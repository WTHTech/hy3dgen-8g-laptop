"""
Hunyuan3D-2mini 分阶段推理封装
================================
核心约束：
  1. Shape / Texture 模型分阶段加载，决不同时驻留显存
  2. 默认只生成形状（调试模式），可选加载纹理
  3. 每个阶段：加载 → 推理 → 可靠卸载（异常时也释放资源）
  4. 仅导出 GLB 时可选加载纹理生成带贴图的彩色模型

硬件适配：RTX 5060 Laptop 8GB VRAM，启用 CPU Offload 降低峰值显存。

用法：
  from scripts.inference import Hunyuan3DGenerator

  gen = Hunyuan3DGenerator(model_dir='D:/AI文创3D打印/models')

  # 图生 3D（仅形状，调试默认模式）
  mesh = gen.generate_shape(image='demo.png')
  gen.export_stl(mesh, 'output.stl')

  # 图生 3D + 纹理 → GLB
  mesh = gen.generate_shape(image='demo.png')
  gen.export_glb(mesh, 'output.glb', with_texture=True, image='demo.png')
"""

import gc
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# ── 路径设置 ─────────────────────────────────────────────
# 将 Hunyuan3D 源码加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUNYUAN3D_ROOT = PROJECT_ROOT / 'Hunyuan3D-2'
MODEL_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# 设置模型缓存环境变量（smart_load_model 会优先读这个）
# 用于 enable_flashvdm 等需要 from_pretrained 的场景
os.environ.setdefault('HY3DGEN_MODELS', str(MODEL_DIR))

if str(HUNYUAN3D_ROOT) not in sys.path:
    sys.path.insert(0, str(HUNYUAN3D_ROOT))


# ── 模型文件映射 ─────────────────────────────────────────
# 当前封装直接从 DiT checkpoint 加载；checkpoint 已包含 VAE 和 conditioner。
MODEL_VARIANTS = {
    'turbo': {
        'dit': 'hunyuan3d-dit-v2-mini-turbo',
        'default_steps': 25,
    },
    'standard': {
        'dit': 'hunyuan3d-dit-v2-mini',
        'default_steps': 50,
    },
}

# 纹理模型需要的两个子模型（需从 HuggingFace 额外下载）
# 路径: {HY3DGEN_MODELS}/tencent/Hunyuan3D-2/{subfolder}
TEX_DELIGHT_SUBFOLDER = 'hunyuan3d-delight-v2-0'
TEX_PAINT_SUBFOLDER = 'hunyuan3d-paint-v2-0'
TEX_PAINT_TURBO_SUBFOLDER = 'hunyuan3d-paint-v2-0-turbo'

# ── 辅助函数 ─────────────────────────────────────────────

def _log(msg: str):
    print(f'[Hunyuan3D] {msg}')


def _free_gpu_memory():
    """在调用方已解除模型引用后回收 Python 对象并清理 CUDA 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_input_image(image):
    """读取输入图片，保留透明 PNG 的 alpha 通道。

    RGB 图片不会在这里自动抠图，调用方应提供干净背景；需要自动抠图时
    使用 ``remove_background=True``。
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            has_alpha = 'A' in opened.getbands() or 'transparency' in opened.info
            return opened.convert('RGBA' if has_alpha else 'RGB').copy()
    if isinstance(image, Image.Image):
        return image.convert('RGBA' if 'A' in image.getbands() else 'RGB')
    raise TypeError(f'image 必须是图片路径或 PIL.Image，实际为: {type(image).__name__}')


def _remove_background(image):
    """按需调用官方 rembg 封装；首次使用可能需要本地 rembg 权重。"""
    from hy3dgen.rembg import BackgroundRemover

    return BackgroundRemover()(image.convert('RGB'))


def _vram_used() -> str:
    """返回当前 CUDA 显存使用量（人类可读）"""
    if not torch.cuda.is_available():
        return 'N/A'
    used = torch.cuda.memory_allocated() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return f'{used:.1f}GB / {total:.1f}GB'


# ── 主类 ─────────────────────────────────────────────────

class Hunyuan3DGenerator:
    """Hunyuan3D-2mini 分阶段推理器

    Parameters
    ----------
    model_dir : str or Path
        模型文件根目录（包含 hunyuan3d-dit-v2-mini/ 等子目录）
    variant : str
        'turbo'（推荐，更快）或 'standard'
    device : str
        推理设备，默认 cuda
    """

    def __init__(self, model_dir=None, variant='turbo', device='cuda'):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.variant = variant
        self.device = device
        self.dtype = torch.float16

        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU')

        # 验证模型文件存在
        self._validate_models()

        # 内部状态：pipeline 决不共存
        self._shape_pipeline = None   # Shape 生成 pipeline
        self._tex_pipeline = None     # 纹理生成 pipeline
        self._tex_available = False   # 纹理模型是否可用

        # 检查纹理模型是否已下载
        self._check_texture_available()

    # ── 模型验证 ──────────────────────────────────────

    def _validate_models(self):
        """验证形状模型文件完整性"""
        cfg = MODEL_VARIANTS.get(self.variant)
        if cfg is None:
            raise ValueError(f'未知 variant: {self.variant}，可选: {list(MODEL_VARIANTS.keys())}')

        dit_dir = self.model_dir / cfg['dit']
        if not dit_dir.exists():
            raise FileNotFoundError(f'Shape 模型目录不存在: {dit_dir}')

        required = ['config.yaml', 'model.fp16.safetensors']
        for fname in required:
            if not (dit_dir / fname).exists():
                raise FileNotFoundError(f'缺少模型文件: {dit_dir / fname}')

        _log(f'模型验证通过: variant={self.variant}, dir={dit_dir}')

    def _check_texture_available(self):
        """检查纹理模型是否已下载"""
        tex_base = self.model_dir / 'tencent' / 'Hunyuan3D-2'
        delight = tex_base / TEX_DELIGHT_SUBFOLDER
        paint = tex_base / TEX_PAINT_TURBO_SUBFOLDER
        paint_std = tex_base / TEX_PAINT_SUBFOLDER

        if delight.exists() and (paint.exists() or paint_std.exists()):
            self._tex_available = True
            self._tex_model_path = str(tex_base)
            _log('纹理模型已就绪，支持 GLB 纹理导出')
        else:
            _log('纹理模型未下载，仅支持无纹理形状生成 (STL/OBJ/GLB)')
            _log(f'  如需纹理: 下载 hunyuan3d-delight-v2-0 + hunyuan3d-paint-v2-0 到 {tex_base}')

    # ── Shape 模型加载/卸载 ───────────────────────────

    def _load_shape_pipeline(self):
        """加载形状生成 pipeline + 启用 CPU Offload（适配 8GB 显存）

        使用 from_single_file 直接从本地文件加载，不走 HuggingFace Hub。
        """
        if self._shape_pipeline is not None:
            _log('Shape pipeline 已在显存中，跳过加载')
            return self._shape_pipeline

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        cfg = MODEL_VARIANTS[self.variant]
        dit_dir = self.model_dir / cfg['dit']
        ckpt_path = str(dit_dir / 'model.fp16.safetensors')
        config_path = str(dit_dir / 'config.yaml')

        _log(f'加载 Shape 模型: {dit_dir.name} ...')
        _log(f'  显存状态(加载前): {_vram_used()}')

        pipeline = None
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                ckpt_path=ckpt_path,
                config_path=config_path,
                device=self.device,
                dtype=self.dtype,
                use_safetensors=True,
            )

            # 当前 ZIP 源码的自定义 pipeline 未定义 diffusers 常见的 components
            # 属性，但 CPU Offload 实现会读取它；在封装层补齐映射，避免修改上游。
            if not hasattr(pipeline, 'components'):
                pipeline.components = {
                    'conditioner': pipeline.conditioner,
                    'model': pipeline.model,
                    'vae': pipeline.vae,
                }

            # 启用 CPU Offload：模型组件按需加载到 GPU，空闲时自动卸载到 CPU。
            pipeline.enable_model_cpu_offload(device=self.device)
            # 上游自定义管线的 offload 实现会把 pipeline.device 留在 CPU，
            # 但采样循环据此创建 latents，随后模型 hook 又输出 CUDA tensor，
            # 会造成 scheduler 的 CPU/CUDA 混用。模块仍由 hook 保持在 CPU，
            # 这里只恢复采样张量应使用的执行设备。
            pipeline.device = torch.device(self.device)
        except Exception:
            if pipeline is not None:
                try:
                    pipeline.to('cpu')
                except Exception:
                    pass
                del pipeline
            _free_gpu_memory()
            raise

        self._shape_pipeline = pipeline
        _log(f'  Shape 模型加载完成，显存: {_vram_used()}')
        return pipeline

    def _unload_shape_pipeline(self):
        """卸载形状生成 pipeline，释放显存"""
        if self._shape_pipeline is None:
            return

        _log(f'卸载 Shape 模型 (卸载前显存: {_vram_used()})')
        pipeline = self._shape_pipeline
        self._shape_pipeline = None
        try:
            # 移除 accelerate hook 并把仍驻留 GPU 的组件迁回 CPU。
            for hook in getattr(pipeline, '_all_hooks', []):
                hook.offload()
                hook.remove()
            pipeline._all_hooks = []
            pipeline.to('cpu')
        except Exception as exc:
            _log(f'  Shape 模型迁回 CPU 时出现警告: {exc}')
        del pipeline
        _free_gpu_memory()
        _log(f'  Shape 模型已卸载，显存: {_vram_used()}')

    # ── 纹理模型加载/卸载 ─────────────────────────────

    def _load_tex_pipeline(self):
        """加载纹理生成 pipeline（仅在需要时调用）"""
        if not self._tex_available:
            raise RuntimeError('纹理模型未下载，无法加载。请先下载纹理模型。')

        if self._tex_pipeline is not None:
            _log('Texture pipeline 已在显存中，跳过加载')
            return self._tex_pipeline

        # 确保 Shape pipeline 已卸载（双模型禁止共存）
        if self._shape_pipeline is not None:
            _log('!! 检测到 Shape pipeline 仍在显存，先卸载...')
            self._unload_shape_pipeline()

        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        _log(f'加载 Texture 模型...')
        _log(f'  显存状态(加载前): {_vram_used()}')

        paint_subfolder = TEX_PAINT_TURBO_SUBFOLDER if (
            self.model_dir / 'tencent' / 'Hunyuan3D-2' / TEX_PAINT_TURBO_SUBFOLDER
        ).exists() else TEX_PAINT_SUBFOLDER

        pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            model_path=self._tex_model_path,
            subfolder=paint_subfolder,
        )

        # 纹理管线也启用 CPU offload
        pipeline.enable_model_cpu_offload()

        self._tex_pipeline = pipeline
        _log(f'  Texture 模型加载完成，显存: {_vram_used()}')
        return pipeline

    def _unload_tex_pipeline(self):
        """卸载纹理生成 pipeline"""
        if self._tex_pipeline is None:
            return

        _log(f'卸载 Texture 模型 (卸载前显存: {_vram_used()})')
        pipeline = self._tex_pipeline
        self._tex_pipeline = None
        try:
            for model_wrapper in getattr(pipeline, 'models', {}).values():
                inner = getattr(model_wrapper, 'pipeline', None)
                for hook in getattr(inner, '_all_hooks', []):
                    hook.offload()
                    hook.remove()
                if inner is not None:
                    inner.to('cpu')
        except Exception as exc:
            _log(f'  Texture 模型迁回 CPU 时出现警告: {exc}')
        del pipeline
        _free_gpu_memory()
        _log(f'  Texture 模型已卸载，显存: {_vram_used()}')

    # ── 核心推理接口 ──────────────────────────────────

    @torch.no_grad()
    def generate_shape(self, image, num_inference_steps=None, guidance_scale=5.0,
                       octree_resolution=256, num_chunks=5000,
                       enable_pbar=True, seed=0, remove_background=False):
        """从图片生成 3D 形状（无纹理）

        Parameters
        ----------
        image : str, Path, or PIL.Image
            输入图片路径或 PIL Image 对象
        num_inference_steps : int, optional
            扩散步数，默认 25 (turbo) / 50 (standard)
        guidance_scale : float
            CFG 引导强度
        octree_resolution : int
            网格提取分辨率，越高越精细但越耗显存
        num_chunks : int
            Marching Cubes 分块数，降低可减少峰值显存
        enable_pbar : bool
            是否显示进度条
        seed : int
            随机种子，用于复现实验结果
        remove_background : bool
            是否调用 rembg 自动抠图；透明 PNG 无需开启

        Returns
        -------
        trimesh.Trimesh
            生成的三角网格（无纹理）
        """
        if num_inference_steps is None:
            num_inference_steps = MODEL_VARIANTS[self.variant]['default_steps']
        if num_inference_steps <= 0:
            raise ValueError('num_inference_steps 必须大于 0')

        image = _load_input_image(image)
        if remove_background:
            image = _remove_background(image)

        pipeline = self._load_shape_pipeline()
        try:
            _log(
                f'开始生成形状... (steps={num_inference_steps}, '
                f'resolution={octree_resolution}, seed={seed})'
            )
            _log(f'  生成前显存: {_vram_used()}')
            generator_device = self.device if str(self.device).startswith('cuda') else 'cpu'
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))
            mesh = pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                enable_pbar=enable_pbar,
                generator=generator,
                output_type='trimesh',
            )[0]
            _log(f'  生成后显存: {_vram_used()}')
            return mesh
        finally:
            # 先解除本方法的引用，再由实例卸载唯一剩余引用。
            del pipeline
            self._unload_shape_pipeline()

    @torch.no_grad()
    def generate_texture(self, mesh, image):
        """为已有网格生成纹理（独立阶段，单独加载/卸载纹理模型）

        Parameters
        ----------
        mesh : trimesh.Trimesh
            已生成的无纹理网格
        image : str, Path, or PIL.Image
            参考图片（用于生成纹理）

        Returns
        -------
        trimesh.Trimesh
            带纹理的网格
        """
        if self._shape_pipeline is not None:
            raise RuntimeError(
                'Shape pipeline 仍在显存中！'
                ' generate_shape() 已自动卸载 Shape 模型，'
                ' 如果你手动加载了，请先调用 _unload_shape_pipeline()'
            )

        image = _load_input_image(image)
        pipeline = self._load_tex_pipeline()
        try:
            _log('开始生成纹理...')
            _log(f'  生成前显存: {_vram_used()}')
            textured_mesh = pipeline(mesh, image=image)
            _log(f'  生成后显存: {_vram_used()}')
            return textured_mesh
        finally:
            del pipeline
            self._unload_tex_pipeline()

    # ── 导出接口 ──────────────────────────────────────

    def export_stl(self, mesh, output_path):
        """导出 STL 格式（无纹理，直接可用于切片打印）

        Parameters
        ----------
        mesh : trimesh.Trimesh
            要导出的网格
        output_path : str or Path
            输出路径，.stl 后缀
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path))
        _log(f'STL 已导出: {path} ({len(mesh.faces)} 面, {len(mesh.vertices)} 顶点)')

    def export_obj(self, mesh, output_path):
        """导出 OBJ 格式"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path))
        _log(f'OBJ 已导出: {path}')

    def export_glb(self, mesh, output_path, with_texture=False, image=None):
        """导出 GLB 格式

        Parameters
        ----------
        mesh : trimesh.Trimesh
            基础网格
        output_path : str or Path
            输出路径
        with_texture : bool
            是否先生成纹理再导出（会额外加载纹理模型）
        image : PIL.Image or str
            纹理参考图（with_texture=True 时必需）
        """
        if with_texture:
            if image is None:
                raise ValueError('with_texture=True 时需要提供 image 参数')
            mesh = self.generate_texture(mesh, image)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path))
        _log(f'GLB 已导出: {path} (纹理: {with_texture})')

    # ── 便捷的全流程接口 ──────────────────────────────

    def image_to_3d(self, image, export_stl=None, export_glb=None,
                    with_texture=False, **shape_kwargs):
        """图生 3D 一站式接口

        Parameters
        ----------
        image : str, Path, or PIL.Image
            输入图片
        export_stl : str or Path, optional
            STL 输出路径
        export_glb : str or Path, optional
            GLB 输出路径
        with_texture : bool
            GLB 是否带纹理（仅在 export_glb 时生效）
        **shape_kwargs
            传给 generate_shape 的参数

        Returns
        -------
        trimesh.Trimesh
        """
        # 阶段1: 生成形状（Shape 模型加载→推理→卸载）
        mesh = self.generate_shape(image, **shape_kwargs)

        # 阶段2: 导出 STL（不需要纹理模型）
        if export_stl:
            self.export_stl(mesh, export_stl)

        # 阶段3: 导出 GLB（可选纹理，独立加载/卸载纹理模型）
        if export_glb:
            self.export_glb(mesh, export_glb,
                           with_texture=with_texture, image=image)

        return mesh

    # ── 显存状态查询 ──────────────────────────────────

    def memory_status(self):
        """打印当前显存和模型加载状态"""
        print(f'── Hunyuan3D 显存状态 ──')
        print(f'  VRAM: {_vram_used()}')
        print(f'  Shape pipeline: {"已加载" if self._shape_pipeline else "未加载"}')
        print(f'  Texture pipeline: {"已加载" if self._tex_pipeline else "未加载"}')
        print(f'  纹理可用: {"是" if self._tex_available else "否（需下载）"}')
        print(f'  双模型共存: {"!! 警告 !!" if self._shape_pipeline and self._tex_pipeline else "安全"}')

    def __del__(self):
        """析构时确保释放所有 GPU 资源"""
        try:
            self._unload_shape_pipeline()
            self._unload_tex_pipeline()
        except Exception:
            # 解释器退出阶段依赖对象可能已经被销毁，析构函数不能再抛异常。
            pass


# ── 便捷函数 ─────────────────────────────────────────────

def create_generator(variant='turbo', model_dir=None):
    """快速创建生成器实例"""
    return Hunyuan3DGenerator(model_dir=model_dir, variant=variant)


# ── 自检脚本 ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('Hunyuan3D-2mini 分阶段推理封装 - 自检')
    print('=' * 60)

    gen = Hunyuan3DGenerator(variant='turbo')
    gen.memory_status()

    # 如果有测试图片，跑一次生成
    test_img = PROJECT_ROOT / 'Hunyuan3D-2' / 'assets' / 'demo.png'
    if test_img.exists():
        print(f'\n── 测试生成: {test_img} ──')
        mesh = gen.generate_shape(image=str(test_img))
        print(f'  结果: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面')

        out_stl = str(OUTPUT_DIR / 'test_output.stl')
        gen.export_stl(mesh, out_stl)
        gen.memory_status()
    else:
        print(f'\n测试图片不存在: {test_img}')
        print('请提供一张图片来做验证：')
        print('  gen = create_generator()')
        print('  mesh = gen.generate_shape(image="your_image.png")')
        print('  gen.export_stl(mesh, "output.stl")')
