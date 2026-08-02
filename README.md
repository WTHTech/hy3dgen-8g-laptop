# AI 文创手办 3D 生成与打印系统

> 🏆 **比赛项目** | AI + 3D 打印 + 文创的交叉创新
>
> **当前阶段**：MVP — 单图生成无纹理 3D 网格并导出 STL/OBJ/GLB
>
> 详细方案与完整路线图见 [`docs/项目方案.md`](docs/项目方案.md)，实时进度见 [`docs/进度跟踪.md`](docs/进度跟踪.md)

---

## 一、项目简介

面向文创手办场景的 AI 驱动 3D 模型生成与打印系统。核心思路是将 **AI 生成** 与 **3D 打印可制造性约束** 深度融合，打造从「图片/草图输入 → 可打印 3D 模型 → 一键切片打印」的全自动闭环。

### 已完成的 MVP 能力

- ✅ 本地 Hunyuan3D-2mini 图生 3D 推理（RTX 5060 8GB VRAM）
- ✅ 分阶段 CPU Offload 显存管理（Shape/Texture 模型不共存）
- ✅ STL / OBJ / GLB 三格式导出
- ✅ 固定 seed 复现 + 轻量回归测试

### 规划中的核心创新

| 创新点 | 说明 | 状态 |
|--------|------|:----:|
| 可打印约束前置生成 | DFM 检测 → 修复 → 复检闭环，提高一次打印成功率 | 🔲 |
| 草图保真分层 Agent | 结构层 + 细节层分治，多 Agent 协作建模 | 🔲 |
| 智能支撑与摆放优化 | AI 识别外观面，自动最优摆放 + 最少支撑 | 🔲 |
| 国潮/校园 IP LoRA 微调 | 非遗纹样、校园地标定向微调专属生成模型 | 🔲 |
| 模块化可拼接生成 | 自动拆分 + 公差卡扣 + BOM 清单 | 🔲 |
| 云端-本地协同架构 | 云端高精度 + 本地轻量兜底 + 离线模板库 | 🔲 |

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| **3D 生成引擎** | [Tencent Hunyuan3D-2mini](https://github.com/Tencent/Hunyuan3D-2)（图生 Shape） |
| **深度学习框架** | PyTorch 2.x + CUDA 12.8 |
| **网格处理** | trimesh, PyMeshLab |
| **显存管理** | accelerate CPU Offload |
| **测试框架** | pytest, unittest |
| **目标部署** | AutoDL / 恒源云 GPU + 本地 RTX 5060 兜底 |

---

## 三、快速开始

### 环境要求

- **GPU**：NVIDIA 显卡，≥ 8GB VRAM
- **CUDA**：≥ 12.8（Blackwell 架构兼容）
- **Python**：3.10
- **OS**：Windows / Linux

### 1. 克隆仓库

```bash
git clone --recurse-submodules https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 2. 创建虚拟环境

```bash
conda create -n hy3d python=3.10 -y
conda activate hy3d
```

### 3. 安装依赖

```bash
# Hunyuan3D-2 依赖
pip install -r Hunyuan3D-2/requirements.txt

# 本项目额外依赖
pip install accelerate trimesh pymeshlab pillow
```

### 4. 下载模型权重

```bash
# 从 HuggingFace 下载（需要安装 huggingface_hub）
pip install huggingface_hub

# Shape 模型（Turbo 加速版，推荐）
huggingface-cli download tencent/Hunyuan3D-2 \
    hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors \
    hunyuan3d-dit-v2-mini-turbo/config.yaml \
    --local-dir ./models/hunyuan3d-dit-v2-mini-turbo

# Shape 模型（标准版，可选）
huggingface-cli download tencent/Hunyuan3D-2 \
    hunyuan3d-dit-v2-mini/model.fp16.safetensors \
    hunyuan3d-dit-v2-mini/config.yaml \
    --local-dir ./models/hunyuan3d-dit-v2-mini

# VAE 模型（可选，后续 FlashVDM 使用）
huggingface-cli download tencent/Hunyuan3D-2 \
    hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors \
    hunyuan3d-vae-v2-mini-turbo/config.yaml \
    --local-dir ./models/hunyuan3d-vae-v2-mini-turbo

# 纹理模型（可选，当前 MVP 尚未验收）
huggingface-cli download tencent/Hunyuan3D-2 \
    hunyuan3d-delight-v2-0 --local-dir ./models/tencent/Hunyuan3D-2/hunyuan3d-delight-v2-0
huggingface-cli download tencent/Hunyuan3D-2 \
    hunyuan3d-paint-v2-0-turbo --local-dir ./models/tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo
```

### 5. 运行自检

```bash
python scripts/inference.py
```

### 6. 交互式使用

```python
import sys
sys.path.insert(0, '.')  # 或项目根目录的绝对路径

from scripts.inference import Hunyuan3DGenerator

gen = Hunyuan3DGenerator(variant='turbo')

# 图生 3D
mesh = gen.generate_shape(image='your_image.png', seed=0)

# 导出
gen.export_stl(mesh, 'output/output.stl')
gen.export_glb(mesh, 'output/output.glb')

# 查看状态
gen.memory_status()
```

---

## 四、项目结构

```
.
├── README.md
├── .gitignore
├── scripts/
│   └── inference.py              # ★ 核心封装：分阶段推理 + CPU Offload
├── tests/
│   └── test_inference_lightweight.py
├── dfm/                           # DFM 检查模块（待开发）
├── docs/
│   ├── 项目方案.md                # 完整方案与路线图
│   ├── 进度跟踪.md                # 实时进度记录
│   ├── 参考文献.md                # 必读论文清单
│   └── 开源项目调研.md            # 竞品分析
├── Hunyuan3D-2/                   # Git Submodule → Tencent/Hunyuan3D-2
├── models/                        # 模型权重（本地，不提交 Git）
├── output/                        # 生成结果（本地，不提交 Git）
└── examples/                      # 精选样例展示
```

---

## 五、当前完成度

| 阶段 | 内容 | 状态 |
|------|------|:----:|
| Phase 0 | 本地环境 + CUDA + 模型导入 | ✅ |
| Phase 1 | 单图生无纹理 3D + 三格式导出冒烟验收 | ✅ |
| Phase 2 | DFM 工具集成与可打印性验证 | 🔲 |
| Phase 3 | 图生 3D MVP 完整性验证（10+ 测试图） | 🔲 |
| Phase 4 | 云端 Hunyuan3D-2 完整版部署 | 🔲 |
| Phase 5 | 全链路跑通（生成→DFM→修复→切片→打印） | 🔲 |

冒烟测试实测数据（turbo 模型，低分辨率）：
- 总用时 52.5s | 39,067 顶点 | 78,146 三角面
- 拓扑：单连通分量、水密、绕序一致
- 推理后显存回到 0.0GB

---

## 六、LICENSE 说明

- 本项目的自有代码（`scripts/`、`tests/`、`dfm/`）采用 [MIT License](LICENSE)
- 依赖的 Hunyuan3D-2 使用 [Tencent Hunyuan 3D 2.0 Community License](https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE)，请注意其使用限制（不适用于欧盟/英国/韩国，月活 >100 万需额外许可）
- `docs/` 中的方案与调研文档为项目参考资料

---

## 七、常见问题

### OMP 报错

```powershell
# 如果遇到 "libiomp5md.dll already initialized"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

### 首次推理慢

CUDA kernel 编译有预热开销，第二次及之后推理会显著加快。
