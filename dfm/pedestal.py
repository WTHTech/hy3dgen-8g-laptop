"""DFM 底座生成器。

为 P1（底面平台）检测失败或接触面积不足的模型自动生成适配底座，
使其能够稳定放置在 3D 打印平台上。

底座添加在 P1 判定之后、拓扑修复之前：合并可能引入新的
非流形边，需要后续修复环处理。

用法::

    from dfm import PedestalGenerator, DFMRules

    rules = DFMRules()
    gen = PedestalGenerator(rules)

    result = gen.generate(prepared_mesh, style='disc')
    if result.success:
        result.mesh.export('with_pedestal.stl')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Optional

import numpy as np
import trimesh

from .rules import DFMRules


@dataclass
class PedestalResult:
    """底座生成结果。"""

    success: bool = False
    mesh: Optional[trimesh.Trimesh] = None
    style: str = ''
    footprint_area_mm2: float = 0.0
    source_contact_area_mm2: float = 0.0
    pedestal_top_area_mm2: float = 0.0
    pedestal_bottom_area_mm2: float = 0.0
    pedestal_thickness_mm: float = 0.0
    pedestal_radius_mm: float = 0.0
    pedestal_bottom_radius_mm: float = 0.0
    pedestal_volume_mm3: float = 0.0
    merge_method: str = ''
    component_count: int = 0
    detail: str = ''
    actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'style': self.style,
            'footprint_area_mm2': round(self.footprint_area_mm2, 3),
            'source_contact_area_mm2': round(self.source_contact_area_mm2, 3),
            'pedestal_top_area_mm2': round(self.pedestal_top_area_mm2, 3),
            'pedestal_bottom_area_mm2': round(self.pedestal_bottom_area_mm2, 3),
            'pedestal_thickness_mm': round(self.pedestal_thickness_mm, 3),
            'pedestal_radius_mm': round(self.pedestal_radius_mm, 3),
            'pedestal_bottom_radius_mm': round(self.pedestal_bottom_radius_mm, 3),
            'pedestal_volume_mm3': round(self.pedestal_volume_mm3, 3),
            'merge_method': self.merge_method,
            'component_count': self.component_count,
            'detail': self.detail,
            'actions': self.actions,
            'vertices': len(self.mesh.vertices) if self.mesh is not None else 0,
            'faces': len(self.mesh.faces) if self.mesh is not None else 0,
        }


class PedestalGenerator:
    """底座生成器。

    为底面接触不足的模型创建适配底座，然后合并成一个网格。
    合并后的网格表面将不水密（底座顶面与模型底面之间有空隙），
    所以调用方必须将其送入修复闭环处理。

    Parameters
    ----------
    rules : DFMRules, optional
        读取 ``min_bottom_area``、``z_tolerance`` 和底座相关阈值。
    """

    # 默认底座设计参数
    DEFAULT_MARGIN = 2.0       # mm，底座相对足迹的外扩余量
    DEFAULT_THICKNESS = 2.0    # mm，底座厚度（层高整数倍）
    DEFAULT_TAPER_ANGLE = 60   # 度（从水平面测量，90°=完全垂直）
    DEFAULT_ATTACHMENT_OVERLAP = 0.4
    DEFAULT_CONTACT_BAND = 0.4
    DEFAULT_MAX_DIMENSION = 180.0

    def __init__(self, rules: Optional[DFMRules] = None):
        self.rules = rules or DFMRules()

    # ── 公开接口 ──────────────────────────────────────

    def generate(
        self,
        mesh: trimesh.Trimesh,
        style: Optional[str] = None,
        margin: Optional[float] = None,
        thickness: Optional[float] = None,
        taper_angle: Optional[float] = None,
        attachment_overlap: Optional[float] = None,
        contact_band: Optional[float] = None,
        max_dimension: Optional[float] = None,
    ) -> PedestalResult:
        """给待检查的毫米网格生成底座并合并。

        参数单位全部是毫米；传入的 mesh 必须是已经标准化并移至
        Z=0 平台的 ``prepared_mesh``。

        Parameters
        ----------
        mesh : trimesh.Trimesh
            prepared_mesh（毫米，底部在 Z≈0）。
        style : {'auto', 'disc', 'block'}
            ``auto`` 按低位支撑轮廓的长宽比选择样式。
            ``disc`` 用圆形底座，适合点/线接触的孤立模型。
            ``block`` 用矩形底座，适合多脚分立模型。
        margin : float, optional
            底座边缘相对足迹的外扩量(mm)；默认 2.0。
        thickness : float, optional
            底座厚度(mm)；默认 2.0（= 层高 × 10）。
        taper_angle : float, optional
            底座侧面从水平面测量的倾角(度)；默认 60°。
            值越小越倾斜，90° = 完全垂直。

        Returns
        -------
        PedestalResult
        """
        # 参数归一化
        if margin is None:
            margin = self.rules.get('pedestal_margin_mm', self.DEFAULT_MARGIN)
        if thickness is None:
            thickness = self.rules.get('pedestal_thickness_mm', self.DEFAULT_THICKNESS)
        if taper_angle is None:
            taper_angle = self.rules.get(
                'pedestal_taper_angle_deg', self.DEFAULT_TAPER_ANGLE,
            )
        if attachment_overlap is None:
            attachment_overlap = self.rules.get(
                'pedestal_attachment_overlap_mm', self.DEFAULT_ATTACHMENT_OVERLAP,
            )
        if contact_band is None:
            contact_band = self.rules.get(
                'pedestal_contact_band_mm', self.DEFAULT_CONTACT_BAND,
            )
        if max_dimension is None:
            max_dimension = self.rules.get(
                'pedestal_max_dimension_mm', self.DEFAULT_MAX_DIMENSION,
            )
        if style is None:
            style = self.rules.get('pedestal_style', 'auto')

        style = str(style).strip().lower()
        if style not in {'auto', 'disc', 'block'}:
            return PedestalResult(
                success=False,
                detail=f'不支持底座样式 {style!r}，可选 auto、disc 或 block',
            )
        for name, value in [
            ('margin', margin), ('thickness', thickness), ('taper_angle', taper_angle),
            ('attachment_overlap', attachment_overlap),
            ('contact_band', contact_band), ('max_dimension', max_dimension),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                return PedestalResult(
                    success=False,
                    detail=f'{name} 必须是正的有限数，实际为 {value!r}',
                )
        if taper_angle > 90:
            return PedestalResult(
                success=False,
                detail=f'taper_angle 必须在 (0, 90] 度内，实际为 {taper_angle!r}',
            )

        if not _mesh_ok(mesh):
            return PedestalResult(
                success=False,
                detail='输入网格无效：顶点少于 3 或面数不足',
            )

        z_tol = max(float(self.rules.get('z_tolerance', 1e-5)), 1e-5)
        min_z = float(mesh.bounds[0, 2])
        if abs(min_z) > z_tol:
            return PedestalResult(
                success=False,
                detail=(
                    '输入必须是已落到 Z=0 的 prepared_mesh，'
                    f'实际 min_z={min_z:.6f}mm'
                ),
            )

        actions: list[dict] = []

        # 1. 提取底面足迹的 2D 轮廓
        source_contact_area = self._bottom_contact_area(mesh, z_tol)
        hull_2d = self._extract_footprint_hull(mesh, float(contact_band))
        if hull_2d is None or len(hull_2d) < 3:
            # 理论上只有损坏网格会走到这里；只围绕最低点创建最小圆，
            # 禁止使用整个模型包围盒，以免张开手臂导致巨型底座。
            footprint_area = 0.0
            lowest = mesh.vertices[np.argmin(mesh.vertices[:, 2]), :2]
            hull_2d = _circle_vertices(
                lowest, np.sqrt(float(self.rules.get('min_bottom_area', 5.0)) / np.pi),
            )
            actions.append({
                'step': '提取底面轮廓',
                'detail': '低位轮廓退化，围绕最低点创建最小支撑圆',
            })
        else:
            footprint_area = self._polygon_area(hull_2d)
            actions.append({
                'step': '提取底面轮廓',
                'detail': f'底面接触点 {len(hull_2d)} 个，足迹面积 {footprint_area:.2f} mm²',
            })

        min_area = float(self.rules.get('min_bottom_area', 5.0))
        if footprint_area < min_area:
            hull_2d = self._inflate_to_min_area(hull_2d, min_area)
            footprint_area = self._polygon_area(hull_2d)
            actions.append({
                'step': '膨胀轮廓',
                'detail': f'原始足迹 < {min_area} mm²，膨胀至 {footprint_area:.2f} mm²',
            })

        # 将重心投影纳入支撑轮廓；底座外扩后，重心会至少落在 margin 内侧。
        support_anchor = self._support_anchor_xy(mesh)
        support_hull = self._convex_hull_2d(np.vstack([hull_2d, support_anchor]))
        if support_hull is not None:
            hull_2d = support_hull

        # 2. 外扩 margin
        hull_2d = self._offset_polygon(hull_2d, margin)
        if hull_2d is None or len(hull_2d) < 3:
            return PedestalResult(
                success=False,
                detail='底座轮廓外扩后退化',
                actions=actions,
            )

        final_footprint = self._polygon_area(hull_2d)
        actions.append({
            'step': '外扩轮廓',
            'detail': f'margin={margin:.1f}mm，最终底面积 {final_footprint:.2f} mm²',
        })

        # 3. 创建底座 3D 网格
        if style == 'auto':
            style = self._select_style(hull_2d)
            actions.append({
                'step': '选择底座样式',
                'detail': f'按支撑轮廓长宽比自动选择 {style}',
            })

        pedestal, pedestal_radius, bottom_radius = self._build_pedestal(
            hull_2d, float(thickness), float(taper_angle), style,
            float(attachment_overlap),
        )
        if pedestal is None:
            return PedestalResult(
                success=False,
                detail='底座 3D 网格创建失败',
                actions=actions,
            )
        pedestal_xy = pedestal.extents[:2]
        build_volume = self.rules.get('build_volume', [220, 220, 250])
        allowed_xy = np.minimum(
            np.asarray(build_volume[:2], dtype=float), float(max_dimension),
        )
        if np.any(pedestal_xy > allowed_xy + 1e-9):
            return PedestalResult(
                success=False,
                detail=(
                    f'自动底座尺寸 {pedestal_xy[0]:.1f}×{pedestal_xy[1]:.1f}mm '
                    f'超过限制 {allowed_xy[0]:.1f}×{allowed_xy[1]:.1f}mm；'
                    '应先调整摆放方向或人工确认'
                ),
                actions=actions,
            )
        bottom_area = self._horizontal_area_at_z(pedestal, float(pedestal.bounds[0, 2]))
        top_area = self._horizontal_area_at_z(pedestal, float(pedestal.bounds[1, 2]))
        actions.append({
            'step': '创建底座网格',
            'detail': f'{style} 底座，厚度 {thickness:.1f}mm，'
                      f'顶部半径 {pedestal_radius:.1f}mm，'
                      f'{len(pedestal.faces)} 面',
        })

        # 4. 合并
        merged, merge_method, merge_detail = self._merge_meshes(mesh, pedestal)
        if merged is None or not _mesh_ok(merged):
            return PedestalResult(
                success=False,
                detail=f'底座与模型未能融合为单一实体：{merge_detail}',
                actions=actions,
            )
        component_count = _component_count(merged)
        # 体素融合会按 pitch 量化边界；最终报告必须以真正输出网格为准，
        # 不能继续记录融合前的理论底面积。
        bottom_area = self._horizontal_area_at_z(merged, float(merged.bounds[0, 2]))
        actions.append({
            'step': '合并网格',
            'detail': f'{merge_method}；{len(mesh.vertices)}→{len(merged.vertices)} 顶点，'
                      f'{len(mesh.faces)}→{len(merged.faces)} 面',
        })

        return PedestalResult(
            success=True,
            mesh=merged,
            style=style,
            footprint_area_mm2=bottom_area,
            source_contact_area_mm2=source_contact_area,
            pedestal_top_area_mm2=top_area,
            pedestal_bottom_area_mm2=bottom_area,
            pedestal_thickness_mm=float(thickness),
            pedestal_radius_mm=pedestal_radius,
            pedestal_bottom_radius_mm=bottom_radius,
            pedestal_volume_mm3=abs(float(pedestal.volume)),
            merge_method=merge_method,
            component_count=component_count,
            detail=f'{style} 底座已融合：实际底面积 {bottom_area:.1f}mm²，'
                   f'厚 {float(thickness):.1f}mm，单连通体；下步需重新 DFM 检查',
            actions=actions,
        )

    # ── 足迹提取 ──────────────────────────────────────

    def _extract_footprint_hull(
        self, mesh: trimesh.Trimesh, contact_band: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """提取最低点以上一个小高度带内顶点的 XY 凸包。

        高度带比 P1 的共面容差更宽，能为点接触的曲面取得局部支撑轮廓，
        但不会把高处的手臂、武器等误当成底座范围。
        """
        if contact_band is None:
            contact_band = float(self.rules.get(
                'pedestal_contact_band_mm', self.DEFAULT_CONTACT_BAND,
            ))
        min_z = float(mesh.vertices[:, 2].min())
        bottom_mask = mesh.vertices[:, 2] <= min_z + float(contact_band)
        bottom_verts = mesh.vertices[bottom_mask]

        hull = self._convex_hull_2d(bottom_verts[:, :2])
        if hull is not None:
            return hull

        if len(bottom_verts) == 0:
            return None
        points = bottom_verts[:, :2]
        center = points.mean(axis=0)
        spread = float(np.linalg.norm(points - center, axis=1).max())
        min_radius = np.sqrt(float(self.rules.get('min_bottom_area', 5.0)) / np.pi)
        return _circle_vertices(center, max(spread, min_radius))

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> Optional[np.ndarray]:
        """返回二维点集的凸包；退化或非有限输入返回 ``None``。"""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            return None
        points = np.unique(points, axis=0)
        if len(points) < 3 or not np.all(np.isfinite(points)):
            return None
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            return None

    @staticmethod
    def _support_anchor_xy(mesh: trimesh.Trimesh) -> np.ndarray:
        """取得用于稳定性扩展的重心投影，失败时回退到表面质心。"""
        try:
            center = np.asarray(
                mesh.center_mass if mesh.is_volume else mesh.centroid,
                dtype=float,
            )
        except Exception:
            center = np.asarray(mesh.centroid, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            center = np.asarray(mesh.centroid, dtype=float)
        return center[:2].reshape(1, 2)

    @staticmethod
    def _select_style(points: np.ndarray) -> str:
        """细长、多脚支撑轮廓用 block，近方形轮廓用 disc。"""
        extents = np.ptp(points, axis=0)
        short = max(float(extents.min()), 1e-9)
        aspect = float(extents.max()) / short
        return 'block' if aspect >= 1.75 else 'disc'

    @staticmethod
    def _bottom_contact_area(mesh: trimesh.Trimesh, tolerance: float) -> float:
        min_z = float(mesh.vertices[:, 2].min())
        face_z = mesh.vertices[mesh.faces][:, :, 2]
        mask = np.all(np.abs(face_z - min_z) <= tolerance, axis=1)
        if not np.any(mask):
            return 0.0
        return float(np.sum(mesh.area_faces[mask] * np.abs(mesh.face_normals[mask, 2])))

    @staticmethod
    def _horizontal_area_at_z(mesh: trimesh.Trimesh, z: float) -> float:
        tolerance = max(1e-7, abs(z) * 1e-9)
        face_z = mesh.vertices[mesh.faces][:, :, 2]
        mask = np.all(np.abs(face_z - z) <= tolerance, axis=1)
        if not np.any(mask):
            return 0.0
        return float(np.sum(mesh.area_faces[mask] * np.abs(mesh.face_normals[mask, 2])))

    # ── 几何计算 ───────────────────────────────────────

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        """鞋带公式计算 2D 简单多边形面积。"""
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @staticmethod
    def _inflate_to_min_area(
        points: np.ndarray, min_area: float
    ) -> np.ndarray:
        """按质心等比膨胀到最小面积，保留原足迹方向和长宽比。"""
        centroid = points.mean(axis=0)
        area = PedestalGenerator._polygon_area(points)
        if area <= 1e-12:
            radius = np.sqrt(min_area / np.pi)
            return _circle_vertices(centroid, radius)
        scale = np.sqrt(min_area / area)
        return centroid + (points - centroid) * scale

    @staticmethod
    def _offset_polygon(
        points: np.ndarray, distance: float
    ) -> Optional[np.ndarray]:
        """对 2D 多边形做外扩缓冲（Shapely 可用）或者缩放回退。"""
        # 优先 Shapely
        try:
            from shapely.geometry import Polygon

            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            try:
                buffered = poly.buffer(distance, quad_segs=8)
            except TypeError:  # Shapely < 2
                buffered = poly.buffer(distance, resolution=8)
            if buffered.is_empty:
                return None
            coords = np.asarray(buffered.exterior.coords)
            return coords[:, :2]
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

        # 回退：以质心为中心均匀缩放
        centroid = points.mean(axis=0)
        max_dist = float(np.linalg.norm(points - centroid, axis=1).max())
        if max_dist < 1e-9:
            return None
        scale = 1.0 + distance / max_dist
        return centroid + (points - centroid) * scale

    # ── 底座网格构建 ───────────────────────────────────

    @staticmethod
    def _build_pedestal(
        hull_2d: np.ndarray,
        thickness: float,
        taper_angle: float,
        style: str,
        attachment_overlap: float,
    ) -> tuple[Optional[trimesh.Trimesh], float, float]:
        """从 2D 轮廓创建底座，顶部伸入模型以保证存在实体交叠。"""
        centroid = hull_2d.mean(axis=0)
        total_height = thickness + attachment_overlap

        if style == 'disc':
            radii = np.linalg.norm(hull_2d - centroid, axis=1)
            radius = float(radii.max())
            # 圆柱圆周上的顶点间距约 1mm，最少 24 段
            sections = min(720, max(24, int(np.ceil(np.pi * 2 * radius))))
            pedestal = trimesh.creation.cylinder(
                radius=radius, height=total_height, sections=sections,
            )
            # 底面位于 -thickness，顶面伸入模型 attachment_overlap。
            pedestal.apply_translation([
                float(centroid[0]), float(centroid[1]),
                (-thickness + attachment_overlap) / 2.0,
            ])

        elif style == 'block':
            min_xy = hull_2d.min(axis=0)
            max_xy = hull_2d.max(axis=0)
            size = max_xy - min_xy
            # 防止零尺寸
            size = np.maximum(size, 1.0)
            center_xy = (min_xy + max_xy) / 2.0
            pedestal = trimesh.creation.box(extents=[
                float(size[0]), float(size[1]), total_height,
            ])
            pedestal.apply_translation([
                float(center_xy[0]), float(center_xy[1]),
                (-thickness + attachment_overlap) / 2.0,
            ])
            radius = float(np.linalg.norm(size) / 2.0)
        else:
            return None, 0.0, 0.0

        # 倾斜侧面：复制顶点对底部 XY 做缩放
        if taper_angle > 0 and taper_angle < 90:
            try:
                angle_rad = np.radians(float(taper_angle))
                bottom_z = -thickness

                verts = pedestal.vertices.copy()
                for i in range(len(verts)):
                    z = verts[i, 2]
                    normalized = (z - bottom_z) / total_height if total_height > 0 else 1.0
                    # 越靠近底部越宽
                    expansion = (1.0 - normalized) * (thickness / np.tan(angle_rad))
                    direction = verts[i, :2] - np.array([float(centroid[0]), float(centroid[1])])
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-9:
                        verts[i, :2] += direction / direction_norm * expansion

                pedestal = trimesh.Trimesh(
                    vertices=verts, faces=pedestal.faces.copy(), process=False,
                )
            except Exception:
                # 锥化失败不致命，用垂直壁继续
                pass

        bottom_expansion = (
            thickness / np.tan(np.radians(float(taper_angle)))
            if 0 < taper_angle < 90 else 0.0
        )
        return pedestal, radius, radius + float(bottom_expansion)

    # ── 合并 ──────────────────────────────────────────

    def _merge_meshes(
        self, model: trimesh.Trimesh, pedestal: trimesh.Trimesh
    ) -> tuple[Optional[trimesh.Trimesh], str, str]:
        """将底座拼接到模型上。

        只有布尔结果为单一连通实体时才算成功。直接拼接不能保证人物
        与底座可共同打印，因此不再作为成功回退路径。
        """
        model_parts = list(model.split(only_watertight=False))
        boolean_error = ''
        try:
            merged = trimesh.boolean.union(
                [*model_parts, pedestal], engine=None,
            )
            if merged is not None and _mesh_ok(merged):
                count = _component_count(merged)
                if count == 1:
                    return merged, 'boolean_union', '布尔并集成功'
                boolean_error = f'布尔结果仍有 {count} 个连通体'
            else:
                boolean_error = '布尔后端未返回有效网格'
        except Exception as exc:
            boolean_error = f'{type(exc).__name__}: {exc}'

        # mini 环境通常没有 manifold3d；优先使用已随 DFM 修复环境提供的
        # pymeshlab/libigl 精确布尔，避免体素化整个模型造成细节和空腔损失。
        pymeshlab_error = ''
        try:
            import pymeshlab

            fused = pedestal.copy()
            for part in model_parts:
                mesh_set = pymeshlab.MeshSet()
                mesh_set.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=fused.vertices,
                        face_matrix=fused.faces,
                    ),
                    'current_union',
                )
                mesh_set.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=part.vertices,
                        face_matrix=part.faces,
                    ),
                    'next_part',
                )
                mesh_set.generate_boolean_union(first_mesh=0, second_mesh=1)
                output = mesh_set.current_mesh()
                fused = trimesh.Trimesh(
                    vertices=output.vertex_matrix(),
                    faces=output.face_matrix(),
                    process=True,
                )
                if not _mesh_ok(fused):
                    raise RuntimeError('pymeshlab 返回空网格')

            count = _component_count(fused)
            if count == 1:
                return fused, 'pymeshlab_boolean_union', 'pymeshlab/libigl 布尔并集成功'
            pymeshlab_error = f'pymeshlab 布尔结果仍有 {count} 个连通体'
        except Exception as exc:
            pymeshlab_error = f'{type(exc).__name__}: {exc}'

        if not bool(self.rules.get('pedestal_allow_voxel_fallback', True)):
            return (
                None,
                'boolean_union',
                f'trimesh 布尔失败: {boolean_error}; pymeshlab 布尔失败: {pymeshlab_error}',
            )

        try:
            combined = trimesh.util.concatenate([*model_parts, pedestal])
            pitch = float(self.rules.get('pedestal_voxel_pitch_mm', 0.4))
            max_cells = int(self.rules.get('pedestal_voxel_max_cells', 12_000_000))
            grid_shape = np.ceil(combined.extents / pitch).astype(np.int64) + 3
            estimated_cells = int(np.prod(grid_shape, dtype=np.int64))
            if estimated_cells > max_cells:
                pitch *= (estimated_cells / max_cells) ** (1.0 / 3.0)

            voxels = combined.voxelized(pitch).fill()
            merged = voxels.marching_cubes
            merged.apply_transform(voxels.transform)
            merged.process(validate=True)
            count = _component_count(merged)
            if _mesh_ok(merged) and count == 1:
                return (
                    merged,
                    'voxel_union',
                    f'体素融合成功 (pitch={pitch:.3f}mm；trimesh: {boolean_error}; '
                    f'pymeshlab: {pymeshlab_error})',
                )
            return None, 'voxel_union', f'体素融合后仍有 {count} 个连通体'
        except Exception as exc:
            return (
                None,
                'voxel_union',
                f'trimesh 布尔失败: {boolean_error}; pymeshlab 布尔失败: '
                f'{pymeshlab_error}; 体素融合失败: {type(exc).__name__}: {exc}',
            )

    # ── 可用性查询 ────────────────────────────────────

    @property
    def available(self) -> bool:
        """当前环境是否至少存在一种实体融合后端。"""
        try:
            if bool(trimesh.boolean.engines_available):
                return True
        except Exception:
            pass
        try:
            import pymeshlab  # noqa: F401
            return True
        except ImportError:
            pass
        if not bool(self.rules.get('pedestal_allow_voxel_fallback', True)):
            return False
        try:
            import skimage  # noqa: F401
            return True
        except ImportError:
            return False


# ── 模块级辅助 ────────────────────────────────────────────

def _circle_vertices(
    center: np.ndarray, radius: float, n: int = 36
) -> np.ndarray:
    """生成近似圆的顶点。"""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])


def _mesh_ok(mesh) -> bool:
    """检查网格是否至少包含有效几何。"""
    return (
        isinstance(mesh, trimesh.Trimesh)
        and len(mesh.vertices) >= 3
        and len(mesh.faces) >= 1
        and np.all(np.isfinite(mesh.vertices))
    )


def _component_count(mesh: trimesh.Trimesh) -> int:
    """安全统计面连通分量；无法拆分时视为融合失败。"""
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return 0
