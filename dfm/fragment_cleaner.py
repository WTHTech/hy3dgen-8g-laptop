"""DFM 孤立碎面清理器。

G6 检测到悬浮/孤立碎片后自动删除，避免用户手动清理。
删除是破坏性的——面数会减少，但主体几何保持不变。

清理添加在底座之后、拓扑修复之前：删除碎片不会引入新的
拓扑缺陷，后续修复环不再受无关碎片干扰。

用法::

    from dfm import FragmentCleaner, DFMRules

    rules = DFMRules()
    cleaner = FragmentCleaner(rules)

    result = cleaner.clean(prepared_mesh)
    if result.cleaned:
        result.mesh.export('cleaned.stl')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Optional

import numpy as np
import trimesh

from .rules import DFMRules


@dataclass
class FragmentCleanResult:
    """碎片清理结果。"""

    success: bool = False         # 分类/清理流程是否完整执行
    cleaned: bool = False         # 是否有碎片被删除
    mesh: Optional[trimesh.Trimesh] = None
    components_before: int = 0
    components_after: int = 0
    removed_component_count: int = 0
    removed_face_count: int = 0
    removed_area_mm2: float = 0.0
    kept_component_count: int = 0
    kept_component_areas_mm2: list[float] = field(default_factory=list)
    kept_component_floating: list[bool] = field(default_factory=list)
    removed_components: list[dict] = field(default_factory=list)
    protected_components: list[dict] = field(default_factory=list)
    detail: str = ''
    actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'cleaned': self.cleaned,
            'components_before': self.components_before,
            'components_after': self.components_after,
            'removed_component_count': self.removed_component_count,
            'removed_face_count': self.removed_face_count,
            'removed_area_mm2': round(self.removed_area_mm2, 3),
            'kept_component_count': self.kept_component_count,
            'kept_component_areas_mm2': [
                round(a, 3) for a in self.kept_component_areas_mm2
            ],
            'kept_component_floating': self.kept_component_floating,
            'removed_components': self.removed_components,
            'protected_components': self.protected_components,
            'detail': self.detail,
            'actions': self.actions,
            'vertices': len(self.mesh.vertices) if self.mesh is not None else 0,
            'faces': len(self.mesh.faces) if self.mesh is not None else 0,
        }


class FragmentCleaner:
    """孤立碎面清理器。

    删除与主体不连通的小碎片和悬浮分量。采用保守策略：
    只删除同时满足相对面积、绝对面积和包围盒尺寸限制，并且与主体
    保持安全距离的明显碎屑。与主体相交或过近的分量一律保护，交给
    后续融合或人工复核。

    Parameters
    ----------
    rules : DFMRules, optional
        读取 ``z_tolerance`` 和碎片判定阈值。
    """

    # 默认阈值
    DEFAULT_MAX_FRAGMENT_RATIO = 0.01    # 碎片面积/主体面积上限
    DEFAULT_MAX_FRAGMENT_AREA_MM2 = 10.0 # mm²，绝对小碎片上限
    DEFAULT_MAX_FRAGMENT_EXTENT_MM = 2.0 # mm，包围盒最长边上限
    DEFAULT_PROTECT_DISTANCE_MM = 0.2    # mm，与主体过近时禁止自动删除

    def __init__(self, rules: Optional[DFMRules] = None):
        self.rules = rules or DFMRules()

    # ── 公开接口 ──────────────────────────────────────

    def clean(
        self,
        mesh: trimesh.Trimesh,
        max_fragment_ratio: Optional[float] = None,
        max_fragment_area_mm2: Optional[float] = None,
        max_fragment_extent_mm: Optional[float] = None,
        protect_distance_mm: Optional[float] = None,
    ) -> FragmentCleanResult:
        """清理网格中的孤立碎面和悬浮碎片。

        Parameters
        ----------
        mesh : trimesh.Trimesh
            prepared_mesh（毫米，底部在 Z≈0）。
        max_fragment_ratio : float, optional
            碎片相对于主体面积的比例上限；默认 0.01（1%）。
        max_fragment_area_mm2 : float, optional
            绝对碎片面积上限(mm²)。自动删除必须同时满足比例和面积限制。
        max_fragment_extent_mm : float, optional
            包围盒最长边上限(mm)。超过此尺寸的部件禁止自动删除。
        protect_distance_mm : float, optional
            与主体表面的保护距离(mm)。AABB 相交或距离不大于该值时保留。

        Returns
        -------
        FragmentCleanResult
        """
        # 参数归一化
        if max_fragment_ratio is None:
            max_fragment_ratio = self.rules.get(
                'fragment_max_ratio', self.DEFAULT_MAX_FRAGMENT_RATIO,
            )
        if max_fragment_area_mm2 is None:
            max_fragment_area_mm2 = self.rules.get(
                'fragment_max_area_mm2', self.DEFAULT_MAX_FRAGMENT_AREA_MM2,
            )
        if max_fragment_extent_mm is None:
            max_fragment_extent_mm = self.rules.get(
                'fragment_max_extent_mm', self.DEFAULT_MAX_FRAGMENT_EXTENT_MM,
            )
        if protect_distance_mm is None:
            protect_distance_mm = self.rules.get(
                'fragment_protect_distance_mm', self.DEFAULT_PROTECT_DISTANCE_MM,
            )
        for name, value in [
            ('max_fragment_ratio', max_fragment_ratio),
            ('max_fragment_area_mm2', max_fragment_area_mm2),
            ('max_fragment_extent_mm', max_fragment_extent_mm),
            ('protect_distance_mm', protect_distance_mm),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                return FragmentCleanResult(
                    detail=f'{name} 必须是正的有限数，实际为 {value!r}',
                )
        if not 0 < max_fragment_ratio < 1:
            return FragmentCleanResult(
                detail=(
                    'max_fragment_ratio 必须在 (0, 1) 内，'
                    f'实际为 {max_fragment_ratio!r}'
                ),
            )

        if not _mesh_ok(mesh):
            return FragmentCleanResult(
                detail='输入网格无效：顶点少于 3 或面数不足',
            )

        if getattr(mesh.visual, 'kind', None) == 'texture':
            return FragmentCleanResult(
                mesh=mesh.copy(),
                detail='检测到纹理网格；碎片拆分/重建可能破坏 UV 或材质，拒绝自动删除',
            )

        actions: list[dict] = []

        # 1. 拆分连通分量
        try:
            components = mesh.split(only_watertight=False)
        except Exception as exc:
            return FragmentCleanResult(
                detail=f'连通分量拆分失败: {exc}',
            )

        if len(components) <= 1:
            actions.append({
                'step': '拆分连通分量',
                'detail': '单连通分量，无需清理',
            })
            return FragmentCleanResult(
                success=True,
                cleaned=False,
                mesh=mesh.copy(),
                components_before=1,
                components_after=1,
                kept_component_count=1,
                kept_component_areas_mm2=[float(mesh.area)],
                kept_component_floating=[float(mesh.bounds[0, 2]) > float(
                    self.rules.get('z_tolerance', 1e-5)
                )],
                detail='单连通分量，无碎片可清理',
                actions=actions,
            )

        # 2. 分类：主体 vs 碎片
        areas = np.array([float(c.area) if hasattr(c, 'area') else 0.0
                          for c in components])
        bed_tol = max(float(self.rules.get('z_tolerance', 1e-5)), 1e-5)

        # 接触平台的判定：最低顶点 Z ≤ bed_tol
        touching_platform = np.array([
            float(c.bounds[0, 2]) <= bed_tol
            for c in components
        ], dtype=bool)

        # 主体：接触平台的分量中面积最大的
        platform_indices = np.where(touching_platform)[0]
        if len(platform_indices) > 0:
            main_idx = int(platform_indices[np.argmax(areas[platform_indices])])
        else:
            # 没有接触平台的 → 取面积最大的作为主体
            main_idx = int(np.argmax(areas))

        main_area = float(areas[main_idx])
        main_component = components[main_idx]

        # 3. 判定每个分量是否碎片
        keep_mask = np.ones(len(components), dtype=bool)
        removal_reasons: list[dict] = []
        protected_components: list[dict] = []

        for i in range(len(components)):
            if i == main_idx:
                continue

            area = float(areas[i])
            floating = not bool(touching_platform[i])
            ratio = area / main_area if main_area > 0 else 1.0
            max_extent = float(np.max(components[i].extents))
            aabb_overlap = _bounds_overlap(
                main_component.bounds,
                components[i].bounds,
                float(protect_distance_mm),
            )
            min_distance = _min_surface_distance(main_component, components[i])
            near_main = aabb_overlap or min_distance <= float(protect_distance_mm)

            # 真正保守的自动删除：三项尺寸条件必须同时满足，且必须与
            # 主体明确分离。任何相交/近邻分量都可能是眼睛、头饰或武器。
            small_by_all_limits = (
                ratio < float(max_fragment_ratio)
                and area < float(max_fragment_area_mm2)
                and max_extent < float(max_fragment_extent_mm)
            )
            remove = small_by_all_limits and not near_main

            record = _component_record(
                index=i,
                component=components[i],
                area=area,
                ratio=ratio,
                floating=floating,
                max_extent=max_extent,
                min_distance=min_distance,
                aabb_overlap=aabb_overlap,
            )

            if remove:
                keep_mask[i] = False
                record['reason'] = (
                    '同时满足相对面积、绝对面积和最长边限制，且与主体明确分离'
                )
                removal_reasons.append(record)
            else:
                if near_main:
                    record['reason'] = '与主体 AABB 相交或距离过近，可能是有意义配件'
                elif not small_by_all_limits:
                    failed_limits = []
                    if ratio >= float(max_fragment_ratio):
                        failed_limits.append('相对面积超限')
                    if area >= float(max_fragment_area_mm2):
                        failed_limits.append('绝对面积超限')
                    if max_extent >= float(max_fragment_extent_mm):
                        failed_limits.append('最长边超限')
                    record['reason'] = '、'.join(failed_limits) + '，禁止自动删除'
                protected_components.append(record)

        # 4. 执行删除
        kept = [c for i, c in enumerate(components) if keep_mask[i]]
        removed_count = int((~keep_mask).sum())

        if removed_count == 0:
            actions.append({
                'step': '分析碎片',
                'detail': f'{len(components)} 个分量均非碎片，未删除',
            })
            return FragmentCleanResult(
                success=True,
                cleaned=False,
                mesh=mesh.copy(),
                components_before=len(components),
                components_after=len(components),
                kept_component_count=len(components),
                kept_component_areas_mm2=[round(float(a), 3) for a in areas],
                kept_component_floating=[not bool(t) for t in touching_platform],
                protected_components=protected_components,
                detail=f'{len(components)} 个分量均非碎片，无需清理',
                actions=actions,
            )

        # 重建网格
        try:
            cleaned_mesh = trimesh.util.concatenate(kept)
        except Exception as exc:
            return FragmentCleanResult(
                detail=f'重建清理后网格失败: {exc}',
                actions=actions,
            )

        actions.append({
            'step': '删除碎片',
            'detail': f'移除 {removed_count} 个碎片（共 '
                      f'{sum(r["faces"] for r in removal_reasons)} 面），'
                      f'保留 {len(kept)} 个分量',
        })

        kept_areas = [float(areas[i]) for i in range(len(components)) if keep_mask[i]]
        kept_floating = [not bool(touching_platform[i]) for i in range(len(components)) if keep_mask[i]]

        total_removed_faces = sum(r['faces'] for r in removal_reasons)
        total_removed_area = sum(float(r['area_mm2']) for r in removal_reasons)

        return FragmentCleanResult(
            success=True,
            cleaned=True,
            mesh=cleaned_mesh,
            components_before=len(components),
            components_after=len(kept),
            removed_component_count=removed_count,
            removed_face_count=total_removed_faces,
            removed_area_mm2=total_removed_area,
            kept_component_count=len(kept),
            kept_component_areas_mm2=[round(a, 3) for a in kept_areas],
            kept_component_floating=kept_floating,
            removed_components=removal_reasons,
            protected_components=protected_components,
            detail=f'移除 {removed_count} 个碎片（{total_removed_faces} 面，'
                   f'{total_removed_area:.2f}mm²），'
                   f'保留 {len(kept)} 个分量',
            actions=actions,
        )

    # ── 可用性 ────────────────────────────────────────

    @property
    def available(self) -> bool:
        return True


def _mesh_ok(mesh) -> bool:
    return (
        isinstance(mesh, trimesh.Trimesh)
        and len(mesh.vertices) >= 3
        and len(mesh.faces) >= 1
        and np.all(np.isfinite(mesh.vertices))
    )


def _bounds_overlap(first: np.ndarray, second: np.ndarray, margin: float) -> bool:
    """判断两个三维 AABB 在给定保护距离内是否重叠。"""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return bool(
        np.all(first[0] <= second[1] + margin)
        and np.all(second[0] <= first[1] + margin)
    )


def _min_surface_distance(
    main: trimesh.Trimesh, component: trimesh.Trimesh,
) -> float:
    """估计组件顶点到主体表面的最小距离；失败时返回无穷大并保留审计。"""
    try:
        query = trimesh.proximity.ProximityQuery(main)
        _, distances, _ = query.on_surface(component.vertices)
        if len(distances) > 0 and np.all(np.isfinite(distances)):
            return float(np.min(distances))
    except Exception:
        pass
    return float('inf')


def _component_record(
    *,
    index: int,
    component: trimesh.Trimesh,
    area: float,
    ratio: float,
    floating: bool,
    max_extent: float,
    min_distance: float,
    aabb_overlap: bool,
) -> dict:
    """生成可序列化、可审计的组件记录。"""
    return {
        'index': int(index),
        'vertices': int(len(component.vertices)),
        'faces': int(len(component.faces)),
        'area_mm2': float(area),
        'area_ratio': float(ratio),
        'max_extent_mm': float(max_extent),
        'centroid_mm': [float(value) for value in component.centroid],
        'bounds_mm': [
            [float(value) for value in component.bounds[0]],
            [float(value) for value in component.bounds[1]],
        ],
        'floating': bool(floating),
        'aabb_overlap_with_main': bool(aabb_overlap),
        'min_distance_to_main_mm': (
            float(min_distance) if np.isfinite(min_distance) else None
        ),
    }
