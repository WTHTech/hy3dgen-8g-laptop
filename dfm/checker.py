"""DFM 可制造性检查。

快速层只给出已有几何依据的结论；尚未实现或计算失败的项目返回
``UNKNOWN``，不会伪装成通过，也不会进入评分。
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .rules import DFMRules


# ── 检查结果数据结构 ───────────────────────────────────

class CheckStatus(str, Enum):
    """单项检查的四态结果。"""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class CheckResult:
    """单项检查结果。"""

    code: str                 # 检查编码，如 'G1', 'S1'
    name: str                 # 检查名称，如 '水密性'
    category: str             # 分类：precheck | topology | overhang
    passed: Optional[bool] = None  # 兼容旧调用；UNKNOWN/N/A 时为 None
    score: Optional[float] = 100.0  # 0~100；UNKNOWN/N/A 不评分
    detail: str = ''          # 人类可读的详情
    metrics: dict = field(default_factory=dict)  # 量化指标
    blocking: bool = True     # 是否阻断级（不通过则终止后续检查）
    status: CheckStatus | str | None = None

    def __post_init__(self):
        if self.status is None:
            self.status = CheckStatus.PASS if self.passed else CheckStatus.FAIL
        else:
            self.status = CheckStatus(self.status)

        if self.status == CheckStatus.PASS:
            self.passed = True
        elif self.status == CheckStatus.FAIL:
            self.passed = False
        else:
            self.passed = None
            self.score = None

    def to_dict(self) -> dict:
        return {
            'code': self.code,
            'name': self.name,
            'category': self.category,
            'status': self.status.value,
            'passed': self.passed,
            'score': self.score,
            'detail': self.detail,
            'metrics': self.metrics,
            'blocking': self.blocking,
        }


@dataclass
class DFMReport:
    """一次 DFM 检查的完整报告。"""

    passed: bool = False      # 全部检查是否通过
    results: list = field(default_factory=list)
    total_score: float = 0.0
    summary: str = ''
    status: str = 'INCOMPLETE'
    complete: bool = False
    units: str = 'mm'
    transform: list = field(default_factory=list)
    prepared_mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        mesh_info = None
        if self.prepared_mesh is not None:
            mesh_info = {
                'vertex_count': int(len(self.prepared_mesh.vertices)),
                'face_count': int(len(self.prepared_mesh.faces)),
                'bounds_mm': self.prepared_mesh.bounds.tolist(),
            }
        return {
            'status': self.status,
            'passed': self.passed,
            'complete': self.complete,
            'total_score': self.total_score,
            'summary': self.summary,
            'units': self.units,
            'transform': self.transform,
            'prepared_mesh': mesh_info,
            'results': [result.to_dict() for result in self.results],
        }

    def to_json(self, path: Optional[str | Path] = None) -> str:
        """序列化报告；可选写入 UTF-8 JSON 文件（网格本体不嵌入）。"""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).write_text(payload, encoding='utf-8')
        return payload


# ── 快速粗筛器 ────────────────────────────────────────

class DFMChecker:
    """DFM 检查器 — 快速拓扑层 + 实验性精检层。

    Parameters
    ----------
    rules : DFMRules
        阈值规则对象

    Usage::

        rules = DFMRules('dfm_config.yaml')
        checker = DFMChecker(rules)
        mesh = trimesh.load('model.stl')
        report = checker.check_quick(mesh)
        if report.passed:
            print(f'粗筛通过，总分 {report.total_score:.0f}')
        else:
            for r in report.results:
                if not r.passed:
                    print(f'  [{r.code}] {r.name}: {r.detail}')
    """

    def __init__(self, rules: DFMRules):
        self.rules = rules

    # ── 入口 ──────────────────────────────────────────

    def check_quick(self, mesh: trimesh.Trimesh,
                    target_height: float = 100.0) -> DFMReport:
        """执行 10 项快速粗筛（P0~P2、G1~G6、S1）。

        Parameters
        ----------
        mesh : trimesh.Trimesh
            待检查网格
        target_height : float
            P0 单位标准化的目标高度(mm)，默认 100mm 手办尺寸

        Returns
        -------
        DFMReport
        """
        report = DFMReport()

        # ── 前置处理：返回的 prepared_mesh 才是后续检查和导出的唯一对象 ──
        mesh = self._prepare_mesh(mesh, target_height, report)
        if mesh is None:
            self._finalize_report(report)
            return report
        self._check_bottom_platform(mesh, report)
        self._check_build_volume(mesh, report)

        # ── 几何拓扑 ──
        self._check_watertight(mesh, report)
        self._check_self_intersection(mesh, report)
        self._check_non_manifold_edges(mesh, report)
        self._check_winding(mesh, report)
        self._check_degenerate_faces(mesh, report)
        self._check_isolated_fragments(mesh, report)

        # ── 悬垂 ──
        self._check_overhang_angle(mesh, report)

        # 汇总
        self._finalize_report(report)
        return report

    def check_detailed(self, mesh: trimesh.Trimesh) -> DFMReport:
        """执行 13 项精检候选（壁厚/空腔/支撑/尺寸）。

        应在粗筛通过后调用。返回独立的 DFMReport。
        """
        report = DFMReport()
        error = self._mesh_validation_error(mesh)
        if error:
            report.results.append(CheckResult(
                code='P0', name='输入网格', category='precheck',
                passed=False, score=0, detail=error,
            ))
            self._finalize_report(report)
            return report
        mesh = mesh.copy()
        report.prepared_mesh = mesh

        # ── 壁厚与结构 ──
        self._check_wall_thickness(mesh, report)
        self._check_min_feature_size(mesh, report)
        self._check_slenderness(mesh, report)
        self._check_lattice_wall(mesh, report)

        # ── 空腔 ──
        self._check_cavity(mesh, report)
        self._check_drain_hole(mesh, report)

        # ── 悬垂与支撑（补充） ──
        self._check_bridging(mesh, report)
        self._check_floating_islands(mesh, report)
        self._check_overhang_contact_area(mesh, report)
        self._check_support_interference(mesh, report)

        # ── 尺寸与装配 ──
        self._check_hole_diameter(mesh, report)
        self._check_assembly_gap(mesh, report)
        self._check_thread_feature(mesh, report)

        self._finalize_report(report)
        return report

    def check_full(self, mesh: trimesh.Trimesh,
                   target_height: float = 100.0) -> DFMReport:
        """执行全量检查：粗筛 10 项 + 精检 13 项，合并为一份报告。

        粗筛任一阻断项失败则跳过精检，直接返回阻断报告。
        """
        quick = self.check_quick(mesh, target_height)
        if quick.status != 'PASS':
            quick.summary = 'BLOCKED: 粗筛阻断，跳过精检 — ' + quick.summary
            return quick

        detailed = self.check_detailed(quick.prepared_mesh)
        # 合并结果
        quick.results.extend(detailed.results)
        self._finalize_report(quick, weighted=True)
        return quick

    # ── P0: 单位标准化 ────────────────────────────────

    def _mesh_validation_error(self, mesh) -> Optional[str]:
        if not isinstance(mesh, trimesh.Trimesh):
            return f'输入必须是 trimesh.Trimesh，实际为 {type(mesh).__name__}'
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return '输入网格为空，必须同时包含顶点和三角面'
        if not np.isfinite(mesh.vertices).all():
            return '输入网格包含 NaN 或无穷坐标'
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
            return '输入必须是三角网格'
        return None

    def _prepare_mesh(self, mesh: trimesh.Trimesh, target_height: float,
                      report: DFMReport) -> Optional[trimesh.Trimesh]:
        """复制、验证网格，将归一化模型缩放至毫米并放置到 Z=0。"""
        error = self._mesh_validation_error(mesh)
        if error:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0, detail=error,
            ))
            return None
        if not isinstance(target_height, (int, float)) or not np.isfinite(target_height) or target_height <= 0:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0,
                detail=f'target_height 必须是正的有限数，实际为 {target_height!r}',
            ))
            return None

        mesh = mesh.copy()
        bounds = mesh.bounds
        current_h = bounds[1, 2] - bounds[0, 2]  # Z 轴高度
        if not np.isfinite(bounds).all() or current_h <= 1e-9:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0, detail='网格边界无效或 Z 轴高度为零',
            ))
            return None

        # 判定是否为归一化坐标：Z 高度 < 5 且整体范围 < 10
        is_normalized = current_h < 5.0 and np.ptp(bounds) < 10.0
        scale = float(target_height / current_h) if is_normalized else 1.0
        mesh.apply_scale(scale)
        min_z = float(mesh.bounds[0, 2])
        mesh.apply_translation([0, 0, -min_z])
        new_h = float(mesh.extents[2])

        transform = np.eye(4)
        transform[:3, :3] *= scale
        transform[2, 3] = -min_z
        report.transform = transform.tolist()
        report.prepared_mesh = mesh

        if is_normalized:
            result = CheckResult(
                code='P0', name='单位标准化', category='precheck',
                passed=True, score=100,
                detail=f'归一化坐标({current_h:.2f}) → 物理尺寸({new_h:.1f}mm)，比例 {scale:.1f}，已移至平台',
                metrics={'original_height': float(current_h),
                         'target_height': target_height,
                         'scale_factor': float(scale),
                         'final_height': float(new_h)},
            )
        else:
            result = CheckResult(
                code='P0', name='单位标准化', category='precheck',
                passed=True, score=100,
                detail=f'按毫米处理 (高度 {current_h:.1f}mm)，并移动至 Z=0 平台',
                metrics={'height_mm': float(current_h),
                         'scale_factor': 1.0, 'z_translation': -min_z},
            )
        report.results.append(result)
        return mesh

    # ── P1: 底面平台 ──────────────────────────────────

    def _check_bottom_platform(self, mesh: trimesh.Trimesh,
                               report: DFMReport):
        """校验底面接触平台的面积是否充足。"""
        min_z = mesh.vertices[:, 2].min()
        eps = self.rules.get('z_tolerance', 1e-5)
        min_area = self.rules.get('min_bottom_area', 5.0)

        metrics = {'min_z': float(min_z)}

        # 负高度穿透
        if min_z < -eps:
            report.results.append(CheckResult(
                code='P1', name='底面平台', category='precheck',
                passed=False, score=0,
                detail=f'模型顶点低于打印平台 (min_z={min_z:.3f}mm)',
                metrics=metrics,
            ))
            return

        # 提取底面附近顶点 → 投影 → 凸包面积
        bottom_mask = mesh.vertices[:, 2] <= min_z + eps
        if bottom_mask.sum() < 3:
            report.results.append(CheckResult(
                code='P1', name='底面平台', category='precheck',
                passed=False, score=0,
                detail=f'底面接触顶点不足 ({bottom_mask.sum()} 个)，无法计算接触面积',
                metrics=metrics,
            ))
            return

        bottom_pts = mesh.vertices[bottom_mask, :2]  # XY 投影
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(bottom_pts)
            contact_area = float(hull.volume)  # 2D convex hull area
        except Exception:
            # 退化情况（点共线/重合）→ 接触面积为 0
            contact_area = 0.0

        metrics['contact_area'] = contact_area

        if contact_area < min_area:
            report.results.append(CheckResult(
                code='P1', name='底面平台', category='precheck',
                passed=False, score=max(0, contact_area / min_area * 100),
                detail=f'底面接触面积不足 ({contact_area:.1f}mm² < {min_area}mm²)，翘边风险',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='P1', name='底面平台', category='precheck',
                passed=True, score=100,
                detail=f'底面接触面积 {contact_area:.1f}mm² ≥ {min_area}mm²',
                metrics=metrics,
            ))

    # ── P2: 外框超限 ──────────────────────────────────

    def _check_build_volume(self, mesh: trimesh.Trimesh,
                            report: DFMReport):
        """校验模型外框是否在打印机最大成型空间内。"""
        build_vol = self.rules.get('build_volume', [220, 220, 250])
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        metrics = {
            'size_x': float(size[0]), 'size_y': float(size[1]),
            'size_z': float(size[2]),
            'build_x': build_vol[0], 'build_y': build_vol[1],
            'build_z': build_vol[2],
        }

        overs = []
        for i, axis in enumerate(['X', 'Y', 'Z']):
            if size[i] > build_vol[i]:
                overs.append(f'{axis}={size[i]:.1f} > {build_vol[i]}mm')

        if overs:
            report.results.append(CheckResult(
                code='P2', name='外框超限', category='precheck',
                passed=False, score=0,
                detail=f'模型超出成型空间: {", ".join(overs)}',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='P2', name='外框超限', category='precheck',
                passed=True, score=100,
                detail=f'外框 {size[0]:.0f}×{size[1]:.0f}×{size[2]:.0f}mm 在 {build_vol[0]}×{build_vol[1]}×{build_vol[2]}mm 内',
                metrics=metrics,
            ))

    # ── G1: 水密性 ────────────────────────────────────

    def _check_watertight(self, mesh: trimesh.Trimesh,
                          report: DFMReport):
        """检测网格是否水密（封闭流形）。"""
        is_wt = bool(mesh.is_watertight)
        if is_wt:
            report.results.append(CheckResult(
                code='G1', name='水密性', category='topology',
                passed=True, score=100,
                detail='网格水密，无破洞或缝隙',
            ))
        else:
            report.results.append(CheckResult(
                code='G1', name='水密性', category='topology',
                passed=False, score=0,
                detail='网格非水密，存在破洞或缝隙 → 切片可能失败',
            ))

    # ── G2: 自相交 ────────────────────────────────────

    def _check_self_intersection(self, mesh: trimesh.Trimesh,
                                 report: DFMReport):
        """检测三角面片是否互相穿插（通过 pymeshlab 选择自相交面）。"""
        max_faces = int(self.rules.get('self_intersection_max_faces', 200_000))
        if len(mesh.faces) > max_faces:
            self._append_unknown(
                report, 'G2', '自相交', 'topology',
                f'面数 {len(mesh.faces)} 超过同步检测上限 {max_faces}，需提交服务器异步检测',
                blocking=True,
            )
            return
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            # 从 trimesh 获取顶点和面片
            ms.add_mesh(
                pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices.astype(np.float64),
                    face_matrix=mesh.faces.astype(np.uint32),
                )
            )
            ms.compute_selection_by_self_intersections_per_face()
            # 检查是否有被选中的自相交面
            face_sel = ms.current_mesh().face_selection_array()
            si_count = int(face_sel.sum()) if face_sel is not None else 0
        except Exception as e:
            self._append_unknown(
                report, 'G2', '自相交', 'topology',
                f'自相交检测未完成: {e}', blocking=True,
            )
            return

        if si_count > 0:
            report.results.append(CheckResult(
                code='G2', name='自相交', category='topology',
                passed=False, score=0,
                detail=f'存在 {si_count} 个自相交面片 → 切片断层/报错',
                metrics={'self_intersecting_faces': si_count},
            ))
        else:
            report.results.append(CheckResult(
                code='G2', name='自相交', category='topology',
                passed=True, score=100,
                detail='无自相交面片',
            ))

    # ── G3: 非流形边 ──────────────────────────────────

    def _check_non_manifold_edges(self, mesh: trimesh.Trimesh,
                                  report: DFMReport):
        """检测被三个或更多三角面共用的非流形边。"""
        try:
            counts = np.bincount(mesh.edges_unique_inverse)
            non_manifold = int(np.count_nonzero(counts > 2))
        except Exception as exc:
            self._append_unknown(
                report, 'G3', '非流形边', 'topology',
                f'非流形边检测未完成: {exc}', blocking=True,
            )
            return
        report.results.append(CheckResult(
            code='G3', name='非流形边', category='topology',
            passed=non_manifold == 0,
            score=100 if non_manifold == 0 else 0,
            detail=('无三面及以上共用边' if non_manifold == 0
                    else f'发现 {non_manifold} 条非流形边'),
            metrics={'non_manifold_edge_count': non_manifold},
        ))

    # ── G4: 面绕序/法向 ───────────────────────────────

    def _check_winding(self, mesh: trimesh.Trimesh,
                       report: DFMReport):
        """检查相邻面绕序；水密网格同时检查整体朝向。"""
        try:
            consistent = bool(mesh.is_winding_consistent)
            signed_volume = float(mesh.volume) if mesh.is_watertight else None
            outward = signed_volume is None or signed_volume > 0
            passed = consistent and outward
        except Exception as exc:
            self._append_unknown(
                report, 'G4', '法向一致性', 'topology',
                f'法向检测未完成: {exc}', blocking=True,
            )
            return
        problems = []
        if not consistent:
            problems.append('相邻面绕序不一致')
        if not outward:
            problems.append('水密网格整体法向朝内')
        report.results.append(CheckResult(
            code='G4', name='法向一致性', category='topology',
            passed=passed, score=100 if passed else 0,
            detail='面绕序一致且整体朝外' if passed else '；'.join(problems),
            metrics={'winding_consistent': consistent,
                     'signed_volume': signed_volume},
        ))

    # ── G5: 重复/退化面 ───────────────────────────────

    def _check_degenerate_faces(self, mesh: trimesh.Trimesh,
                                report: DFMReport):
        """检测顶点索引重复的面和零面积三角面。"""
        try:
            canonical = np.sort(mesh.faces, axis=1)
            duplicate_count = int(len(canonical) - len(np.unique(canonical, axis=0)))
            tolerance = max(float(mesh.scale) ** 2 * 1e-12, 1e-12)
            degenerate_count = int(np.count_nonzero(mesh.area_faces <= tolerance))
            passed = duplicate_count == 0 and degenerate_count == 0
        except Exception as exc:
            self._append_unknown(
                report, 'G5', '重复/退化面', 'topology',
                f'重复/退化面检测未完成: {exc}', blocking=True,
            )
            return
        report.results.append(CheckResult(
            code='G5', name='重复/退化面', category='topology',
            passed=passed, score=100 if passed else 0,
            detail=('无重复或退化三角面' if passed else
                    f'重复面 {duplicate_count} 个，退化面 {degenerate_count} 个'),
            metrics={'duplicate_face_count': duplicate_count,
                     'degenerate_face_count': degenerate_count,
                     'area_tolerance': tolerance},
        ))

    # ── G6: 孤立碎面 ──────────────────────────────────

    def _check_isolated_fragments(self, mesh: trimesh.Trimesh,
                                  report: DFMReport):
        """检测与主体不连通的悬浮碎面。"""
        try:
            components = mesh.split(only_watertight=False)
        except Exception as exc:
            self._append_unknown(
                report, 'G6', '孤立碎面', 'topology',
                f'连通分量拆分未完成: {exc}', blocking=True,
            )
            return

        if len(components) <= 1:
            report.results.append(CheckResult(
                code='G6', name='孤立碎面', category='topology',
                passed=True, score=100,
                detail=f'单连通分量，无孤立碎面',
                metrics={'component_count': 1},
            ))
            return

        # 按面积排序，主体为最大分量
        areas = [c.area for c in components]
        total_area = sum(areas)
        main_area = max(areas)
        small = [(i, a) for i, a in enumerate(areas) if a < main_area * 0.01]
        frag_count = len(small)

        metrics = {
            'component_count': len(components),
            'total_area': float(total_area),
            'fragment_count': frag_count,
        }

        if frag_count > 0:
            frag_total = sum(a for _, a in small)
            report.results.append(CheckResult(
                code='G6', name='孤立碎面', category='topology',
                passed=frag_count == 0,
                score=max(60, 100 - frag_count * 10),
                detail=f'发现 {frag_count} 个孤立碎面（总面积 {frag_total:.2f}mm²），建议清理',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='G6', name='孤立碎面', category='topology',
                passed=True, score=100,
                detail=f'{len(components)} 个连通分量均无碎面',
                metrics=metrics,
            ))

    # ── S1: 悬垂角 ────────────────────────────────────

    def _check_overhang_angle(self, mesh: trimesh.Trimesh,
                              report: DFMReport):
        """按面积检测朝下且接近水平的悬垂面，排除贴平台底面。"""
        angle_threshold = self.rules.get('critical_angle', 45)
        max_ratio = self.rules.get('max_overhang_area_ratio', 0.30)
        normals = mesh.face_normals
        z_component = normals[:, 2]
        # downward horizontal: nz=-1 -> severe; vertical wall: nz=0 -> safe
        overhang_mask = z_component < -np.cos(np.radians(angle_threshold))

        face_z = mesh.vertices[mesh.faces][:, :, 2]
        bed_tol = max(float(self.rules.get('z_tolerance', 1e-5)), 1e-5)
        on_build_plate = np.all(face_z <= bed_tol, axis=1)
        overhang_mask &= ~on_build_plate

        areas = mesh.area_faces
        total_area = float(areas.sum())
        overhang_area = float(areas[overhang_mask].sum())
        overhang_count = int(overhang_mask.sum())
        overhang_ratio = overhang_area / total_area if total_area > 0 else 0.0

        metrics = {
            'overhang_face_count': overhang_count,
            'total_face_count': len(mesh.faces),
            'overhang_area_mm2': overhang_area,
            'total_area_mm2': total_area,
            'overhang_ratio': float(overhang_ratio),
            'critical_angle': angle_threshold,
        }

        if overhang_ratio > max_ratio:
            report.results.append(CheckResult(
                code='S1', name='悬垂角', category='overhang',
                passed=False, score=max(0, 100 - overhang_ratio * 100),
                detail=f'非底面悬垂面积占比 {overhang_ratio:.1%} > {max_ratio:.0%}，大量区域需支撑',
                metrics=metrics,
            ))
        elif overhang_ratio > 0.1:
            report.results.append(CheckResult(
                code='S1', name='悬垂角', category='overhang',
                passed=True, score=100 - overhang_ratio * 50,
                detail=f'悬垂面占比 {overhang_ratio:.1%}，少量区域需支撑',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='S1', name='悬垂角', category='overhang',
                passed=True, score=100,
                detail=f'悬垂面占比 {overhang_ratio:.1%}，打印姿态良好',
                metrics=metrics,
            ))

    # ── W1: 最小壁厚 ──────────────────────────────────

    def _check_wall_thickness(self, mesh: trimesh.Trimesh,
                              report: DFMReport):
        """射线法估算全局最小壁厚。"""
        min_wall = self.rules.get('min_wall_thickness', 1.2)
        try:
            sample_count = int(self.rules.get('wall_sample_count', 800))
            samples, face_idx = trimesh.sample.sample_surface(
                mesh, sample_count, seed=0,
            )
            normals = mesh.face_normals[face_idx]
            inward = -normals
            epsilon = max(float(mesh.scale) * 1e-7, 1e-6)
            origins = samples + inward * epsilon
            locations, index_ray, _ = mesh.ray.intersects_location(
                ray_origins=origins, ray_directions=inward,
                multiple_hits=True,
            )
            if len(locations) == 0:
                self._append_unknown(
                    report, 'W1', '最小壁厚', 'wall',
                    '壁厚射线没有命中对侧表面，无法给出可靠结论',
                    blocking=False,
                )
                return
            else:
                distances = np.linalg.norm(locations - origins[index_ray], axis=1)
                valid = distances > epsilon * 10
                distances = distances[valid]
                hit_rays = index_ray[valid]
                nearest = {}
                for ray_id, distance in zip(hit_rays, distances):
                    ray_id = int(ray_id)
                    nearest[ray_id] = min(nearest.get(ray_id, float('inf')), float(distance))
                thicknesses = np.asarray(list(nearest.values()), dtype=float)
            if len(thicknesses) < max(10, sample_count // 20):
                self._append_unknown(
                    report, 'W1', '最小壁厚', 'wall',
                    f'仅 {len(thicknesses)}/{sample_count} 条射线获得有效壁厚，样本不足',
                    blocking=False,
                )
                return
            p5 = float(np.percentile(thicknesses, 5))
            p50 = float(np.percentile(thicknesses, 50))
            metrics = {
                'thickness_p5': round(p5, 3),
                'thickness_p50': round(p50, 3),
                'valid_ray_count': len(thicknesses),
                'sample_count': sample_count,
            }
        except Exception as e:
            self._append_unknown(
                report, 'W1', '最小壁厚', 'wall',
                f'壁厚采样未完成: {e}', blocking=False,
            )
            return

        if p5 < min_wall:
            report.results.append(CheckResult(
                code='W1', name='最小壁厚', category='wall',
                passed=False, score=max(0, p5 / min_wall * 100),
                detail=f'第5百分位壁厚 {p5:.2f}mm < {min_wall}mm，局部过薄',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='W1', name='最小壁厚', category='wall',
                passed=True, score=100,
                detail=f'壁厚 P5={p5:.2f}mm, P50={p50:.2f}mm，满足最小值 {min_wall}mm',
                metrics=metrics,
            ))

    # ── W2: 最小特征尺寸 ──────────────────────────────

    def _check_min_feature_size(self, mesh: trimesh.Trimesh,
                                report: DFMReport):
        """最小特征需基于局部厚度/骨架，而不能用三角边长代替。"""
        self._append_unknown(
            report, 'W2', '最小特征尺寸', 'wall',
            '尚未接入局部厚度/骨架分析；三角边长只反映网格密度，已停止据此判定',
            blocking=False,
        )

    # ── W3: 细长比 ────────────────────────────────────

    def _check_slenderness(self, mesh: trimesh.Trimesh,
                           report: DFMReport):
        """细长比必须定位局部杆状结构，不能使用整模包围盒。"""
        self._append_unknown(
            report, 'W3', '局部细长比', 'wall',
            '待接入局部骨架分段后计算；整模高度/中截面宽度会误判，已停止使用',
            blocking=False,
        )

    # ── W4: 镂空点阵壁厚 ──────────────────────────────

    def _check_lattice_wall(self, mesh: trimesh.Trimesh,
                            report: DFMReport):
        """镂空点阵壁厚检测（简化版：检查是否有极薄区域）。

        完整 SDF 体素化方案待 CuraEngine 集成后实施，当前用射线法近似。
        """
        report.results.append(CheckResult(
            code='W4', name='镂空点阵壁厚', category='wall',
            status=CheckStatus.UNKNOWN, score=None,
            detail='镂空壁厚精检尚未实现，当前不参与通过判定和评分',
            blocking=False,
        ))

    # ── C1: 封闭空腔 ──────────────────────────────────

    def _check_cavity(self, mesh: trimesh.Trimesh,
                      report: DFMReport):
        """SDF 体素法检测完全封闭空腔。"""
        self._append_unknown(
            report, 'C1', '封闭空腔', 'cavity',
            '封闭空腔需对“空域”体素做边界洪泛；原算法检测的是实体内部，已停用',
            blocking=False,
        )
        return
        min_vol = self.rules.get('min_cavity_volume', 1.0)
        pitch = self.rules.get('sdf_pitch', 1.5)
        try:
            from scipy import ndimage
            bounds = mesh.bounds
            size = bounds[1] - bounds[0]
            # 安全限制：网格点数不超过 500K，自动增大 pitch
            max_voxels = 500_000
            est_voxels = (size[0] / pitch) * (size[1] / pitch) * (size[2] / pitch)
            if est_voxels > max_voxels:
                pitch = max(pitch, (size[0] * size[1] * size[2] / max_voxels) ** (1 / 3))
            axes = [np.arange(bounds[0, i] + pitch / 2,
                              bounds[1, i], pitch) for i in range(3)]
            if any(len(a) == 0 for a in axes):
                report.results.append(CheckResult(
                    code='C1', name='封闭空腔', category='cavity',
                    passed=True, score=100,
                    detail=f'模型尺寸过小，无法以 pitch={pitch}mm 体素化',
                ))
                return
            grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
            pts = grid.reshape(-1, 3)
            # SDF: 内部为负
            sdf = -trimesh.proximity.signed_distance(mesh, pts)
            interior = sdf < 0
            if interior.sum() == 0:
                report.results.append(CheckResult(
                    code='C1', name='封闭空腔', category='cavity',
                    passed=True, score=100,
                    detail='未检测到内部空腔',
                    metrics={'cavity_count': 0},
                ))
                return
            # 3D 连通域
            shape = tuple(len(a) for a in axes)
            labels, n = ndimage.label(interior.reshape(shape))
            # 排除触及边界的连通域（半开放凹槽/外部）
            cavity_count = 0
            cavities = []
            for lid in range(1, n + 1):
                mask = labels == lid
                voxel_count = int(mask.sum())
                volume = voxel_count * pitch ** 3
                if volume < min_vol:
                    continue
                # 检查是否触及体素网格边界（6面任一）
                touches_boundary = (
                    mask[0, :, :].any() or mask[-1, :, :].any()
                    or mask[:, 0, :].any() or mask[:, -1, :].any()
                    or mask[:, :, 0].any() or mask[:, :, -1].any()
                )
                if not touches_boundary:
                    cavity_count += 1
                    coords = np.argwhere(mask)
                    center = (coords.mean(axis=0) * pitch
                              + [axes[i][0] for i in range(3)])
                    cavities.append({
                        'volume': round(volume, 2),
                        'center': [round(float(c), 1) for c in center],
                    })
        except Exception as e:
            report.results.append(CheckResult(
                code='C1', name='封闭空腔', category='cavity',
                passed=False, score=50,
                detail=f'空腔 SDF 检测异常: {e}',
            ))
            return

        if cavity_count > 0:
            total_v = sum(c['volume'] for c in cavities)
            report.results.append(CheckResult(
                code='C1', name='封闭空腔', category='cavity',
                passed=False, score=max(0, 100 - cavity_count * 20),
                detail=f'发现 {cavity_count} 个封闭空腔（总体积 {total_v:.1f}mm³），需排液/排气',
                metrics={'cavity_count': cavity_count,
                         'total_volume': round(total_v, 1),
                         'cavities': cavities},
            ))
        else:
            report.results.append(CheckResult(
                code='C1', name='封闭空腔', category='cavity',
                passed=True, score=100,
                detail='无封闭空腔',
                metrics={'cavity_count': 0},
            ))

    # ── C2: 排液孔（光固化） ───────────────────────────

    def _check_drain_hole(self, mesh: trimesh.Trimesh,
                          report: DFMReport):
        """光固化中空模型必须含排液孔。"""
        if self.rules.process != 'SLA':
            self._append_na(report, 'C2', '排液孔', 'cavity', '仅适用于 SLA 工艺')
            return
        self._append_unknown(
            report, 'C2', '排液孔', 'cavity',
            'SLA 排液孔必须先识别空腔并验证其与外界连通，当前尚未实现',
            blocking=False,
        )

    # ── S2: 桥接检测 ──────────────────────────────────

    def _check_bridging(self, mesh: trimesh.Trimesh,
                        report: DFMReport):
        """简化桥接检测：检查层间悬空截面跨度。"""
        self._append_unknown(
            report, 'S2', '桥接检测', 'overhang',
            '桥接跨度需比较相邻层支撑区域；整层截面宽度不能代表桥接，原算法已停用',
            blocking=False,
        )
        return
        max_span = self.rules.get('bridge_max_span', 15.0)
        try:
            bbox = mesh.bounds
            h = bbox[1, 2] - bbox[0, 2]
            layer_h = self.rules.get('layer_height', 0.2)
            n_slices = max(5, int(h / (layer_h * 5)))  # 每5层抽检一次
            max_gap = 0.0
            for z in np.linspace(bbox[0, 2] + layer_h, bbox[1, 2], n_slices):
                s = mesh.section(plane_origin=[0, 0, z],
                                 plane_normal=[0, 0, 1])
                if s is None:
                    continue
                # 取截面轮廓的 AABB 最长边作为桥接跨度近似
                sb = s.bounds
                span = max(sb[1, 0] - sb[0, 0], sb[1, 1] - sb[0, 1])
                max_gap = max(max_gap, span)
            metrics = {'max_span': round(float(max_gap), 1),
                       'bridge_limit': max_span}
        except Exception as e:
            report.results.append(CheckResult(
                code='S2', name='桥接检测', category='overhang',
                passed=True, score=80,
                detail=f'桥接分析异常: {e}',
            ))
            return

        if max_gap > max_span:
            report.results.append(CheckResult(
                code='S2', name='桥接检测', category='overhang',
                passed=False, score=max(0, 100 - (max_gap - max_span) * 5),
                detail=f'最大桥接跨度 {max_gap:.1f}mm > {max_span}mm → 拉丝/塌陷风险',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='S2', name='桥接检测', category='overhang',
                passed=True, score=100,
                detail=f'最大截面跨度 {max_gap:.1f}mm <= {max_span}mm',
                metrics=metrics,
            ))

    # ── S3: 悬空孤岛 ──────────────────────────────────

    def _check_floating_islands(self, mesh: trimesh.Trimesh,
                                report: DFMReport):
        """分层切片 + XY 重叠检测悬空孤岛。

        仅当新增轮廓在上一层无重叠 XY 区域时才判定为孤岛，
        避免复杂模型的自然轮廓变化引起误报。
        """
        self._append_unknown(
            report, 'S3', '悬空孤岛', 'overhang',
            '需使用层面积与下层膨胀支撑区域的差集；当前轮廓算法会漏掉断层后的孤岛，已停用',
            blocking=False,
        )
        return
        try:
            bbox = mesh.bounds
            h = bbox[1, 2] - bbox[0, 2]
            layer_h = self.rules.get('layer_height', 0.2)
            n_slices = max(10, int(h / layer_h))
            prev_polygons = []  # 上一层的多边形列表（Shapely）
            island_count = 0
            island_zs = []
            for z in np.linspace(bbox[0, 2], bbox[1, 2], n_slices):
                s = mesh.section(plane_origin=[0, 0, z],
                                 plane_normal=[0, 0, 1])
                if s is None:
                    prev_polygons = []
                    continue
                try:
                    planar, _ = s.to_planar()
                    cur_polygons = list(planar.polygons_full)
                except Exception:
                    cur_polygons = []
                # 检查当前层每个轮廓是否与上层有 XY 重叠
                for poly in cur_polygons:
                    if poly.is_empty:
                        continue
                    has_support = any(
                        poly.intersects(prev) or poly.within(prev)
                        for prev in prev_polygons
                    ) if prev_polygons else True  # 第一层假支撑
                    if not has_support:
                        island_count += 1
                        island_zs.append(round(float(z), 1))
                prev_polygons = cur_polygons
            metrics = {'island_count': island_count,
                       'island_z_layers': island_zs[:5]}
        except Exception as e:
            report.results.append(CheckResult(
                code='S3', name='悬空孤岛', category='overhang',
                passed=True, score=70,
                detail=f'孤岛检测异常: {e}',
            ))
            return

        if island_count > 0:
            report.results.append(CheckResult(
                code='S3', name='悬空孤岛', category='overhang',
                passed=False, score=max(50, 100 - island_count * 15),
                detail=f'检测到 {island_count} 个悬空孤岛（Z={island_zs[:3]}...），打印时坍塌',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='S3', name='悬空孤岛', category='overhang',
                passed=True, score=100,
                detail='未检测到悬空孤岛',
                metrics=metrics,
            ))

    # ── S4: 最小支撑接触面积 ──────────────────────────

    def _check_overhang_contact_area(self, mesh: trimesh.Trimesh,
                                     report: DFMReport):
        """计算悬垂面在 XY 平面的投影面积，校验支撑接触是否充足。"""
        self._append_unknown(
            report, 'S4', '支撑接触面积', 'overhang',
            '必须在生成支撑后测量真实接触斑块；悬垂投影总面积不能替代接触面积',
            blocking=False,
        )
        return
        min_contact = self.rules.get('min_contact_area', 3.0)
        angle_threshold = self.rules.get('critical_angle', 45)
        try:
            normals = (mesh.face_normals if hasattr(mesh, 'face_normals')
                       else mesh.vertex_normals)
            cos_th = np.cos(np.radians(angle_threshold))
            overhang = normals[:, 2] < cos_th
            if overhang.sum() == 0:
                report.results.append(CheckResult(
                    code='S4', name='支撑接触面积', category='overhang',
                    passed=True, score=100,
                    detail='无悬垂面，无需支撑',
                    metrics={'overhang_projected_area': 0.0},
                ))
                return
            areas = (mesh.area_faces if hasattr(mesh, 'area_faces')
                     else np.ones(len(normals)))
            proj_area = float((areas * np.abs(normals[:, 2]) * overhang).sum())
            metrics = {'overhang_projected_area': round(proj_area, 2),
                       'min_contact_area': min_contact}
        except Exception as e:
            report.results.append(CheckResult(
                code='S4', name='支撑接触面积', category='overhang',
                passed=True, score=70,
                detail=f'接触面积计算异常: {e}',
            ))
            return

        if proj_area < min_contact:
            report.results.append(CheckResult(
                code='S4', name='支撑接触面积', category='overhang',
                passed=False, score=max(0, proj_area / min_contact * 100),
                detail=f'悬垂投影面积 {proj_area:.1f}mm² < {min_contact}mm²，支撑易脱落',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='S4', name='支撑接触面积', category='overhang',
                passed=True, score=100,
                detail=f'悬垂投影面积 {proj_area:.1f}mm² >= {min_contact}mm²',
                metrics=metrics,
            ))

    # ── S5: 支撑去除干涉 ───────────────────────────────

    def _check_support_interference(self, mesh: trimesh.Trimesh,
                                    report: DFMReport):
        """支撑-模型干涉检测（stub：需支撑生成后做布尔运算）。"""
        report.results.append(CheckResult(
            code='S5', name='支撑干涉', category='overhang',
            status=CheckStatus.UNKNOWN, score=None,
            detail='支撑干涉检测需在生成支撑后实施布尔运算，当前不参与判定和评分',
            blocking=False,
        ))

    # ── D1: 内孔最小孔径 ──────────────────────────────

    def _check_hole_diameter(self, mesh: trimesh.Trimesh,
                             report: DFMReport):
        """分层切片 + Shapely 内轮廓检测最小内孔直径。"""
        self._append_unknown(
            report, 'D1', '内孔孔径', 'dimension',
            '任意方向内孔需要轴向识别；仅做水平切片会漏检或误测，当前不参与判定',
            blocking=False,
        )
        return
        min_dia = self.rules.get('min_hole_diameter', 0.4)
        try:
            from shapely.geometry import Polygon
            bbox = mesh.bounds
            h = bbox[1, 2] - bbox[0, 2]
            n_slices = max(5, int(h / 2))  # 每 2mm 一层
            min_width = 999.0
            for z in np.linspace(bbox[0, 2] + 0.1, bbox[1, 2] - 0.1, n_slices):
                s = mesh.section(plane_origin=[0, 0, z],
                                 plane_normal=[0, 0, 1])
                if s is None:
                    continue
                try:
                    planar, _ = s.to_planar()
                    for poly in planar.polygons_full:
                        for interior in poly.interiors:
                            ip = Polygon(interior)
                            if ip.is_empty:
                                continue
                            w = ip.minimum_rotated_rectangle
                            ww = min(
                                w.exterior.coords[1][0] - w.exterior.coords[0][0],
                                w.exterior.coords[2][1] - w.exterior.coords[1][1],
                                key=abs)
                            ww = abs(ww)
                            if 0 < ww < min_width:
                                min_width = ww
                except Exception:
                    continue
            min_width = min_width if min_width < 999.0 else 0.0
            metrics = {'min_hole_diameter': round(float(min_width), 3),
                       'nozzle_diameter': min_dia}
        except ImportError:
            report.results.append(CheckResult(
                code='D1', name='内孔孔径', category='dimension',
                passed=True, score=70,
                detail='Shapely 未安装，跳过内孔检测；pip install shapely',
            ))
            return
        except Exception as e:
            report.results.append(CheckResult(
                code='D1', name='内孔孔径', category='dimension',
                passed=True, score=70,
                detail=f'内孔分析异常: {e}',
            ))
            return

        if min_width > 0 and min_width < min_dia:
            report.results.append(CheckResult(
                code='D1', name='内孔孔径', category='dimension',
                passed=False, score=max(0, min_width / min_dia * 100),
                detail=f'最小内孔 {min_width:.3f}mm < {min_dia}mm（喷嘴直径），打印堵死',
                metrics=metrics,
            ))
        elif min_width == 0.0:
            report.results.append(CheckResult(
                code='D1', name='内孔孔径', category='dimension',
                passed=True, score=100,
                detail='未检测到内孔结构',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='D1', name='内孔孔径', category='dimension',
                passed=True, score=100,
                detail=f'最小内孔 {min_width:.3f}mm >= {min_dia}mm',
                metrics=metrics,
            ))

    # ── D2: 装配间隙 ──────────────────────────────────

    def _check_assembly_gap(self, mesh: trimesh.Trimesh,
                            report: DFMReport):
        """多部件最近距离检测，小于阈值则有粘连风险。"""
        gap_th = self.rules.get('gap_threshold', 0.25)
        try:
            components = mesh.split(only_watertight=False)
        except Exception:
            self._append_unknown(
                report, 'D2', '装配间隙', 'dimension',
                '无法拆分连通分量，装配间隙检测未完成', blocking=False,
            )
            return

        if len(components) <= 1:
            self._append_na(
                report, 'D2', '装配间隙', 'dimension',
                '单部件模型，无装配间隙可检查',
                metrics={'component_count': len(components)},
            )
            return

        try:
            min_gap = float('inf')
            gap_pairs = []
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    pq = trimesh.proximity.ProximityQuery(components[i])
                    _, dist, _ = pq.on_surface(components[j].vertices)
                    d = float(dist.min()) if len(dist) > 0 else float('inf')
                    if d < min_gap:
                        min_gap = d
                    if d < gap_th:
                        gap_pairs.append((i, j, round(d, 3)))
            metrics = {'min_gap': round(float(min_gap), 3),
                       'component_count': len(components),
                       'gap_threshold': gap_th,
                       'risk_pairs': gap_pairs}
        except Exception as e:
            self._append_unknown(
                report, 'D2', '装配间隙', 'dimension',
                f'装配间隙检测未完成: {e}', blocking=False,
            )
            return

        if min_gap < gap_th:
            report.results.append(CheckResult(
                code='D2', name='装配间隙', category='dimension',
                passed=False, score=max(0, min_gap / gap_th * 100),
                detail=f'最小间隙 {min_gap:.3f}mm < {gap_th}mm，{len(gap_pairs)} 处粘连风险',
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='D2', name='装配间隙', category='dimension',
                passed=True, score=100,
                detail=f'最小间隙 {min_gap:.3f}mm >= {gap_th}mm',
                metrics=metrics,
            ))

    # ── D3: 螺纹特征 ──────────────────────────────────

    def _check_thread_feature(self, mesh: trimesh.Trimesh,
                              report: DFMReport):
        """螺纹特征校验（stub：需高频切片 + 周期分析）。"""
        report.results.append(CheckResult(
            code='D3', name='螺纹特征', category='dimension',
            status=CheckStatus.UNKNOWN, score=None,
            detail='螺纹检测尚未实现；若含标准螺纹请手动校验',
            blocking=False,
        ))

    # ── 辅助 ──────────────────────────────────────────

    def _append_unknown(self, report: DFMReport, code: str, name: str,
                        category: str, detail: str, blocking: bool = False,
                        metrics: Optional[dict] = None):
        report.results.append(CheckResult(
            code=code, name=name, category=category,
            status=CheckStatus.UNKNOWN, score=None, detail=detail,
            metrics=metrics or {}, blocking=blocking,
        ))

    def _append_na(self, report: DFMReport, code: str, name: str,
                   category: str, detail: str,
                   metrics: Optional[dict] = None):
        report.results.append(CheckResult(
            code=code, name=name, category=category,
            status=CheckStatus.NOT_APPLICABLE, score=None, detail=detail,
            metrics=metrics or {}, blocking=False,
        ))

    def _finalize_report(self, report: DFMReport, weighted: bool = False):
        failed = [r for r in report.results if r.status == CheckStatus.FAIL]
        unknown = [r for r in report.results if r.status == CheckStatus.UNKNOWN]
        report.complete = not unknown
        if failed:
            report.status = 'FAIL'
        elif unknown:
            report.status = 'INCOMPLETE'
        else:
            report.status = 'PASS'
        report.passed = report.status == 'PASS'
        report.total_score = (
            self._calc_weighted_score(report.results)
            if weighted else self._calc_total_score(report.results)
        )
        report.summary = self._make_summary(report)

    def _calc_total_score(self, results: list[CheckResult]) -> float:
        """只对实际完成的 PASS/FAIL 项计算算术平均分。"""
        scores = [
            float(r.score) for r in results
            if r.status in (CheckStatus.PASS, CheckStatus.FAIL)
            and r.score is not None
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _calc_weighted_score(self, results: list[CheckResult]) -> float:
        """按 DFM 文档六维权重计算加权总分（全量检查时使用）。"""
        dims = {'watertight_topology': [], 'wall_thickness': [],
                'overhang_support': [], 'dimension_compliance': [],
                'feature_structure': [], 'cavity': []}

        cat_map = {
            'topology': 'watertight_topology',
            'wall': 'wall_thickness',
            'overhang': 'overhang_support',
            'cavity': 'cavity',
            'dimension': 'dimension_compliance',
            'precheck': 'watertight_topology',
        }

        for r in results:
            if r.status not in (CheckStatus.PASS, CheckStatus.FAIL) or r.score is None:
                continue
            bucket = cat_map.get(r.category, 'feature_structure')
            dims[bucket].append(float(r.score))

        configured_weights = {
            dim: float(self.rules.get(dim, 0)) for dim in dims
        }
        total = 0.0
        total_weight = 0.0
        for dim, weight in configured_weights.items():
            scores = dims.get(dim, [])
            if scores:
                total += sum(scores) / len(scores) * weight
                total_weight += weight

        return total / total_weight if total_weight > 0 else 0.0

    def _make_summary(self, report: DFMReport) -> str:
        failed = [r for r in report.results if r.status == CheckStatus.FAIL]
        unknown = [r for r in report.results if r.status == CheckStatus.UNKNOWN]
        na_count = sum(r.status == CheckStatus.NOT_APPLICABLE for r in report.results)
        if failed:
            codes = ', '.join(r.code for r in failed)
            suffix = f'；另有 {len(unknown)} 项未完成' if unknown else ''
            return f'FAIL: {len(failed)} 项未通过 [{codes}]{suffix}'
        if unknown:
            codes = ', '.join(r.code for r in unknown)
            return f'INCOMPLETE: 已完成项无失败，{len(unknown)} 项未得出结论 [{codes}]'
        suffix = f'（{na_count} 项不适用）' if na_count else ''
        return f'PASS: 全部适用检查通过{suffix}'
