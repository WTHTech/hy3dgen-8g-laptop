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
        report = checker.check_quick(mesh, input_units='mm')
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
                    target_height: float = 100.0,
                    input_units: str = 'auto') -> DFMReport:
        """执行 11 项快速粗筛（P0~P3、G1~G6、S1）。

        Parameters
        ----------
        mesh : trimesh.Trimesh
            待检查网格
        target_height : float
            P0 单位标准化的目标高度(mm)，默认 100mm 手办尺寸
        input_units : {'auto', 'mm', 'normalized'}
            ``normalized`` 用于 AI 归一化输出，``mm`` 用于已有物理模型。
            ``auto`` 遇到小尺寸歧义时返回 INCOMPLETE，不会擅自放大。

        Returns
        -------
        DFMReport
        """
        report = DFMReport()

        # ── 前置处理：返回的 prepared_mesh 才是后续检查和导出的唯一对象 ──
        mesh = self._prepare_mesh(mesh, target_height, report, input_units)
        if mesh is None:
            self._finalize_report(report)
            return report
        self._check_bottom_platform(mesh, report)
        self._check_stability(mesh, report)
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
                   target_height: float = 100.0,
                   input_units: str = 'auto') -> DFMReport:
        """执行全量检查：粗筛 11 项 + 精检 13 项，合并为一份报告。

        粗筛任一阻断项失败则跳过精检，直接返回阻断报告。
        """
        quick = self.check_quick(mesh, target_height, input_units=input_units)
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
                      report: DFMReport,
                      input_units: str = 'auto') -> Optional[trimesh.Trimesh]:
        """复制、验证网格，将归一化模型缩放至毫米并放置到 Z=0。"""
        error = self._mesh_validation_error(mesh)
        if error:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0, detail=error,
            ))
            return None
        mode = str(input_units).strip().lower()
        if mode not in {'auto', 'mm', 'normalized'}:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0,
                detail=f'input_units 必须是 auto/mm/normalized，实际为 {input_units!r}',
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
        if not np.isfinite(bounds).all() or float(mesh.extents.max()) <= 1e-9:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0, detail='网格边界无效或尺寸为零',
            ))
            return None

        current_h = float(mesh.extents[2])
        looks_normalized = current_h < 5.0 and float(mesh.extents.max()) < 10.0
        if mode == 'auto' and looks_normalized:
            self._append_unknown(
                report, 'P0', '单位标准化', 'precheck',
                '小尺寸模型可能是毫米零件，也可能是归一化 AI 输出；请显式指定 input_units="mm" 或 "normalized"',
                blocking=True,
                metrics={'height': float(current_h),
                         'extents': mesh.extents.tolist()},
            )
            return None
        is_normalized = mode == 'normalized'
        source_up_axis = 'z'
        orientation = np.eye(4)
        if is_normalized:
            source_up_axis = str(
                self.rules.get('normalized_up_axis', 'y')
            ).strip().lower()
            orientation = self._up_axis_to_z_transform(source_up_axis)
            mesh.apply_transform(orientation)

        oriented_h = float(mesh.extents[2])
        if oriented_h <= 1e-9:
            report.results.append(CheckResult(
                code='P0', name='单位与输入标准化', category='precheck',
                passed=False, score=0,
                detail=f'{source_up_axis.upper()} 轴转换到打印 Z 轴后高度为零',
            ))
            return None

        scale = float(target_height / oriented_h) if is_normalized else 1.0
        mesh.apply_scale(scale)
        min_z = float(mesh.bounds[0, 2])
        mesh.apply_translation([0, 0, -min_z])
        new_h = float(mesh.extents[2])

        scale_transform = np.eye(4)
        scale_transform[:3, :3] *= scale
        translation = np.eye(4)
        translation[2, 3] = -min_z
        transform = translation @ scale_transform @ orientation
        report.transform = transform.tolist()
        report.prepared_mesh = mesh

        if is_normalized:
            result = CheckResult(
                code='P0', name='单位标准化', category='precheck',
                passed=True, score=100,
                detail=(
                    f'归一化 {source_up_axis.upper()}-up → 打印 Z-up，'
                    f'高度 {oriented_h:.2f} → {new_h:.1f}mm，'
                    f'比例 {scale:.1f}，已移至平台'
                ),
                metrics={'original_height': float(oriented_h),
                         'target_height': target_height,
                         'scale_factor': float(scale),
                         'final_height': float(new_h),
                         'source_up_axis': source_up_axis,
                         'print_up_axis': 'z',
                         'original_extents': np.asarray(bounds[1] - bounds[0]).tolist(),
                         'final_extents': mesh.extents.tolist(),
                         'input_units': mode},
            )
        else:
            result = CheckResult(
                code='P0', name='单位标准化', category='precheck',
                passed=True, score=100,
                detail=f'按毫米处理 (高度 {current_h:.1f}mm)，并移动至 Z=0 平台',
                metrics={'height_mm': float(current_h),
                         'scale_factor': 1.0, 'z_translation': -min_z,
                         'input_units': 'mm' if mode == 'auto' else mode},
            )
        report.results.append(result)
        return mesh

    @staticmethod
    def _up_axis_to_z_transform(source_up_axis: str) -> np.ndarray:
        """把源坐标的上方向旋转到打印坐标 Z 轴，不做镜像。"""
        axis = str(source_up_axis).strip().lower()
        transform = np.eye(4)
        if axis == 'z':
            return transform
        if axis == 'y':
            # 绕 X 轴 +90°：旧 Y → 新 Z，旧 Z → 新 -Y。
            transform[:3, :3] = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ])
            return transform
        if axis == 'x':
            # 绕 Y 轴 -90°：旧 X → 新 Z，旧 Z → 新 -X。
            transform[:3, :3] = np.array([
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ])
            return transform
        raise ValueError(f'不支持的源上方向: {source_up_axis!r}')

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

        # 只统计真正共面的底部三角面。最低点/最低边不具有可承载面积，
        # 不能把分散接触点的凸包当作实际接触面积。
        face_z = mesh.vertices[mesh.faces][:, :, 2]
        bottom_faces = np.all(np.abs(face_z - min_z) <= eps, axis=1)
        bottom_count = int(bottom_faces.sum())
        if bottom_count == 0:
            report.results.append(CheckResult(
                code='P1', name='底面平台', category='precheck',
                passed=False, score=0,
                detail='模型仅以点或边接触平台，没有可承载的平面区域',
                metrics={**metrics, 'bottom_face_count': 0,
                         'contact_area': 0.0},
            ))
            return

        contact_area = float(
            np.sum(mesh.area_faces[bottom_faces]
                   * np.abs(mesh.face_normals[bottom_faces, 2]))
        )
        metrics['contact_area'] = contact_area
        metrics['bottom_face_count'] = bottom_count

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

    # ── P3: 静态稳定性 ────────────────────────────────

    def _check_stability(self, mesh: trimesh.Trimesh,
                         report: DFMReport):
        """检查重心 XY 投影是否位于真实底面支撑凸包的安全裕量内。"""
        min_z = float(mesh.vertices[:, 2].min())
        eps = max(float(self.rules.get('z_tolerance', 1e-5)), 1e-5)
        required_margin = float(self.rules.get('stability_margin_mm', 1.0))
        face_z = mesh.vertices[mesh.faces][:, :, 2]
        bottom_faces = np.all(np.abs(face_z - min_z) <= eps, axis=1)
        support_points = np.unique(
            mesh.vertices[mesh.faces[bottom_faces]].reshape(-1, 3)[:, :2], axis=0,
        ) if np.any(bottom_faces) else np.empty((0, 2))

        if len(support_points) < 3:
            self._append_na(
                report, 'P3', '静态稳定性', 'precheck',
                '没有可形成支撑多边形的真实底面；由 P1 处理点/边接触问题',
                metrics={'support_point_count': int(len(support_points))},
            )
            return

        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(support_points)
            equations = np.asarray(hull.equations, dtype=float)
            support_area = float(hull.volume)  # 二维 ConvexHull.volume 即面积
        except Exception as exc:
            self._append_unknown(
                report, 'P3', '静态稳定性', 'precheck',
                f'支撑多边形计算未完成: {exc}', blocking=True,
            )
            return

        center_source = 'volume_center_mass'
        try:
            if mesh.is_volume:
                center = np.asarray(mesh.center_mass, dtype=float)
            else:
                center = np.asarray(mesh.centroid, dtype=float)
                center_source = 'surface_centroid'
        except Exception:
            center = np.asarray(mesh.centroid, dtype=float)
            center_source = 'surface_centroid'
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            center = np.asarray(mesh.centroid, dtype=float)
            center_source = 'surface_centroid'
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            self._append_unknown(
                report, 'P3', '静态稳定性', 'precheck',
                '模型重心无法可靠计算', blocking=True,
            )
            return

        normals = equations[:, :2]
        offsets = equations[:, 2]
        normal_lengths = np.linalg.norm(normals, axis=1)
        signed_distances = -(
            normals @ center[:2] + offsets
        ) / np.maximum(normal_lengths, 1e-12)
        stability_margin = float(signed_distances.min())
        metrics = {
            'center_x_mm': float(center[0]),
            'center_y_mm': float(center[1]),
            'center_source': center_source,
            'support_area_mm2': support_area,
            'stability_margin_mm': stability_margin,
            'required_margin_mm': required_margin,
        }

        if stability_margin < required_margin:
            report.results.append(CheckResult(
                code='P3', name='静态稳定性', category='precheck',
                passed=False,
                score=max(0.0, min(100.0, stability_margin / required_margin * 100.0)),
                detail=(
                    f'重心投影安全裕量 {stability_margin:.2f}mm '
                    f'< {required_margin:.2f}mm，存在倾倒风险'
                ),
                metrics=metrics,
            ))
        else:
            report.results.append(CheckResult(
                code='P3', name='静态稳定性', category='precheck',
                passed=True, score=100,
                detail=f'重心投影位于支撑区内，安全裕量 {stability_margin:.2f}mm',
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
        bed_tol = max(float(self.rules.get('z_tolerance', 1e-5)), 1e-5)
        touching_platform = [
            float(component.bounds[0, 2]) <= bed_tol
            for component in components
        ]
        platform_indices = [
            index for index, touching in enumerate(touching_platform) if touching
        ]
        main_idx = (
            max(platform_indices, key=lambda index: areas[index])
            if platform_indices else int(np.argmax(areas))
        )
        main_area = float(areas[main_idx])
        fragment_ratio = float(self.rules.get('fragment_max_ratio', 0.01))
        fragment_area = float(self.rules.get('fragment_max_area_mm2', 10.0))
        fragment_extent = float(self.rules.get('fragment_max_extent_mm', 2.0))
        small = [
            (i, a) for i, (a, component) in enumerate(zip(areas, components))
            if (
                a < main_area * fragment_ratio
                and a < fragment_area
                and float(np.max(component.extents)) < fragment_extent
            )
        ]
        frag_count = len(small)
        floating = [
            index for index, touching in enumerate(touching_platform)
            if not touching
        ]

        metrics = {
            'component_count': len(components),
            'main_component_index': int(main_idx),
            'total_area': float(total_area),
            'fragment_count': frag_count,
            'fragment_candidate_indices': [index for index, _ in small],
            'fragment_max_ratio': fragment_ratio,
            'fragment_max_area_mm2': fragment_area,
            'fragment_max_extent_mm': fragment_extent,
            'floating_component_count': len(floating),
            'floating_component_indices': floating[:20],
        }

        if floating:
            report.results.append(CheckResult(
                code='G6', name='孤立碎面', category='topology',
                passed=False, score=0,
                detail=f'发现 {len(floating)} 个未接触打印平台的悬空连通分量',
                metrics=metrics,
            ))
        elif frag_count > 0:
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
        """镂空点阵壁厚尚未实现，明确返回 UNKNOWN。"""
        report.results.append(CheckResult(
            code='W4', name='镂空点阵壁厚', category='wall',
            status=CheckStatus.UNKNOWN, score=None,
            detail='镂空壁厚精检尚未实现，当前不参与通过判定和评分',
            blocking=False,
        ))

    # ── C1: 封闭空腔 ──────────────────────────────────

    def _check_cavity(self, mesh: trimesh.Trimesh,
                      report: DFMReport):
        """封闭空腔空域洪泛尚未实现，明确返回 UNKNOWN。"""
        self._append_unknown(
            report, 'C1', '封闭空腔', 'cavity',
            '封闭空腔需对“空域”体素做边界洪泛；原算法检测的是实体内部，已停用',
            blocking=False,
        )
        return

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
        """层间桥接检测尚未实现，明确返回 UNKNOWN。"""
        self._append_unknown(
            report, 'S2', '桥接检测', 'overhang',
            '桥接跨度需比较相邻层支撑区域；整层截面宽度不能代表桥接，原算法已停用',
            blocking=False,
        )
        return

   # ── S3: 悬空孤岛 ──────────────────────────────────

    def _check_floating_islands(self, mesh: trimesh.Trimesh,
                                report: DFMReport):
        """层间悬空孤岛检测尚未实现，明确返回 UNKNOWN。"""
        self._append_unknown(
            report, 'S3', '悬空孤岛', 'overhang',
            '需使用层面积与下层膨胀支撑区域的差集；当前轮廓算法会漏掉断层后的孤岛，已停用',
            blocking=False,
        )
        return

   # ── S4: 最小支撑接触面积 ──────────────────────────

    def _check_overhang_contact_area(self, mesh: trimesh.Trimesh,
                                     report: DFMReport):
        """真实支撑接触面积需要支撑网格，当前返回 UNKNOWN。"""
        self._append_unknown(
            report, 'S4', '支撑接触面积', 'overhang',
            '必须在生成支撑后测量真实接触斑块；悬垂投影总面积不能替代接触面积',
            blocking=False,
        )
        return

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
        """任意方向内孔识别尚未实现，明确返回 UNKNOWN。"""
        self._append_unknown(
            report, 'D1', '内孔孔径', 'dimension',
            '任意方向内孔需要轴向识别；仅做水平切片会漏检或误测，当前不参与判定',
            blocking=False,
        )
        return

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

        for r in results:
            if r.status not in (CheckStatus.PASS, CheckStatus.FAIL) or r.score is None:
                continue
            if r.code.startswith('G'):
                bucket = 'watertight_topology'
            elif r.code in {'W1', 'W4'}:
                bucket = 'wall_thickness'
            elif r.code in {'W2', 'W3'}:
                bucket = 'feature_structure'
            elif r.code.startswith('S') or r.code in {'P1', 'P3'}:
                bucket = 'overhang_support'
            elif r.code.startswith('D') or r.code == 'P2':
                bucket = 'dimension_compliance'
            elif r.code.startswith('C'):
                bucket = 'cavity'
            else:
                # P0 是单位/输入准备步骤，不代表制造质量，不计入加权分。
                continue
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
