"""DFM 分级修复与复检闭环。

修复策略：
  轻量 — 合并重叠顶点、删除重复/零面积面、统一法向、清理孤立顶点 → pymeshlab
  中度 — 填补破洞、修复非流形边/顶点 → pymeshlab / MeshFix CLI
  重度 — 重网格化、简化（预留接口）
  兜底 — 无法自动修复时建议换 seed 重新生成

用法::

    from dfm import DFMChecker, DFMRepairer, DFMRules

    rules = DFMRules()
    checker = DFMChecker(rules)
    repairer = DFMRepairer()

    result, final_report = repairer.repair_and_recheck(
        mesh, checker, input_units='normalized'
    )
    if result.success:
        result.final_mesh.export('repaired.stl')
    else:
        print(f'修复失败: {result.summary}')
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import trimesh

from .checker import CheckResult, CheckStatus, DFMChecker, DFMReport
from .fragment_cleaner import FragmentCleaner
from .pedestal import PedestalGenerator
from .repair_policy import evaluate_repair_candidate
from .rules import DFMRules


# ── 数据结构 ─────────────────────────────────────────────

class RepairLevel(str, Enum):
    """修复级别。"""
    LIGHT = "light"        # 无损修复：合并顶点、去重、统一法向
    MEDIUM = "medium"      # 拓扑修复：补洞、修非流形、MeshFix
    HEAVY = "heavy"        # 重网格化（预留）
    FALLBACK = "fallback"  # 无法修复


@dataclass
class RepairAction:
    """单次修复操作的记录。"""

    level: RepairLevel
    description: str       # 人类可读的操作描述
    success: bool          # 操作是否成功执行
    detail: str = ""       # 补充信息
    vertices_before: int = 0
    faces_before: int = 0
    vertices_after: int = 0
    faces_after: int = 0
    stage: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            'level': self.level.value,
            'description': self.description,
            'success': self.success,
            'detail': self.detail,
            'vertices_before': self.vertices_before,
            'faces_before': self.faces_before,
            'vertices_after': self.vertices_after,
            'faces_after': self.faces_after,
            'stage': self.stage,
            'duration_seconds': round(float(self.duration_seconds), 3),
            'timed_out': self.timed_out,
        }


@dataclass
class RepairResult:
    """一次修复流程的完整记录。"""

    actions: list[RepairAction] = field(default_factory=list)
    final_mesh: Optional[trimesh.Trimesh] = None
    success: bool = False
    summary: str = ""
    verification_scope: str = "none"
    requires_detailed_check: bool = True
    timed_out: bool = False
    timeout_stage: str = ""
    resume_from_stage: str = ""

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'summary': self.summary,
            'verification_scope': self.verification_scope,
            'requires_detailed_check': self.requires_detailed_check,
            'timed_out': self.timed_out,
            'timeout_stage': self.timeout_stage,
            'resume_from_stage': self.resume_from_stage,
            'action_count': len(self.actions),
            'actions': [a.to_dict() for a in self.actions],
            'final_vertices': (
                len(self.final_mesh.vertices) if self.final_mesh is not None else 0
            ),
            'final_faces': (
                len(self.final_mesh.faces) if self.final_mesh is not None else 0
            ),
        }


@dataclass
class _StageExecution:
    """一次隔离修复阶段的内部执行结果。"""

    mesh: trimesh.Trimesh
    actions: list[RepairAction] = field(default_factory=list)
    success: bool = False
    changed: bool = False
    timed_out: bool = False
    duration_seconds: float = 0.0
    detail: str = ""


# ── 修复器 ───────────────────────────────────────────────

class DFMRepairer:
    """DFM 分级修复器。

    Parameters
    ----------
    rules : DFMRules, optional
        阈值规则对象，用于获取合并顶点容差等参数。
    meshfix_path : str or Path, optional
        MeshFix CLI 可执行文件路径；不提供则自动在 ``tools/meshfix/`` 下查找。
    """

    _LIGHT_CODES = frozenset({'G4', 'G5'})
    _MEDIUM_CODES = frozenset({'G1', 'G2', 'G3'})
    _REPAIRABLE_CODES = _LIGHT_CODES | _MEDIUM_CODES
    _PEDESTAL_CODES = frozenset({'P1', 'P3'})
    _FRAGMENT_CODES = frozenset({'G6'})

    def __init__(
        self,
        rules: Optional[DFMRules] = None,
        meshfix_path: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
        isolate_stages: bool = True,
    ):
        self.rules = rules or DFMRules()
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError('progress_callback 必须可调用或为 None')
        if not isinstance(isolate_stages, bool):
            raise TypeError('isolate_stages 必须是布尔值')
        self._progress_callback = progress_callback
        self._isolate_stages = isolate_stages
        self._meshfix_path: Optional[str] = None
        self._pymeshlab_available: bool = False

        # 定位 MeshFix
        if meshfix_path is not None:
            candidate = Path(meshfix_path)
            if candidate.exists():
                self._meshfix_path = str(candidate)
        else:
            self._meshfix_path = self._find_meshfix()

        # 检测 pymeshlab
        try:
            import pymeshlab  # noqa: F401
            self._pymeshlab_available = True
        except ImportError:
            self._pymeshlab_available = False

    def _emit_progress(
        self,
        stage: str,
        status: str,
        duration_seconds: float = 0.0,
        detail: str = '',
    ) -> None:
        """发出结构化阶段事件；回调异常不得破坏修复流程。"""
        event = {
            'stage': stage,
            'status': status,
            'duration_seconds': round(float(duration_seconds), 3),
            'detail': detail,
        }
        logging.getLogger(__name__).info(
            'repair_stage stage=%s status=%s duration=%.3fs detail=%s',
            stage, status, float(duration_seconds), detail,
        )
        if self._progress_callback is not None:
            try:
                self._progress_callback(event)
            except Exception:
                logging.getLogger(__name__).exception('修复进度回调异常，已忽略')

    @staticmethod
    def _stage_level(stage: str) -> RepairLevel:
        return RepairLevel.MEDIUM if stage in {'medium', 'meshfix'} else RepairLevel.LIGHT

    @staticmethod
    def _stage_display_name(stage: str) -> str:
        return {
            'pedestal': '底座生成',
            'fragment': '碎面清理',
            'light': '轻量修复',
            'medium': '中度修复',
            'meshfix': 'MeshFix 修复',
        }.get(stage, stage)

    def _execute_stage_direct(
        self,
        stage: str,
        mesh: trimesh.Trimesh,
        repair_func: Optional[Callable] = None,
    ) -> _StageExecution:
        """在当前进程执行一个阶段；仅供子进程或显式测试注入使用。"""
        before = _counts(mesh)
        if stage == 'pedestal':
            pedestal = PedestalGenerator(self.rules).generate(mesh)
            ok = pedestal.success and pedestal.mesh is not None
            output = pedestal.mesh if ok else mesh.copy()
            action = RepairAction(
                level=RepairLevel.LIGHT,
                description=(
                    f'底座生成: {pedestal.detail}' if ok else '底座生成'
                ),
                success=ok,
                detail=pedestal.detail,
                vertices_before=before[0],
                faces_before=before[1],
                vertices_after=len(output.vertices),
                faces_after=len(output.faces),
                stage=stage,
            )
            return _StageExecution(
                mesh=output, actions=[action], success=ok, changed=ok,
                detail=pedestal.detail,
            )

        if stage == 'fragment':
            cleaned = FragmentCleaner(self.rules).clean(mesh)
            changed = cleaned.cleaned and cleaned.mesh is not None
            output = cleaned.mesh if changed else mesh.copy()
            action = RepairAction(
                level=RepairLevel.LIGHT,
                description=(
                    f'碎面清理: {cleaned.detail}' if changed else '碎面清理'
                ),
                success=changed,
                detail=cleaned.detail,
                vertices_before=before[0],
                faces_before=before[1],
                vertices_after=len(output.vertices),
                faces_after=len(output.faces),
                stage=stage,
            )
            return _StageExecution(
                mesh=output, actions=[action], success=changed,
                changed=changed, detail=cleaned.detail,
            )

        if repair_func is None:
            repair_func = {
                'light': self.repair_light,
                'medium': self.repair_medium,
                'meshfix': self.repair_meshfix,
            }.get(stage)
        if repair_func is None:
            raise ValueError(f'未知修复阶段: {stage!r}')
        output, actions = repair_func(mesh)
        ok = _mesh_ok(output) and bool(actions) and all(
            action.success for action in actions
        )
        for action in actions:
            if not action.stage:
                action.stage = stage
        return _StageExecution(
            mesh=output if _mesh_ok(output) else mesh.copy(),
            actions=actions,
            success=ok,
            changed=_mesh_ok(output),
        )

    def _execute_stage(
        self,
        stage: str,
        mesh: trimesh.Trimesh,
        repair_func: Optional[Callable] = None,
    ) -> _StageExecution:
        """在可终止子进程执行危险阶段，并保留最后有效网格。"""
        started = time.perf_counter()
        self._emit_progress(stage, 'started')

        # 动态替换/注入的方法不能可靠跨 Windows spawn 序列化，显式在
        # 当前进程执行；生产默认方法仍全部隔离。
        default_func = getattr(type(self), f'repair_{stage}', None)
        injected_func = (
            repair_func is not None
            and getattr(repair_func, '__func__', None) is not default_func
        )
        # MeshFix 本身已是带 subprocess.run(timeout=...) 的外部进程，
        # 直接执行可避免 worker 超时后留下孙进程。
        if not self._isolate_stages or injected_func or stage == 'meshfix':
            outcome = self._execute_stage_direct(stage, mesh, repair_func)
            outcome.duration_seconds = time.perf_counter() - started
            for action in outcome.actions:
                action.duration_seconds = outcome.duration_seconds
            self._emit_progress(
                stage, 'completed' if outcome.success else 'failed',
                outcome.duration_seconds, outcome.detail,
            )
            return outcome

        timeout = float(self.rules.get('repair_stage_timeout_seconds', 120))
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix=f'dfm_{stage}_') as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / 'input.ply'
            output_path = tmp_dir / 'output.ply'
            result_path = tmp_dir / 'result.json'
            rules_path = tmp_dir / 'rules.json'
            mesh.export(input_path)
            rules_path.write_text(
                json.dumps(self.rules.to_dict(), ensure_ascii=False),
                encoding='utf-8',
            )
            command = [
                sys.executable, '-m', 'dfm.repair_worker',
                '--stage', stage,
                '--input', str(input_path),
                '--output', str(output_path),
                '--result', str(result_path),
                '--rules', str(rules_path),
            ]
            if self._meshfix_path:
                command.extend(['--meshfix', self._meshfix_path])
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - started
                detail = (
                    f'{self._stage_display_name(stage)}超过 {timeout:g}s，'
                    '隔离子进程已终止；保留进入该阶段前的有效网格'
                )
                action = RepairAction(
                    level=self._stage_level(stage),
                    description=f'{self._stage_display_name(stage)}超时',
                    success=False,
                    detail=detail,
                    vertices_before=len(mesh.vertices),
                    faces_before=len(mesh.faces),
                    vertices_after=len(mesh.vertices),
                    faces_after=len(mesh.faces),
                    stage=stage,
                    duration_seconds=duration,
                    timed_out=True,
                )
                self._emit_progress(stage, 'timed_out', duration, detail)
                return _StageExecution(
                    mesh=mesh.copy(), actions=[action], success=False,
                    timed_out=True, duration_seconds=duration, detail=detail,
                )

            duration = time.perf_counter() - started
            if completed.returncode != 0 or not result_path.exists():
                stderr = (completed.stderr or '').strip()
                detail = (
                    f'隔离修复进程失败(returncode={completed.returncode})'
                    + (f': {stderr[-1000:]}' if stderr else '')
                )
                action = RepairAction(
                    level=self._stage_level(stage),
                    description=self._stage_display_name(stage),
                    success=False,
                    detail=detail,
                    vertices_before=len(mesh.vertices),
                    faces_before=len(mesh.faces),
                    vertices_after=len(mesh.vertices),
                    faces_after=len(mesh.faces),
                    stage=stage,
                    duration_seconds=duration,
                )
                self._emit_progress(stage, 'failed', duration, detail)
                return _StageExecution(
                    mesh=mesh.copy(), actions=[action], success=False,
                    duration_seconds=duration, detail=detail,
                )

            try:
                payload = json.loads(result_path.read_text(encoding='utf-8'))
                actions = [
                    RepairAction(
                        level=RepairLevel(item['level']),
                        description=item['description'],
                        success=bool(item['success']),
                        detail=item.get('detail', ''),
                        vertices_before=int(item.get('vertices_before', 0)),
                        faces_before=int(item.get('faces_before', 0)),
                        vertices_after=int(item.get('vertices_after', 0)),
                        faces_after=int(item.get('faces_after', 0)),
                        stage=item.get('stage', stage),
                        duration_seconds=duration,
                        timed_out=bool(item.get('timed_out', False)),
                    )
                    for item in payload.get('actions', [])
                ]
                changed = bool(payload.get('changed')) and output_path.exists()
                output_mesh = (
                    trimesh.load(output_path, force='mesh', process=False)
                    if changed else mesh.copy()
                )
                if not _mesh_ok(output_mesh):
                    raise ValueError('worker 返回的网格无效')
            except Exception as exc:
                detail = f'无法读取隔离修复结果，已保留原网格: {type(exc).__name__}: {exc}'
                action = RepairAction(
                    level=self._stage_level(stage),
                    description=self._stage_display_name(stage),
                    success=False,
                    detail=detail,
                    vertices_before=len(mesh.vertices),
                    faces_before=len(mesh.faces),
                    vertices_after=len(mesh.vertices),
                    faces_after=len(mesh.faces),
                    stage=stage,
                    duration_seconds=duration,
                )
                self._emit_progress(stage, 'failed', duration, detail)
                return _StageExecution(
                    mesh=mesh.copy(), actions=[action], success=False,
                    duration_seconds=duration, detail=detail,
                )
            outcome = _StageExecution(
                mesh=output_mesh,
                actions=actions,
                success=bool(payload.get('success')),
                changed=changed,
                duration_seconds=duration,
                detail=payload.get('detail', ''),
            )
            self._emit_progress(
                stage, 'completed' if outcome.success else 'failed',
                duration, outcome.detail,
            )
            return outcome

    def _timeout_result(
        self,
        mesh: trimesh.Trimesh,
        actions: list[RepairAction],
        stage: str,
    ) -> RepairResult:
        """构造可恢复的超时结果，禁止使用阶段内的部分产物。"""
        return RepairResult(
            actions=actions,
            final_mesh=mesh.copy() if _mesh_ok(mesh) else None,
            success=False,
            verification_scope='quick',
            requires_detailed_check=True,
            timed_out=True,
            timeout_stage=stage,
            resume_from_stage=stage,
            summary=(
                f'{self._stage_display_name(stage)}超时；已保留阶段前网格，'
                f'可从 {stage} 阶段续跑或转服务器处理'
            ),
        )

    def _transactional_recheck(
        self,
        *,
        stage: str,
        previous_mesh: trimesh.Trimesh,
        candidate_mesh: trimesh.Trimesh,
        previous_report: DFMReport,
        checker: DFMChecker,
        target_height: float,
        actions: list[RepairAction],
    ) -> tuple[DFMReport, trimesh.Trimesh, bool]:
        """复检候选并只提交不退化的网格，否则保留阶段前检查点。"""
        candidate_report = checker.check_quick(
            candidate_mesh, target_height, input_units='mm',
        )
        decision = evaluate_repair_candidate(previous_report, candidate_report)
        if decision.accepted:
            committed = (
                candidate_report.prepared_mesh
                if candidate_report.prepared_mesh is not None
                else candidate_mesh
            )
            return candidate_report, committed, True

        rollback = RepairAction(
            level=self._stage_level(stage),
            description=(
                f'{self._stage_display_name(stage)}事务性验收回滚'
            ),
            success=False,
            detail=decision.detail,
            vertices_before=len(previous_mesh.vertices),
            faces_before=len(previous_mesh.faces),
            vertices_after=len(previous_mesh.vertices),
            faces_after=len(previous_mesh.faces),
            stage=stage,
        )
        actions.append(rollback)
        self._emit_progress(stage, 'rolled_back', detail=decision.detail)
        return previous_report, previous_mesh, False

    # ── 公开入口 ──────────────────────────────────────

    def repair_light(
        self, mesh: trimesh.Trimesh
    ) -> tuple[trimesh.Trimesh, list[RepairAction]]:
        """执行轻量修复（无损）：合并重叠顶点、删除重复/零面积面、
        清理孤立顶点、统一法向。

        这些操作不改变模型的几何形状，适合作为任何修复流程的第一步。

        Returns
        -------
        (repaired_mesh, actions)
        """
        actions: list[RepairAction] = []
        if not _mesh_ok(mesh):
            actions.append(_action(
                RepairLevel.LIGHT, "输入网格校验", False,
                "网格为空或类型无效", _safe_counts(mesh), (0, 0),
            ))
            return _safe_mesh_copy(mesh), actions
        if _has_visual_payload(mesh):
            actions.append(_action(
                RepairLevel.LIGHT, "外观属性保护", False,
                "检测到纹理/顶点色/面颜色；当前几何修复会丢失外观属性，请在纹理生成前修复几何",
                _counts(mesh), _counts(mesh),
            ))
            return mesh.copy(), actions
        if not self._pymeshlab_available:
            actions.append(_action(
                RepairLevel.LIGHT, "pymeshlab 轻量修复", False,
                "pymeshlab 不可用，未执行任何修复", _counts(mesh), _counts(mesh),
            ))
            return mesh.copy(), actions

        count_before = _counts(mesh)

        try:
            ms = _to_pymeshlab(mesh)
        except Exception as exc:
            actions.append(_action(
                RepairLevel.LIGHT, "加载网格到 pymeshlab",
                False, str(exc), count_before, (0, 0),
            ))
            return _safe_mesh_copy(mesh), actions

        # 1. 合并重叠顶点
        v_before = ms.current_mesh().vertex_number()
        try:
            import pymeshlab
            tolerance = float(self.rules.get('merge_close_tolerance', 0.01))
            ms.meshing_merge_close_vertices(
                threshold=pymeshlab.PureValue(tolerance),
            )
            v_after = ms.current_mesh().vertex_number()
            actions.append(_action(
                RepairLevel.LIGHT, "合并重叠顶点",
                True, f'容差 {tolerance}mm，顶点 {v_before} → {v_after}',
                count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.LIGHT, "合并重叠顶点",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 2. 删除重复面
        f_before = ms.current_mesh().face_number()
        try:
            ms.meshing_remove_duplicate_faces()
            f_after = ms.current_mesh().face_number()
            actions.append(_action(
                RepairLevel.LIGHT, "删除重复三角面",
                True, f'面 {f_before} → {f_after}',
                count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.LIGHT, "删除重复三角面",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 3. 删除零面积面
        f_before = ms.current_mesh().face_number()
        try:
            ms.meshing_remove_null_faces()
            f_after = ms.current_mesh().face_number()
            actions.append(_action(
                RepairLevel.LIGHT, "删除零面积退化面",
                True, f'面 {f_before} → {f_after}',
                count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.LIGHT, "删除零面积退化面",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 4. 清理孤立顶点
        try:
            ms.meshing_remove_unreferenced_vertices()
            actions.append(_action(
                RepairLevel.LIGHT, "清理未引用顶点",
                True, "", count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.LIGHT, "清理未引用顶点",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 5. 统一法向朝向
        try:
            ms.meshing_re_orient_faces_coherently()
            oriented = _from_pymeshlab(ms)
            flipped = False
            if oriented.is_watertight and float(oriented.volume) < 0:
                ms.meshing_invert_face_orientation(forceflip=True)
                flipped = True
            actions.append(_action(
                RepairLevel.LIGHT, "统一面法向朝向",
                True, "已整体翻转朝外" if flipped else "相邻面绕序已统一",
                count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.LIGHT, "统一面法向朝向",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        repaired = _from_pymeshlab(ms)
        return repaired, actions

    def repair_medium(
        self, mesh: trimesh.Trimesh
    ) -> tuple[trimesh.Trimesh, list[RepairAction]]:
        """执行中度修复：填补破洞、修复非流形边/顶点，并尝试 MeshFix CLI。

        注意：中度修复可能改变模型拓扑（例如补洞会增加面片），
        修复后应重新运行 DFM 检查验证。

        Returns
        -------
        (repaired_mesh, actions)
        """
        actions: list[RepairAction] = []
        if not _mesh_ok(mesh):
            actions.append(_action(
                RepairLevel.MEDIUM, "输入网格校验", False,
                "网格为空或类型无效", _safe_counts(mesh), (0, 0),
            ))
            return _safe_mesh_copy(mesh), actions
        if _has_visual_payload(mesh):
            actions.append(_action(
                RepairLevel.MEDIUM, "外观属性保护", False,
                "带外观属性的网格禁止进入会丢失属性的拓扑修复",
                _counts(mesh), _counts(mesh),
            ))
            return mesh.copy(), actions
        if not self._pymeshlab_available:
            if self._meshfix_path is not None:
                fixed, action = self._run_meshfix(mesh)
                actions.append(action)
                return fixed if fixed is not None else mesh.copy(), actions
            actions.append(_action(
                RepairLevel.MEDIUM, "中度修复后端", False,
                "pymeshlab 与 MeshFix 均不可用", _counts(mesh), _counts(mesh),
            ))
            return mesh.copy(), actions

        count_before = _counts(mesh)

        try:
            ms = _to_pymeshlab(mesh)
        except Exception as exc:
            actions.append(_action(
                RepairLevel.MEDIUM, "加载网格到 pymeshlab",
                False, str(exc), count_before, (0, 0),
            ))
            return _safe_mesh_copy(mesh), actions

        # 1. 按边界环边数填补小破洞
        f_before = ms.current_mesh().face_number()
        try:
            max_hole_edges = int(self.rules.get('max_hole_edges', 30))
            ms.meshing_close_holes(maxholesize=max_hole_edges)
            f_after = ms.current_mesh().face_number()
            closed = f_after - f_before
            actions.append(_action(
                RepairLevel.MEDIUM, "填补破洞",
                True,
                f'新增 {closed} 面（边界环最多 {max_hole_edges} 条边）',
                count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.MEDIUM, "填补破洞",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 2. 修复非流形边
        try:
            ms.meshing_repair_non_manifold_edges()
            actions.append(_action(
                RepairLevel.MEDIUM, "修复非流形边",
                True, "", count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.MEDIUM, "修复非流形边",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 3. 修复非流形顶点
        try:
            ms.meshing_repair_non_manifold_vertices()
            actions.append(_action(
                RepairLevel.MEDIUM, "修复非流形顶点",
                True, "", count_before, _counts_from_ms(ms),
            ))
        except Exception as exc:
            actions.append(_action(
                RepairLevel.MEDIUM, "修复非流形顶点",
                False, str(exc), count_before, _counts_from_ms(ms),
            ))

        # 4. 清理
        try:
            ms.meshing_remove_unreferenced_vertices()
        except Exception:
            pass

        pymeshlab_repaired = _from_pymeshlab(ms)

        return pymeshlab_repaired, actions

    def repair_meshfix(
        self, mesh: trimesh.Trimesh
    ) -> tuple[trimesh.Trimesh, list[RepairAction]]:
        """单独调用 MeshFix；应只在 pymeshlab 修复复检仍失败后执行。"""
        if not _mesh_ok(mesh):
            return _safe_mesh_copy(mesh), [_action(
                RepairLevel.MEDIUM, "MeshFix CLI 修复", False,
                "网格为空或类型无效", _safe_counts(mesh), (0, 0),
            )]
        if _has_visual_payload(mesh):
            return mesh.copy(), [_action(
                RepairLevel.MEDIUM, "外观属性保护", False,
                "带外观属性的网格禁止进入 STL 中转的 MeshFix 修复",
                _counts(mesh), _counts(mesh),
            )]
        if self._meshfix_path is None:
            return mesh.copy(), [_action(
                RepairLevel.MEDIUM, "MeshFix CLI 修复", False,
                "MeshFix 不可用", _counts(mesh), _counts(mesh),
            )]
        fixed, action = self._run_meshfix(mesh)
        return fixed if fixed is not None else mesh.copy(), [action]

    def repair(
        self,
        mesh: trimesh.Trimesh,
        issues: Optional[list[CheckResult]] = None,
    ) -> RepairResult:
        """执行分级修复流程：轻量 → 中度。

        根据 DFM 检查结果中失败的项自动选择修复策略。
        如果不提供 issues，默认执行轻量修复。

        Parameters
        ----------
        mesh : trimesh.Trimesh
            待修复网格。
        issues : list[CheckResult], optional
            DFM 检查中失败的项，用于判断需要哪些修复。

        Returns
        -------
        RepairResult
        """
        result = RepairResult()
        if not _mesh_ok(mesh):
            result.summary = '输入网格为空或类型无效，未执行修复'
            return result
        current = mesh.copy()

        failed_codes = self._failed_codes(issues or [])
        if issues is not None and not (failed_codes & self._REPAIRABLE_CODES):
            result.final_mesh = current
            result.summary = '失败项不属于当前自动拓扑修复范围，未修改网格'
            return result

        # 未提供 issues 时保留“仅执行轻量清理”的兼容行为。
        planned_codes = failed_codes if issues is not None else set(self._LIGHT_CODES)
        attempted: set[str] = set()
        stage = self._next_repair_stage(planned_codes, attempted)
        if stage is not None:
            stage_key, _, repair_func = stage
            stage_result = self._execute_stage(
                stage_key, current, repair_func=repair_func,
            )
            result.actions.extend(stage_result.actions)
            if stage_result.timed_out:
                return self._timeout_result(
                    current, result.actions, stage_key,
                )
            stage_actions = stage_result.actions
            current = stage_result.mesh
            attempted.add(stage_key)

            # 一次性入口保持原有行为：同时存在中度缺陷，或轻量修复执行失败时，
            # 再执行一个由共享策略选出的升级阶段；阶段之间不在这里复检。
            needs_followup = (
                stage_key == 'light'
                and (
                    bool(planned_codes & self._MEDIUM_CODES)
                    or not all(action.success for action in stage_actions)
                )
            )
            if needs_followup:
                followup = self._next_repair_stage(planned_codes, attempted)
                if followup is not None:
                    followup_key, _, followup_func = followup
                    followup_result = self._execute_stage(
                        followup_key, current, repair_func=followup_func,
                    )
                    result.actions.extend(followup_result.actions)
                    if followup_result.timed_out:
                        return self._timeout_result(
                            current, result.actions, followup_key,
                        )
                    current = followup_result.mesh
                    attempted.add(followup_key)

        # ── 判断整体成功 ──
        all_ok = bool(result.actions) and all(a.success for a in result.actions)
        # 至少有面片和顶点
        has_content = (
            len(current.vertices) >= 3
            and len(current.faces) >= 1
        )
        result.success = all_ok and has_content
        result.final_mesh = current if has_content else None

        if result.success:
            result.summary = (
                f'修复完成：{len(result.actions)} 步，'
                f'最终网格 {len(current.vertices)} 顶点 {len(current.faces)} 面'
            )
        else:
            failed = [a for a in result.actions if not a.success]
            reasons = '; '.join(
                f'{a.description}({a.detail})' for a in failed
            )
            if not reasons:
                reasons = '没有可用的修复后端或未执行任何修复操作'
            result.summary = f'修复未完全成功: {reasons}'

        return result

    def repair_and_recheck(
        self,
        mesh: trimesh.Trimesh,
        checker: DFMChecker,
        target_height: float = 100.0,
        max_rounds: int = 2,
        input_units: str = 'auto',
    ) -> tuple[RepairResult, DFMReport]:
        """修复 + 复检闭环：修复后自动运行快速粗筛验证。

        流程::

            初始检查 → [FAIL]
                → [P1/P3] → 底座生成与实体融合 → 复检
                → [G6] → 碎片清理 → 复检
                → 轻量修复 → 复检
                → [仍 FAIL] → 中度修复 → 复检
                → [仍 FAIL] → 标记 FALLBACK

        Parameters
        ----------
        mesh : trimesh.Trimesh
            待修复网格。
        checker : DFMChecker
            DFM 检查器实例，用于复检。
        target_height : float
            目标高度(mm)，传递给 check_quick。
        max_rounds : int
            最多修复轮数（轻量 + 中度各算一轮），默认 2。
        input_units : {'auto', 'mm', 'normalized'}
            原始输入单位模式；AI 生成网格应显式传 ``normalized``。

        Returns
        -------
        (RepairResult, DFMReport)
            修复结果和最后一次 DFM 检查报告。
        """
        # 初始检查。UNKNOWN 表示检查未完成，不是可送入 MeshFix 的几何缺陷。
        initial_report = checker.check_quick(
            mesh, target_height, input_units=input_units,
        )
        if initial_report.prepared_mesh is None:
            result = RepairResult(
                final_mesh=None,
                success=False,
                summary=f'输入未进入修复：{initial_report.summary}',
            )
            return result, initial_report

        if initial_report.status == 'INCOMPLETE':
            result = RepairResult(
                final_mesh=initial_report.prepared_mesh,
                success=False,
                summary='检查未完成（UNKNOWN），应转异步检查或人工复核，未执行几何修复',
            )
            return result, initial_report

        if initial_report.status == 'PASS':
            result = RepairResult()
            result.success = True
            result.final_mesh = initial_report.prepared_mesh
            result.verification_scope = 'quick'
            result.summary = '无需修复：初始快速层已通过；仍需精检与切片验证'
            return result, initial_report

        current = initial_report.prepared_mesh
        last_report = initial_report
        all_actions: list[RepairAction] = []
        pedestal_attempts = 0
        fragment_attempts = 0

        if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 1:
            result = RepairResult(
                final_mesh=current,
                success=False,
                summary='max_rounds 必须是大于等于 1 的整数，未执行修复',
            )
            return result, last_report

        # ── 底座：P1 失败时先加底座再继续 ──
        initial_codes = self._failed_codes(initial_report.results)
        if initial_codes & self._PEDESTAL_CODES:
            pedestal_attempts += 1
            pedestal_stage = self._execute_stage('pedestal', current)
            all_actions.extend(pedestal_stage.actions)
            if pedestal_stage.timed_out:
                return self._timeout_result(
                    current, all_actions, 'pedestal',
                ), last_report
            if pedestal_stage.success and pedestal_stage.changed:
                last_report, current, _ = self._transactional_recheck(
                    stage='pedestal',
                    previous_mesh=current,
                    candidate_mesh=pedestal_stage.mesh,
                    previous_report=last_report,
                    checker=checker,
                    target_height=target_height,
                    actions=all_actions,
                )

        if last_report.status == 'PASS':
            result = RepairResult(
                actions=all_actions,
                final_mesh=last_report.prepared_mesh,
                success=True,
                verification_scope='quick',
                requires_detailed_check=True,
                summary='自动底座融合成功，快速层通过；仍需精检与切片验证',
            )
            return result, last_report

        if last_report.status == 'INCOMPLETE':
            result = RepairResult(
                actions=all_actions,
                final_mesh=current,
                success=False,
                summary='底座处理后检查返回 UNKNOWN，停止继续修改并转异步/人工复核',
            )
            return result, last_report

        # ── 碎面清理：G6 失败时删除孤立/悬浮碎片 ──
        failed_after_pedestal = self._failed_codes(last_report.results)
        fragment_changed = False
        if failed_after_pedestal & self._FRAGMENT_CODES:
            fragment_attempts += 1
            fragment_stage = self._execute_stage('fragment', current)
            all_actions.extend(fragment_stage.actions)
            if fragment_stage.timed_out:
                return self._timeout_result(
                    current, all_actions, 'fragment',
                ), last_report
            if fragment_stage.success and fragment_stage.changed:
                last_report, current, fragment_changed = (
                    self._transactional_recheck(
                        stage='fragment',
                        previous_mesh=current,
                        candidate_mesh=fragment_stage.mesh,
                        previous_report=last_report,
                        checker=checker,
                        target_height=target_height,
                        actions=all_actions,
                    )
                )

        # P1/P3 与 G6 同时出现时，首次底座可能因悬浮碎片而融合失败。
        # 删除明确碎屑并复检后，必须根据最新状态立即重试一次底座。
        post_fragment_codes = self._failed_codes(last_report.results)
        if (
            fragment_changed
            and post_fragment_codes & self._PEDESTAL_CODES
            and pedestal_attempts < 2
        ):
            pedestal_attempts += 1
            pedestal_stage = self._execute_stage('pedestal', current)
            for action in pedestal_stage.actions:
                action.description = f'碎片清理后重试{action.description}'
            all_actions.extend(pedestal_stage.actions)
            if pedestal_stage.timed_out:
                return self._timeout_result(
                    current, all_actions, 'pedestal',
                ), last_report
            if pedestal_stage.success and pedestal_stage.changed:
                last_report, current, _ = self._transactional_recheck(
                    stage='pedestal',
                    previous_mesh=current,
                    candidate_mesh=pedestal_stage.mesh,
                    previous_report=last_report,
                    checker=checker,
                    target_height=target_height,
                    actions=all_actions,
                )

        if last_report.status == 'PASS':
            result = RepairResult(
                actions=all_actions,
                final_mesh=last_report.prepared_mesh,
                success=True,
                verification_scope='quick',
                requires_detailed_check=True,
                summary='碎片清理成功，快速层通过；仍需精检与切片验证',
            )
            return result, last_report

        if last_report.status == 'INCOMPLETE':
            result = RepairResult(
                actions=all_actions,
                final_mesh=current,
                success=False,
                summary='碎片清理后检查返回 UNKNOWN，停止继续修改并转异步/人工复核',
            )
            return result, last_report

        initial_codes = self._failed_codes(last_report.results)
        if not (initial_codes & self._REPAIRABLE_CODES):
            return self._fallback_result(
                current, all_actions, initial_codes,
                '当前失败项需要摆放、底座加固、尺寸或支撑策略，不能用拓扑修复解决',
            ), last_report

        attempted: set[str] = set()
        for round_idx in range(max_rounds):
            remaining = self._failed_codes(last_report.results)
            if not (remaining & self._REPAIRABLE_CODES):
                return self._fallback_result(
                    current, all_actions, remaining,
                    '拓扑缺陷已处理，但仍存在不可由网格修复解决的问题',
                ), last_report

            stage = self._next_repair_stage(remaining, attempted)
            if stage is None:
                return self._fallback_result(
                    current, all_actions, remaining,
                    '没有适用于剩余缺陷的未执行修复阶段',
                ), last_report

            stage_key, stage_name, repair_func = stage
            attempted.add(stage_key)
            repair_stage = self._execute_stage(
                stage_key, current, repair_func=repair_func,
            )
            actions = repair_stage.actions
            all_actions.extend(actions)
            if repair_stage.timed_out:
                return self._timeout_result(
                    current, all_actions, stage_key,
                ), last_report
            repaired = repair_stage.mesh

            if not _mesh_ok(repaired):
                result = RepairResult(
                    actions=all_actions,
                    final_mesh=None,
                    success=False,
                    summary='修复导致网格损坏（顶点或面数不足）',
                )
                return result, last_report

            # 复检后只有不增加缺陷、不降分且不新增 UNKNOWN 的候选才提交。
            last_report, current, repair_committed = self._transactional_recheck(
                stage=stage_key,
                previous_mesh=current,
                candidate_mesh=repaired,
                previous_report=last_report,
                checker=checker,
                target_height=target_height,
                actions=all_actions,
            )
            if not repair_committed:
                remaining_after_rollback = self._failed_codes(
                    last_report.results
                )
                return self._fallback_result(
                    current,
                    all_actions,
                    remaining_after_rollback,
                    f'{stage_name}修复候选未通过事务性验收，已回滚并停止派生修复',
                ), last_report

            # 拓扑修复也可能暴露或产生新的孤立小分量；复检后按最新
            # G6 状态再执行一次保守清理，而不是沿用初始计划。
            remaining_after_repair = self._failed_codes(last_report.results)
            if (
                remaining_after_repair & self._FRAGMENT_CODES
                and fragment_attempts < 2
            ):
                fragment_attempts += 1
                fragment_stage = self._execute_stage('fragment', current)
                for action in fragment_stage.actions:
                    action.description = f'拓扑修复后{action.description}'
                all_actions.extend(fragment_stage.actions)
                if fragment_stage.timed_out:
                    return self._timeout_result(
                        current, all_actions, 'fragment',
                    ), last_report
                if fragment_stage.success and fragment_stage.changed:
                    last_report, current, _ = self._transactional_recheck(
                        stage='fragment',
                        previous_mesh=current,
                        candidate_mesh=fragment_stage.mesh,
                        previous_report=last_report,
                        checker=checker,
                        target_height=target_height,
                        actions=all_actions,
                    )

            # 如果初次底座融合受原模型拓扑缺陷阻碍，在拓扑修复后允许重试一次。
            # 这样 P1/P3 与 G 类问题同时出现时不会在 G 类修好后直接 FALLBACK。
            remaining_after_repair = self._failed_codes(last_report.results)
            if (
                remaining_after_repair & self._PEDESTAL_CODES
                and pedestal_attempts < 2
            ):
                pedestal_attempts += 1
                pedestal_stage = self._execute_stage('pedestal', current)
                for action in pedestal_stage.actions:
                    action.description = f'拓扑修复后重试{action.description}'
                all_actions.extend(pedestal_stage.actions)
                if pedestal_stage.timed_out:
                    return self._timeout_result(
                        current, all_actions, 'pedestal',
                    ), last_report
                if pedestal_stage.success and pedestal_stage.changed:
                    last_report, current, _ = self._transactional_recheck(
                        stage='pedestal',
                        previous_mesh=current,
                        candidate_mesh=pedestal_stage.mesh,
                        previous_report=last_report,
                        checker=checker,
                        target_height=target_height,
                        actions=all_actions,
                    )

            if last_report.status == 'PASS':
                result = RepairResult(
                    actions=all_actions,
                    final_mesh=last_report.prepared_mesh,
                    success=True,
                    verification_scope='quick',
                    requires_detailed_check=True,
                    summary=(
                        f'{stage_name}修复成功（第 {round_idx + 1} 轮，'
                        f'{len(actions)} 步），快速层通过；仍需精检与切片验证'
                    ),
                )
                return result, last_report

            if last_report.status == 'INCOMPLETE':
                result = RepairResult(
                    actions=all_actions,
                    final_mesh=current,
                    success=False,
                    summary='修复后检查返回 UNKNOWN，停止继续修改并转异步/人工复核',
                )
                return result, last_report

        remaining = self._failed_codes(last_report.results)
        return self._fallback_result(
            current, all_actions, remaining,
            f'达到修复轮数上限 {max_rounds}',
        ), last_report

    @staticmethod
    def _failed_codes(results: list[CheckResult]) -> set[str]:
        """提取明确失败的检查编码，供所有修复入口共享。"""
        return {
            item.code for item in results
            if item.status == CheckStatus.FAIL
        }

    def _next_repair_stage(
        self,
        failed_codes: set[str],
        attempted: set[str],
    ) -> Optional[tuple[
        str,
        str,
        Callable[[trimesh.Trimesh], tuple[trimesh.Trimesh, list[RepairAction]]],
    ]]:
        """根据最新失败项选择下一个尚未执行的修复阶段。

        轻量缺陷首次选择轻量修复；若复检后仍有轻量缺陷，则升级到
        中度修复。中度修复后仍失败且 MeshFix 可用时再使用 MeshFix。
        """
        if failed_codes & self._LIGHT_CODES and 'light' not in attempted:
            return 'light', '轻量', self.repair_light

        needs_medium = bool(failed_codes & self._MEDIUM_CODES) or (
            bool(failed_codes & self._LIGHT_CODES) and 'light' in attempted
        )
        if not needs_medium:
            return None

        if self.pymeshlab_available and 'medium' not in attempted:
            return 'medium', '中度', self.repair_medium
        if self.meshfix_available and 'meshfix' not in attempted:
            return 'meshfix', 'MeshFix', self.repair_meshfix
        if (
            not self.pymeshlab_available
            and not self.meshfix_available
            and 'medium' not in attempted
        ):
            # 执行一次以留下“后端不可用”的明确 RepairAction。
            return 'medium', '中度', self.repair_medium
        return None

    def _fallback_result(
        self,
        mesh: trimesh.Trimesh,
        actions: list[RepairAction],
        codes: set[str],
        reason: str,
    ) -> RepairResult:
        """构造不再自动修改的安全兜底结果。"""
        code_text = ', '.join(sorted(codes)) if codes else '无'
        actions.append(RepairAction(
            level=RepairLevel.FALLBACK,
            description='兜底：停止自动修复',
            success=False,
            detail=f'{reason}；剩余失败项: {code_text}',
            vertices_before=len(mesh.vertices),
            faces_before=len(mesh.faces),
            vertices_after=len(mesh.vertices),
            faces_after=len(mesh.faces),
        ))
        return RepairResult(
            actions=actions,
            final_mesh=mesh if _mesh_ok(mesh) else None,
            success=False,
            verification_scope='quick',
            requires_detailed_check=True,
            summary=f'{reason}，剩余失败项 [{code_text}]',
        )

    # ── 内部方法 ──────────────────────────────────────

    def _run_meshfix(
        self, mesh: trimesh.Trimesh
    ) -> tuple[Optional[trimesh.Trimesh], RepairAction]:
        """调用 MeshFix CLI 修复网格。"""
        count_before = _counts(mesh)
        tmp_dir = tempfile.mkdtemp(prefix='dfm_meshfix_')
        input_path = os.path.join(tmp_dir, 'input.stl')
        output_path = os.path.join(tmp_dir, 'output.stl')

        try:
            mesh.export(input_path)
            cmd = [self._meshfix_path, input_path, output_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(self.rules.get('meshfix_timeout_seconds', 120)),
            )
            if proc.returncode != 0:
                stderr = proc.stderr.strip() or '未知错误'
                return None, _action(
                    RepairLevel.MEDIUM, "MeshFix CLI 修复",
                    False, f'MeshFix 返回码 {proc.returncode}: {stderr[:120]}',
                    count_before, (0, 0),
                )

            if not os.path.exists(output_path):
                return None, _action(
                    RepairLevel.MEDIUM, "MeshFix CLI 修复",
                    False, 'MeshFix 未生成输出文件',
                    count_before, (0, 0),
                )

            repaired = trimesh.load(output_path, force='mesh')
            if not isinstance(repaired, trimesh.Trimesh):
                return None, _action(
                    RepairLevel.MEDIUM, "MeshFix CLI 修复",
                    False, 'MeshFix 输出无法解析为 Trimesh',
                    count_before, (0, 0),
                )

            count_after = _counts(repaired)
            return repaired, _action(
                RepairLevel.MEDIUM, "MeshFix CLI 修复",
                True,
                f'MeshFix 完成，顶点 {count_before[0]}→{count_after[0]}，'
                f'面 {count_before[1]}→{count_after[1]}',
                count_before, count_after,
            )
        except subprocess.TimeoutExpired:
            timeout = int(self.rules.get('meshfix_timeout_seconds', 120))
            return None, _action(
                RepairLevel.MEDIUM, "MeshFix CLI 修复",
                False, f'MeshFix 执行超时 ({timeout}s)',
                count_before, (0, 0),
            )
        except Exception as exc:
            return None, _action(
                RepairLevel.MEDIUM, "MeshFix CLI 修复",
                False, str(exc),
                count_before, (0, 0),
            )
        finally:
            # 清理临时文件
            try:
                for f in [input_path, output_path]:
                    if os.path.exists(f):
                        os.unlink(f)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    @staticmethod
    def _find_meshfix() -> Optional[str]:
        """在项目 tools/ 目录下查找 MeshFix 可执行文件。"""
        candidates = [
            Path(__file__).resolve().parent.parent
            / "tools" / "meshfix" / "MeshFix-V2.1-master" / "bin64" / "MeshFix.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    @property
    def meshfix_available(self) -> bool:
        """MeshFix CLI 是否可用。"""
        return self._meshfix_path is not None

    @property
    def pymeshlab_available(self) -> bool:
        """pymeshlab 是否可用。"""
        return self._pymeshlab_available


# ── 模块级辅助函数 ───────────────────────────────────────

def _counts(mesh: trimesh.Trimesh) -> tuple[int, int]:
    return (len(mesh.vertices), len(mesh.faces))


def _safe_counts(mesh) -> tuple[int, int]:
    if not isinstance(mesh, trimesh.Trimesh):
        return (0, 0)
    return _counts(mesh)


def _safe_mesh_copy(mesh):
    """Trimesh 输入始终返回独立副本；非网格无可复制状态，原样返回。"""
    return mesh.copy() if isinstance(mesh, trimesh.Trimesh) else mesh


def _has_visual_payload(mesh: trimesh.Trimesh) -> bool:
    """拓扑修复目前不保留外观属性，发现后必须拒绝而不是静默丢失。"""
    return getattr(mesh.visual, 'kind', None) in {'vertex', 'face', 'texture'}


def _counts_from_ms(ms) -> tuple[int, int]:
    m = ms.current_mesh()
    return (m.vertex_number(), m.face_number())


def _action(
    level: RepairLevel,
    description: str,
    success: bool,
    detail: str,
    before: tuple[int, int],
    after: tuple[int, int],
) -> RepairAction:
    return RepairAction(
        level=level,
        description=description,
        success=success,
        detail=detail,
        vertices_before=before[0],
        faces_before=before[1],
        vertices_after=after[0],
        faces_after=after[1],
    )


def _to_pymeshlab(mesh: trimesh.Trimesh):
    """将 trimesh 网格转为 pymeshlab MeshSet。"""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=mesh.vertices.astype(np.float64),
        face_matrix=mesh.faces.astype(np.uint32),
    ))
    return ms


def _from_pymeshlab(ms) -> trimesh.Trimesh:
    """从 pymeshlab MeshSet 提取 trimesh 网格。"""
    m = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=m.vertex_matrix(),
        faces=m.face_matrix(),
        process=False,
    )


def _mesh_ok(mesh: trimesh.Trimesh) -> bool:
    """检查网格是否至少包含有效几何。"""
    return (
        isinstance(mesh, trimesh.Trimesh)
        and len(mesh.vertices) >= 3
        and len(mesh.faces) >= 1
    )
