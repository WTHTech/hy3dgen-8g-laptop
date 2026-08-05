"""复用已落盘 raw mesh 的可恢复 DFM 批量验证入口。

该脚本不运行图生 3D。每个样本按“初检 → 修复/复检 → 精检候选 →
PASS 门禁切片”执行，并在每个阶段完成后原子写入状态与产物。
中断后使用 ``--resume`` 会复用已完成阶段；修复超时默认保留为可审计
终态，只有显式传 ``--retry-timeouts`` 才重新执行修复阶段。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import trimesh

# 直接执行 ``python scripts/batch_dfm_benchmark.py`` 时，将项目根目录加入路径。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dfm import (  # noqa: E402
    CheckResult,
    CuraSlicer,
    DFMChecker,
    DFMRepairer,
    DFMReport,
    DFMRules,
)


SCHEMA_VERSION = 1
TERMINAL_STAGE_STATUSES = frozenset({'COMPLETED', 'SKIPPED', 'TIMED_OUT'})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _atomic_write_json(path: Path, payload: object) -> None:
    """在目标目录写临时文件后原子替换，避免中断留下半截 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise


def _atomic_export_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    """按目标格式导出到同目录临时文件，再原子替换。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f'.{path.stem}.{uuid.uuid4().hex}{path.suffix}'
    try:
        mesh.export(temp_path)
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise RuntimeError(f'网格导出未生成有效文件: {path}')
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force='mesh', process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f'无法从 {path} 加载三角网格')
    if len(loaded.vertices) < 3 or len(loaded.faces) < 1:
        raise ValueError(f'raw mesh 为空或没有有效三角面: {path}')
    return loaded


def _load_report(report_path: Path, mesh_path: Path) -> DFMReport:
    """从分离的 JSON 元数据和网格文件恢复报告，供阶段续跑。"""
    payload = json.loads(report_path.read_text(encoding='utf-8'))
    mesh = _load_mesh(mesh_path)
    results = [
        CheckResult(
            code=item['code'],
            name=item['name'],
            category=item['category'],
            status=item.get('status'),
            score=item.get('score'),
            detail=item.get('detail', ''),
            metrics=item.get('metrics', {}),
            blocking=bool(item.get('blocking', True)),
        )
        for item in payload.get('results', [])
    ]
    return DFMReport(
        passed=bool(payload.get('passed', False)),
        results=results,
        total_score=float(payload.get('total_score', 0.0)),
        summary=str(payload.get('summary', '')),
        status=str(payload.get('status', 'INCOMPLETE')),
        complete=bool(payload.get('complete', False)),
        units=str(payload.get('units', 'mm')),
        transform=payload.get('transform', []),
        prepared_mesh=mesh,
    )


def _failure_codes(report: DFMReport) -> list[str]:
    return [
        item.code for item in report.results
        if getattr(item.status, 'value', item.status) == 'FAIL'
    ]


def _unknown_codes(report: DFMReport) -> list[str]:
    return [
        item.code for item in report.results
        if getattr(item.status, 'value', item.status) == 'UNKNOWN'
    ]


@dataclass(frozen=True)
class BatchConfig:
    output_root: Path
    run_id: str
    target_height_mm: float = 100.0
    resume: bool = False
    retry_timeouts: bool = False
    repair_timeout_seconds: Optional[float] = None
    slice_timeout_seconds: Optional[int] = None
    max_repair_rounds: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, 'output_root', Path(self.output_root).resolve())
        if not self.run_id or self.run_id in {'.', '..'}:
            raise ValueError('run_id 不能为空')
        if Path(self.run_id).name != self.run_id:
            raise ValueError('run_id 不能包含路径分隔符')
        if self.target_height_mm <= 0:
            raise ValueError('target_height_mm 必须为正数')
        if self.repair_timeout_seconds is not None and self.repair_timeout_seconds <= 0:
            raise ValueError('repair_timeout_seconds 必须为正数')
        if self.slice_timeout_seconds is not None and self.slice_timeout_seconds < 1:
            raise ValueError('slice_timeout_seconds 必须为正整数')
        if self.max_repair_rounds < 1:
            raise ValueError('max_repair_rounds 必须至少为 1')


class BatchDFMBenchmark:
    """单进程批量编排器；昂贵修复阶段由 DFMRepairer 自行隔离。"""

    def __init__(
        self,
        config: BatchConfig,
        *,
        rules: Optional[DFMRules] = None,
        checker=None,
        repairer=None,
        slicer=None,
    ):
        self.config = config
        base_rules = rules or DFMRules()
        if config.repair_timeout_seconds is not None:
            rules_payload = base_rules.to_dict()
            rules_payload['repair']['repair_stage_timeout_seconds'] = float(
                config.repair_timeout_seconds
            )
            base_rules = DFMRules.from_dict(rules_payload)
        self.rules = base_rules
        self.checker = checker or DFMChecker(self.rules)
        self._repairer_override = repairer
        self._slicer_override = slicer
        self.run_root = config.output_root / config.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._config_fingerprint = hashlib.sha256(
            json.dumps({
                'rules': self.rules.to_dict(),
                'target_height_mm': config.target_height_mm,
                'max_repair_rounds': config.max_repair_rounds,
                'slice_timeout_seconds': config.slice_timeout_seconds,
            }, ensure_ascii=False, sort_keys=True).encode('utf-8')
        ).hexdigest()

    def _sample_dir(self, image_id: str) -> Path:
        if not image_id or Path(image_id).name != image_id:
            raise ValueError(f'image_id 非法: {image_id!r}')
        return self.run_root / image_id

    def _new_state(self, image_id: str, raw_mesh: Path, source_hash: str) -> dict:
        now = _now()
        return {
            'schema_version': SCHEMA_VERSION,
            'image_id': image_id,
            'run_id': self.config.run_id,
            'status': 'PENDING',
            'source_raw_mesh': str(raw_mesh),
            'source_sha256': source_hash,
            'config_fingerprint': self._config_fingerprint,
            'target_height_mm': self.config.target_height_mm,
            'created_at': now,
            'updated_at': now,
            'stages': {},
        }

    def _load_or_create_state(
        self, image_id: str, raw_mesh: Path, sample_dir: Path,
    ) -> dict:
        state_path = sample_dir / 'state.json'
        source_hash = _sha256(raw_mesh)
        if not state_path.exists():
            state = self._new_state(image_id, raw_mesh, source_hash)
            _atomic_write_json(state_path, state)
            return state
        if not self.config.resume:
            raise FileExistsError(
                f'运行目录已存在: {sample_dir}；请使用 --resume 或更换 --run-id'
            )
        state = json.loads(state_path.read_text(encoding='utf-8'))
        expected = {
            'schema_version': SCHEMA_VERSION,
            'image_id': image_id,
            'run_id': self.config.run_id,
            'source_sha256': source_hash,
            'config_fingerprint': self._config_fingerprint,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items() if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f'恢复状态与当前输入/配置不一致: {mismatches}')
        return state

    @staticmethod
    def _stage(state: dict, name: str) -> dict:
        return state.setdefault('stages', {}).setdefault(name, {})

    def _write_state(self, sample_dir: Path, state: dict) -> None:
        state['updated_at'] = _now()
        _atomic_write_json(sample_dir / 'state.json', state)

    def _begin_stage(self, sample_dir: Path, state: dict, name: str) -> None:
        stage = self._stage(state, name)
        stage.clear()
        stage.update({'status': 'RUNNING', 'started_at': _now()})
        state['status'] = 'RUNNING'
        state['current_stage'] = name
        self._write_state(sample_dir, state)

    def _finish_stage(
        self,
        sample_dir: Path,
        state: dict,
        name: str,
        status: str,
        **metadata,
    ) -> None:
        stage = self._stage(state, name)
        stage.update(metadata)
        stage['status'] = status
        stage['completed_at'] = _now()
        state.pop('current_stage', None)
        self._write_state(sample_dir, state)

    def _stage_reusable(self, state: dict, name: str, sample_dir: Path) -> bool:
        stage = state.get('stages', {}).get(name, {})
        status = stage.get('status')
        if status not in TERMINAL_STAGE_STATUSES:
            return False
        if name == 'repair_recheck' and status == 'TIMED_OUT' and self.config.retry_timeouts:
            return False
        required = {
            'initial_check': ('initial_report.json', 'initial_prepared.stl'),
            'repair_recheck': (
                'repair_result.json', 'final_report.json', 'final_prepared.stl',
            ),
            'detailed_check': (
                ('detailed_report.json',) if status == 'COMPLETED' else ()
            ),
            'slice': ('slice_result.json',),
        }.get(name, ())
        return all((sample_dir / item).is_file() for item in required)

    def _make_repairer(self, sample_dir: Path):
        if self._repairer_override is not None:
            return self._repairer_override
        progress_path = sample_dir / 'repair_progress.json'
        events: list[dict] = []

        def record(event: dict) -> None:
            events.append(event)
            _atomic_write_json(progress_path, {'events': events})

        return DFMRepairer(self.rules, progress_callback=record)

    def _make_slicer(self):
        return self._slicer_override or CuraSlicer(rules=self.rules)

    def _run_sample_inner(
        self,
        image_id: str,
        raw_mesh_path: Path,
        sample_dir: Path,
        state: dict,
    ) -> dict:
        raw_mesh = _load_mesh(raw_mesh_path)

        # 1) 快速初检
        if self._stage_reusable(state, 'initial_check', sample_dir):
            initial_report = _load_report(
                sample_dir / 'initial_report.json',
                sample_dir / 'initial_prepared.stl',
            )
        else:
            self._begin_stage(sample_dir, state, 'initial_check')
            initial_report = self.checker.check_quick(
                raw_mesh,
                self.config.target_height_mm,
                input_units='normalized',
            )
            if initial_report.prepared_mesh is None:
                raise RuntimeError('初检没有生成 prepared_mesh')
            _atomic_write_json(
                sample_dir / 'initial_report.json', initial_report.to_dict(),
            )
            _atomic_export_mesh(
                initial_report.prepared_mesh,
                sample_dir / 'initial_prepared.stl',
            )
            self._finish_stage(
                sample_dir, state, 'initial_check', 'COMPLETED',
                dfm_status=initial_report.status,
                score=initial_report.total_score,
            )

        # 2) 修复与快速复检
        repair_timed_out = False
        if self._stage_reusable(state, 'repair_recheck', sample_dir):
            repair_payload = json.loads(
                (sample_dir / 'repair_result.json').read_text(encoding='utf-8')
            )
            repair_timed_out = bool(repair_payload.get('timed_out', False))
            final_report = _load_report(
                sample_dir / 'final_report.json',
                sample_dir / 'final_prepared.stl',
            )
        else:
            self._begin_stage(sample_dir, state, 'repair_recheck')
            repairer = self._make_repairer(sample_dir)
            repair_result, final_report = repairer.repair_and_recheck(
                raw_mesh,
                self.checker,
                target_height=self.config.target_height_mm,
                max_rounds=self.config.max_repair_rounds,
                input_units='normalized',
            )
            final_mesh = final_report.prepared_mesh or repair_result.final_mesh
            if final_mesh is None:
                raise RuntimeError('修复/复检没有返回最后有效网格')
            final_report.prepared_mesh = final_mesh
            repair_result.final_mesh = final_mesh
            repair_timed_out = bool(repair_result.timed_out)
            _atomic_write_json(
                sample_dir / 'repair_result.json', repair_result.to_dict(),
            )
            _atomic_write_json(
                sample_dir / 'final_report.json', final_report.to_dict(),
            )
            _atomic_export_mesh(final_mesh, sample_dir / 'final_prepared.stl')
            self._finish_stage(
                sample_dir,
                state,
                'repair_recheck',
                'TIMED_OUT' if repair_timed_out else 'COMPLETED',
                dfm_status=final_report.status,
                repair_success=repair_result.success,
                timed_out=repair_timed_out,
                timeout_stage=repair_result.timeout_stage,
                resume_from_stage=repair_result.resume_from_stage,
            )

        # 3) 精检候选。快速层不通过时按设计阻断，不伪造精检结果。
        detailed_report: Optional[DFMReport] = None
        if self._stage_reusable(state, 'detailed_check', sample_dir):
            detailed_stage = state.get('stages', {}).get('detailed_check', {})
            if detailed_stage.get('status') == 'COMPLETED':
                detailed_report = _load_report(
                    sample_dir / 'detailed_report.json',
                    sample_dir / 'final_prepared.stl',
                )
        else:
            if final_report.status == 'PASS':
                self._begin_stage(sample_dir, state, 'detailed_check')
                detailed_report = self.checker.check_detailed(
                    final_report.prepared_mesh
                )
                _atomic_write_json(
                    sample_dir / 'detailed_report.json', detailed_report.to_dict(),
                )
                self._finish_stage(
                    sample_dir, state, 'detailed_check', 'COMPLETED',
                    dfm_status=detailed_report.status,
                    score=detailed_report.total_score,
                )
            else:
                self._finish_stage(
                    sample_dir, state, 'detailed_check', 'SKIPPED',
                    reason=f'快速复检状态为 {final_report.status}，按门禁跳过精检',
                )

        # 4) 只有快速复检 PASS 才进入 CuraEngine。
        slice_payload: dict
        if self._stage_reusable(state, 'slice', sample_dir):
            slice_payload = json.loads(
                (sample_dir / 'slice_result.json').read_text(encoding='utf-8')
            )
        elif final_report.status == 'PASS':
            self._begin_stage(sample_dir, state, 'slice')
            slicer = self._make_slicer()
            slice_result = slicer.slice_prepared(
                final_report,
                sample_dir / 'model.gcode',
                timeout=self.config.slice_timeout_seconds,
            )
            slice_payload = slice_result.to_dict()
            slice_payload['skipped'] = False
            _atomic_write_json(sample_dir / 'slice_result.json', slice_payload)
            self._finish_stage(
                sample_dir, state, 'slice', 'COMPLETED',
                success=slice_result.success,
                error=slice_result.error,
            )
        else:
            slice_payload = {
                'success': False,
                'skipped': True,
                'reason': f'快速复检状态为 {final_report.status}，未进入切片',
            }
            _atomic_write_json(sample_dir / 'slice_result.json', slice_payload)
            self._finish_stage(
                sample_dir, state, 'slice', 'SKIPPED',
                reason=slice_payload['reason'],
            )

        status = 'COMPLETED_WITH_TIMEOUT' if repair_timed_out else 'COMPLETED'
        manufacturing_status = (
            final_report.status
            if final_report.status != 'PASS'
            else (
                detailed_report.status
                if detailed_report is not None else 'INCOMPLETE'
            )
        )
        state['status'] = status
        state['completed_at'] = _now()
        state.pop('error', None)
        self._write_state(sample_dir, state)
        return {
            'image_id': image_id,
            'status': status,
            'initial_dfm_status': initial_report.status,
            'initial_score': initial_report.total_score,
            'initial_failed_codes': _failure_codes(initial_report),
            'initial_unknown_codes': _unknown_codes(initial_report),
            # 兼容首轮汇总字段；其语义明确为“最终快速层”，不是完整制造性结论。
            'final_dfm_status': final_report.status,
            'final_quick_status': final_report.status,
            'final_score': final_report.total_score,
            'final_failed_codes': _failure_codes(final_report),
            'final_unknown_codes': _unknown_codes(final_report),
            'detailed_dfm_status': (
                detailed_report.status if detailed_report is not None else 'SKIPPED'
            ),
            'detailed_score': (
                detailed_report.total_score if detailed_report is not None else None
            ),
            'detailed_failed_codes': (
                _failure_codes(detailed_report) if detailed_report is not None else []
            ),
            'detailed_unknown_codes': (
                _unknown_codes(detailed_report) if detailed_report is not None else []
            ),
            'manufacturing_status': manufacturing_status,
            'repair_timed_out': repair_timed_out,
            'timeout_stage': state['stages']['repair_recheck'].get('timeout_stage', ''),
            'resume_from_stage': state['stages']['repair_recheck'].get(
                'resume_from_stage', ''
            ),
            'slice_success': bool(slice_payload.get('success', False)),
            'slice_skipped': bool(slice_payload.get('skipped', False)),
            'sample_dir': str(sample_dir),
        }

    def run_sample(self, image_id: str, raw_mesh: str | Path) -> dict:
        raw_mesh_path = Path(raw_mesh).expanduser().resolve()
        if not raw_mesh_path.is_file():
            return {
                'image_id': image_id,
                'status': 'FAILED',
                'error': f'raw mesh 不存在: {raw_mesh_path}',
            }
        sample_dir = self._sample_dir(image_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        state: Optional[dict] = None
        try:
            state = self._load_or_create_state(
                image_id, raw_mesh_path, sample_dir,
            )
            state['status'] = 'RUNNING'
            self._write_state(sample_dir, state)
            return self._run_sample_inner(
                image_id, raw_mesh_path, sample_dir, state,
            )
        except Exception as exc:
            # 只有通过输入哈希与配置指纹校验、归属于本次调用的状态才能
            # 写失败信息。配置不兼容或未传 --resume 时不得污染旧检查点。
            if state is not None:
                try:
                    current_stage = state.get('current_stage')
                    if current_stage:
                        stage = self._stage(state, current_stage)
                        stage.update({
                            'status': 'FAILED',
                            'completed_at': _now(),
                            'error': f'{type(exc).__name__}: {exc}',
                        })
                    state['status'] = 'FAILED'
                    state['error'] = f'{type(exc).__name__}: {exc}'
                    state['traceback'] = traceback.format_exc()
                    state.pop('current_stage', None)
                    self._write_state(sample_dir, state)
                except Exception:
                    pass
            return {
                'image_id': image_id,
                'status': 'FAILED',
                'error': f'{type(exc).__name__}: {exc}',
                'sample_dir': str(sample_dir),
            }

    def run(self, samples: Mapping[str, str | Path]) -> dict:
        invocation_started_at = _now()
        started_at = invocation_started_at
        existing_summary_path = self.run_root / 'batch_summary.json'
        if self.config.resume and existing_summary_path.is_file():
            try:
                existing_summary = json.loads(
                    existing_summary_path.read_text(encoding='utf-8')
                )
                started_at = str(existing_summary.get('started_at') or started_at)
            except (OSError, ValueError, TypeError):
                # 单样本 state.json 才是恢复依据；旧汇总损坏不应阻断重建。
                pass
        results: list[dict] = []
        for image_id, raw_mesh in samples.items():
            try:
                result = self.run_sample(image_id, raw_mesh)
            except Exception as exc:
                # 防御性边界：即使 run_sample 被外部替换/扩展，也不能阻断全批次。
                result = {
                    'image_id': image_id,
                    'status': 'FAILED',
                    'error': f'{type(exc).__name__}: {exc}',
                }
            results.append(result)
            _atomic_write_json(
                self.run_root / 'batch_progress.json',
                {'run_id': self.config.run_id, 'samples': results},
            )

        counts = dict(Counter(item['status'] for item in results))
        summary = {
            'schema_version': SCHEMA_VERSION,
            'run_id': self.config.run_id,
            'started_at': started_at,
            'last_invocation_started_at': invocation_started_at,
            'completed_at': _now(),
            'sample_count': len(results),
            'counts': counts,
            'samples': results,
        }
        _atomic_write_json(self.run_root / 'batch_summary.json', summary)
        csv_fields = [
            'image_id', 'status', 'initial_dfm_status', 'initial_score',
            'initial_failed_codes', 'initial_unknown_codes',
            'final_quick_status', 'final_score', 'final_failed_codes',
            'final_unknown_codes', 'repair_timed_out', 'timeout_stage',
            'detailed_dfm_status', 'detailed_score', 'detailed_failed_codes',
            'detailed_unknown_codes', 'manufacturing_status',
            'resume_from_stage', 'slice_success', 'slice_skipped',
            'error', 'sample_dir',
        ]
        csv_rows = []
        for item in results:
            row = {key: item.get(key, '') for key in csv_fields}
            for key in (
                'initial_failed_codes', 'initial_unknown_codes',
                'final_failed_codes', 'final_unknown_codes',
                'detailed_failed_codes', 'detailed_unknown_codes',
            ):
                if isinstance(row[key], list):
                    row[key] = '|'.join(row[key])
            csv_rows.append(row)
        _atomic_write_csv(self.run_root / 'batch_summary.csv', csv_rows, csv_fields)
        return summary


def discover_raw_meshes(source_runs_root: Path, image_ids: list[str]) -> dict[str, Path]:
    """为每个 image_id 查找唯一的 ``raw_mesh.*``，歧义时拒绝猜测。"""
    source_runs_root = Path(source_runs_root).resolve()
    result: dict[str, Path] = {}
    for image_id in image_ids:
        sample_root = source_runs_root / image_id
        if not sample_root.is_dir():
            raise FileNotFoundError(f'缺少样本目录: {sample_root}')
        matches = sorted(
            path for path in sample_root.rglob('raw_mesh.*')
            if path.suffix.lower() in {'.glb', '.gltf', '.ply', '.stl', '.obj'}
        )
        if len(matches) != 1:
            raise ValueError(
                f'{image_id} 必须且只能有一个 raw_mesh，实际找到 {len(matches)} 个: '
                f'{[str(path) for path in matches]}'
            )
        result[image_id] = matches[0]
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='复用 raw mesh 执行可恢复 DFM/修复/复检/切片批量验证',
    )
    parser.add_argument(
        '--source-runs-root', type=Path,
        default=PROJECT_ROOT / 'benchmarks' / 'dfm_v1' / 'runs',
    )
    parser.add_argument(
        '--output-root', type=Path,
        default=PROJECT_ROOT / 'benchmarks' / 'dfm_v1' / 'reruns',
    )
    parser.add_argument('--run-id', default='dfm_full_rerun_20260804')
    parser.add_argument(
        '--image-ids', nargs='+', default=['A01', 'B01', 'C01', 'D01', 'E01'],
    )
    parser.add_argument('--target-height', type=float, default=100.0)
    parser.add_argument('--repair-timeout', type=float)
    parser.add_argument('--slice-timeout', type=int)
    parser.add_argument('--max-repair-rounds', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-timeouts', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = BatchConfig(
        output_root=args.output_root,
        run_id=args.run_id,
        target_height_mm=args.target_height,
        resume=args.resume,
        retry_timeouts=args.retry_timeouts,
        repair_timeout_seconds=args.repair_timeout,
        slice_timeout_seconds=args.slice_timeout,
        max_repair_rounds=args.max_repair_rounds,
    )
    try:
        samples = discover_raw_meshes(args.source_runs_root, args.image_ids)
        runner = BatchDFMBenchmark(config)
        summary = runner.run(samples)
    except Exception as exc:
        print(f'[batch-dfm] 启动失败: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

    for item in summary['samples']:
        print(
            f"[batch-dfm] {item['image_id']}: {item['status']} "
            f"initial={item.get('initial_dfm_status', '-')} "
            f"final={item.get('final_dfm_status', '-')} "
            f"slice={item.get('slice_success', False)}"
        )
    print(f"[batch-dfm] 汇总: {config.output_root / config.run_id / 'batch_summary.json'}")
    return 1 if summary['counts'].get('FAILED', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
