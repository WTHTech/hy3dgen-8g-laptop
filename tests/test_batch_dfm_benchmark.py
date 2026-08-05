"""可恢复 DFM 批量脚本的行为回归测试。"""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import trimesh

from dfm import DFMReport, RepairResult, SliceResult
from scripts.batch_dfm_benchmark import BatchConfig, BatchDFMBenchmark


def _report(mesh: trimesh.Trimesh, status: str) -> DFMReport:
    return DFMReport(
        passed=status == 'PASS',
        complete=True,
        total_score=100.0 if status == 'PASS' else 50.0,
        summary=f'{status} test report',
        status=status,
        units='mm',
        transform=[],
        prepared_mesh=mesh.copy(),
    )


class _Checker:
    def __init__(
        self, *, fail_if_quick_called: bool = False,
        detailed_status: str = 'PASS',
    ):
        self.fail_if_quick_called = fail_if_quick_called
        self.detailed_status = detailed_status
        self.quick_calls = 0

    def check_quick(self, mesh, target_height=100.0, input_units='auto'):
        if self.fail_if_quick_called:
            raise AssertionError('恢复时不应重跑已完成的初检阶段')
        self.quick_calls += 1
        return _report(mesh, 'FAIL')

    def check_detailed(self, mesh):
        return _report(mesh, self.detailed_status)


class _FailingRepairer:
    def repair_and_recheck(self, *args, **kwargs):
        raise RuntimeError('simulated repair crash')


class _SuccessfulRepairer:
    def repair_and_recheck(self, mesh, checker, **kwargs):
        return (
            RepairResult(
                final_mesh=mesh.copy(), success=True,
                verification_scope='quick', summary='repaired',
            ),
            _report(mesh, 'PASS'),
        )


class _TimeoutRepairer:
    def repair_and_recheck(self, mesh, checker, **kwargs):
        return (
            RepairResult(
                final_mesh=mesh.copy(), success=False,
                verification_scope='quick', summary='timed out',
                timed_out=True, timeout_stage='pedestal',
                resume_from_stage='pedestal',
            ),
            _report(mesh, 'FAIL'),
        )


class _FailIfUsedRepairer:
    def repair_and_recheck(self, *args, **kwargs):
        raise AssertionError('已落盘超时结果默认不应在 resume 时自动重试')


class _Slicer:
    def slice_prepared(self, report, output_path, **kwargs):
        output_path = Path(output_path)
        output_path.write_text('; test gcode\n', encoding='utf-8')
        return SliceResult(success=True, gcode_path=str(output_path))


class _FailIfUsedSlicer:
    def slice_prepared(self, *args, **kwargs):
        raise AssertionError('最终快速报告不是 PASS 时不得进入切片')


class BatchDFMBenchmarkTests(unittest.TestCase):
    def _raw_mesh(self, root: Path, image_id: str = 'A01') -> Path:
        path = root / image_id / 'source' / 'raw_mesh.glb'
        path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(extents=[0.5, 1.0, 0.25]).export(path)
        return path

    def _config(
        self,
        root: Path,
        *,
        resume: bool,
        retry_timeouts: bool = False,
        target_height_mm: float = 100.0,
    ) -> BatchConfig:
        return BatchConfig(
            output_root=root / 'output',
            run_id='unit_resume',
            target_height_mm=target_height_mm,
            resume=resume,
            retry_timeouts=retry_timeouts,
        )

    def test_resume_reuses_initial_check_after_repair_failure(self):
        """初检已原子落盘后修复崩溃，续跑不得重复初检。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._raw_mesh(root)
            first_checker = _Checker()
            first = BatchDFMBenchmark(
                self._config(root, resume=False),
                checker=first_checker,
                repairer=_FailingRepairer(),
                slicer=_FailIfUsedSlicer(),
            )

            first_result = first.run_sample('A01', raw)
            self.assertEqual(first_result['status'], 'FAILED')
            self.assertEqual(first_checker.quick_calls, 1)

            resumed = BatchDFMBenchmark(
                self._config(root, resume=True),
                checker=_Checker(fail_if_quick_called=True),
                repairer=_SuccessfulRepairer(),
                slicer=_Slicer(),
            )
            resumed_result = resumed.run_sample('A01', raw)

            self.assertEqual(resumed_result['status'], 'COMPLETED')
            self.assertEqual(resumed_result['final_quick_status'], 'PASS')
            self.assertEqual(resumed_result['detailed_dfm_status'], 'PASS')
            self.assertEqual(resumed_result['manufacturing_status'], 'PASS')
            sample_dir = root / 'output' / 'unit_resume' / 'A01'
            self.assertTrue((sample_dir / 'initial_report.json').is_file())
            self.assertTrue((sample_dir / 'final_report.json').is_file())
            self.assertTrue((sample_dir / 'slice_result.json').is_file())
            self.assertTrue((sample_dir / 'model.gcode').is_file())

    def test_resume_keeps_timeout_terminal_until_retry_is_requested(self):
        """超时结果可审计复用，显式 retry_timeouts 才重新进入修复。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._raw_mesh(root)
            initial = BatchDFMBenchmark(
                self._config(root, resume=False),
                checker=_Checker(), repairer=_TimeoutRepairer(),
                slicer=_FailIfUsedSlicer(),
            )
            timed_out = initial.run_sample('A01', raw)
            self.assertEqual(timed_out['status'], 'COMPLETED_WITH_TIMEOUT')

            reused = BatchDFMBenchmark(
                self._config(root, resume=True),
                checker=_Checker(fail_if_quick_called=True),
                repairer=_FailIfUsedRepairer(),
                slicer=_FailIfUsedSlicer(),
            )
            reused_result = reused.run_sample('A01', raw)
            self.assertEqual(reused_result['status'], 'COMPLETED_WITH_TIMEOUT')

            retried = BatchDFMBenchmark(
                self._config(root, resume=True, retry_timeouts=True),
                checker=_Checker(fail_if_quick_called=True),
                repairer=_SuccessfulRepairer(),
                slicer=_Slicer(),
            )
            retried_result = retried.run_sample('A01', raw)
            self.assertEqual(retried_result['status'], 'COMPLETED')

    def test_detailed_failure_is_not_hidden_by_successful_slice_simulation(self):
        """快速层 PASS 可做切片仿真，但精检 FAIL 必须成为综合制造性结论。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._raw_mesh(root)
            runner = BatchDFMBenchmark(
                self._config(root, resume=False),
                checker=_Checker(detailed_status='FAIL'),
                repairer=_SuccessfulRepairer(),
                slicer=_Slicer(),
            )

            result = runner.run_sample('A01', raw)

            self.assertEqual(result['final_quick_status'], 'PASS')
            self.assertEqual(result['detailed_dfm_status'], 'FAIL')
            self.assertEqual(result['manufacturing_status'], 'FAIL')
            self.assertTrue(result['slice_success'])

    def test_batch_continues_after_one_sample_fails(self):
        """单个样本异常必须写入汇总，但不能阻断后续样本。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_a = self._raw_mesh(root, 'A01')
            raw_b = self._raw_mesh(root, 'B01')
            runner = BatchDFMBenchmark(
                self._config(root, resume=False),
                checker=_Checker(), repairer=_SuccessfulRepairer(),
                slicer=_Slicer(),
            )
            original_run_sample = runner.run_sample

            def run_sample(image_id, raw_mesh):
                if image_id == 'A01':
                    raise RuntimeError('simulated sample failure')
                return original_run_sample(image_id, raw_mesh)

            runner.run_sample = run_sample
            summary = runner.run({'A01': raw_a, 'B01': raw_b})

            by_id = {item['image_id']: item for item in summary['samples']}
            self.assertEqual(by_id['A01']['status'], 'FAILED')
            self.assertEqual(by_id['B01']['status'], 'COMPLETED')
            self.assertEqual(summary['counts']['FAILED'], 1)
            self.assertEqual(summary['counts']['COMPLETED'], 1)
            self.assertTrue(
                (root / 'output' / 'unit_resume' / 'batch_summary.json').is_file()
            )

    def test_resume_preserves_original_batch_start_time(self):
        """汇总续写不能把首次批次开始时间改成恢复命令的时间。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._raw_mesh(root)
            first = BatchDFMBenchmark(
                self._config(root, resume=False),
                checker=_Checker(), repairer=_SuccessfulRepairer(),
                slicer=_Slicer(),
            )
            first.run({'A01': raw})
            summary_path = root / 'output' / 'unit_resume' / 'batch_summary.json'
            payload = json.loads(summary_path.read_text(encoding='utf-8'))
            payload['started_at'] = '2020-01-01T00:00:00+00:00'
            summary_path.write_text(json.dumps(payload), encoding='utf-8')

            resumed = BatchDFMBenchmark(
                self._config(root, resume=True),
                checker=_Checker(fail_if_quick_called=True),
                repairer=_FailIfUsedRepairer(), slicer=_FailIfUsedSlicer(),
            )
            summary = resumed.run({'A01': raw})

            self.assertEqual(summary['started_at'], '2020-01-01T00:00:00+00:00')
            self.assertIn('last_invocation_started_at', summary)

    def test_incompatible_resume_does_not_overwrite_valid_state(self):
        """输入或配置指纹不一致时应拒绝恢复，但必须保留旧检查点。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._raw_mesh(root)
            first = BatchDFMBenchmark(
                self._config(root, resume=False),
                checker=_Checker(), repairer=_SuccessfulRepairer(),
                slicer=_Slicer(),
            )
            first_result = first.run_sample('A01', raw)
            self.assertEqual(first_result['status'], 'COMPLETED')
            state_path = root / 'output' / 'unit_resume' / 'A01' / 'state.json'
            original_state = state_path.read_bytes()

            incompatible = BatchDFMBenchmark(
                self._config(root, resume=True, target_height_mm=80.0),
                checker=_Checker(fail_if_quick_called=True),
                repairer=_FailIfUsedRepairer(), slicer=_FailIfUsedSlicer(),
            )
            result = incompatible.run_sample('A01', raw)

            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('配置不一致', result['error'])
            self.assertEqual(state_path.read_bytes(), original_state)


if __name__ == '__main__':
    unittest.main()
