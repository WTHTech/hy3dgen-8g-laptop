"""DFM 修复模块回归测试。"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from dfm import (
    CheckStatus,
    CheckResult,
    DFMChecker,
    DFMRepairer,
    DFMReport,
    DFMRules,
    RepairAction,
    RepairLevel,
)


class DFMRepairTests(unittest.TestCase):
    def setUp(self):
        self.rules = DFMRules()
        self.checker = DFMChecker(self.rules)
        self.repairer = DFMRepairer()

    # ── 轻量修复 ────────────────────────────────────

    def test_light_repair_preserves_clean_mesh(self):
        """完好网格经轻量修复后拓扑不变。"""
        cube = trimesh.creation.box(extents=[20, 20, 20])
        repaired, actions = self.repairer.repair_light(cube)

        self.assertEqual(len(repaired.vertices), 8)
        self.assertEqual(len(repaired.faces), 12)
        self.assertTrue(all(a.success for a in actions))

    def test_light_repair_removes_duplicate_face(self):
        """重复面被删除。"""
        box = trimesh.creation.box(extents=[20, 20, 20])
        faces = np.vstack([box.faces, box.faces[0]])
        bad = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=faces, process=False,
        )

        repaired, actions = self.repairer.repair_light(bad)

        self.assertEqual(len(repaired.faces), 12)
        dup_action = next(a for a in actions if '重复' in a.description)
        self.assertTrue(dup_action.success)

    def test_light_repair_unifies_normals(self):
        """法向被统一。"""
        box = trimesh.creation.box(extents=[20, 20, 20])
        # 翻转一半的面
        faces = box.faces.copy()
        faces[6:] = faces[6:, ::-1]
        bad = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=faces, process=False,
        )

        self.assertFalse(bad.is_winding_consistent)

        repaired, actions = self.repairer.repair_light(bad)

        self.assertTrue(repaired.is_winding_consistent)

    # ── 修复闭环 ────────────────────────────────────

    def test_repair_and_recheck_clean_mesh_returns_immediately(self):
        """完好的网格直接返回 PASS，无修复操作。"""
        cube = trimesh.creation.box(extents=[20, 20, 20])
        result, report = self.repairer.repair_and_recheck(cube, self.checker)

        self.assertTrue(result.success)
        self.assertEqual(report.status, 'PASS')
        self.assertEqual(len(result.actions), 0)

    def test_repair_and_recheck_fixes_non_watertight(self):
        """非水密网格经修复后通过粗筛。"""
        v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64) * 20
        f = np.array([
            [0, 2, 1], [0, 3, 2],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.uint32)
        open_box = trimesh.Trimesh(vertices=v, faces=f, process=False)

        self.assertFalse(open_box.is_watertight)

        result, report = self.repairer.repair_and_recheck(open_box, self.checker)

        self.assertTrue(result.success)
        self.assertEqual(report.status, 'PASS')
        self.assertTrue(result.final_mesh.is_watertight)

    def test_repair_and_recheck_handles_defective_mesh(self):
        """有重复面和非流形边的网格修复后通过。"""
        box = trimesh.creation.box(extents=[20, 20, 20])
        faces = np.vstack([box.faces, box.faces[0]])
        bad = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=faces, process=False,
        )

        result, report = self.repairer.repair_and_recheck(bad, self.checker)

        self.assertTrue(result.success)
        self.assertEqual(report.status, 'PASS')

    # ── 数据结构 ────────────────────────────────────

    def test_repair_result_to_dict(self):
        """RepairResult.to_dict() 返回合法 JSON 结构。"""
        result, _ = self.repairer.repair_and_recheck(
            trimesh.creation.box(extents=[20, 20, 20]),
            self.checker,
        )
        d = result.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn('success', d)
        self.assertIn('actions', d)
        self.assertIsInstance(d['actions'], list)
        self.assertEqual(d['verification_scope'], 'quick')
        self.assertTrue(d['requires_detailed_check'])

    # ── 工具可用性 ──────────────────────────────────

    def test_pymeshlab_and_meshfix_available(self):
        """当前环境 pymeshlab 和 MeshFix 均应可用。"""
        self.assertTrue(self.repairer.pymeshlab_available)
        self.assertTrue(self.repairer.meshfix_available)

    # ── 边界情况 ────────────────────────────────────

    def test_repair_empty_mesh_does_not_crash(self):
        """空网格不应导致崩溃。"""
        empty = trimesh.Trimesh()
        result = self.repairer.repair(empty)

        # 空网格无法修复，但不应抛异常
        self.assertFalse(result.success)
        self.assertIsNone(result.final_mesh)

    def test_stage_failure_paths_return_independent_mesh_copies(self):
        """各公开修复阶段失败时不得把输入 Trimesh 引用直接返回。"""
        empty = trimesh.Trimesh()

        for repair_func in (
            self.repairer.repair_light,
            self.repairer.repair_medium,
            self.repairer.repair_meshfix,
        ):
            returned, actions = repair_func(empty)
            self.assertIsNot(returned, empty)
            self.assertFalse(actions[0].success)

        box = trimesh.creation.box([20, 20, 20])
        with patch('dfm.repair._to_pymeshlab', side_effect=RuntimeError('转换失败')):
            for repair_func in (
                self.repairer.repair_light,
                self.repairer.repair_medium,
            ):
                returned, actions = repair_func(box)
                self.assertIsNot(returned, box)
                np.testing.assert_array_equal(returned.vertices, box.vertices)
                self.assertFalse(actions[0].success)

    def test_recheck_replans_stage_from_latest_failure_codes(self):
        """轻量复检暴露中度缺陷后，应按最新报告动态升级修复。"""
        box = trimesh.creation.box([20, 20, 20])

        def report(status, codes):
            return DFMReport(
                passed=status == 'PASS',
                complete=True,
                status=status,
                prepared_mesh=box.copy(),
                results=[
                    CheckResult(
                        code=code,
                        name=code,
                        category='topology',
                        status=CheckStatus.FAIL,
                    )
                    for code in codes
                ],
            )

        checker = Mock()
        checker.check_quick.side_effect = [
            report('FAIL', ['G4']),
            report('FAIL', ['G2']),
            report('PASS', []),
        ]
        light_action = RepairAction(RepairLevel.LIGHT, '轻量测试修复', True)
        medium_action = RepairAction(RepairLevel.MEDIUM, '中度测试修复', True)

        with (
            patch.object(
                self.repairer,
                'repair_light',
                side_effect=lambda mesh: (mesh.copy(), [light_action]),
            ) as repair_light,
            patch.object(
                self.repairer,
                'repair_medium',
                side_effect=lambda mesh: (mesh.copy(), [medium_action]),
            ) as repair_medium,
        ):
            result, final_report = self.repairer.repair_and_recheck(
                box, checker, input_units='mm', max_rounds=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(final_report.status, 'PASS')
        repair_light.assert_called_once()
        repair_medium.assert_called_once()
        self.assertEqual(
            [action.level for action in result.actions],
            [RepairLevel.LIGHT, RepairLevel.MEDIUM],
        )

    def test_transaction_rolls_back_stage_that_introduces_more_failures(self):
        """B01 类修复回归不得覆盖阶段前分数更高、缺陷更少的网格。"""
        original = trimesh.creation.box([20, 20, 20])
        degraded = original.copy()
        degraded.apply_translation([7, 0, 0])

        def report(mesh, score, codes):
            return DFMReport(
                passed=False,
                complete=True,
                status='FAIL',
                total_score=score,
                prepared_mesh=mesh.copy(),
                results=[
                    CheckResult(
                        code=code,
                        name=code,
                        category='topology',
                        status=CheckStatus.FAIL,
                    )
                    for code in codes
                ],
            )

        checker = Mock()
        checker.check_quick.side_effect = [
            report(original, 70.0, ['G5']),
            report(degraded, 50.0, ['G1', 'G2', 'G4']),
            report(degraded, 100.0, []),
        ]
        light_action = RepairAction(
            RepairLevel.LIGHT, '产生退化结果的轻量修复', True,
        )

        with (
            patch.object(
                self.repairer,
                'repair_light',
                side_effect=lambda mesh: (degraded.copy(), [light_action]),
            ),
            patch.object(
                self.repairer,
                'repair_medium',
                side_effect=lambda mesh: (
                    degraded.copy(),
                    [RepairAction(RepairLevel.MEDIUM, '不应执行的中度修复', True)],
                ),
            ) as repair_medium,
        ):
            result, final_report = self.repairer.repair_and_recheck(
                original, checker, input_units='mm', max_rounds=2,
            )

        self.assertFalse(result.success)
        self.assertEqual(final_report.total_score, 70.0)
        self.assertEqual(
            {item.code for item in final_report.results}, {'G5'},
        )
        np.testing.assert_allclose(result.final_mesh.vertices, original.vertices)
        rollback = [
            action for action in result.actions
            if '事务性验收回滚' in action.description
        ]
        self.assertEqual(len(rollback), 1)
        self.assertIn('70.000', rollback[0].detail)
        self.assertIn('50.000', rollback[0].detail)
        repair_medium.assert_not_called()

    def test_repair_and_recheck_returns_mesh_on_fallback(self):
        """即使修复不完全成功，也应尽量返回可用的网格。"""
        # 两根完全分离的细杆 — 修复也无法让它们通过所有检查
        cyl1 = trimesh.creation.cylinder(radius=0.1, height=50, sections=6)
        cyl1.apply_translation([-10, 0, 25])
        cyl2 = trimesh.creation.cylinder(radius=0.1, height=50, sections=6)
        cyl2.apply_translation([10, 0, 25])
        combined = trimesh.util.concatenate([cyl1, cyl2])

        result, report = self.repairer.repair_and_recheck(combined, self.checker)

        # 不应崩溃，且至少返回了某种网格
        self.assertIsNotNone(result.final_mesh)
        self.assertIsInstance(result.summary, str)
        self.assertGreater(len(result.actions), 0)

    def test_repair_and_recheck_empty_mesh_returns_failure(self):
        """初始检查无法准备网格时不得继续调用修复后端。"""
        result, report = self.repairer.repair_and_recheck(
            trimesh.Trimesh(), self.checker,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.final_mesh)
        self.assertEqual(report.status, 'FAIL')
        self.assertEqual(result.actions, [])

    def test_unknown_check_does_not_trigger_geometry_repair(self):
        """UNKNOWN 应转异步/人工复核，不能进入 MeshFix。"""
        rules = DFMRules()
        rules._raw['performance']['self_intersection_max_faces'] = 1
        checker = DFMChecker(rules)

        result, report = self.repairer.repair_and_recheck(
            trimesh.creation.box([20, 20, 20]), checker,
        )

        self.assertFalse(result.success)
        self.assertEqual(report.status, 'INCOMPLETE')
        self.assertEqual(result.actions, [])

    def test_missing_backends_cannot_report_zero_action_success(self):
        """没有执行修复操作时 success 必须为 False。"""
        box = trimesh.creation.box([20, 20, 20])
        faces = np.vstack([box.faces, box.faces[0]])
        bad = trimesh.Trimesh(vertices=box.vertices, faces=faces, process=False)
        repairer = DFMRepairer()
        repairer._pymeshlab_available = False
        repairer._meshfix_path = None
        repairer._isolate_stages = False

        issue = self.checker.check_quick(bad, input_units='mm').results
        result = repairer.repair(bad, issue)

        self.assertFalse(result.success)
        self.assertGreater(len(result.actions), 0)
        self.assertTrue(any(not action.success for action in result.actions))

    def test_textured_or_colored_mesh_is_rejected_without_data_loss(self):
        """当前修复器无法保留外观属性时，应拒绝而不是静默丢弃。"""
        colored = trimesh.creation.box([20, 20, 20])
        colored.visual.vertex_colors = np.tile(
            np.array([[255, 0, 0, 255]], dtype=np.uint8),
            (len(colored.vertices), 1),
        )

        repaired, actions = self.repairer.repair_light(colored)

        self.assertEqual(repaired.visual.kind, 'vertex')
        self.assertEqual(len(actions), 1)
        self.assertFalse(actions[0].success)

    def test_fully_inward_mesh_is_flipped_outward(self):
        """一致但整体朝内的水密网格也必须修复。"""
        inward = trimesh.creation.box([20, 20, 20])
        inward.invert()

        result, report = self.repairer.repair_and_recheck(
            inward, self.checker, max_rounds=1,
        )

        self.assertTrue(result.success)
        self.assertEqual(report.status, 'PASS')
        self.assertGreater(result.final_mesh.volume, 0)

    def test_platform_failure_does_not_run_meshfix(self):
        """点接触由底座融合修复，不应误用 MeshFix。"""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=10)
        sphere.apply_translation([0, 0, 10])

        result, report = self.repairer.repair_and_recheck(
            sphere, self.checker, input_units='mm',
        )

        self.assertTrue(result.success, result.summary)
        self.assertEqual(report.status, 'PASS')
        self.assertIsNotNone(result.final_mesh)
        self.assertIn(RepairLevel.LIGHT, [a.level for a in result.actions])
        # 不应触发 MeshFix（没有 G 类缺陷）
        meshfix_actions = [
            a for a in result.actions
            if 'MeshFix' in a.description
        ]
        self.assertEqual(meshfix_actions, [])

    def test_stage_timeout_returns_recoverable_mesh_and_progress_state(self):
        """阻塞修复必须被终止，并返回可续跑的最后有效网格。"""
        rules = DFMRules()
        rules._raw['repair']['repair_stage_timeout_seconds'] = 0.001
        events = []
        repairer = DFMRepairer(rules=rules, progress_callback=events.append)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=10)
        sphere.apply_translation([0, 0, 10])

        result, report = repairer.repair_and_recheck(
            sphere, self.checker, input_units='mm',
        )

        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.timeout_stage, 'pedestal')
        self.assertEqual(result.resume_from_stage, 'pedestal')
        self.assertIsNotNone(result.final_mesh)
        np.testing.assert_allclose(result.final_mesh.vertices, sphere.vertices)
        self.assertEqual(report.status, 'FAIL')
        self.assertTrue(any(action.timed_out for action in result.actions))
        self.assertEqual(events[0]['status'], 'started')
        self.assertEqual(events[-1]['status'], 'timed_out')
        payload = result.to_dict()
        self.assertTrue(payload['timed_out'])
        self.assertEqual(payload['resume_from_stage'], 'pedestal')

    def test_low_level_repair_uses_the_same_timeout_boundary(self):
        """低层 repair() 也不得绕过隔离阶段的硬超时。"""
        rules = DFMRules()
        rules._raw['repair']['repair_stage_timeout_seconds'] = 0.001
        repairer = DFMRepairer(rules=rules)
        mesh = trimesh.creation.box([20, 20, 20])
        issues = [
            CheckResult(
                code='G5', name='退化面', category='topology',
                status=CheckStatus.FAIL,
            )
        ]

        result = repairer.repair(mesh, issues)

        self.assertTrue(result.timed_out)
        self.assertEqual(result.timeout_stage, 'light')
        self.assertEqual(result.resume_from_stage, 'light')
        self.assertIsNotNone(result.final_mesh)

    def test_static_instability_triggers_pedestal_and_rechecks(self):
        """P1 已通过但重心越界时，也应自动扩展并融合底座。"""
        foot = trimesh.creation.box([4, 4, 4])
        foot.apply_translation([0, 0, 2])
        upper = trimesh.creation.box([10, 4, 10])
        upper.apply_translation([3, 0, 8.5])
        top_heavy = trimesh.util.concatenate([foot, upper])

        initial = self.checker.check_quick(top_heavy, input_units='mm')
        initial_codes = {
            item.code for item in initial.results
            if item.status == CheckStatus.FAIL
        }
        self.assertNotIn('P1', initial_codes)
        self.assertIn('P3', initial_codes)

        result, report = self.repairer.repair_and_recheck(
            top_heavy, self.checker, input_units='mm',
        )

        self.assertTrue(result.success, result.summary)
        self.assertEqual(report.status, 'PASS')
        self.assertEqual(len(result.final_mesh.split(only_watertight=False)), 1)
        self.assertTrue(any('底座' in action.description for action in result.actions))


if __name__ == '__main__':
    unittest.main()
