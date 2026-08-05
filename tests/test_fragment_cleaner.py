"""DFM 碎片清理器回归测试。"""

import unittest

import numpy as np
import trimesh

from dfm import DFMRules, FragmentCleaner
from dfm.checker import DFMChecker
from dfm.repair import DFMRepairer


class FragmentCleanerUnitTests(unittest.TestCase):
    def setUp(self):
        self.rules = DFMRules()
        self.cleaner = FragmentCleaner(self.rules)

    # ── 基础功能 ────────────────────────────────────

    def test_single_component_returns_unchanged(self):
        """单连通体无需清理。"""
        cube = trimesh.creation.box([20, 20, 20])

        result = self.cleaner.clean(cube)

        self.assertFalse(result.cleaned)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.mesh)
        self.assertEqual(len(result.mesh.faces), 12)
        self.assertEqual(result.components_before, 1)
        self.assertEqual(result.components_after, 1)

    def test_empty_mesh_returns_failure(self):
        result = self.cleaner.clean(trimesh.Trimesh())
        self.assertFalse(result.cleaned)
        self.assertIsNone(result.mesh)

    def test_removes_tiny_floating_fragment(self):
        """主体+小悬浮碎片 → 碎片被删除。"""
        main_body = trimesh.creation.box([20, 20, 20])
        # 悬浮在高处的小碎片
        fragment = trimesh.creation.box([0.5, 0.5, 0.5])
        fragment.apply_translation([5, 5, 30])
        combined = trimesh.util.concatenate([main_body, fragment])

        result = self.cleaner.clean(combined)

        self.assertTrue(result.cleaned)
        self.assertEqual(result.components_before, 2)
        self.assertEqual(result.components_after, 1)
        self.assertEqual(result.removed_component_count, 1)

    def test_keeps_large_floating_component(self):
        """大悬浮分量不应被当作碎片删除。"""
        main_body = trimesh.creation.box([20, 20, 20])
        large_fragment = trimesh.creation.box([10, 10, 10])
        large_fragment.apply_translation([0, 0, 35])
        combined = trimesh.util.concatenate([main_body, large_fragment])

        result = self.cleaner.clean(combined)

        # 10×10×10 相对 20×20×20 比例较大，不应被删
        self.assertFalse(result.cleaned)
        self.assertEqual(result.components_after, 2)

    def test_keeps_large_absolute_accessory_even_when_ratio_is_below_one_percent(self):
        """相对主体占比很小但绝对尺寸明显的配件不得自动删除。"""
        main_body = trimesh.creation.box([200, 200, 200])
        main_body.apply_translation([0, 0, 100])
        accessory = trimesh.creation.box([15, 15, 15])
        accessory.apply_translation([0, 0, 230])

        result = self.cleaner.clean(
            trimesh.util.concatenate([main_body, accessory]),
        )

        self.assertTrue(result.success)
        self.assertFalse(result.cleaned)
        self.assertEqual(result.components_after, 2)
        self.assertGreater(result.protected_components[0]['area_mm2'], 1000)

    def test_protects_small_component_whose_bounds_overlap_main_body(self):
        """嵌入主体但未焊接的小外壳可能是眼睛/纽扣，应保护而不是删除。"""
        main_body = trimesh.creation.box([20, 20, 20])
        accessory = trimesh.creation.box([1, 1, 1])
        accessory.apply_translation([9.75, 0, 0])

        result = self.cleaner.clean(
            trimesh.util.concatenate([main_body, accessory]),
        )

        self.assertFalse(result.cleaned)
        self.assertEqual(len(result.protected_components), 1)
        self.assertTrue(
            result.protected_components[0]['aabb_overlap_with_main'],
        )

    def test_removes_platform_debris(self):
        """接触平台但面积极小的碎屑也应删除。"""
        main_body = trimesh.creation.box([50, 50, 20])
        # 很小的平台碎屑
        debris = trimesh.creation.box([0.3, 0.3, 0.3])
        debris.apply_translation([30, 0, 0.15])
        combined = trimesh.util.concatenate([main_body, debris])

        result = self.cleaner.clean(combined)

        self.assertTrue(result.cleaned)
        self.assertEqual(result.components_after, 1)

    def test_two_platform_components_keep_both(self):
        """两个接触平台的分量都应保留。"""
        left = trimesh.creation.box([10, 10, 20])
        left.apply_translation([-20, 0, 10])
        right = trimesh.creation.box([10, 10, 20])
        right.apply_translation([20, 0, 10])
        combined = trimesh.util.concatenate([left, right])

        result = self.cleaner.clean(combined)

        self.assertFalse(result.cleaned)
        self.assertEqual(result.components_after, 2)

    def test_all_floating_takes_largest_as_main(self):
        """所有分量都不接触平台时，以面积最大的为主体。"""
        body1 = trimesh.creation.box([15, 15, 15])
        body1.apply_translation([0, 0, 10])
        body2 = trimesh.creation.box([0.8, 0.8, 0.8])  # 小碎片
        body2.apply_translation([20, 0, 10])
        combined = trimesh.util.concatenate([body1, body2])

        result = self.cleaner.clean(combined)

        self.assertTrue(result.cleaned, result.detail)
        self.assertEqual(result.components_after, 1)

    # ── 参数校验 ────────────────────────────────────

    def test_negative_ratio_rejected(self):
        result = self.cleaner.clean(
            trimesh.creation.box([20, 20, 20]), max_fragment_ratio=-1,
        )
        self.assertFalse(result.cleaned)
        self.assertIsNone(result.mesh)

    def test_ratio_at_or_above_one_is_rejected(self):
        result = self.cleaner.clean(
            trimesh.creation.box([20, 20, 20]), max_fragment_ratio=1,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.mesh)
        self.assertIn('(0, 1)', result.detail)

    # ── 结果序列化 ──────────────────────────────────

    def test_result_to_dict_serializable(self):
        main_body = trimesh.creation.box([20, 20, 20])
        fragment = trimesh.creation.box([0.5, 0.5, 0.5])
        fragment.apply_translation([5, 5, 30])
        combined = trimesh.util.concatenate([main_body, fragment])

        result = self.cleaner.clean(combined)
        d = result.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn('cleaned', d)
        self.assertIn('removed_component_count', d)
        self.assertIn('removed_components', d)
        self.assertIn('actions', d)
        self.assertGreater(len(d['removed_components']), 0)
        self.assertIn('bounds_mm', d['removed_components'][0])
        self.assertIn('reason', d['removed_components'][0])

    # ── 修复闭环融合 ────────────────────────────────

    def test_g6_failure_triggers_fragment_cleaning(self):
        """G6 失败时修复闭环自动清理碎片。"""
        main_body = trimesh.creation.box([20, 20, 20])
        # 悬浮在上方的小碎片
        fragment = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
        fragment.apply_translation([6, 6, 35])
        combined = trimesh.util.concatenate([main_body, fragment])

        checker = DFMChecker(self.rules)
        repairer = DFMRepairer(self.rules)

        # 先确认初始检查 G6 失败
        init_report = checker.check_quick(combined, input_units='mm')
        g6 = next(r for r in init_report.results if r.code == 'G6')
        self.assertEqual(g6.status.value, 'FAIL')

        result, report = repairer.repair_and_recheck(
            combined, checker, input_units='mm', max_rounds=2,
        )

        # 应该有碎片清理动作
        clean_actions = [
            a for a in result.actions if '碎片' in a.description
        ]
        self.assertGreater(len(clean_actions), 0)
        self.assertTrue(clean_actions[0].success)

        # 修复后应变成单连通体
        self.assertTrue(result.success)
        self.assertEqual(report.status, 'PASS')

    def test_combined_p1_and_g6_retries_pedestal_after_cleaning(self):
        """小碎片阻碍首次融合时，清理后必须重新加底座。"""
        body = trimesh.creation.icosphere(subdivisions=2, radius=10)
        body.apply_translation([0, 0, 10])
        fragment = trimesh.creation.box([0.5, 0.5, 0.5])
        fragment.apply_translation([5, 5, 30])
        combined = trimesh.util.concatenate([body, fragment])
        checker = DFMChecker(self.rules)

        initial = checker.check_quick(combined, input_units='mm')
        failed = {
            item.code for item in initial.results
            if item.status.value == 'FAIL'
        }
        self.assertIn('P1', failed)
        self.assertIn('G6', failed)

        result, report = DFMRepairer(self.rules).repair_and_recheck(
            combined, checker, input_units='mm', max_rounds=2,
        )

        self.assertTrue(result.success, result.summary)
        self.assertEqual(report.status, 'PASS')
        self.assertTrue(any(
            '碎片清理后重试底座' in action.description
            for action in result.actions
        ))

    # ── 可用性 ──────────────────────────────────────

    def test_cleaner_is_available(self):
        self.assertTrue(self.cleaner.available)


if __name__ == '__main__':
    unittest.main()
