"""DFM 底座生成器回归测试。"""

import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from dfm import DFMRules, PedestalGenerator, PedestalResult
from dfm.repair import DFMRepairer
from dfm.checker import DFMChecker


class PedestalUnitTests(unittest.TestCase):
    def setUp(self):
        self.rules = DFMRules()
        self.gen = PedestalGenerator(self.rules)

    # ── 基础功能 ────────────────────────────────────

    def test_empty_mesh_returns_failure(self):
        """空网格返回失败。"""
        result = self.gen.generate(trimesh.Trimesh())
        self.assertFalse(result.success)
        self.assertIsNone(result.mesh)

    def test_nan_vertices_return_failure(self):
        """含 NaN 的网格返回失败。"""
        bad = trimesh.creation.box([20, 20, 20])
        bad.vertices[0, 0] = np.nan
        result = self.gen.generate(bad)
        self.assertFalse(result.success)

    def test_disc_on_sphere_point_contact(self):
        """球体点接触：底座生成成功，底面积达标。"""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=10)
        sphere.apply_translation([0, 0, 10])

        result = self.gen.generate(sphere, style='disc')

        self.assertTrue(result.success, result.detail)
        self.assertIsNotNone(result.mesh)
        self.assertGreater(result.footprint_area_mm2, self.rules.get('min_bottom_area'))
        self.assertGreater(len(result.mesh.faces), len(sphere.faces))
        self.assertEqual(len(result.mesh.split(only_watertight=False)), 1)
        self.assertEqual(result.component_count, 1)
        self.assertIn(
            result.merge_method,
            {'boolean_union', 'pymeshlab_boolean_union', 'voxel_union'},
        )
        checked = DFMChecker(self.rules).check_quick(result.mesh, input_units='mm')
        p1 = next(item for item in checked.results if item.code == 'P1')
        self.assertAlmostEqual(
            result.pedestal_bottom_area_mm2,
            p1.metrics['contact_area'],
            places=5,
        )
        # 底座应在模型下方
        self.assertLess(result.mesh.bounds[0, 2], -0.5)

    def test_block_on_two_separated_cylinders(self):
        """两个分离圆柱：block 底座覆盖两个接触点。"""
        cyl1 = trimesh.creation.cylinder(radius=2, height=30, sections=16)
        cyl1.apply_translation([-15, 0, 15])
        cyl2 = trimesh.creation.cylinder(radius=2, height=30, sections=16)
        cyl2.apply_translation([15, 0, 15])
        combined = trimesh.util.concatenate([cyl1, cyl2])

        result = self.gen.generate(combined, style='auto')

        self.assertTrue(result.success, result.detail)
        self.assertEqual(result.style, 'block')
        self.assertEqual(len(result.mesh.split(only_watertight=False)), 1)
        # block 应覆盖两柱之间的跨度
        bounds = result.mesh.bounds
        self.assertGreater(bounds[1, 0] - bounds[0, 0], 30)

    def test_good_contact_mesh_still_gets_stable_base(self):
        """即使接触面积本身较小，底座也应当生成且大于原足迹。"""
        cube = trimesh.creation.box([20, 20, 20])
        cube.apply_translation([0, 0, 10])
        result = self.gen.generate(cube, style='disc')

        self.assertTrue(result.success, result.detail)
        # 底面积应大于原始足迹 400mm²（加上 margin 扩展）
        self.assertGreater(result.footprint_area_mm2, 400)

    def test_unprepared_mesh_is_rejected(self):
        """公开入口不得把未落地网格的中部误当成底座连接面。"""
        result = self.gen.generate(trimesh.creation.box([20, 20, 20]))

        self.assertFalse(result.success)
        self.assertIn('prepared_mesh', result.detail)

    def test_merge_failure_is_not_reported_as_successful_concatenation(self):
        """布尔不可用且禁用体素回退时，禁止退化为多实体直接拼接。"""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=10)
        sphere.apply_translation([0, 0, 10])
        self.rules._raw['repair']['pedestal_allow_voxel_fallback'] = False
        generator = PedestalGenerator(self.rules)

        with (
            patch('trimesh.boolean.union', side_effect=RuntimeError('backend down')),
            patch.dict('sys.modules', {'pymeshlab': None}),
        ):
            result = generator.generate(sphere)

        self.assertFalse(result.success)
        self.assertIsNone(result.mesh)
        self.assertIn('未能融合', result.detail)

    # ── 参数校验 ────────────────────────────────────

    def test_invalid_style_rejected(self):
        result = self.gen.generate(
            trimesh.creation.box([20, 20, 20]), style='hexagon',
        )
        self.assertFalse(result.success)
        self.assertIn('不支持', result.detail)

    def test_negative_margin_rejected(self):
        result = self.gen.generate(
            trimesh.creation.box([20, 20, 20]), margin=-1,
        )
        self.assertFalse(result.success)

    def test_zero_thickness_rejected(self):
        result = self.gen.generate(
            trimesh.creation.box([20, 20, 20]), thickness=0,
        )
        self.assertFalse(result.success)

    def test_out_of_range_taper_angle_rejected(self):
        cube = trimesh.creation.box([20, 20, 20])
        cube.apply_translation([0, 0, 10])

        result = self.gen.generate(cube, taper_angle=120)

        self.assertFalse(result.success)
        self.assertIn('(0, 90]', result.detail)

    # ── 结果序列化 ──────────────────────────────────

    def test_result_to_dict_serializable(self):
        cube = trimesh.creation.box([20, 20, 20])
        cube.apply_translation([0, 0, 10])
        result = self.gen.generate(cube)
        d = result.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn('success', d)
        self.assertIn('footprint_area_mm2', d)
        self.assertIn('actions', d)
        self.assertIn('pedestal_bottom_area_mm2', d)
        self.assertIn('merge_method', d)
        self.assertIsInstance(d['actions'], list)

    # ── 融合到修复闭环 ──────────────────────────────

    def test_p1_failure_auto_generates_pedestal_in_repair_loop(self):
        """P1 失败时修复闭环应自动生成底座，然后继续拓扑修复。"""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=10)
        sphere.apply_translation([0, 0, 10])

        checker = DFMChecker(self.rules)
        repairer = DFMRepairer(self.rules)

        result, report = repairer.repair_and_recheck(
            sphere, checker, input_units='mm', max_rounds=2,
        )

        # 底座生成应在 actions 中
        pedestal_actions = [
            a for a in result.actions
            if '底座' in a.description
        ]
        self.assertGreater(len(pedestal_actions), 0)
        self.assertTrue(pedestal_actions[0].success)

        self.assertTrue(result.success, result.summary)
        self.assertEqual(report.status, 'PASS')
        self.assertIsNotNone(result.final_mesh)
        self.assertEqual(len(result.final_mesh.split(only_watertight=False)), 1)
        # 底座后的网格应有足够底面积
        final_checker = DFMChecker(self.rules)
        final_report = final_checker.check_quick(
            result.final_mesh, input_units='mm',
        )
        p1 = next(r for r in final_report.results if r.code == 'P1')
        self.assertEqual(p1.status.value, 'PASS')
        self.assertAlmostEqual(
            result.actions[0].vertices_after,
            len(result.final_mesh.vertices),
            delta=max(2, len(result.final_mesh.vertices) * 0.02),
        )

    # ── 可用性 ──────────────────────────────────────

    def test_generator_is_available(self):
        self.assertTrue(self.gen.available)


if __name__ == '__main__':
    unittest.main()
