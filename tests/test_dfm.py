"""DFM 几何真值与回归测试。"""

import json
import unittest

import numpy as np
import trimesh

from dfm import CheckStatus, DFMChecker, DFMReport, DFMRules


def by_code(report, code):
    return next(result for result in report.results if result.code == code)


class DFMGeometryTests(unittest.TestCase):
    def setUp(self):
        self.checker = DFMChecker(DFMRules())

    def test_normalized_mesh_is_scaled_and_same_mesh_is_used_by_full_check(self):
        source = trimesh.creation.box(extents=[0.1, 0.1, 2.0])

        report = self.checker.check_full(source, target_height=100.0)

        self.assertIsNot(report.prepared_mesh, source)
        np.testing.assert_allclose(report.prepared_mesh.extents, [5.0, 5.0, 100.0])
        self.assertAlmostEqual(report.prepared_mesh.bounds[0, 2], 0.0)
        self.assertAlmostEqual(by_code(report, 'W1').metrics['thickness_p5'], 5.0)
        np.testing.assert_allclose(source.extents, [0.1, 0.1, 2.0])

    def test_cube_vertical_walls_are_not_overhangs(self):
        report = self.checker.check_quick(trimesh.creation.box(extents=[20, 20, 20]))

        result = by_code(report, 'S1')
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertAlmostEqual(result.metrics['overhang_ratio'], 0.0)

    def test_elevated_horizontal_plate_is_overhang(self):
        plate = trimesh.creation.box(extents=[40, 40, 2])
        plate.apply_translation([0, 0, 11])
        report = DFMReport(prepared_mesh=plate)

        self.checker._check_overhang_angle(plate, report)

        result = by_code(report, 'S1')
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertGreater(result.metrics['overhang_ratio'], 0.4)

    def test_wall_thickness_ray_hits_are_paired_with_their_origin(self):
        cube = trimesh.creation.box(extents=[20, 20, 20])
        report = self.checker.check_detailed(cube)

        result = by_code(report, 'W1')
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.metrics['valid_ray_count'], 800)
        self.assertAlmostEqual(result.metrics['thickness_p5'], 20.0)

    def test_unverified_rules_are_unknown_and_do_not_enter_score(self):
        report = self.checker.check_detailed(trimesh.creation.box(extents=[20, 20, 20]))

        self.assertEqual(report.status, 'INCOMPLETE')
        self.assertFalse(report.passed)
        self.assertEqual(by_code(report, 'C1').status, CheckStatus.UNKNOWN)
        self.assertEqual(by_code(report, 'C2').status, CheckStatus.NOT_APPLICABLE)
        self.assertAlmostEqual(report.total_score, 100.0)

    def test_empty_and_non_finite_meshes_fail_cleanly(self):
        empty = self.checker.check_quick(trimesh.Trimesh())
        self.assertEqual(empty.status, 'FAIL')
        self.assertEqual(by_code(empty, 'P0').status, CheckStatus.FAIL)

        invalid = trimesh.creation.box()
        invalid.vertices[0, 0] = np.nan
        report = self.checker.check_quick(invalid)
        self.assertEqual(report.status, 'FAIL')
        self.assertIn('NaN', by_code(report, 'P0').detail)

    def test_duplicate_face_is_reported_as_topology_failure(self):
        box = trimesh.creation.box(extents=[20, 20, 20])
        faces = np.vstack([box.faces, box.faces[0]])
        duplicate = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=faces, process=False,
        )

        report = DFMReport(prepared_mesh=duplicate)
        self.checker._check_non_manifold_edges(duplicate, report)
        self.checker._check_degenerate_faces(duplicate, report)

        self.assertEqual(by_code(report, 'G3').status, CheckStatus.FAIL)
        self.assertEqual(by_code(report, 'G5').status, CheckStatus.FAIL)
        self.assertEqual(by_code(report, 'G5').metrics['duplicate_face_count'], 1)

    def test_process_override_is_case_insensitive_and_uses_real_key(self):
        rules = DFMRules()
        rules.set_process('sla')

        self.assertEqual(rules.process, 'SLA')
        self.assertEqual(rules.get('critical_angle'), 30)
        self.assertTrue(rules.get('require_drain_hole'))
        with self.assertRaises(ValueError):
            rules.set_process('unknown')

    def test_report_json_excludes_mesh_payload(self):
        report = self.checker.check_quick(trimesh.creation.box(extents=[20, 20, 20]))
        payload = json.loads(report.to_json())

        self.assertEqual(payload['status'], 'PASS')
        self.assertEqual(payload['prepared_mesh']['face_count'], 12)
        self.assertNotIn('vertices', payload['prepared_mesh'])


if __name__ == '__main__':
    unittest.main()
