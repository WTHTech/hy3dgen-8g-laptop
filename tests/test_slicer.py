"""CuraEngine 切片封装回归测试。"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from dfm import CheckResult, CheckStatus, CuraSlicer, DFMReport
from dfm.slicer import _parse_gcode


GCODE_FIXTURE = """;FLAVOR:Marlin
;TIME:6666
;Filament used: 0m
;Layer height: 0.2
;LAYER_COUNT:2
M82
G92 E0
G1 E-2
;LAYER:0
G1 X0 Y0 Z0.2 E0
G1 X10 Y0 E10
G1 E8
G1 E10
;TIME_ELAPSED:30
;LAYER:1
G1 X20 Y5 Z0.4 E15
;TIME_ELAPSED:60
"""


class CuraSlicerUnitTests(unittest.TestCase):
    def _fake_slicer(self, root: Path) -> CuraSlicer:
        runtime = root / 'runtime'
        definitions = runtime / 'share' / 'cura' / 'resources' / 'definitions'
        definitions.mkdir(parents=True)
        engine = runtime / ('CuraEngine.exe' if os.name == 'nt' else 'CuraEngine')
        engine.write_bytes(b'fake')
        printer_def = definitions / 'fdmprinter.def.json'
        extruder_def = definitions / 'fdmextruder.def.json'
        printer_def.write_text('{}', encoding='utf-8')
        extruder_def.write_text('{}', encoding='utf-8')
        return CuraSlicer(
            engine_path=engine,
            resources_path=runtime / 'share' / 'cura' / 'resources',
        )

    def test_parse_gcode_prefers_elapsed_time_and_tracks_extrusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'fixture.gcode'
            path.write_text(GCODE_FIXTURE, encoding='utf-8')
            stats = _parse_gcode(path)

        self.assertAlmostEqual(stats['print_time_min'], 1.0)
        self.assertAlmostEqual(stats['filament_mm'], 15.0)
        self.assertEqual(stats['layer_count'], 2)
        self.assertEqual(stats['extrusion_move_count'], 2)
        np.testing.assert_allclose(
            stats['toolpath_bounds_mm'],
            [[0.0, 0.0, 0.2], [20.0, 5.0, 0.4]],
        )
        self.assertTrue(any('TIME_ELAPSED' in item for item in stats['warnings']))

    def test_command_creates_global_and_extruder_setting_stacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            slicer = self._fake_slicer(Path(temp_dir))
            settings = {
                'layer_height': 0.2,
                'material_diameter': 1.75,
                'machine_nozzle_size': 0.4,
                'machine_nozzle_id': 'AA 0.4',
                'machine_nozzle_offset_x': 1.25,
                'machine_nozzle_offset_y': -0.5,
                'machine_extruder_start_pos_x': 10,
                'machine_extruder_start_pos_y': 20,
            }
            command = slicer._build_command(
                Path('model.stl'), Path('output.gcode'), settings, {}
            )

        extruder_index = command.index('-e0')
        global_args = command[:extruder_index]
        extruder_args = command[extruder_index:]
        self.assertIn('layer_height=0.2', global_args)
        self.assertNotIn('layer_height=0.2', extruder_args)
        self.assertIn('material_diameter=1.75', global_args)
        self.assertIn('material_diameter=1.75', extruder_args)
        self.assertIn('machine_nozzle_size=0.4', extruder_args)
        self.assertIn('machine_nozzle_id=AA 0.4', extruder_args)
        self.assertIn('machine_nozzle_offset_x=1.25', extruder_args)
        self.assertIn('machine_nozzle_offset_y=-0.5', extruder_args)
        self.assertIn('machine_extruder_start_pos_x=10', extruder_args)
        self.assertIn('machine_extruder_start_pos_y=20', extruder_args)
        self.assertIn('-d', command)

    def test_slice_prepared_blocks_failed_report_by_default(self):
        mesh = trimesh.creation.box([20, 20, 20])
        report = DFMReport(
            status='FAIL',
            complete=True,
            prepared_mesh=mesh,
            results=[
                CheckResult(
                    code='P1', name='平台', category='precheck',
                    status=CheckStatus.FAIL,
                )
            ],
        )

        result = CuraSlicer().slice_prepared(report)

        self.assertFalse(result.success)
        self.assertIn('未进入切片', result.error)

    def test_slice_rejects_non_mm_input(self):
        result = CuraSlicer().slice(
            trimesh.creation.box([1, 1, 1]), input_units='normalized'
        )
        self.assertFalse(result.success)
        self.assertIn('prepared_mesh', result.error)

    def test_successful_slice_atomically_replaces_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slicer = self._fake_slicer(root)
            output = root / 'nested' / 'result.gcode'
            output.parent.mkdir()
            output.write_text('old', encoding='utf-8')

            def fake_run(command, **kwargs):
                if len(command) > 1 and command[1] == 'help':
                    return subprocess.CompletedProcess(
                        command, 0, 'Cura_SteamEngine version 5.13.0', ''
                    )
                generated = Path(command[command.index('-o') + 1])
                generated.write_text(GCODE_FIXTURE, encoding='utf-8')
                return subprocess.CompletedProcess(command, 0, '', '')

            mesh = trimesh.creation.box([20, 20, 20])
            mesh.apply_translation([0, 0, 10])
            with patch('dfm.slicer.subprocess.run', side_effect=fake_run):
                result = slicer.slice(mesh, output, input_units='mm')

            self.assertTrue(result.success)
            self.assertEqual(output.read_text(encoding='utf-8'), GCODE_FIXTURE)
            self.assertEqual(result.engine_version, '5.13.0')
            self.assertEqual(list(output.parent.glob('*.part')), [])

    def test_failed_slice_preserves_existing_output_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slicer = self._fake_slicer(root)
            output = root / 'result.gcode'
            output.write_text('known-good', encoding='utf-8')

            def fake_run(command, **kwargs):
                if len(command) > 1 and command[1] == 'help':
                    return subprocess.CompletedProcess(
                        command, 0, 'Cura_SteamEngine version 5.13.0', ''
                    )
                generated = Path(command[command.index('-o') + 1])
                generated.write_text('partial', encoding='utf-8')
                return subprocess.CompletedProcess(
                    command, 1, '', '[error] simulated failure'
                )

            mesh = trimesh.creation.box([20, 20, 20])
            with patch('dfm.slicer.subprocess.run', side_effect=fake_run):
                result = slicer.slice(mesh, output, input_units='mm')

            self.assertFalse(result.success)
            self.assertEqual(output.read_text(encoding='utf-8'), 'known-good')
            self.assertEqual(list(root.glob('*.part')), [])


class CuraSlicerIntegrationTests(unittest.TestCase):
    def test_real_curaengine_slices_mm_cube_when_runtime_available(self):
        slicer = CuraSlicer()
        if not slicer.available:
            self.skipTest(slicer.availability_error)
        mesh = trimesh.creation.box([20, 20, 20])
        mesh.apply_translation([0, 0, 10])

        result = slicer.slice(mesh, timeout=60, input_units='mm')
        if result.gcode_path:
            self.addCleanup(Path(result.gcode_path).unlink, missing_ok=True)

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.engine_version, '5.13.0')
        self.assertEqual(result.layer_count, 100)
        self.assertGreater(result.print_time_min, 0)
        self.assertGreater(result.filament_mm, 0)
        self.assertGreater(result.filament_g, 0)
        self.assertTrue(result.simulation_only)


if __name__ == '__main__':
    unittest.main()
