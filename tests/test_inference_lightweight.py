"""不加载模型权重的推理封装回归测试。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import trimesh
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import inference  # noqa: E402


class _FailingPipeline:
    def __call__(self, **kwargs):
        raise RuntimeError("simulated inference failure")


class InferenceLightweightTests(unittest.TestCase):
    @staticmethod
    def _generator_with_mesh(mesh):
        generator = object.__new__(inference.Hunyuan3DGenerator)
        generator.generate_shape = lambda image, **kwargs: mesh.copy()
        return generator

    def test_preserves_rgba_input(self):
        image = Image.new("RGBA", (8, 8), (10, 20, 30, 40))
        loaded = inference._load_input_image(image)

        self.assertEqual(loaded.mode, "RGBA")
        self.assertEqual(loaded.getpixel((0, 0)), (10, 20, 30, 40))

    def test_keeps_rgb_input_as_rgb(self):
        image = Image.new("RGB", (8, 8), (10, 20, 30))
        loaded = inference._load_input_image(image)

        self.assertEqual(loaded.mode, "RGB")

    def test_rejects_non_image_input(self):
        with self.assertRaises(TypeError):
            inference._load_input_image(object())

    def test_generate_shape_unloads_after_failure(self):
        generator = object.__new__(inference.Hunyuan3DGenerator)
        generator.variant = "turbo"
        generator.device = "cpu"
        generator._shape_pipeline = _FailingPipeline()
        generator._load_shape_pipeline = lambda: generator._shape_pipeline
        unload_calls = []

        def unload():
            unload_calls.append(True)
            generator._shape_pipeline = None

        generator._unload_shape_pipeline = unload

        with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
            generator.generate_shape(Image.new("RGBA", (8, 8)), enable_pbar=False)

        self.assertEqual(unload_calls, [True])
        self.assertIsNone(generator._shape_pipeline)

    def test_text_to_3d_is_not_exposed_without_model(self):
        self.assertFalse(hasattr(inference.Hunyuan3DGenerator, "text_to_3d"))

    def test_failed_report_export_requires_explicit_diagnostic_override(self):
        from dfm import DFMReport

        generator = self._generator_with_mesh(trimesh.creation.box())
        report = DFMReport(
            status='FAIL',
            prepared_mesh=trimesh.creation.box([10, 10, 10]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'failed.stl'
            with self.assertRaisesRegex(ValueError, '拒绝导出'):
                generator.export_stl(report, output)
            self.assertFalse(output.exists())

            diagnostic = generator.export_stl(
                report, output, allow_failed=True,
            )
            self.assertEqual(diagnostic, output)
            self.assertTrue(output.is_file())

            with self.assertRaisesRegex(ValueError, '\\.stl 后缀'):
                generator.export_stl(report.prepared_mesh, Path(temp_dir) / 'wrong.obj')

    def test_printable_pipeline_blocks_failed_export(self):
        # 球体与平台只有点接触，P1 必须失败，且默认不得写出文件。
        generator = self._generator_with_mesh(
            trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'blocked.stl'
            result = generator.image_to_prepared_3d(
                'unused.png', target_height=100,
                export_stl=output, auto_repair=False,
            )

            self.assertEqual(result['report'].status, 'FAIL')
            self.assertFalse(result['ready_for_slicing'])
            self.assertFalse(result['printable_verified'])
            self.assertFalse(output.exists())
            self.assertEqual(result['exported_paths'], {})
            self.assertIn('未导出', result['export_blocked_reason'])
            self.assertIs(result['prepared_mesh'], result['report'].prepared_mesh)

            diagnostic = Path(temp_dir) / 'diagnostic.stl'
            diagnostic_result = generator.image_to_prepared_3d(
                'unused.png', target_height=100,
                export_stl=diagnostic, auto_repair=False,
                allow_failed_export=True,
            )
            self.assertTrue(diagnostic.is_file())
            self.assertIn('stl', diagnostic_result['exported_paths'])

    def test_repair_reuses_rules_mm_units_and_latest_report(self):
        from dfm import DFMReport, DFMRules, RepairResult

        generator = self._generator_with_mesh(
            trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        )
        rules = DFMRules()
        final_mesh = trimesh.creation.box([1, 1, 1])
        final_report = DFMReport(
            status='FAIL', complete=True, summary='latest recheck',
            prepared_mesh=final_mesh,
        )
        repair_result = RepairResult(
            final_mesh=final_mesh, success=False, summary='still failed',
        )

        with patch('dfm.DFMRepairer') as repairer_class:
            repairer = repairer_class.return_value
            repairer.repair_and_recheck.return_value = (
                repair_result, final_report,
            )
            result = generator.image_to_prepared_3d(
                'unused.png', target_height=1,
                dfm_rules=rules, auto_repair=True,
            )

        repairer_class.assert_called_once_with(rules=rules)
        _, call_kwargs = repairer.repair_and_recheck.call_args
        self.assertEqual(call_kwargs['input_units'], 'mm')
        self.assertEqual(call_kwargs['target_height'], 1)
        self.assertIs(result['report'], final_report)
        self.assertIs(result['prepared_mesh'], final_report.prepared_mesh)

    def test_passed_pipeline_exports_scaled_prepared_mesh(self):
        generator = self._generator_with_mesh(
            trimesh.creation.box([1, 1, 1])
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'prepared.stl'
            glb_output = Path(temp_dir) / 'prepared.glb'
            result = generator.image_to_prepared_3d(
                'unused.png', target_height=20,
                export_stl=output, export_glb=glb_output,
                auto_repair=False,
            )

            self.assertEqual(result['report'].status, 'PASS')
            self.assertTrue(result['ready_for_slicing'])
            self.assertFalse(result['printable_verified'])
            self.assertEqual(result['export_blocked_reason'], '')
            self.assertEqual(result['exported_paths']['stl'], str(output.resolve()))
            self.assertEqual(
                result['exported_paths']['glb'], str(glb_output.resolve())
            )
            reloaded = trimesh.load(output, force='mesh', process=False)
            self.assertAlmostEqual(float(reloaded.extents[2]), 20.0, places=5)
            reloaded_glb = trimesh.load(glb_output, force='mesh', process=False)
            self.assertAlmostEqual(float(reloaded_glb.extents[2]), 20.0, places=5)
            self.assertIs(result['prepared_mesh'], result['report'].prepared_mesh)

            # 旧名称只保留兼容委托，验证语义不变。
            alias_result = generator.image_to_printable_3d(
                'unused.png', target_height=20, auto_repair=False,
            )
            self.assertEqual(alias_result['report'].status, 'PASS')
            self.assertFalse(alias_result['printable_verified'])

    def test_prepared_pipeline_generates_from_selected_preprocessed_image(self):
        mesh = trimesh.creation.box([1, 1, 1])
        generator = object.__new__(inference.Hunyuan3DGenerator)
        generated_from = []

        def generate_shape(image, **kwargs):
            generated_from.append(image)
            return mesh.copy()

        generator.generate_shape = generate_shape
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            selected = temp_dir / 'pedestal.png'
            Image.new('RGBA', (16, 24), (10, 20, 30, 255)).save(selected)
            preprocess_result = SimpleNamespace(selected_image_path=selected)

            class FakePreprocessor:
                def __init__(self):
                    self.calls = []

                def process(self, image, output_dir, options):
                    self.calls.append((image, Path(output_dir), options))
                    return preprocess_result

            preprocessor = FakePreprocessor()
            options = object()
            result = generator.image_to_prepared_3d(
                'original.png',
                target_height=20,
                auto_repair=False,
                image_preprocessor=preprocessor,
                image_preprocess_options=options,
                preprocess_output_dir=temp_dir / 'preprocess',
            )

        self.assertEqual(generated_from, [selected])
        self.assertIs(result['preprocess_result'], preprocess_result)
        self.assertEqual(
            result['generation_image'], str(selected.resolve()),
        )
        self.assertEqual(len(preprocessor.calls), 1)
        self.assertEqual(preprocessor.calls[0][0], 'original.png')
        self.assertIs(preprocessor.calls[0][2], options)

    def test_preprocessor_requires_output_directory_before_generation(self):
        generator = self._generator_with_mesh(trimesh.creation.box())

        class FakePreprocessor:
            def process(self, image, output_dir, options):
                raise AssertionError('must fail before provider call')

        with self.assertRaisesRegex(ValueError, 'preprocess_output_dir'):
            generator.image_to_prepared_3d(
                'original.png',
                auto_repair=False,
                image_preprocessor=FakePreprocessor(),
            )


if __name__ == "__main__":
    unittest.main()
