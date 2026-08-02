"""不加载模型权重的推理封装回归测试。"""

import sys
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import inference  # noqa: E402


class _FailingPipeline:
    def __call__(self, **kwargs):
        raise RuntimeError("simulated inference failure")


class InferenceLightweightTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
