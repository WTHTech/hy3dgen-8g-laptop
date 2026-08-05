"""AI 底座图像预处理的无网络回归测试。"""

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from image_preprocess import (  # noqa: E402
    PedestalImagePreprocessor,
    PreprocessOptions,
    analyze_image,
    build_edit_canvas,
    should_add_pedestal,
)
from image_preprocess.providers import (  # noqa: E402
    ImageEditError,
    OpenAIImageEditProvider,
    PedestalEditRequest,
)
from scripts import preprocess_image  # noqa: E402


def _figurine_image(size=(64, 64)):
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 3, 52, 43), fill=(200, 100, 80, 255))
    draw.rectangle((18, 28, 46, 52), fill=(50, 100, 180, 255))
    draw.rectangle((24, 52, 28, 61), fill=(50, 50, 50, 255))
    draw.rectangle((36, 52, 40, 61), fill=(50, 50, 50, 255))
    return image


def _png_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class _FakeImagesEndpoint:
    def __init__(self, output_image=None, error=None):
        self.output_image = output_image
        self.error = error
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=_png_b64(self.output_image))]
        )


class _FakeClient:
    def __init__(self, endpoint):
        self.images = endpoint


class _StatusError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


class _NeverCalledProvider:
    name = 'never-called'

    def add_pedestal(self, image, mask, request):
        raise AssertionError('provider should not be called')


class _SuccessfulProvider:
    name = 'fake-success'

    def __init__(self):
        self.calls = []

    def add_pedestal(self, image, mask, request):
        self.calls.append((image.copy(), mask.copy(), request))
        output = image.copy()
        draw = ImageDraw.Draw(output)
        draw.ellipse(
            (output.width // 6, output.height * 3 // 4,
             output.width * 5 // 6, output.height - 4),
            fill=(100, 100, 100, 255),
        )
        return output


class _FailingProvider:
    name = 'fake-failure'

    def add_pedestal(self, image, mask, request):
        raise ImageEditError('provider_error')


class _OpaqueProvider:
    name = 'fake-opaque'

    def add_pedestal(self, image, mask, request):
        return Image.new('RGBA', image.size, (240, 240, 240, 255))


class ImagePreprocessTests(unittest.TestCase):
    def test_auto_requests_pedestal_for_narrow_figurine_contact(self):
        analysis = analyze_image(_figurine_image())

        self.assertGreater(analysis.subject_area_ratio, 0)
        self.assertLess(analysis.bottom_contact_ratio, 0.25)
        self.assertTrue(
            should_add_pedestal('auto', 'figurine', analysis)
        )
        self.assertFalse(
            should_add_pedestal('auto', 'pendant', analysis)
        )
        self.assertTrue(
            should_add_pedestal('always', 'pendant', analysis)
        )
        self.assertFalse(
            should_add_pedestal('never', 'figurine', analysis)
        )

    def test_build_edit_canvas_protects_upper_region_and_opens_bottom(self):
        canvas, mask, placement = build_edit_canvas(
            _figurine_image(), canvas_size=(96, 128),
        )

        self.assertEqual(canvas.mode, 'RGBA')
        self.assertEqual(mask.mode, 'RGBA')
        self.assertEqual(canvas.size, mask.size)
        self.assertEqual(mask.getpixel((5, 5))[3], 255)
        self.assertEqual(mask.getpixel((5, 127))[3], 0)
        self.assertLess(placement.edit_start_y, placement.subject_bottom_y)
        self.assertGreater(placement.subject_bottom_y, placement.subject_top_y)

    def test_openai_provider_sends_mask_and_decodes_transparent_png(self):
        edited = Image.new('RGBA', (32, 48), (10, 20, 30, 0))
        endpoint = _FakeImagesEndpoint(output_image=edited)
        provider = OpenAIImageEditProvider(
            client=_FakeClient(endpoint), timeout_seconds=45,
        )
        request = PedestalEditRequest(
            prompt='add a base', size='1024x1536', quality='medium',
        )

        output = provider.add_pedestal(
            Image.new('RGBA', (32, 48)),
            Image.new('RGBA', (32, 48), (0, 0, 0, 255)),
            request,
        )

        self.assertEqual(output.mode, 'RGBA')
        self.assertEqual(output.size, (32, 48))
        self.assertEqual(len(endpoint.calls), 1)
        call = endpoint.calls[0]
        self.assertEqual(call['model'], 'gpt-image-2')
        self.assertEqual(call['background'], 'transparent')
        self.assertEqual(call['input_fidelity'], 'high')
        self.assertEqual(call['output_format'], 'png')
        self.assertEqual(call['response_format'], 'b64_json')
        self.assertEqual(call['timeout'], 45)
        self.assertEqual(call['prompt'], 'add a base')
        self.assertTrue(hasattr(call['image'], 'read'))
        self.assertTrue(hasattr(call['mask'], 'read'))

    def test_openai_provider_maps_status_without_leaking_error_text(self):
        endpoint = _FakeImagesEndpoint(
            error=_StatusError(401, 'sensitive upstream response'),
        )
        provider = OpenAIImageEditProvider(client=_FakeClient(endpoint))

        with self.assertRaises(ImageEditError) as raised:
            provider.add_pedestal(
                Image.new('RGBA', (32, 48)),
                Image.new('RGBA', (32, 48)),
                PedestalEditRequest(prompt='add a base'),
            )

        self.assertEqual(raised.exception.code, 'authentication_error')
        self.assertNotIn('sensitive', str(raised.exception))

    def test_openai_provider_requires_environment_key_without_client(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ImageEditError, 'missing_api_key'):
                OpenAIImageEditProvider()

    def test_openai_provider_does_not_inherit_third_party_base_url(self):
        with patch.dict(
            os.environ,
            {
                'OPENAI_API_KEY': 'unit-test-key',
                'OPENAI_BASE_URL': 'https://third-party.example/v1',
            },
            clear=False,
        ):
            with patch('openai.OpenAI') as client_factory:
                OpenAIImageEditProvider()

        self.assertEqual(client_factory.call_count, 1)
        self.assertEqual(
            client_factory.call_args.kwargs['base_url'],
            'https://api.openai.com/v1',
        )

    def test_pipeline_saves_traceable_success_artifacts(self):
        provider = _SuccessfulProvider()
        options = PreprocessOptions(
            mode='always', artifact_type='figurine',
            canvas_size=(96, 128), api_size='1024x1536',
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = PedestalImagePreprocessor(provider).process(
                _figurine_image(), Path(temp_dir), options,
            )

            self.assertTrue(result.applied)
            self.assertFalse(result.fallback_used)
            self.assertEqual(result.selected_image_path, result.edited_image_path)
            self.assertTrue(result.original_image_path.is_file())
            self.assertTrue(result.input_canvas_path.is_file())
            self.assertTrue(result.mask_path.is_file())
            self.assertTrue(result.edited_image_path.is_file())
            self.assertTrue(result.manifest_path.is_file())
            manifest = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'EDITED')
            self.assertEqual(manifest['provider'], 'fake-success')
            self.assertIn('sha256', manifest['artifacts']['original'])
            self.assertNotIn('OPENAI_API_KEY', result.manifest_path.read_text('utf-8'))
            self.assertEqual(len(provider.calls), 1)

    def test_pipeline_falls_back_without_exposing_provider_exception(self):
        secret = 'do-not-leak-this-key'
        options = PreprocessOptions(
            mode='always', artifact_type='figurine', canvas_size=(96, 128),
        )
        with patch.dict(os.environ, {'OPENAI_API_KEY': secret}, clear=False):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = PedestalImagePreprocessor(_FailingProvider()).process(
                    _figurine_image(), Path(temp_dir), options,
                )

                self.assertFalse(result.applied)
                self.assertTrue(result.fallback_used)
                self.assertEqual(
                    result.selected_image_path, result.original_image_path,
                )
                manifest_text = result.manifest_path.read_text(encoding='utf-8')
                manifest = json.loads(manifest_text)
                self.assertEqual(manifest['status'], 'FALLBACK')
                self.assertEqual(manifest['reason'], 'provider_error')
                self.assertNotIn(secret, manifest_text)
                self.assertNotIn('do-not-leak', manifest_text)

    def test_pipeline_rejects_opaque_background_and_falls_back(self):
        options = PreprocessOptions(
            mode='always', artifact_type='figurine', canvas_size=(96, 128),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = PedestalImagePreprocessor(_OpaqueProvider()).process(
                _figurine_image(), Path(temp_dir), options,
            )

            self.assertFalse(result.applied)
            self.assertTrue(result.fallback_used)
            self.assertEqual(result.reason, 'opaque_background')
            self.assertEqual(result.selected_image_path, result.original_image_path)
            self.assertIsNone(result.edited_image_path)
            self.assertFalse((Path(temp_dir) / 'pedestal.png').exists())

    def test_never_mode_does_not_build_mask_or_call_provider(self):
        options = PreprocessOptions(
            mode='never', artifact_type='figurine', canvas_size=(96, 128),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = PedestalImagePreprocessor(_NeverCalledProvider()).process(
                _figurine_image(), Path(temp_dir), options,
            )

            self.assertFalse(result.applied)
            self.assertFalse(result.fallback_used)
            self.assertIsNone(result.mask_path)
            self.assertIsNone(result.edited_image_path)
            self.assertEqual(result.selected_image_path, result.original_image_path)
            manifest = json.loads(result.manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'SKIPPED')
            self.assertEqual(manifest['reason'], 'mode_never')

    def test_cli_never_mode_is_offline_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / 'input.png'
            output_dir = temp_dir / 'result'
            _figurine_image().save(input_path)

            exit_code = preprocess_image.main([
                str(input_path),
                '--output-dir', str(output_dir),
                '--mode', 'never',
            ])

            self.assertEqual(exit_code, 0)
            manifest = json.loads(
                (output_dir / 'preprocess_manifest.json').read_text('utf-8')
            )
            self.assertEqual(manifest['status'], 'SKIPPED')
            self.assertEqual(manifest['reason'], 'mode_never')


if __name__ == '__main__':
    unittest.main()
