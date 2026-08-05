"""火山方舟 Seedream 图像编辑提供方的离线测试。"""

import base64
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from image_preprocess.providers import (  # noqa: E402
    ImageEditError,
    PedestalEditRequest,
    VolcengineSeedreamProvider,
)
from scripts import preprocess_image  # noqa: E402


def _png_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _figurine_image(size=(64, 64)):
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 3, 52, 43), fill=(200, 100, 80, 255))
    draw.rectangle((18, 28, 46, 52), fill=(50, 100, 180, 255))
    draw.rectangle((24, 52, 28, 61), fill=(50, 50, 50, 255))
    draw.rectangle((36, 52, 40, 61), fill=(50, 50, 50, 255))
    return image


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHttpClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class _TransparentProvider:
    name = 'fake-seedream'

    def __init__(self):
        self.requests = []

    def add_pedestal(self, image, mask, request):
        self.requests.append(request)
        return image.copy()


class VolcengineSeedreamTests(unittest.TestCase):
    def test_sends_single_reference_image_and_decodes_png(self):
        edited = Image.new('RGBA', (32, 48), (10, 20, 30, 0))
        client = _FakeHttpClient(_FakeResponse(200, {
            'data': [{'b64_json': _png_b64(edited)}],
        }))
        provider = VolcengineSeedreamProvider(
            client=client,
            api_key='unit-test-key',
            timeout_seconds=45,
        )

        output = provider.add_pedestal(
            Image.new('RGBA', (32, 48)),
            Image.new('RGBA', (32, 48), (0, 0, 0, 255)),
            PedestalEditRequest(
                prompt='add a base', size='2K', quality='medium',
            ),
        )

        self.assertEqual(output.mode, 'RGBA')
        self.assertEqual(output.size, (32, 48))
        self.assertEqual(len(client.calls), 1)
        url, call = client.calls[0]
        self.assertEqual(
            url,
            'https://ark.cn-beijing.volces.com/api/v3/images/generations',
        )
        self.assertEqual(call['headers']['Authorization'], 'Bearer unit-test-key')
        self.assertEqual(call['timeout'], 45)
        payload = call['json']
        self.assertEqual(payload['model'], 'doubao-seedream-5-0-pro-260628')
        self.assertEqual(payload['prompt'], 'add a base')
        self.assertTrue(payload['image'].startswith('data:image/png;base64,'))
        self.assertEqual(payload['size'], '2K')
        self.assertEqual(payload['response_format'], 'b64_json')
        self.assertEqual(payload['output_format'], 'png')
        self.assertNotIn('sequential_image_generation', payload)
        self.assertNotIn('stream', payload)
        self.assertFalse(payload['watermark'])

    def test_maps_model_not_open_without_leaking_response_message(self):
        client = _FakeHttpClient(_FakeResponse(404, {
            'error': {
                'code': 'ModelNotOpen',
                'message': 'sensitive upstream response',
            },
        }))
        provider = VolcengineSeedreamProvider(
            client=client, api_key='unit-test-key',
        )

        with self.assertRaises(ImageEditError) as raised:
            provider.add_pedestal(
                Image.new('RGBA', (32, 48)),
                Image.new('RGBA', (32, 48)),
                PedestalEditRequest(prompt='add a base', size='2K'),
            )

        self.assertEqual(raised.exception.code, 'model_not_open')
        self.assertNotIn('sensitive', str(raised.exception))

    def test_removes_connected_opaque_background_and_restores_protected_region(self):
        source = Image.new('RGBA', (32, 48), (0, 0, 0, 0))
        ImageDraw.Draw(source).rectangle(
            (10, 4, 22, 30), fill=(200, 30, 30, 255),
        )
        mask = Image.new('RGBA', source.size, (0, 0, 0, 0))
        ImageDraw.Draw(mask).rectangle(
            (0, 0, 31, 31), fill=(255, 255, 255, 255),
        )
        generated = Image.new('RGBA', source.size, (250, 250, 250, 255))
        draw = ImageDraw.Draw(generated)
        draw.rectangle((10, 4, 22, 30), fill=(20, 20, 220, 255))
        draw.ellipse((5, 31, 27, 44), fill=(80, 80, 80, 255))
        client = _FakeHttpClient(_FakeResponse(200, {
            'data': [{'b64_json': _png_b64(generated)}],
        }))
        provider = VolcengineSeedreamProvider(
            client=client, api_key='unit-test-key',
        )

        output = provider.add_pedestal(
            source,
            mask,
            PedestalEditRequest(prompt='add a base', size='2K'),
        )

        self.assertEqual(output.getpixel((0, 0))[3], 0)
        self.assertEqual(output.getpixel((16, 16)), (200, 30, 30, 255))
        self.assertGreater(output.getpixel((16, 40))[3], 0)

    def test_requires_key_and_rejects_non_official_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ImageEditError, 'missing_api_key'):
                VolcengineSeedreamProvider(client=_FakeHttpClient())

        with self.assertRaisesRegex(ImageEditError, 'untrusted_base_url'):
            VolcengineSeedreamProvider(
                client=_FakeHttpClient(),
                api_key='unit-test-key',
                base_url='https://third-party.example/api/v3',
            )

    def test_cli_selects_seedream_and_uses_2k_request(self):
        provider = _TransparentProvider()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / 'input.png'
            _figurine_image().save(input_path)

            with patch(
                'scripts.preprocess_image.VolcengineSeedreamProvider',
                return_value=provider,
            ) as provider_factory:
                exit_code = preprocess_image.main([
                    str(input_path),
                    '--output-dir', str(temp_dir / 'output'),
                    '--mode', 'always',
                    '--provider', 'seedream',
                    '--model', 'doubao-seedream-5-0-pro-260628',
                ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(provider_factory.call_count, 1)
        self.assertEqual(
            provider_factory.call_args.kwargs['model'],
            'doubao-seedream-5-0-pro-260628',
        )
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].size, '2K')


if __name__ == '__main__':
    unittest.main()
