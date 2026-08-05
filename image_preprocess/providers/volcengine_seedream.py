"""火山方舟 Seedream 5.0 Pro 图像编辑适配器。"""

import base64
import io
import os
from typing import Any, Optional
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from .base import ImageEditError, PedestalEditRequest


VOLCENGINE_ARK_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
SEEDREAM_5_PRO_MODEL = 'doubao-seedream-5-0-pro-260628'


def _validate_base_url(base_url: str) -> str:
    normalized = str(base_url).strip().rstrip('/')
    parsed = urlparse(normalized)
    trusted = (
        parsed.scheme == 'https'
        and parsed.hostname == 'ark.cn-beijing.volces.com'
        and parsed.path.rstrip('/') == '/api/v3'
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ''
        and parsed.fragment == ''
        and parsed.port in (None, 443)
    )
    if not trusted:
        raise ImageEditError('untrusted_base_url')
    return normalized


def _safe_http_error_code(status_code: int, payload: Any) -> str:
    provider_code = ''
    if isinstance(payload, dict):
        error = payload.get('error')
        if isinstance(error, dict):
            provider_code = str(error.get('code') or '').lower()
    if provider_code == 'modelnotopen':
        return 'model_not_open'
    return {
        400: 'invalid_request',
        401: 'authentication_error',
        403: 'permission_denied',
        404: 'model_or_endpoint_not_found',
        408: 'provider_timeout',
        409: 'provider_conflict',
        429: 'rate_limit_or_quota',
    }.get(
        status_code,
        'provider_unavailable' if status_code >= 500 else 'provider_error',
    )


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert('RGBA').save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


class VolcengineSeedreamProvider:
    """使用火山方舟官方图片生成 API 对单张参考图进行编辑。"""

    name = f'volcengine:{SEEDREAM_5_PRO_MODEL}'

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
        model: str = SEEDREAM_5_PRO_MODEL,
    ):
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds 必须大于 0')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model 不能为空')

        resolved_key = (
            api_key
            or os.environ.get('ARK_API_KEY')
            or os.environ.get('OPENAI_API_KEY')
        )
        if not resolved_key or not resolved_key.strip():
            raise ImageEditError('missing_api_key')
        resolved_url = (
            base_url
            or os.environ.get('ARK_BASE_URL')
            or os.environ.get('BASE_URL')
            or VOLCENGINE_ARK_BASE_URL
        )
        self.base_url = _validate_base_url(resolved_url)

        if client is None:
            import httpx

            client = httpx
        self._client = client
        self._api_key = resolved_key.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.model = model.strip()
        self.name = f'volcengine:{self.model}'

    def add_pedestal(
        self,
        image: Image.Image,
        mask: Image.Image,
        request: PedestalEditRequest,
    ) -> Image.Image:
        if not isinstance(request, PedestalEditRequest):
            raise TypeError('request 必须是 PedestalEditRequest')
        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise TypeError('image 和 mask 必须是 PIL.Image')
        if image.size != mask.size:
            raise ImageEditError('mask_size_mismatch')

        payload = {
            'model': self.model,
            'prompt': request.prompt,
            'image': _png_data_url(image),
            'size': request.size,
            'response_format': 'b64_json',
            'output_format': 'png',
            'watermark': False,
        }
        try:
            response = self._client.post(
                f'{self.base_url}/images/generations',
                headers={
                    'Authorization': f'Bearer {self._api_key}',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            try:
                response_payload = response.json()
            except Exception:
                response_payload = None
            status_code = int(getattr(response, 'status_code', 0))
            if status_code != 200:
                raise ImageEditError(
                    _safe_http_error_code(status_code, response_payload)
                )
            if not isinstance(response_payload, dict):
                raise ImageEditError('invalid_image_response')
            data = response_payload.get('data')
            encoded = (
                data[0].get('b64_json')
                if isinstance(data, list) and data and isinstance(data[0], dict)
                else None
            )
            if not encoded:
                raise ImageEditError('missing_image_data')
            if isinstance(encoded, str) and encoded.startswith('data:'):
                encoded = encoded.partition(',')[2]
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as opened:
                return opened.convert('RGBA').copy()
        except ImageEditError:
            raise
        except (ValueError, TypeError, UnidentifiedImageError):
            raise ImageEditError('invalid_image_response') from None
        except Exception as exc:
            class_name = type(exc).__name__
            reason = (
                'provider_timeout'
                if class_name in {'TimeoutException', 'ReadTimeout'}
                else 'provider_connection_error'
                if class_name in {'ConnectError', 'NetworkError'}
                else 'provider_error'
            )
            raise ImageEditError(reason) from None
