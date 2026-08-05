"""OpenAI GPT Image 2 图像编辑适配器。"""

import base64
import io
import os
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError

from .base import ImageEditError, PedestalEditRequest


OPENAI_OFFICIAL_BASE_URL = 'https://api.openai.com/v1'


def _safe_provider_error_code(error: Exception) -> str:
    """仅按异常类型/状态码分类，不保留供应商原始错误文本。"""
    class_name = type(error).__name__
    if class_name == 'APITimeoutError':
        return 'provider_timeout'
    if class_name == 'APIConnectionError':
        return 'provider_connection_error'

    status_code = getattr(error, 'status_code', None)
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
        'provider_unavailable'
        if isinstance(status_code, int) and status_code >= 500
        else 'provider_error',
    )


class OpenAIImageEditProvider:
    name = 'openai:gpt-image-2'

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        model: str = 'gpt-image-2',
    ):
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds 必须大于 0')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model 不能为空')

        if client is None:
            resolved_key = api_key or os.environ.get('OPENAI_API_KEY')
            if not resolved_key or not resolved_key.strip():
                raise ImageEditError('missing_api_key')
            from openai import OpenAI

            # 本提供方的数据边界是 OpenAI 官方服务。不能隐式继承全局
            # OPENAI_BASE_URL，否则用户原图可能被发送到未授权的兼容端点。
            client = OpenAI(
                api_key=resolved_key,
                base_url=OPENAI_OFFICIAL_BASE_URL,
                max_retries=1,
            )
        self._client = client
        self.timeout_seconds = float(timeout_seconds)
        self.model = model
        self.name = f'openai:{model}'

    @staticmethod
    def _png_file(image: Image.Image, name: str) -> io.BytesIO:
        if not isinstance(image, Image.Image):
            raise TypeError('image 和 mask 必须是 PIL.Image')
        buffer = io.BytesIO()
        image.convert('RGBA').save(buffer, format='PNG')
        buffer.seek(0)
        buffer.name = name
        return buffer

    def add_pedestal(
        self,
        image: Image.Image,
        mask: Image.Image,
        request: PedestalEditRequest,
    ) -> Image.Image:
        if not isinstance(request, PedestalEditRequest):
            raise TypeError('request 必须是 PedestalEditRequest')
        if image.size != mask.size:
            raise ImageEditError('mask_size_mismatch')

        input_file = self._png_file(image, 'input.png')
        mask_file = self._png_file(mask, 'mask.png')
        try:
            response = self._client.images.edit(
                model=self.model,
                image=input_file,
                mask=mask_file,
                prompt=request.prompt,
                background='transparent',
                input_fidelity='high',
                output_format='png',
                response_format='b64_json',
                quality=request.quality,
                size=request.size,
                n=1,
                timeout=self.timeout_seconds,
            )
            data = getattr(response, 'data', None)
            encoded = getattr(data[0], 'b64_json', None) if data else None
            if not encoded:
                raise ImageEditError('missing_image_data')
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as opened:
                return opened.convert('RGBA').copy()
        except ImageEditError:
            raise
        except (ValueError, UnidentifiedImageError):
            raise ImageEditError('invalid_image_response') from None
        except Exception as exc:
            # 供应商异常可能包含请求头或敏感上下文，不能向上透传原文。
            raise ImageEditError(_safe_provider_error_code(exc)) from None
        finally:
            input_file.close()
            mask_file.close()
