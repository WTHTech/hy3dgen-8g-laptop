"""图像编辑提供方的公共边界。"""

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class ImageEditError(RuntimeError):
    """不包含供应商原始响应或凭据的安全错误。"""

    def __init__(self, code: str):
        self.code = str(code)
        super().__init__(self.code)


@dataclass(frozen=True)
class PedestalEditRequest:
    prompt: str
    size: str = '1024x1536'
    quality: str = 'medium'


class ImageEditProvider(Protocol):
    name: str

    def add_pedestal(
        self,
        image: Image.Image,
        mask: Image.Image,
        request: PedestalEditRequest,
    ) -> Image.Image:
        ...

