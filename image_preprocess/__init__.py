"""可制造性感知的图像预处理。"""

from .analyzer import ImageAnalysis, analyze_image, should_add_pedestal
from .mask import CanvasPlacement, build_edit_canvas
from .pipeline import (
    PedestalImagePreprocessor,
    PreprocessOptions,
    PreprocessResult,
)

__all__ = [
    'CanvasPlacement',
    'ImageAnalysis',
    'PedestalImagePreprocessor',
    'PreprocessOptions',
    'PreprocessResult',
    'analyze_image',
    'build_edit_canvas',
    'should_add_pedestal',
]
