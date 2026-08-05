"""图生 3D 前的轻量图片分析。"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from PIL import Image


BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageAnalysis:
    width: int
    height: int
    subject_bbox: BoundingBox
    subject_area_ratio: float
    bottom_contact_ratio: float
    bottom_margin_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _subject_bbox(image: Image.Image, alpha_threshold: int) -> BoundingBox:
    if not 0 <= alpha_threshold <= 255:
        raise ValueError('alpha_threshold 必须在 0 到 255 之间')
    rgba = image.convert('RGBA')
    alpha = rgba.getchannel('A').point(
        lambda value: 255 if value > alpha_threshold else 0
    )
    bbox: Optional[BoundingBox] = alpha.getbbox()
    if bbox is None:
        raise ValueError('图片没有可见主体')
    return bbox


def analyze_image(
    image: Image.Image,
    *,
    alpha_threshold: int = 16,
    bottom_band_ratio: float = 0.04,
) -> ImageAnalysis:
    """分析透明主体的边界及底部横向接触比例。"""
    if not isinstance(image, Image.Image):
        raise TypeError('image 必须是 PIL.Image')
    if not 0 < bottom_band_ratio <= 0.25:
        raise ValueError('bottom_band_ratio 必须在 0 到 0.25 之间')

    rgba = image.convert('RGBA')
    left, top, right, bottom = _subject_bbox(rgba, alpha_threshold)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width <= 0 or subject_height <= 0:
        raise ValueError('主体边界无效')

    alpha = rgba.getchannel('A')
    band_height = max(1, round(subject_height * bottom_band_ratio))
    band_top = max(top, bottom - band_height)
    contact_columns = 0
    for x in range(left, right):
        if any(
            alpha.getpixel((x, y)) > alpha_threshold
            for y in range(band_top, bottom)
        ):
            contact_columns += 1

    histogram = alpha.histogram()
    visible_pixels = sum(histogram[alpha_threshold + 1:])
    return ImageAnalysis(
        width=rgba.width,
        height=rgba.height,
        subject_bbox=(left, top, right, bottom),
        subject_area_ratio=visible_pixels / float(rgba.width * rgba.height),
        bottom_contact_ratio=contact_columns / float(subject_width),
        bottom_margin_ratio=(rgba.height - bottom) / float(rgba.height),
    )


def should_add_pedestal(
    mode: str,
    artifact_type: str,
    analysis: ImageAnalysis,
    *,
    contact_threshold: float = 0.25,
) -> bool:
    """按用户模式、作品类型和底部接触风险决定是否调用编辑服务。"""
    if mode not in {'auto', 'always', 'never'}:
        raise ValueError('mode 必须是 auto、always 或 never')
    if artifact_type not in {'figurine', 'pendant', 'other'}:
        raise ValueError('artifact_type 必须是 figurine、pendant 或 other')
    if not 0 < contact_threshold <= 1:
        raise ValueError('contact_threshold 必须在 0 到 1 之间')
    if not isinstance(analysis, ImageAnalysis):
        raise TypeError('analysis 必须是 ImageAnalysis')

    if mode == 'never':
        return False
    if mode == 'always':
        return True
    return (
        artifact_type == 'figurine'
        and analysis.bottom_contact_ratio < contact_threshold
    )
