"""为 AI 底座编辑创建标准画布与底部蒙版。"""

from dataclasses import dataclass
from typing import Tuple

from PIL import Image, ImageDraw

from .analyzer import _subject_bbox


@dataclass(frozen=True)
class CanvasPlacement:
    subject_left_x: int
    subject_top_y: int
    subject_right_x: int
    subject_bottom_y: int
    edit_start_y: int


def build_edit_canvas(
    image: Image.Image,
    *,
    canvas_size: Tuple[int, int] = (1024, 1536),
    max_subject_width_ratio: float = 0.88,
    max_subject_height_ratio: float = 0.72,
    top_margin_ratio: float = 0.04,
    overlap_ratio: float = 0.05,
) -> tuple[Image.Image, Image.Image, CanvasPlacement]:
    """将主体放到透明画布，并把主体底部以下标为可编辑区域。"""
    if not isinstance(image, Image.Image):
        raise TypeError('image 必须是 PIL.Image')
    if (
        not isinstance(canvas_size, tuple)
        or len(canvas_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in canvas_size)
    ):
        raise ValueError('canvas_size 必须是两个正整数')
    for name, value in {
        'max_subject_width_ratio': max_subject_width_ratio,
        'max_subject_height_ratio': max_subject_height_ratio,
        'top_margin_ratio': top_margin_ratio,
        'overlap_ratio': overlap_ratio,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f'{name} 必须在 0 到 1 之间')

    rgba = image.convert('RGBA')
    bbox = _subject_bbox(rgba, 16)
    subject = rgba.crop(bbox)
    canvas_width, canvas_height = canvas_size
    scale = min(
        canvas_width * max_subject_width_ratio / subject.width,
        canvas_height * max_subject_height_ratio / subject.height,
    )
    target_size = (
        max(1, round(subject.width * scale)),
        max(1, round(subject.height * scale)),
    )
    subject = subject.resize(target_size, Image.Resampling.LANCZOS)

    left = (canvas_width - subject.width) // 2
    top = min(
        round(canvas_height * top_margin_ratio),
        max(0, canvas_height - subject.height),
    )
    right = left + subject.width
    bottom = top + subject.height
    overlap_pixels = max(2, round(subject.height * overlap_ratio))
    edit_start = max(top, bottom - overlap_pixels)

    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    canvas.alpha_composite(subject, (left, top))
    mask = Image.new('RGBA', canvas_size, (0, 0, 0, 255))
    ImageDraw.Draw(mask).rectangle(
        (0, edit_start, canvas_width, canvas_height),
        fill=(0, 0, 0, 0),
    )
    return canvas, mask, CanvasPlacement(
        subject_left_x=left,
        subject_top_y=top,
        subject_right_x=right,
        subject_bottom_y=bottom,
        edit_start_y=edit_start,
    )

