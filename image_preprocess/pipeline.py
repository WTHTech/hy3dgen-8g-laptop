"""AI 底座图像预处理编排、落盘与安全回退。"""

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image

from .analyzer import analyze_image, should_add_pedestal
from .mask import build_edit_canvas
from .prompts import pedestal_prompt
from .providers import ImageEditError, ImageEditProvider, PedestalEditRequest


ImageInput = Union[str, os.PathLike, Image.Image]


@dataclass(frozen=True)
class PreprocessOptions:
    mode: str = 'auto'
    artifact_type: str = 'figurine'
    canvas_size: Tuple[int, int] = (1024, 1536)
    api_size: str = '1024x1536'
    quality: str = 'medium'
    contact_threshold: float = 0.25

    def __post_init__(self):
        if self.mode not in {'auto', 'always', 'never'}:
            raise ValueError('mode 必须是 auto、always 或 never')
        if self.artifact_type not in {'figurine', 'pendant', 'other'}:
            raise ValueError('artifact_type 必须是 figurine、pendant 或 other')
        if self.quality not in {'low', 'medium', 'high', 'auto'}:
            raise ValueError('quality 必须是 low、medium、high 或 auto')
        if not 0 < self.contact_threshold <= 1:
            raise ValueError('contact_threshold 必须在 0 到 1 之间')


@dataclass(frozen=True)
class PreprocessResult:
    applied: bool
    fallback_used: bool
    reason: str
    selected_image_path: Path
    original_image_path: Path
    input_canvas_path: Optional[Path]
    mask_path: Optional[Path]
    edited_image_path: Optional[Path]
    manifest_path: Path


def _load_rgba(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGBA').copy()
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            return opened.convert('RGBA').copy()
    raise TypeError('image 必须是图片路径或 PIL.Image')


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f'.{path.stem}.{uuid.uuid4().hex}.tmp'
    try:
        image.convert('RGBA').save(temp, format='PNG')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f'.{path.name}.{uuid.uuid4().hex}.tmp'
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    return {
        'path': str(path.resolve()),
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }


class PedestalImagePreprocessor:
    def __init__(self, provider: ImageEditProvider):
        if provider is None or not hasattr(provider, 'add_pedestal'):
            raise TypeError('provider 必须实现 add_pedestal')
        self.provider = provider

    def process(
        self,
        image: ImageInput,
        output_dir: Union[str, os.PathLike],
        options: Optional[PreprocessOptions] = None,
    ) -> PreprocessResult:
        options = options or PreprocessOptions()
        if not isinstance(options, PreprocessOptions):
            raise TypeError('options 必须是 PreprocessOptions')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        original = _load_rgba(image)
        analysis = analyze_image(original)
        original_path = output_dir / 'original.png'
        manifest_path = output_dir / 'preprocess_manifest.json'
        _atomic_save_png(original, original_path)

        requested = should_add_pedestal(
            options.mode,
            options.artifact_type,
            analysis,
            contact_threshold=options.contact_threshold,
        )
        if not requested:
            reason = (
                'mode_never' if options.mode == 'never'
                else 'pedestal_not_required'
            )
            result = PreprocessResult(
                applied=False,
                fallback_used=False,
                reason=reason,
                selected_image_path=original_path,
                original_image_path=original_path,
                input_canvas_path=None,
                mask_path=None,
                edited_image_path=None,
                manifest_path=manifest_path,
            )
            self._save_manifest(result, analysis.to_dict(), options, {})
            return result

        canvas, mask, placement = build_edit_canvas(
            original, canvas_size=options.canvas_size,
        )
        canvas_path = output_dir / 'input_canvas.png'
        mask_path = output_dir / 'preprocess_mask.png'
        edited_path = output_dir / 'pedestal.png'
        _atomic_save_png(canvas, canvas_path)
        _atomic_save_png(mask, mask_path)
        request = PedestalEditRequest(
            prompt=pedestal_prompt(),
            size=options.api_size,
            quality=options.quality,
        )

        try:
            edited = self.provider.add_pedestal(canvas, mask, request)
            if not isinstance(edited, Image.Image):
                raise ImageEditError('invalid_image_response')
            edited = edited.convert('RGBA')
            alpha_min, _ = edited.getchannel('A').getextrema()
            if alpha_min == 255:
                raise ImageEditError('opaque_background')
            edited_analysis = analyze_image(edited)
            if edited_analysis.subject_area_ratio < 0.01:
                raise ImageEditError('subject_missing')
            _atomic_save_png(edited, edited_path)
            result = PreprocessResult(
                applied=True,
                fallback_used=False,
                reason='edited',
                selected_image_path=edited_path,
                original_image_path=original_path,
                input_canvas_path=canvas_path,
                mask_path=mask_path,
                edited_image_path=edited_path,
                manifest_path=manifest_path,
            )
            extra = {
                'placement': asdict(placement),
                'edited_analysis': edited_analysis.to_dict(),
                'request': {
                    'size': request.size,
                    'quality': request.quality,
                    'prompt_sha256': hashlib.sha256(
                        request.prompt.encode('utf-8')
                    ).hexdigest(),
                },
            }
            self._save_manifest(result, analysis.to_dict(), options, extra)
            return result
        except ImageEditError as exc:
            reason = exc.code
        except Exception:
            reason = 'provider_error'

        result = PreprocessResult(
            applied=False,
            fallback_used=True,
            reason=reason,
            selected_image_path=original_path,
            original_image_path=original_path,
            input_canvas_path=canvas_path,
            mask_path=mask_path,
            edited_image_path=None,
            manifest_path=manifest_path,
        )
        self._save_manifest(
            result,
            analysis.to_dict(),
            options,
            {'placement': asdict(placement)},
        )
        return result

    def _save_manifest(
        self,
        result: PreprocessResult,
        analysis: dict,
        options: PreprocessOptions,
        extra: dict,
    ) -> None:
        status = (
            'EDITED' if result.applied
            else 'FALLBACK' if result.fallback_used
            else 'SKIPPED'
        )
        payload = {
            'schema_version': 1,
            'status': status,
            'reason': result.reason,
            'provider': getattr(self.provider, 'name', type(self.provider).__name__),
            'options': asdict(options),
            'analysis': analysis,
            'artifacts': {
                'original': _artifact(result.original_image_path),
                'input_canvas': _artifact(result.input_canvas_path),
                'mask': _artifact(result.mask_path),
                'edited': _artifact(result.edited_image_path),
                'selected': _artifact(result.selected_image_path),
            },
        }
        payload.update(extra)
        _atomic_save_json(payload, result.manifest_path)
