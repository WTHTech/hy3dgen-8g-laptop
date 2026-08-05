"""在 Hunyuan3D 前为需要落地的手办图片生成 AI 底座。"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_preprocess import PedestalImagePreprocessor, PreprocessOptions
from image_preprocess.providers import (
    ImageEditError,
    OpenAIImageEditProvider,
    VolcengineSeedreamProvider,
)


class _DisabledProvider:
    name = 'disabled'

    def add_pedestal(self, image, mask, request):
        raise AssertionError('禁用模式不应调用图像编辑提供方')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='使用 AI 在图片底部生成可制造性底座，不执行图生 3D。',
    )
    parser.add_argument('image', type=Path, help='输入 PNG/JPG 图片')
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='保存原图、蒙版、派生图和 manifest 的目录',
    )
    parser.add_argument(
        '--mode', choices=('auto', 'always', 'never'), default='auto',
    )
    parser.add_argument(
        '--artifact-type',
        choices=('figurine', 'pendant', 'other'),
        default='figurine',
    )
    parser.add_argument(
        '--provider', choices=('openai', 'seedream'), default='openai',
    )
    parser.add_argument(
        '--quality', choices=('low', 'medium', 'high', 'auto'),
        default='medium',
    )
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--model', default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.is_file():
        raise FileNotFoundError(f'输入图片不存在: {args.image}')
    if args.timeout <= 0:
        raise ValueError('--timeout 必须大于 0')

    try:
        if args.mode == 'never':
            provider = _DisabledProvider()
        elif args.provider == 'seedream':
            provider_kwargs = {'timeout_seconds': args.timeout}
            if args.model:
                provider_kwargs['model'] = args.model
            provider = VolcengineSeedreamProvider(**provider_kwargs)
        else:
            provider_kwargs = {'timeout_seconds': args.timeout}
            if args.model:
                provider_kwargs['model'] = args.model
            provider = OpenAIImageEditProvider(**provider_kwargs)
    except ImageEditError as exc:
        print(
            json.dumps({'status': 'ERROR', 'reason': exc.code}),
            file=sys.stderr,
        )
        return 2

    result = PedestalImagePreprocessor(provider).process(
        args.image,
        args.output_dir,
        PreprocessOptions(
            mode=args.mode,
            artifact_type=args.artifact_type,
            quality=args.quality,
            api_size='2K' if args.provider == 'seedream' else '1024x1536',
        ),
    )
    print(json.dumps({
        'status': (
            'EDITED' if result.applied
            else 'FALLBACK' if result.fallback_used
            else 'SKIPPED'
        ),
        'reason': result.reason,
        'selected_image_path': str(result.selected_image_path.resolve()),
        'manifest_path': str(result.manifest_path.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
