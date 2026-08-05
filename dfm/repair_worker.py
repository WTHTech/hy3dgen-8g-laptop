"""DFM 单阶段隔离修复进程。

该模块只由 :class:`dfm.repair.DFMRepairer` 通过 ``python -m`` 调用。
把可能阻塞的 pymeshlab、布尔并集和体素操作放入独立进程，父进程
可以在超时后真正终止计算，并继续使用阶段开始前的有效网格。
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import trimesh

from .repair import DFMRepairer, _mesh_ok
from .rules import DFMRules


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='运行一个隔离 DFM 修复阶段')
    parser.add_argument(
        '--stage', required=True,
        choices=('pedestal', 'fragment', 'light', 'medium', 'meshfix'),
    )
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--rules', required=True)
    parser.add_argument('--meshfix')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result_path = Path(args.result)
    try:
        rules_payload = json.loads(
            Path(args.rules).read_text(encoding='utf-8')
        )
        rules = DFMRules.from_dict(rules_payload)
        mesh = trimesh.load(args.input, force='mesh', process=False)
        if not _mesh_ok(mesh):
            raise ValueError('隔离修复输入不是有效三角网格')

        repairer = DFMRepairer(
            rules=rules,
            meshfix_path=args.meshfix,
            isolate_stages=False,
        )
        outcome = repairer._execute_stage_direct(args.stage, mesh)
        changed = bool(outcome.changed and _mesh_ok(outcome.mesh))
        if changed:
            outcome.mesh.export(args.output)
        payload = {
            'success': outcome.success,
            'changed': changed,
            'detail': outcome.detail,
            'actions': [action.to_dict() for action in outcome.actions],
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return 0
    except Exception as exc:
        result_path.write_text(
            json.dumps({
                'success': False,
                'changed': False,
                'detail': str(exc),
                'error_type': type(exc).__name__,
                'traceback': traceback.format_exc(),
                'actions': [],
            }, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
