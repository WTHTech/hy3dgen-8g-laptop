"""修复候选网格的事务性提交策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .checker import CheckStatus, DFMReport


@dataclass(frozen=True)
class RepairCandidateDecision:
    """候选修复相对阶段前报告的验收结论。"""

    accepted: bool
    detail: str
    previous_failures: frozenset[str]
    candidate_failures: frozenset[str]
    new_failures: frozenset[str]
    resolved_failures: frozenset[str]
    new_unknown: frozenset[str]
    previous_score: float
    candidate_score: float


def _codes(report: DFMReport, status: CheckStatus) -> frozenset[str]:
    return frozenset(
        item.code for item in report.results if item.status == status
    )


def evaluate_repair_candidate(
    previous: DFMReport,
    candidate: DFMReport,
    *,
    score_tolerance: float = 1e-6,
) -> RepairCandidateDecision:
    """决定候选网格能否覆盖阶段前检查点。

    安全门要求候选不增加 FAIL 数量、不降低快速层分数，也不引入新的
    UNKNOWN。允许在失败数量不增加且分数不下降时用一种缺陷替换另一种，
    以保留“轻量修复暴露可继续处理中度缺陷”的既有升级能力。
    """
    previous_failures = _codes(previous, CheckStatus.FAIL)
    candidate_failures = _codes(candidate, CheckStatus.FAIL)
    previous_unknown = _codes(previous, CheckStatus.UNKNOWN)
    candidate_unknown = _codes(candidate, CheckStatus.UNKNOWN)
    new_failures = candidate_failures - previous_failures
    resolved_failures = previous_failures - candidate_failures
    new_unknown = candidate_unknown - previous_unknown
    previous_score = float(previous.total_score)
    candidate_score = float(candidate.total_score)

    reasons: list[str] = []
    if candidate.prepared_mesh is None:
        reasons.append('候选报告没有 prepared_mesh')
    if not math.isfinite(candidate_score):
        reasons.append('候选分数不是有限数')
    if len(candidate_failures) > len(previous_failures):
        reasons.append(
            f'FAIL 数量 {len(previous_failures)}→{len(candidate_failures)}'
        )
    if candidate_score < previous_score - score_tolerance:
        reasons.append(f'分数 {previous_score:.3f}→{candidate_score:.3f}')
    if new_unknown:
        reasons.append(f'新增 UNKNOWN [{", ".join(sorted(new_unknown))}]')

    accepted = not reasons
    if accepted:
        detail = (
            f'事务性验收通过：FAIL {len(previous_failures)}→'
            f'{len(candidate_failures)}，分数 {previous_score:.3f}→'
            f'{candidate_score:.3f}'
        )
    else:
        detail = (
            '事务性验收拒绝候选并回滚：' + '；'.join(reasons)
            + f'；新增 FAIL [{", ".join(sorted(new_failures)) or "无"}]'
            + f'；已解决 FAIL [{", ".join(sorted(resolved_failures)) or "无"}]'
        )

    return RepairCandidateDecision(
        accepted=accepted,
        detail=detail,
        previous_failures=previous_failures,
        candidate_failures=candidate_failures,
        new_failures=new_failures,
        resolved_failures=resolved_failures,
        new_unknown=new_unknown,
        previous_score=previous_score,
        candidate_score=candidate_score,
    )

