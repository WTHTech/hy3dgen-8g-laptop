# DFM (Design for Manufacturing) — 可制造性检查模块
#
# 用法：
#   from dfm import DFMChecker, DFMRules
#
#   rules = DFMRules('dfm_config.yaml')
#   checker = DFMChecker(rules)
#
#   # 粗筛（10 项；自相交耗时随面数增长）
#   report = checker.check_quick(mesh)
#
#   # 精检（13 项候选；未实现项返回 UNKNOWN）
#   report = checker.check_detailed(mesh)
#
#   # 全量 (粗筛阻断则跳过精检)
#   report = checker.check_full(mesh, target_height=100)

from .rules import DFMRules
from .checker import CheckResult, CheckStatus, DFMChecker, DFMReport

__all__ = ['DFMRules', 'DFMChecker', 'CheckResult', 'CheckStatus', 'DFMReport']
