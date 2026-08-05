# DFM (Design for Manufacturing) — 可制造性检查模块
#
# 用法：
#   from dfm import DFMChecker, DFMRules, DFMRepairer
#
#   rules = DFMRules('dfm_config.yaml')
#   checker = DFMChecker(rules)
#
#   # 粗筛（10 项；自相交耗时随面数增长）
#   report = checker.check_quick(mesh, input_units='normalized')
#
#   # 精检（13 项候选；未实现项返回 UNKNOWN）
#   report = checker.check_detailed(mesh)
#
#   # 全量 (粗筛阻断则跳过精检)
#   report = checker.check_full(
#       mesh, target_height=100, input_units='normalized'
#   )
#
#   # 修复闭环
#   repairer = DFMRepairer()
#   result, final_report = repairer.repair_and_recheck(
#       mesh, checker, input_units='normalized'
#   )

#   # 切片仿真
#   from dfm import CuraSlicer
#   slicer = CuraSlicer()
#   result = slicer.slice_prepared(report, 'output/test.gcode')
#   print(f'打印耗时: {result.print_time_min:.0f} min, 耗材: {result.filament_g:.1f}g')

from .rules import DFMRules
from .checker import CheckResult, CheckStatus, DFMChecker, DFMReport
from .fragment_cleaner import FragmentCleanResult, FragmentCleaner
from .pedestal import PedestalGenerator, PedestalResult
from .repair import (
    DFMRepairer,
    RepairAction,
    RepairLevel,
    RepairResult,
)
from .slicer import CuraSlicer, SliceResult

__all__ = [
    'DFMRules', 'DFMChecker', 'DFMRepairer',
    'CheckResult', 'CheckStatus', 'DFMReport',
    'FragmentCleaner', 'FragmentCleanResult',
    'PedestalGenerator', 'PedestalResult',
    'RepairAction', 'RepairLevel', 'RepairResult',
    'CuraSlicer', 'SliceResult',
]
