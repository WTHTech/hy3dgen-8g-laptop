"""CuraEngine 5.x 切片仿真封装。

本模块默认生成用于 DFM 验证的仿真 G-code，不把通用配置生成的文件宣称为
可直接上机文件。业务入口应优先使用 :meth:`CuraSlicer.slice_prepared`，确保
传入的是已经统一为毫米并完成 DFM 检查/修复的 ``prepared_mesh``。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .checker import DFMReport
from .rules import DFMRules


# Cura 5.x CLI 的通用 FDM 仿真设置。打印机、材料和质量配置最终应由服务器上的
# 固定 profile 替换；这里仅用于几何可切片性、层数、时间和耗材估算。
DEFAULT_SLICE_SETTINGS: dict[str, object] = {
    'machine_name': 'DFM_Generic_FDM_Simulation',
    'machine_width': 220,
    'machine_depth': 220,
    'machine_height': 250,
    'machine_center_is_zero': True,
    'machine_extruder_count': 1,
    'machine_heated_bed': False,
    'machine_nozzle_size': 0.4,
    # 仿真文件不携带可直接上机的启停、归零和加热命令。
    'machine_start_gcode': '',
    'machine_end_gcode': '',
    'material_print_temp_prepend': False,
    'material_print_temp_wait': False,
    'material_bed_temp_prepend': False,
    'material_bed_temp_wait': False,
    # CLI 模式下把模型放到以平台中心为原点的坐标系。
    'center_object': True,
    'mesh_position_x': 0,
    'mesh_position_y': 0,
    # 几何与填充。
    'layer_height': 0.2,
    'layer_height_0': 0.2,
    'wall_line_count': 3,
    'top_layers': 6,
    'bottom_layers': 6,
    'infill_sparse_density': 20,
    'adhesion_type': 'skirt',
    'skirt_line_count': 3,
    'support_enable': False,
    'support_type': 'buildplate',
    # PLA 参考值。温度命令在仿真模式关闭，但这些值仍用于设置解析。
    'default_material_print_temperature': 200,
    'material_print_temperature': 200,
    'material_print_temperature_layer_0': 200,
    'material_initial_print_temperature': 200,
    'material_final_print_temperature': 200,
    'default_material_bed_temperature': 60,
    'material_bed_temperature': 60,
    'material_bed_temperature_layer_0': 60,
    'material_diameter': 1.75,
    'material_flow': 100,
    # 速度与冷却。
    'speed_print': 50,
    'speed_wall': 25,
    'speed_topbottom': 30,
    'speed_travel': 150,
    'cool_fan_enabled': True,
    'cool_fan_speed_min': 100,
    # Cura 5.13 CLI 不能从通用定义解析出的必需项。
    'roofing_layer_count': 0,
    'flooring_layer_count': 0,
    'support_z_seam_away_from_model': True,
    'support_z_seam_min_distance': 1.0,
    'lightning_infill_support_angle': 40,
    'scarf_joint_seam_end_height_ratio': 0,
    'reset_flow_duration': 2.0,
}

_EXTRUDER_PREFIXES = (
    'material_', 'default_material_', 'speed_', 'cool_', 'retraction_',
    # Cura 的 fdmextruder 定义把喷嘴属性、偏移以及挤出机启停位置等
    # machine_* 设置放在挤出机栈。使用稳定前缀可覆盖同类新增设置，
    # 避免维护一个容易遗漏的逐项白名单。
    'machine_extruder_', 'machine_nozzle_',
)
_NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_AXIS_RE = re.compile(rf'(?:^|\s)([XYZE])({_NUMBER})', re.IGNORECASE)


@dataclass
class SliceResult:
    """一次 CuraEngine 切片仿真的结构化结果。"""

    success: bool = False
    gcode_path: Optional[str] = None
    gcode_size_bytes: int = 0
    print_time_min: float = 0.0
    filament_mm: float = 0.0
    filament_g: float = 0.0
    layer_count: int = 0
    layer_height_mm: float = 0.0
    toolpath_bounds_mm: Optional[list[list[float]]] = None
    extrusion_move_count: int = 0
    engine_version: str = ''
    profile_name: str = 'DFM_Generic_FDM_Simulation'
    simulation_only: bool = True
    error: str = ''
    stderr: str = ''
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'gcode_path': self.gcode_path,
            'gcode_size_bytes': self.gcode_size_bytes,
            'print_time_min': self.print_time_min,
            'filament_mm': self.filament_mm,
            'filament_g': self.filament_g,
            'layer_count': self.layer_count,
            'layer_height_mm': self.layer_height_mm,
            'toolpath_bounds_mm': self.toolpath_bounds_mm,
            'extrusion_move_count': self.extrusion_move_count,
            'engine_version': self.engine_version,
            'profile_name': self.profile_name,
            'simulation_only': self.simulation_only,
            'error': self.error,
            'warnings': list(self.warnings),
        }


class CuraSlicer:
    """CuraEngine 5.x CLI 的单挤出机 FDM 仿真封装。

    路径发现顺序：显式参数 → ``CURA_ENGINE_PATH`` / 
    ``CURA_ENGINE_RESOURCES`` → 项目 ``.cache`` → 系统 ``PATH``。
    """

    def __init__(
        self,
        engine_path: Optional[str | Path] = None,
        def_path: Optional[str | Path] = None,
        extruder_def_path: Optional[str | Path] = None,
        resources_path: Optional[str | Path] = None,
        rules: Optional[DFMRules] = None,
        material_density_g_cm3: Optional[float] = None,
    ):
        self.rules = rules or DFMRules()
        density = (
            material_density_g_cm3
            if material_density_g_cm3 is not None
            else self.rules.get('filament_density_g_cm3', 1.24)
        )
        if isinstance(density, bool) or not isinstance(density, (int, float)) or density <= 0:
            raise ValueError('material_density_g_cm3 必须是正数')
        self.material_density_g_cm3 = float(density)

        self._engine_path = self._normalise_optional_path(engine_path) or self._find_engine()
        self._resources_path = (
            self._normalise_optional_path(resources_path)
            or self._find_resources(self._engine_path)
        )

        default_def = None
        default_extruder_def = None
        if self._resources_path:
            definitions = Path(self._resources_path) / 'definitions'
            default_def = definitions / 'fdmprinter.def.json'
            default_extruder_def = definitions / 'fdmextruder.def.json'

        self._def_path = self._normalise_optional_path(def_path or default_def)
        self._extruder_def_path = self._normalise_optional_path(
            extruder_def_path or default_extruder_def
        )
        self._engine_version: Optional[str] = None

    def slice_prepared(
        self,
        report: DFMReport,
        output_path: Optional[str | Path] = None,
        settings: Optional[dict[str, object]] = None,
        extruder_settings: Optional[dict[str, object]] = None,
        timeout: Optional[int] = None,
        *,
        allow_failed: bool = False,
    ) -> SliceResult:
        """切片检查报告中的毫米 ``prepared_mesh``。

        默认只接受 ``PASS`` 报告。诊断失败模型时必须显式传
        ``allow_failed=True``，避免原始或未通过 DFM 的网格误入导出链。
        """
        if not isinstance(report, DFMReport) or report.prepared_mesh is None:
            return SliceResult(error='DFM 报告没有可用的 prepared_mesh')
        if report.status != 'PASS' and not allow_failed:
            return SliceResult(
                error=f'DFM 状态为 {report.status}，未进入切片；诊断时可显式 allow_failed=True'
            )
        return self.slice(
            report.prepared_mesh,
            output_path,
            settings,
            extruder_settings,
            timeout,
            input_units='mm',
        )

    def slice(
        self,
        mesh: trimesh.Trimesh | str | Path,
        output_path: Optional[str | Path] = None,
        settings: Optional[dict[str, object]] = None,
        extruder_settings: Optional[dict[str, object]] = None,
        timeout: Optional[int] = None,
        *,
        input_units: str = 'mm',
    ) -> SliceResult:
        """低层切片入口；输入必须已经是毫米 STL/Trimesh。"""
        if self.rules.process != 'FDM':
            return SliceResult(error='当前 CuraEngine 封装仅支持 FDM；SLA 需要独立切片器')
        if input_units != 'mm':
            return SliceResult(
                error='CuraEngine 只接受毫米输入；normalized/auto 网格必须先通过 DFM prepared_mesh'
            )
        if not self.available:
            return SliceResult(error=self.availability_error)

        actual_timeout = timeout
        if actual_timeout is None:
            actual_timeout = self.rules.get('cura_timeout_seconds', 120)
        if (
            not isinstance(actual_timeout, int)
            or isinstance(actual_timeout, bool)
            or actual_timeout < 1
        ):
            return SliceResult(error='timeout 必须是大于等于 1 的整数秒')

        try:
            merged = dict(DEFAULT_SLICE_SETTINGS)
            merged.update(self._settings_from_rules())
            merged.update(_validate_settings(settings, 'settings'))
            explicit_extruder = _validate_settings(
                extruder_settings, 'extruder_settings'
            )
            filament_diameter = float(merged.get('material_diameter', 1.75))
            if filament_diameter <= 0:
                return SliceResult(error='material_diameter 必须是正数')
        except (TypeError, ValueError) as exc:
            return SliceResult(error=f'切片设置无效: {exc}')

        stl_path: Optional[Path] = None
        temporary_stl: Optional[Path] = None
        working_output: Optional[Path] = None
        final_output: Optional[Path] = None

        try:
            if isinstance(mesh, (str, Path)):
                stl_path = Path(mesh).expanduser().resolve()
                if not stl_path.is_file():
                    raise ValueError(f'STL 文件不存在: {stl_path}')
                if stl_path.suffix.lower() != '.stl':
                    raise ValueError('CuraEngine CLI 路径输入当前只接受 .stl')
            elif isinstance(mesh, trimesh.Trimesh):
                if not _mesh_ok(mesh):
                    raise ValueError('网格为空、包含非有限坐标或没有有效三角面')
                fd, temp_name = tempfile.mkstemp(suffix='.stl', prefix='dfm_slice_')
                os.close(fd)
                temporary_stl = Path(temp_name)
                mesh.export(temporary_stl)
                stl_path = temporary_stl
            else:
                raise TypeError(f'不支持的输入类型: {type(mesh).__name__}')

            if output_path is None:
                fd, output_name = tempfile.mkstemp(
                    suffix='.gcode', prefix='dfm_slice_out_'
                )
                os.close(fd)
                working_output = Path(output_name)
                final_output = working_output
            else:
                final_output = Path(output_path).expanduser().resolve()
                if final_output.exists() and final_output.is_dir():
                    raise ValueError(f'G-code 输出路径是目录: {final_output}')
                final_output.parent.mkdir(parents=True, exist_ok=True)
                fd, output_name = tempfile.mkstemp(
                    suffix='.gcode.part',
                    prefix=f'.{final_output.stem}_',
                    dir=final_output.parent,
                )
                os.close(fd)
                working_output = Path(output_name)

            # 要求 CuraEngine 从不存在的路径创建输出，避免把旧文件误判为成功。
            working_output.unlink(missing_ok=True)
            cmd = self._build_command(
                stl_path,
                working_output,
                merged,
                explicit_extruder,
            )
            env = os.environ.copy()
            search_path = os.pathsep.join(str(path) for path in self.definition_search_paths)
            if search_path:
                env['CURA_ENGINE_SEARCH_PATH'] = search_path

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=actual_timeout,
                encoding='utf-8',
                errors='replace',
                cwd=str(Path(self._engine_path).parent),
                env=env,
            )
            stderr = proc.stderr or ''
            stdout = proc.stdout or ''
            engine_errors = _engine_error_lines(stderr + '\n' + stdout)
            if proc.returncode != 0 or engine_errors or not working_output.is_file():
                detail = engine_errors[-1] if engine_errors else '未生成新的 G-code'
                return SliceResult(
                    error=f'CuraEngine 返回码 {proc.returncode}: {detail[-300:]}',
                    stderr=stderr[-4000:],
                    engine_version=self.engine_version,
                )

            gcode_size = working_output.stat().st_size
            if gcode_size == 0:
                return SliceResult(
                    error='CuraEngine 生成了空 G-code 文件',
                    stderr=stderr[-4000:],
                    engine_version=self.engine_version,
                )

            stats = _parse_gcode(
                working_output,
                filament_diameter_mm=filament_diameter,
                material_density_g_cm3=self.material_density_g_cm3,
            )
            validation_error = self._validate_stats(stats, merged)
            if validation_error:
                return SliceResult(
                    error=validation_error,
                    stderr=stderr[-4000:],
                    warnings=stats['warnings'],
                    engine_version=self.engine_version,
                )

            if output_path is not None:
                os.replace(working_output, final_output)
            # 成功输出由调用方持有，不在 finally 中清理。
            working_output = None

            return SliceResult(
                success=True,
                gcode_path=str(final_output),
                gcode_size_bytes=gcode_size,
                print_time_min=stats['print_time_min'],
                filament_mm=stats['filament_mm'],
                filament_g=stats['filament_g'],
                layer_count=stats['layer_count'],
                layer_height_mm=stats['layer_height_mm'],
                toolpath_bounds_mm=stats['toolpath_bounds_mm'],
                extrusion_move_count=stats['extrusion_move_count'],
                engine_version=self.engine_version,
                stderr=stderr[-4000:] if stderr else '',
                warnings=stats['warnings'],
            )
        except subprocess.TimeoutExpired:
            return SliceResult(
                error=f'CuraEngine 执行超时 ({actual_timeout}s)',
                engine_version=self.engine_version,
            )
        except Exception as exc:
            return SliceResult(
                error=f'切片异常: {exc}',
                engine_version=self.engine_version,
            )
        finally:
            _unlink_quietly(temporary_stl)
            _unlink_quietly(working_output)

    def _build_command(
        self,
        stl_path: Path,
        output_path: Path,
        settings: dict[str, object],
        explicit_extruder: dict[str, object],
    ) -> list[str]:
        search_path = os.pathsep.join(str(path) for path in self.definition_search_paths)
        cmd = [str(self._engine_path), 'slice']
        if search_path:
            cmd.extend(['-d', search_path])
        cmd.extend(['-j', str(self._def_path)])

        # 全局栈先接收全部值；Cura 会忽略不属于该栈的 per-extruder 值。
        for key, value in settings.items():
            cmd.extend(['-s', f'{key}={_format_setting_value(value)}'])

        # 单挤出机栈必须显式创建，并重新写入 per-extruder 值。
        cmd.extend(['-e0', '-j', str(self._extruder_def_path)])
        auto_extruder = {
            key: value for key, value in settings.items()
            if _is_extruder_setting(key)
        }
        auto_extruder.update(explicit_extruder)
        for key, value in auto_extruder.items():
            cmd.extend(['-s', f'{key}={_format_setting_value(value)}'])

        cmd.extend(['-l', str(stl_path), '-o', str(output_path)])
        return cmd

    def _settings_from_rules(self) -> dict[str, object]:
        build_vol = self.rules.get('build_volume', [220, 220, 250])
        nozzle = float(self.rules.get('nozzle_diameter', 0.4))
        layer_h = float(self.rules.get('layer_height', 0.2))
        min_wall = float(self.rules.get('min_wall_thickness', nozzle * 3))
        return {
            'machine_width': build_vol[0],
            'machine_depth': build_vol[1],
            'machine_height': build_vol[2],
            'machine_nozzle_size': nozzle,
            'layer_height': layer_h,
            'layer_height_0': layer_h,
            'wall_line_count': max(1, int(np.ceil(min_wall / nozzle))),
        }

    def _validate_stats(
        self,
        stats: dict,
        settings: dict[str, object],
    ) -> str:
        if stats['layer_count'] < 1:
            return 'G-code 没有有效打印层'
        if stats['extrusion_move_count'] < 1 or stats['filament_mm'] <= 0:
            return 'G-code 没有有效挤出路径'
        if stats['print_time_min'] <= 0:
            return 'G-code 没有可用的打印时间估算'
        bounds = stats['toolpath_bounds_mm']
        if bounds is None:
            return 'G-code 没有可用的刀路边界'
        extents = np.asarray(bounds[1]) - np.asarray(bounds[0])
        build = np.asarray([
            float(settings['machine_width']),
            float(settings['machine_depth']),
            float(settings['machine_height']),
        ])
        if np.any(~np.isfinite(extents)) or np.any(extents < 0):
            return 'G-code 刀路边界无效'
        if np.any(extents > build + 1e-6):
            return f'G-code 刀路尺寸 {extents.tolist()} 超出平台 {build.tolist()}'
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if bool(settings.get('machine_center_is_zero', False)):
            platform_min = np.asarray([-build[0] / 2, -build[1] / 2, 0.0])
            platform_max = np.asarray([build[0] / 2, build[1] / 2, build[2]])
        else:
            platform_min = np.zeros(3, dtype=np.float64)
            platform_max = build
        if (
            np.any(bounds_array[0] < platform_min - 1e-6)
            or np.any(bounds_array[1] > platform_max + 1e-6)
        ):
            return (
                f'G-code 刀路边界 {bounds} 超出平台坐标范围 '
                f'{[platform_min.tolist(), platform_max.tolist()]}'
            )
        return ''

    @property
    def definition_search_paths(self) -> list[Path]:
        paths: list[Path] = []
        if self._def_path:
            paths.append(Path(self._def_path).parent)
        if self._extruder_def_path:
            paths.append(Path(self._extruder_def_path).parent)
        if self._resources_path:
            resources = Path(self._resources_path)
            paths.extend([resources / 'definitions', resources / 'extruders'])
        result: list[Path] = []
        for path in paths:
            resolved = path.resolve()
            if resolved.is_dir() and resolved not in result:
                result.append(resolved)
        return result

    @property
    def available(self) -> bool:
        return not self.availability_error

    @property
    def availability_error(self) -> str:
        required = {
            'CuraEngine 可执行文件': self._engine_path,
            '打印机定义': self._def_path,
            '挤出机定义': self._extruder_def_path,
        }
        missing = [
            label for label, value in required.items()
            if not value or not Path(value).is_file()
        ]
        if missing:
            return f'CuraEngine 不可用，缺少: {", ".join(missing)}'
        return ''

    @property
    def engine_version(self) -> str:
        if self._engine_version is None:
            self._engine_version = self._detect_engine_version()
        return self._engine_version

    def _detect_engine_version(self) -> str:
        if not self._engine_path or not Path(self._engine_path).is_file():
            return ''
        try:
            proc = subprocess.run(
                [self._engine_path, 'help'],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace',
                cwd=str(Path(self._engine_path).parent),
            )
            match = re.search(
                r'Cura_(?:SteamEngine|Engine) version\s+([^\s]+)',
                (proc.stdout or '') + '\n' + (proc.stderr or ''),
            )
            return match.group(1) if match else 'unknown'
        except Exception:
            return 'unknown'

    @property
    def engine_path(self) -> Optional[str]:
        return self._engine_path

    @property
    def def_path(self) -> Optional[str]:
        return self._def_path

    @property
    def extruder_def_path(self) -> Optional[str]:
        return self._extruder_def_path

    @property
    def resources_path(self) -> Optional[str]:
        return self._resources_path

    @staticmethod
    def _normalise_optional_path(value) -> Optional[str]:
        if value is None:
            return None
        return str(Path(value).expanduser().resolve())

    @staticmethod
    def _find_engine() -> Optional[str]:
        env_path = os.environ.get('CURA_ENGINE_PATH')
        if env_path and Path(env_path).expanduser().is_file():
            return str(Path(env_path).expanduser().resolve())

        root = Path(__file__).resolve().parent.parent
        names = ('CuraEngine.exe', 'CuraEngine')
        cache_root = root / '.cache' / 'tools' / 'curaengine'
        candidates: list[Path] = []
        if cache_root.is_dir():
            for name in names:
                candidates.extend(cache_root.glob(f'*/*/{name}'))
                candidates.extend(cache_root.glob(f'*/{name}'))
        legacy = root / 'tools' / 'curaengine' / 'curaengine'
        candidates.extend(legacy / name for name in names)
        for candidate in sorted(candidates, key=_version_sort_key, reverse=True):
            if candidate.is_file():
                return str(candidate.resolve())

        for executable in ('CuraEngine', 'curaengine'):
            found = shutil.which(executable)
            if found:
                return str(Path(found).resolve())
        return None

    @staticmethod
    def _find_resources(engine_path: Optional[str]) -> Optional[str]:
        env_path = os.environ.get('CURA_ENGINE_RESOURCES')
        if env_path and Path(env_path).expanduser().is_dir():
            return str(Path(env_path).expanduser().resolve())
        if not engine_path:
            return None
        base = Path(engine_path).resolve().parent
        candidates = [
            base / 'share' / 'cura' / 'resources',
            base / 'resources',
            base.parent / 'share' / 'cura' / 'resources',
        ]
        for candidate in candidates:
            if (candidate / 'definitions' / 'fdmprinter.def.json').is_file():
                return str(candidate.resolve())
        return None


def _parse_gcode(
    gcode_path: str | Path,
    *,
    filament_diameter_mm: float = 1.75,
    material_density_g_cm3: float = 1.24,
) -> dict:
    """扫描完整 G-code，提取可靠的时间、耗材、层数和刀路边界。"""
    time_header = 0.0
    time_elapsed = 0.0
    filament_header_mm = 0.0
    layer_count = 0
    max_layer = -1
    layer_height = 0.0
    seen_layer = False
    extrusion_absolute = True
    position_absolute = True
    active_tool = 0
    current_e: dict[int, float] = {0: 0.0}
    retraction_debt: dict[int, float] = {0: 0.0}
    filament_mm = 0.0
    extrusion_moves = 0
    position = {'X': None, 'Y': None, 'Z': None}
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    warnings: list[str] = []
    elapsed_snapshot = None

    time_re = re.compile(rf'^;TIME:({_NUMBER})', re.IGNORECASE)
    elapsed_re = re.compile(rf'^;TIME_ELAPSED:({_NUMBER})', re.IGNORECASE)
    filament_re = re.compile(
        rf'^;Filament used(?: \[m\])?:?\s*({_NUMBER})\s*m?',
        re.IGNORECASE,
    )
    layer_count_re = re.compile(r'^;LAYER_COUNT:\s*(\d+)', re.IGNORECASE)
    layer_re = re.compile(r'^;LAYER:\s*(-?\d+)', re.IGNORECASE)
    layer_h_re = re.compile(rf'^;Layer height:\s*({_NUMBER})', re.IGNORECASE)

    with Path(gcode_path).open('r', encoding='utf-8', errors='replace') as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue

            match = time_re.match(line)
            if match:
                time_header = max(time_header, float(match.group(1)))
            match = elapsed_re.match(line)
            if match:
                elapsed_value = float(match.group(1))
                if elapsed_value >= time_elapsed:
                    time_elapsed = elapsed_value
                    elapsed_snapshot = (
                        filament_mm,
                        extrusion_moves,
                        bounds_min.copy(),
                        bounds_max.copy(),
                    )
            match = filament_re.match(line)
            if match:
                filament_header_mm = max(
                    filament_header_mm, float(match.group(1)) * 1000.0
                )
            match = layer_count_re.match(line)
            if match:
                layer_count = max(layer_count, int(match.group(1)))
            match = layer_re.match(line)
            if match:
                layer_index = int(match.group(1))
                if layer_index >= 0:
                    seen_layer = True
                    max_layer = max(max_layer, layer_index)
            match = layer_h_re.match(line)
            if match:
                layer_height = float(match.group(1))

            command_text = line.split(';', 1)[0].strip()
            if not command_text:
                continue
            command = command_text.split(maxsplit=1)[0].upper()
            if re.fullmatch(r'T\d+', command):
                active_tool = int(command[1:])
                current_e.setdefault(active_tool, 0.0)
                retraction_debt.setdefault(active_tool, 0.0)
                continue
            if command == 'M82':
                extrusion_absolute = True
                continue
            if command == 'M83':
                extrusion_absolute = False
                continue
            if command == 'G90':
                position_absolute = True
                continue
            if command == 'G91':
                position_absolute = False
                continue

            axes = {
                name.upper(): float(value)
                for name, value in _AXIS_RE.findall(command_text)
            }
            if command == 'G92':
                if 'E' in axes:
                    current_e[active_tool] = axes['E']
                for axis in ('X', 'Y', 'Z'):
                    if axis in axes:
                        position[axis] = axes[axis]
                continue
            if command not in {'G0', 'G1'}:
                continue

            if 'E' in axes:
                old_e = current_e.get(active_tool, 0.0)
                delta = axes['E'] - old_e if extrusion_absolute else axes['E']
                current_e[active_tool] = axes['E'] if extrusion_absolute else old_e + axes['E']
                debt = retraction_debt.get(active_tool, 0.0)
                if delta < 0:
                    retraction_debt[active_tool] = debt - delta
                elif delta > 0:
                    consumed = min(delta, debt)
                    printable = delta - consumed
                    retraction_debt[active_tool] = debt - consumed
                    if seen_layer and printable > 0:
                        filament_mm += printable
                        extrusion_moves += 1

            for axis in ('X', 'Y', 'Z'):
                if axis not in axes:
                    continue
                previous = position[axis]
                if position_absolute or previous is None:
                    position[axis] = axes[axis]
                else:
                    position[axis] = previous + axes[axis]
            if seen_layer and all(position[axis] is not None for axis in ('X', 'Y', 'Z')):
                point = np.asarray([position['X'], position['Y'], position['Z']])
                bounds_min = np.minimum(bounds_min, point)
                bounds_max = np.maximum(bounds_max, point)

    # 最后一个 TIME_ELAPSED 位于 Cura 生成的结束脚本之前；使用该快照，避免
    # 归零、退料或呈现模型等结束动作污染耗材和刀路边界。
    if elapsed_snapshot is not None:
        filament_mm, extrusion_moves, bounds_min, bounds_max = elapsed_snapshot

    if layer_count == 0 and max_layer >= 0:
        layer_count = max_layer + 1
    if filament_mm <= 0 and filament_header_mm > 0:
        filament_mm = filament_header_mm
        warnings.append('未能从 E 轴刀路计算耗材，已使用 G-code 头部耗材值')
    if time_elapsed > 0:
        print_time_seconds = time_elapsed
        if time_header > 0 and abs(time_header - time_elapsed) / time_elapsed > 0.2:
            warnings.append('G-code 头部 TIME 与最终 TIME_ELAPSED 差异过大，已采用后者')
    else:
        print_time_seconds = time_header
        if time_header > 0:
            warnings.append('缺少 TIME_ELAPSED，打印时间退回使用 G-code 头部 TIME')

    if np.all(np.isfinite(bounds_min)) and np.all(np.isfinite(bounds_max)):
        toolpath_bounds = [bounds_min.tolist(), bounds_max.tolist()]
    else:
        toolpath_bounds = None

    radius = filament_diameter_mm / 2.0
    volume_cm3 = filament_mm * np.pi * radius ** 2 / 1000.0
    filament_g = round(volume_cm3 * material_density_g_cm3, 3)
    return {
        'print_time_min': print_time_seconds / 60.0,
        'filament_mm': filament_mm,
        'filament_g': filament_g,
        'layer_count': layer_count,
        'layer_height_mm': layer_height,
        'toolpath_bounds_mm': toolpath_bounds,
        'extrusion_move_count': extrusion_moves,
        'warnings': warnings,
    }


def _validate_settings(
    settings: Optional[dict[str, object]],
    label: str,
) -> dict[str, object]:
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise TypeError(f'{label} 必须是字典')
    result: dict[str, object] = {}
    for key, value in settings.items():
        if not isinstance(key, str) or not key or any(char.isspace() for char in key):
            raise ValueError(f'{label} 包含非法设置名: {key!r}')
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f'{label}[{key!r}] 仅支持字符串、数字或布尔值')
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f'{label}[{key!r}] 必须是有限数值')
        if isinstance(value, str) and '\x00' in value:
            raise ValueError(f'{label}[{key!r}] 不能包含 NUL 字符')
        result[key] = value
    return result


def _format_setting_value(value: object) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _is_extruder_setting(key: str) -> bool:
    return key.startswith(_EXTRUDER_PREFIXES)


def _mesh_ok(mesh: trimesh.Trimesh) -> bool:
    return (
        isinstance(mesh, trimesh.Trimesh)
        and len(mesh.vertices) >= 3
        and len(mesh.faces) >= 1
        and np.all(np.isfinite(mesh.vertices))
    )


def _engine_error_lines(output: str) -> list[str]:
    return [
        line.strip() for line in output.splitlines()
        if '[error]' in line.lower()
    ]


def _unlink_quietly(path: Optional[Path]):
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _version_sort_key(path: Path) -> tuple[int, ...]:
    for part in path.parts:
        if re.fullmatch(r'\d+(?:\.\d+)+', part):
            return tuple(int(value) for value in part.split('.'))
    return (0,)
