"""DFM 阈值规则加载与校验。

配置统一使用毫米和角度制，支持 FDM / SLA 两种工艺。加载时会进行
结构、类型和取值范围校验，避免错误阈值静默进入制造性判定。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class DFMRules:
    """加载并管理 DFM 检查阈值。"""

    _SECTIONS = (
        "wall", "overhang", "cavity", "dimension",
        "platform", "printer", "performance", "scoring",
    )
    _PROCESSES = {"FDM", "SLA"}
    _POSITIVE_KEYS = {
        "nozzle_diameter", "layer_height", "min_wall_thickness",
        "min_feature_size", "max_slenderness_ratio", "lattice_min_wall",
        "min_contact_area", "bridge_max_span", "min_cavity_volume",
        "sdf_pitch", "min_hole_diameter", "gap_threshold",
        "min_thread_pitch", "min_bottom_area", "wall_sample_count",
        "self_intersection_max_faces",
    }

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "dfm_config.yaml"
        self._path = Path(config_path)
        self._base: dict[str, Any] = {}
        self._raw: dict[str, Any] = {}
        self.reload()

    def reload(self):
        """重新读取、校验配置，并重新应用所选工艺的覆盖项。"""
        with self._path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise ValueError(f"DFM 配置必须是 YAML 映射: {self._path}")
        self._validate_structure(loaded)
        self._base = deepcopy(loaded)
        self.set_process(str(loaded.get("process", "FDM")))

    def set_process(self, process: str):
        """切换工艺；大小写不敏感，最终统一保存为大写。"""
        normalized = str(process).strip().upper()
        if normalized not in self._PROCESSES:
            raise ValueError(f"不支持的打印工艺 {process!r}，仅支持 FDM 或 SLA")

        self._raw = deepcopy(self._base)
        self._raw["process"] = normalized
        if normalized == "SLA":
            for key, value in self._base.get("sla_overrides", {}).items():
                self._apply_override(key, value)
        self._validate_values(self._raw)

    def _apply_override(self, key: str, value: Any):
        """将工艺覆盖值写入已有配置键；拼写错误直接报错。"""
        matches = [
            section for section in self._SECTIONS
            if key in self._raw.get(section, {})
        ]
        if len(matches) != 1:
            raise ValueError(
                f"SLA 覆盖项 {key!r} 必须且只能匹配一个现有配置键，实际匹配 {matches}"
            )
        self._raw[matches[0]][key] = value

    def _validate_structure(self, config: dict[str, Any]):
        for section in self._SECTIONS:
            if section not in config:
                raise ValueError(f"DFM 配置缺少节: {section}")
            if not isinstance(config[section], dict):
                raise ValueError(f"DFM 配置节 {section} 必须是映射")
        if not isinstance(config.get("sla_overrides", {}), dict):
            raise ValueError("sla_overrides 必须是映射")

        process = str(config.get("process", "FDM")).strip().upper()
        if process not in self._PROCESSES:
            raise ValueError(f"process 必须是 FDM 或 SLA，实际为 {process!r}")

    def _validate_values(self, config: dict[str, Any]):
        flat: dict[str, Any] = {}
        for section in self._SECTIONS:
            flat.update(config.get(section, {}))

        for key in self._POSITIVE_KEYS:
            if key in flat:
                value = flat[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"{key} 必须是正数，实际为 {value!r}")

        angle = flat.get("critical_angle")
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not 0 < angle < 90:
            raise ValueError(f"critical_angle 必须在 (0, 90) 度内，实际为 {angle!r}")

        ratio = flat.get("max_overhang_area_ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 <= ratio <= 1:
            raise ValueError(
                f"max_overhang_area_ratio 必须在 [0, 1] 内，实际为 {ratio!r}"
            )

        volume = flat.get("build_volume")
        if (
            not isinstance(volume, list)
            or len(volume) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in volume)
        ):
            raise ValueError("build_volume 必须是三个正数 [X, Y, Z]")

        for key, weight in config.get("scoring", {}).items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(f"评分权重 {key} 必须是正数")

    @property
    def process(self) -> str:
        return self._raw["process"]

    def get(self, key: str, default=None):
        """按顶层、各配置节的顺序查找一个配置键。"""
        if key in self._raw and not isinstance(self._raw[key], dict):
            return self._raw[key]
        for section in self._SECTIONS:
            table = self._raw.get(section, {})
            if key in table:
                return table[key]
        return default

    def __getitem__(self, key: str):
        value = self.get(key)
        if value is None:
            raise KeyError(f"配置中未找到: {key}")
        return value

    def __repr__(self):
        return (
            f"DFMRules(process={self.process!r}, "
            f"wall={self.get('min_wall_thickness')}mm, "
            f"nozzle={self.get('nozzle_diameter')}mm)"
        )
