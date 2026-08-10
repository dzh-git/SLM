"""Configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration is physically or structurally invalid."""


@dataclass
class OpticsConfig:
    wavelength_nm: float
    focal_length_mm: float
    objective_na: float
    input_profile: str = "gaussian"
    input_beam_radius_mm: Optional[float] = None


@dataclass
class TargetConfig:
    rows: int
    columns: int
    spot_radius_um: float
    pitch_um: float
    max_position_error_um: float
    radius_definition: str = "1/e2"


@dataclass
class SLMConfig:
    pixel_pitch_um: float
    resolution_x: int
    resolution_y: int
    fill_factor: float = 1.0
    phase_levels: int = 256


@dataclass
class DesignConfig:
    fov_margin_spot_radii: float = 3.0
    pupil_power_fraction: float = 0.99


@dataclass
class SimulationConfig:
    finite_aperture: bool = True
    apply_fill_factor: bool = True
    quantize_phase: bool = True
    roi_radius_spot_radii: float = 3.0
    measurement_oversampling: int = 1
    workers: int = -1


@dataclass
class OptimizationConfig:
    max_iterations: int = 80
    weight_exponent: float = 0.7
    cv_tolerance: float = 0.05
    min_to_max_power_ratio: float = 0.9
    max_relative_power_deviation: float = 0.1
    patience: int = 5
    seed: int = 42


@dataclass
class OutputConfig:
    directory: str = "outputs/run"


@dataclass
class ProjectConfig:
    optics: OpticsConfig
    target: TargetConfig
    slm: SLMConfig
    design: DesignConfig
    simulation: SimulationConfig
    optimization: OptimizationConfig
    output: OutputConfig


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError("配置节 '%s' 必须是映射。" % name)
    return value


def load_config(path: str) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("YAML 顶层必须是映射。")

    try:
        config = ProjectConfig(
            optics=OpticsConfig(**_section(data, "optics")),
            target=TargetConfig(**_section(data, "target")),
            slm=SLMConfig(**_section(data, "slm")),
            design=DesignConfig(**_section(data, "design")),
            simulation=SimulationConfig(**_section(data, "simulation")),
            optimization=OptimizationConfig(**_section(data, "optimization")),
            output=OutputConfig(**_section(data, "output")),
        )
    except TypeError as exc:
        raise ConfigurationError("配置字段缺失或名称错误：%s" % exc) from exc
    validate_config(config)
    return config


def validate_config(config: ProjectConfig) -> None:
    optics = config.optics
    target = config.target
    slm = config.slm
    design = config.design
    simulation = config.simulation
    optimization = config.optimization

    positive = {
        "wavelength_nm": optics.wavelength_nm,
        "focal_length_mm": optics.focal_length_mm,
        "objective_na": optics.objective_na,
        "spot_radius_um": target.spot_radius_um,
        "pitch_um": target.pitch_um,
        "max_position_error_um": target.max_position_error_um,
        "pixel_pitch_um": slm.pixel_pitch_um,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ConfigurationError("%s 必须大于 0。" % name)
    if optics.focal_length_mm <= 30.0:
        raise ConfigurationError("本项目要求 focal_length_mm > 30 mm。")
    if not 0 < optics.objective_na < 0.5:
        raise ConfigurationError("本项目要求 0 < objective_na < 0.5。")
    if optics.input_profile not in ("gaussian", "uniform"):
        raise ConfigurationError("input_profile 只能是 gaussian 或 uniform。")
    if optics.input_beam_radius_mm is not None and optics.input_beam_radius_mm <= 0:
        raise ConfigurationError("input_beam_radius_mm 必须大于 0 或留空。")
    if target.rows < 1 or target.columns < 1:
        raise ConfigurationError("阵列行列数必须大于 0。")
    if target.rows % 2 == 0 or target.columns % 2 == 0:
        raise ConfigurationError("当前实现要求奇数行和奇数列，以便阵列关于光轴对称。")
    if target.radius_definition != "1/e2":
        raise ConfigurationError("当前版本仅支持 1/e2 强度半径。")
    if slm.resolution_x < 2 or slm.resolution_y < 2:
        raise ConfigurationError("SLM 分辨率必须至少为 2×2。")
    if not 0 < slm.fill_factor <= 1:
        raise ConfigurationError("fill_factor 必须在 (0, 1] 内。")
    if slm.phase_levels < 2:
        raise ConfigurationError("phase_levels 必须至少为 2。")
    if not 0 < design.pupil_power_fraction < 1:
        raise ConfigurationError("pupil_power_fraction 必须在 (0, 1) 内。")
    if design.fov_margin_spot_radii < 0:
        raise ConfigurationError("fov_margin_spot_radii 不能为负数。")
    if simulation.roi_radius_spot_radii <= 0:
        raise ConfigurationError("roi_radius_spot_radii 必须大于 0。")
    if (
        isinstance(simulation.measurement_oversampling, bool)
        or not isinstance(simulation.measurement_oversampling, int)
        or simulation.measurement_oversampling < 1
    ):
        raise ConfigurationError("measurement_oversampling 必须是大于等于 1 的整数。")
    if optimization.max_iterations < 1 or optimization.patience < 1:
        raise ConfigurationError("优化迭代次数和 patience 必须大于 0。")
    if not 0 < optimization.cv_tolerance < 1:
        raise ConfigurationError("cv_tolerance 必须在 (0, 1) 内。")
    if not 0 < optimization.min_to_max_power_ratio <= 1:
        raise ConfigurationError("min_to_max_power_ratio 必须在 (0, 1] 内。")
    if not 0 < optimization.max_relative_power_deviation < 1:
        raise ConfigurationError("max_relative_power_deviation 必须在 (0, 1) 内。")
    if not 0 < optimization.weight_exponent <= 1:
        raise ConfigurationError("weight_exponent 必须在 (0, 1] 内。")
