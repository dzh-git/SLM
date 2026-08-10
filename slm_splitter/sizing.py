"""Analytical SLM requirement calculations."""

import math
from dataclasses import asdict
from typing import Any, Dict, List

from .config import ProjectConfig


def _focal_limit(wavelength_m: float, focal_length_m: float, pixel_pitch_m: float) -> float:
    """Paraxial focal-plane half-width represented by the sampled SLM."""
    return wavelength_m * focal_length_m / (2.0 * pixel_pitch_m)


def calculate_requirements(config: ProjectConfig) -> Dict[str, Any]:
    optics = config.optics
    target = config.target
    slm = config.slm
    design = config.design

    wavelength = optics.wavelength_nm * 1e-9
    focal_length = optics.focal_length_mm * 1e-3
    spot_radius = target.spot_radius_um * 1e-6
    pitch = target.pitch_um * 1e-6
    max_error = target.max_position_error_um * 1e-6
    pixel_pitch = slm.pixel_pitch_um * 1e-6

    half_span_x = 0.5 * (target.columns - 1) * pitch
    half_span_y = 0.5 * (target.rows - 1) * pitch
    margin = design.fov_margin_spot_radii * spot_radius
    required_x = half_span_x + margin
    required_y = half_span_y + margin
    required_radius = max(required_x, required_y)

    effective_na = wavelength / (math.pi * spot_radius)
    beam_radius = wavelength * focal_length / (math.pi * spot_radius)
    if optics.input_beam_radius_mm is not None:
        configured_beam_radius = optics.input_beam_radius_mm * 1e-3
    else:
        configured_beam_radius = beam_radius

    containment_factor = math.sqrt(-math.log(1.0 - design.pupil_power_fraction) / 2.0)
    aperture_for_beam = 2.0 * containment_factor * configured_beam_radius
    # The current optimizer snaps targets to a square DFT grid. Guarantee the
    # requested radial error, not merely each Cartesian component.
    aperture_for_position = wavelength * focal_length / (math.sqrt(2.0) * max_error)
    minimum_active_size = max(aperture_for_beam, aperture_for_position)

    maximum_pixel_pitch = wavelength * focal_length / (2.0 * required_radius)
    objective_na_for_containment = containment_factor * configured_beam_radius / focal_length
    minimum_resolution_x = int(math.ceil(minimum_active_size / pixel_pitch))
    minimum_resolution_y = minimum_resolution_x

    active_width = slm.resolution_x * pixel_pitch
    active_height = slm.resolution_y * pixel_pitch
    sampling_x = wavelength * focal_length / active_width
    sampling_y = wavelength * focal_length / active_height
    radial_sampling_error = 0.5 * math.hypot(sampling_x, sampling_y)
    focal_limit = _focal_limit(wavelength, focal_length, pixel_pitch)
    carrier_samples = wavelength * focal_length / (required_radius * pixel_pitch)

    checks = {
        "pixel_pitch": pixel_pitch <= maximum_pixel_pitch,
        "active_width": active_width >= minimum_active_size,
        "active_height": active_height >= minimum_active_size,
        "objective_na": optics.objective_na >= effective_na,
        "objective_power_containment": optics.objective_na >= objective_na_for_containment,
        "position_sampling_radial": radial_sampling_error <= max_error,
        "field_of_view_x": focal_limit >= required_x,
        "field_of_view_y": focal_limit >= required_y,
    }

    failures: List[str] = []
    failure_text = {
        "pixel_pitch": "像元过大，无法表示最外侧光斑所需的空间频率。",
        "active_width": "SLM 有效宽度不足，无法满足光束包络或位置采样要求。",
        "active_height": "SLM 有效高度不足，无法满足光束包络或位置采样要求。",
        "objective_na": "物镜 NA 小于形成目标 Gaussian 焦斑所需的有效 NA。",
        "objective_power_containment": "物镜 NA 不足以通过设计指定比例的 Gaussian 光束功率。",
        "position_sampling_radial": "焦平面 DFT 网格的二维最坏位置量化误差超限。",
        "field_of_view_x": "x 方向 Nyquist 焦平面视场不足。",
        "field_of_view_y": "y 方向 Nyquist 焦平面视场不足。",
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(failure_text[name])

    tradeoff = []
    seen = set()
    for p_um in [
        maximum_pixel_pitch * 1e6,
        maximum_pixel_pitch * 0.8e6,
        maximum_pixel_pitch * 0.6e6,
        slm.pixel_pitch_um,
    ]:
        key = round(p_um, 6)
        if key in seen:
            continue
        seen.add(key)
        tradeoff.append(
            {
                "pixel_pitch_um": p_um,
                "minimum_square_resolution": int(math.ceil(minimum_active_size / (p_um * 1e-6))),
            }
        )

    return {
        "input": {
            "optics": asdict(optics),
            "target": asdict(target),
            "slm": asdict(slm),
        },
        "requirements": {
            "array_half_span_x_um": half_span_x * 1e6,
            "array_half_span_y_um": half_span_y * 1e6,
            "required_focal_radius_um": required_radius * 1e6,
            "effective_na": effective_na,
            "minimum_objective_na_for_spot": effective_na,
            "minimum_objective_na_for_power_fraction": objective_na_for_containment,
            "required_gaussian_beam_radius_mm": beam_radius * 1e3,
            "required_gaussian_beam_diameter_mm": 2.0 * beam_radius * 1e3,
            "beam_containment_radius_factor": containment_factor,
            "minimum_active_size_mm": minimum_active_size * 1e3,
            "active_size_from_beam_mm": aperture_for_beam * 1e3,
            "active_size_from_position_mm": aperture_for_position * 1e3,
            "active_size_from_numerical_position_mm": aperture_for_position * 1e3,
            "maximum_pixel_pitch_um": maximum_pixel_pitch * 1e6,
            "minimum_resolution_x_at_candidate_pitch": minimum_resolution_x,
            "minimum_resolution_y_at_candidate_pitch": minimum_resolution_y,
        },
        "candidate": {
            "active_width_mm": active_width * 1e3,
            "active_height_mm": active_height * 1e3,
            "focal_sampling_x_um": sampling_x * 1e6,
            "focal_sampling_y_um": sampling_y * 1e6,
            "nearest_bin_error_x_um": 0.5 * sampling_x * 1e6,
            "nearest_bin_error_y_um": 0.5 * sampling_y * 1e6,
            "nearest_bin_radial_error_um": radial_sampling_error * 1e6,
            "nyquist_focal_limit_um": focal_limit * 1e6,
            "samples_per_outer_carrier_period": carrier_samples,
            "checks": checks,
            "passed": all(checks.values()),
            "failures": failures,
            "warnings": (
                ["最外侧载频每周期不足 2.5 个像元；虽未混叠，但工程抗误差余量较小。"]
                if carrier_samples < 2.5
                else []
            ),
        },
        "pixel_pitch_resolution_tradeoff": tradeoff,
    }


def candidate_is_feasible(report: Dict[str, Any]) -> bool:
    return bool(report["candidate"]["passed"])
