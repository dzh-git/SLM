"""Focal-array measurement utilities."""

from typing import Any, Dict, List, Tuple

import numpy as np

from .config import ProjectConfig
from .optics import OpticalGrid, map_targets_to_pixels, target_coordinates


def power_uniformity(powers: np.ndarray, config: ProjectConfig) -> Dict[str, Any]:
    """Return equal-power statistics and the combined acceptance result."""
    power_array = np.asarray(powers, dtype=np.float64)
    mean_power = float(np.mean(power_array))
    safe_mean = max(mean_power, 1e-20)
    maximum_power = max(float(np.max(power_array)), 1e-20)
    cv = float(np.std(power_array) / safe_mean)
    min_to_max = float(np.min(power_array) / maximum_power)
    relative = power_array / safe_mean
    maximum_relative_deviation = float(np.max(np.abs(relative - 1.0)))
    cv_passed = cv <= config.optimization.cv_tolerance
    ratio_passed = min_to_max >= config.optimization.min_to_max_power_ratio
    deviation_passed = maximum_relative_deviation <= config.optimization.max_relative_power_deviation
    score = max(
        cv / config.optimization.cv_tolerance,
        config.optimization.min_to_max_power_ratio / max(min_to_max, 1e-20),
        maximum_relative_deviation / config.optimization.max_relative_power_deviation,
    )
    return {
        "power_cv": cv,
        "minimum_to_maximum_power_ratio": min_to_max,
        "maximum_relative_power_deviation": maximum_relative_deviation,
        "minimum_relative_power": float(np.min(relative)),
        "maximum_relative_power": float(np.max(relative)),
        "uniformity_score": float(score),
        "cv_passed": bool(cv_passed),
        "power_ratio_passed": bool(ratio_passed),
        "relative_deviation_passed": bool(deviation_passed),
        "uniformity_passed": bool(cv_passed and ratio_passed and deviation_passed),
    }


def _fit_gaussian_radius(
    signal: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    fallback_second_moment: float,
) -> float:
    """Fit log(I)=constant-2*r^2/w^2 with a moment fallback."""
    peak = float(np.max(signal))
    valid = signal > 0.05 * peak
    if np.count_nonzero(valid) >= 4:
        radius_squared = ((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2)[valid]
        log_intensity = np.log(signal[valid])
        slope, _intercept = np.polyfit(radius_squared, log_intensity, 1)
        if np.isfinite(slope) and slope < 0:
            return float(np.sqrt(-2.0 / slope))
    return float(np.sqrt(max(2.0 * fallback_second_moment, 0.0)))


def evaluate_focal_field(
    focal_field: np.ndarray,
    config: ProjectConfig,
    grid: OpticalGrid,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    intensity = np.abs(focal_field) ** 2
    coordinates = target_coordinates(config)
    rows, cols, bin_errors = map_targets_to_pixels(coordinates, grid.x_focus_m, grid.y_focus_m)
    requested_radius = config.simulation.roi_radius_spot_radii * config.target.spot_radius_um * 1e-6
    roi_radius = min(requested_radius, 0.45 * config.target.pitch_um * 1e-6)

    records: List[Dict[str, float]] = []
    powers = []
    peaks = []
    position_errors = []
    radius_estimates = []

    for index, ((x_target, y_target), row, col, bin_error) in enumerate(
        zip(coordinates, rows, cols, bin_errors)
    ):
        x_mask = np.abs(grid.x_focus_m - x_target) <= roi_radius
        y_mask = np.abs(grid.y_focus_m - y_target) <= roi_radius
        x_indices = np.flatnonzero(x_mask)
        y_indices = np.flatnonzero(y_mask)
        if x_indices.size == 0 or y_indices.size == 0:
            raise ValueError("焦平面采样过粗，光斑积分区域为空。")
        patch = intensity[np.ix_(y_indices, x_indices)].astype(np.float64)
        xx, yy = np.meshgrid(grid.x_focus_m[x_indices], grid.y_focus_m[y_indices])
        radial_mask = (xx - x_target) ** 2 + (yy - y_target) ** 2 <= roi_radius ** 2
        patch_values = patch[radial_mask]
        floor = float(np.percentile(patch_values, 10.0)) if patch_values.size > 4 else 0.0
        signal = np.maximum(patch - floor, 0.0) * radial_mask
        power = float(np.sum(signal))
        if power <= 0:
            centroid_x = float(grid.x_focus_m[col])
            centroid_y = float(grid.y_focus_m[row])
            radius = float("nan")
        else:
            centroid_x = float(np.sum(signal * xx) / power)
            centroid_y = float(np.sum(signal * yy) / power)
            second_moment = float(np.sum(signal * ((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2)) / power)
            radius = _fit_gaussian_radius(signal, xx, yy, centroid_x, centroid_y, second_moment)
        position_error = float(np.hypot(centroid_x - x_target, centroid_y - y_target))
        peak = float(np.max(patch_values))

        powers.append(power)
        peaks.append(peak)
        position_errors.append(position_error)
        if np.isfinite(radius):
            radius_estimates.append(radius)
        records.append(
            {
                "index": float(index),
                "target_x_um": x_target * 1e6,
                "target_y_um": y_target * 1e6,
                "mapped_x_um": float(grid.x_focus_m[col] * 1e6),
                "mapped_y_um": float(grid.y_focus_m[row] * 1e6),
                "bin_error_um": float(bin_error * 1e6),
                "centroid_x_um": centroid_x * 1e6,
                "centroid_y_um": centroid_y * 1e6,
                "position_error_um": position_error * 1e6,
                "integrated_power": power,
                "peak_intensity": peak,
                "radius_1e2_um": radius * 1e6,
            }
        )

    power_array = np.asarray(powers)
    radius_array = np.asarray(radius_estimates, dtype=np.float64)
    total_power = float(np.sum(intensity))
    numerical_roi_efficiency = float(np.sum(power_array) / max(total_power, 1e-20))
    uniformity = power_uniformity(power_array, config)
    target_radius = config.target.spot_radius_um * 1e-6
    if radius_array.size:
        radius_metrics = {
            "median_radius_1e2_um": float(np.median(radius_array) * 1e6),
            "mean_radius_1e2_um": float(np.mean(radius_array) * 1e6),
            "minimum_radius_1e2_um": float(np.min(radius_array) * 1e6),
            "radius_p05_1e2_um": float(np.percentile(radius_array, 5.0) * 1e6),
            "radius_p95_1e2_um": float(np.percentile(radius_array, 95.0) * 1e6),
            "maximum_radius_1e2_um": float(np.max(radius_array) * 1e6),
            "median_radius_relative_error": float(abs(np.median(radius_array) / target_radius - 1.0)),
            "maximum_radius_relative_error": float(np.max(np.abs(radius_array / target_radius - 1.0))),
        }
    else:
        radius_metrics = {
            "median_radius_1e2_um": float("nan"),
            "mean_radius_1e2_um": float("nan"),
            "minimum_radius_1e2_um": float("nan"),
            "radius_p05_1e2_um": float("nan"),
            "radius_p95_1e2_um": float("nan"),
            "maximum_radius_1e2_um": float("nan"),
            "median_radius_relative_error": float("nan"),
            "maximum_radius_relative_error": float("nan"),
        }
    metrics = {
        "spot_count": len(records),
        **uniformity,
        "diffraction_efficiency": numerical_roi_efficiency,
        "numerical_roi_efficiency": numerical_roi_efficiency,
        "maximum_position_error_um": float(np.max(position_errors) * 1e6),
        "rms_position_error_um": float(np.sqrt(np.mean(np.square(position_errors))) * 1e6),
        "maximum_bin_mapping_error_um": float(np.max(bin_errors) * 1e6),
        **radius_metrics,
        "mean_peak_intensity": float(np.mean(peaks)),
        "position_passed": bool(np.max(position_errors) * 1e6 <= config.target.max_position_error_um),
    }
    return metrics, records
