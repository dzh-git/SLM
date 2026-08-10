"""Weighted Gerchberg-Saxton phase-only hologram optimization."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import fft

from .config import ProjectConfig
from .metrics import power_uniformity
from .optics import (
    OpticalGrid,
    map_targets_to_pixels,
    pixel_aperture_envelope,
    quantize_phase,
    target_coordinates,
)


@dataclass
class OptimizationResult:
    phase: np.ndarray
    cv_history: List[float]
    min_to_max_ratio_history: List[float]
    max_relative_deviation_history: List[float]
    uniformity_score_history: List[float]
    iterations: int
    converged: bool
    target_pixel_errors_m: np.ndarray
    best_power_cv: float
    best_min_to_max_power_ratio: float
    best_max_relative_power_deviation: float
    best_uniformity_score: float


def _target_roi_indices(
    config: ProjectConfig,
    grid: OpticalGrid,
    coordinates: Sequence[Tuple[float, float]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    requested = config.simulation.roi_radius_spot_radii * config.target.spot_radius_um * 1e-6
    radius = min(requested, 0.45 * config.target.pitch_um * 1e-6)
    regions: List[Tuple[np.ndarray, np.ndarray]] = []
    for x_target, y_target in coordinates:
        x_indices = np.flatnonzero(np.abs(grid.x_focus_m - x_target) <= radius)
        y_indices = np.flatnonzero(np.abs(grid.y_focus_m - y_target) <= radius)
        xx, yy = np.meshgrid(grid.x_focus_m[x_indices], grid.y_focus_m[y_indices])
        inside = (xx - x_target) ** 2 + (yy - y_target) ** 2 <= radius ** 2
        local_rows, local_cols = np.nonzero(inside)
        regions.append((y_indices[local_rows], x_indices[local_cols]))
    return regions


def _integrated_powers(intensity: np.ndarray, regions: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.asarray([np.sum(intensity[rows, cols], dtype=np.float64) for rows, cols in regions])


def optimize_phase(config: ProjectConfig, grid: OpticalGrid) -> OptimizationResult:
    coordinates = target_coordinates(config)
    rows, cols, errors = map_targets_to_pixels(coordinates, grid.x_focus_m, grid.y_focus_m)
    regions = _target_roi_indices(config, grid, coordinates)
    if any(region_rows.size == 0 for region_rows, _region_cols in regions):
        raise ValueError("焦平面采样过粗，至少一个目标光斑的积分区域为空。")
    rng = np.random.RandomState(config.optimization.seed)

    target_field = np.zeros(grid.input_amplitude.shape, dtype=np.complex64)
    target_field[rows, cols] = np.exp(1j * rng.uniform(0, 2.0 * np.pi, len(rows))).astype(np.complex64)
    slm_guess = fft.fftshift(
        fft.ifft2(fft.ifftshift(target_field), norm="ortho", workers=config.simulation.workers)
    )
    phase = np.angle(slm_guess).astype(np.float32)
    transfer = pixel_aperture_envelope(config, grid)
    history: List[float] = []
    ratio_history: List[float] = []
    deviation_history: List[float] = []
    score_history: List[float] = []
    stable_count = 0
    converged = False
    best_score = float("inf")
    best_statistics = None
    best_phase = phase.copy()

    for _iteration in range(config.optimization.max_iterations):
        effective_phase = (
            quantize_phase(phase, config.slm.phase_levels)
            if config.simulation.quantize_phase
            else phase
        )
        slm_field = grid.input_amplitude.astype(np.complex64) * np.exp(1j * effective_phase).astype(np.complex64)
        raw_focal = fft.fftshift(
            fft.fft2(fft.ifftshift(slm_field), norm="ortho", workers=config.simulation.workers)
        )
        focal = raw_focal * transfer
        powers = _integrated_powers(np.abs(focal) ** 2, regions)
        mean_power = float(np.mean(powers))
        statistics = power_uniformity(powers, config)
        history.append(statistics["power_cv"])
        ratio_history.append(statistics["minimum_to_maximum_power_ratio"])
        deviation_history.append(statistics["maximum_relative_power_deviation"])
        score_history.append(statistics["uniformity_score"])

        improvement = best_score - statistics["uniformity_score"]
        if improvement > 0:
            best_score = statistics["uniformity_score"]
            best_statistics = statistics
            best_phase = effective_phase.copy()
        if statistics["uniformity_passed"]:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= config.optimization.patience:
            converged = True
            break
        # Do not return an unevaluated phase when the iteration budget is
        # exhausted. The current phase and the last history item stay aligned.
        if _iteration == config.optimization.max_iterations - 1:
            break

        # Scale each complete target ROI, rather than only its center pixel.
        # The square root converts the power correction into a field correction.
        correction = (mean_power / np.maximum(powers, 1e-20)) ** (
            0.5 * config.optimization.weight_exponent
        )
        constrained = raw_focal.copy()
        for factor, (region_rows, region_cols) in zip(correction, regions):
            constrained[region_rows, region_cols] = raw_focal[region_rows, region_cols] * factor
        inverse = fft.fftshift(
            fft.ifft2(fft.ifftshift(constrained), norm="ortho", workers=config.simulation.workers)
        )
        phase = np.angle(inverse).astype(np.float32)

    returned_phase = np.mod(best_phase, 2.0 * np.pi).astype(np.float32)
    if config.simulation.quantize_phase:
        returned_phase = quantize_phase(returned_phase, config.slm.phase_levels)
    if best_statistics is None:
        raise RuntimeError("优化器未评价任何相位。")
    return OptimizationResult(
        phase=returned_phase,
        cv_history=history,
        min_to_max_ratio_history=ratio_history,
        max_relative_deviation_history=deviation_history,
        uniformity_score_history=score_history,
        iterations=len(history),
        converged=converged,
        target_pixel_errors_m=errors,
        best_power_cv=best_statistics["power_cv"],
        best_min_to_max_power_ratio=best_statistics["minimum_to_maximum_power_ratio"],
        best_max_relative_power_deviation=best_statistics["maximum_relative_power_deviation"],
        best_uniformity_score=best_statistics["uniformity_score"],
    )
