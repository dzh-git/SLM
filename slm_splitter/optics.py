"""Scalar Fourier-optics primitives."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import fft

from .config import ProjectConfig


@dataclass
class OpticalGrid:
    x_slm_m: np.ndarray
    y_slm_m: np.ndarray
    x_focus_m: np.ndarray
    y_focus_m: np.ndarray
    input_amplitude: np.ndarray


def _focus_axis(count: int, pixel_pitch_m: float, wavelength_m: float, focal_length_m: float) -> np.ndarray:
    frequencies = np.fft.fftshift(np.fft.fftfreq(count, d=pixel_pitch_m))
    return wavelength_m * focal_length_m * frequencies


def build_grid(config: ProjectConfig) -> OpticalGrid:
    nx = config.slm.resolution_x
    ny = config.slm.resolution_y
    pixel_pitch = config.slm.pixel_pitch_um * 1e-6
    wavelength = config.optics.wavelength_nm * 1e-9
    focal_length = config.optics.focal_length_mm * 1e-3

    x = (np.arange(nx, dtype=np.float32) - nx // 2) * pixel_pitch
    y = (np.arange(ny, dtype=np.float32) - ny // 2) * pixel_pitch
    xx, yy = np.meshgrid(x, y)

    if config.optics.input_profile == "gaussian":
        if config.optics.input_beam_radius_mm is None:
            beam_radius = wavelength * focal_length / (np.pi * config.target.spot_radius_um * 1e-6)
        else:
            beam_radius = config.optics.input_beam_radius_mm * 1e-3
        amplitude = np.exp(-(xx * xx + yy * yy) / (beam_radius * beam_radius)).astype(np.float32)
    else:
        amplitude = np.ones((ny, nx), dtype=np.float32)

    if config.simulation.finite_aperture:
        pupil_radius = focal_length * config.optics.objective_na
        amplitude *= ((xx * xx + yy * yy) <= pupil_radius * pupil_radius)

    return OpticalGrid(
        x_slm_m=x,
        y_slm_m=y,
        x_focus_m=_focus_axis(nx, pixel_pitch, wavelength, focal_length),
        y_focus_m=_focus_axis(ny, pixel_pitch, wavelength, focal_length),
        input_amplitude=amplitude,
    )


def quantize_phase(phase: np.ndarray, levels: int) -> np.ndarray:
    wrapped = np.mod(phase, 2.0 * np.pi)
    indices = np.mod(np.rint(wrapped * levels / (2.0 * np.pi)), levels)
    return (indices * 2.0 * np.pi / levels).astype(np.float32)


def pixel_aperture_envelope(config: ProjectConfig, grid: OpticalGrid) -> np.ndarray:
    """Return the field envelope of square sample-and-hold pixels.

    fill_factor is interpreted as illuminated area fraction. The linear
    aperture width is sqrt(fill_factor)*p. Even a 100% fill factor has the sinc
    envelope of a width-p pixel; disabling apply_fill_factor selects the ideal
    point-sampled model.
    """
    if not config.simulation.apply_fill_factor:
        return np.ones((grid.y_focus_m.size, grid.x_focus_m.size), dtype=np.float32)
    wavelength = config.optics.wavelength_nm * 1e-9
    focal_length = config.optics.focal_length_mm * 1e-3
    pixel_pitch = config.slm.pixel_pitch_um * 1e-6
    aperture_width = math.sqrt(config.slm.fill_factor) * pixel_pitch
    envelope_x = np.sinc(aperture_width * grid.x_focus_m / (wavelength * focal_length))
    envelope_y = np.sinc(aperture_width * grid.y_focus_m / (wavelength * focal_length))
    return (envelope_y[:, None] * envelope_x[None, :]).astype(np.float32)


def _apply_pixel_aperture_envelope(
    focal_field: np.ndarray,
    config: ProjectConfig,
    grid: OpticalGrid,
) -> None:
    """Apply the separable pixel envelope in place to limit peak memory."""
    if not config.simulation.apply_fill_factor:
        return
    wavelength = config.optics.wavelength_nm * 1e-9
    focal_length = config.optics.focal_length_mm * 1e-3
    pixel_pitch = config.slm.pixel_pitch_um * 1e-6
    aperture_width = math.sqrt(config.slm.fill_factor) * pixel_pitch
    envelope_x = np.sinc(aperture_width * grid.x_focus_m / (wavelength * focal_length)).astype(np.float32)
    envelope_y = np.sinc(aperture_width * grid.y_focus_m / (wavelength * focal_length)).astype(np.float32)
    focal_field *= envelope_y[:, None]
    focal_field *= envelope_x[None, :]


def propagate(
    amplitude: np.ndarray,
    phase: np.ndarray,
    config: ProjectConfig,
    grid: OpticalGrid,
) -> np.ndarray:
    effective_phase = phase
    if config.simulation.quantize_phase:
        effective_phase = quantize_phase(phase, config.slm.phase_levels)
    slm_field = amplitude.astype(np.complex64) * np.exp(1j * effective_phase).astype(np.complex64)
    focal_field = fft.fftshift(
        fft.fft2(fft.ifftshift(slm_field), norm="ortho", workers=config.simulation.workers)
    )
    _apply_pixel_aperture_envelope(focal_field, config, grid)
    return focal_field.astype(np.complex64, copy=False)


def propagate_for_measurement(
    amplitude: np.ndarray,
    phase: np.ndarray,
    config: ProjectConfig,
    grid: OpticalGrid,
) -> Tuple[np.ndarray, OpticalGrid]:
    """Propagate on a zero-padded grid for final focal-plane measurement.

    Zero padding does not add optical information. It evaluates the same
    discrete pupil field on a finer Fourier grid so centroids and Gaussian
    radii are not limited to the native FFT-bin spacing.
    """
    factor = config.simulation.measurement_oversampling
    if factor == 1:
        return propagate(amplitude, phase, config, grid), grid

    effective_phase = phase
    if config.simulation.quantize_phase:
        effective_phase = quantize_phase(phase, config.slm.phase_levels)

    ny, nx = amplitude.shape
    padded_ny = factor * ny
    padded_nx = factor * nx
    offset_y = (padded_ny - ny) // 2
    offset_x = (padded_nx - nx) // 2
    padded_field = np.zeros((padded_ny, padded_nx), dtype=np.complex64)
    padded_field[offset_y : offset_y + ny, offset_x : offset_x + nx] = (
        amplitude.astype(np.complex64) * np.exp(1j * effective_phase).astype(np.complex64)
    )
    unshifted_field = fft.ifftshift(padded_field)
    del padded_field
    unshifted_focal = fft.fft2(
        unshifted_field,
        norm="ortho",
        overwrite_x=True,
        workers=config.simulation.workers,
    )
    focal_field = fft.fftshift(unshifted_focal)
    del unshifted_field, unshifted_focal

    wavelength = config.optics.wavelength_nm * 1e-9
    focal_length = config.optics.focal_length_mm * 1e-3
    pixel_pitch = config.slm.pixel_pitch_um * 1e-6
    measurement_grid = OpticalGrid(
        x_slm_m=grid.x_slm_m,
        y_slm_m=grid.y_slm_m,
        x_focus_m=_focus_axis(padded_nx, pixel_pitch, wavelength, focal_length),
        y_focus_m=_focus_axis(padded_ny, pixel_pitch, wavelength, focal_length),
        input_amplitude=grid.input_amplitude,
    )
    _apply_pixel_aperture_envelope(focal_field, config, measurement_grid)
    return focal_field.astype(np.complex64, copy=False), measurement_grid


def target_coordinates(config: ProjectConfig) -> List[Tuple[float, float]]:
    pitch = config.target.pitch_um * 1e-6
    xs = (np.arange(config.target.columns) - config.target.columns // 2) * pitch
    ys = (np.arange(config.target.rows) - config.target.rows // 2) * pitch
    return [(float(x), float(y)) for y in ys for x in xs]


def map_targets_to_pixels(
    coordinates: Sequence[Tuple[float, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    cols = []
    errors = []
    for x_target, y_target in coordinates:
        col = int(np.argmin(np.abs(x_axis - x_target)))
        row = int(np.argmin(np.abs(y_axis - y_target)))
        rows.append(row)
        cols.append(col)
        errors.append(float(np.hypot(x_axis[col] - x_target, y_axis[row] - y_target)))
    rows_array = np.asarray(rows, dtype=np.intp)
    cols_array = np.asarray(cols, dtype=np.intp)
    if len(set(zip(rows_array.tolist(), cols_array.tolist()))) != len(coordinates):
        raise ValueError("多个目标光斑映射到同一个 FFT 像素；需要更大的有效面积或更小的阵列间距。")
    return rows_array, cols_array, np.asarray(errors)
