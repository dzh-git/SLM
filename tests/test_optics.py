from copy import deepcopy

import numpy as np
import pytest

from slm_splitter.config import load_config
from slm_splitter.optics import (
    build_grid,
    pixel_aperture_envelope,
    propagate,
    propagate_for_measurement,
    quantize_phase,
)


def _sampling_config():
    config = deepcopy(load_config("examples/quick_demo.yaml"))
    config.target.rows = 5
    config.target.columns = 5
    config.target.spot_radius_um = 8.0
    config.target.pitch_um = 40.0
    config.target.max_position_error_um = 10.0
    config.slm.pixel_pitch_um = 20.0
    config.slm.resolution_x = 512
    config.slm.resolution_y = 512
    config.optics.objective_na = 0.05
    return config


def test_linear_phase_grating_moves_focus_to_requested_coordinate():
    config = _sampling_config()
    grid = build_grid(config)
    target_col = config.slm.resolution_x // 2 + 20
    target_x = grid.x_focus_m[target_col]
    wavelength = config.optics.wavelength_nm * 1e-9
    focal_length = config.optics.focal_length_mm * 1e-3
    spatial_frequency = np.sin(np.arctan(target_x / focal_length)) / wavelength
    phase = (2.0 * np.pi * spatial_frequency * grid.x_slm_m[None, :]).astype(np.float32)
    phase = np.broadcast_to(phase, grid.input_amplitude.shape)

    intensity = np.abs(propagate(grid.input_amplitude, phase, config, grid)) ** 2
    row, col = np.unravel_index(np.argmax(intensity), intensity.shape)

    assert grid.x_focus_m[col] == pytest.approx(target_x, abs=config.target.max_position_error_um * 1e-6)
    assert abs(grid.y_focus_m[row]) <= config.target.max_position_error_um * 1e-6


def test_unmodulated_gaussian_has_expected_1e2_radius():
    config = _sampling_config()
    grid = build_grid(config)
    phase = np.zeros(grid.input_amplitude.shape, dtype=np.float32)
    intensity = np.abs(propagate(grid.input_amplitude, phase, config, grid)) ** 2
    center_row = config.slm.resolution_y // 2
    normalized = intensity[center_row] / np.max(intensity[center_row])
    positive = np.arange(config.slm.resolution_x // 2, config.slm.resolution_x)
    index = positive[np.argmin(np.abs(normalized[positive] - np.exp(-2.0)))]
    measured_radius_um = grid.x_focus_m[index] * 1e6

    focal_sampling_um = abs(grid.x_focus_m[index + 1] - grid.x_focus_m[index]) * 1e6
    assert measured_radius_um == pytest.approx(config.target.spot_radius_um, abs=1.5 * focal_sampling_um)


def test_phase_quantization_range_and_levels():
    phase = np.linspace(-1.0, 8.0, 100, dtype=np.float32)
    result = quantize_phase(phase, 16)

    assert np.min(result) >= 0
    assert np.max(result) < 2.0 * np.pi
    assert np.unique(result).size <= 16


def test_fill_factor_is_interpreted_as_area_fraction():
    config = _sampling_config()
    config.slm.fill_factor = 0.81
    config.simulation.apply_fill_factor = True
    grid = build_grid(config)
    envelope = pixel_aperture_envelope(config, grid)
    col = config.slm.resolution_x // 2 + 10
    wavelength = config.optics.wavelength_nm * 1e-9
    focal_length = config.optics.focal_length_mm * 1e-3
    linear_width = 0.9 * config.slm.pixel_pitch_um * 1e-6
    expected = np.sinc(linear_width * grid.x_focus_m[col] / (wavelength * focal_length))

    assert envelope[config.slm.resolution_y // 2, col] == pytest.approx(expected)


def test_zero_padding_refines_measurement_grid_without_moving_focus():
    config = _sampling_config()
    config.simulation.measurement_oversampling = 4
    config.simulation.apply_fill_factor = False
    config.simulation.quantize_phase = False
    grid = build_grid(config)
    phase = np.zeros(grid.input_amplitude.shape, dtype=np.float32)

    focal, measurement_grid = propagate_for_measurement(grid.input_amplitude, phase, config, grid)
    native_sampling = grid.x_focus_m[1] - grid.x_focus_m[0]
    refined_sampling = measurement_grid.x_focus_m[1] - measurement_grid.x_focus_m[0]
    row, col = np.unravel_index(np.argmax(np.abs(focal) ** 2), focal.shape)

    assert focal.shape == (4 * config.slm.resolution_y, 4 * config.slm.resolution_x)
    assert refined_sampling == pytest.approx(native_sampling / 4.0)
    assert measurement_grid.x_focus_m[col] == pytest.approx(0.0, abs=1e-12)
    assert measurement_grid.y_focus_m[row] == pytest.approx(0.0, abs=1e-12)
