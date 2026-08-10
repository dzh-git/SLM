from copy import deepcopy

import numpy as np

from slm_splitter.config import load_config
from slm_splitter.metrics import evaluate_focal_field
from slm_splitter.optics import build_grid, propagate_for_measurement
from slm_splitter.optimizer import optimize_phase


def test_21x21_quick_pipeline_meets_uniformity_and_position_limits():
    config = load_config("examples/quick_demo.yaml")
    grid = build_grid(config)
    result = optimize_phase(config, grid)
    focal, measurement_grid = propagate_for_measurement(grid.input_amplitude, result.phase, config, grid)
    metrics, records = evaluate_focal_field(focal, config, measurement_grid)

    assert result.phase.shape == (512, 512)
    assert result.converged
    assert np.min(result.phase) >= 0
    assert np.max(result.phase) < 2.0 * np.pi
    assert metrics["spot_count"] == 441
    assert len(records) == 441
    assert metrics["power_cv"] <= 0.05
    assert metrics["minimum_to_maximum_power_ratio"] >= 0.90
    assert metrics["maximum_relative_power_deviation"] <= 0.10
    assert metrics["uniformity_passed"]
    assert metrics["maximum_position_error_um"] < config.target.max_position_error_um
