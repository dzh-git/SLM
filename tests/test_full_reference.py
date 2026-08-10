"""Opt-in full-resolution acceptance test.

Run with: $env:SLM_RUN_FULL='1'; python -m pytest -m slow -q
"""

import os

import pytest

from slm_splitter.config import load_config
from slm_splitter.metrics import evaluate_focal_field
from slm_splitter.optics import build_grid, propagate_for_measurement
from slm_splitter.optimizer import optimize_phase


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("SLM_RUN_FULL") != "1", reason="requires the 4096x4096 full grid")
def test_full_reference_21x21_acceptance():
    config = load_config("examples/reference_21x21.yaml")
    grid = build_grid(config)
    result = optimize_phase(config, grid)
    focal, measurement_grid = propagate_for_measurement(grid.input_amplitude, result.phase, config, grid)
    metrics, _records = evaluate_focal_field(focal, config, measurement_grid)

    assert metrics["spot_count"] == 441
    assert metrics["power_cv"] <= 0.05
    assert metrics["minimum_to_maximum_power_ratio"] >= 0.90
    assert metrics["maximum_relative_power_deviation"] <= 0.10
    assert metrics["uniformity_passed"]
    assert metrics["maximum_position_error_um"] < config.target.max_position_error_um
