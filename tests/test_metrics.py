from copy import deepcopy

import numpy as np

from slm_splitter.config import load_config
from slm_splitter.metrics import power_uniformity


def test_power_cv_cannot_hide_a_large_outlier():
    config = deepcopy(load_config("examples/quick_demo.yaml"))
    powers = np.ones(441)
    powers[-1] = 1.5

    metrics = power_uniformity(powers, config)

    assert metrics["power_cv"] < config.optimization.cv_tolerance
    assert metrics["minimum_to_maximum_power_ratio"] < config.optimization.min_to_max_power_ratio
    assert metrics["maximum_relative_power_deviation"] > config.optimization.max_relative_power_deviation
    assert not metrics["uniformity_passed"]
