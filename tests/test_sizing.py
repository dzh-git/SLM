from copy import deepcopy

import pytest

from slm_splitter.config import load_config
from slm_splitter.sizing import calculate_requirements


def test_reference_sizing_matches_analytical_values():
    config = load_config("examples/reference_21x21.yaml")
    report = calculate_requirements(config)
    requirements = report["requirements"]

    assert requirements["array_half_span_x_um"] == pytest.approx(2000.0)
    assert requirements["effective_na"] == pytest.approx(0.16934086, rel=1e-6)
    assert requirements["maximum_pixel_pitch_um"] == pytest.approx(4.1168248, rel=1e-6)
    assert requirements["minimum_active_size_mm"] == pytest.approx(15.93167, rel=1e-5)
    assert requirements["minimum_objective_na_for_power_fraction"] == pytest.approx(0.2569624, rel=1e-6)
    assert requirements["minimum_resolution_x_at_candidate_pitch"] == 3983
    assert report["candidate"]["passed"]


def test_candidate_reports_independent_failures():
    config = deepcopy(load_config("examples/reference_21x21.yaml"))
    config.slm.pixel_pitch_um = 8.0
    config.slm.resolution_x = 1000
    config.slm.resolution_y = 1000
    config.optics.objective_na = 0.1
    report = calculate_requirements(config)

    assert not report["candidate"]["passed"]
    assert not report["candidate"]["checks"]["pixel_pitch"]
    assert not report["candidate"]["checks"]["active_width"]
    assert not report["candidate"]["checks"]["objective_na"]
    assert not report["candidate"]["checks"]["objective_power_containment"]
    assert report["candidate"]["failures"]


def test_wavelength_scaling_changes_na_and_pixel_limit():
    base = load_config("examples/reference_21x21.yaml")
    doubled = deepcopy(base)
    doubled.optics.wavelength_nm *= 2.0
    base_report = calculate_requirements(base)
    doubled_report = calculate_requirements(doubled)

    assert doubled_report["requirements"]["effective_na"] == pytest.approx(
        2.0 * base_report["requirements"]["effective_na"]
    )
    assert doubled_report["requirements"]["maximum_pixel_pitch_um"] == pytest.approx(
        2.0 * base_report["requirements"]["maximum_pixel_pitch_um"]
    )


def test_position_sampling_uses_two_dimensional_radial_error():
    config = deepcopy(load_config("examples/reference_21x21.yaml"))
    report = calculate_requirements(config)
    single_axis = report["candidate"]["nearest_bin_error_x_um"]
    radial = report["candidate"]["nearest_bin_radial_error_um"]

    assert radial == pytest.approx(2 ** 0.5 * single_axis)
    config.target.max_position_error_um = 0.6
    strict_report = calculate_requirements(config)
    assert not strict_report["candidate"]["checks"]["position_sampling_radial"]
