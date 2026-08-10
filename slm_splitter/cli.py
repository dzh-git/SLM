"""Command-line interface."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ConfigurationError, ProjectConfig, load_config
from .metrics import evaluate_focal_field
from .optics import build_grid, propagate_for_measurement
from .optimizer import OptimizationResult, optimize_phase
from .output import (
    ensure_output,
    save_convergence,
    save_focal_plots,
    save_phase,
    write_json,
    write_sizing_report,
    write_spot_table,
)
from .sizing import calculate_requirements, candidate_is_feasible


def _output_directory(config: ProjectConfig, override: Optional[str]) -> Path:
    return ensure_output(override or config.output.directory)


def _require_feasible(report):
    if not candidate_is_feasible(report):
        details = "\n".join("- " + item for item in report["candidate"]["failures"])
        raise ConfigurationError("候选 SLM 不满足设计约束，已停止仿真/优化：\n" + details)


def command_size(config: ProjectConfig, output: Path):
    report = calculate_requirements(config)
    write_sizing_report(output, report)
    print("SLM sizing: %s" % ("PASS" if report["candidate"]["passed"] else "FAIL"))
    print("Report: %s" % (output / "slm_requirements.md"))
    return report


def command_optimize(config: ProjectConfig, output: Path, report=None) -> OptimizationResult:
    report = report or calculate_requirements(config)
    _require_feasible(report)
    grid = build_grid(config)
    result = optimize_phase(config, grid)
    save_phase(output, result.phase)
    save_convergence(output, result.cv_history)
    write_json(
        output / "optimization_metrics.json",
        {
            "iterations": result.iterations,
            "converged": result.converged,
            "final_target_power_cv": result.best_power_cv,
            "final_minimum_to_maximum_power_ratio": result.best_min_to_max_power_ratio,
            "final_maximum_relative_power_deviation": result.best_max_relative_power_deviation,
            "final_uniformity_score": result.best_uniformity_score,
            "minimum_target_power_cv": min(result.cv_history),
            "last_iteration_power_cv": result.cv_history[-1],
            "maximum_target_bin_error_um": float(np.max(result.target_pixel_errors_m) * 1e6),
            "cv_history": result.cv_history,
            "min_to_max_ratio_history": result.min_to_max_ratio_history,
            "max_relative_deviation_history": result.max_relative_deviation_history,
            "uniformity_score_history": result.uniformity_score_history,
        },
    )
    print(
        "Optimization: %d iterations, CV=%.6f, min/max=%.6f, max deviation=%.6f"
        % (
            result.iterations,
            result.best_power_cv,
            result.best_min_to_max_power_ratio,
            result.best_max_relative_power_deviation,
        )
    )
    return result


def command_simulate(
    config: ProjectConfig,
    output: Path,
    phase_path: Optional[str] = None,
    phase: Optional[np.ndarray] = None,
    report=None,
):
    report = report or calculate_requirements(config)
    _require_feasible(report)
    if phase is None:
        source = Path(phase_path) if phase_path else output / "slm_phase.npy"
        if not source.exists():
            raise ConfigurationError("找不到相位文件：%s。请先运行 optimize 或使用 run。" % source)
        phase = np.load(str(source))
    expected_shape = (config.slm.resolution_y, config.slm.resolution_x)
    if phase.shape != expected_shape:
        raise ConfigurationError("相位数组形状 %s 与 SLM 形状 %s 不一致。" % (phase.shape, expected_shape))
    grid = build_grid(config)
    focal, measurement_grid = propagate_for_measurement(grid.input_amplitude, phase, config, grid)
    metrics, records = evaluate_focal_field(focal, config, measurement_grid)
    metrics["measurement_oversampling"] = config.simulation.measurement_oversampling
    metrics["measurement_sampling_x_um"] = float(
        abs(measurement_grid.x_focus_m[1] - measurement_grid.x_focus_m[0]) * 1e6
    )
    metrics["measurement_sampling_y_um"] = float(
        abs(measurement_grid.y_focus_m[1] - measurement_grid.y_focus_m[0]) * 1e6
    )
    write_json(output / "simulation_metrics.json", metrics)
    write_spot_table(output, records)
    save_focal_plots(output, focal, measurement_grid, config)
    # Keep terminal output ASCII-safe on Windows consoles that still use GBK.
    print(
        "Simulation: %d spots, CV=%.6f, min/max=%.6f, max deviation=%.6f, position=%.4f um"
        % (
            metrics["spot_count"],
            metrics["power_cv"],
            metrics["minimum_to_maximum_power_ratio"],
            metrics["maximum_relative_power_deviation"],
            metrics["maximum_position_error_um"],
        )
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLM 21×21 分束设计、优化与仿真")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("size", "optimize", "run"):
        item = subparsers.add_parser(command)
        item.add_argument("config", help="YAML 配置路径")
        item.add_argument("--output", help="覆盖配置中的输出目录")
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("config", help="YAML 配置路径")
    simulate.add_argument("--output", help="覆盖配置中的输出目录")
    simulate.add_argument("--phase", help="相位 .npy 文件；默认读取输出目录中的 slm_phase.npy")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        output = _output_directory(config, args.output)
        if args.command == "size":
            command_size(config, output)
        elif args.command == "optimize":
            command_optimize(config, output)
        elif args.command == "simulate":
            command_simulate(config, output, phase_path=args.phase)
        elif args.command == "run":
            report = command_size(config, output)
            result = command_optimize(config, output, report=report)
            command_simulate(config, output, phase=result.phase, report=report)
    except (ConfigurationError, ValueError, OSError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
