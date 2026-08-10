"""Output serialization and plotting."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import ProjectConfig
from .optics import OpticalGrid


def ensure_output(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_sizing_report(output: Path, report: Dict[str, Any]) -> None:
    write_json(output / "slm_requirements.json", report)
    req = report["requirements"]
    candidate = report["candidate"]
    status = "通过" if candidate["passed"] else "不通过"
    lines = [
        "# SLM 参数计算结果",
        "",
        "## 最低要求",
        "",
        "- 最大像元尺寸：%.4f µm" % req["maximum_pixel_pitch_um"],
        "- 最小有效边长：%.4f mm" % req["minimum_active_size_mm"],
        "- 候选像元尺寸下最低分辨率：%d × %d" % (
            req["minimum_resolution_x_at_candidate_pitch"],
            req["minimum_resolution_y_at_candidate_pitch"],
        ),
        "- 所需有效 NA：%.6f" % req["effective_na"],
        "- 指定功率包络所需物镜 NA：%.6f" % req["minimum_objective_na_for_power_fraction"],
        "- 所需 Gaussian 1/e² 强度直径：%.4f mm" % req["required_gaussian_beam_diameter_mm"],
        "",
        "## 候选 SLM",
        "",
        "- 结论：**%s**" % status,
        "- 有效尺寸：%.4f mm × %.4f mm" % (candidate["active_width_mm"], candidate["active_height_mm"]),
        "- 焦平面采样：%.4f µm × %.4f µm" % (candidate["focal_sampling_x_um"], candidate["focal_sampling_y_um"]),
    ]
    if candidate["failures"]:
        lines.extend(["", "## 未满足约束", ""])
        lines.extend("- " + item for item in candidate["failures"])
    if candidate.get("warnings"):
        lines.extend(["", "## 工程警告", ""])
        lines.extend("- " + item for item in candidate["warnings"])
    (output / "slm_requirements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    wrapped = np.mod(phase, 2.0 * np.pi)
    return np.mod(np.rint(wrapped * 256.0 / (2.0 * np.pi)), 256).astype(np.uint8)


def save_phase(output: Path, phase: np.ndarray) -> None:
    wrapped = np.mod(phase, 2.0 * np.pi).astype(np.float32)
    np.save(str(output / "slm_phase.npy"), wrapped)
    # Phase is circular: 0 and 2π are identical. Map the 256 half-open bins
    # [0, 2π) to integer device codes 0..255 without dropping the top code.
    gray = phase_to_uint8(wrapped)
    Image.fromarray(gray, mode="L").save(str(output / "slm_phase_8bit.png"))
    plt.figure(figsize=(7, 6))
    plt.imshow(wrapped, cmap="twilight", origin="lower", vmin=0, vmax=2 * np.pi)
    plt.colorbar(label="Phase (rad)")
    plt.title("SLM phase")
    plt.tight_layout()
    plt.savefig(str(output / "slm_phase_preview.png"), dpi=160)
    plt.close()


def save_convergence(output: Path, history: List[float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel("Iteration")
    plt.ylabel("Target power CV")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output / "convergence.png"), dpi=160)
    plt.close()


def _crop_indices(axis: np.ndarray, limit: float) -> np.ndarray:
    indices = np.flatnonzero(np.abs(axis) <= limit)
    return indices if indices.size else np.arange(axis.size)


def _max_pool_for_display(image: np.ndarray, maximum_side: int = 1400) -> np.ndarray:
    """Preserve diffraction peaks when a large FFT grid is rasterized for a PNG."""
    factor = max(1, int(np.ceil(max(image.shape) / maximum_side)))
    if factor == 1:
        return image
    pad_y = (-image.shape[0]) % factor
    pad_x = (-image.shape[1]) % factor
    padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode="constant")
    return padded.reshape(
        padded.shape[0] // factor,
        factor,
        padded.shape[1] // factor,
        factor,
    ).max(axis=(1, 3))


def save_focal_plots(
    output: Path,
    focal_field: np.ndarray,
    grid: OpticalGrid,
    config: ProjectConfig,
) -> None:
    intensity = np.abs(focal_field) ** 2
    max_value = max(float(np.max(intensity)), 1e-30)
    normalized = intensity / max_value
    limit_x = (0.5 * (config.target.columns - 1) + 1.0) * config.target.pitch_um * 1e-6
    limit_y = (0.5 * (config.target.rows - 1) + 1.0) * config.target.pitch_um * 1e-6
    x_idx = _crop_indices(grid.x_focus_m, limit_x)
    y_idx = _crop_indices(grid.y_focus_m, limit_y)
    crop = normalized[np.ix_(y_idx, x_idx)]
    display_crop = _max_pool_for_display(crop)
    extent = [
        grid.x_focus_m[x_idx[0]] * 1e6,
        grid.x_focus_m[x_idx[-1]] * 1e6,
        grid.y_focus_m[y_idx[0]] * 1e6,
        grid.y_focus_m[y_idx[-1]] * 1e6,
    ]

    for name, image, label in [
        ("focal_plane_linear.png", display_crop, "Normalized intensity"),
        ("focal_plane_log.png", 10.0 * np.log10(np.maximum(display_crop, 1e-8)), "Intensity (dB)"),
    ]:
        plt.figure(figsize=(8, 7))
        plt.imshow(image, origin="lower", extent=extent, cmap="inferno", aspect="equal")
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.colorbar(label=label)
        plt.tight_layout()
        plt.savefig(str(output / name), dpi=180)
        plt.close()

    center_row = int(np.argmin(np.abs(grid.y_focus_m)))
    plt.figure(figsize=(8, 4))
    plt.plot(grid.x_focus_m[x_idx] * 1e6, normalized[center_row, x_idx])
    plt.xlabel("x (µm)")
    plt.ylabel("Normalized intensity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output / "central_cross_section.png"), dpi=160)
    plt.close()

    local_limit = max(5.0 * config.target.spot_radius_um, 10.0) * 1e-6
    local_x = _crop_indices(grid.x_focus_m, local_limit)
    local_y = _crop_indices(grid.y_focus_m, local_limit)
    local = normalized[np.ix_(local_y, local_x)]
    local_extent = [
        grid.x_focus_m[local_x[0]] * 1e6,
        grid.x_focus_m[local_x[-1]] * 1e6,
        grid.y_focus_m[local_y[0]] * 1e6,
        grid.y_focus_m[local_y[-1]] * 1e6,
    ]
    plt.figure(figsize=(6, 5))
    plt.imshow(local, origin="lower", extent=local_extent, cmap="inferno")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.colorbar(label="Normalized intensity")
    plt.tight_layout()
    plt.savefig(str(output / "central_spot.png"), dpi=180)
    plt.close()


def write_spot_table(output: Path, records: List[Dict[str, float]]) -> None:
    if not records:
        return
    with (output / "spot_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
