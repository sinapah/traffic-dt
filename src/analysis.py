from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import OutagePeriod
from src.prediction import _METRIC_NAMES
from src.telemetry import SystemState, TelemetryBus

_METRIC_LABELS = {
    "queue_length": "Queue Length",
    "arrival_rate": "Arrival Rate",
    "utilization": "Utilization",
    "processing_rate": "Processing Rate",
    "mean_wait_time": "Mean Wait Time",
}


def run_distribution_analysis(
    bus: TelemetryBus,
    estimated_archive: list[SystemState],
    outages: list[OutagePeriod],
    output_dir: str = "output",
    strategy: str = "unknown",
) -> list[str]:
    paths: list[str] = []
    out = Path(output_dir)

    actual = _extract_ground_truth(bus, outages)
    synth = _extract_estimates(estimated_archive, outages)
    if actual is None or synth is None:
        return paths

    p = str(out / f"dt_distribution_histograms_{strategy}.png")
    _plot_histograms(actual, synth, p)
    paths.append(p)

    p = str(out / f"dt_distribution_correlations_{strategy}.png")
    _plot_correlation_matrices(actual, synth, p)
    paths.append(p)

    p = str(out / f"dt_distribution_joint_{strategy}.png")
    _plot_joint_distributions(actual, synth, p)
    paths.append(p)

    errors = _compute_errors(actual, synth)
    error_path = out / f"dt_distribution_errors_{strategy}.json"
    with open(error_path, "w") as f:
        json.dump(errors, f, indent=2)
    paths.append(str(error_path))

    print("\nDistribution Analysis:")
    n_actual = len(next(iter(actual.values())))
    n_synth = len(next(iter(synth.values())))
    print(f"  Samples: {n_actual} ground truth, {n_synth} synthesized")
    for metric in _METRIC_NAMES:
        e = errors[metric]
        print(f"  {_METRIC_LABELS[metric]:20s}  MAE={e['mae']:.4f}  RMSE={e['rmse']:.4f}")
    print(f"  Correlation diff (mean abs off-diag): {errors['correlation_diff']:.4f}")
    print(f"  Plots: {paths}")

    return paths


def _extract_ground_truth(
    bus: TelemetryBus,
    outages: list[OutagePeriod],
) -> dict[str, np.ndarray] | None:
    if not outages:
        return None

    values = {m: [] for m in _METRIC_NAMES}
    for pkt in bus.all_packets:
        for o in outages:
            if o.start <= pkt.timestamp < o.end:
                for m in _METRIC_NAMES:
                    values[m].append(float(getattr(pkt, m)))
                break

    if all(len(v) == 0 for v in values.values()):
        return None

    return {m: np.array(v) for m, v in values.items()}


def _extract_estimates(
    estimated_archive: list[SystemState],
    outages: list[OutagePeriod],
) -> dict[str, np.ndarray] | None:
    if not estimated_archive or not outages:
        return None

    values = {m: [] for m in _METRIC_NAMES}
    for state in estimated_archive:
        for o in outages:
            if o.start <= state.timestamp < o.end:
                for edge_state in state.edges.values():
                    for m in _METRIC_NAMES:
                        values[m].append(float(getattr(edge_state, m)))
                break

    if all(len(v) == 0 for v in values.values()):
        return None

    return {m: np.array(v) for m, v in values.items()}


def _plot_histograms(
    actual: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    save_path: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for i, metric in enumerate(_METRIC_NAMES):
        ax = axes_flat[i]
        a = actual[metric]
        p = predicted[metric]

        combined = np.concatenate([a, p])
        lo, hi = float(np.percentile(combined, 1)), float(np.percentile(combined, 99))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 30)

        ax.hist(a, bins=bins, alpha=0.6, label="Ground truth", color="#1f77b4", density=True)
        ax.hist(p, bins=bins, alpha=0.6, label="Synthesized", color="#ff7f0e", density=True)
        ax.set_title(_METRIC_LABELS[metric])
        ax.legend(fontsize=8)
        ax.set_ylabel("Density")

    axes_flat[-1].axis("off")
    fig.suptitle("Distribution Comparison: Ground Truth vs DT-Synthesized Telemetry", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_correlation_matrices(
    actual: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    save_path: str,
) -> None:
    def _corr_matrix(data):
        arr = np.column_stack([data[m] for m in _METRIC_NAMES])
        return np.corrcoef(arr, rowvar=False)

    corr_actual = _corr_matrix(actual)
    corr_pred = _corr_matrix(predicted)

    diff = np.abs(corr_actual - corr_pred)
    n = diff.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_diff = float(diff[mask].mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmin = min(corr_actual.min(), corr_pred.min())
    vmax = max(corr_actual.max(), corr_pred.max())

    labels = [_METRIC_LABELS[m] for m in _METRIC_NAMES]

    for ax, corr, title in [
        (axes[0], corr_actual, "Ground Truth"),
        (axes[1], corr_pred, "Synthesized"),
    ]:
        im = ax.imshow(corr, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.suptitle(f"Correlation Matrices  |  Mean Abs Diff (off-diag): {mean_diff:.4f}", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_joint_distributions(
    actual: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    save_path: str,
) -> None:
    pairs = [
        ("arrival_rate", "queue_length", "Arrival Rate vs Queue Length"),
        ("arrival_rate", "utilization", "Arrival Rate vs Utilization"),
        ("queue_length", "mean_wait_time", "Queue Length vs Mean Wait Time"),
        ("utilization", "processing_rate", "Utilization vs Processing Rate"),
        ("arrival_rate", "mean_wait_time", "Arrival Rate vs Mean Wait Time"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for idx, (x_metric, y_metric, title) in enumerate(pairs):
        ax = axes_flat[idx]
        ax.scatter(actual[x_metric], actual[y_metric], alpha=0.3, s=8, label="Ground truth", color="#1f77b4")
        ax.scatter(predicted[x_metric], predicted[y_metric], alpha=0.3, s=8, label="Synthesized", color="#ff7f0e")
        ax.set_xlabel(_METRIC_LABELS[x_metric])
        ax.set_ylabel(_METRIC_LABELS[y_metric])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes_flat[-1].axis("off")
    fig.suptitle("Joint Distribution: Ground Truth vs Synthesized", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _compute_errors(
    actual: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
) -> dict:
    errors: dict = {}
    for metric in _METRIC_NAMES:
        a = actual[metric]
        p = predicted[metric]
        min_len = min(len(a), len(p))
        diff = a[:min_len] - p[:min_len]
        errors[metric] = {
            "mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "n_ground_truth": int(len(a)),
            "n_synthesized": int(len(p)),
            "mean_ground_truth": float(a.mean()),
            "mean_synthesized": float(p[:min_len].mean()),
        }

    def _corr_mat(data):
        arr = np.column_stack([data[m] for m in _METRIC_NAMES])
        return np.corrcoef(arr, rowvar=False)

    cr = _corr_mat(actual)
    cp = _corr_mat(predicted)
    n = cr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    errors["correlation_diff"] = float(np.abs(cr - cp)[mask].mean())

    return errors
