from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import OutagePeriod
from src.metrics import MetricsCollector


def _add_outage_bands(ax, outages: list[OutagePeriod]) -> None:
    for o in outages:
        ax.axvspan(o.start, o.end, alpha=0.2, color="red", label="Telemetry outage")


def plot_queue_length(
    metrics: MetricsCollector,
    outages: list[OutagePeriod] | None = None,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    edge_ids = sorted({e.tags.get("edge_id") for e in metrics.entries if e.name == "queue_length"})
    for eid in edge_ids:
        t, v = metrics.get_series("queue_length", eid)
        ax.plot(t, v, label=f"Edge {eid}", linewidth=0.8)
    if outages:
        _add_outage_bands(ax, outages)
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Queue length")
    ax.set_title("Queue Length per Edge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_utilization(
    metrics: MetricsCollector,
    outages: list[OutagePeriod] | None = None,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    edge_ids = sorted({e.tags.get("edge_id") for e in metrics.entries if e.name == "utilization"})
    for eid in edge_ids:
        t, v = metrics.get_series("utilization", eid)
        ax.plot(t, v, label=f"Edge {eid}", linewidth=0.8)
    if outages:
        _add_outage_bands(ax, outages)
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Utilization")
    ax.set_title("Edge Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_latency(
    metrics: MetricsCollector,
    outages: list[OutagePeriod] | None = None,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    edge_ids = sorted({e.tags.get("edge_id") for e in metrics.entries if e.name == "wait_time"})
    for eid in edge_ids:
        t, v = metrics.get_series("wait_time", eid)
        ax.plot(t, v, label=f"Edge {eid}", linewidth=0.8)
    if outages:
        _add_outage_bands(ax, outages)
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Mean wait time")
    ax.set_title("Mean Frame Wait Time per Edge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_throughput(
    metrics: MetricsCollector,
    outages: list[OutagePeriod] | None = None,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    edge_ids = sorted({e.tags.get("edge_id") for e in metrics.entries if e.name == "processing_rate"})
    for eid in edge_ids:
        t, v = metrics.get_series("processing_rate", eid)
        ax.plot(t, v, label=f"Edge {eid}", linewidth=0.8)
    if outages:
        _add_outage_bands(ax, outages)
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Processing rate (frames/unit time)")
    ax.set_title("Processing Throughput per Edge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fl_convergence(
    metrics: MetricsCollector,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    t, v = metrics.get_series("fl_convergence")
    if t:
        ax.plot(t, v, linewidth=1.2, color="purple")
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Convergence metric")
    ax.set_title("Federated Learning Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_dt_estimation_error(
    dt_errors: list[tuple[float, float]],
    outages: list[OutagePeriod] | None = None,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    if dt_errors:
        t, e = zip(*dt_errors)
        ax.plot(t, e, linewidth=1.0, color="darkred")
    if outages:
        _add_outage_bands(ax, outages)
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Estimation error (avg)")
    ax.set_title("DT Telemetry Estimation Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_all(
    metrics: MetricsCollector,
    outages: list[OutagePeriod] | None = None,
    dt_errors: list[tuple[float, float]] | None = None,
    output_dir: str = "output",
    prefix: str = "",
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, func in [
        ("queue_length", plot_queue_length),
        ("utilization", plot_utilization),
        ("latency", plot_latency),
        ("throughput", plot_throughput),
        ("fl_convergence", plot_fl_convergence),
    ]:
        p = str(out / f"{prefix}{name}.png")
        if name == "fl_convergence":
            func(metrics, save_path=p)
        else:
            func(metrics, outages=outages, save_path=p)
        paths.append(p)
    if dt_errors:
        p = str(out / f"{prefix}dt_estimation_error.png")
        plot_dt_estimation_error(dt_errors, outages=outages, save_path=p)
        paths.append(p)
    return paths
