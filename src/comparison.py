from __future__ import annotations

import ctypes
import gc
import json
import logging
from pathlib import Path

from src.analysis import run_distribution_analysis
from src.config import SimulationConfig
from src.simulation import Simulation, SimulationResult
from src.visualization import plot_all

log = logging.getLogger(__name__)


def run_baseline(config: SimulationConfig) -> SimulationResult:
    cfg = SimulationConfig.from_dict(config.to_dict())
    cfg.orchestrator.enabled = False
    sim = Simulation(cfg)
    return sim.run(dt_enabled=False)


def run_dt_driven(config: SimulationConfig) -> SimulationResult:
    sim = Simulation(config)
    return sim.run(dt_enabled=True)


def _free_result(result: SimulationResult) -> None:
    """Explicitly release the heavy objects (models, dataset) held by a SimulationResult."""
    for edge in result.edges.values():
        edge.model = None
        edge.assigned_dataset = None
    result.fl_coordinator.global_model = None
    result.fl_coordinator.val_loader = None
    result.cameras.clear()
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def compare(
    config: SimulationConfig,
    output_dir: str = "output",
) -> tuple[SimulationResult, SimulationResult, list[str]]:
    out = Path(output_dir)
    baseline_dir = str(out / "baseline")
    dt_dir = str(out / "dt")

    # --- Baseline run ---
    log.info("=== Baseline run (no digital twin, no orchestrator) ===")
    baseline = run_baseline(config)
    log.info("Baseline run complete — generating plots...")

    baseline_paths = plot_all(
        baseline.metrics,
        outages=config.outages,
        output_dir=baseline_dir,
    )
    b_summary = baseline.metrics.get_summary()
    _save_summary(b_summary, out / "baseline" / "metrics_summary.json")

    # Snapshot the lightweight stats we need for the final report, then free
    # all of the heavy objects (models, dataset) so they don't sit in memory
    # alongside the DT simulation's models.
    b_fl_rounds = baseline.fl_coordinator.round_number
    b_fl_convergence = baseline.fl_coordinator.current_convergence
    b_mAP = baseline.fl_coordinator.round_accuracy[-1].get("mAP", 0.0) if baseline.fl_coordinator.round_accuracy else None
    b_loss = baseline.fl_coordinator.round_losses[-1] if baseline.fl_coordinator.round_losses else None

    log.info("Freeing baseline models and dataset from memory...")
    _free_result(baseline)

    # --- DT-driven run ---
    log.info("=== DT-driven run (digital twin + orchestrator enabled) ===")
    dt_result = run_dt_driven(config)
    log.info("DT-driven run complete — generating plots...")

    dt_paths = plot_all(
        dt_result.metrics,
        outages=config.outages,
        dt_errors=dt_result.digital_twin.estimation_errors if dt_result.digital_twin else None,
        output_dir=dt_dir,
    )
    d_summary = dt_result.metrics.get_summary()
    _save_summary(d_summary, out / "dt" / "metrics_summary.json")

    # --- Print comparison summary ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for metric_name in ["queue_length", "utilization", "wait_time", "dropped_frames"]:
        if metric_name in b_summary and metric_name in d_summary:
            b = b_summary[metric_name]
            d = d_summary[metric_name]
            print(f"\n{metric_name}:")
            print(f"  Baseline  - mean: {b['mean']:.3f}, max: {b['max']:.3f}, p95: {b['p95']:.3f}")
            print(f"  DT-driven - mean: {d['mean']:.3f}, max: {d['max']:.3f}, p95: {d['p95']:.3f}")

    print(f"\nFL rounds completed: baseline={b_fl_rounds}, dt={dt_result.fl_coordinator.round_number}")
    print(f"FL convergence:     baseline={b_fl_convergence:.4f}, dt={dt_result.fl_coordinator.current_convergence:.4f}")

    if b_mAP is not None:
        d_mAP = dt_result.fl_coordinator.round_accuracy[-1].get("mAP", 0.0) if dt_result.fl_coordinator.round_accuracy else 0.0
        print(f"FL final mAP:       baseline={b_mAP:.4f}, dt={d_mAP:.4f}")
    if b_loss is not None:
        d_loss = dt_result.fl_coordinator.round_losses[-1] if dt_result.fl_coordinator.round_losses else 0.0
        print(f"FL final loss:      baseline={b_loss:.4f}, dt={d_loss:.4f}")

    if dt_result.digital_twin and dt_result.digital_twin.estimation_errors:
        errors = [e for _, e in dt_result.digital_twin.estimation_errors]
        print(f"DT estimation error: mean={sum(errors)/len(errors):.4f}, max={max(errors):.4f}")

    analysis_paths: list[str] = []
    if dt_result.digital_twin:
        strategy = config.dt.prediction.strategy
        analysis_dir = str(out / "dt" / strategy)
        analysis_paths = run_distribution_analysis(
            bus=dt_result.telemetry_bus,
            estimated_archive=dt_result.digital_twin.estimated_archive,
            outages=config.outages,
            output_dir=analysis_dir,
        )

    all_paths = baseline_paths + dt_paths + analysis_paths

    print(f"\nBaseline plots: {baseline_paths}")
    print(f"DT-driven plots: {dt_paths}")
    if analysis_paths:
        print(f"Distribution analysis: {analysis_paths}")
    print("=" * 60)

    return baseline, dt_result, all_paths


def _save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Metrics summary saved to {path}")
