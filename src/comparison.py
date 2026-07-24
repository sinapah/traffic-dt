from __future__ import annotations

from src.config import SimulationConfig
from src.simulation import Simulation, SimulationResult
from src.visualization import plot_all


def run_baseline(config: SimulationConfig) -> SimulationResult:
    cfg = SimulationConfig.from_dict(config.to_dict())
    cfg.orchestrator.enabled = False
    sim = Simulation(cfg)
    return sim.run(dt_enabled=False)


def run_dt_driven(config: SimulationConfig) -> SimulationResult:
    sim = Simulation(config)
    return sim.run(dt_enabled=True)


def compare(
    config: SimulationConfig,
    output_dir: str = "output",
) -> tuple[SimulationResult, SimulationResult, list[str]]:
    baseline = run_baseline(config)
    dt_result = run_dt_driven(config)

    baseline_paths = plot_all(
        baseline.metrics,
        outages=config.outages,
        output_dir=output_dir,
        prefix="baseline_",
    )
    dt_paths = plot_all(
        dt_result.metrics,
        outages=config.outages,
        dt_errors=dt_result.digital_twin.estimation_errors if dt_result.digital_twin else None,
        output_dir=output_dir,
        prefix="dt_",
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    b_summary = baseline.metrics.get_summary()
    d_summary = dt_result.metrics.get_summary()
    for metric_name in ["queue_length", "utilization", "wait_time", "dropped_frames"]:
        if metric_name in b_summary and metric_name in d_summary:
            b = b_summary[metric_name]
            d = d_summary[metric_name]
            print(f"\n{metric_name}:")
            print(f"  Baseline  - mean: {b['mean']:.3f}, max: {b['max']:.3f}, p95: {b['p95']:.3f}")
            print(f"  DT-driven - mean: {d['mean']:.3f}, max: {d['max']:.3f}, p95: {d['p95']:.3f}")

    b_total_dropped = sum(
        b_summary.get(f"dropped_frames", {}).get("max", 0) for _ in [1]
    )
    d_total_dropped = sum(
        d_summary.get(f"dropped_frames", {}).get("max", 0) for _ in [1]
    )
    print(f"\nFL rounds completed: baseline={baseline.fl_coordinator.round_number}, dt={dt_result.fl_coordinator.round_number}")
    print(f"FL convergence:     baseline={baseline.fl_coordinator.current_convergence:.4f}, dt={dt_result.fl_coordinator.current_convergence:.4f}")

    if dt_result.digital_twin and dt_result.digital_twin.estimation_errors:
        errors = [e for _, e in dt_result.digital_twin.estimation_errors]
        print(f"DT estimation error: mean={sum(errors)/len(errors):.4f}, max={max(errors):.4f}")

    print(f"\nBaseline plots: {baseline_paths}")
    print(f"DT-driven plots: {dt_paths}")
    print("=" * 60)

    return baseline, dt_result, baseline_paths + dt_paths
