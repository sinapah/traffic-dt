from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import SimulationConfig
from src.simulation import Simulation
from src.comparison import compare
from src.visualization import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic DT Simulation")
    parser.add_argument("--config", type=str, default="config/default.json", help="Config file path")
    parser.add_argument(
        "--mode",
        choices=["single", "compare", "baseline"],
        default="compare",
        help="Run mode: single (DT-driven only), baseline (no DT), compare (both)",
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = SimulationConfig.from_file(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = SimulationConfig()

    if args.seed is not None:
        config.seed = args.seed

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        baseline, dt_result, plots = compare(config, output_dir=str(output_dir))
    elif args.mode == "baseline":
        from src.comparison import run_baseline
        result = run_baseline(config)
        plots = plot_all(
            result.metrics,
            outages=config.outages,
            output_dir=str(output_dir),
            prefix="baseline_",
        )
        print(f"\nBaseline run complete. Plots: {plots}")
    else:
        sim = Simulation(config)
        result = sim.run(dt_enabled=True)
        plots = plot_all(
            result.metrics,
            outages=config.outages,
            dt_errors=result.digital_twin.estimation_errors if result.digital_twin else None,
            output_dir=str(output_dir),
            prefix="dt_",
        )
        print(f"\nDT-driven run complete. Plots: {plots}")

    summary = {}
    if args.mode in ("single", "baseline"):
        summary = result.metrics.get_summary()
    else:
        summary = baseline.metrics.get_summary()

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Metrics summary saved to {summary_path}")


if __name__ == "__main__":
    main()
