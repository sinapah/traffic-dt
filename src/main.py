from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import SimulationConfig
from src.simulation import Simulation
from src.comparison import compare
from src.visualization import plot_all

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Metrics summary saved to {path}")


def main() -> None:
    _configure_logging()
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
        log.info("Config loaded from %s", config_path)
    else:
        log.warning("Config file not found: %s — using defaults", config_path)
        config = SimulationConfig()

    if args.seed is not None:
        config.seed = args.seed

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    if args.mode == "compare":
        log.info("Mode: compare (baseline + DT-driven)")
        compare(config, output_dir=str(output_dir))
    elif args.mode == "baseline":
        log.info("Mode: baseline")
        from src.comparison import run_baseline
        result = run_baseline(config)
        base_dir = output_dir / "baseline"
        plots = plot_all(
            result.metrics,
            outages=config.outages,
            output_dir=str(base_dir),
        )
        _save_summary(result.metrics.get_summary(), base_dir / "metrics_summary.json")
        print(f"\nBaseline run complete. Plots: {plots}")
    else:
        log.info("Mode: single (DT-driven only)")
        sim = Simulation(config)
        result = sim.run(dt_enabled=True)
        dt_dir = output_dir / "dt"
        plots = plot_all(
            result.metrics,
            outages=config.outages,
            dt_errors=result.digital_twin.estimation_errors if result.digital_twin else None,
            output_dir=str(dt_dir),
        )
        _save_summary(result.metrics.get_summary(), dt_dir / "metrics_summary.json")
        print(f"\nDT-driven run complete. Plots: {plots}")


if __name__ == "__main__":
    main()
