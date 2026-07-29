from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import SimulationConfig
from src.metrics import MetricsCollector
from src.simulation import Simulation
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


def _save_telemetry_data(
    metrics: MetricsCollector,
    dt_errors: list[tuple[float, float]] | None,
    config: SimulationConfig,
    output_dir: Path,
) -> None:
    ts_path = output_dir / "metrics_timeseries.json"
    with open(ts_path, "w") as f:
        json.dump(metrics.to_dicts(), f, indent=2, default=str)
    log.info("Telemetry timeseries saved to %s", ts_path)

    if dt_errors is not None:
        de_path = output_dir / "dt_errors.json"
        with open(de_path, "w") as f:
            json.dump(dt_errors, f, indent=2)
        log.info("DT estimation errors saved to %s", de_path)

    cfg_path = output_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    log.info("Config snapshot saved to %s", cfg_path)


def _run_plot_mode(input_dir_str: str) -> None:
    input_dir = Path(input_dir_str)
    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    cfg_path = input_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = SimulationConfig.from_dict(json.load(f))
        outages = config.outages
    else:
        log.warning("No config.json found — no outage bands on plots")
        outages = None

    ts_path = input_dir / "metrics_timeseries.json"
    if not ts_path.exists():
        log.error("No metrics_timeseries.json found in %s", input_dir)
        sys.exit(1)
    with open(ts_path) as f:
        metrics = MetricsCollector.from_dicts(json.load(f))
    log.info("Loaded %d telemetry entries", len(metrics.entries))

    de_path = input_dir / "dt_errors.json"
    dt_errors = None
    if de_path.exists():
        with open(de_path) as f:
            dt_errors = [(t, e) for t, e in json.load(f)]
        log.info("Loaded %d DT estimation error entries", len(dt_errors))

    plots = plot_all(
        metrics,
        outages=outages,
        dt_errors=dt_errors,
        output_dir=str(input_dir),
    )
    print(f"\nPlots regenerated in {input_dir}: {plots}")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Traffic DT Simulation")
    parser.add_argument("--config", type=str, default="config/default.json", help="Config file path")
    parser.add_argument(
        "--mode",
        choices=["baseline", "dt", "plot"],
        default="dt",
        help="Run mode: baseline (no DT), dt (DT-driven), plot (regenerate plots from saved data)",
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--outage",
        choices=["kde", "wgan"],
        default=None,
        help="Outage mode for dt: enables telemetry outage with specified prediction strategy",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory with saved telemetry data (required for --mode plot)",
    )
    args = parser.parse_args()

    if args.mode == "plot":
        if not args.input:
            parser.error("--input is required with --mode plot")
        _run_plot_mode(args.input)
        return

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

    if args.mode == "baseline":
        if args.outage is not None:
            parser.error("--outage is only valid with --mode dt")
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
        _save_telemetry_data(result.metrics, None, config, base_dir)
        print(f"\nBaseline run complete. Plots: {plots}")
    else:
        if args.outage is not None:
            config.dt.prediction.strategy = args.outage
        else:
            config.outages = []
        log.info("Mode: dt (DT-driven only)")
        sim = Simulation(config)
        result = sim.run(dt_enabled=True)
        dt_dir = output_dir / "dt"
        dt_errors = result.digital_twin.estimation_errors if result.digital_twin else None
        plots = plot_all(
            result.metrics,
            outages=config.outages,
            dt_errors=dt_errors,
            output_dir=str(dt_dir),
        )
        _save_summary(result.metrics.get_summary(), dt_dir / "metrics_summary.json")
        _save_telemetry_data(result.metrics, dt_errors, config, dt_dir)
        print(f"\nDT-driven run complete. Plots: {plots}")


if __name__ == "__main__":
    main()
