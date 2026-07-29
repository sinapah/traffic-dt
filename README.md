# Traffic DT — Smart Traffic Monitoring Simulation

Discrete-event simulation of a smart traffic monitoring system with three cameras, three edge nodes, a federated learning coordinator, and a system-level Digital Twin (DT). Built with SimPy.

The simulation evaluates whether a DT can improve resource orchestration under dynamic workloads and telemetry outages over a 48-minute accelerated timeline (1 sim-min = 1 real-world hour, 2-day cycle).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# DT-driven run (no telemetry outage by default)
python -m src.main --mode dt

# DT-driven run with telemetry outage using KDE prediction
python -m src.main --mode dt --outage kde

# DT-driven run with telemetry outage using WGAN prediction
python -m src.main --mode dt --outage wgan

# Baseline run (no DT, no orchestrator)
python -m src.main --mode baseline

# Regenerate plots from saved telemetry data
python -m src.main --mode plot --input output/dt
python -m src.main --mode plot --input output/baseline
```

Outputs go to `output/`: PNG plots, metrics summary, full telemetry timeseries, and a config snapshot per run.

## Architecture

```
Camera 0 ──┐
Camera 1 ──┤──▶ Edge Nodes (queues + processing) ──▶ TelemetryBus ──▶ Digital Twin
Camera 2 ──┘         │                                    │              │
                     │                                    │         Orchestrator
              FL Coordinator ◀────────────────────────────┘              │
                     │                                                   │
                     └── local training ──────────────────────────▶ Edge Nodes
```

### Components

| File | Role |
|---|---|
| `config.py` | All simulation parameters as dataclasses, JSON I/O |
| `workload.py` | Time-dependent arrival rate schedule (night/rush-hour/midday pattern) |
| `queue.py` | Bounded FIFO queue with drop tracking and wait time measurement |
| `camera.py` | SimPy process generating frames as a time-varying Poisson process |
| `edge_node.py` | Per-edge queue processing, FL training, and telemetry emission |
| `fl_coordinator.py` | Periodic aggregation rounds with logistic convergence model |
| `telemetry.py` | Packet structures and publish-subscribe bus with outage toggle |
| `digital_twin.py` | State estimation, outage handling, recommendation generation |
| `prediction.py` | Pluggable prediction strategies (historical, KDE, WGAN interfaces) |
| `orchestrator.py` | Applies DT recommendations as parameter changes to edges |
| `metrics.py` | Centralized time-series metric collection with summary statistics |
| `simulation.py` | Top-level SimPy orchestrator wiring all components |
| `visualization.py` | Matplotlib plots with outage band highlighting |
| `comparison.py` | Baseline vs DT-driven side-by-side execution and summary |

## Queue Model

Each edge node runs an independent **G/G/1/∞/FCFS** queue:

```
Camera i ──▶ [Bounded Queue (cap=500)] ──▶ 1 Server (exponential service)
```

- **Arrivals:** Time-varying Poisson process (rate = base_fps × time-of-day multiplier)
- **Service:** Exponential distribution (mean = 1/service_rate)
- **Capacity:** 500 frames; excess arrivals are dropped
- **Servers:** One per edge — there is no pooled/cross-edge queue (not G/G/s)

Edge service rates are heterogeneous:

| Edge | Service Rate | Model |
|---|---|---|
| Edge 0 | 25 fps | Mid-range hardware |
| Edge 1 | 35 fps | High-end hardware |
| Edge 2 | 15 fps | Low-end hardware |

## Workload

The default schedule defines a 2-day cycle (48 sim-min) with time-varying multipliers applied to each camera's base frame rate (10 fps):

| Period | Sim Time | Real Hour | Multiplier | Effective fps |
|---|---|---|---|---|
| Night | 0–6, 24–30 | 00:00–06:00 | 0.2 | 2 |
| Morning rush | 6–9, 30–33 | 06:00–09:00 | 0.2→2.5 | 2→25 |
| Midday | 9–16, 33–40 | 09:00–16:00 | 1.2–2.5 | 12–25 |
| Evening rush | 16–19, 40–43 | 16:00–19:00 | 1.2→2.8 | 12→28 |
| Evening | 19–24, 43–48 | 19:00–24:00 | 0.2–0.5 | 2–5 |

Edge 2 (15 fps) becomes overloaded during rush hours (arrival rate ~28 fps), causing queue buildup and dropped frames.

## Federated Learning

- Operates on a configurable aggregation interval (default: every 5 sim-min)
- Each round: coordinator triggers local training on participating edges
- Training cost = `local_epochs × batch_size × dataset_size × cost_per_unit`
- Convergence modeled as a logistic curve: `ceiling × (1 - e^{-speed × round})`
- Configurable participation rate (default: 100%)
- **Resource contention:** During FL training, the edge's effective service rate is reduced by a configurable fraction (default: 50%), creating real competition for processing capacity

## Digital Twin

### What It Does

1. **Collects telemetry** from all edges via the TelemetryBus (queue length, utilization, processing rate, arrival rate, FL stats)
2. **Maintains state** — current snapshot of every edge and the aggregate system
3. **Records history** — timestamped state snapshots for later analysis
4. **Estimates state during outages** — uses a prediction strategy with weighted moving average and linear trend extrapolation to fill gaps
5. **Generates recommendations** — tiered parameter adjustments passed to the orchestrator
6. **Self-calibrates** — uses post-outage estimation error to adjust future predictions via bias correction

### What It Controls (Two Knobs)

| Parameter | Range | Effect |
|---|---|---|
| `local_epochs` | 1–10 | FL training intensity; more epochs = longer training = more resource contention |
| `sampling_rate` | 0.2–1.0 | Frame acceptance rate; lower = fewer frames processed = less queue pressure |

### Recommendation Logic

The DT checks each edge's utilization against tiered thresholds:
- If utilization > 1.0 (overloaded): set `sampling_rate` proportional to `service_rate / arrival_rate`, reduce `local_epochs` to minimum
- If utilization > target × 1.1 (mild overload): reduce `sampling_rate` by 20%, reduce `local_epochs` by 2
- If utilization < target × 0.7 (underloaded): increase `sampling_rate` by 10%, increase `local_epochs` by 1

### Telemetry Outages

Default config creates an outage during sim-min 18–28 (real-world hour 18–28). During outage:

1. Edges keep processing frames and FL continues normally
2. `TelemetryBus` stops delivering packets to the DT
3. DT generates synthetic state using the active prediction strategy
4. After outage ends, DT compares estimated vs actual and computes error

## Limitations and Known Issues

### 1. G/G/1 per edge, not G/G/s

Each edge has an independent single-server queue. There is no cross-edge pooling or load balancing. If G/G/s (shared queue with parallel servers) is needed, the queue architecture would need to be restructured.

### 2. Baseline vs Historical — important distinction

Two terms that sound similar but mean different things:

| Term | What it is | Applies to |
|---|---|---|
| **Baseline** | A **run mode** (`--mode baseline` or the internal comparison run). Runs the simulation **without a Digital Twin at all** — no telemetry estimation, no orchestration. Shows what happens naturally. | Entire simulation |
| **Historical** | A **prediction strategy** (`dt.prediction.strategy: "historical"`). Used *within* the DT to estimate telemetry during an outage using exponentially-weighted moving averages. | DT's outage estimator |

In other words: the baseline tells you "what would happen with no DT." The historical strategy tells you "how the DT fills gaps when it's active." They are not interchangeable.

### 3. Prediction strategies

The DT supports three pluggable prediction strategies set via `dt.prediction.strategy`:

| Strategy | Description |
|---|---|
| `historical` | Exponentially-weighted moving average with linear trend extrapolation and post-outage bias correction. Default. |
| `kde` | Multivariate Gaussian Kernel Density Estimation. Models the joint 5-dimensional distribution (queue_length, arrival_rate, utilization, processing_rate, mean_wait_time) per workload phase, preserving cross-metric correlations. Falls back to `historical` when fewer than `kde_min_samples` observations exist for the current phase. Requires only NumPy. |
| `wgan` | Wasserstein GAN with gradient penalty (WGAN-GP). Trains a small generator MLP (latent noise + phase one-hot → 5 metrics) online during normal operation, retraining every 5 sim-minutes. During outages, samples from the generator conditioned on the current workload phase. Falls back to `kde` → `historical` when training data are insufficient. Requires PyTorch. |

KDE and WGAN both condition predictions on the current **workload phase** (night / morning-ramp / morning-peak / midday / evening-ramp / evening), improving accuracy when outages span periods with different traffic intensities.

Key new config parameters under `dt.prediction`:

```json
{
  "strategy": "kde",
  "kde_min_samples": 5,
  "wgan_latent_dim": 8,
  "wgan_hidden_dim": 32,
  "wgan_train_epochs": 100,
  "wgan_lambda_gp": 10.0
}
```

## Configuration

The default config file is `config/default.json`. Key parameters:

```json
{
  "duration": 48.0,              // Simulation length in minutes
  "outages": [{"start": 18.0, "end": 28.0}],
  "fl": {
    "aggregation_interval": 5.0,
    "local_epochs": 2,
    "batch_size": 32,
    "cost_per_unit": 0.001
  },
  "dt": {
    "telemetry_interval": 0.5,
    "prediction": { "strategy": "historical", "history_window": 10 }
  },
  "orchestrator": {
    "target_utilization": 0.80,
    "adjustment_interval": 2.0
  }
}
```

## Outputs

Each run creates a subdirectory under `output/` with the following files:

```
output/
  baseline/                      # Baseline run (no DT)
    queue_length.png
    utilization.png
    latency.png
    throughput.png
    fl_convergence.png
    metrics_summary.json         # Aggregate stats (mean, max, p50, p95)
    metrics_timeseries.json      # Full (timestamp, value, tags) entries
    config.json                  # Config snapshot used for the run
  dt/                            # DT-driven run
    queue_length.png
    utilization.png
    latency.png
    throughput.png
    fl_convergence.png
    dt_estimation_error.png      # Only if outage was active
    metrics_summary.json
    metrics_timeseries.json
    dt_errors.json               # DT estimation error timeseries (outage only)
    config.json
    kde/                         # Strategy-specific distribution analysis
      distribution_histograms.png
      distribution_correlations.png
      distribution_joint.png
      distribution_errors.json
    wgan/
      ...
```

| Plot | What it shows |
|---|---|
| `queue_length.png` | Per-edge queue length over time |
| `utilization.png` | Per-edge load ratio (arrival_rate / service_rate) over time |
| `latency.png` | Mean frame wait time per edge |
| `throughput.png` | Processing rate per edge |
| `fl_convergence.png` | FL convergence metric over rounds |
| `dt_estimation_error.png` | DT estimation error during/after outage |
| `distribution_histograms.png` | Ground truth vs synthesized distribution overlay (per metric) |
| `distribution_correlations.png` | Ground truth vs synthesized correlation matrices |
| `distribution_joint.png` | Joint-distribution scatter plots for metric pairs |
| `distribution_errors.json` | Per-metric MAE, RMSE, and correlation fidelity score |
| `metrics_summary.json` | Aggregate stats (mean, max, p50, p95) for all metrics |

All plots include red shaded bands marking telemetry outage periods.
