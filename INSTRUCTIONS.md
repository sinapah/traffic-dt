# Task: Fix DT Effectiveness in Traffic-DT Simulation

You are working on a discrete-event simulation in `/home/ubuntu/traffic-dt`. The codebase is functional and runs end-to-end, but the Digital Twin (DT) currently has no observable effect — baseline and DT-driven configurations produce identical results. Your job is to fix three root-cause issues so the DT demonstrably improves system performance, and to improve the predictor used during telemetry outages.

Read `README.md` and `agents.md` for full project context before starting.

### Background

The simulation has three cameras, three edges (service rates 25/35/15 fps), a federated learning coordinator, a DT, and an orchestrator. During rush hours, Edge 2 (15 fps) should be overloaded (arrival rate ~28 fps). The DT should detect this and recommend reducing `sampling_rate` to prevent queue overflow. The orchestrator applies these recommendations. During telemetry outages, the DT estimates state using a predictor.

Currently this doesn't work because of three interlocking issues.

---

### Fix 1: Utilization metric is wrong

**Problem:** `queue.py:32` defines `utilization = len(queue) / capacity`. With capacity=500 and typical queue length ~50, utilization peaks at ~10%. The DT's threshold is `target_utilization * 1.1 = 0.88`, so it never triggers.

**Fix:** Compute utilization as `arrival_rate / service_rate` — the true load ratio. This requires:

1. **`src/queue.py`** — Remove the `utilization` property from `BoundedQueue` (it's the wrong abstraction at the queue level).

2. **`src/edge_node.py`** — Add a proper `utilization` property on `EdgeNode` that computes `self._arrival_rate / self.service_rate`, clamped to [0.0, 1.0+]. This uses the already-tracked `_arrival_rate` from `_update_rates()`. The `EdgeNode.utilization` property at line 47 currently delegates to `self.queue.utilization` — replace it.

3. **`src/edge_node.py:101`** — The telemetry packet currently sets `utilization=self.queue.utilization`. Change to `utilization=self.utilization` (the new EdgeNode property).

4. **`src/simulation.py:110`** — The metrics collector records `edge.queue.utilization`. Change to `edge.utilization`.

5. **`src/digital_twin.py:86-98`** — `_update_current_state` builds `EdgeState` from telemetry packets. The `utilization` field in `EdgeState` will now carry the correct load-ratio value from the packet. No change needed here, but verify it works.

**Expected result:** Edge 2 should show utilization >1.0 during rush hours (arrival rate 28 fps / service rate 15 fps = 1.87). Edge 0 and Edge 1 should stay below 1.0. The DT should now trigger the "overloaded" path for Edge 2.

---

### Fix 2: No resource contention between processing and FL

**Problem:** `_process_loop` and `_training_loop` in `edge_node.py` run as independent SimPy processes. FL training does not reduce processing capacity. Adjusting `local_epochs` has zero effect on frame throughput.

**Fix:** When FL training is active, the processing loop should slow down. The simplest model: during training, the effective service rate is reduced by a configurable fraction (e.g., 50%).

1. **`src/config.py`** — Add a field to `FLConfig`:
   ```python
   training_resource_contention: float = 0.5  # 0.0 = no contention, 1.0 = fully blocked
   ```

2. **`src/edge_node.py`** — Modify `_process_loop` (line 65). When `self._training_active` is True, use a diluted service rate:
   ```python
   effective_rate = self.service_rate * (1.0 - fl_config.training_resource_contention)
   ```
   Pass the `fl_config` into the edge node (store it in `__init__` or pass it when training starts). Use `effective_rate` instead of `self.service_rate` for the exponential service time when training is active.

3. **`src/config/default.json`** and **`src/config/baseline.json`** — Add `"training_resource_contention": 0.5` to the `fl` section.

**Expected result:** When FL training fires (every 5 sim-min), Edge 2's effective service rate drops from 15 to 7.5 fps while arrival rate is ~28 fps. Queue builds rapidly during training windows. The DT, seeing high utilization, should recommend reducing `sampling_rate` on Edge 2. This creates a visible difference between baseline (no adaptation) and DT-driven (adaptation).

---

### Fix 3: DT recommendation thresholds need tuning

**Problem:** Even with correct utilization, the current thresholds (`> target*1.1` to reduce, `< target*0.7` to increase) may be too wide. With `target_utilization=0.80`, the "reduce" threshold is 0.88. Edge 2 will exceed this during rush hours, but the magnitude of the response (reduce `sampling_rate` by 10%, reduce `local_epochs` by 1) may be too gradual.

**Fix:** Make the DT respond more aggressively when utilization is critically high.

1. **`src/digital_twin.py:182-204`** — Rework `get_recommendation()`. Replace the single threshold with a tiered response:
   - utilization > 1.0 (overloaded): set `sampling_rate` proportional to `service_rate / arrival_rate` (hard cap to prevent overflow), reduce `local_epochs` to minimum
   - utilization > target * 1.1 (mild overload): reduce `sampling_rate` by 20%, reduce `local_epochs` by 2
   - utilization < target * 0.7 (underloaded): increase `sampling_rate` by 10%, increase `local_epochs` by 1

2. **`src/digital_twin.py`** — Also use the *current* `arrival_rate` from the edge state (not just utilization) when deciding how aggressively to cut sampling rate. If arrival rate is 28 fps and service rate is 15 fps, the DT should recommend `sampling_rate ≈ 15/28 ≈ 0.54` to balance the queue.

**Expected result:** During rush hours, the DT should reduce Edge 2's `sampling_rate` from 1.0 to ~0.5-0.6. This prevents dropped frames (at the cost of deliberately skipping some frames). The baseline run should show dropped frames; the DT-driven run should show fewer or zero drops but lower throughput on Edge 2 during peaks.

---

### Fix 4: Improve the historical predictor

**Problem:** `HistoricalPredictor` in `prediction.py:29-57` computes a flat rolling average of the last N telemetry packets. During an outage, if arrival rate is rising (entering rush hour), the estimate will lag badly — it predicts the past average, not the current trajectory.

**Fix:** Use a weighted moving average that gives more weight to recent observations, plus a simple linear trend extrapolation.

1. **`src/prediction.py`** — Rewrite `HistoricalPredictor.predict_edge_state()`:
   - Compute a weighted moving average: weight `i` for the `i`-th most recent packet (newer = higher weight). Use exponential weights: `w_i = alpha^(N-1-i)` where `alpha` is ~0.8.
   - Compute a simple linear trend from the last `min(window, len(history))` observations of `queue_length` and `arrival_rate`. Extrapolate the trend forward by estimating the time gap between the last observation and `current_time`.
   - Combine: `predicted_value = weighted_avg + trend_extrapolation`

2. **`src/config.py`** — Add to `PredictionConfig`:
   ```python
   trend_weight: float = 0.3  # 0.0 = flat average only, 1.0 = full trend extrapolation
   ```

**Expected result:** During the outage (min 18-28), if the workload is entering a rush hour, the predictor should produce rising estimates rather than flat ones. The estimation error (plotted in `dt_dt_estimation_error.png`) should decrease compared to the current flat-average approach.

---

### Fix 5: Use estimation error for self-calibration

**Problem:** `digital_twin.py:138-146` computes post-outage estimation error but discards it. The predictor never learns.

**Fix:** After an outage ends, compute the error and use it to adjust the predictor's behavior for the next outage.

1. **`src/prediction.py`** — Add a `calibrate(actual: EdgeState, estimated: EdgeState, dt: float)` method to `PredictionStrategy`. The base implementation is a no-op.

2. **`src/prediction.py`** — In `HistoricalPredictor`, implement `calibrate()` to store a `bias_correction` factor per metric (e.g., if estimates were systematically 20% low, multiply future predictions by 1.2). Use exponential moving average of past errors:
   ```python
   self.bias_correction = 0.9 * self.bias_correction + 0.1 * (actual / estimated)
   ```
   Apply this correction in `predict_edge_state()`.

3. **`src/digital_twin.py`** — In `_compute_estimation_error()`, after computing errors, call `self.predictor.calibrate(actual_state, estimated_state, ...)` for each edge.

**Expected result:** If a second outage occurs later in the simulation, the predictor should produce better estimates because it has learned from the first outage's error. You can test this by adding a second outage to the config (e.g., `{"start": 38.0, "end": 44.0}`).

---

### Verification

After making all fixes, run:

```bash
source .venv/bin/activate
python -m src.main --mode compare --output output
```

The comparison output should now show **different results** for baseline vs DT-driven. Specifically:

| Metric | Baseline (expected) | DT-driven (expected) |
|---|---|---|
| Edge 2 peak queue length | High (100+), dropped frames | Lower, fewer/no drops |
| Edge 2 mean dropped_frames | > 0 | ~0 |
| Edge 2 mean sampling_rate | 1.0 (fixed) | < 1.0 during peaks |
| Edge 2 peak utilization | > 1.5 (overloaded) | Reduced during peaks |
| FL convergence | Same | Same (FL itself is unaffected) |
| DT estimation error | N/A | Lower with improved predictor |

Also verify:
- `output/baseline_queue_length.png` should show Edge 2's queue spiking during rush hours
- `output/dt_queue_length.png` should show Edge 2's queue staying more controlled
- `output/dt_dt_estimation_error.png` should show lower error than before
- The `metrics_summary.json` should show different values for baseline vs DT

Update `README.md` to reflect the fixes — remove or update the "Limitations" section to mark fixes 1-5 as resolved.
