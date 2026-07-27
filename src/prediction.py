from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from src.telemetry import EdgeState, SystemState, TelemetryPacket

if TYPE_CHECKING:
    from src.config import WorkloadPhase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Six coarse workload phases derived from the default 24-hour schedule:
#   0 = Night         (real hours  0– 6)  multiplier ~0.2
#   1 = Morning ramp  (real hours  6– 9)  multiplier 0.2→2.5
#   2 = Morning peak  (real hours  9–11)  multiplier ~2.5
#   3 = Midday        (real hours 11–16)  multiplier 1.2–2.5
#   4 = Evening ramp  (real hours 16–19)  multiplier 1.2→2.8
#   5 = Evening       (real hours 19–24)  multiplier 2.8→0.2
_NUM_PHASES = 6
_PHASE_BOUNDARIES = [0.0, 6.0, 9.0, 11.0, 16.0, 19.0, 24.0]  # hour boundaries

# Metric names used throughout this module
_METRIC_NAMES = [
    "queue_length",
    "arrival_rate",
    "utilization",
    "processing_rate",
    "mean_wait_time",
]

# ---------------------------------------------------------------------------
# Phase helper
# ---------------------------------------------------------------------------


def _get_phase(sim_time: float, duration: float, num_phases: int = _NUM_PHASES) -> int:
    """Map *sim_time* to a coarse workload-phase index in [0, num_phases).

    The simulation cycle length is *duration* (default 48 sim-min = 2 real days).
    We fold sim_time into a 24-hour clock by taking ``sim_time % 24`` and then
    bucket by the standard phase boundaries defined above.  If *num_phases* is
    different from 6 we fall back to uniform bucketing.
    """
    hour = sim_time % 24.0  # fold into a single 24-hour cycle
    if num_phases == _NUM_PHASES:
        for i in range(len(_PHASE_BOUNDARIES) - 1):
            if hour < _PHASE_BOUNDARIES[i + 1]:
                return i
        return num_phases - 1
    # Fallback: uniform bucketing
    bucket_size = 24.0 / num_phases
    return min(int(hour / bucket_size), num_phases - 1)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PredictionStrategy(ABC):
    @abstractmethod
    def predict_edge_state(
        self,
        edge_id: int,
        history: list[TelemetryPacket],
        current_time: float,
    ) -> EdgeState:
        ...

    @abstractmethod
    def predict_system_state(
        self,
        edge_histories: dict[int, list[TelemetryPacket]],
        current_time: float,
        fl_round: int,
        fl_convergence: float,
    ) -> SystemState:
        ...

    def calibrate(self, actual: EdgeState, estimated: EdgeState, dt: float) -> None:
        pass


# ---------------------------------------------------------------------------
# HistoricalPredictor  (unchanged)
# ---------------------------------------------------------------------------


class HistoricalPredictor(PredictionStrategy):
    def __init__(self, window: int = 10, trend_weight: float = 0.3):
        self.window = window
        self.trend_weight = trend_weight
        self.alpha = 0.8
        self.bias_correction: dict[str, float] = {
            "queue_length": 1.0,
            "utilization": 1.0,
            "processing_rate": 1.0,
            "arrival_rate": 1.0,
            "mean_wait_time": 1.0,
        }

    def _weighted_avg(self, values: list[float]) -> float:
        n = len(values)
        if n == 0:
            return 0.0
        weights = [self.alpha ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / total_w if total_w > 0 else 0.0

    def _linear_trend(self, values: list[float], dt: float) -> float:
        n = len(values)
        if n < 2 or dt <= 0:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        slope = num / den
        return slope * dt * self.trend_weight

    def predict_edge_state(
        self,
        edge_id: int,
        history: list[TelemetryPacket],
        current_time: float,
    ) -> EdgeState:
        recent = history[-self.window:] if history else []
        if not recent:
            return EdgeState(edge_id=edge_id)

        time_gap = current_time - recent[-1].timestamp if recent else 0.0

        queue_lengths = [float(p.queue_length) for p in recent]
        arrival_rates = [p.arrival_rate for p in recent]
        utilizations = [p.utilization for p in recent]
        processing_rates = [p.processing_rate for p in recent]
        wait_times = [p.mean_wait_time for p in recent]

        ql_wa = self._weighted_avg(queue_lengths)
        ar_wa = self._weighted_avg(arrival_rates)
        ut_wa = self._weighted_avg(utilizations)
        pr_wa = self._weighted_avg(processing_rates)
        wt_wa = self._weighted_avg(wait_times)

        ql_trend = self._linear_trend(queue_lengths, time_gap)
        ar_trend = self._linear_trend(arrival_rates, time_gap)

        predicted_queue = max(0, int((ql_wa + ql_trend) * self.bias_correction["queue_length"]))
        predicted_arrival = max(0.0, (ar_wa + ar_trend) * self.bias_correction["arrival_rate"])
        predicted_util = max(0.0, ut_wa * self.bias_correction["utilization"])
        predicted_processing = max(0.0, pr_wa * self.bias_correction["processing_rate"])
        predicted_wait = max(0.0, wt_wa * self.bias_correction["mean_wait_time"])

        return EdgeState(
            edge_id=edge_id,
            queue_length=predicted_queue,
            queue_capacity=recent[-1].queue_capacity,
            utilization=predicted_util,
            processing_rate=predicted_processing,
            arrival_rate=predicted_arrival,
            total_processed=recent[-1].total_processed,
            total_dropped=recent[-1].total_dropped,
            mean_wait_time=predicted_wait,
            training_active=False,
            local_epochs=recent[-1].local_epochs,
            sampling_rate=recent[-1].sampling_rate,
        )

    def predict_system_state(
        self,
        edge_histories: dict[int, list[TelemetryPacket]],
        current_time: float,
        fl_round: int,
        fl_convergence: float,
    ) -> SystemState:
        edge_states = {}
        total_arrival = 0.0
        total_processing = 0.0
        total_queue = 0
        total_dropped = 0

        for edge_id, history in edge_histories.items():
            state = self.predict_edge_state(edge_id, history, current_time)
            edge_states[edge_id] = state
            total_arrival += state.arrival_rate
            total_processing += state.processing_rate
            total_queue += state.queue_length
            total_dropped += state.total_dropped

        return SystemState(
            timestamp=current_time,
            edges=edge_states,
            fl_round=fl_round,
            fl_convergence=fl_convergence,
            active_participants=len(edge_states),
            aggregate_arrival_rate=total_arrival,
            aggregate_processing_rate=total_processing,
            aggregate_queue_length=total_queue,
            aggregate_dropped=total_dropped,
        )

    def calibrate(self, actual: EdgeState, estimated: EdgeState, dt: float) -> None:
        for key, actual_val, est_val in [
            ("queue_length", float(actual.queue_length), float(estimated.queue_length)),
            ("utilization", actual.utilization, estimated.utilization),
            ("processing_rate", actual.processing_rate, estimated.processing_rate),
            ("arrival_rate", actual.arrival_rate, estimated.arrival_rate),
            ("mean_wait_time", actual.mean_wait_time, estimated.mean_wait_time),
        ]:
            if est_val > 0.01:
                ratio = actual_val / est_val
                self.bias_correction[key] = 0.9 * self.bias_correction[key] + 0.1 * ratio


# ---------------------------------------------------------------------------
# KDE helpers
# ---------------------------------------------------------------------------


class _MultiVariateKDE:
    """Multivariate Gaussian KDE (d=5) using only NumPy.

    Models the joint 5-dimensional distribution of all telemetry metrics,
    preserving cross-metric correlations (e.g. high arrival_rate ↔ high
    queue_length).

    Bandwidth selection: multivariate Scott's rule —
        H = n^(-2/(d+4)) * cov(data)
    where d=5 and cov(data) is the empirical covariance matrix.

    Sampling: pick a random observation then add Gaussian noise with
    covariance H (Silverman's sample-smooth approach).
    """

    def __init__(self) -> None:
        self._obs: list[np.ndarray] = []

    def add(self, metrics: dict[str, float]) -> None:
        vec = np.array([metrics[m] for m in _METRIC_NAMES])
        self._obs.append(vec)

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        n = len(self._obs)
        if n == 0:
            return {m: 0.0 for m in _METRIC_NAMES}

        idx = int(rng.integers(n))
        obs = self._obs[idx]

        if n >= 2:
            d = len(_METRIC_NAMES)
            arr = np.array(self._obs)
            cov = np.cov(arr, rowvar=False)
            cov += 1e-6 * np.eye(d)
            h = n ** (-1.0 / (d + 4))
            noise = rng.multivariate_normal(np.zeros(d), h ** 2 * cov)
            vals = obs + noise
        else:
            vals = obs

        return {m: max(0.0, float(vals[i])) for i, m in enumerate(_METRIC_NAMES)}

    @property
    def n(self) -> int:
        return len(self._obs)


class _PhaseKDEStore:
    """Per-phase multivariate KDE storage for one edge.

    Internally structured as ``{phase_idx: _MultiVariateKDE}``.
    Unlike the previous per-metric 1-D KDE approach, this models the
    joint 5-dimensional distribution of all telemetry metrics, preserving
    cross-metric correlations.

    Observations are ingested with :meth:`add_observation`; the predictor
    calls :meth:`can_sample` to gate whether it has enough data, and
    :meth:`sample` to draw a full metric dict for a given phase.
    """

    def __init__(self, num_phases: int = _NUM_PHASES, min_samples: int = 5) -> None:
        self._num_phases = num_phases
        self._min_samples = min_samples
        self._kdes: dict[int, _MultiVariateKDE] = {}
        self._seen_timestamps: dict[int, set[float]] = {}

    def _ensure_phase(self, phase: int) -> _MultiVariateKDE:
        if phase not in self._kdes:
            self._kdes[phase] = _MultiVariateKDE()
            self._seen_timestamps[phase] = set()
        return self._kdes[phase]

    def add_observation(
        self,
        phase: int,
        timestamp: float,
        metrics: dict[str, float],
    ) -> None:
        seen = self._seen_timestamps.get(phase)
        if seen is None:
            self._ensure_phase(phase)
            seen = self._seen_timestamps[phase]
        if timestamp in seen:
            return
        seen.add(timestamp)
        self._kdes[phase].add(metrics)

    def can_sample(self, phase: int) -> bool:
        kde = self._kdes.get(phase)
        return kde is not None and kde.n >= self._min_samples

    def sample(self, phase: int, rng: np.random.Generator) -> dict[str, float]:
        return self._kdes[phase].sample(rng)


# ---------------------------------------------------------------------------
# KDEPredictor
# ---------------------------------------------------------------------------


class KDEPredictor(PredictionStrategy):
    """Synthesises telemetry by sampling from per-phase Gaussian KDEs.

    During normal (non-outage) operation the predictor passively ingests
    telemetry packets into phase-partitioned KDE stores.  When called during
    an outage it samples from the KDE for the current workload phase, falling
    back to :class:`HistoricalPredictor` if insufficient data exist for that
    phase.

    Time-of-day conditioning
    ------------------------
    Packets are tagged with a workload phase (0–5) derived from their
    timestamp via :func:`_get_phase`.  During prediction the same function
    maps *current_time* to a phase so the synthesised values reflect the
    expected traffic intensity at that time of day.

    Bias correction
    ---------------
    Inherits the same post-outage calibration mechanism as
    :class:`HistoricalPredictor`: the :meth:`calibrate` method adjusts per-
    metric multiplicative bias-correction factors via exponential smoothing.
    """

    def __init__(
        self,
        window: int = 10,
        trend_weight: float = 0.3,
        duration: float = 48.0,
        min_samples: int = 5,
        seed: int | None = None,
    ) -> None:
        self.window = window
        self._duration = duration
        self._fallback = HistoricalPredictor(window, trend_weight)
        self._rng = np.random.default_rng(seed)
        self._stores: dict[int, _PhaseKDEStore] = {}
        self.bias_correction: dict[str, float] = {m: 1.0 for m in _METRIC_NAMES}
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store(self, edge_id: int) -> _PhaseKDEStore:
        if edge_id not in self._stores:
            self._stores[edge_id] = _PhaseKDEStore(
                num_phases=_NUM_PHASES, min_samples=self._min_samples
            )
        return self._stores[edge_id]

    def _ingest(self, edge_id: int, history: list[TelemetryPacket]) -> None:
        """Feed history packets into the KDE store (idempotent via timestamp dedup)."""
        store = self._store(edge_id)
        for pkt in history:
            phase = _get_phase(pkt.timestamp, self._duration)
            store.add_observation(
                phase,
                pkt.timestamp,
                {
                    "queue_length": float(pkt.queue_length),
                    "arrival_rate": pkt.arrival_rate,
                    "utilization": pkt.utilization,
                    "processing_rate": pkt.processing_rate,
                    "mean_wait_time": pkt.mean_wait_time,
                },
            )

    def _build_edge_state(
        self,
        edge_id: int,
        sampled: dict[str, float],
        reference_packet: TelemetryPacket,
    ) -> EdgeState:
        """Clamp sampled values, apply bias correction, return EdgeState."""
        bc = self.bias_correction
        return EdgeState(
            edge_id=edge_id,
            queue_length=max(0, int(sampled["queue_length"] * bc["queue_length"])),
            queue_capacity=reference_packet.queue_capacity,
            utilization=max(0.0, sampled["utilization"] * bc["utilization"]),
            processing_rate=max(0.0, sampled["processing_rate"] * bc["processing_rate"]),
            arrival_rate=max(0.0, sampled["arrival_rate"] * bc["arrival_rate"]),
            total_processed=reference_packet.total_processed,
            total_dropped=reference_packet.total_dropped,
            mean_wait_time=max(0.0, sampled["mean_wait_time"] * bc["mean_wait_time"]),
            training_active=False,
            local_epochs=reference_packet.local_epochs,
            sampling_rate=reference_packet.sampling_rate,
        )

    # ------------------------------------------------------------------
    # PredictionStrategy interface
    # ------------------------------------------------------------------

    def predict_edge_state(
        self,
        edge_id: int,
        history: list[TelemetryPacket],
        current_time: float,
    ) -> EdgeState:
        # Always ingest whatever history we have (duplicate-safe)
        self._ingest(edge_id, history)

        phase = _get_phase(current_time, self._duration)
        store = self._store(edge_id)

        if store.can_sample(phase):
            sampled = store.sample(phase, self._rng)
            ref = history[-1] if history else None
            if ref is not None:
                return self._build_edge_state(edge_id, sampled, ref)

        # Fallback
        return self._fallback.predict_edge_state(edge_id, history, current_time)

    def predict_system_state(
        self,
        edge_histories: dict[int, list[TelemetryPacket]],
        current_time: float,
        fl_round: int,
        fl_convergence: float,
    ) -> SystemState:
        edge_states: dict[int, EdgeState] = {}
        total_arrival = 0.0
        total_processing = 0.0
        total_queue = 0
        total_dropped = 0

        for edge_id, history in edge_histories.items():
            state = self.predict_edge_state(edge_id, history, current_time)
            edge_states[edge_id] = state
            total_arrival += state.arrival_rate
            total_processing += state.processing_rate
            total_queue += state.queue_length
            total_dropped += state.total_dropped

        return SystemState(
            timestamp=current_time,
            edges=edge_states,
            fl_round=fl_round,
            fl_convergence=fl_convergence,
            active_participants=len(edge_states),
            aggregate_arrival_rate=total_arrival,
            aggregate_processing_rate=total_processing,
            aggregate_queue_length=total_queue,
            aggregate_dropped=total_dropped,
        )

    def calibrate(self, actual: EdgeState, estimated: EdgeState, dt: float) -> None:
        pairs = [
            ("queue_length", float(actual.queue_length), float(estimated.queue_length)),
            ("utilization", actual.utilization, estimated.utilization),
            ("processing_rate", actual.processing_rate, estimated.processing_rate),
            ("arrival_rate", actual.arrival_rate, estimated.arrival_rate),
            ("mean_wait_time", actual.mean_wait_time, estimated.mean_wait_time),
        ]
        for key, actual_val, est_val in pairs:
            if est_val > 0.01:
                ratio = actual_val / est_val
                self.bias_correction[key] = 0.9 * self.bias_correction[key] + 0.1 * ratio
        # Propagate to fallback as well (used when KDE data is insufficient)
        self._fallback.calibrate(actual, estimated, dt)


# ---------------------------------------------------------------------------
# WGAN building blocks  (PyTorch)
# ---------------------------------------------------------------------------


def _try_import_torch():
    """Lazy import so KDE/historical strategies do not require PyTorch."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the WGAN predictor. "
            "Install it with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


class _Generator:
    """Wraps a PyTorch MLP: latent noise + phase one-hot → 5 metrics (all ≥ 0)."""

    def __init__(self, latent_dim: int, num_phases: int, hidden_dim: int) -> None:
        torch, nn = _try_import_torch()
        self._torch = torch
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_phases, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, len(_METRIC_NAMES)),
            nn.Softplus(),  # smooth non-negative activation (better gradients than ReLU)
        )
        self.latent_dim = latent_dim
        self.num_phases = num_phases

    def parameters(self):
        return self.net.parameters()

    def __call__(self, z, phase_oh):
        return self.net(self._torch.cat([z, phase_oh], dim=1))

    def eval(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class _Critic:
    """Wraps a PyTorch MLP: 5 metrics + phase one-hot → scalar (no sigmoid)."""

    def __init__(self, num_phases: int, hidden_dim: int) -> None:
        torch, nn = _try_import_torch()
        self._torch = torch
        self.net = nn.Sequential(
            nn.Linear(len(_METRIC_NAMES) + num_phases, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.num_phases = num_phases

    def parameters(self):
        return self.net.parameters()

    def __call__(self, x, phase_oh):
        return self.net(self._torch.cat([x, phase_oh], dim=1))

    def train_mode(self):
        self.net.train()


class _WGANTrainer:
    """Manages WGAN-GP training and sampling for a single edge node.

    Architecture
    ------------
    - Generator  G : z ∈ ℝ^latent_dim, phase ∈ one-hot^num_phases → x ∈ ℝ^5
    - Critic     D : x ∈ ℝ^5, phase ∈ one-hot^num_phases           → score ∈ ℝ

    Training
    --------
    Uses the Wasserstein-1 distance with gradient penalty (WGAN-GP, Gulrajani
    et al. 2017).  For each generator update we run *n_critic* = 5 critic
    updates.  Both networks use Adam(lr=1e-3, betas=(0.5, 0.9)).

    Normalisation
    -------------
    Each metric column is z-score normalised before training; the generator
    output is un-normalised before being returned to the caller.
    """

    _N_CRITIC = 5  # critic steps per generator step

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_phases: int,
        lambda_gp: float,
    ) -> None:
        torch, nn = _try_import_torch()
        self._torch = torch
        self._nn = nn
        self._device = torch.device("cpu")

        self._latent_dim = latent_dim
        self._num_phases = num_phases
        self._lambda_gp = lambda_gp

        self._gen = _Generator(latent_dim, num_phases, hidden_dim)
        self._critic = _Critic(num_phases, hidden_dim)

        self._gen.net.to(self._device)
        self._critic.net.to(self._device)

        self._opt_g = torch.optim.Adam(
            self._gen.parameters(), lr=1e-3, betas=(0.5, 0.9)
        )
        self._opt_d = torch.optim.Adam(
            self._critic.parameters(), lr=1e-3, betas=(0.5, 0.9)
        )

        # Normalisation stats: {phase: (mean_vec, std_vec)} both np.ndarray shape (5,)
        self._norm: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Which phases have been successfully trained
        self._trained_phases: set[int] = set()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _fit_normalise(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-6] = 1.0  # avoid division by zero for constant columns
        return mean, std

    def _normalise(self, data: np.ndarray, phase: int) -> np.ndarray:
        mean, std = self._norm[phase]
        return (data - mean) / std

    def _unnormalise(self, data: np.ndarray, phase: int) -> np.ndarray:
        mean, std = self._norm[phase]
        return data * std + mean

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP)
    # ------------------------------------------------------------------

    def _gradient_penalty(self, real: "torch.Tensor", fake: "torch.Tensor", phase_oh: "torch.Tensor") -> "torch.Tensor":
        torch = self._torch
        bsz = real.size(0)
        alpha = torch.rand(bsz, 1, device=self._device)
        interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        d_interp = self._critic(interpolated, phase_oh)
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        phase_data: dict[int, np.ndarray],
        epochs: int,
        batch_size: int,
    ) -> None:
        """Train on *phase_data* = ``{phase: ndarray of shape (N, 5)}``."""
        torch = self._torch
        for phase, raw in phase_data.items():
            n = len(raw)
            if n < batch_size:
                continue  # not enough data for this phase

            mean, std = self._fit_normalise(raw)
            self._norm[phase] = (mean, std)
            normed = self._normalise(raw, phase)

            data_t = torch.tensor(normed, dtype=torch.float32, device=self._device)
            phase_oh = torch.zeros(1, self._num_phases, device=self._device)
            phase_oh[0, phase] = 1.0

            self._gen.train_mode()
            self._critic.train_mode()

            step = 0
            for _ in range(epochs):
                # --- Critic update (N_CRITIC times) ---
                for _ in range(self._N_CRITIC):
                    idx = torch.randint(0, n, (batch_size,))
                    real = data_t[idx]
                    ph_batch = phase_oh.expand(batch_size, -1)

                    z = torch.randn(batch_size, self._latent_dim, device=self._device)
                    fake = self._gen(z, ph_batch).detach()

                    d_real = self._critic(real, ph_batch).mean()
                    d_fake = self._critic(fake, ph_batch).mean()
                    gp = self._gradient_penalty(real, fake.requires_grad_(False), ph_batch)

                    loss_d = d_fake - d_real + self._lambda_gp * gp
                    self._opt_d.zero_grad()
                    loss_d.backward()
                    self._opt_d.step()

                # --- Generator update ---
                z = torch.randn(batch_size, self._latent_dim, device=self._device)
                ph_batch = phase_oh.expand(batch_size, -1)
                fake = self._gen(z, ph_batch)
                loss_g = -self._critic(fake, ph_batch).mean()
                self._opt_g.zero_grad()
                loss_g.backward()
                self._opt_g.step()

                step += 1

            self._trained_phases.add(phase)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def can_sample(self, phase: int) -> bool:
        return phase in self._trained_phases

    def sample(self, phase: int, n: int = 1) -> np.ndarray:
        """Generate *n* samples for *phase*.  Returns ndarray of shape (n, 5)."""
        torch = self._torch
        self._gen.eval()
        with torch.no_grad():
            z = torch.randn(n, self._latent_dim, device=self._device)
            phase_oh = torch.zeros(n, self._num_phases, device=self._device)
            phase_oh[:, phase] = 1.0
            out = self._gen(z, phase_oh).cpu().numpy()
        # Un-normalise and clamp to non-negative
        out = self._unnormalise(out, phase)
        return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# WGANPredictor
# ---------------------------------------------------------------------------


class WGANPredictor(PredictionStrategy):
    """Synthesises telemetry by sampling from a per-edge WGAN-GP generator.

    Training strategy
    -----------------
    The predictor buffers raw telemetry packets by edge and workload phase.
    Training is triggered lazily inside :meth:`predict_edge_state` whenever:

    * at least *batch_size* observations are available for a phase, **and**
    * at least *retrain_interval* sim-minutes have elapsed since the last
      training for that edge.

    The retrain interval is set to 5 sim-minutes so the model is retrained
    approximately every 10 telemetry ticks during normal operation, giving it
    continuously improving coverage before the outage begins.

    Fallback
    --------
    If the WGAN has not yet been trained for the current phase (either because
    there is insufficient data or because training has not fired yet), the
    predictor falls back to :class:`KDEPredictor`, which itself falls back to
    :class:`HistoricalPredictor`.

    Bias correction
    ---------------
    Post-outage calibration is inherited from :class:`HistoricalPredictor`.
    """

    _RETRAIN_INTERVAL = 5.0  # sim-minutes between retraining
    _MAX_BUFFER = 400        # cap per-phase buffer to bound memory & training cost

    def __init__(
        self,
        window: int = 10,
        trend_weight: float = 0.3,
        duration: float = 48.0,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        train_epochs: int = 100,
        batch_size: int = 32,
        lambda_gp: float = 10.0,
        min_samples: int = 5,
        seed: int | None = None,
    ) -> None:
        self._duration = duration
        self._train_epochs = train_epochs
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._lambda_gp = lambda_gp
        self._min_samples = min_samples

        self._rng = np.random.default_rng(seed)

        # Fallback chain: WGAN → KDE → historical
        self._kde_fallback = KDEPredictor(
            window=window,
            trend_weight=trend_weight,
            duration=duration,
            min_samples=min_samples,
            seed=seed,
        )

        # Per-edge WGAN trainer
        self._trainers: dict[int, _WGANTrainer] = {}
        # Per-edge, per-phase raw observation buffer: list of 5-element lists
        self._buffers: dict[int, dict[int, list[list[float]]]] = {}
        # Per-edge timestamp of last WGAN training
        self._last_trained: dict[int, float] = {}

        self.bias_correction: dict[str, float] = {m: 1.0 for m in _METRIC_NAMES}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trainer(self, edge_id: int) -> _WGANTrainer:
        if edge_id not in self._trainers:
            self._trainers[edge_id] = _WGANTrainer(
                latent_dim=self._latent_dim,
                hidden_dim=self._hidden_dim,
                num_phases=_NUM_PHASES,
                lambda_gp=self._lambda_gp,
            )
        return self._trainers[edge_id]

    def _buffer_packet(self, edge_id: int, pkt: TelemetryPacket) -> None:
        phase = _get_phase(pkt.timestamp, self._duration)
        if edge_id not in self._buffers:
            self._buffers[edge_id] = {}
        phase_buf = self._buffers[edge_id]
        if phase not in phase_buf:
            phase_buf[phase] = []
        row = [
            float(pkt.queue_length),
            pkt.arrival_rate,
            pkt.utilization,
            pkt.processing_rate,
            pkt.mean_wait_time,
        ]
        phase_buf[phase].append(row)
        # Keep buffer bounded
        if len(phase_buf[phase]) > self._MAX_BUFFER:
            phase_buf[phase] = phase_buf[phase][-self._MAX_BUFFER:]

    def _maybe_train(self, edge_id: int, current_time: float) -> None:
        last = self._last_trained.get(edge_id, -999.0)
        if current_time - last < self._RETRAIN_INTERVAL:
            return

        phase_buf = self._buffers.get(edge_id, {})
        phase_data: dict[int, np.ndarray] = {}
        for phase, rows in phase_buf.items():
            if len(rows) >= self._batch_size:
                phase_data[phase] = np.array(rows[-self._MAX_BUFFER:], dtype=np.float32)

        if not phase_data:
            return

        trainer = self._trainer(edge_id)
        trainer.train(phase_data, self._train_epochs, self._batch_size)
        self._last_trained[edge_id] = current_time

    def _build_edge_state(
        self,
        edge_id: int,
        sample_vec: np.ndarray,
        reference_packet: TelemetryPacket,
    ) -> EdgeState:
        bc = self.bias_correction
        # sample_vec columns: queue_length, arrival_rate, utilization, processing_rate, mean_wait_time
        return EdgeState(
            edge_id=edge_id,
            queue_length=max(0, int(sample_vec[0] * bc["queue_length"])),
            queue_capacity=reference_packet.queue_capacity,
            utilization=max(0.0, float(sample_vec[2]) * bc["utilization"]),
            processing_rate=max(0.0, float(sample_vec[3]) * bc["processing_rate"]),
            arrival_rate=max(0.0, float(sample_vec[1]) * bc["arrival_rate"]),
            total_processed=reference_packet.total_processed,
            total_dropped=reference_packet.total_dropped,
            mean_wait_time=max(0.0, float(sample_vec[4]) * bc["mean_wait_time"]),
            training_active=False,
            local_epochs=reference_packet.local_epochs,
            sampling_rate=reference_packet.sampling_rate,
        )

    # ------------------------------------------------------------------
    # PredictionStrategy interface
    # ------------------------------------------------------------------

    def predict_edge_state(
        self,
        edge_id: int,
        history: list[TelemetryPacket],
        current_time: float,
    ) -> EdgeState:
        # 1. Buffer all packets (duplicate observation is harmless here; buffers
        #    are capped by _MAX_BUFFER)
        for pkt in history:
            self._buffer_packet(edge_id, pkt)

        # 2. Also feed KDE fallback (it deduplicates internally)
        self._kde_fallback._ingest(edge_id, history)

        # 3. Try training / retraining WGAN
        self._maybe_train(edge_id, current_time)

        # 4. Attempt WGAN sampling
        phase = _get_phase(current_time, self._duration)
        trainer = self._trainer(edge_id)
        if trainer.can_sample(phase) and history:
            vec = trainer.sample(phase, n=1)[0]
            return self._build_edge_state(edge_id, vec, history[-1])

        # 5. Fall back to KDE (which itself falls back to historical)
        return self._kde_fallback.predict_edge_state(edge_id, history, current_time)

    def predict_system_state(
        self,
        edge_histories: dict[int, list[TelemetryPacket]],
        current_time: float,
        fl_round: int,
        fl_convergence: float,
    ) -> SystemState:
        edge_states: dict[int, EdgeState] = {}
        total_arrival = 0.0
        total_processing = 0.0
        total_queue = 0
        total_dropped = 0

        for edge_id, history in edge_histories.items():
            state = self.predict_edge_state(edge_id, history, current_time)
            edge_states[edge_id] = state
            total_arrival += state.arrival_rate
            total_processing += state.processing_rate
            total_queue += state.queue_length
            total_dropped += state.total_dropped

        return SystemState(
            timestamp=current_time,
            edges=edge_states,
            fl_round=fl_round,
            fl_convergence=fl_convergence,
            active_participants=len(edge_states),
            aggregate_arrival_rate=total_arrival,
            aggregate_processing_rate=total_processing,
            aggregate_queue_length=total_queue,
            aggregate_dropped=total_dropped,
        )

    def calibrate(self, actual: EdgeState, estimated: EdgeState, dt: float) -> None:
        pairs = [
            ("queue_length", float(actual.queue_length), float(estimated.queue_length)),
            ("utilization", actual.utilization, estimated.utilization),
            ("processing_rate", actual.processing_rate, estimated.processing_rate),
            ("arrival_rate", actual.arrival_rate, estimated.arrival_rate),
            ("mean_wait_time", actual.mean_wait_time, estimated.mean_wait_time),
        ]
        for key, actual_val, est_val in pairs:
            if est_val > 0.01:
                ratio = actual_val / est_val
                self.bias_correction[key] = 0.9 * self.bias_correction[key] + 0.1 * ratio
        # Propagate to fallback chain
        self._kde_fallback.calibrate(actual, estimated, dt)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_predictor(
    strategy: str,
    window: int = 10,
    trend_weight: float = 0.3,
    duration: float = 48.0,
    workload_schedule: list | None = None,
    kde_min_samples: int = 5,
    wgan_latent_dim: int = 8,
    wgan_hidden_dim: int = 32,
    wgan_train_epochs: int = 100,
    wgan_lambda_gp: float = 10.0,
    seed: int | None = None,
) -> PredictionStrategy:
    """Instantiate a :class:`PredictionStrategy` by name.

    Parameters
    ----------
    strategy:
        One of ``"historical"``, ``"kde"``, or ``"wgan"``.
    window:
        History window size (used by historical and KDE predictors).
    trend_weight:
        Linear-trend extrapolation weight for the historical predictor.
    duration:
        Simulation duration (sim-minutes); used for phase computation.
    workload_schedule:
        Optional list of ``WorkloadPhase`` objects.  Currently used to
        derive phase boundaries; if empty the default 6-phase schedule is
        used.
    kde_min_samples:
        Minimum observations per phase before KDE sampling is used.
    wgan_latent_dim:
        Dimension of the WGAN generator's noise vector.
    wgan_hidden_dim:
        Hidden layer width for both generator and critic MLPs.
    wgan_train_epochs:
        Number of full training epochs run each time the WGAN is retrained.
    wgan_lambda_gp:
        Gradient-penalty coefficient λ for WGAN-GP.
    seed:
        Optional RNG seed for reproducibility.
    """
    if strategy == "kde":
        return KDEPredictor(
            window=window,
            trend_weight=trend_weight,
            duration=duration,
            min_samples=kde_min_samples,
            seed=seed,
        )
    elif strategy == "wgan":
        return WGANPredictor(
            window=window,
            trend_weight=trend_weight,
            duration=duration,
            latent_dim=wgan_latent_dim,
            hidden_dim=wgan_hidden_dim,
            train_epochs=wgan_train_epochs,
            batch_size=max(4, wgan_latent_dim * 4),  # sensible minimum
            lambda_gp=wgan_lambda_gp,
            min_samples=kde_min_samples,
            seed=seed,
        )
    else:
        return HistoricalPredictor(window=window, trend_weight=trend_weight)
