from __future__ import annotations

import simpy

from src.config import DTConfig, SimulationConfig
from src.telemetry import (
    EdgeState,
    SystemState,
    TelemetryBus,
    TelemetryPacket,
)
from src.prediction import PredictionStrategy, create_predictor


class DigitalTwin:
    def __init__(
        self,
        env: simpy.Environment,
        config: DTConfig,
        sim_config: SimulationConfig,
        bus: TelemetryBus,
    ):
        self.env = env
        self.config = config
        self.sim_config = sim_config
        self.bus = bus
        self.predictor: PredictionStrategy = create_predictor(
            config.prediction.strategy, config.prediction.history_window, config.prediction.trend_weight
        )

        self._edge_histories: dict[int, list[TelemetryPacket]] = {}
        self._current_state = SystemState()
        self._estimated_state: SystemState | None = None
        self._outage_active = False
        self._outage_start: float | None = None

        self.state_history: list[SystemState] = []
        self.estimated_history: list[SystemState] = []
        self.estimation_errors: list[tuple[float, float]] = []

        self._last_fl_round = 0
        self._last_fl_convergence = 0.0

        bus.subscribe(self._on_telemetry)

    def run(self) -> simpy.Process:
        return self.env.process(self._update_loop())

    def _on_telemetry(self, packet: TelemetryPacket) -> None:
        if packet.edge_id not in self._edge_histories:
            self._edge_histories[packet.edge_id] = []
        self._edge_histories[packet.edge_id].append(packet)

        if not self._outage_active:
            self._update_current_state()
            self.state_history.append(self._clone_state(self._current_state))

    def _update_current_state(self) -> None:
        total_arrival = 0.0
        total_processing = 0.0
        total_queue = 0
        total_dropped = 0

        for edge_id, history in self._edge_histories.items():
            if history:
                p = history[-1]
                total_arrival += p.arrival_rate
                total_processing += p.processing_rate
                total_queue += p.queue_length
                total_dropped += p.total_dropped

        self._current_state = SystemState(
            timestamp=self.env.now,
            edges={},
            fl_round=self._last_fl_round,
            fl_convergence=self._last_fl_convergence,
            active_participants=len(self._edge_histories),
            aggregate_arrival_rate=total_arrival,
            aggregate_processing_rate=total_processing,
            aggregate_queue_length=total_queue,
            aggregate_dropped=total_dropped,
        )
        for edge_id, history in self._edge_histories.items():
            if history:
                p = history[-1]
                self._current_state.edges[edge_id] = EdgeState(
                    edge_id=edge_id,
                    queue_length=p.queue_length,
                    queue_capacity=p.queue_capacity,
                    utilization=p.utilization,
                    processing_rate=p.processing_rate,
                    arrival_rate=p.arrival_rate,
                    total_processed=p.total_processed,
                    total_dropped=p.total_dropped,
                    mean_wait_time=p.mean_wait_time,
                    training_active=p.training_active,
                    local_epochs=p.local_epochs,
                    sampling_rate=p.sampling_rate,
                )

    def _update_loop(self) -> simpy.Generator:
        for outage in self.sim_config.outages:
            yield self.env.event() | self.env.timeout(0)

        while True:
            yield self.env.timeout(self.config.telemetry_interval)
            self._check_outage()
            if self._outage_active:
                self._estimate_state()
            elif self._estimated_state is not None:
                self._compute_estimation_error()
                self._estimated_state = None

    def _check_outage(self) -> None:
        t = self.env.now
        was_outage = self._outage_active
        self._outage_active = False
        for outage in self.sim_config.outages:
            if outage.start <= t < outage.end:
                self._outage_active = True
                if not was_outage:
                    self._outage_start = outage.start
                break
        if was_outage and not self._outage_active:
            self._outage_start = None

    def _estimate_state(self) -> None:
        state = self.predictor.predict_system_state(
            self._edge_histories,
            self.env.now,
            self._last_fl_round,
            self._last_fl_convergence,
        )
        state.timestamp = self.env.now
        self._estimated_state = state
        self.estimated_history.append(self._clone_state(state))

    def _compute_estimation_error(self) -> None:
        if not self.estimated_history or not self.state_history:
            return
        actual = self.state_history[-1]
        for est in self.estimated_history:
            q_error = abs(est.aggregate_queue_length - actual.aggregate_queue_length)
            u_error = abs(est.aggregate_arrival_rate - actual.aggregate_arrival_rate)
            self.estimation_errors.append((est.timestamp, (q_error + u_error) / 2.0))
            dt = est.timestamp - (self.estimated_history[0].timestamp if len(self.estimated_history) > 1 else est.timestamp)
            for edge_id, est_edge in est.edges.items():
                if edge_id in actual.edges:
                    self.predictor.calibrate(actual.edges[edge_id], est_edge, dt)
        self.estimated_history.clear()

    @staticmethod
    def _clone_state(state: SystemState) -> SystemState:
        return SystemState(
            timestamp=state.timestamp,
            edges={
                eid: EdgeState(
                    edge_id=s.edge_id,
                    queue_length=s.queue_length,
                    queue_capacity=s.queue_capacity,
                    utilization=s.utilization,
                    processing_rate=s.processing_rate,
                    arrival_rate=s.arrival_rate,
                    total_processed=s.total_processed,
                    total_dropped=s.total_dropped,
                    mean_wait_time=s.mean_wait_time,
                    training_active=s.training_active,
                    local_epochs=s.local_epochs,
                    sampling_rate=s.sampling_rate,
                )
                for eid, s in state.edges.items()
            },
            fl_round=state.fl_round,
            fl_convergence=state.fl_convergence,
            active_participants=state.active_participants,
            aggregate_arrival_rate=state.aggregate_arrival_rate,
            aggregate_processing_rate=state.aggregate_processing_rate,
            aggregate_queue_length=state.aggregate_queue_length,
            aggregate_dropped=state.aggregate_dropped,
        )

    def update_fl_stats(self, round_number: int, convergence: float) -> None:
        self._last_fl_round = round_number
        self._last_fl_convergence = convergence

    def get_recommendation(self) -> dict | None:
        if self._outage_active and self._estimated_state:
            state = self._estimated_state
        elif self.state_history:
            state = self.state_history[-1]
        else:
            return None

        orch = self.sim_config.orchestrator
        recs = {}
        for edge_id, edge_state in state.edges.items():
            target_util = orch.target_utilization
            if edge_state.utilization > 1.0:
                if edge_state.arrival_rate > 0:
                    desired_sampling = edge_state.processing_rate / edge_state.arrival_rate
                    new_sampling = max(orch.min_sampling_rate, min(orch.max_sampling_rate, desired_sampling))
                else:
                    new_sampling = orch.min_sampling_rate
                recs[edge_id] = {
                    "local_epochs": orch.min_local_epochs,
                    "sampling_rate": new_sampling,
                }
            elif edge_state.utilization > target_util * 1.1:
                recs[edge_id] = {
                    "local_epochs": max(orch.min_local_epochs, edge_state.local_epochs - 2),
                    "sampling_rate": max(orch.min_sampling_rate, edge_state.sampling_rate * 0.8),
                }
            elif edge_state.utilization < target_util * 0.7:
                recs[edge_id] = {
                    "local_epochs": min(orch.max_local_epochs, edge_state.local_epochs + 1),
                    "sampling_rate": min(orch.max_sampling_rate, edge_state.sampling_rate * 1.1),
                }
        return recs if recs else None

    @property
    def is_outage(self) -> bool:
        return self._outage_active
