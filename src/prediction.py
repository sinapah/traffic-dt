from __future__ import annotations

from abc import ABC, abstractmethod

from src.telemetry import EdgeState, SystemState, TelemetryPacket


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


class HistoricalPredictor(PredictionStrategy):
    def __init__(self, window: int = 10):
        self.window = window

    def predict_edge_state(
        self,
        edge_id: int,
        history: list[TelemetryPacket],
        current_time: float,
    ) -> EdgeState:
        recent = history[-self.window:] if history else []
        if not recent:
            return EdgeState(edge_id=edge_id)

        n = len(recent)
        return EdgeState(
            edge_id=edge_id,
            queue_length=sum(p.queue_length for p in recent) // n,
            queue_capacity=recent[-1].queue_capacity,
            utilization=sum(p.utilization for p in recent) / n,
            processing_rate=sum(p.processing_rate for p in recent) / n,
            arrival_rate=sum(p.arrival_rate for p in recent) / n,
            total_processed=recent[-1].total_processed,
            total_dropped=recent[-1].total_dropped,
            mean_wait_time=sum(p.mean_wait_time for p in recent) / n,
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


class KDEPredictor(PredictionStrategy):
    def __init__(self, window: int = 10):
        self.window = window
        self._fallback = HistoricalPredictor(window)

    def predict_edge_state(self, edge_id, history, current_time):
        return self._fallback.predict_edge_state(edge_id, history, current_time)

    def predict_system_state(self, edge_histories, current_time, fl_round, fl_convergence):
        return self._fallback.predict_system_state(
            edge_histories, current_time, fl_round, fl_convergence
        )


class WGANPredictor(PredictionStrategy):
    def __init__(self, window: int = 10):
        self.window = window
        self._fallback = HistoricalPredictor(window)

    def predict_edge_state(self, edge_id, history, current_time):
        return self._fallback.predict_edge_state(edge_id, history, current_time)

    def predict_system_state(self, edge_histories, current_time, fl_round, fl_convergence):
        return self._fallback.predict_system_state(
            edge_histories, current_time, fl_round, fl_convergence
        )


def create_predictor(strategy: str, window: int = 10) -> PredictionStrategy:
    strategies = {
        "historical": HistoricalPredictor,
        "kde": KDEPredictor,
        "wgan": WGANPredictor,
    }
    cls = strategies.get(strategy, HistoricalPredictor)
    return cls(window=window)
