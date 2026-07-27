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

    def calibrate(self, actual: EdgeState, estimated: EdgeState, dt: float) -> None:
        pass


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

        n = len(recent)
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


class KDEPredictor(PredictionStrategy):
    def __init__(self, window: int = 10, trend_weight: float = 0.3):
        self.window = window
        self._fallback = HistoricalPredictor(window, trend_weight)

    def predict_edge_state(self, edge_id, history, current_time):
        return self._fallback.predict_edge_state(edge_id, history, current_time)

    def predict_system_state(self, edge_histories, current_time, fl_round, fl_convergence):
        return self._fallback.predict_system_state(
            edge_histories, current_time, fl_round, fl_convergence
        )

    def calibrate(self, actual, estimated, dt):
        self._fallback.calibrate(actual, estimated, dt)


class WGANPredictor(PredictionStrategy):
    def __init__(self, window: int = 10, trend_weight: float = 0.3):
        self.window = window
        self._fallback = HistoricalPredictor(window, trend_weight)

    def predict_edge_state(self, edge_id, history, current_time):
        return self._fallback.predict_edge_state(edge_id, history, current_time)

    def predict_system_state(self, edge_histories, current_time, fl_round, fl_convergence):
        return self._fallback.predict_system_state(
            edge_histories, current_time, fl_round, fl_convergence
        )

    def calibrate(self, actual, estimated, dt):
        self._fallback.calibrate(actual, estimated, dt)


def create_predictor(strategy: str, window: int = 10, trend_weight: float = 0.3) -> PredictionStrategy:
    strategies = {
        "historical": HistoricalPredictor,
        "kde": KDEPredictor,
        "wgan": WGANPredictor,
    }
    cls = strategies.get(strategy, HistoricalPredictor)
    return cls(window=window, trend_weight=trend_weight)
