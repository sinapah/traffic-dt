from __future__ import annotations

import simpy
import numpy as np

from src.config import EdgeConfig, FLConfig, SimulationConfig
from src.queue import BoundedQueue, Frame
from src.telemetry import TelemetryBus, TelemetryPacket


class EdgeNode:
    def __init__(
        self,
        env: simpy.Environment,
        config: EdgeConfig,
        sim_config: SimulationConfig,
        bus: TelemetryBus,
        rng: np.random.Generator,
    ):
        self.env = env
        self.config = config
        self.sim_config = sim_config
        self.bus = bus
        self.rng = rng
        self.queue = BoundedQueue(config.queue_capacity)

        self.service_rate = config.service_rate
        self.sampling_rate = 1.0
        self.local_epochs = sim_config.fl.local_epochs

        self._total_arrivals_window = 0
        self._total_processed_window = 0
        self._window_start = 0.0
        self._processing_rate = 0.0
        self._arrival_rate = 0.0
        self._training_active = False
        self._training_done = simpy.Event(env)

        self.fl_round = 0
        self.fl_convergence = 0.0
        self._last_processing_rate = 0.0

        self.training_process: simpy.Process | None = None
        self._training_duration = 0.0

    @property
    def utilization(self) -> float:
        return self.queue.utilization

    def run(self) -> simpy.Process:
        return self.env.process(self._process_loop())

    def run_telemetry(self, interval: float) -> simpy.Process:
        return self.env.process(self._telemetry_loop(interval))

    def trigger_training(self, fl_round: int, convergence: float, fl_config: FLConfig) -> simpy.Event:
        self.fl_round = fl_round
        self.fl_convergence = convergence
        self._training_done = simpy.Event(self.env)
        self.training_process = self.env.process(
            self._training_loop(fl_config)
        )
        return self._training_done

    def _process_loop(self) -> simpy.Generator:
        while True:
            frame = self.queue.get(self.env.now)
            if frame is None:
                yield self.env.timeout(0.1)
                continue

            if self.rng.random() > self.sampling_rate:
                continue

            service_time = self.rng.exponential(1.0 / self.service_rate)
            yield self.env.timeout(service_time)

    def _training_loop(self, fl_config: FLConfig) -> simpy.Generator:
        self._training_active = True
        cost = (
            fl_config.local_epochs
            * fl_config.batch_size
            * fl_config.dataset_size
            * fl_config.cost_per_unit
        )
        training_time = cost
        self._training_duration = training_time
        yield self.env.timeout(training_time)
        self._training_active = False
        self._training_done.succeed()

    def _telemetry_loop(self, interval: float) -> simpy.Generator:
        while True:
            yield self.env.timeout(interval)
            self._update_rates()
            packet = TelemetryPacket(
                timestamp=self.env.now,
                edge_id=self.config.edge_id,
                queue_length=self.queue.length,
                queue_capacity=self.config.queue_capacity,
                utilization=self.queue.utilization,
                processing_rate=self._processing_rate,
                arrival_rate=self._arrival_rate,
                total_processed=self.queue.total_processed,
                total_dropped=self.queue.total_dropped,
                mean_wait_time=self.queue.mean_wait_time,
                training_active=self._training_active,
                fl_round=self.fl_round,
                fl_convergence=self.fl_convergence,
                local_epochs=self.local_epochs,
                sampling_rate=self.sampling_rate,
            )
            self.bus.publish(packet)

    def _update_rates(self) -> None:
        dt = self.env.now - self._window_start
        if dt > 0:
            self._processing_rate = (
                (self.queue.total_processed - self._total_processed_window) / dt
            )
            self._arrival_rate = (
                (self.queue.total_arrivals - self._total_arrivals_window) / dt
            )
        self._total_processed_window = self.queue.total_processed
        self._total_arrivals_window = self.queue.total_arrivals
        self._window_start = self.env.now

    def apply_parameters(
        self,
        local_epochs: int | None = None,
        sampling_rate: float | None = None,
    ) -> None:
        if local_epochs is not None:
            self.local_epochs = local_epochs
        if sampling_rate is not None:
            self.sampling_rate = max(
                self.sim_config.orchestrator.min_sampling_rate,
                min(self.sim_config.orchestrator.max_sampling_rate, sampling_rate),
            )
