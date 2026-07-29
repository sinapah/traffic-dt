from __future__ import annotations

import copy
import logging
import threading

import simpy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import EdgeConfig, FLConfig, SimulationConfig
from src.dataset import DetracDataset
from src.queue import BoundedQueue, Frame
from src.telemetry import TelemetryBus, TelemetryPacket

log = logging.getLogger(__name__)


class EdgeNode:
    def __init__(
        self,
        env: simpy.Environment,
        config: EdgeConfig,
        sim_config: SimulationConfig,
        bus: TelemetryBus,
        rng: np.random.Generator,
        model: torch.nn.Module | None = None,
        assigned_dataset: torch.utils.data.Dataset | None = None,
    ):
        self.env = env
        self.config = config
        self.sim_config = sim_config
        self.bus = bus
        self.rng = rng
        self.queue = BoundedQueue(config.queue_capacity)
        self.model = model
        self.assigned_dataset = assigned_dataset

        # Paths of images processed since the last FL round, used as training data
        self.processed_image_paths: list[str] = []

        # Lookup from image path to index in the full dataset for building Subsets
        self._path_to_idx: dict[str, int] = {}
        if assigned_dataset is not None:
            for idx in assigned_dataset.indices:
                path = assigned_dataset.dataset.samples[idx]["image_path"]
                self._path_to_idx[path] = idx

        self.service_rate = config.service_rate
        self.sampling_rate = 1.0
        self.local_epochs = sim_config.fl.local_epochs
        self._fl_config: FLConfig | None = None

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
        self._trained_state: dict[str, torch.Tensor] | None = None
        self._training_loss: float = 0.0

    @property
    def utilization(self) -> float:
        if self.service_rate > 0:
            return min(self._arrival_rate / self.service_rate, float("inf"))
        return 0.0

    def run(self) -> simpy.Process:
        return self.env.process(self._process_loop())

    def run_telemetry(self, interval: float) -> simpy.Process:
        return self.env.process(self._telemetry_loop(interval))

    def trigger_training(self, fl_round: int, convergence: float, fl_config: FLConfig) -> simpy.Event:
        self.fl_round = fl_round
        self.fl_convergence = convergence
        self._fl_config = fl_config
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

            effective_rate = self.service_rate
            if self._training_active and self._fl_config is not None:
                effective_rate *= (1.0 - self._fl_config.training_resource_contention)
            service_time = self.rng.exponential(1.0 / effective_rate)
            yield self.env.timeout(service_time)

            # Collect the path so we can train on it at the next FL round
            if frame.image_path is not None:
                self.processed_image_paths.append(frame.image_path)

    def _training_loop(self, fl_config: FLConfig) -> simpy.Generator:
        self._training_active = True
        self._trained_state = None
        self._training_loss = 0.0

        # Use the images the edge actually processed since the last round
        paths = list(set(self.processed_image_paths))
        n_samples = len(paths)

        if self.model is not None and self.assigned_dataset is not None and n_samples > 0:
            result = {}
            log.info(
                "Edge %d: starting local training (FL round %d, %d images processed since last round, %d epoch(s)) ...",
                self.config.edge_id, self.fl_round, n_samples, fl_config.fine_tune_epochs,
            )

            def _train():
                indices = [
                    self._path_to_idx[p] for p in paths
                    if p in self._path_to_idx
                ]
                train_subset = Subset(self.assigned_dataset.dataset, indices)
                loader = DataLoader(
                    train_subset,
                    batch_size=fl_config.batch_size,
                    shuffle=True,
                    collate_fn=DetracDataset.collate_fn,
                    num_workers=0,
                )
                model_copy = copy.deepcopy(self.model)
                model_copy.train()
                params = [p for p in model_copy.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(params, lr=fl_config.learning_rate, momentum=0.9, weight_decay=1e-4)
                total_loss = 0.0
                num_batches = 0
                for _ in range(fl_config.fine_tune_epochs):
                    for images, targets in loader:
                        images = [img.to("cpu") for img in images]
                        targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
                        loss_dict = model_copy(images, targets)
                        loss = sum(loss_dict.values())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1
                result["state_dict"] = copy.deepcopy(model_copy.state_dict())
                result["loss"] = total_loss / max(num_batches, 1)

            thread = threading.Thread(target=_train, daemon=True)
            thread.start()

            # Simulated training cost scales with actual number of images processed
            cost = (
                fl_config.local_epochs
                * fl_config.batch_size
                * n_samples
                * fl_config.cost_per_unit
            )
            self._training_duration = cost
            yield self.env.timeout(cost)
            thread.join()
            self._trained_state = result.get("state_dict")
            self._training_loss = result.get("loss", 0.0)
            # Clear so the next round starts fresh
            self.processed_image_paths.clear()
            log.info(
                "Edge %d: training complete (loss=%.4f)",
                self.config.edge_id, self._training_loss,
            )
        else:
            if n_samples == 0 and self.model is not None:
                log.info(
                    "Edge %d: no images processed since last round, skipping training.",
                    self.config.edge_id,
                )
            cost = (
                fl_config.local_epochs
                * fl_config.batch_size
                * max(n_samples, 1)
                * fl_config.cost_per_unit
            )
            self._training_duration = cost
            yield self.env.timeout(cost)

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
                utilization=self.utilization,
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

    def get_trained_state(self) -> dict[str, torch.Tensor] | None:
        return self._trained_state

    def get_training_loss(self) -> float:
        return self._training_loss

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
