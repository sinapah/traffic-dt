from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import simpy
import torch
from torch.utils.data import DataLoader

from src.config import SimulationConfig, CameraConfig, EdgeConfig
from src.workload import WorkloadSchedule
from src.camera import Camera
from src.edge_node import EdgeNode
from src.fl_coordinator import FLCoordinator
from src.telemetry import TelemetryBus
from src.digital_twin import DigitalTwin
from src.orchestrator import Orchestrator
from src.metrics import MetricsCollector
from torch.utils.data import Subset
from src.dataset import DetracDataset
from src.model import create_model

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    config: SimulationConfig
    metrics: MetricsCollector
    telemetry_bus: TelemetryBus
    digital_twin: DigitalTwin | None
    orchestrator: Orchestrator
    edges: dict[int, EdgeNode]
    fl_coordinator: FLCoordinator
    cameras: list[Camera]


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def run(self, dt_enabled: bool = True) -> SimulationResult:
        env = simpy.Environment()
        metrics = MetricsCollector()
        bus = TelemetryBus()
        workload = WorkloadSchedule(self.config)

        dataset: DetracDataset | None = None
        train_indices: list[int] | None = None
        val_loader: DataLoader | None = None
        base_seed = self.config.seed or 42
        if self.config.dataset_path and self.config.annotation_path:
            log.info(
                "Loading dataset from %s ...",
                self.config.dataset_path,
            )
            dataset = DetracDataset(
                image_root=self.config.dataset_path,
                annotation_root=self.config.annotation_path,
            )
            n = len(dataset)
            rng_val = torch.Generator().manual_seed(base_seed + 1)
            perm = torch.randperm(n, generator=rng_val).tolist()
            n_val = max(1, int(n * self.config.global_val_ratio))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            log.info(
                "Dataset loaded: %d samples (%d train, %d val)",
                n, len(train_idx), n_val,
            )
            val_dataset = Subset(dataset, val_idx)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.fl.batch_size,
                shuffle=False,
                collate_fn=DetracDataset.collate_fn,
                num_workers=0,
            )
            train_indices = train_idx

        edges: dict[int, EdgeNode] = {}
        edge_configs = self.config.edges or [
            EdgeConfig(edge_id=i, service_rate=sr, queue_capacity=500)
            for i, sr in enumerate([25.0, 35.0, 15.0])
        ]
        for ec in edge_configs:
            edge_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            log.info("Creating model for edge %d ...", ec.edge_id)
            edge_model = create_model(pretrained=True) if dataset is not None else None
            edge_ds = None
            if dataset is not None and train_indices is not None:
                n_train = len(train_indices)
                per_edge = n_train // len(edge_configs)
                rng_edge = torch.Generator().manual_seed(base_seed + 2 + ec.edge_id)
                perm_t = torch.randperm(n_train, generator=rng_edge).tolist()
                start = ec.edge_id * per_edge
                end = start + per_edge if ec.edge_id < len(edge_configs) - 1 else n_train
                edge_indices = [train_indices[perm_t[i]] for i in range(start, end)]
                edge_ds = Subset(dataset, edge_indices)
            edges[ec.edge_id] = EdgeNode(
                env=env,
                config=ec,
                sim_config=self.config,
                bus=bus,
                rng=edge_rng,
                model=edge_model,
                assigned_dataset=edge_ds,
            )

        camera_configs = self.config.cameras or [
            CameraConfig(camera_id=i, edge_id=i, base_fps=10.0)
            for i in range(self.config.num_cameras)
        ]
        cameras = []
        for cc in camera_configs:
            cam_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            edge_image_paths = []
            if dataset is not None and train_indices is not None:
                n_train = len(train_indices)
                per_edge = n_train // len(edge_configs)
                rng_cam = torch.Generator().manual_seed(base_seed + 2 + cc.edge_id)
                perm_cam = torch.randperm(n_train, generator=rng_cam).tolist()
                start = cc.edge_id * per_edge
                end = start + per_edge if cc.edge_id < len(edge_configs) - 1 else n_train
                edge_idx = [train_indices[perm_cam[i]] for i in range(start, end)]
                edge_image_paths = [dataset.samples[i]["image_path"] for i in edge_idx]
            cam = Camera(
                env=env,
                config=cc,
                sim_config=self.config,
                queue=edges[cc.edge_id].queue,
                workload=workload,
                rng=cam_rng,
                image_paths=edge_image_paths,
            )
            cameras.append(cam)

        log.info("Creating global FL model ...")
        global_model = create_model(pretrained=True) if dataset is not None else None
        fl_coordinator = FLCoordinator(
            env,
            self.config.fl,
            rng=random.Random(int(self.rng.integers(0, 2**31))),
            global_model=global_model,
            val_loader=val_loader,
        )

        dt: DigitalTwin | None = None
        if dt_enabled:
            dt = DigitalTwin(
                env=env,
                config=self.config.dt,
                sim_config=self.config,
                bus=bus,
            )

        def on_fl_round(**kwargs):
            if dt is not None:
                dt.update_fl_stats(kwargs["round_number"], kwargs["convergence"])
            round_num = kwargs.get("round_number", 0)
            metrics.record(env.now, "fl_round", round_num)
            metrics.record(env.now, "fl_convergence", kwargs.get("convergence", 0.0))
            if fl_coordinator.round_losses:
                metrics.record(env.now, "fl_loss", fl_coordinator.round_losses[-1])
            if fl_coordinator.round_accuracy:
                acc = fl_coordinator.round_accuracy[-1]
                for k, v in acc.items():
                    safe_key = k.replace(".", "_")
                    metrics.record(env.now, f"fl_{safe_key}", v)

        orchestrator = Orchestrator(
            env=env,
            config=self.config.orchestrator,
            sim_config=self.config,
            digital_twin=dt,
            edges=edges,
        )

        def outage_controller():
            for outage in self.config.outages:
                yield env.timeout(outage.start - env.now)
                bus.set_outage(True)
                yield env.timeout(outage.end - outage.start)
                bus.set_outage(False)

        def metrics_collector():
            while True:
                yield env.timeout(0.25)
                for edge_id, edge in edges.items():
                    metrics.record(env.now, "queue_length", edge.queue.length, edge_id=edge_id)
                    metrics.record(env.now, "utilization", edge.utilization, edge_id=edge_id)
                    metrics.record(env.now, "processing_rate", edge._processing_rate, edge_id=edge_id)
                    metrics.record(env.now, "arrival_rate", edge._arrival_rate, edge_id=edge_id)
                    metrics.record(env.now, "wait_time", edge.queue.mean_wait_time, edge_id=edge_id)
                    metrics.record(env.now, "dropped_frames", edge.queue.total_dropped, edge_id=edge_id)
                    metrics.record(env.now, "total_processed", edge.queue.total_processed, edge_id=edge_id)
                if dt:
                    metrics.record(env.now, "dt_outage", float(dt.is_outage))

        for edge in edges.values():
            edge.run()
            edge.run_telemetry(self.config.dt.telemetry_interval)
        for cam in cameras:
            cam.run()
        fl_coordinator.run(list(edges.values()), on_fl_round)
        if dt:
            dt.run()
        orchestrator.run()
        env.process(outage_controller())
        env.process(metrics_collector())

        log.info(
            "Running simulation (duration=%.1f, dt_enabled=%s) ...",
            self.config.duration, dt_enabled,
        )
        env.run(until=self.config.duration)
        log.info("Simulation complete.")

        return SimulationResult(
            config=self.config,
            metrics=metrics,
            telemetry_bus=bus,
            digital_twin=dt,
            orchestrator=orchestrator,
            edges=edges,
            fl_coordinator=fl_coordinator,
            cameras=cameras,
        )
