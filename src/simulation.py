from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import simpy

from src.config import SimulationConfig, CameraConfig, EdgeConfig
from src.workload import WorkloadSchedule
from src.camera import Camera
from src.edge_node import EdgeNode
from src.fl_coordinator import FLCoordinator
from src.telemetry import TelemetryBus
from src.digital_twin import DigitalTwin
from src.orchestrator import Orchestrator
from src.metrics import MetricsCollector


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

        edges: dict[int, EdgeNode] = {}
        edge_configs = self.config.edges or [
            EdgeConfig(edge_id=i, service_rate=sr, queue_capacity=500)
            for i, sr in enumerate([25.0, 35.0, 15.0])
        ]
        for ec in edge_configs:
            edge_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            edges[ec.edge_id] = EdgeNode(
                env=env,
                config=ec,
                sim_config=self.config,
                bus=bus,
                rng=edge_rng,
            )

        camera_configs = self.config.cameras or [
            CameraConfig(camera_id=i, edge_id=i, base_fps=10.0)
            for i in range(self.config.num_cameras)
        ]
        cameras = []
        for cc in camera_configs:
            cam_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            cam = Camera(
                env=env,
                config=cc,
                sim_config=self.config,
                queue=edges[cc.edge_id].queue,
                workload=workload,
                rng=cam_rng,
            )
            cameras.append(cam)

        fl_coordinator = FLCoordinator(env, self.config.fl, rng=random.Random(int(self.rng.integers(0, 2**31))))

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
                metrics.record(env.now, "fl_round", fl_coordinator.round_number)
                metrics.record(env.now, "fl_convergence", fl_coordinator.current_convergence)
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

        env.run(until=self.config.duration)

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
