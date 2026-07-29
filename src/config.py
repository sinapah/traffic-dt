from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    camera_id: int
    edge_id: int
    base_fps: float = 10.0


@dataclass
class EdgeConfig:
    edge_id: int
    service_rate: float = 25.0
    queue_capacity: int = 500


@dataclass
class FLConfig:
    aggregation_interval: float = 5.0
    local_epochs: int = 2
    batch_size: int = 32
    dataset_size: int = 100
    cost_per_unit: float = 0.001
    participation_rate: float = 1.0
    convergence_ceiling: float = 0.95
    convergence_speed: float = 0.3
    training_resource_contention: float = 0.5  # 0.0 = no contention, 1.0 = fully blocked
    learning_rate: float = 0.001
    fine_tune_epochs: int = 1


@dataclass
class PredictionConfig:
    strategy: str = "historical"
    history_window: int = 10
    trend_weight: float = 0.3  # 0.0 = flat average only, 1.0 = full trend extrapolation
    duration: float = 48.0
    workload_schedule: list[WorkloadPhase] = field(default_factory=list)
    kde_min_samples: int = 5
    wgan_latent_dim: int = 8
    wgan_hidden_dim: int = 32
    wgan_train_epochs: int = 100
    wgan_lambda_gp: float = 10.0


@dataclass
class DTConfig:
    telemetry_interval: float = 0.5
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


@dataclass
class OrchestratorConfig:
    enabled: bool = True
    adjustment_interval: float = 2.0
    target_utilization: float = 0.80
    max_local_epochs: int = 10
    min_local_epochs: int = 1
    max_sampling_rate: float = 1.0
    min_sampling_rate: float = 0.2


@dataclass
class OutagePeriod:
    start: float
    end: float


@dataclass
class WorkloadPhase:
    hour: float
    multiplier: float


@dataclass
class SimulationConfig:
    duration: float = 48.0
    time_scale: float = 60.0
    num_cameras: int = 3
    num_edges: int = 3
    cameras: list[CameraConfig] = field(default_factory=list)
    edges: list[EdgeConfig] = field(default_factory=list)
    fl: FLConfig = field(default_factory=FLConfig)
    dt: DTConfig = field(default_factory=DTConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    outages: list[OutagePeriod] = field(default_factory=list)
    workload_schedule: list[WorkloadPhase] = field(default_factory=list)
    seed: int | None = None
    dataset_path: str = "DETRAC-Images/DETRAC-Images"
    annotation_path: str = "DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML"
    global_val_ratio: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        d = {
            "duration": self.duration,
            "time_scale": self.time_scale,
            "num_cameras": self.num_cameras,
            "num_edges": self.num_edges,
            "fl": {
                "aggregation_interval": self.fl.aggregation_interval,
                "local_epochs": self.fl.local_epochs,
                "batch_size": self.fl.batch_size,
                "dataset_size": self.fl.dataset_size,
                "cost_per_unit": self.fl.cost_per_unit,
                "participation_rate": self.fl.participation_rate,
                "convergence_ceiling": self.fl.convergence_ceiling,
                "convergence_speed": self.fl.convergence_speed,
                "training_resource_contention": self.fl.training_resource_contention,
                "learning_rate": self.fl.learning_rate,
                "fine_tune_epochs": self.fl.fine_tune_epochs,
            },
            "dt": {
                "telemetry_interval": self.dt.telemetry_interval,
                "prediction": {
                    "strategy": self.dt.prediction.strategy,
                    "history_window": self.dt.prediction.history_window,
                    "trend_weight": self.dt.prediction.trend_weight,
                    "duration": self.dt.prediction.duration,
                    "workload_schedule": [
                        {"hour": w.hour, "multiplier": w.multiplier}
                        for w in self.dt.prediction.workload_schedule
                    ],
                    "kde_min_samples": self.dt.prediction.kde_min_samples,
                    "wgan_latent_dim": self.dt.prediction.wgan_latent_dim,
                    "wgan_hidden_dim": self.dt.prediction.wgan_hidden_dim,
                    "wgan_train_epochs": self.dt.prediction.wgan_train_epochs,
                    "wgan_lambda_gp": self.dt.prediction.wgan_lambda_gp,
                },
            },
            "orchestrator": {
                "enabled": self.orchestrator.enabled,
                "adjustment_interval": self.orchestrator.adjustment_interval,
                "target_utilization": self.orchestrator.target_utilization,
                "max_local_epochs": self.orchestrator.max_local_epochs,
                "min_local_epochs": self.orchestrator.min_local_epochs,
                "max_sampling_rate": self.orchestrator.max_sampling_rate,
                "min_sampling_rate": self.orchestrator.min_sampling_rate,
            },
            "outages": [{"start": o.start, "end": o.end} for o in self.outages],
            "workload_schedule": [
                {"hour": w.hour, "multiplier": w.multiplier}
                for w in self.workload_schedule
            ],
            "seed": self.seed,
            "dataset_path": self.dataset_path,
            "annotation_path": self.annotation_path,
            "global_val_ratio": self.global_val_ratio,
        }
        if self.cameras:
            d["cameras"] = [
                {"camera_id": c.camera_id, "edge_id": c.edge_id, "base_fps": c.base_fps}
                for c in self.cameras
            ]
        if self.edges:
            d["edges"] = [
                {"edge_id": e.edge_id, "service_rate": e.service_rate, "queue_capacity": e.queue_capacity}
                for e in self.edges
            ]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimulationConfig:
        cameras = [
            CameraConfig(camera_id=c["camera_id"], edge_id=c["edge_id"], base_fps=c.get("base_fps", 10.0))
            for c in d.get("cameras", [])
        ]
        edges = [
            EdgeConfig(edge_id=e["edge_id"], service_rate=e["service_rate"], queue_capacity=e.get("queue_capacity", 500))
            for e in d.get("edges", [])
        ]
        fl_d = d.get("fl", {})
        fl = FLConfig(
            aggregation_interval=fl_d.get("aggregation_interval", 5.0),
            local_epochs=fl_d.get("local_epochs", 2),
            batch_size=fl_d.get("batch_size", 32),
            dataset_size=fl_d.get("dataset_size", 100),
            cost_per_unit=fl_d.get("cost_per_unit", 0.001),
            participation_rate=fl_d.get("participation_rate", 1.0),
            convergence_ceiling=fl_d.get("convergence_ceiling", 0.95),
            convergence_speed=fl_d.get("convergence_speed", 0.3),
            training_resource_contention=fl_d.get("training_resource_contention", 0.5),
            learning_rate=fl_d.get("learning_rate", 0.001),
            fine_tune_epochs=fl_d.get("fine_tune_epochs", 1),
        )
        dt_d = d.get("dt", {})
        pred_d = dt_d.get("prediction", {})
        ws_list = pred_d.get("workload_schedule", [])
        if ws_list is None:
            ws_list = []
        workload_schedule_pred = [
            WorkloadPhase(hour=w["hour"], multiplier=w["multiplier"]) for w in ws_list
        ]
        prediction = PredictionConfig(
            strategy=pred_d.get("strategy", "historical"),
            history_window=pred_d.get("history_window", 10),
            trend_weight=pred_d.get("trend_weight", 0.3),
            duration=pred_d.get("duration", 48.0),
            workload_schedule=workload_schedule_pred,
            kde_min_samples=pred_d.get("kde_min_samples", 5),
            wgan_latent_dim=pred_d.get("wgan_latent_dim", 8),
            wgan_hidden_dim=pred_d.get("wgan_hidden_dim", 32),
            wgan_train_epochs=pred_d.get("wgan_train_epochs", 100),
            wgan_lambda_gp=pred_d.get("wgan_lambda_gp", 10.0),
        )
        dt = DTConfig(
            telemetry_interval=dt_d.get("telemetry_interval", 0.5),
            prediction=prediction,
        )
        orch_d = d.get("orchestrator", {})
        orchestrator = OrchestratorConfig(
            enabled=orch_d.get("enabled", True),
            adjustment_interval=orch_d.get("adjustment_interval", 2.0),
            target_utilization=orch_d.get("target_utilization", 0.80),
            max_local_epochs=orch_d.get("max_local_epochs", 10),
            min_local_epochs=orch_d.get("min_local_epochs", 1),
            max_sampling_rate=orch_d.get("max_sampling_rate", 1.0),
            min_sampling_rate=orch_d.get("min_sampling_rate", 0.2),
        )
        outages = [OutagePeriod(start=o["start"], end=o["end"]) for o in d.get("outages", [])]
        workload = [WorkloadPhase(hour=w["hour"], multiplier=w["multiplier"]) for w in d.get("workload_schedule", [])]

        return cls(
            duration=d.get("duration", 48.0),
            time_scale=d.get("time_scale", 60.0),
            num_cameras=d.get("num_cameras", 3),
            num_edges=d.get("num_edges", 3),
            cameras=cameras,
            edges=edges,
            fl=fl,
            dt=dt,
            orchestrator=orchestrator,
            outages=outages,
            workload_schedule=workload,
            seed=d.get("seed"),
            dataset_path=d.get("dataset_path", "DETRAC-Images/DETRAC-Images"),
            annotation_path=d.get("annotation_path", "DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML"),
            global_val_ratio=d.get("global_val_ratio", 0.01),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SimulationConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
