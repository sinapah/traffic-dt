from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from src.config import OrchestratorConfig, SimulationConfig
from src.digital_twin import DigitalTwin
from src.edge_node import EdgeNode


@dataclass
class OrchestrationAction:
    timestamp: float
    edge_id: int
    local_epochs: int
    sampling_rate: float


class Orchestrator:
    def __init__(
        self,
        env: simpy.Environment,
        config: OrchestratorConfig,
        sim_config: SimulationConfig,
        digital_twin: DigitalTwin | None,
        edges: dict[int, EdgeNode],
    ):
        self.env = env
        self.config = config
        self.sim_config = sim_config
        self.dt = digital_twin
        self.edges = edges
        self.actions: list[OrchestrationAction] = []

    def run(self) -> simpy.Process:
        if not self.config.enabled:
            return self.env.process(self._noop())
        return self.env.process(self._orchestration_loop())

    def _orchestration_loop(self) -> simpy.Generator:
        while True:
            yield self.env.timeout(self.config.adjustment_interval)
            if self.dt is None:
                continue
            recs = self.dt.get_recommendation()
            if recs:
                for edge_id, params in recs.items():
                    if edge_id in self.edges:
                        edge = self.edges[edge_id]
                        edge.apply_parameters(
                            local_epochs=params.get("local_epochs"),
                            sampling_rate=params.get("sampling_rate"),
                        )
                        self.actions.append(
                            OrchestrationAction(
                                timestamp=self.env.now,
                                edge_id=edge_id,
                                local_epochs=edge.local_epochs,
                                sampling_rate=edge.sampling_rate,
                            )
                        )

    def _noop(self) -> simpy.Generator:
        while True:
            yield self.env.timeout(self.config.adjustment_interval)

    def get_action_log(self) -> list[dict]:
        return [
            {
                "timestamp": a.timestamp,
                "edge_id": a.edge_id,
                "local_epochs": a.local_epochs,
                "sampling_rate": a.sampling_rate,
            }
            for a in self.actions
        ]
