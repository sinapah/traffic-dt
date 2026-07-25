from __future__ import annotations

import math
import random

import simpy

from src.config import FLConfig


class FLCoordinator:
    def __init__(self, env: simpy.Environment, fl_config: FLConfig, rng: random.Random | None = None):
        self.env = env
        self.config = fl_config
        self.rng = rng or random.Random()
        self.round_number = 0
        self.round_durations: list[float] = []
        self.round_participants: list[int] = []
        self.round_convergence: list[float] = []
        self.aggregation_count = 0
        self._current_convergence = 0.0

    def run(self, edges: list, callback) -> simpy.Process:
        return self.env.process(self._aggregation_loop(edges, callback))

    def _aggregation_loop(
        self, edges: list, callback
    ) -> simpy.Generator:
        while True:
            yield self.env.timeout(self.config.aggregation_interval)
            self.round_number += 1
            round_start = self.env.now

            training_events = []
            active_edges = []

            for edge in edges:
                if self.rng.random() < self.config.participation_rate:
                    evt = edge.trigger_training(
                        self.round_number, self._current_convergence, self.config
                    )
                    training_events.append(evt)
                    active_edges.append(edge)

            if training_events:
                yield simpy.AnyOf(self.env, training_events)

            round_duration = self.env.now - round_start
            self.round_durations.append(round_duration)
            self.round_participants.append(len(active_edges))
            self.aggregation_count += 1

            self._current_convergence = self._compute_convergence(self.round_number)
            self.round_convergence.append(self._current_convergence)

            callback(
                round_number=self.round_number,
                duration=round_duration,
                participants=len(active_edges),
                convergence=self._current_convergence,
            )

    def _compute_convergence(self, round_num: int) -> float:
        ceiling = self.config.convergence_ceiling
        speed = self.config.convergence_speed
        return ceiling * (1.0 - math.exp(-speed * round_num))

    @property
    def current_convergence(self) -> float:
        return self._current_convergence

    def get_stats(self) -> dict:
        return {
            "round_number": self.round_number,
            "aggregation_count": self.aggregation_count,
            "mean_round_duration": (
                sum(self.round_durations) / len(self.round_durations)
                if self.round_durations else 0.0
            ),
            "current_convergence": self._current_convergence,
            "mean_participants": (
                sum(self.round_participants) / len(self.round_participants)
                if self.round_participants else 0.0
            ),
        }
