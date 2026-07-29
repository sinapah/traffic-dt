from __future__ import annotations

import math
import random
import logging
import gc
import ctypes
from collections import OrderedDict

import simpy
import torch
from torch.utils.data import DataLoader

from src.config import FLConfig
from src.dataset import DetracDataset
from src.model import evaluate

log = logging.getLogger(__name__)


class FLCoordinator:
    def __init__(
        self,
        env: simpy.Environment,
        fl_config: FLConfig,
        rng: random.Random | None = None,
        global_model: torch.nn.Module | None = None,
        val_loader: DataLoader | None = None,
    ):
        self.env = env
        self.config = fl_config
        self.rng = rng or random.Random()
        self.global_model = global_model
        self.val_loader = val_loader
        self.round_number = 0
        self.round_durations: list[float] = []
        self.round_participants: list[int] = []
        self.round_convergence: list[float] = []
        self.round_losses: list[float] = []
        self.round_accuracy: list[dict[str, float]] = []
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

            # Trigger all participating edges concurrently — each trains only on
            # images it processed since the last round, so per-edge memory is small.
            for edge in edges:
                if self.rng.random() < self.config.participation_rate:
                    evt = edge.trigger_training(
                        self.round_number, self._current_convergence, self.config
                    )
                    training_events.append(evt)
                    active_edges.append(edge)

            if training_events:
                yield simpy.AllOf(self.env, training_events)

            log.info(
                "FL round %d — aggregating %d edge(s) ...",
                self.round_number, len(active_edges),
            )

            state_dicts = []
            losses = []
            for edge in active_edges:
                sd = edge.get_trained_state()
                if sd is not None:
                    state_dicts.append(sd)
                    losses.append(edge.get_training_loss())

            if state_dicts:
                avg_state = self._fedavg(state_dicts)
                self._broadcast(edges, avg_state)

                if self.global_model is not None:
                    self.global_model.load_state_dict(avg_state)

                avg_loss = sum(losses) / len(losses) if losses else 0.0
                self.round_losses.append(avg_loss)

                if self.val_loader is not None and self.global_model is not None:
                    log.info("FL round %d — evaluating global model on validation set ...", self.round_number)
                    acc = evaluate(self.global_model, self.val_loader)
                    self.round_accuracy.append(acc)
                    mAP = acc.get("mAP", 0.0)
                    self._current_convergence = mAP
                    log.info(
                        "FL round %d complete — loss=%.4f  mAP=%.4f",
                        self.round_number, avg_loss, mAP,
                    )
                else:
                    self._current_convergence = self._compute_convergence(self.round_number)
                    log.info(
                        "FL round %d complete — loss=%.4f  convergence=%.4f",
                        self.round_number, avg_loss, self._current_convergence,
                    )
            else:
                self._current_convergence = self._compute_convergence(self.round_number)
                log.info(
                    "FL round %d — no participants, convergence=%.4f",
                    self.round_number, self._current_convergence,
                )

            round_duration = self.env.now - round_start
            self.round_durations.append(round_duration)
            self.round_participants.append(len(active_edges))
            self.aggregation_count += 1
            self.round_convergence.append(self._current_convergence)

            callback(
                round_number=self.round_number,
                duration=round_duration,
                participants=len(active_edges),
                convergence=self._current_convergence,
            )

            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)

    def _compute_convergence(self, round_num: int) -> float:
        ceiling = self.config.convergence_ceiling
        speed = self.config.convergence_speed
        return ceiling * (1.0 - math.exp(-speed * round_num))

    @staticmethod
    def _fedavg(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        avg: OrderedDict[str, torch.Tensor] = OrderedDict()
        keys = state_dicts[0].keys()
        for key in keys:
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            avg[key] = stacked.mean(dim=0)
        return avg

    def _broadcast(self, edges: list, state_dict: dict[str, torch.Tensor]) -> None:
        for edge in edges:
            if edge.model is not None:
                edge.model.load_state_dict(state_dict)

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
