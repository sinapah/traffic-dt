from __future__ import annotations

import simpy
import numpy as np

from src.config import CameraConfig, SimulationConfig
from src.queue import BoundedQueue, Frame
from src.workload import WorkloadSchedule


class Camera:
    def __init__(
        self,
        env: simpy.Environment,
        config: CameraConfig,
        sim_config: SimulationConfig,
        queue: BoundedQueue,
        workload: WorkloadSchedule,
        rng: np.random.Generator,
    ):
        self.env = env
        self.config = config
        self.sim_config = sim_config
        self.queue = queue
        self.workload = workload
        self.rng = rng
        self.frame_counter = 0

    def run(self) -> simpy.Process:
        return self.env.process(self._generate_frames())

    def _generate_frames(self) -> simpy.Generator:
        while True:
            rate = self.workload.get_arrival_rate(self.env.now, self.config.base_fps)
            rate = max(rate, 0.01)
            inter_arrival = self.rng.exponential(1.0 / rate)
            yield self.env.timeout(inter_arrival)
            self.frame_counter += 1
            frame = Frame(
                timestamp=self.env.now,
                camera_id=self.config.camera_id,
                frame_id=self.frame_counter,
            )
            self.queue.put(frame)
