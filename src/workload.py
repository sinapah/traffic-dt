from __future__ import annotations

import math
from src.config import SimulationConfig, WorkloadPhase


class WorkloadSchedule:
    def __init__(self, config: SimulationConfig):
        self.duration = config.duration
        if config.workload_schedule:
            self.phases = sorted(config.workload_schedule, key=lambda p: p.hour)
        else:
            self.phases = self._default_schedule()

    @staticmethod
    def _default_schedule() -> list[WorkloadPhase]:
        hours = list(range(49))
        multipliers = []
        for h in hours:
            real_hour = h % 24
            if 0 <= real_hour < 6:
                m = 0.2
            elif 6 <= real_hour < 9:
                m = 0.2 + (real_hour - 6) * (2.5 - 0.2) / 3.0
            elif 9 <= real_hour < 11:
                m = 2.5
            elif 11 <= real_hour < 14:
                m = 2.5 - (real_hour - 11) * (2.5 - 1.2) / 3.0
            elif 14 <= real_hour < 16:
                m = 1.2
            elif 16 <= real_hour < 19:
                m = 1.2 + (real_hour - 16) * (2.8 - 1.2) / 3.0
            elif 19 <= real_hour < 21:
                m = 2.8 - (real_hour - 19) * (2.8 - 0.5) / 2.0
            elif 21 <= real_hour < 24:
                m = 0.5 - (real_hour - 21) * (0.5 - 0.2) / 3.0
            else:
                m = 0.2
            multipliers.append(m)
        return [WorkloadPhase(hour=float(h), multiplier=m) for h, m in zip(hours, multipliers)]

    def get_multiplier(self, sim_time: float) -> float:
        if not self.phases:
            return 1.0
        t = sim_time % self.duration
        for i in range(len(self.phases) - 1):
            p0 = self.phases[i]
            p1 = self.phases[i + 1]
            if p0.hour <= t < p1.hour:
                frac = (t - p0.hour) / (p1.hour - p0.hour) if p1.hour > p0.hour else 0
                return p0.multiplier + frac * (p1.multiplier - p0.multiplier)
        return self.phases[-1].multiplier

    def get_arrival_rate(self, sim_time: float, base_fps: float) -> float:
        return base_fps * self.get_multiplier(sim_time)
