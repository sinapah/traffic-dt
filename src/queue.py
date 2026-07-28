from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Frame:
    timestamp: float
    camera_id: int
    frame_id: int
    image_path: str | None = None


class BoundedQueue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._queue: deque[Frame] = deque()
        self.total_arrivals = 0
        self.total_dropped = 0
        self.total_processed = 0
        self._wait_times: list[float] = []

    @property
    def length(self) -> int:
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self.capacity

    @property
    def mean_wait_time(self) -> float:
        return sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0

    def put(self, frame: Frame) -> bool:
        self.total_arrivals += 1
        if self.is_full:
            self.total_dropped += 1
            return False
        self._queue.append(frame)
        return True

    def get(self, current_time: float) -> Frame | None:
        if not self._queue:
            return None
        frame = self._queue.popleft()
        self._wait_times.append(current_time - frame.timestamp)
        self.total_processed += 1
        return frame

    def reset_wait_times(self) -> None:
        self._wait_times.clear()
