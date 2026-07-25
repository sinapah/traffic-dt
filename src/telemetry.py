from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TelemetryPacket:
    timestamp: float
    edge_id: int
    queue_length: int
    queue_capacity: int
    utilization: float
    processing_rate: float
    arrival_rate: float
    total_processed: int
    total_dropped: int
    mean_wait_time: float
    training_active: bool
    fl_round: int
    fl_convergence: float
    local_epochs: int
    sampling_rate: float


@dataclass
class EdgeState:
    edge_id: int
    queue_length: int = 0
    queue_capacity: int = 500
    utilization: float = 0.0
    processing_rate: float = 0.0
    arrival_rate: float = 0.0
    total_processed: int = 0
    total_dropped: int = 0
    mean_wait_time: float = 0.0
    training_active: bool = False
    local_epochs: int = 2
    sampling_rate: float = 1.0


@dataclass
class SystemState:
    timestamp: float = 0.0
    edges: dict[int, EdgeState] = field(default_factory=dict)
    fl_round: int = 0
    fl_convergence: float = 0.0
    active_participants: int = 0
    aggregate_arrival_rate: float = 0.0
    aggregate_processing_rate: float = 0.0
    aggregate_queue_length: int = 0
    aggregate_dropped: int = 0


class TelemetryBus:
    def __init__(self):
        self._packets: list[TelemetryPacket] = []
        self._subscribers: list = []
        self._outage_active = False

    def subscribe(self, callback) -> None:
        self._subscribers.append(callback)

    def set_outage(self, active: bool) -> None:
        self._outage_active = active

    @property
    def outage_active(self) -> bool:
        return self._outage_active

    def publish(self, packet: TelemetryPacket) -> None:
        self._packets.append(packet)
        if not self._outage_active:
            for cb in self._subscribers:
                cb(packet)

    @property
    def all_packets(self) -> list[TelemetryPacket]:
        return self._packets
