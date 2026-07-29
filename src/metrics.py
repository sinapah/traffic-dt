from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricEntry:
    timestamp: float
    name: str
    value: float
    tags: dict = field(default_factory=dict)


class MetricsCollector:
    def __init__(self):
        self._entries: list[MetricEntry] = []

    def record(self, timestamp: float, name: str, value: float, **tags) -> None:
        self._entries.append(MetricEntry(timestamp=timestamp, name=name, value=value, tags=tags))

    @property
    def entries(self) -> list[MetricEntry]:
        return self._entries

    def get_series(self, name: str, edge_id: int | None = None) -> tuple[list[float], list[float]]:
        times = []
        values = []
        for e in self._entries:
            if e.name == name:
                if edge_id is not None and e.tags.get("edge_id") != edge_id:
                    continue
                times.append(e.timestamp)
                values.append(e.value)
        return times, values

    def get_summary(self) -> dict[str, dict[str, float]]:
        from collections import defaultdict
        grouped: dict[str, list[float]] = defaultdict(list)
        for e in self._entries:
            grouped[e.name].append(e.value)
        summary = {}
        for name, vals in grouped.items():
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            summary[name] = {
                "count": n,
                "mean": sum(vals) / n if n else 0,
                "min": vals_sorted[0] if n else 0,
                "max": vals_sorted[-1] if n else 0,
                "p50": vals_sorted[n // 2] if n else 0,
                "p95": vals_sorted[int(n * 0.95)] if n else 0,
            }
        return summary

    def to_dicts(self) -> list[dict]:
        return [
            {
                "timestamp": e.timestamp,
                "name": e.name,
                "value": e.value,
                **e.tags,
            }
            for e in self._entries
        ]

    @classmethod
    def from_dicts(cls, data: list[dict]) -> MetricsCollector:
        mc = cls()
        for d in data:
            tags = {k: v for k, v in d.items() if k not in ("timestamp", "name", "value")}
            mc.record(d["timestamp"], d["name"], d["value"], **tags)
        return mc
