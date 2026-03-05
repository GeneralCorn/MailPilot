"""Trace recording for runtime steps."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceEvent:
    """A single trace entry."""
    agent: str
    event: str
    time: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "event": self.event,
            "time": self.time.isoformat(),
            **self.payload,
        }


class TraceRecord:
    """Append-only trace of runtime events."""
    _events: list[TraceEvent]

    def __init__(self) -> None:
        self._events = []

    def append(self, agent: str, event: str, **payload: Any) -> None:
        self._events.append(
            TraceEvent(agent=agent, event=event, time=datetime.now(timezone.utc), payload=dict(payload))
        )

    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def clear(self) -> None:
        self._events.clear()
