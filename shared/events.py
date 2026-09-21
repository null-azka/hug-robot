"""
Shared event schema for the hug robot.

Every trigger source (Module 1, Module 2, and the button) emits an Event
with one of these types. The Brain (Module 4) doesn't care which module
fired -- it just watches for any of the three.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    FACE_SAD_DETECTED = "face_sad_detected"
    SPEECH_HUG_REQUEST = "speech_hug_request"
    BUTTON_PRESSED = "button_pressed"


@dataclass
class Event:
    type: EventType
    source: str                    # which module fired it, e.g. "module1_face"
    timestamp: float                # time.time() when it fired
    meta: Optional[dict] = None     # optional extra info (e.g. confidence score)

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "Event":
        d = json.loads(s)
        return Event(
            type=EventType(d["type"]),
            source=d["source"],
            timestamp=d["timestamp"],
            meta=d.get("meta"),
        )


def make_event(event_type: EventType, source: str, **meta) -> Event:
    """Convenience constructor -- fills in the timestamp for you."""
    return Event(type=event_type, source=source, timestamp=time.time(), meta=meta or None)