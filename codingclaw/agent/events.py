from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def make_event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
