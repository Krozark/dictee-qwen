from __future__ import annotations

_STATUS_DISPLAY = {
    "idle": ("Idle", "grey"),
    "recording": ("Recording", "red"),
    "paused": ("Paused", "yellow"),
}


def status_label(state: str) -> tuple[str, str]:
    try:
        return _STATUS_DISPLAY[state]
    except KeyError:
        raise ValueError(f"unknown state {state!r}") from None
