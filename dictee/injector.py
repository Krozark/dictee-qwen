from __future__ import annotations

APP_WINDOW_TITLE = "DicteeQwenOverlay"

_PUNCTUATION_NO_SPACE_BEFORE = {",", ".", "!", "?", ";", ":", ")", "]", "}", "…"}


def join_segment(existing_text: str, new_segment: str) -> str:
    existing_text = existing_text.strip()
    new_segment = new_segment.strip()
    if not new_segment:
        return existing_text
    if not existing_text:
        return new_segment
    if new_segment[0] in _PUNCTUATION_NO_SPACE_BEFORE:
        return existing_text + new_segment
    return existing_text + " " + new_segment


def should_inject(active_window_title: str) -> bool:
    return active_window_title != APP_WINDOW_TITLE
