from __future__ import annotations

import subprocess
from typing import Callable

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


Runner = Callable[..., "subprocess.CompletedProcess"]


def get_active_window_id(runner: Runner = subprocess.run) -> str:
    result = runner(["xdotool", "getactivewindow"], capture_output=True, check=True, text=True)
    return result.stdout.strip()


def get_window_name(window_id: str, runner: Runner = subprocess.run) -> str:
    result = runner(["xdotool", "getwindowname", window_id], capture_output=True, check=True, text=True)
    return result.stdout.strip()


def set_clipboard(text: str, runner: Runner = subprocess.run) -> None:
    runner(["xclip", "-selection", "clipboard"], input=text, check=True, text=True)


def inject_text(window_id: str, text: str, runner: Runner = subprocess.run) -> None:
    set_clipboard(text, runner=runner)
    runner(["xdotool", "key", "--window", window_id, "ctrl+v"], check=True)
