from __future__ import annotations

import subprocess
import time
from typing import Callable

APP_WINDOW_TITLE = "DicteeQwenOverlay"

# xdotool's `windowactivate --sync` only waits for the window manager to report
# the activation as complete (e.g. the _NET_ACTIVE_WINDOW property changing). It
# does NOT guarantee the target application has finished processing the
# focus-in event and is ready to receive a paste keystroke. These settle delays
# bridge that race window; tune them here if a particular WM/app combo still
# drops or truncates pastes.
ACTIVATE_SETTLE_SECONDS = 0.15
PASTE_SETTLE_SECONDS = 0.1

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


def inject_text(
    window_id: str,
    text: str,
    runner: Runner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    set_clipboard(text, runner=runner)
    previous_window_id = get_active_window_id(runner=runner)
    needs_focus_switch = previous_window_id != window_id
    if needs_focus_switch:
        runner(["xdotool", "windowactivate", "--sync", window_id], check=True)
        sleep(ACTIVATE_SETTLE_SECONDS)
    runner(["xdotool", "key", "ctrl+v"], check=True)
    if needs_focus_switch:
        sleep(PASTE_SETTLE_SECONDS)
        runner(["xdotool", "windowactivate", "--sync", previous_window_id], check=True)
