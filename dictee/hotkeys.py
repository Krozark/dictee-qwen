from __future__ import annotations

from typing import Callable

from Xlib import X, XK
from Xlib.display import Display

_MODIFIER_MASKS = {
    "Super": X.Mod4Mask,
    "Ctrl": X.ControlMask,
    "Alt": X.Mod1Mask,
    "Shift": X.ShiftMask,
}


def parse_shortcut(shortcut: str) -> tuple[int, str]:
    modifiers = 0
    remainder = shortcut
    while remainder.startswith("<"):
        end = remainder.index(">")
        name = remainder[1:end]
        if name not in _MODIFIER_MASKS:
            raise ValueError(f"Unknown modifier {name!r} in shortcut {shortcut!r}")
        modifiers |= _MODIFIER_MASKS[name]
        remainder = remainder[end + 1:]
    if not remainder:
        raise ValueError(f"Shortcut {shortcut!r} has no key")
    return modifiers, remainder


class HotkeyListener:
    def __init__(
        self,
        bindings: dict[str, Callable[[], None]],
        display_factory: Callable[[], "Display"] = Display,
    ):
        self._bindings = bindings
        self._display = display_factory()
        self._root = self._display.screen().root
        self._keycode_map: dict[tuple[int, int], Callable[[], None]] = {}

    def grab_all(self) -> None:
        for shortcut, callback in self._bindings.items():
            modifiers, key = parse_shortcut(shortcut)
            keysym = XK.string_to_keysym(key)
            keycode = self._display.keysym_to_keycode(keysym)
            self._root.grab_key(keycode, modifiers, True, X.GrabModeAsync, X.GrabModeAsync)
            self._keycode_map[(keycode, modifiers)] = callback

    def run(self) -> None:
        """Blocks, dispatching matching key-press events to their callback.
        Requires a live X11 connection; exercised by the manual integration
        checklist in README.md, not by unit tests."""
        while True:
            event = self._display.next_event()
            if event.type == X.KeyPress:
                callback = self._keycode_map.get((event.detail, event.state))
                if callback is not None:
                    callback()
