import pytest
from Xlib import X, XK

from dictee.hotkeys import HotkeyListener, parse_shortcut


def test_parse_shortcut_single_modifier():
    assert parse_shortcut("<Super>d") == (X.Mod4Mask, "d")


def test_parse_shortcut_stacked_modifiers():
    assert parse_shortcut("<Ctrl><Alt>p") == (X.ControlMask | X.Mod1Mask, "p")


def test_parse_shortcut_rejects_unknown_modifier():
    with pytest.raises(ValueError, match="Foo"):
        parse_shortcut("<Foo>d")


def test_parse_shortcut_rejects_missing_key():
    with pytest.raises(ValueError, match="no key"):
        parse_shortcut("<Super>")


class FakeRootWindow:
    def __init__(self):
        self.grab_calls = []

    def grab_key(self, keycode, modifiers, owner_events, pointer_mode, keyboard_mode):
        self.grab_calls.append((keycode, modifiers))


class FakeScreenInfo:
    def __init__(self, root):
        self.root = root


class FakeDisplay:
    def __init__(self):
        self.root = FakeRootWindow()
        self._screen_info = FakeScreenInfo(self.root)

    def screen(self):
        return self._screen_info

    def keysym_to_keycode(self, keysym):
        return keysym  # identity stand-in, sufficient to verify wiring


def test_grab_all_grabs_each_configured_shortcut():
    fake_display = FakeDisplay()
    listener = HotkeyListener(
        {"<Super>d": lambda: None, "<Super>p": lambda: None},
        display_factory=lambda: fake_display,
    )

    listener.grab_all()

    lock_variants = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)
    expected = {
        (XK.string_to_keysym("d"), X.Mod4Mask | lock_bits) for lock_bits in lock_variants
    } | {
        (XK.string_to_keysym("p"), X.Mod4Mask | lock_bits) for lock_bits in lock_variants
    }
    assert set(fake_display.root.grab_calls) == expected


def test_grab_all_registers_callbacks_by_keycode_and_modifiers():
    fake_display = FakeDisplay()
    calls = []
    listener = HotkeyListener(
        {"<Super>d": lambda: calls.append("d")}, display_factory=lambda: fake_display
    )
    listener.grab_all()

    keysym = XK.string_to_keysym("d")
    callback = listener._keycode_map[(keysym, X.Mod4Mask)]
    callback()

    assert calls == ["d"]


def test_grab_all_grabs_lock_modifier_variants():
    fake_display = FakeDisplay()
    listener = HotkeyListener({"<Super>d": lambda: None}, display_factory=lambda: fake_display)
    listener.grab_all()

    keysym = XK.string_to_keysym("d")
    expected = {
        (keysym, X.Mod4Mask),
        (keysym, X.Mod4Mask | X.LockMask),
        (keysym, X.Mod4Mask | X.Mod2Mask),
        (keysym, X.Mod4Mask | X.LockMask | X.Mod2Mask),
    }
    assert set(fake_display.root.grab_calls) == expected


def test_grab_all_rejects_conflicting_shortcuts():
    fake_display = FakeDisplay()
    # Modifier order doesn't affect the resulting bitmask, so these two
    # distinct shortcut strings both resolve to the same (keycode, modifiers).
    listener = HotkeyListener(
        {"<Ctrl><Alt>p": lambda: None, "<Alt><Ctrl>p": lambda: None},
        display_factory=lambda: fake_display,
    )
    with pytest.raises(ValueError, match="conflicts"):
        listener.grab_all()
