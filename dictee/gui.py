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


_BUTTON_LABELS = {
    "idle": ("Démarrer", "Mettre en pause"),
    "recording": ("Arrêter", "Mettre en pause"),
    "paused": ("Arrêter", "Reprendre"),
}


def button_labels(state: str) -> tuple[str, str]:
    try:
        return _BUTTON_LABELS[state]
    except KeyError:
        raise ValueError(f"unknown state {state!r}") from None


import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

from .injector import APP_WINDOW_TITLE  # noqa: E402


class DictationWindow(Gtk.Window):
    def __init__(self, *, on_toggle_record, on_toggle_pause, on_copy):
        super().__init__(title=APP_WINDOW_TITLE)
        self._on_toggle_record = on_toggle_record
        self._on_toggle_pause = on_toggle_pause
        self._on_copy = on_copy

        self.set_default_size(360, 240)
        self.set_resizable(True)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(box)

        self._status_label = Gtk.Label(label="Idle")
        box.pack_start(self._status_label, False, False, 4)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._text_view = Gtk.TextView()
        self._text_view.set_editable(False)
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        scroller.add(self._text_view)
        box.pack_start(scroller, True, True, 0)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.pack_start(button_box, False, False, 4)

        initial_record_label, initial_pause_label = button_labels("idle")

        self._record_button = Gtk.Button(label=initial_record_label)
        self._record_button.connect("clicked", lambda _btn: self._on_toggle_record())
        button_box.pack_start(self._record_button, True, True, 0)

        self._pause_button = Gtk.Button(label=initial_pause_label)
        self._pause_button.connect("clicked", lambda _btn: self._on_toggle_pause())
        button_box.pack_start(self._pause_button, True, True, 0)

        copy_button = Gtk.Button(label="Copier")
        copy_button.connect("clicked", lambda _btn: self._on_copy())
        button_box.pack_start(copy_button, True, True, 0)

        self.connect("destroy", lambda _win: Gtk.main_quit())

    def append_segment(self, text: str) -> None:
        GLib.idle_add(self._append_segment_on_main_thread, text)

    def _append_segment_on_main_thread(self, text: str) -> bool:
        buffer = self._text_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), text + " ")
        return False

    def set_status(self, state: str) -> None:
        GLib.idle_add(self._set_status_on_main_thread, state)

    def _set_status_on_main_thread(self, state: str) -> bool:
        text, _color = status_label(state)
        self._status_label.set_text(text)
        record_label, pause_label = button_labels(state)
        self._record_button.set_label(record_label)
        self._pause_button.set_label(pause_label)
        return False

    def clear_transcript(self) -> None:
        GLib.idle_add(self._clear_transcript_on_main_thread)

    def _clear_transcript_on_main_thread(self) -> bool:
        self._text_view.get_buffer().set_text("")
        return False
