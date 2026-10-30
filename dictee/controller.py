from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .injector import join_segment, should_inject


class VadModel(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...


class VadSegmenterProtocol(Protocol):
    def feed_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray | None: ...


class AsrBackendProtocol(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class BackupWriterProtocol(Protocol):
    def start_session(self) -> None: ...
    def append_segment(self, text: str) -> None: ...


class GuiProtocol(Protocol):
    def set_status(self, state: str) -> None: ...
    def append_segment(self, text: str) -> None: ...


class Controller:
    def __init__(
        self,
        *,
        vad_segmenter: VadSegmenterProtocol,
        vad_model: VadModel,
        asr_backend: AsrBackendProtocol,
        backup_writer: BackupWriterProtocol,
        gui: GuiProtocol,
        get_active_window_id: Callable[[], str],
        get_window_name: Callable[[str], str],
        inject_text: Callable[[str, str], None],
        set_clipboard: Callable[[str], None],
        sample_rate: int,
    ):
        self._vad_segmenter = vad_segmenter
        self._vad_model = vad_model
        self._asr_backend = asr_backend
        self._backup_writer = backup_writer
        self._gui = gui
        self._get_active_window_id = get_active_window_id
        self._get_window_name = get_window_name
        self._inject_text = inject_text
        self._set_clipboard = set_clipboard
        self._sample_rate = sample_rate
        self.state = "idle"
        self.transcript = ""

    def toggle_record(self) -> None:
        if self.state == "idle":
            self.transcript = ""
            self._backup_writer.start_session()
            self.state = "recording"
        else:
            self.state = "idle"
        self._gui.set_status(self.state)

    def toggle_pause(self) -> None:
        if self.state == "recording":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "recording"
        else:
            return
        self._gui.set_status(self.state)

    def copy(self) -> None:
        self._set_clipboard(self.transcript)

    def process_frame(self, frame: np.ndarray) -> None:
        if self.state != "recording":
            return
        is_speech = self._vad_model.is_speech(frame)
        segment_audio = self._vad_segmenter.feed_frame(frame, is_speech)
        if segment_audio is None:
            return
        text = self._asr_backend.transcribe(segment_audio, self._sample_rate)
        if not text:
            return
        self.transcript = join_segment(self.transcript, text)
        self._backup_writer.append_segment(text)
        self._gui.append_segment(text)
        self._maybe_inject(text)

    def _maybe_inject(self, text: str) -> None:
        window_id = self._get_active_window_id()
        window_name = self._get_window_name(window_id)
        if should_inject(window_name):
            self._inject_text(window_id, text)
