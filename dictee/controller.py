from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

import numpy as np

from .injector import join_segment, should_inject

logger = logging.getLogger(__name__)


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
    def clear_transcript(self) -> None: ...


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
        inject_text: Callable[[str], None],
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
        self._lock = threading.Lock()
        self.state = "idle"
        self.transcript = ""

    def toggle_record(self) -> None:
        with self._lock:
            starting = self.state == "idle"
            if starting:
                self.transcript = ""
        if starting:
            self._backup_writer.start_session()
            self._gui.clear_transcript()
        with self._lock:
            self.state = "recording" if starting else "idle"
            new_state = self.state
        self._gui.set_status(new_state)

    def toggle_pause(self) -> None:
        with self._lock:
            if self.state == "recording":
                self.state = "paused"
            elif self.state == "paused":
                self.state = "recording"
            else:
                return
            new_state = self.state
        self._gui.set_status(new_state)

    def copy(self) -> None:
        with self._lock:
            transcript = self.transcript
        self._set_clipboard(transcript)

    def process_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if self.state != "recording":
                return
        is_speech = self._vad_model.is_speech(frame)
        segment_audio = self._vad_segmenter.feed_frame(frame, is_speech)
        if segment_audio is None:
            return
        try:
            text = self._asr_backend.transcribe(segment_audio, self._sample_rate)
        except Exception:
            logger.exception("ASR transcription failed for a segment; skipping it")
            return
        if not text:
            return
        with self._lock:
            previous_transcript = self.transcript
            self.transcript = join_segment(self.transcript, text)
            injected_text = self.transcript[len(previous_transcript) :]
        self._backup_writer.append_segment(text)
        self._gui.append_segment(text)
        self._maybe_inject(injected_text)

    def _maybe_inject(self, text: str) -> None:
        try:
            window_id = self._get_active_window_id()
            window_name = self._get_window_name(window_id)
            if should_inject(window_name):
                self._inject_text(text)
        except Exception:
            logger.exception("Failed to inject transcribed text into the active window")
