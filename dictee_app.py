#!/usr/bin/env python3
from __future__ import annotations

import logging
import queue
import signal
import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib  # noqa: E402

from dictee.asr_backends.qwen_backend import QwenBackend
from dictee.asr_backends.whisper_backend import WhisperBackend
from dictee.audio_capture import SAMPLE_RATE, AudioCapture
from dictee.backup_writer import BackupWriter
from dictee.config import load_config
from dictee.controller import Controller
from dictee.gui import DictationWindow, TrayIcon
from dictee.hotkeys import HotkeyListener
from dictee.injector import get_active_window_id, get_window_name, inject_text, set_clipboard
from dictee.pid_lock import AlreadyRunningError, PidLock
from dictee.silero_vad import load_silero_vad
from dictee.vad_segmenter import VadSegmenter

PID_LOCK_PATH = "/tmp/dictee_app.pid"

QWEN_MODEL_IDS = {
    "qwen-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
    "qwen-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
}


def build_asr_backend(config):
    if config.backend in QWEN_MODEL_IDS:
        return QwenBackend(QWEN_MODEL_IDS[config.backend], language=config.language)
    if config.backend == "whisper":
        return WhisperBackend(base_url="http://localhost:8080")
    raise ValueError(f"Unknown backend {config.backend!r}")


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    config = load_config(config_path)

    lock = PidLock(PID_LOCK_PATH)
    try:
        lock.acquire()
    except AlreadyRunningError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        asr_backend = build_asr_backend(config)
        backup_writer = BackupWriter(config.backup_file)
        vad_segmenter = VadSegmenter(
            frame_duration_ms=32,  # 512 samples at 16kHz
            silence_ms=config.vad_silence_ms,
            min_segment_ms=config.vad_min_segment_ms,
            max_segment_ms=config.vad_max_segment_ms,
        )
        vad_model = load_silero_vad(sample_rate=SAMPLE_RATE)

        frame_queue: "queue.Queue" = queue.Queue()
        audio_capture = AudioCapture(frame_queue)

        window = DictationWindow(
            on_toggle_record=lambda: controller.toggle_record() or audio_toggle(),
            on_toggle_pause=lambda: controller.toggle_pause() or audio_pause_toggle(),
            on_copy=lambda: controller.copy(),
        )

        controller = Controller(
            vad_segmenter=vad_segmenter,
            vad_model=vad_model,
            asr_backend=asr_backend,
            backup_writer=backup_writer,
            gui=window,
            get_active_window_id=get_active_window_id,
            get_window_name=get_window_name,
            inject_text=inject_text,
            set_clipboard=set_clipboard,
            sample_rate=SAMPLE_RATE,
        )

        def audio_toggle():
            if controller.state == "recording":
                audio_capture.start()
            else:
                audio_capture.stop()

        def audio_pause_toggle():
            if controller.state == "recording":
                audio_capture.start()
            elif controller.state == "paused":
                audio_capture.stop()

        def process_frames_loop():
            while True:
                frame = frame_queue.get()
                controller.process_frame(frame)

        processing_thread = threading.Thread(target=process_frames_loop, daemon=True)
        processing_thread.start()

        hotkeys = HotkeyListener(
            {
                config.hotkey_toggle_record: lambda: (controller.toggle_record(), audio_toggle()),
                config.hotkey_toggle_pause: lambda: (controller.toggle_pause(), audio_pause_toggle()),
                config.hotkey_copy: lambda: controller.copy(),
            }
        )
        hotkeys.grab_all()
        hotkey_thread = threading.Thread(target=hotkeys.run, daemon=True)
        hotkey_thread.start()

        def shutdown(*_args):
            audio_capture.stop()
            asr_backend.close()
            if config.backend == "whisper":
                pass  # whisper-server lifecycle is managed outside this process for now
            lock.release()
            Gtk.main_quit()
            return False

        tray_icon = TrayIcon(
            get_state=lambda: controller.state,
            on_toggle_record=lambda: controller.toggle_record() or audio_toggle(),
            on_toggle_pause=lambda: controller.toggle_pause() or audio_pause_toggle(),
            on_copy=lambda: controller.copy(),
            on_show_window=window.present,
            on_quit=shutdown,
        )
    except Exception as exc:
        lock.release()
        print(f"Failed to start dictee: {exc}", file=sys.stderr)
        return 1

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, shutdown)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, shutdown)

    window.show_all()
    try:
        Gtk.main()
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
