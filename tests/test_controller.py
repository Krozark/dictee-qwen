import numpy as np

from dictee.controller import Controller
from dictee.injector import APP_WINDOW_TITLE


class FakeVadModel:
    def __init__(self, is_speech=True):
        self.is_speech_value = is_speech

    def is_speech(self, frame):
        return self.is_speech_value


class FakeVadSegmenter:
    def __init__(self, results):
        self._results = list(results)

    def feed_frame(self, frame, is_speech):
        return self._results.pop(0) if self._results else None


class FakeAsrBackend:
    def __init__(self, text="bonjour"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        return self.text


class FakeBackupWriter:
    def __init__(self):
        self.started = False
        self.segments = []

    def start_session(self):
        self.started = True
        self.segments = []

    def append_segment(self, text):
        self.segments.append(text)


class FakeGui:
    def __init__(self):
        self.statuses = []
        self.segments = []
        self.clear_calls = 0

    def set_status(self, state):
        self.statuses.append(state)

    def append_segment(self, text):
        self.segments.append(text)

    def clear_transcript(self):
        self.clear_calls += 1


def make_controller(vad_results=(), asr_text="bonjour", active_window_title="Mozilla Firefox"):
    backup_writer = FakeBackupWriter()
    gui = FakeGui()
    asr_backend = FakeAsrBackend(text=asr_text)
    inject_calls = []
    clipboard_calls = []

    controller = Controller(
        vad_segmenter=FakeVadSegmenter(vad_results),
        vad_model=FakeVadModel(),
        asr_backend=asr_backend,
        backup_writer=backup_writer,
        gui=gui,
        get_active_window_id=lambda: "42",
        get_window_name=lambda window_id: active_window_title,
        inject_text=lambda text: inject_calls.append(text),
        set_clipboard=lambda text: clipboard_calls.append(text),
        sample_rate=16000,
    )
    return controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls


def test_toggle_record_starts_recording_resets_transcript_and_backup():
    controller, backup_writer, gui, *_ = make_controller()
    controller.toggle_record()
    assert controller.state == "recording"
    assert backup_writer.started is True
    assert gui.statuses == ["recording"]


def test_toggle_record_again_stops_recording():
    controller, *_ = make_controller()
    controller.toggle_record()
    controller.toggle_record()
    assert controller.state == "idle"


def test_toggle_pause_from_recording_goes_to_paused_and_back():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller()
    controller.toggle_record()
    controller.toggle_pause()
    assert controller.state == "paused"
    controller.toggle_pause()
    assert controller.state == "recording"


def test_toggle_pause_while_idle_is_noop():
    controller, backup_writer, gui, *_ = make_controller()
    controller.toggle_pause()
    assert controller.state == "idle"
    assert gui.statuses == []


def test_copy_sends_transcript_to_clipboard():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller()
    controller.transcript = "bonjour le monde"
    controller.copy()
    assert clipboard_calls == ["bonjour le monde"]


def test_process_frame_ignored_when_not_recording():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[np.zeros(10, dtype=np.float32)]
    )
    controller.process_frame(np.zeros(10, dtype=np.float32))
    assert asr_backend.calls == []


def test_process_frame_with_no_completed_segment_does_nothing():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[None]
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))
    assert asr_backend.calls == []
    assert gui.segments == []


def test_process_frame_with_completed_segment_updates_transcript_backup_gui_and_injects():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio], asr_text="bonjour", active_window_title="Mozilla Firefox"
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))

    assert controller.transcript == "bonjour"
    assert backup_writer.segments == ["bonjour"]
    assert gui.segments == ["bonjour"]
    assert inject_calls == ["bonjour"]
    assert asr_backend.calls == [(segment_audio, 16000)]


def test_process_frame_skips_injection_when_focus_is_own_window():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio], asr_text="bonjour", active_window_title=APP_WINDOW_TITLE
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))

    assert gui.segments == ["bonjour"]
    assert backup_writer.segments == ["bonjour"]
    assert inject_calls == []


def test_process_frame_ignores_empty_transcription():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio], asr_text=""
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))

    assert gui.segments == []
    assert backup_writer.segments == []
    assert inject_calls == []


def test_process_frame_ignored_while_paused():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio]
    )
    controller.toggle_record()
    controller.toggle_pause()
    controller.process_frame(np.zeros(10, dtype=np.float32))
    assert asr_backend.calls == []


def test_process_frame_catches_asr_backend_exception_and_continues():
    class RaisingAsrBackend:
        def transcribe(self, audio, sample_rate):
            raise RuntimeError("boom")

    backup_writer = FakeBackupWriter()
    gui = FakeGui()
    controller = Controller(
        vad_segmenter=FakeVadSegmenter([np.ones(10, dtype=np.float32)]),
        vad_model=FakeVadModel(),
        asr_backend=RaisingAsrBackend(),
        backup_writer=backup_writer,
        gui=gui,
        get_active_window_id=lambda: "42",
        get_window_name=lambda window_id: "Mozilla Firefox",
        inject_text=lambda text: None,
        set_clipboard=lambda text: None,
        sample_rate=16000,
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))  # must not raise

    assert controller.state == "recording"
    assert controller.transcript == ""
    assert backup_writer.segments == []
    assert gui.segments == []


def test_process_frame_catches_injection_exception_and_continues():
    segment_audio = np.ones(10, dtype=np.float32)
    backup_writer = FakeBackupWriter()
    gui = FakeGui()

    def raising_inject_text(text):
        raise RuntimeError("xdotool exploded")

    controller = Controller(
        vad_segmenter=FakeVadSegmenter([segment_audio]),
        vad_model=FakeVadModel(),
        asr_backend=FakeAsrBackend(text="bonjour"),
        backup_writer=backup_writer,
        gui=gui,
        get_active_window_id=lambda: "42",
        get_window_name=lambda window_id: "Mozilla Firefox",
        inject_text=raising_inject_text,
        set_clipboard=lambda text: None,
        sample_rate=16000,
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))  # must not raise

    assert controller.state == "recording"
    assert controller.transcript == "bonjour"
    assert backup_writer.segments == ["bonjour"]
    assert gui.segments == ["bonjour"]


def test_toggle_record_clears_gui_transcript_when_starting_a_new_session():
    controller, backup_writer, gui, *_ = make_controller()
    controller.toggle_record()
    assert gui.clear_calls == 1


def test_toggle_record_does_not_clear_gui_transcript_when_stopping():
    controller, backup_writer, gui, *_ = make_controller()
    controller.toggle_record()  # start
    controller.toggle_record()  # stop
    assert gui.clear_calls == 1  # only cleared once, on the start transition, not on stop
