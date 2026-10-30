import queue

import numpy as np

from dictee.audio_capture import FRAME_SAMPLES, SAMPLE_RATE, AudioCapture


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def test_start_creates_stream_with_expected_parameters():
    created = {}

    def stream_factory(**kwargs):
        created["kwargs"] = kwargs
        return FakeStream(**kwargs)

    capture = AudioCapture(queue.Queue(), stream_factory=stream_factory)
    capture.start()

    assert created["kwargs"]["samplerate"] == SAMPLE_RATE
    assert created["kwargs"]["channels"] == 1
    assert created["kwargs"]["blocksize"] == FRAME_SAMPLES
    assert created["kwargs"]["dtype"] == "float32"
    assert capture._stream.started is True


def test_callback_pushes_mono_frame_copy_to_queue():
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()
    capture = AudioCapture(frame_queue, stream_factory=FakeStream)
    indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

    capture._callback(indata, 3, None, None)

    pushed = frame_queue.get_nowait()
    np.testing.assert_array_equal(pushed, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    indata[0][0] = 99.0  # mutate original to prove the pushed array is a copy
    assert pushed[0] == np.float32(0.1)


def test_stop_stops_and_closes_stream():
    capture = AudioCapture(queue.Queue(), stream_factory=FakeStream)
    capture.start()
    stream = capture._stream
    capture.stop()

    assert stream.started is False
    assert stream.closed is True
    assert capture._stream is None
