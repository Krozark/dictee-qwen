from __future__ import annotations

import logging
import queue
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 512
SAMPLE_RATE = 16000

StreamFactory = Callable[..., object]


class AudioCapture:
    def __init__(
        self,
        frame_queue: "queue.Queue[np.ndarray]",
        stream_factory: StreamFactory = sd.InputStream,
    ):
        self._frame_queue = frame_queue
        self._stream_factory = stream_factory
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("audio input status: %s", status)
        self._frame_queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._stream is not None:
            self.stop()
        self._stream = self._stream_factory(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SAMPLES,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
