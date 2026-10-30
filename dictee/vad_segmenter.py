from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VadSegmenter:
    frame_duration_ms: int
    silence_ms: int
    min_segment_ms: int
    max_segment_ms: int
    _frames: list = field(default_factory=list, init=False)
    _silence_accum_ms: int = field(default=0, init=False)
    _speech_accum_ms: int = field(default=0, init=False)

    def feed_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray | None:
        if is_speech:
            self._frames.append(frame)
            self._speech_accum_ms += self.frame_duration_ms
            self._silence_accum_ms = 0
            if self._speech_accum_ms >= self.max_segment_ms:
                return self._finalize(force=True)
            return None

        if not self._frames:
            return None

        self._silence_accum_ms += self.frame_duration_ms
        if self._silence_accum_ms >= self.silence_ms:
            return self._finalize(force=False)
        return None

    def _finalize(self, force: bool) -> np.ndarray | None:
        frames = self._frames
        duration_ms = self._speech_accum_ms
        self._frames = []
        self._silence_accum_ms = 0
        self._speech_accum_ms = 0
        if duration_ms < self.min_segment_ms and not force:
            return None
        return np.concatenate(frames)
