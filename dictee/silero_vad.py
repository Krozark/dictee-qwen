from __future__ import annotations

from typing import Callable

import numpy as np
import torch

SPEECH_THRESHOLD = 0.5

ModelCall = Callable[[np.ndarray, int], float]


class SileroVad:
    def __init__(self, model_call: ModelCall, sample_rate: int = 16000):
        self._model_call = model_call
        self._sample_rate = sample_rate

    def is_speech(self, frame: np.ndarray) -> bool:
        probability = self._model_call(frame, self._sample_rate)
        return probability >= SPEECH_THRESHOLD


def load_silero_vad(sample_rate: int = 16000) -> SileroVad:
    """Loads the real Silero VAD model via torch.hub. Requires network access
    on first run (downloads and caches the model) and `torch`/`torchaudio`
    installed. Not covered by unit tests - exercised by the manual
    integration checklist in README.md."""
    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )

    def model_call(frame: np.ndarray, sr: int) -> float:
        tensor = torch.from_numpy(frame).float()
        return float(model(tensor, sr).item())

    return SileroVad(model_call=model_call, sample_rate=sample_rate)
