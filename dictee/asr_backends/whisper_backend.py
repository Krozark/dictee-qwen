from __future__ import annotations

import io
import wave
from typing import Callable

import numpy as np
import requests

from .base import AsrBackend

HttpPost = Callable[..., "requests.Response"]

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 30


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class WhisperBackend(AsrBackend):
    def __init__(self, base_url: str, http_post: HttpPost = requests.post):
        self._base_url = base_url.rstrip("/")
        self._http_post = http_post

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        wav_bytes = encode_wav(audio, sample_rate)
        response = self._http_post(
            f"{self._base_url}/inference",
            files={"file": ("segment.wav", wav_bytes, "audio/wav")},
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return response.json()["text"].strip()
