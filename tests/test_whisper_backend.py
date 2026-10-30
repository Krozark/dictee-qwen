import wave
import io

import numpy as np

from dictee.asr_backends.whisper_backend import WhisperBackend, encode_wav


def test_encode_wav_produces_a_valid_mono_16bit_wav():
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    wav_bytes = encode_wav(audio, sample_rate=16000)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 5


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True

    def json(self):
        return self._json_data


def test_whisper_backend_posts_wav_and_returns_stripped_text():
    calls = []

    def fake_http_post(url, files):
        calls.append((url, files))
        return FakeResponse({"text": " bonjour le monde "})

    backend = WhisperBackend(base_url="http://localhost:8080", http_post=fake_http_post)
    result = backend.transcribe(np.zeros(1600, dtype=np.float32), sample_rate=16000)

    assert result == "bonjour le monde"
    assert len(calls) == 1
    url, files = calls[0]
    assert url == "http://localhost:8080/inference"
    assert "file" in files


def test_whisper_backend_strips_trailing_slash_from_base_url():
    captured_urls = []

    def fake_http_post(url, files):
        captured_urls.append(url)
        return FakeResponse({"text": "bonjour"})

    backend = WhisperBackend(base_url="http://localhost:8080/", http_post=fake_http_post)
    backend.transcribe(np.zeros(1600, dtype=np.float32), sample_rate=16000)

    assert captured_urls == ["http://localhost:8080/inference"]
