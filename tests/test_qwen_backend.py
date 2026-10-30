import numpy as np
import pytest

from dictee.asr_backends.qwen_backend import AsrModelLoadError, QwenBackend


def test_wraps_cuda_oom_error_with_clear_message():
    def oom_loader(model_id):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(AsrModelLoadError, match="Qwen/Qwen3-ASR-1.7B-hf"):
        QwenBackend("Qwen/Qwen3-ASR-1.7B-hf", loader=oom_loader)


def test_reraises_non_oom_runtime_errors_unchanged():
    def failing_loader(model_id):
        raise RuntimeError("weights file not found")

    with pytest.raises(RuntimeError, match="weights file not found"):
        QwenBackend("Qwen/Qwen3-ASR-1.7B-hf", loader=failing_loader)


class FakeInputs(dict):
    def to(self, device, dtype):
        self.to_args = (device, dtype)
        return self


class FakeModel:
    device = "cuda:0"
    dtype = "bfloat16"

    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.zeros((1, kwargs["input_ids"].shape[1] + 3))


class FakeProcessor:
    def __init__(self):
        self.apply_calls = []
        self.decode_calls = []

    def apply_transcription_request(self, audio, language=None, **kwargs):
        self.apply_calls.append({"audio": audio, "language": language})
        return FakeInputs({"input_ids": np.zeros((1, 5))})

    def decode(self, generated_ids, return_format="raw"):
        self.decode_calls.append({"generated_ids": generated_ids, "return_format": return_format})
        return ["bonjour le monde"]


def test_transcribe_forces_configured_language_and_returns_decoded_text():
    fake_model = FakeModel()
    fake_processor = FakeProcessor()

    def loader(model_id):
        return fake_model, fake_processor

    backend = QwenBackend("Qwen/Qwen3-ASR-1.7B-hf", language="fr", loader=loader)
    audio = np.zeros(1600, dtype=np.float32)
    result = backend.transcribe(audio, sample_rate=16000)

    assert result == "bonjour le monde"
    assert fake_processor.apply_calls == [{"audio": audio, "language": "fr"}]
    assert fake_processor.decode_calls[0]["return_format"] == "transcription_only"
    assert len(fake_model.generate_calls) == 1
