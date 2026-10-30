import numpy as np
import pytest

from dictee.asr_backends.base import AsrBackend


def test_asr_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AsrBackend()


def test_minimal_subclass_can_be_instantiated_and_called():
    class FakeBackend(AsrBackend):
        def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
            return "bonjour"

    backend = FakeBackend()
    assert backend.transcribe(np.zeros(10), 16000) == "bonjour"
    backend.close()  # default no-op must not raise
