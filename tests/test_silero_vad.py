import numpy as np

from dictee.silero_vad import SileroVad


def test_is_speech_true_when_probability_above_threshold():
    vad = SileroVad(model_call=lambda frame, sr: 0.9)
    assert vad.is_speech(np.zeros(512, dtype=np.float32)) is True


def test_is_speech_false_when_probability_below_threshold():
    vad = SileroVad(model_call=lambda frame, sr: 0.1)
    assert vad.is_speech(np.zeros(512, dtype=np.float32)) is False


def test_is_speech_true_at_exact_threshold():
    vad = SileroVad(model_call=lambda frame, sr: 0.5)
    assert vad.is_speech(np.zeros(512, dtype=np.float32)) is True


def test_is_speech_passes_configured_sample_rate_to_model_call():
    received = {}

    def model_call(frame, sr):
        received["sr"] = sr
        return 1.0

    vad = SileroVad(model_call=model_call, sample_rate=8000)
    vad.is_speech(np.zeros(512, dtype=np.float32))
    assert received["sr"] == 8000
