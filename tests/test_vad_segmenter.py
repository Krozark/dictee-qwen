import numpy as np

from dictee.vad_segmenter import VadSegmenter


def make_segmenter():
    return VadSegmenter(frame_duration_ms=20, silence_ms=100, min_segment_ms=40, max_segment_ms=200)


def frame(value=1.0):
    return np.full(10, value, dtype=np.float32)


def test_returns_none_while_speech_continues():
    seg = make_segmenter()
    assert seg.feed_frame(frame(), is_speech=True) is None
    assert seg.feed_frame(frame(), is_speech=True) is None
    assert seg.feed_frame(frame(), is_speech=True) is None


def test_finalizes_segment_after_silence_threshold():
    seg = make_segmenter()
    seg.feed_frame(frame(), is_speech=True)
    seg.feed_frame(frame(), is_speech=True)
    seg.feed_frame(frame(), is_speech=True)  # 60ms of speech, above the 40ms minimum
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 20ms silence
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 40ms silence
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 60ms silence
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 80ms silence
    result = seg.feed_frame(frame(0.0), is_speech=False)  # 100ms silence: finalize
    assert result is not None
    assert result.shape == (30,)


def test_discards_segment_shorter_than_min_duration():
    seg = make_segmenter()
    seg.feed_frame(frame(), is_speech=True)  # 20ms of speech, below the 40ms minimum
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 20ms silence
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 40ms silence
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 60ms silence
    assert seg.feed_frame(frame(0.0), is_speech=False) is None  # 80ms silence
    result = seg.feed_frame(frame(0.0), is_speech=False)  # 100ms silence: finalize, but discard (too short)
    assert result is None


def test_force_finalizes_at_max_duration():
    seg = make_segmenter()
    result = None
    for _ in range(10):  # 10 * 20ms = 200ms = max_segment_ms
        result = seg.feed_frame(frame(), is_speech=True)
    assert result is not None
    assert result.shape == (100,)


def test_state_resets_after_finalize_so_a_new_segment_can_start():
    seg = make_segmenter()
    for _ in range(10):
        seg.feed_frame(frame(), is_speech=True)
    assert seg.feed_frame(frame(), is_speech=True) is None
