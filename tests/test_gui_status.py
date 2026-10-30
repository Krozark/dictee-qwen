import pytest

from dictee.gui import status_label


def test_status_label_for_idle():
    assert status_label("idle") == ("Idle", "grey")


def test_status_label_for_recording():
    assert status_label("recording") == ("Recording", "red")


def test_status_label_for_paused():
    assert status_label("paused") == ("Paused", "yellow")


def test_status_label_raises_for_unknown_state():
    with pytest.raises(ValueError, match="unknown"):
        status_label("unknown")
