import pytest

from dictee.gui import button_labels


def test_button_labels_for_idle():
    assert button_labels("idle") == ("Démarrer", "Mettre en pause")


def test_button_labels_for_recording():
    assert button_labels("recording") == ("Arrêter", "Mettre en pause")


def test_button_labels_for_paused():
    assert button_labels("paused") == ("Arrêter", "Reprendre")


def test_button_labels_raises_for_unknown_state():
    with pytest.raises(ValueError, match="unknown"):
        button_labels("unknown")
