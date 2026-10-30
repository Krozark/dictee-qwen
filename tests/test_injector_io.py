import subprocess

import pytest

from dictee.injector import get_active_window_id, get_window_name, inject_text, set_clipboard


class FakeResult:
    def __init__(self, stdout=""):
        self.stdout = stdout


class FakeRunner:
    def __init__(self, stdout_by_args=None):
        self.calls = []
        self._stdout_by_args = stdout_by_args or {}

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        return FakeResult(stdout=self._stdout_by_args.get(tuple(args), ""))


def test_get_active_window_id_parses_stdout():
    runner = FakeRunner({("xdotool", "getactivewindow"): "12345\n"})
    assert get_active_window_id(runner=runner) == "12345"
    assert runner.calls == [(["xdotool", "getactivewindow"], {"capture_output": True, "check": True, "text": True})]


def test_get_window_name_parses_stdout():
    runner = FakeRunner({("xdotool", "getwindowname", "12345"): "Mozilla Firefox\n"})
    assert get_window_name("12345", runner=runner) == "Mozilla Firefox"


def test_set_clipboard_pipes_text_to_xclip():
    runner = FakeRunner()
    set_clipboard("bonjour", runner=runner)
    assert runner.calls == [
        (["xclip", "-selection", "clipboard"], {"input": "bonjour", "check": True, "text": True})
    ]


def test_inject_text_pastes_directly_when_target_already_active():
    runner = FakeRunner({("xdotool", "getactivewindow"): "12345\n"})
    inject_text("12345", "bonjour", runner=runner)
    assert runner.calls == [
        (["xclip", "-selection", "clipboard"], {"input": "bonjour", "check": True, "text": True}),
        (["xdotool", "getactivewindow"], {"capture_output": True, "check": True, "text": True}),
        (["xdotool", "key", "ctrl+v"], {"check": True}),
    ]


def test_inject_text_activates_target_then_restores_previous_window():
    runner = FakeRunner({("xdotool", "getactivewindow"): "99999\n"})
    inject_text("12345", "bonjour", runner=runner)
    assert runner.calls == [
        (["xclip", "-selection", "clipboard"], {"input": "bonjour", "check": True, "text": True}),
        (["xdotool", "getactivewindow"], {"capture_output": True, "check": True, "text": True}),
        (["xdotool", "windowactivate", "--sync", "12345"], {"check": True}),
        (["xdotool", "key", "ctrl+v"], {"check": True}),
        (["xdotool", "windowactivate", "--sync", "99999"], {"check": True}),
    ]


def test_inject_text_propagates_runner_failure():
    def failing_runner(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    with pytest.raises(subprocess.CalledProcessError):
        inject_text("12345", "bonjour", runner=failing_runner)
