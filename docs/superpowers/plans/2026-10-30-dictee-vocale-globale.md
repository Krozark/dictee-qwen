# Dictée Vocale Globale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-process Linux Mint (X11/Cinnamon) application that transcribes French speech locally on GPU (Qwen3-ASR, with whisper.cpp as a fallback backend) and injects the text at the cursor in real time, with a floating GTK overlay and global keyboard shortcuts, as specified in `docs/superpowers/specs/2026-10-30-dictee-vocale-globale-design.md`.

**Architecture:** One process (`dictee_app.py`) hosts a GTK overlay, a global X11 hotkey listener, an audio-capture thread, a VAD-based segmenter, a pluggable ASR backend (Qwen3-ASR via `transformers`, or whisper.cpp via its HTTP server), and an `xdotool`/`xclip`-based injector. A `Controller` class holds the session state machine and is the single place that decides what happens to each transcribed segment (display, backup file, injection). All hardware-facing code (audio I/O, GTK, X11, subprocess calls, model loading) is wrapped behind small interfaces so the orchestration logic can be unit-tested with fakes.

**Tech Stack:** Python 3.11+, `transformers`/`torch` (CUDA), `sounddevice`, Silero VAD (`torch.hub`), `python-xlib`, `PyGObject` (GTK3), `xdotool`/`xclip` (subprocess), `requests` (whisper.cpp HTTP), `pytest`.

---

## Task 0: Project scaffolding

**Files:**
- Create: `dictee/__init__.py`
- Create: `dictee/asr_backends/__init__.py`
- Create: `tests/__init__.py` (not created — pytest discovers `tests/` without it)
- Create: `requirements.txt`

- [ ] **Step 1: Create package directories and empty `__init__.py` files**

Run:
```bash
mkdir -p dictee/asr_backends tests
touch dictee/__init__.py dictee/asr_backends/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
numpy
sounddevice
requests
python-xlib
PyGObject
transformers>=4.46
accelerate
tomli; python_version < "3.11"
pytest
```

Note (do not act on this now, it's for Task 18/`setup.sh`): `torch`, `torchaudio` and the Silero VAD weights are installed separately with the CUDA-matched index URL, not pinned here.

- [ ] **Step 3: Install dependencies into a venv and verify pytest runs**

Run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest --collect-only
```
Expected: exits 0, reports "no tests ran" (no test files yet).

- [ ] **Step 4: Commit**

```bash
git add dictee/__init__.py dictee/asr_backends/__init__.py requirements.txt
git commit -m "chore: scaffold dictee package and requirements"
```

---

## Task 1: Config loading (`dictee/config.py`)

**Files:**
- Create: `dictee/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import pytest

from dictee.config import Config, load_config


def test_load_config_returns_defaults_when_path_is_none():
    config = load_config(None)
    assert config == Config()


def test_load_config_returns_defaults_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.toml"
    config = load_config(str(missing))
    assert config == Config()


def test_load_config_overrides_fields_from_file(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text('backend = "whisper"\nlanguage = "en"\n')
    config = load_config(str(toml_path))
    assert config.backend == "whisper"
    assert config.language == "en"
    assert config.vad_silence_ms == Config().vad_silence_ms


def test_load_config_rejects_invalid_backend(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text('backend = "not-a-backend"\n')
    with pytest.raises(ValueError, match="backend"):
        load_config(str(toml_path))


def test_load_config_rejects_non_positive_vad_silence_ms(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("vad_silence_ms = 0\n")
    with pytest.raises(ValueError, match="vad_silence_ms"):
        load_config(str(toml_path))


def test_load_config_rejects_max_segment_not_greater_than_min(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("vad_min_segment_ms = 500\nvad_max_segment_ms = 500\n")
    with pytest.raises(ValueError, match="vad_max_segment_ms"):
        load_config(str(toml_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.config'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/config.py
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11

VALID_BACKENDS = {"qwen-1.7b", "qwen-0.6b", "whisper"}


@dataclass(frozen=True)
class Config:
    backend: str = "qwen-1.7b"
    language: str = "fr"
    hotkey_toggle_record: str = "<Super>d"
    hotkey_toggle_pause: str = "<Super>p"
    hotkey_copy: str = "<Super>c"
    vad_silence_ms: int = 500
    vad_min_segment_ms: int = 300
    vad_max_segment_ms: int = 15000
    backup_file: str = "/tmp/dictation_live.txt"


def validate_config(config: Config) -> None:
    if config.backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend {config.backend!r}, must be one of {sorted(VALID_BACKENDS)}"
        )
    if config.vad_silence_ms <= 0:
        raise ValueError("vad_silence_ms must be positive")
    if config.vad_min_segment_ms <= 0:
        raise ValueError("vad_min_segment_ms must be positive")
    if config.vad_max_segment_ms <= config.vad_min_segment_ms:
        raise ValueError("vad_max_segment_ms must be greater than vad_min_segment_ms")


def load_config(path: str | None) -> Config:
    config = Config()
    if path is None:
        return config
    file_path = Path(path)
    if not file_path.exists():
        return config
    with file_path.open("rb") as f:
        data = tomllib.load(f)
    config = replace(config, **data)
    validate_config(config)
    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/config.py tests/test_config.py
git commit -m "feat: add config.toml loading and validation"
```

---

## Task 2: Injector pure logic — text joining and injection-target decision

**Files:**
- Create: `dictee/injector.py`
- Test: `tests/test_injector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_injector.py
from dictee.injector import APP_WINDOW_TITLE, join_segment, should_inject


def test_join_segment_first_segment_strips_whitespace():
    assert join_segment("", "  bonjour  ") == "bonjour"


def test_join_segment_adds_space_between_words():
    assert join_segment("bonjour", "le monde") == "bonjour le monde"


def test_join_segment_no_space_before_punctuation():
    assert join_segment("bonjour", ", comment ça va") == "bonjour, comment ça va"


def test_join_segment_ignores_empty_new_segment():
    assert join_segment("bonjour", "   ") == "bonjour"


def test_should_inject_true_for_other_window():
    assert should_inject("Mozilla Firefox") is True


def test_should_inject_false_for_app_window():
    assert should_inject(APP_WINDOW_TITLE) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_injector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.injector'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/injector.py
from __future__ import annotations

APP_WINDOW_TITLE = "DicteeQwenOverlay"

_PUNCTUATION_NO_SPACE_BEFORE = {",", ".", "!", "?", ";", ":", ")", "]", "}", "…"}


def join_segment(existing_text: str, new_segment: str) -> str:
    new_segment = new_segment.strip()
    if not new_segment:
        return existing_text
    if not existing_text:
        return new_segment
    if new_segment[0] in _PUNCTUATION_NO_SPACE_BEFORE:
        return existing_text + new_segment
    return existing_text + " " + new_segment


def should_inject(active_window_title: str) -> bool:
    return active_window_title != APP_WINDOW_TITLE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_injector.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/injector.py tests/test_injector.py
git commit -m "feat: add segment-joining and injection-target logic"
```

---

## Task 3: Injector I/O — `xdotool`/`xclip` wrappers

**Files:**
- Modify: `dictee/injector.py`
- Test: `tests/test_injector_io.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_injector_io.py
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


def test_inject_text_sets_clipboard_then_pastes_into_target_window():
    runner = FakeRunner()
    inject_text("12345", "bonjour", runner=runner)
    assert runner.calls == [
        (["xclip", "-selection", "clipboard"], {"input": "bonjour", "check": True, "text": True}),
        (["xdotool", "key", "--window", "12345", "ctrl+v"], {"check": True}),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_injector_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_active_window_id'`

- [ ] **Step 3: Add the implementation to `dictee/injector.py`**

```python
# Add to dictee/injector.py (after the existing imports/constants)
import subprocess
from typing import Callable

Runner = Callable[..., "subprocess.CompletedProcess"]


def get_active_window_id(runner: Runner = subprocess.run) -> str:
    result = runner(["xdotool", "getactivewindow"], capture_output=True, check=True, text=True)
    return result.stdout.strip()


def get_window_name(window_id: str, runner: Runner = subprocess.run) -> str:
    result = runner(["xdotool", "getwindowname", window_id], capture_output=True, check=True, text=True)
    return result.stdout.strip()


def set_clipboard(text: str, runner: Runner = subprocess.run) -> None:
    runner(["xclip", "-selection", "clipboard"], input=text, check=True, text=True)


def inject_text(window_id: str, text: str, runner: Runner = subprocess.run) -> None:
    set_clipboard(text, runner=runner)
    runner(["xdotool", "key", "--window", window_id, "ctrl+v"], check=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_injector_io.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/injector.py tests/test_injector_io.py
git commit -m "feat: add xdotool/xclip injection I/O wrappers"
```

---

## Task 4: VAD segmenter pure logic (`dictee/vad_segmenter.py`)

**Files:**
- Create: `dictee/vad_segmenter.py`
- Test: `tests/test_vad_segmenter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vad_segmenter.py
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
    for _ in range(4):
        result = seg.feed_frame(frame(0.0), is_speech=False)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vad_segmenter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.vad_segmenter'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/vad_segmenter.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VadSegmenter:
    frame_duration_ms: int
    silence_ms: int
    min_segment_ms: int
    max_segment_ms: int
    _frames: list = field(default_factory=list, init=False)
    _silence_accum_ms: int = field(default=0, init=False)
    _speech_accum_ms: int = field(default=0, init=False)

    def feed_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray | None:
        if is_speech:
            self._frames.append(frame)
            self._speech_accum_ms += self.frame_duration_ms
            self._silence_accum_ms = 0
            if self._speech_accum_ms >= self.max_segment_ms:
                return self._finalize(force=True)
            return None

        if not self._frames:
            return None

        self._silence_accum_ms += self.frame_duration_ms
        if self._silence_accum_ms >= self.silence_ms:
            return self._finalize(force=False)
        return None

    def _finalize(self, force: bool) -> np.ndarray | None:
        frames = self._frames
        duration_ms = self._speech_accum_ms
        self._frames = []
        self._silence_accum_ms = 0
        self._speech_accum_ms = 0
        if duration_ms < self.min_segment_ms and not force:
            return None
        return np.concatenate(frames)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vad_segmenter.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/vad_segmenter.py tests/test_vad_segmenter.py
git commit -m "feat: add silence-based audio segmenter"
```

---

## Task 5: Silero VAD adapter (`dictee/silero_vad.py`)

**Files:**
- Create: `dictee/silero_vad.py`
- Test: `tests/test_silero_vad.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_silero_vad.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_silero_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.silero_vad'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/silero_vad.py
from __future__ import annotations

from typing import Callable

import numpy as np

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
    installed. Not covered by unit tests — exercised by the manual
    integration checklist in README.md."""
    import torch

    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )

    def model_call(frame: np.ndarray, sr: int) -> float:
        tensor = torch.from_numpy(frame).float()
        return float(model(tensor, sr).item())

    return SileroVad(model_call=model_call, sample_rate=sample_rate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_silero_vad.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/silero_vad.py tests/test_silero_vad.py
git commit -m "feat: add Silero VAD adapter with injectable model call"
```

---

## Task 6: ASR backend interface (`dictee/asr_backends/base.py`)

**Files:**
- Create: `dictee/asr_backends/base.py`
- Test: `tests/test_asr_base.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_asr_base.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_asr_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.asr_backends.base'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/asr_backends/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AsrBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_asr_base.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/asr_backends/base.py tests/test_asr_base.py
git commit -m "feat: add ASR backend interface"
```

---

## Task 7: Whisper HTTP backend (`dictee/asr_backends/whisper_backend.py`)

**Files:**
- Create: `dictee/asr_backends/whisper_backend.py`
- Test: `tests/test_whisper_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_whisper_backend.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_whisper_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.asr_backends.whisper_backend'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/asr_backends/whisper_backend.py
from __future__ import annotations

import io
import wave
from typing import Callable

import numpy as np
import requests

from .base import AsrBackend

HttpPost = Callable[..., "requests.Response"]


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
        )
        response.raise_for_status()
        return response.json()["text"].strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_whisper_backend.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/asr_backends/whisper_backend.py tests/test_whisper_backend.py
git commit -m "feat: add whisper.cpp HTTP backend"
```

---

## Task 8: Qwen3-ASR backend (`dictee/asr_backends/qwen_backend.py`)

**Files:**
- Create: `dictee/asr_backends/qwen_backend.py`
- Test: `tests/test_qwen_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qwen_backend.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qwen_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.asr_backends.qwen_backend'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/asr_backends/qwen_backend.py
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import AsrBackend

Loader = Callable[[str], tuple]  # (model_id) -> (model, processor)


class AsrModelLoadError(RuntimeError):
    pass


def default_loader(model_id: str):
    """Loads the real Qwen3-ASR model via transformers. Requires `torch` and
    a CUDA device; not exercised by unit tests (see README manual checklist)."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = (
        AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    return model, processor


class QwenBackend(AsrBackend):
    def __init__(self, model_id: str, language: str = "fr", loader: Loader = default_loader):
        try:
            self._model, self._processor = loader(model_id)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise AsrModelLoadError(
                    f"Not enough VRAM to load {model_id}. "
                    "Try the 0.6B model or the whisper backend instead."
                ) from exc
            raise
        self._language = language

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        inputs = self._processor.apply_transcription_request(
            audio=audio, language=self._language
        ).to(self._model.device, self._model.dtype)
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(generated_ids, return_format="transcription_only")[0]
        return text.strip()

    def close(self) -> None:
        import torch

        del self._model
        torch.cuda.empty_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qwen_backend.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/asr_backends/qwen_backend.py tests/test_qwen_backend.py
git commit -m "feat: add Qwen3-ASR backend with OOM-aware loading"
```

---

## Task 9: Audio capture wrapper (`dictee/audio_capture.py`)

**Files:**
- Create: `dictee/audio_capture.py`
- Test: `tests/test_audio_capture.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_audio_capture.py
import queue

import numpy as np

from dictee.audio_capture import FRAME_SAMPLES, SAMPLE_RATE, AudioCapture


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def test_start_creates_stream_with_expected_parameters():
    created = {}

    def stream_factory(**kwargs):
        created["kwargs"] = kwargs
        return FakeStream(**kwargs)

    capture = AudioCapture(queue.Queue(), stream_factory=stream_factory)
    capture.start()

    assert created["kwargs"]["samplerate"] == SAMPLE_RATE
    assert created["kwargs"]["channels"] == 1
    assert created["kwargs"]["blocksize"] == FRAME_SAMPLES
    assert created["kwargs"]["dtype"] == "float32"
    assert capture._stream.started is True


def test_callback_pushes_mono_frame_copy_to_queue():
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()
    capture = AudioCapture(frame_queue, stream_factory=FakeStream)
    indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

    capture._callback(indata, 3, None, None)

    pushed = frame_queue.get_nowait()
    np.testing.assert_array_equal(pushed, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    indata[0][0] = 99.0  # mutate original to prove the pushed array is a copy
    assert pushed[0] == np.float32(0.1)


def test_stop_stops_and_closes_stream():
    capture = AudioCapture(queue.Queue(), stream_factory=FakeStream)
    capture.start()
    stream = capture._stream
    capture.stop()

    assert stream.started is False
    assert stream.closed is True
    assert capture._stream is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.audio_capture'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/audio_capture.py
from __future__ import annotations

import queue
from typing import Callable

import numpy as np
import sounddevice as sd

FRAME_SAMPLES = 512
SAMPLE_RATE = 16000

StreamFactory = Callable[..., object]


class AudioCapture:
    def __init__(
        self,
        frame_queue: "queue.Queue[np.ndarray]",
        stream_factory: StreamFactory = sd.InputStream,
    ):
        self._frame_queue = frame_queue
        self._stream_factory = stream_factory
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        self._frame_queue.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream = self._stream_factory(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SAMPLES,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_capture.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/audio_capture.py tests/test_audio_capture.py
git commit -m "feat: add sounddevice audio capture wrapper"
```

---

## Task 10: Global hotkeys (`dictee/hotkeys.py`)

**Files:**
- Create: `dictee/hotkeys.py`
- Test: `tests/test_hotkeys.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hotkeys.py
import pytest
from Xlib import X, XK

from dictee.hotkeys import HotkeyListener, parse_shortcut


def test_parse_shortcut_single_modifier():
    assert parse_shortcut("<Super>d") == (X.Mod4Mask, "d")


def test_parse_shortcut_stacked_modifiers():
    assert parse_shortcut("<Ctrl><Alt>p") == (X.ControlMask | X.Mod1Mask, "p")


def test_parse_shortcut_rejects_unknown_modifier():
    with pytest.raises(ValueError, match="Foo"):
        parse_shortcut("<Foo>d")


def test_parse_shortcut_rejects_missing_key():
    with pytest.raises(ValueError, match="no key"):
        parse_shortcut("<Super>")


class FakeRootWindow:
    def __init__(self):
        self.grab_calls = []

    def grab_key(self, keycode, modifiers, owner_events, pointer_mode, keyboard_mode):
        self.grab_calls.append((keycode, modifiers))


class FakeScreenInfo:
    def __init__(self, root):
        self.root = root


class FakeDisplay:
    def __init__(self):
        self.root = FakeRootWindow()
        self._screen_info = FakeScreenInfo(self.root)

    def screen(self):
        return self._screen_info

    def keysym_to_keycode(self, keysym):
        return keysym  # identity stand-in, sufficient to verify wiring


def test_grab_all_grabs_each_configured_shortcut():
    fake_display = FakeDisplay()
    listener = HotkeyListener(
        {"<Super>d": lambda: None, "<Super>p": lambda: None},
        display_factory=lambda: fake_display,
    )

    listener.grab_all()

    expected = {
        (XK.string_to_keysym("d"), X.Mod4Mask),
        (XK.string_to_keysym("p"), X.Mod4Mask),
    }
    assert set(fake_display.root.grab_calls) == expected


def test_grab_all_registers_callbacks_by_keycode_and_modifiers():
    fake_display = FakeDisplay()
    calls = []
    listener = HotkeyListener(
        {"<Super>d": lambda: calls.append("d")}, display_factory=lambda: fake_display
    )
    listener.grab_all()

    keysym = XK.string_to_keysym("d")
    callback = listener._keycode_map[(keysym, X.Mod4Mask)]
    callback()

    assert calls == ["d"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hotkeys.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.hotkeys'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/hotkeys.py
from __future__ import annotations

from typing import Callable

from Xlib import X, XK
from Xlib.display import Display

_MODIFIER_MASKS = {
    "Super": X.Mod4Mask,
    "Ctrl": X.ControlMask,
    "Alt": X.Mod1Mask,
    "Shift": X.ShiftMask,
}


def parse_shortcut(shortcut: str) -> tuple[int, str]:
    modifiers = 0
    remainder = shortcut
    while remainder.startswith("<"):
        end = remainder.index(">")
        name = remainder[1:end]
        if name not in _MODIFIER_MASKS:
            raise ValueError(f"Unknown modifier {name!r} in shortcut {shortcut!r}")
        modifiers |= _MODIFIER_MASKS[name]
        remainder = remainder[end + 1:]
    if not remainder:
        raise ValueError(f"Shortcut {shortcut!r} has no key")
    return modifiers, remainder


class HotkeyListener:
    def __init__(
        self,
        bindings: dict[str, Callable[[], None]],
        display_factory: Callable[[], "Display"] = Display,
    ):
        self._bindings = bindings
        self._display = display_factory()
        self._root = self._display.screen().root
        self._keycode_map: dict[tuple[int, int], Callable[[], None]] = {}

    def grab_all(self) -> None:
        for shortcut, callback in self._bindings.items():
            modifiers, key = parse_shortcut(shortcut)
            keysym = XK.string_to_keysym(key)
            keycode = self._display.keysym_to_keycode(keysym)
            self._root.grab_key(keycode, modifiers, True, X.GrabModeAsync, X.GrabModeAsync)
            self._keycode_map[(keycode, modifiers)] = callback

    def run(self) -> None:
        """Blocks, dispatching matching key-press events to their callback.
        Requires a live X11 connection; exercised by the manual integration
        checklist in README.md, not by unit tests."""
        while True:
            event = self._display.next_event()
            if event.type == X.KeyPress:
                callback = self._keycode_map.get((event.detail, event.state))
                if callback is not None:
                    callback()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hotkeys.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/hotkeys.py tests/test_hotkeys.py
git commit -m "feat: add global X11 hotkey listener"
```

---

## Task 11: Single-instance PID lock (`dictee/pid_lock.py`)

**Files:**
- Create: `dictee/pid_lock.py`
- Test: `tests/test_pid_lock.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pid_lock.py
import os

import pytest

from dictee.pid_lock import AlreadyRunningError, PidLock


def test_acquire_writes_current_pid_when_no_existing_file(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock = PidLock(str(lock_path))
    lock.acquire()
    assert lock_path.read_text().strip() == str(os.getpid())


def test_acquire_raises_when_existing_pid_is_alive(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock_path.write_text(str(os.getpid()))  # our own process, always alive
    lock = PidLock(str(lock_path))
    with pytest.raises(AlreadyRunningError, match=str(os.getpid())):
        lock.acquire()


def test_acquire_succeeds_when_existing_pid_is_dead(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock_path.write_text("424242")
    lock = PidLock(str(lock_path), pid_is_alive=lambda pid: False)
    lock.acquire()
    assert lock_path.read_text().strip() == str(os.getpid())


def test_release_removes_file(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock = PidLock(str(lock_path))
    lock.acquire()
    lock.release()
    assert not lock_path.exists()


def test_release_is_noop_when_file_missing(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock = PidLock(str(lock_path))
    lock.release()  # must not raise
    assert not lock_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pid_lock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.pid_lock'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/pid_lock.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class AlreadyRunningError(RuntimeError):
    pass


def _default_pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PidLock:
    def __init__(self, path: str, pid_is_alive: Callable[[int], bool] = _default_pid_is_alive):
        self._path = Path(path)
        self._pid_is_alive = pid_is_alive

    def acquire(self) -> None:
        if self._path.exists():
            existing_pid = int(self._path.read_text().strip())
            if self._pid_is_alive(existing_pid):
                raise AlreadyRunningError(
                    f"dictee_app is already running with PID {existing_pid}"
                )
        self._path.write_text(str(os.getpid()))

    def release(self) -> None:
        if self._path.exists():
            self._path.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pid_lock.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/pid_lock.py tests/test_pid_lock.py
git commit -m "feat: add single-instance PID lock"
```

---

## Task 12: Live backup file writer (`dictee/backup_writer.py`)

**Files:**
- Create: `dictee/backup_writer.py`
- Test: `tests/test_backup_writer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backup_writer.py
from dictee.backup_writer import BackupWriter


def test_start_session_creates_empty_file(tmp_path):
    path = tmp_path / "dictation_live.txt"
    writer = BackupWriter(str(path))
    writer.start_session()
    assert path.read_text() == ""


def test_append_segment_appends_text_with_trailing_space(tmp_path):
    path = tmp_path / "dictation_live.txt"
    writer = BackupWriter(str(path))
    writer.start_session()
    writer.append_segment("bonjour")
    assert path.read_text() == "bonjour "


def test_append_segment_multiple_calls_accumulate(tmp_path):
    path = tmp_path / "dictation_live.txt"
    writer = BackupWriter(str(path))
    writer.start_session()
    writer.append_segment("bonjour")
    writer.append_segment("le monde")
    assert path.read_text() == "bonjour le monde "


def test_start_session_clears_previous_session_content(tmp_path):
    path = tmp_path / "dictation_live.txt"
    path.write_text("leftover from a previous crashed session")
    writer = BackupWriter(str(path))
    writer.start_session()
    assert path.read_text() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backup_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.backup_writer'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/backup_writer.py
from __future__ import annotations

from pathlib import Path


class BackupWriter:
    def __init__(self, path: str):
        self._path = Path(path)

    def start_session(self) -> None:
        self._path.write_text("")

    def append_segment(self, text: str) -> None:
        with self._path.open("a") as f:
            f.write(text)
            f.write(" ")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backup_writer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/backup_writer.py tests/test_backup_writer.py
git commit -m "feat: add live transcript backup file writer"
```

---

## Task 13: GUI status label helper

**Files:**
- Create: `dictee/gui.py` (status helper only; the window class is added in Task 14)
- Test: `tests/test_gui_status.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gui_status.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gui_status.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.gui'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/gui.py
from __future__ import annotations

_STATUS_DISPLAY = {
    "idle": ("Idle", "grey"),
    "recording": ("Recording", "red"),
    "paused": ("Paused", "yellow"),
}


def status_label(state: str) -> tuple[str, str]:
    try:
        return _STATUS_DISPLAY[state]
    except KeyError:
        raise ValueError(f"unknown state {state!r}") from None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gui_status.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/gui.py tests/test_gui_status.py
git commit -m "feat: add GUI status label mapping"
```

---

## Task 14: GTK overlay window

**Files:**
- Modify: `dictee/gui.py`

This task adds the real GTK window. It is not covered by automated tests
(rendering a GTK window requires a running X11 display) — it is exercised by
the manual integration checklist in `README.md` (Task 19).

- [ ] **Step 1: Append the window class to `dictee/gui.py`**

```python
# Add to dictee/gui.py
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

from .injector import APP_WINDOW_TITLE  # noqa: E402


class DictationWindow(Gtk.Window):
    def __init__(self, *, on_toggle_record, on_toggle_pause, on_copy):
        super().__init__(title=APP_WINDOW_TITLE)
        self._on_toggle_record = on_toggle_record
        self._on_toggle_pause = on_toggle_pause
        self._on_copy = on_copy

        self.set_keep_above(True)
        self.set_default_size(360, 240)
        self.set_resizable(True)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(box)

        self._status_label = Gtk.Label(label="Idle")
        box.pack_start(self._status_label, False, False, 4)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._text_view = Gtk.TextView()
        self._text_view.set_editable(False)
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        scroller.add(self._text_view)
        box.pack_start(scroller, True, True, 0)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.pack_start(button_box, False, False, 4)

        record_button = Gtk.Button(label="Start/Stop")
        record_button.connect("clicked", lambda _btn: self._on_toggle_record())
        button_box.pack_start(record_button, True, True, 0)

        pause_button = Gtk.Button(label="Pause/Reprendre")
        pause_button.connect("clicked", lambda _btn: self._on_toggle_pause())
        button_box.pack_start(pause_button, True, True, 0)

        copy_button = Gtk.Button(label="Copier")
        copy_button.connect("clicked", lambda _btn: self._on_copy())
        button_box.pack_start(copy_button, True, True, 0)

        self.connect("destroy", lambda _win: Gtk.main_quit())

    def append_segment(self, text: str) -> None:
        GLib.idle_add(self._append_segment_on_main_thread, text)

    def _append_segment_on_main_thread(self, text: str) -> bool:
        buffer = self._text_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), text + " ")
        return False

    def set_status(self, state: str) -> None:
        GLib.idle_add(self._set_status_on_main_thread, state)

    def _set_status_on_main_thread(self, state: str) -> bool:
        text, _color = status_label(state)
        self._status_label.set_text(text)
        return False

    def clear_transcript(self) -> None:
        GLib.idle_add(self._clear_transcript_on_main_thread)

    def _clear_transcript_on_main_thread(self) -> bool:
        self._text_view.get_buffer().set_text("")
        return False
```

- [ ] **Step 2: Manually verify the window renders**

Run (requires an X11 session and the venv active):
```bash
python3 -c "
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from dictee.gui import DictationWindow
win = DictationWindow(on_toggle_record=lambda: print('toggle'), on_toggle_pause=lambda: print('pause'), on_copy=lambda: print('copy'))
win.show_all()
Gtk.main()
"
```
Expected: a small always-on-top window appears with a status label, a text area, and three buttons; clicking each button prints the corresponding message to the terminal; closing the window exits the process.

- [ ] **Step 3: Commit**

```bash
git add dictee/gui.py
git commit -m "feat: add GTK floating overlay window"
```

---

## Task 15: Controller — session state machine and orchestration

**Files:**
- Create: `dictee/controller.py`
- Test: `tests/test_controller.py`

This is the core of the merged app: it decides what happens to every
transcribed segment (append to transcript, write to backup file, update the
GUI, and inject at the cursor unless the active window is the app itself).
All hardware-facing collaborators are injected as fakes so the state machine
is fully covered without touching real audio/GPU/X11.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_controller.py
import numpy as np

from dictee.controller import Controller
from dictee.injector import APP_WINDOW_TITLE


class FakeVadModel:
    def __init__(self, is_speech=True):
        self.is_speech_value = is_speech

    def is_speech(self, frame):
        return self.is_speech_value


class FakeVadSegmenter:
    def __init__(self, results):
        self._results = list(results)

    def feed_frame(self, frame, is_speech):
        return self._results.pop(0) if self._results else None


class FakeAsrBackend:
    def __init__(self, text="bonjour"):
        self.text = text
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        return self.text


class FakeBackupWriter:
    def __init__(self):
        self.started = False
        self.segments = []

    def start_session(self):
        self.started = True
        self.segments = []

    def append_segment(self, text):
        self.segments.append(text)


class FakeGui:
    def __init__(self):
        self.statuses = []
        self.segments = []

    def set_status(self, state):
        self.statuses.append(state)

    def append_segment(self, text):
        self.segments.append(text)


def make_controller(vad_results=(), asr_text="bonjour", active_window_title="Mozilla Firefox"):
    backup_writer = FakeBackupWriter()
    gui = FakeGui()
    asr_backend = FakeAsrBackend(text=asr_text)
    inject_calls = []
    clipboard_calls = []

    controller = Controller(
        vad_segmenter=FakeVadSegmenter(vad_results),
        vad_model=FakeVadModel(),
        asr_backend=asr_backend,
        backup_writer=backup_writer,
        gui=gui,
        get_active_window_id=lambda: "42",
        get_window_name=lambda window_id: active_window_title,
        inject_text=lambda window_id, text: inject_calls.append((window_id, text)),
        set_clipboard=lambda text: clipboard_calls.append(text),
        sample_rate=16000,
    )
    return controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls


def test_toggle_record_starts_recording_resets_transcript_and_backup():
    controller, backup_writer, gui, *_ = make_controller()
    controller.toggle_record()
    assert controller.state == "recording"
    assert backup_writer.started is True
    assert gui.statuses == ["recording"]


def test_toggle_record_again_stops_recording():
    controller, *_ = make_controller()
    controller.toggle_record()
    controller.toggle_record()
    assert controller.state == "idle"


def test_toggle_pause_from_recording_goes_to_paused_and_back():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller()
    controller.toggle_record()
    controller.toggle_pause()
    assert controller.state == "paused"
    controller.toggle_pause()
    assert controller.state == "recording"


def test_toggle_pause_while_idle_is_noop():
    controller, backup_writer, gui, *_ = make_controller()
    controller.toggle_pause()
    assert controller.state == "idle"
    assert gui.statuses == []


def test_copy_sends_transcript_to_clipboard():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller()
    controller.transcript = "bonjour le monde"
    controller.copy()
    assert clipboard_calls == ["bonjour le monde"]


def test_process_frame_ignored_when_not_recording():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[np.zeros(10, dtype=np.float32)]
    )
    controller.process_frame(np.zeros(10, dtype=np.float32))
    assert asr_backend.calls == []


def test_process_frame_with_no_completed_segment_does_nothing():
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[None]
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))
    assert asr_backend.calls == []
    assert gui.segments == []


def test_process_frame_with_completed_segment_updates_transcript_backup_gui_and_injects():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio], asr_text="bonjour", active_window_title="Mozilla Firefox"
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))

    assert controller.transcript == "bonjour"
    assert backup_writer.segments == ["bonjour"]
    assert gui.segments == ["bonjour"]
    assert inject_calls == [("42", "bonjour")]
    assert asr_backend.calls == [(segment_audio, 16000)]


def test_process_frame_skips_injection_when_focus_is_own_window():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio], asr_text="bonjour", active_window_title=APP_WINDOW_TITLE
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))

    assert gui.segments == ["bonjour"]
    assert backup_writer.segments == ["bonjour"]
    assert inject_calls == []


def test_process_frame_ignores_empty_transcription():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio], asr_text=""
    )
    controller.toggle_record()
    controller.process_frame(np.zeros(10, dtype=np.float32))

    assert gui.segments == []
    assert backup_writer.segments == []
    assert inject_calls == []


def test_process_frame_ignored_while_paused():
    segment_audio = np.ones(10, dtype=np.float32)
    controller, backup_writer, gui, asr_backend, inject_calls, clipboard_calls = make_controller(
        vad_results=[segment_audio]
    )
    controller.toggle_record()
    controller.toggle_pause()
    controller.process_frame(np.zeros(10, dtype=np.float32))
    assert asr_backend.calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_controller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dictee.controller'`

- [ ] **Step 3: Write the implementation**

```python
# dictee/controller.py
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .injector import join_segment, should_inject


class VadModel(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...


class VadSegmenterProtocol(Protocol):
    def feed_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray | None: ...


class AsrBackendProtocol(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class BackupWriterProtocol(Protocol):
    def start_session(self) -> None: ...
    def append_segment(self, text: str) -> None: ...


class GuiProtocol(Protocol):
    def set_status(self, state: str) -> None: ...
    def append_segment(self, text: str) -> None: ...


class Controller:
    def __init__(
        self,
        *,
        vad_segmenter: VadSegmenterProtocol,
        vad_model: VadModel,
        asr_backend: AsrBackendProtocol,
        backup_writer: BackupWriterProtocol,
        gui: GuiProtocol,
        get_active_window_id: Callable[[], str],
        get_window_name: Callable[[str], str],
        inject_text: Callable[[str, str], None],
        set_clipboard: Callable[[str], None],
        sample_rate: int,
    ):
        self._vad_segmenter = vad_segmenter
        self._vad_model = vad_model
        self._asr_backend = asr_backend
        self._backup_writer = backup_writer
        self._gui = gui
        self._get_active_window_id = get_active_window_id
        self._get_window_name = get_window_name
        self._inject_text = inject_text
        self._set_clipboard = set_clipboard
        self._sample_rate = sample_rate
        self.state = "idle"
        self.transcript = ""

    def toggle_record(self) -> None:
        if self.state == "idle":
            self.transcript = ""
            self._backup_writer.start_session()
            self.state = "recording"
        else:
            self.state = "idle"
        self._gui.set_status(self.state)

    def toggle_pause(self) -> None:
        if self.state == "recording":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "recording"
        else:
            return
        self._gui.set_status(self.state)

    def copy(self) -> None:
        self._set_clipboard(self.transcript)

    def process_frame(self, frame: np.ndarray) -> None:
        if self.state != "recording":
            return
        is_speech = self._vad_model.is_speech(frame)
        segment_audio = self._vad_segmenter.feed_frame(frame, is_speech)
        if segment_audio is None:
            return
        text = self._asr_backend.transcribe(segment_audio, self._sample_rate)
        if not text:
            return
        self.transcript = join_segment(self.transcript, text)
        self._backup_writer.append_segment(text)
        self._gui.append_segment(text)
        self._maybe_inject(text)

    def _maybe_inject(self, text: str) -> None:
        window_id = self._get_active_window_id()
        window_name = self._get_window_name(window_id)
        if should_inject(window_name):
            self._inject_text(window_id, text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_controller.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add dictee/controller.py tests/test_controller.py
git commit -m "feat: add session controller orchestrating VAD, ASR, GUI and injection"
```

---

## Task 16: Default `config.toml`

**Files:**
- Create: `config.toml`

- [ ] **Step 1: Write the default config file**

```toml
# config.toml — dictee-qwen default configuration

# One of: "qwen-1.7b", "qwen-0.6b", "whisper"
backend = "qwen-1.7b"

language = "fr"

# Global keyboard shortcuts (captured in X11, work even when the window
# is minimized). Modifiers: <Super> <Ctrl> <Alt> <Shift>, stackable.
hotkey_toggle_record = "<Super>d"
hotkey_toggle_pause = "<Super>p"
hotkey_copy = "<Super>c"

# Voice activity detection tuning (milliseconds)
vad_silence_ms = 500
vad_min_segment_ms = 300
vad_max_segment_ms = 15000

backup_file = "/tmp/dictation_live.txt"
```

- [ ] **Step 2: Verify it loads without error**

Run:
```bash
python3 -c "from dictee.config import load_config; print(load_config('config.toml'))"
```
Expected: prints a `Config(...)` with `backend='qwen-1.7b'`, no exception.

- [ ] **Step 3: Commit**

```bash
git add config.toml
git commit -m "chore: add default config.toml"
```

---

## Task 17: Application entrypoint (`dictee_app.py`)

**Files:**
- Create: `dictee_app.py`

This task wires every previously-built module into the running application.
It is integration glue, exercised by the manual checklist in Task 19, not by
unit tests (the pieces it wires together are already unit-tested
individually).

- [ ] **Step 1: Write `dictee_app.py`**

```python
#!/usr/bin/env python3
from __future__ import annotations

import queue
import signal
import subprocess
import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from dictee.audio_capture import SAMPLE_RATE, AudioCapture
from dictee.backup_writer import BackupWriter
from dictee.config import load_config
from dictee.controller import Controller
from dictee.gui import DictationWindow
from dictee.hotkeys import HotkeyListener
from dictee.injector import get_active_window_id, get_window_name, inject_text, set_clipboard
from dictee.pid_lock import AlreadyRunningError, PidLock
from dictee.silero_vad import load_silero_vad
from dictee.vad_segmenter import VadSegmenter

PID_LOCK_PATH = "/tmp/dictee_app.pid"


def build_asr_backend(config):
    if config.backend == "qwen-1.7b":
        from dictee.asr_backends.qwen_backend import QwenBackend

        return QwenBackend("Qwen/Qwen3-ASR-1.7B-hf", language=config.language)
    if config.backend == "qwen-0.6b":
        from dictee.asr_backends.qwen_backend import QwenBackend

        return QwenBackend("Qwen/Qwen3-ASR-0.6B-hf", language=config.language)
    if config.backend == "whisper":
        from dictee.asr_backends.whisper_backend import WhisperBackend

        return WhisperBackend(base_url="http://localhost:8080")
    raise ValueError(f"Unknown backend {config.backend!r}")


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    config = load_config(config_path)

    lock = PidLock(PID_LOCK_PATH)
    try:
        lock.acquire()
    except AlreadyRunningError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    asr_backend = build_asr_backend(config)
    backup_writer = BackupWriter(config.backup_file)
    vad_segmenter = VadSegmenter(
        frame_duration_ms=32,  # 512 samples at 16kHz
        silence_ms=config.vad_silence_ms,
        min_segment_ms=config.vad_min_segment_ms,
        max_segment_ms=config.vad_max_segment_ms,
    )
    vad_model = load_silero_vad(sample_rate=SAMPLE_RATE)

    frame_queue: "queue.Queue" = queue.Queue()
    audio_capture = AudioCapture(frame_queue)

    window = DictationWindow(
        on_toggle_record=lambda: controller.toggle_record() or audio_toggle(),
        on_toggle_pause=lambda: controller.toggle_pause() or audio_pause_toggle(),
        on_copy=lambda: controller.copy(),
    )

    controller = Controller(
        vad_segmenter=vad_segmenter,
        vad_model=vad_model,
        asr_backend=asr_backend,
        backup_writer=backup_writer,
        gui=window,
        get_active_window_id=get_active_window_id,
        get_window_name=get_window_name,
        inject_text=inject_text,
        set_clipboard=set_clipboard,
        sample_rate=SAMPLE_RATE,
    )

    def audio_toggle():
        if controller.state == "recording":
            audio_capture.start()
        else:
            audio_capture.stop()

    def audio_pause_toggle():
        if controller.state == "recording":
            audio_capture.start()
        elif controller.state == "paused":
            audio_capture.stop()

    def process_frames_loop():
        while True:
            frame = frame_queue.get()
            controller.process_frame(frame)

    processing_thread = threading.Thread(target=process_frames_loop, daemon=True)
    processing_thread.start()

    hotkeys = HotkeyListener(
        {
            config.hotkey_toggle_record: lambda: (controller.toggle_record(), audio_toggle()),
            config.hotkey_toggle_pause: lambda: (controller.toggle_pause(), audio_pause_toggle()),
            config.hotkey_copy: lambda: controller.copy(),
        }
    )
    hotkeys.grab_all()
    hotkey_thread = threading.Thread(target=hotkeys.run, daemon=True)
    hotkey_thread.start()

    def shutdown(*_args):
        audio_capture.stop()
        asr_backend.close()
        if config.backend == "whisper":
            pass  # whisper-server lifecycle is managed outside this process for now
        lock.release()
        Gtk.main_quit()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    window.show_all()
    try:
        Gtk.main()
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Manually verify the app starts and shuts down cleanly**

Run (requires an X11 session, a working microphone, and `config.toml` with
`backend = "whisper"` or a GPU available for the Qwen backend):
```bash
python3 dictee_app.py config.toml
```
Expected: the overlay window appears, pressing `Ctrl+C` in the terminal
cleanly stops the app without a traceback, and `/tmp/dictee_app.pid` is
removed afterwards.

- [ ] **Step 3: Commit**

```bash
git add dictee_app.py
git commit -m "feat: wire audio, VAD, ASR, injector, hotkeys and GUI into the app entrypoint"
```

---

## Task 18: `setup.sh`

**Files:**
- Create: `setup.sh`

- [ ] **Step 1: Write the installation script**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "== Installing system dependencies =="
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    xdotool \
    xclip \
    portaudio19-dev \
    python3-venv \
    python3-gi \
    gir1.2-gtk-3.0 \
    libx11-dev

echo "== Creating Python virtual environment =="
python3 -m venv venv
source venv/bin/activate

echo "== Installing PyTorch with CUDA 12.1 support =="
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "== Installing project dependencies =="
pip install -r requirements.txt

echo "== Verifying CUDA availability =="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

read -rp "Build whisper.cpp with CUDA support as a fallback backend? [y/N] " build_whisper
if [[ "${build_whisper:-N}" =~ ^[Yy]$ ]]; then
    echo "== Cloning and building whisper.cpp with CUDA =="
    git clone https://github.com/ggerganov/whisper.cpp.git vendor/whisper.cpp
    (
        cd vendor/whisper.cpp
        GGML_CUDA=1 make -j"$(nproc)"
        ./models/download-ggml-model.sh large-v3-turbo
    )
    echo "Start the backend with:"
    echo "  vendor/whisper.cpp/build/bin/whisper-server -m vendor/whisper.cpp/models/ggml-large-v3-turbo.bin --port 8080"
fi

echo "== Done. Activate the environment with: source venv/bin/activate =="
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x setup.sh`

- [ ] **Step 3: Commit**

```bash
git add setup.sh
git commit -m "chore: add setup.sh for system and Python dependencies"
```

---

## Task 19: README — autostart integration and manual test checklist

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# dictee-qwen

Dictée vocale locale (français) pour Linux Mint / Cinnamon / X11, avec
transcription GPU (Qwen3-ASR, whisper.cpp en secours), overlay flottant et
raccourcis clavier globaux. Voir le design complet dans
`docs/superpowers/specs/2026-10-30-dictee-vocale-globale-design.md`.

## Installation

```bash
./setup.sh
source venv/bin/activate
pytest  # vérifie que la logique interne fonctionne avant de lancer l'app
```

## Lancement manuel

```bash
source venv/bin/activate
python3 dictee_app.py config.toml
```

## Démarrage automatique à l'ouverture de session (Cinnamon)

Les raccourcis clavier sont captés par l'application elle-même (X11), donc
aucune configuration dans *Paramètres système > Clavier* n'est nécessaire.
Il faut seulement que `dictee_app.py` tourne dès l'ouverture de session :

1. Ouvrir *Paramètres système > Démarrage*.
2. Ajouter une entrée personnalisée :
   - Nom : `Dictée Qwen`
   - Commande : `/chemin/vers/dictee-qwen/venv/bin/python3 /chemin/vers/dictee-qwen/dictee_app.py /chemin/vers/dictee-qwen/config.toml`
   - Retard de démarrage : quelques secondes (pour laisser le bureau se charger).

## Checklist de tests manuels (non automatisés)

- [ ] Démarrage de l'app : la fenêtre flottante apparaît, le modèle se
      charge sans erreur (`torch.cuda.is_available()` vrai).
- [ ] Appuyer sur le raccourci "toggle record" alors que la fenêtre est
      minimisée : l'enregistrement démarre (statut GUI passe à "Recording"
      après réaffichage).
- [ ] Parler dans un navigateur avec le curseur dans un champ de texte : le
      texte apparaît au niveau du curseur au fur et à mesure.
- [ ] Mettre le focus sur la fenêtre de l'app elle-même pendant la dictée :
      le texte n'est plus injecté, seulement affiché dans la zone de texte.
- [ ] Utiliser le bouton "Pause/Reprendre" puis le raccourci clavier
      correspondant : les deux doivent produire le même effet.
- [ ] Utiliser le bouton "Copier" : le transcript complet est dans le
      presse-papiers (`xclip -o -selection clipboard`).
- [ ] Dictée longue (10–15 minutes) : vérifier qu'aucune donnée n'est perdue
      en comparant avec `/tmp/dictation_live.txt`.
- [ ] Débrancher le micro en cours de dictée : l'app ne doit pas planter
      (un message d'erreur est acceptable).
- [ ] Arrêter l'app (`Ctrl+C` ou fermeture de la fenêtre) : le fichier
      `/tmp/dictee_app.pid` est supprimé, `nvidia-smi` ne montre plus le
      processus en mémoire GPU.
- [ ] Relancer l'app juste après : pas de message "already running"
      bloquant à tort.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup, autostart and manual test checklist"
```

---

## Self-review notes

- **Spec coverage:** every section of the design doc maps to a task —
  architecture/single-process (Tasks 15, 17), ASR backends (Tasks 6–8), VAD
  (Tasks 4–5), GUI (Tasks 13–14), hotkeys (Task 10), injection logic (Tasks
  2–3), error handling/OOM (Task 8), cleanup/PID lock (Task 11), backup file
  (Task 12), project structure/config/setup/README (Tasks 0, 16, 18, 19).
- **Type consistency:** `Controller` constructor keyword names match exactly
  between Task 15's implementation and the fakes used in its own tests;
  `dictee_app.py` (Task 17) calls every collaborator with the same
  constructor signature defined in its own task (`AudioCapture(frame_queue)`,
  `VadSegmenter(frame_duration_ms=..., silence_ms=..., min_segment_ms=...,
  max_segment_ms=...)`, `HotkeyListener({shortcut: callback})`,
  `PidLock(path)`, `BackupWriter(path)`, `DictationWindow(on_toggle_record=,
  on_toggle_pause=, on_copy=)`) — all taken from their respective task
  definitions, not re-guessed.
- **No placeholders:** every step contains complete, runnable code; the two
  genuinely hardware-dependent pieces (GTK rendering in Task 14, the full
  app loop in Task 17) are explicitly marked as manually verified rather than
  left with a vague "test this" instruction.
