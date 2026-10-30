from __future__ import annotations

from dataclasses import dataclass, fields, replace
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
    valid_fields = {f.name for f in fields(Config)}
    unknown_keys = set(data) - valid_fields
    if unknown_keys:
        raise ValueError(f"Unknown config key(s) {sorted(unknown_keys)} in {path}")
    config = replace(config, **data)
    validate_config(config)
    return config
