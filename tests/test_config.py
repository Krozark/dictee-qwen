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


def test_load_config_rejects_unknown_key(tmp_path):
    toml_path = tmp_path / "config.toml"
    toml_path.write_text('backend_ = "whisper"\n')
    with pytest.raises(ValueError, match="backend_"):
        load_config(str(toml_path))
