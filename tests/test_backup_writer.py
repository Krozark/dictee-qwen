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
