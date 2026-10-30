from __future__ import annotations

import os
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
            f.flush()
            os.fsync(f.fileno())
