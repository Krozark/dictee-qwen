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
