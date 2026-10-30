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


def test_acquire_succeeds_when_existing_file_is_corrupted(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock_path.write_text("")
    lock = PidLock(str(lock_path))
    lock.acquire()
    assert lock_path.read_text().strip() == str(os.getpid())


def test_acquire_succeeds_when_existing_file_is_garbage(tmp_path):
    lock_path = tmp_path / "dictee.pid"
    lock_path.write_text("not-a-number")
    lock = PidLock(str(lock_path))
    lock.acquire()
    assert lock_path.read_text().strip() == str(os.getpid())
