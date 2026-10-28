import contextlib
import fcntl
import hashlib
import os
import pickle
import tempfile
from pathlib import Path


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class LocalCheckpoints:
    """Trusted self-created pickle only. Digest detects corruption, not hostile replacements."""

    magic = b"CB01"

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, identity):
        return self.directory / identity / "checkpoint.bin"

    def load(self, identity):
        path = self.path(identity)
        if not path.exists():
            return None
        blob = path.read_bytes()
        if blob[:4] != self.magic or hashlib.sha256(blob[68:]).hexdigest().encode() != blob[4:68]:
            raise ValueError("checkpoint checksum/schema mismatch")
        return pickle.loads(blob[68:])

    def save(self, identity, state):
        payload = pickle.dumps(state, protocol=5)
        atomic_write(self.path(identity), self.magic + hashlib.sha256(payload).hexdigest().encode() + payload)


@contextlib.contextmanager
def run_lock(directory, identity):
    path = Path(directory) / identity
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"run already active: {identity}") from e
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextlib.contextmanager
def publication_lock(directory, identity):
    """Serialize shared catalog/registry mutations across worker subprocesses."""
    path = Path(directory) / ".publication-locks"
    path.mkdir(parents=True, exist_ok=True)
    with (path / hashlib.sha256(identity.encode()).hexdigest()).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
