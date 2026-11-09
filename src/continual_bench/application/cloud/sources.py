"""Catalog-driven public sources and a bounded, resumable, verified HTTPS downloader.

The state machine never receives URLs: the dataset catalog yields the pinned file names,
revisions and SHA-256 digests, and this module derives the only admitted URL per file.
Redirects are followed only over HTTPS to the hosts the source declares.
"""

import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from continual_bench.datasets.zia import DATASETS

HF_BASE = "https://huggingface.co/datasets/ziadatalabs"
# huggingface.co resolves LFS/Xet files through CDN hosts under these domains.
ALLOWED_HOST_SUFFIXES = ("huggingface.co", "hf.co")
MAX_REDIRECTS = 5
CHUNK = 8 * 1024**2
DEFAULT_MAX_BYTES = 48 * 1024**3


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    sha256: str
    allowed_hosts: tuple[str, ...] = ALLOWED_HOST_SUFFIXES
    max_bytes: int = DEFAULT_MAX_BYTES


def source_files(dataset: str, revision: str) -> list[SourceFile]:
    spec = DATASETS[dataset]
    if revision != spec["revision"]:
        raise ValueError("revision is not admitted by the dataset catalog")
    files = [(spec["file"], spec["sha256"])]
    if "nodes" in spec:
        files.append((spec["nodes"], spec["nodes_sha256"]))
    return [
        SourceFile(name, f"{HF_BASE}/{spec['repo']}/resolve/{revision}/{name}", digest)
        for name, digest in files
    ]


def host_allowed(url: str, allowed=ALLOWED_HOST_SUFFIXES) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(host == s or host.endswith("." + s) for s in allowed)


class _AllowlistRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed):
        self.allowed = allowed
        self.hops = 0

    def redirect_request(self, request, fp, code, message, headers, new_url):
        self.hops += 1
        if self.hops > MAX_REDIRECTS:
            raise urllib.error.HTTPError(new_url, code, "too many redirects", headers, fp)
        if not host_allowed(new_url, self.allowed):
            raise urllib.error.HTTPError(
                new_url, code, "redirect to a host outside the source allowlist", headers, fp
            )
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected is not None and request.has_header("Range"):
            redirected.add_unredirected_header("Range", request.get_header("Range"))
        return redirected


def default_opener(allowed):
    return urllib.request.build_opener(_AllowlistRedirects(allowed)).open


def sha256_file(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(
    source: SourceFile,
    destination,
    *,
    timeout=60,
    retries=6,
    deadline_seconds=6 * 3600,
    opener=None,
    sleep=time.sleep,
) -> dict:
    """Download with HTTP Range resume, bounded size/time/retries and SHA-256 verification."""
    if not host_allowed(source.url, source.allowed_hosts):
        raise ValueError(f"source URL outside the allowlist: {source.url}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == source.sha256:
        return {
            "sha256": source.sha256,
            "bytes": destination.stat().st_size,
            "reused": True,
            "attempts": 0,
            "resumed_bytes": 0,
        }
    partial = destination.with_name(destination.name + ".partial")
    open_url = opener or default_opener(source.allowed_hosts)
    started, attempts, resumed = time.monotonic(), 0, 0
    while True:
        attempts += 1
        offset = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(source.url, headers={"User-Agent": "continual-bench-acquisition/1"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with open_url(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if offset and status == 206:
                    resumed += offset
                    mode = "ab"
                else:
                    mode, offset = "wb", 0
                length = response.headers.get("Content-Length")
                if length is not None and offset + int(length) > source.max_bytes:
                    raise ValueError(f"{source.name} exceeds the {source.max_bytes} byte limit")
                with partial.open(mode) as stream:
                    written = offset
                    while chunk := response.read(CHUNK):
                        written += len(chunk)
                        if written > source.max_bytes:
                            raise ValueError(f"{source.name} exceeds the {source.max_bytes} byte limit")
                        if time.monotonic() - started > deadline_seconds:
                            raise TimeoutError(f"{source.name} download deadline exceeded")
                        stream.write(chunk)
            break
        except urllib.error.HTTPError as error:
            if error.code == 416 and offset:
                break  # Range starts at EOF: the partial file is complete; verify below.
            if error.code < 500 and error.code != 429 or attempts > retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if isinstance(error, TimeoutError) and time.monotonic() - started > deadline_seconds:
                raise
            if attempts > retries:
                raise
        sleep(min(60, 2**attempts))
    actual = sha256_file(partial)
    if actual != source.sha256:
        partial.unlink()
        raise ValueError(f"downloaded checksum mismatch for {source.name}: {actual}")
    partial.replace(destination)
    return {
        "sha256": actual,
        "bytes": destination.stat().st_size,
        "reused": False,
        "attempts": attempts,
        "resumed_bytes": resumed,
    }
