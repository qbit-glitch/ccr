"""Cached dataset downloads using only the standard library."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.error
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
USER_AGENT = "ccr-bench"


class DownloadError(Exception):
    """Raised when a dataset cannot be fetched or fails its checksum."""


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, cache_dir: str, expected_sha256: str | None = None) -> str:
    """Download ``url`` into ``cache_dir`` once and return the local path.

    A cached file is reused without contacting the server. The file's sha256 is
    written beside it as ``<file>.sha256`` for provenance in reports.
    """
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    name = os.path.basename(urlparse(url).path) or "download"
    target_dir = os.path.join(cache_dir, key)
    path = os.path.join(target_dir, name)
    if os.path.isfile(path):
        return path
    os.makedirs(target_dir, exist_ok=True)
    part = path + ".part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, CHUNK)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _remove(part)
        raise DownloadError(f"could not fetch {url}: {exc}") from exc
    actual = _sha256_of(part)
    if expected_sha256 is not None and actual != expected_sha256:
        _remove(part)
        raise DownloadError(
            f"sha256 mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    os.replace(part, path)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(actual + "\n")
    logger.info("fetched %s -> %s", url, path)
    return path


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
