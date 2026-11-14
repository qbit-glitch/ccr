"""Tests for the cached dataset downloader."""

from __future__ import annotations

import hashlib
import os

import pytest

from ccr.bench.download import DownloadError, fetch


def _src(tmp_path, content: bytes = b"hello"):
    p = tmp_path / "src" / "data.json"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(content)
    return p


def test_fetch_copies_once_then_serves_from_cache(tmp_path) -> None:
    src = _src(tmp_path)
    cache = str(tmp_path / "cache")
    first = fetch(src.as_uri(), cache)
    src.write_bytes(b"changed upstream")
    second = fetch(src.as_uri(), cache)
    assert first == second
    assert open(first, "rb").read() == b"hello"


def test_fetch_records_sha256_sidecar(tmp_path) -> None:
    src = _src(tmp_path)
    path = fetch(src.as_uri(), str(tmp_path / "cache"))
    expected = hashlib.sha256(b"hello").hexdigest()
    assert open(path + ".sha256").read().strip() == expected


def test_wrong_expected_sha_raises_and_leaves_no_cached_file(tmp_path) -> None:
    src = _src(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(DownloadError, match="sha256"):
        fetch(src.as_uri(), str(cache), expected_sha256="0" * 64)
    leftovers = [f for _, _, files in os.walk(cache) for f in files]
    assert leftovers == []


def test_correct_expected_sha_passes(tmp_path) -> None:
    src = _src(tmp_path)
    good = hashlib.sha256(b"hello").hexdigest()
    assert fetch(src.as_uri(), str(tmp_path / "cache"), expected_sha256=good)


def test_missing_source_raises_download_error(tmp_path) -> None:
    missing = (tmp_path / "nope.json").as_uri()
    with pytest.raises(DownloadError):
        fetch(missing, str(tmp_path / "cache"))
