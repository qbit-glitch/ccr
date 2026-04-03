"""Tests for working memory scratchpad (AgeMem-inspired)."""

import json
import os
import tempfile
import threading
import time

import pytest

from ccr.core.scratchpad import Scratchpad, ScratchpadEntry


@pytest.fixture
def scratch_path(tmp_path):
    return str(tmp_path / "scratchpad.json")


@pytest.fixture
def scratchpad(scratch_path):
    return Scratchpad(scratch_path)


class TestScratchpadEntry:
    def test_to_dict(self):
        entry = ScratchpadEntry(
            key="test", value="hello", created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z", access_count=3,
        )
        d = entry.to_dict()
        assert d["value"] == "hello"
        assert d["access_count"] == 3
        assert "key" not in d  # key is the dict key, not in value

    def test_from_dict(self):
        d = {"value": "world", "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z", "access_count": 5}
        entry = ScratchpadEntry.from_dict("mykey", d)
        assert entry.key == "mykey"
        assert entry.value == "world"
        assert entry.access_count == 5

    def test_from_dict_defaults(self):
        d = {"value": "minimal"}
        entry = ScratchpadEntry.from_dict("k", d)
        assert entry.created_at == ""
        assert entry.access_count == 0


class TestScratchpad:
    def test_set_and_get(self, scratchpad):
        scratchpad.set("focus", "authentication module")
        entry = scratchpad.get("focus")
        assert entry is not None
        assert entry.value == "authentication module"
        assert entry.access_count == 1

    def test_get_missing_key(self, scratchpad):
        assert scratchpad.get("nonexistent") is None

    def test_set_updates_existing(self, scratchpad):
        scratchpad.set("focus", "v1")
        scratchpad.set("focus", "v2")
        entry = scratchpad.get("focus")
        assert entry.value == "v2"

    def test_get_increments_access_count(self, scratchpad):
        scratchpad.set("key", "val")
        scratchpad.get("key")
        scratchpad.get("key")
        entry = scratchpad.get("key")
        assert entry.access_count == 3

    def test_list_entries_sorted_by_updated(self, scratchpad):
        scratchpad.set("a", "first")
        scratchpad.set("b", "second")
        scratchpad.set("c", "third")
        entries = scratchpad.list_entries()
        assert len(entries) == 3
        assert entries[0].key == "c"  # most recently updated

    def test_list_entries_empty(self, scratchpad):
        assert scratchpad.list_entries() == []

    def test_delete_existing(self, scratchpad):
        scratchpad.set("temp", "data")
        assert scratchpad.delete("temp") is True
        assert scratchpad.get("temp") is None

    def test_delete_nonexistent(self, scratchpad):
        assert scratchpad.delete("ghost") is False

    def test_clear(self, scratchpad):
        scratchpad.set("a", "1")
        scratchpad.set("b", "2")
        scratchpad.set("c", "3")
        count = scratchpad.clear()
        assert count == 3
        assert scratchpad.size == 0

    def test_clear_empty(self, scratchpad):
        assert scratchpad.clear() == 0

    def test_size(self, scratchpad):
        assert scratchpad.size == 0
        scratchpad.set("x", "y")
        assert scratchpad.size == 1
        scratchpad.set("a", "b")
        assert scratchpad.size == 2

    def test_persistence(self, scratch_path):
        sp1 = Scratchpad(scratch_path)
        sp1.set("persist", "me")
        sp2 = Scratchpad(scratch_path)
        entry = sp2.get("persist")
        assert entry is not None
        assert entry.value == "me"

    def test_persistence_after_delete(self, scratch_path):
        sp1 = Scratchpad(scratch_path)
        sp1.set("a", "1")
        sp1.set("b", "2")
        sp1.delete("a")
        sp2 = Scratchpad(scratch_path)
        assert sp2.get("a") is None
        assert sp2.get("b") is not None

    def test_format_for_context_empty(self, scratchpad):
        assert scratchpad.format_for_context() == ""

    def test_format_for_context(self, scratchpad):
        scratchpad.set("focus", "auth module")
        scratchpad.set("hypothesis", "race condition in login")
        text = scratchpad.format_for_context()
        assert "# Working Memory" in text
        assert "**focus**" in text
        assert "auth module" in text
        assert "**hypothesis**" in text

    def test_thread_safety(self, scratchpad):
        """Concurrent set/get operations should not corrupt state."""
        errors = []
        def writer(n):
            try:
                for i in range(20):
                    scratchpad.set(f"key-{n}-{i}", f"val-{n}-{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    scratchpad.list_entries()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_corrupt_file_handled(self, scratch_path):
        os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
        with open(scratch_path, "w") as f:
            f.write("not json{{{")
        sp = Scratchpad(scratch_path)
        assert sp.size == 0  # gracefully empty

    def test_missing_file_handled(self, tmp_path):
        sp = Scratchpad(str(tmp_path / "nonexistent" / "scratchpad.json"))
        assert sp.size == 0

    def test_set_timestamps(self, scratchpad):
        entry = scratchpad.set("ts", "test")
        assert entry.created_at != ""
        assert entry.updated_at != ""
        assert entry.created_at == entry.updated_at  # same on creation

    def test_update_preserves_created_at(self, scratchpad):
        e1 = scratchpad.set("ts", "v1")
        created = e1.created_at
        e2 = scratchpad.set("ts", "v2")
        assert e2.created_at == created  # created_at unchanged


class TestScratchpadTTL:
    def test_get_does_not_write_disk(self, scratch_path):
        """get() must not write to disk — access_count increment is in-memory only."""
        sp = Scratchpad(scratch_path)
        sp.set("key", "value")
        mtime_before = os.path.getmtime(scratch_path)
        # Call get() several times and verify the file mtime is unchanged
        sp.get("key")
        sp.get("key")
        mtime_after = os.path.getmtime(scratch_path)
        assert mtime_before == mtime_after

    def test_ttl_expiry(self, scratchpad):
        """An entry set with ttl_seconds=1 should return None after ~1 second."""
        scratchpad.set("short", "lived", ttl_seconds=1)
        # Confirm it's there immediately
        assert scratchpad.get("short") is not None
        # Wait for expiry
        time.sleep(1.1)
        assert scratchpad.get("short") is None

    def test_ttl_not_expired(self, scratchpad):
        """An entry set with ttl_seconds=60 should still be readable immediately."""
        scratchpad.set("long", "lived", ttl_seconds=60)
        entry = scratchpad.get("long")
        assert entry is not None
        assert entry.value == "lived"

    def test_no_ttl_never_expires(self, scratchpad):
        """An entry set without TTL should have expires_at=None."""
        entry = scratchpad.set("permanent", "data")
        assert entry.expires_at is None

    def test_ttl_entry_excluded_from_list(self, scratchpad):
        """Expired entries should not appear in list_entries()."""
        scratchpad.set("live", "yes", ttl_seconds=60)
        scratchpad.set("dead", "no", ttl_seconds=1)
        # Verify both initially present
        assert len(scratchpad.list_entries()) == 2
        time.sleep(1.1)
        # After expiry, only the live entry should appear
        entries = scratchpad.list_entries()
        keys = [e.key for e in entries]
        assert "live" in keys
        assert "dead" not in keys

    def test_ttl_entry_excluded_from_format_for_context(self, scratchpad):
        """Expired entries should not appear in format_for_context()."""
        scratchpad.set("active", "yes", ttl_seconds=60)
        scratchpad.set("expired", "no", ttl_seconds=1)
        time.sleep(1.1)
        text = scratchpad.format_for_context()
        assert "active" in text
        assert "expired" not in text

    def test_ttl_entry_persists_expires_at_on_disk(self, scratch_path):
        """The expires_at field should be written to and read from disk."""
        sp1 = Scratchpad(scratch_path)
        sp1.set("ttl-key", "ttl-val", ttl_seconds=300)
        # Load fresh instance from disk
        sp2 = Scratchpad(scratch_path)
        entry = sp2.get("ttl-key")
        assert entry is not None
        assert entry.expires_at is not None
