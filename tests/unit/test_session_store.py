"""Unit tests for ccr.core.session_store.SessionStore."""

from __future__ import annotations

import json
import threading

import pytest

from ccr.core.session_store import SessionStore, _estimate_tokens, _make_session_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "sessions.db")


@pytest.fixture
def store(db_path):
    s = SessionStore(db_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_make_session_id_format():
    sid = _make_session_id()
    assert sid.startswith("ses_")
    assert len(sid) > 12


def test_estimate_tokens():
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("a" * 4) == 1
    assert _estimate_tokens("a" * 400) == 100


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_session_returns_unique_ids(store):
    sid1 = store.create_session()
    sid2 = store.create_session()
    assert sid1 != sid2
    assert sid1.startswith("ses_")
    assert sid2.startswith("ses_")


def test_create_session_stores_project(store):
    sid = store.create_session(project="/some/project")
    sessions = store.list_sessions()
    assert any(s["id"] == sid and s["project"] == "/some/project" for s in sessions)


def test_finalize_session_sets_ended_at(store):
    sid = store.create_session()
    # ended_at is NULL before finalize
    sessions = store.list_sessions()
    row = next(s for s in sessions if s["id"] == sid)
    assert row["ended_at"] is None
    store.finalize_session(sid)
    sessions = store.list_sessions()
    row = next(s for s in sessions if s["id"] == sid)
    assert row["ended_at"] is not None


def test_list_sessions_ordered_newest_first(store):
    sid1 = store.create_session()
    sid2 = store.create_session()
    sessions = store.list_sessions()
    ids = [s["id"] for s in sessions]
    assert ids.index(sid2) < ids.index(sid1)


# ---------------------------------------------------------------------------
# Turn logging
# ---------------------------------------------------------------------------


def test_log_turn_increments_turn_number(store):
    sid = store.create_session()
    t1 = store.log_turn(sid, "hello", "world")
    t2 = store.log_turn(sid, "foo", "bar")
    t3 = store.log_turn(sid, "baz", "qux")
    assert t1 == 1
    assert t2 == 2
    assert t3 == 3


def test_log_turn_stores_and_retrieves_text(store):
    sid = store.create_session()
    store.log_turn(sid, "What is 2+2?", "The answer is 4.")
    turns = store.get_session_turns(sid)
    assert len(turns) == 1
    assert turns[0]["user_message"] == "What is 2+2?"
    assert turns[0]["assistant_message"] == "The answer is 4."


def test_log_turn_stores_tool_calls(store):
    sid = store.create_session()
    tools = [{"name": "Read", "input": {"file_path": "foo.py"}}]
    store.log_turn(sid, "u", "a", tool_calls=tools)
    turns = store.get_session_turns(sid)
    assert json.loads(turns[0]["tool_calls_json"]) == tools


def test_log_turn_stores_files_touched(store):
    sid = store.create_session()
    files = ["ccr/core/session_store.py", "tests/unit/test_session_store.py"]
    store.log_turn(sid, "u", "a", files_touched=files)
    turns = store.get_session_turns(sid)
    assert json.loads(turns[0]["files_touched_json"]) == files


def test_log_turn_auto_creates_session_if_missing(store):
    # Passing a session_id that doesn't exist should create a transient session
    turn_num = store.log_turn("nonexistent_session_id", "u", "a")
    assert turn_num == 1
    turns = store.get_session_turns("nonexistent_session_id")
    assert len(turns) == 1


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_get_session_turns_limit_and_offset(store):
    sid = store.create_session()
    for i in range(5):
        store.log_turn(sid, f"q{i}", f"a{i}")
    page1 = store.get_session_turns(sid, limit=2, offset=0)
    page2 = store.get_session_turns(sid, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0]["turn_number"] == 1
    assert page2[0]["turn_number"] == 3


def test_get_session_turns_empty_session(store):
    sid = store.create_session()
    turns = store.get_session_turns(sid)
    assert turns == []


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_turns_finds_matching_text(store):
    sid = store.create_session()
    store.log_turn(sid, "how do I use sqlite3?", "Use the sqlite3 module from stdlib.")
    store.log_turn(sid, "what is python?", "A high-level language.")
    results = store.search_turns("sqlite3")
    assert len(results) >= 1
    # Check the session_id and turn_number are present
    assert results[0]["session_id"] == sid
    assert results[0]["turn_number"] == 1


def test_search_turns_empty_result(store):
    sid = store.create_session()
    store.log_turn(sid, "hello", "world")
    results = store.search_turns("zzznomatch_xyz")
    assert results == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_json_format(store):
    sid = store.create_session()
    store.log_turn(sid, "user q", "assistant a")
    data_str = store.export_session(sid, fmt="json")
    data = json.loads(data_str)
    assert "session" in data
    assert "turns" in data
    assert len(data["turns"]) == 1


def test_export_jsonl_format(store):
    sid = store.create_session()
    store.log_turn(sid, "q1", "a1")
    store.log_turn(sid, "q2", "a2")
    data_str = store.export_session(sid, fmt="jsonl")
    lines = [l for l in data_str.strip().splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "messages" in obj
        assert obj["messages"][0]["role"] == "user"
        assert obj["messages"][1]["role"] == "assistant"


def test_export_markdown_format(store):
    sid = store.create_session()
    store.log_turn(sid, "what is 1+1?", "It is 2.")
    data_str = store.export_session(sid, fmt="markdown")
    assert "## Turn 1" in data_str
    assert "what is 1+1?" in data_str
    assert "It is 2." in data_str


def test_export_unknown_format_raises(store):
    sid = store.create_session()
    with pytest.raises(ValueError, match="Unknown export format"):
        store.export_session(sid, fmt="csv")


# ---------------------------------------------------------------------------
# source field
# ---------------------------------------------------------------------------


def test_log_turn_default_source_is_direct(store):
    """log_turn without source= stores 'direct'."""
    sid = store.create_session()
    store.log_turn(sid, "hello", "world")
    turns = store.get_session_turns(sid)
    assert turns[0]["source"] == "direct"


def test_log_turn_source_field(store):
    """log_turn with source='transcript' is stored and retrievable."""
    sid = store.create_session()
    store.log_turn(sid, "reconstructed user", "reconstructed assistant", source="transcript")
    turns = store.get_session_turns(sid)
    assert len(turns) == 1
    assert turns[0]["source"] == "transcript"
    assert turns[0]["user_message"] == "reconstructed user"
    assert turns[0]["assistant_message"] == "reconstructed assistant"


def test_log_turn_mixed_sources(store):
    """Direct and transcript turns can coexist in the same session."""
    sid = store.create_session()
    store.log_turn(sid, "q1", "a1", source="direct")
    store.log_turn(sid, "q2", "a2", source="transcript")
    turns = store.get_session_turns(sid)
    assert turns[0]["source"] == "direct"
    assert turns[1]["source"] == "transcript"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety(db_path):
    """Multiple threads can log turns concurrently without data corruption."""
    store = SessionStore(db_path)
    sid = store.create_session()
    errors = []

    def worker():
        try:
            for _ in range(5):
                store.log_turn(sid, "u", "a")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store.close()
    assert errors == [], f"Thread errors: {errors}"
    # All 15 turns should be present
    store2 = SessionStore(db_path)
    turns = store2.get_session_turns(sid, limit=100)
    store2.close()
    assert len(turns) == 15


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence(db_path):
    """Data written in one store instance survives close/reopen."""
    s1 = SessionStore(db_path)
    sid = s1.create_session(project="test-project")
    s1.log_turn(sid, "hello", "world")
    s1.close()

    s2 = SessionStore(db_path)
    turns = s2.get_session_turns(sid)
    s2.close()
    assert len(turns) == 1
    assert turns[0]["user_message"] == "hello"
    assert turns[0]["assistant_message"] == "world"
