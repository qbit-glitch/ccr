"""Tests for the Codex CCR wrapper lifecycle."""

from __future__ import annotations

import json
import os

from ccr.core.session_store import SessionStore
from ccr.hooks.state_accumulator import SessionState, save_state, state_path


def _write_codex_transcript(path: str, user: str, assistant: str, cwd: str) -> None:
    lines = [
        json.dumps({"type": "session_meta", "payload": {"cwd": cwd}}),
        json.dumps({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            },
        }),
        json.dumps({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": assistant}],
            },
        }),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_run_codex_invokes_real_binary_and_finalizes(tmp_path, monkeypatch):
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nexit 23\n")
    fake_codex.chmod(0o755)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CCR_CODEX_BINARY", str(fake_codex))

    from ccr.cli_codex import run_codex

    assert run_codex(["--version"]) == 23
    assert (tmp_path / ".ccr").is_dir()
    assert (tmp_path / ".ccr" / "memory.db").is_file()
    assert not (tmp_path / ".ccr" / ".codex_wrapper_state.json").exists()


def test_global_codex_helper_delegates_to_python_wrapper(tmp_path):
    from ccr.cli_global import _write_codex_ccr

    _write_codex_ccr(str(tmp_path), "/usr/bin/python3")

    helper = tmp_path / "codex-ccr"
    text = helper.read_text()
    assert helper.exists()
    assert os.access(helper, os.X_OK)
    assert "-m ccr.cli_codex" in text


def test_finalize_codex_session_reconciles_transcript_and_cleans_state(tmp_path, monkeypatch):
    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()

    db_path = ccr_root / "sessions.db"
    store = SessionStore(str(db_path))
    sid = store.create_session(project=str(tmp_path))
    store.close()

    (ccr_root / ".current_session_id").write_text(sid)
    (ccr_root / ".pending_user_msg").write_text("Implement wrapper")
    (ccr_root / ".session_active").write_text("12345")
    save_state(
        str(ccr_root),
        SessionState(start_time=123.0, context_tokens=42, session_id="shortid"),
    )

    transcript = tmp_path / "codex.jsonl"
    _write_codex_transcript(str(transcript), "Implement wrapper", "Done.", str(tmp_path))

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("ccr.hooks.on_stop._auto_baseline_commit", lambda *args: None)
    monkeypatch.setattr("ccr.hooks.on_stop._write_session_metrics", lambda *args, **kwargs: None)

    from ccr.cli_codex import finalize_codex_session

    finalize_codex_session(str(tmp_path), transcript_path=str(transcript))

    store2 = SessionStore(str(db_path))
    turns = store2.get_session_turns(sid)
    store2.close()
    assert len(turns) == 1
    assert turns[0]["user_message"] == "Implement wrapper"
    assert turns[0]["assistant_message"] == "Done."
    assert turns[0]["source"] == "transcript"
    assert not os.path.isfile(state_path(str(ccr_root)))
    assert not (ccr_root / ".current_session_id").exists()
    assert not (ccr_root / ".pending_user_msg").exists()
    assert not (ccr_root / ".session_active").exists()
