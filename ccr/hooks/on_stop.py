#!/usr/bin/env python3
"""Claude Code hook: fires on Stop (session end).

Reads accumulated session state and auto-commits if meaningful
progress was detected. Replaces the bare-bones on_session_end.py.

Also reconciles any Q&A turns present in the Claude Code transcript
but not yet logged via session_log_turn (transcript-based fallback).

Prints a brief confirmation to stdout so Claude Code sees the commit.
"""

import json
import os
import sys


def _log_hook_error(error_text: str) -> None:
    """Write error to .ccr/.hook_errors.log — non-fatal, never raises."""
    import datetime
    try:
        proj = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
        log_path = os.path.join(proj, ".ccr", ".hook_errors.log")
        if os.path.isdir(os.path.dirname(log_path)):
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- {ts} [on_stop] ---\n{error_text}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Transcript parsing helpers
# ---------------------------------------------------------------------------

def _read_stdin_payload() -> dict:
    """Read and parse the JSON payload from stdin (non-fatal)."""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                return json.loads(raw)
    except Exception:
        pass
    return {}


def _extract_text_from_content(content) -> str:
    """Extract plain text from a message content field.

    Content may be a plain string or a list of content blocks.
    We only capture text-type blocks; tool_use and tool_result are skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _parse_transcript(transcript_path: str) -> list[dict]:
    """Parse the JSONL transcript and return a list of (user, assistant) turn dicts.

    Only text-bearing human/assistant pairs are included.  Tool-use messages,
    tool_result messages, and any content-less messages are skipped.

    Returns:
        List of dicts with keys ``user_message`` and ``assistant_message``.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as fh:
            raw_lines = fh.readlines()
    except (OSError, IOError):
        return []

    # Parse all objects; skip malformed lines
    objects = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Walk through objects pairing user → assistant
    turns = []
    pending_user: str | None = None

    for obj in objects:
        obj_type = obj.get("type", "")
        message = obj.get("message", {})
        role = message.get("role", "")

        # Some transcript formats use "human" as role alias
        if obj_type in ("user", "human") or role in ("user", "human"):
            content = message.get("content", obj.get("content", ""))
            text = _extract_text_from_content(content)
            # Skip empty or tool-result-only messages
            if text:
                pending_user = text
        elif obj_type == "assistant" or role == "assistant":
            content = message.get("content", obj.get("content", ""))
            text = _extract_text_from_content(content)
            if text and pending_user is not None:
                turns.append({
                    "user_message": pending_user,
                    "assistant_message": text,
                })
                pending_user = None
            elif not text:
                # Assistant message with no extractable text (e.g. pure tool_use block)
                # Don't consume the pending user message — next assistant may have text
                pass

    return turns


def _reconcile_transcript(
    store,
    session_id: str,
    transcript_path: str,
) -> int:
    """Insert any transcript turns that were not already logged in the DB.

    Comparison is by turn_number (ordinal position).  If the DB already has
    N turns logged, only transcript turns beyond index N are inserted.

    Returns the number of turns inserted.
    """
    transcript_turns = _parse_transcript(transcript_path)
    if not transcript_turns:
        return 0

    # Build a set of turn_numbers already in the DB (handles sparse logging)
    try:
        existing_turns = store.get_session_turns(session_id, limit=10000)
        logged_turn_numbers: set[int] = {t["turn_number"] for t in existing_turns}
    except Exception:
        logged_turn_numbers = set()

    inserted = 0
    for idx, turn in enumerate(transcript_turns):
        turn_number_1based = idx + 1
        if turn_number_1based in logged_turn_numbers:
            # Already logged by session_log_turn — skip
            continue
        try:
            store.log_turn(
                session_id=session_id,
                user_message=turn["user_message"],
                assistant_message=turn["assistant_message"],
                tool_calls=[{"source": "transcript"}],
                source="transcript",
            )
            inserted += 1
        except Exception as _exc:
            if "UNIQUE" in str(_exc):
                pass  # Already logged by session_log_turn — expected on duplicate
            else:
                import traceback as _tb
                _log_hook_error(f"Transcript reconciliation turn {idx + 1}: {_tb.format_exc()}")

    return inserted


# ---------------------------------------------------------------------------
# C1: Session metrics writer
# ---------------------------------------------------------------------------

def _write_session_metrics(ccr_root: str, state, turn_count: int = 0) -> None:
    """Append a one-line JSON record to .ccr/metrics/sessions.jsonl (non-fatal).

    Records context_tokens injected and duration for 'ccr stats' ROI dashboard.
    Skips sessions where context_tokens == 0 (no memory was injected — likely
    a session that started before ccr install ran).

    Args:
        ccr_root: Path to the .ccr/ directory.
        state: SessionState object with context_tokens, start_time, session_id,
            and tool_calls (count of tool-use turns this session).
        turn_count: Number of Q&A turns in this session (from SessionStore).
            Used to estimate per-turn reminder overhead injected by CCR.
            Reminders fire only on turns where tool use has occurred, so overhead
            is capped at min(turn_count - 1, state.tool_calls) * 32 tokens.
            Pure Q&A sessions (tool_calls == 0) have zero reminder overhead.
            32 = measured length of _handle_subsequent_prompt output (129 chars ÷ 4).
    """
    if not getattr(state, "context_tokens", 0):
        return  # Nothing injected — skip to avoid skewing averages

    import datetime  # noqa: PLC0415

    try:
        metrics_dir = os.path.join(ccr_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = getattr(state, "start_time", 0)
        start_dt = (
            datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc)
            if start_time
            else now
        )
        duration_min = max(0, round((now - start_dt).total_seconds() / 60, 1))

        # Per-turn reminder overhead: _handle_subsequent_prompt emits ~32 tokens only
        # on turns where tool use has occurred. Cap overhead at actual tool-using turns
        # to avoid overcounting for pure-Q&A sessions (tool_calls == 0 → no reminders).
        # 32 = measured reminder text length (129 chars ÷ 4 chars/token).
        tool_call_turns = min(max(0, turn_count - 1), getattr(state, "tool_calls", 0))
        reminder_overhead_tokens = tool_call_turns * 32

        record = {
            "session_id": getattr(state, "session_id", "") or "",
            "start": start_dt.isoformat(),
            "end": now.isoformat(),
            "context_tokens": getattr(state, "context_tokens", 0),
            "duration_min": duration_min,
            "branch": ccr_root,  # coarse identifier; ccr stats reads branch from commits
            "reminder_overhead_tokens": reminder_overhead_tokens,
        }

        jsonl_path = os.path.join(metrics_dir, "sessions.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # Non-fatal — never block session end


# ---------------------------------------------------------------------------
# Main hook logic
# ---------------------------------------------------------------------------

def _auto_baseline_commit(mem: object, state: object, store: object, session_id: str) -> None:
    """Create a structural commit if no explicit gcc_commit was made this session.

    Priority 1: State-accumulator auto-commit when state.is_meaningful() is True
                (files modified, ops accumulated via on_tool_use hooks).
    Priority 2: Baseline summary from session turns (for pure-conversation sessions).

    Non-fatal: all errors logged, never propagated.

    Args:
        mem: MemoryManager instance (already initialized).
        state: SessionState (already loaded — do not reload after clear_state).
        store: SessionStore instance (already open), or None if session creation failed.
        session_id: Active session ID string, may be empty.
    """
    marker_path = os.path.join(mem.ccr_root, ".session_explicit_commit")

    # Clean up marker unconditionally; record whether it was present
    had_explicit = False
    try:
        if os.path.isfile(marker_path):
            had_explicit = True
            os.unlink(marker_path)
    except OSError:
        pass

    if had_explicit:
        return  # Explicit gcc_commit was made — no baseline needed

    # Priority 1: State accumulator (files touched / ops accumulated)
    if state.is_meaningful():
        try:
            fields = state.to_commit_fields()
            mem.commit(
                title=fields["title"],
                what=fields["what"],
                why=fields["why"],
                files_changed=fields["files_changed"],
                next_step=fields["next_step"],
                patterns_learned=fields["patterns_learned"],
            )
            print(f"[CCR] Auto-committed: {fields['title']}")
        except Exception as exc:
            _log_hook_error(f"Auto-commit failed: {exc}")
        return

    # Priority 2: Baseline summary from session turns
    has_files = bool(state.files_touched)
    turns: list = []
    if session_id and store is not None:
        try:
            turns = store.get_session_turns(session_id, limit=10)
        except Exception:
            pass

    if not has_files and not turns:
        return  # Truly empty session — skip

    # Quality gate: skip trivial sessions that would pollute the commit log
    # (e.g. "hi", "thanks", single-word greetings with no file activity)
    if not has_files:
        substantive = [
            t for t in turns
            if len(t.get("user_message", "").strip().split()) >= 3
        ]
        if not substantive:
            return  # Only trivial messages — skip

    try:
        from ccr.hooks.state_accumulator import extract_baseline_summary  # noqa: PLC0415
        fields = extract_baseline_summary(state, turns)
        mem.commit(
            title=fields["title"],
            what=fields["what"],
            why=fields["why"],
            files_changed=fields["files_changed"],
            next_step=fields["next_step"],
            author="[auto-baseline]",
        )
        print(f"[CCR] Auto-baseline: {fields['title']}")
    except Exception as exc:
        import traceback as _tb
        _log_hook_error(f"Auto-baseline commit failed: {_tb.format_exc()}")


def main() -> None:
    # Read stdin payload FIRST — stdin is only available at hook entry
    payload = _read_stdin_payload()
    transcript_path: str = payload.get("transcript_path", "")
    stdin_session_id: str = payload.get("session_id", "")

    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig
    from ccr.hooks.state_accumulator import clear_state, load_state

    mem = MemoryManager(project_root, CCRConfig())
    if not os.path.isdir(mem.ccr_root):
        if os.environ.get("CCR_AUTO_INIT", "").lower() in ("1", "true", "yes"):
            mem.ensure_structure()
        else:
            return

    state = load_state(mem.ccr_root)

    # Clean up session state
    clear_state(mem.ccr_root)

    # Delete session marker so next session gets full memory context injection
    # (Without this, _is_first_prompt() sees the stale marker and injects only
    # the 19-token reminder instead of the full gcc_context block.)
    try:
        os.unlink(os.path.join(mem.ccr_root, ".session_active"))
    except (FileNotFoundError, OSError):
        pass  # Non-fatal — marker may already be absent

    # Finalize session logger DB entry and run transcript reconciliation (non-fatal)
    try:
        id_file = os.path.join(mem.ccr_root, ".current_session_id")
        session_id = ""
        if os.path.isfile(id_file):
            with open(id_file, "r", encoding="utf-8") as f:
                session_id = f.read().strip()

        # Fall back to session_id from stdin payload if file is missing/empty
        if not session_id and stdin_session_id:
            session_id = stdin_session_id

        turn_count = 0
        store = None
        if session_id:
            from ccr.core.session_store import SessionStore  # noqa: PLC0415
            store = SessionStore(os.path.join(mem.ccr_root, "sessions.db"))
            store.finalize_session(session_id)

            # Transcript reconciliation: insert any turns Claude forgot to log
            if transcript_path and os.path.isfile(transcript_path):
                inserted = _reconcile_transcript(store, session_id, transcript_path)
                if inserted:
                    print(f"[CCR] Transcript reconciliation: inserted {inserted} missing turn(s).")
        else:
            _log_hook_error(
                "on_stop: session_id is empty — skipping DB finalize and reconciliation. "
                "Auto-baseline will still fire using file state only."
            )

        # Auto-baseline: fires regardless of session_id so no session is ever silently dropped.
        # When session_id is empty, store=None and _auto_baseline_commit falls back to
        # file-only state (files_touched). This covers the case where _create_db_session
        # silently failed at session start.
        _auto_baseline_commit(mem, state, store, session_id)

        # Read final turn count for reminder-overhead estimate in session metrics
        if session_id and store is not None:
            try:
                conn_rows = store.get_session_turns(session_id, limit=10000)
                turn_count = len(conn_rows)
            except Exception:
                turn_count = 0

        # C1: Write session metrics record — done after store is open so we
        # can pass the real turn count for per-turn reminder overhead estimation.
        _write_session_metrics(mem.ccr_root, state, turn_count=turn_count)

        if os.path.isfile(id_file):
            os.unlink(id_file)
    except Exception:
        import traceback
        _log_hook_error(traceback.format_exc())

    # Clean up any pending user message buffer
    try:
        pending = os.path.join(mem.ccr_root, ".pending_user_msg")
        if os.path.isfile(pending):
            os.unlink(pending)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        _log_hook_error(traceback.format_exc())
