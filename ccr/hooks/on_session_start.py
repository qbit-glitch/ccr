#!/usr/bin/env python3
"""Claude Code hook: fires on UserPromptSubmit.

Injects ACE playbook + level-1 memory context into the conversation
by printing to stdout (Claude Code captures hook stdout as context).

On first prompt of a session, outputs a strong directive to use CCR tools.
On subsequent prompts, outputs a lighter reminder to commit if needed.
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
                f.write(f"\n--- {ts} [on_session_start] ---\n{error_text}\n")
    except Exception:
        pass


_MARKER_MAX_AGE_SECONDS = 7200  # 2 hours — auto-invalidate force-killed sessions


def _is_first_prompt(ccr_root: str) -> tuple[bool, bool]:
    """Check if this is the first prompt of the session using a marker file.

    Three-layer stale detection:
    1. Atomic O_CREAT|O_EXCL create — succeeds only if no marker exists.
    2. PID validation — if marker exists, check if owning process is alive.
    3. Age check — markers older than 2 hours are treated as stale even if
       the PID happens to be reused by a different process (covers force-kill).

    Returns:
        (is_first_prompt, was_stale) — was_stale=True when a stale marker was
        replaced, so the caller can emit a crash-recovery notice.
    """
    import time as _time

    marker = os.path.join(ccr_root, ".session_active")
    try:
        # Atomic create — succeeds only if no marker exists
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True, False
    except FileExistsError:
        # Marker exists — check age first (catches force-kill / kill -9)
        try:
            mtime = os.path.getmtime(marker)
            if _time.time() - mtime > _MARKER_MAX_AGE_SECONDS:
                # Marker is older than 2 hours — definitely stale
                with open(marker, "w") as f:
                    f.write(str(os.getpid()))
                return True, True
        except OSError:
            pass

        # Age is fine — check if the owning process is still alive
        try:
            with open(marker, "r") as f:
                stored_pid = int(f.read().strip())
            os.kill(stored_pid, 0)  # Raises OSError if process is dead
            return False, False  # Genuine mid-session (process alive)
        except (ValueError, OSError):
            # Stale marker: process is dead or PID unreadable — replace
            try:
                with open(marker, "w") as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass
            return True, True
    except OSError:
        return False, False


def _project_has_commits(mem) -> bool:
    """Return True if at least one commit exists on the active branch."""
    import re
    try:
        branch = mem.get_active_branch()
        recent = mem._read_commits_window(branch, 0, 1)
        return bool(re.search(r"## \[C\d{3,}\]", recent))
    except Exception:
        return False


def _read_prompt_from_stdin() -> str:
    """Try to parse the user prompt from stdin JSON (UserPromptSubmit hook payload)."""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                data = json.loads(raw)
                return data.get("prompt", "")
    except Exception:
        pass
    return ""


def main():
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Read user prompt from stdin before any other I/O
    user_prompt = _read_prompt_from_stdin()

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig

    mem = MemoryManager(project_root, CCRConfig())
    if not os.path.isdir(mem.ccr_root):
        if os.environ.get("CCR_AUTO_INIT", "").lower() in ("1", "true", "yes"):
            mem.ensure_structure()
        else:
            return  # No .ccr/ directory yet — skip

    first_prompt, was_stale = _is_first_prompt(mem.ccr_root)

    if first_prompt:
        # Initialize fresh session state for auto-commit accumulation
        from ccr.hooks.state_accumulator import initialize_state
        initialize_state(mem.ccr_root)
        _handle_session_start(mem, project_root, user_prompt, was_stale=was_stale)
    else:
        _handle_subsequent_prompt(mem, user_prompt)


def _buffer_user_prompt(ccr_root: str, prompt: str) -> None:
    """Write user prompt to .pending_user_msg for session_log_turn to consume."""
    if not prompt:
        return
    try:
        tmp = os.path.join(ccr_root, ".pending_user_msg.tmp")
        dst = os.path.join(ccr_root, ".pending_user_msg")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(prompt)
        os.replace(tmp, dst)
    except OSError:
        pass


def _create_db_session(ccr_root: str, project_root: str) -> None:
    """Create a new session in sessions.db and write .current_session_id (non-fatal)."""
    try:
        from ccr.core.session_store import SessionStore  # noqa: PLC0415
        db_path = os.path.join(ccr_root, "sessions.db")
        store = SessionStore(db_path)
        sid = store.create_session(project=project_root)
        tmp = os.path.join(ccr_root, ".current_session_id.tmp")
        dst = os.path.join(ccr_root, ".current_session_id")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(sid)
        os.replace(tmp, dst)
    except Exception:
        pass  # never block session start


_DIRECTIVES: dict[str, str] = {
    "default": """\
<MANDATORY_CCR_ACTIONS>
Memory already loaded. Respond directly to the user.

REQUIRED after each task:
  gcc_commit(title, what, why, files_changed, next_step)

WHEN NEEDED:
  gcc_context(level=3) — deeper context
  gcc_search(query)    — find past decisions
  ace_get_playbook()   — see evolved strategies
  session_log_turn(assistant_message="...") — only if you need real-time turn access;
                                              turns are auto-captured at session end
</MANDATORY_CCR_ACTIONS>""",

    "ml": """\
<MANDATORY_CCR_ACTIONS>
Memory already loaded. ML Research mode active.

REQUIRED after each experiment/task:
  gcc_commit(title, what, why, files_changed, next_step,
             experiment={"id": "...", "hypothesis": "...",
                         "metrics": {...}, "conclusion": "..."})

WHEN NEEDED:
  gcc_experiments()           — browse experiment history + metrics
  gcc_branch(name, purpose)   — isolate a hypothesis
  index_search(query)         — find code/config files
  session_log_turn(assistant_message="...") — only if real-time turn access needed
</MANDATORY_CCR_ACTIONS>""",

    "academic": """\
<MANDATORY_CCR_ACTIONS>
Memory already loaded. Academic Research mode active.

REQUIRED after each writing/analysis task:
  gcc_commit(title, what, why, files_changed, next_step)

WHEN NEEDED:
  gcc_discuss(topic, position) — record argument or decision
  gcc_discussions()            — retrieve past positions/notes
  gcc_search(query)            — search all past analysis
  session_search(query)        — search conversation history
  session_log_turn(assistant_message="...") — only if real-time turn access needed
</MANDATORY_CCR_ACTIONS>""",
}


def _get_preset(ccr_root: str) -> str:
    """Read preset from .ccr/metadata.yaml. Returns 'default' if unset or on error."""
    try:
        import yaml  # noqa: PLC0415 — lazy import keeps hook startup fast
        meta_path = os.path.join(ccr_root, "metadata.yaml")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return (yaml.safe_load(f) or {}).get("preset", "default")
    except ImportError:
        _log_hook_error(
            "pyyaml not installed — preset from metadata.yaml cannot be read. "
            "Install it: pip install pyyaml"
        )
    except Exception:
        pass
    return "default"


def _handle_session_start(mem, project_root: str = "", user_prompt: str = "", was_stale: bool = False):
    """First prompt of session: inject full context + strong directive."""
    # Create DB session and buffer user prompt (non-fatal)
    _create_db_session(mem.ccr_root, project_root or "")
    _buffer_user_prompt(mem.ccr_root, user_prompt)

    # Empty-project UX: skip MANDATORY_CCR_ACTIONS when there's nothing to recall
    if not _project_has_commits(mem):
        print("""<ccr_ready>
CCR is active. This project has no memory yet — that's normal on first use.

After finishing your first task, call:
  gcc_commit(title="...", what="...", why="...", files_changed=[...], next_step="...")

CCR will then remember your progress across all future sessions automatically.
See: docs/quickstart-students.md for a 5-minute guide.
Tip: use `ccr export-context` to export memory for claude.ai sessions.
</ccr_ready>""")
        return

    # Get level-2 context (rolling summary + last 3 commits) — pre-injected so
    # Claude doesn't need to call gcc_context(level=2) before responding.
    context = mem.get_context(level=2)

    # Get global playbook (~/.ccr/)
    global_playbook_path = os.path.expanduser("~/.ccr/global_playbook.txt")
    global_text = ""
    if os.path.isfile(global_playbook_path):
        with open(global_playbook_path, "r", encoding="utf-8") as f:
            global_text = f.read().strip()

    # Get project playbook (.ccr/)
    playbook_path = os.path.join(mem.ccr_root, "playbook.txt")
    playbook_text = ""
    if os.path.isfile(playbook_path):
        with open(playbook_path, "r", encoding="utf-8") as f:
            playbook_text = f.read().strip()

    # Log session start
    mem.log_ota(
        tool_name="session-start",
        observation="New Claude Code session",
        thought="Injecting memory context and playbook",
        action="Hook fired on UserPromptSubmit",
    )

    # Output for Claude Code to capture
    parts = []
    if was_stale:
        parts.append(
            "⚡ CCR recovered from an unclean shutdown — memory context freshly injected. "
            "Everything is OK; no action needed. "
            "If this message appears every session, run `ccr doctor` to check for hook errors."
        )
    if context.strip():
        parts.append(f"<gcc_context>\n{context}\n</gcc_context>")
    if global_text or playbook_text:
        pb_parts = []
        if global_text:
            pb_parts.append(f"# GLOBAL STRATEGIES (all projects)\n{global_text}")
        if playbook_text:
            pb_parts.append(f"# PROJECT STRATEGIES (this project)\n{playbook_text}")
        parts.append(f"<ace_playbook>\n{chr(10).join(pb_parts)}\n</ace_playbook>")

    # Session-start directive: preset-aware, surfaces only the relevant tool subset.
    # Level-2 context already injected above — no need to call gcc_context again.
    preset = _get_preset(mem.ccr_root)
    parts.append(_DIRECTIVES.get(preset, _DIRECTIVES["default"]))

    if parts:
        output = "\n\n".join(parts)
        print(output)

        # C1: Estimate tokens injected and save to session state for ccr stats
        try:
            from ccr.hooks.state_accumulator import load_state, save_state  # noqa: PLC0415
            ctx_chars = len(output)
            ctx_tokens = max(1, ctx_chars // 4)  # ~4 chars/token heuristic
            state = load_state(mem.ccr_root)
            state.context_tokens = ctx_tokens
            save_state(mem.ccr_root, state)
        except Exception:
            pass  # Never block session start


def _handle_subsequent_prompt(mem, user_prompt: str = ""):
    """Subsequent prompts: light reminder to commit if progress was made."""
    # Buffer user prompt for session_log_turn to consume
    _buffer_user_prompt(mem.ccr_root, user_prompt)

    # Skip reminder if no tool use has occurred yet — nothing to commit
    try:
        from ccr.hooks.state_accumulator import load_state  # noqa: PLC0415
        state = load_state(mem.ccr_root)
        if state.tool_calls == 0:
            return  # No work done yet — reminder would be noise
    except Exception:
        pass  # On any error, default to showing the reminder

    print("""<ccr_reminder>
Remember: call gcc_commit after completing tasks. Call ace_update_counters after significant work.
</ccr_reminder>""")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        _log_hook_error(traceback.format_exc())
