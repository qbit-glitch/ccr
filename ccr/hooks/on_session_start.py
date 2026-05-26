#!/usr/bin/env python3
"""Hook: fires at session start (first prompt).

Injects ACE playbook + level-1 memory context into the conversation
by printing to stdout (agent captures hook stdout as context).

On first prompt of a session, outputs a strong directive to use CCR tools.
On subsequent prompts, outputs a lighter reminder to commit if needed.

Provider-agnostic: reads CanonicalEvent payload from stdin, uses the
agent-declared ContextFormat for output.
"""

import os
import re
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

    Returns:
        (is_first_prompt, was_stale) — was_stale=True when a stale marker was
        replaced, so the caller can emit a crash-recovery notice.
    """
    import time as _time

    marker = os.path.join(ccr_root, ".session_active")
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True, False
    except FileExistsError:
        try:
            mtime = os.path.getmtime(marker)
            if _time.time() - mtime > _MARKER_MAX_AGE_SECONDS:
                with open(marker, "w") as f:
                    f.write(str(os.getpid()))
                return True, True
        except OSError:
            pass

        try:
            with open(marker, "r") as f:
                stored_pid = int(f.read().strip())
            os.kill(stored_pid, 0)
            return False, False
        except (ValueError, OSError):
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
        pass


# Preset-aware directives — content only, formatting is applied later
_DIRECTIVES = {
    "default": (
        "<!-- MANDATORY_CCR_ACTIONS -->\n"
        "Memory already loaded. Respond directly to the user.\n\n"
        "REQUIRED after each task:\n"
        "  gcc_commit(title, what, why, files_changed, next_step)\n\n"
        "WHEN NEEDED:\n"
        "  gcc_context(level=3) — deeper context\n"
        "  gcc_search(query)    — find past decisions\n"
        "  ace_get_playbook()   — see evolved strategies\n"
        "  session_log_turn(assistant_message=\"...\") — only if you need real-time turn access"
    ),
    "ml": (
        "<!-- MANDATORY_CCR_ACTIONS -->\n"
        "Memory already loaded. ML Research mode active.\n\n"
        "REQUIRED after each experiment/task:\n"
        '  gcc_commit(title, what, why, files_changed, next_step,\n'
        '             experiment={"id": "...", "hypothesis": "...",\n'
        '                         "metrics": {...}, "conclusion": "..."})\n\n'
        "WHEN NEEDED:\n"
        "  gcc_experiments()           — browse experiment history + metrics\n"
        "  gcc_branch(name, purpose)   — isolate a hypothesis\n"
        "  index_search(query)         — find code/config files\n"
        "  session_log_turn(assistant_message=\"...\") — only if real-time turn access needed"
    ),
    "academic": (
        "<!-- MANDATORY_CCR_ACTIONS -->\n"
        "Memory already loaded. Academic Research mode active.\n\n"
        "REQUIRED after each writing/analysis task:\n"
        "  gcc_commit(title, what, why, files_changed, next_step)\n\n"
        "WHEN NEEDED:\n"
        "  gcc_discuss(topic, position) — record argument or decision\n"
        "  gcc_discussions()            — retrieve past positions/notes\n"
        "  gcc_search(query)            — search all past analysis\n"
        "  session_search(query)        — search conversation history\n"
        "  session_log_turn(assistant_message=\"...\") — only if real-time turn access needed"
    ),
}


_CODEX_DIRECTIVES = {
    "default": (
        "Memory already loaded. Respond directly to the user.\n\n"
        "CCR SAVE POLICY FOR CODEX:\n"
        "  Do not call gcc_commit after routine reads, searches, or small edits.\n"
        "  Call gcc_commit only after a meaningful task milestone, before likely context loss, "
        "or when the user explicitly asks to save/update CCR memory.\n"
        "  Prefer one concise commit for a completed task; long sessions should rarely need "
        "more than 2-3 CCR commits.\n\n"
        "WHEN NEEDED:\n"
        "  gcc_context(level=3) — deeper context\n"
        "  gcc_search(query)    — find past decisions\n"
        "  ace_get_playbook()   — see evolved strategies\n"
        "  session_log_turn(assistant_message=\"...\") — only if you need real-time turn access"
    ),
    "ml": (
        "Memory already loaded. ML Research mode active.\n\n"
        "CCR SAVE POLICY FOR CODEX:\n"
        "  Do not call gcc_commit after routine reads, searches, or exploratory commands.\n"
        "  Call gcc_commit after a meaningful experiment result, implementation milestone, "
        "before likely context loss, or when the user explicitly asks to save/update CCR memory.\n"
        "  Prefer one concise commit per completed experiment/task; long sessions should rarely "
        "need more than 2-3 CCR commits.\n\n"
        "WHEN NEEDED:\n"
        "  gcc_experiments()           — browse experiment history + metrics\n"
        "  gcc_branch(name, purpose)   — isolate a hypothesis\n"
        "  index_search(query)         — find code/config files\n"
        "  session_log_turn(assistant_message=\"...\") — only if real-time turn access needed"
    ),
    "academic": (
        "Memory already loaded. Academic Research mode active.\n\n"
        "CCR SAVE POLICY FOR CODEX:\n"
        "  Do not call gcc_commit after routine reads, searches, or minor notes.\n"
        "  Call gcc_commit after a meaningful analysis/writing milestone, before likely context "
        "loss, or when the user explicitly asks to save/update CCR memory.\n"
        "  Prefer one concise commit per completed task; long sessions should rarely need "
        "more than 2-3 CCR commits.\n\n"
        "WHEN NEEDED:\n"
        "  gcc_discuss(topic, position) — record argument or decision\n"
        "  gcc_discussions()            — retrieve past positions/notes\n"
        "  gcc_search(query)            — search all past analysis\n"
        "  session_search(query)        — search conversation history\n"
        "  session_log_turn(assistant_message=\"...\") — only if real-time turn access needed"
    ),
}


def _is_codex_agent() -> bool:
    return os.environ.get("CCR_HOOK_AGENT", "").strip().lower() == "codex"


def _squash_line(text: str, max_chars: int = 220) -> str:
    """Collapse whitespace and clip long hook output for Codex visibility."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _context_section(context: str, heading: str) -> str:
    match = re.search(
        rf"(?ms)^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)",
        context or "",
    )
    return match.group(1).strip() if match else ""


def _first_recent_item(context: str) -> str:
    for pattern in (
        r"(?m)^-\s+\[[^\]]+\]\s+\([^)]+\)\s+(.+)$",
        r"(?m)^##\s+\[C\d+\].*?\|\s+branch:[^|]+\s+\|\s+(.+)$",
    ):
        match = re.search(pattern, context or "")
        if match:
            return match.group(1).strip()
    return ""


def _codex_session_summary(
    context: str,
    global_text: str = "",
    playbook_text: str = "",
    was_stale: bool = False,
) -> str:
    """Summarize full CCR context into the short stdout Codex exposes."""
    focus = _context_section(context, "Current Focus")
    focus = focus.splitlines()[0].strip() if focus else ""
    recent = _first_recent_item(context)
    bullet_count = len(re.findall(r"(?m)^\[[a-z]+-\d+\]", f"{global_text}\n{playbook_text}"))

    prefix = "CCR recovered from an unclean shutdown; " if was_stale else ""
    line1 = prefix + "CCR retrieved full memory"
    if bullet_count:
        line1 += f" and {bullet_count} playbook rule{'s' if bullet_count != 1 else ''}"
    if focus:
        line1 += f": {_squash_line(focus, 180)}"
    elif recent:
        line1 += f": {_squash_line(recent, 180)}"
    else:
        line1 += "."

    line2 = "Use gcc_context/gcc_search only for deeper recall; save with gcc_commit only after meaningful milestones or when asked."
    if focus and recent and recent not in focus:
        line2 = f"Recent: {_squash_line(recent, 120)}. " + line2
    return f"{line1}\n{line2}"


def _get_preset(ccr_root: str) -> str:
    """Read preset from .ccr/metadata.yaml. Returns 'default' if unset or on error."""
    try:
        import yaml  # noqa: PLC0415
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


def main():
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    # Parse canonical event payload from stdin (provider-agnostic)
    from ccr.hooks.canonical import HookPayload
    from ccr.hooks.codex import codex_ccr_config, resolve_project_root

    payload = HookPayload.from_stdin()
    project_root = resolve_project_root(payload.raw)
    fmt = payload.format
    user_prompt = payload.prompt or ""

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig

    config = codex_ccr_config() if _is_codex_agent() else CCRConfig()
    mem = MemoryManager(project_root, config)
    if not os.path.isdir(mem.ccr_root):
        if os.environ.get("CCR_AUTO_INIT", "").lower() in ("1", "true", "yes"):
            mem.ensure_structure()
        else:
            return

    first_prompt, was_stale = _is_first_prompt(mem.ccr_root)

    if first_prompt:
        from ccr.hooks.state_accumulator import initialize_state
        initialize_state(mem.ccr_root)
        _handle_session_start(mem, project_root, user_prompt, was_stale=was_stale, fmt=fmt)
    else:
        _handle_subsequent_prompt(mem, user_prompt, fmt=fmt)


def _handle_session_start(
    mem, project_root: str = "", user_prompt: str = "", was_stale: bool = False, fmt=None
):
    """First prompt of session: inject full context + strong directive."""
    from ccr.hooks.canonical import (
        format_context_block, format_playbook_block, format_directive,
        format_ready_message, ContextFormat
    )

    if fmt is None:
        fmt = ContextFormat.XML

    _create_db_session(mem.ccr_root, project_root or "")
    _buffer_user_prompt(mem.ccr_root, user_prompt)

    # Empty-project UX
    if not _project_has_commits(mem):
        if _is_codex_agent():
            print(
                "CCR active: no saved project memory yet. "
                "Save a concise gcc_commit after the first meaningful task."
            )
            return
        msg = (
            "CCR is active. This project has no memory yet — that's normal on first use.\n\n"
            "After finishing your first task, call:\n"
            '  gcc_commit(title="...", what="...", why="...", files_changed=[...], next_step="...")\n\n'
            "CCR will then remember your progress across all future sessions automatically.\n"
            "See: docs/quickstart-students.md for a 5-minute guide.\n"
            "Tip: use `ccr export-context` to export memory for claude.ai sessions."
        )
        print(format_ready_message(msg, fmt))
        return

    # Get context and playbooks
    context = mem.get_context(level=2)

    global_playbook_path = os.path.expanduser("~/.ccr/global_playbook.txt")
    global_text = ""
    if os.path.isfile(global_playbook_path):
        with open(global_playbook_path, "r", encoding="utf-8") as f:
            global_text = f.read().strip()

    playbook_path = os.path.join(mem.ccr_root, "playbook.txt")
    playbook_text = ""
    if os.path.isfile(playbook_path):
        with open(playbook_path, "r", encoding="utf-8") as f:
            playbook_text = f.read().strip()

    mem.log_ota(
        tool_name="session-start",
        observation="New CCR session",
        thought="Injecting memory context and playbook",
        action="Hook fired on session start",
    )

    # Build output parts using the agent's preferred format
    if _is_codex_agent():
        output = _codex_session_summary(
            context,
            global_text=global_text,
            playbook_text=playbook_text,
            was_stale=was_stale,
        )
        print(output)
        try:
            from ccr.hooks.state_accumulator import load_state, save_state
            state = load_state(mem.ccr_root)
            state.context_tokens = max(1, len(output) // 4)
            save_state(mem.ccr_root, state)
        except Exception:
            pass
        return

    parts = []
    if was_stale:
        parts.append(
            "⚡ CCR recovered from an unclean shutdown — memory context freshly injected. "
            "Everything is OK; no action needed. "
            "If this message appears every session, run `ccr doctor` to check for hook errors."
        )
    if context.strip():
        parts.append(format_context_block(context, tag="gcc_context", fmt=fmt))
    if global_text or playbook_text:
        parts.append(format_playbook_block(global_text, playbook_text, fmt=fmt))

    preset = _get_preset(mem.ccr_root)
    directives = _CODEX_DIRECTIVES if _is_codex_agent() else _DIRECTIVES
    directive = directives.get(preset, directives["default"])
    parts.append(format_directive(directive, fmt=fmt))

    if parts:
        output = "\n\n".join(parts)
        print(output)

        # Estimate tokens injected and save to session state
        try:
            from ccr.hooks.state_accumulator import load_state, save_state
            ctx_chars = len(output)
            ctx_tokens = max(1, ctx_chars // 4)
            state = load_state(mem.ccr_root)
            state.context_tokens = ctx_tokens
            save_state(mem.ccr_root, state)
        except Exception:
            pass


def _handle_subsequent_prompt(mem, user_prompt: str = "", fmt=None):
    """Subsequent prompts: light reminder to commit if progress was made."""
    from ccr.hooks.canonical import format_reminder, ContextFormat

    if fmt is None:
        fmt = ContextFormat.XML

    _buffer_user_prompt(mem.ccr_root, user_prompt)

    try:
        from ccr.hooks.state_accumulator import load_state
        state = load_state(mem.ccr_root)
        if state.tool_calls == 0:
            return
    except Exception:
        pass

    reminder = (
        "Remember: call gcc_commit after completing tasks. "
        "Call ace_update_counters after significant work."
    )
    print(format_reminder(reminder, fmt=fmt))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        _log_hook_error(traceback.format_exc())
