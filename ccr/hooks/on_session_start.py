#!/usr/bin/env python3
"""Claude Code hook: fires on UserPromptSubmit.

Injects ACE playbook + level-1 memory context into the conversation
by printing to stdout (Claude Code captures hook stdout as context).

On first prompt of a session, outputs a strong directive to use CCR tools.
On subsequent prompts, outputs a lighter reminder to commit if needed.
"""

import os
import sys


# Track whether this is the first invocation in this process lifetime.
# Hooks are invoked as separate processes each time, so we use a marker file.
_SESSION_MARKER = "/tmp/.ccr_session_{pid}.marker"


def _is_first_prompt(ccr_root: str) -> bool:
    """Check if this is the first prompt of the session using a marker file (M6: atomic create)."""
    marker = os.path.join(ccr_root, ".session_active")
    try:
        # M6: Atomic create — O_CREAT|O_EXCL fails if file already exists
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError:
        # Fallback: if directory doesn't exist or permissions issue
        return False


def main():
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig

    mem = MemoryManager(project_root, CCRConfig())
    if not os.path.isdir(mem.ccr_root):
        return  # No .ccr/ directory yet — skip

    first_prompt = _is_first_prompt(mem.ccr_root)

    if first_prompt:
        _handle_session_start(mem)
    else:
        _handle_subsequent_prompt(mem)


def _handle_session_start(mem):
    """First prompt of session: inject full context + strong directive."""
    # Get level-1 context
    context = mem.get_context(level=1)

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
    if context.strip():
        parts.append(f"<gcc_context>\n{context}\n</gcc_context>")
    if global_text or playbook_text:
        pb_parts = []
        if global_text:
            pb_parts.append(f"# GLOBAL STRATEGIES (all projects)\n{global_text}")
        if playbook_text:
            pb_parts.append(f"# PROJECT STRATEGIES (this project)\n{playbook_text}")
        parts.append(f"<ace_playbook>\n{chr(10).join(pb_parts)}\n</ace_playbook>")

    # Strong session-start directive
    parts.append("""<MANDATORY_CCR_ACTIONS>
You have 18 CCR MCP tools (gcc_*, ace_*, rlm_*, index_*) available RIGHT NOW.

IMMEDIATE ACTION REQUIRED — before responding to the user's message:
1. Call gcc_context(level=2) to load full project memory
2. Call gcc_status to check memory state

ONGOING REQUIREMENTS for this entire session:
- After completing ANY task: call gcc_commit with what/why/files_changed/next_step
- After significant work: call ace_get_playbook, then ace_update_counters
- If you learn something new: call ace_apply_delta with ADD
- Before context gets large: call gcc_commit to preserve state
- Do NOT wait for the user to ask — commit proactively after each milestone

These are MCP tools you call directly, not theoretical concepts. Use them NOW.
</MANDATORY_CCR_ACTIONS>""")

    if parts:
        print("\n\n".join(parts))


def _handle_subsequent_prompt(mem):
    """Subsequent prompts: light reminder to commit if progress was made."""
    print("""<ccr_reminder>
Remember: call gcc_commit after completing tasks. Call ace_update_counters after significant work.
</ccr_reminder>""")


if __name__ == "__main__":
    main()
