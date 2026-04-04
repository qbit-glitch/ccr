#!/usr/bin/env python3
"""Claude Code hook: fires on Stop (session end).

Reads accumulated session state and auto-commits if meaningful
progress was detected. Replaces the bare-bones on_session_end.py.

Prints a brief confirmation to stdout so Claude Code sees the commit.
"""

import os
import sys


def main() -> None:
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig
    from ccr.hooks.state_accumulator import clear_state, load_state

    mem = MemoryManager(project_root, CCRConfig())
    if not os.path.isdir(mem.ccr_root):
        return

    state = load_state(mem.ccr_root)

    if state.is_meaningful():
        fields = state.to_commit_fields()
        try:
            result = mem.commit(
                title=fields["title"],
                what=fields["what"],
                why=fields["why"],
                files_changed=fields["files_changed"],
                next_step=fields["next_step"],
                patterns_learned=fields["patterns_learned"],
            )
            print(f"[CCR] Auto-committed: {fields['title']}")
        except Exception as exc:
            # Don't crash the hook on commit failure
            print(f"[CCR] Auto-commit failed: {exc}")
    else:
        # Log session end even without auto-commit
        mem.log_ota(
            tool_name="session-end",
            observation="Claude Code session ending",
            thought="No meaningful progress to auto-commit",
            action="Session ended cleanly",
        )

    # Clean up session state
    clear_state(mem.ccr_root)


if __name__ == "__main__":
    main()
