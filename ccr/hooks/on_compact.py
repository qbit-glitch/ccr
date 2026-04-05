#!/usr/bin/env python3
"""Claude Code hook: fires on PreCompact (context approaching limit).

Reminds Claude Code to commit progress before context is compacted,
preventing loss of reasoning state.
"""

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
                f.write(f"\n--- {ts} [on_compact] ---\n{error_text}\n")
    except Exception:
        pass


def main():
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig

    mem = MemoryManager(project_root, CCRConfig())
    if not os.path.isdir(mem.ccr_root):
        return

    mem.log_ota(
        tool_name="pre-compact",
        observation="Context approaching compaction threshold",
        thought="Should commit progress before compaction",
        action="Triggered pre-compact reminder",
    )

    print(
        "REMINDER: Context is about to be compacted. "
        "Use gcc_commit to save your progress before state is lost."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        _log_hook_error(traceback.format_exc())
