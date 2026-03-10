#!/usr/bin/env python3
"""Claude Code hook: fires on Stop / session end.

Auto-commits any uncommitted progress and logs session end to OTA.
"""

import os
import sys


def main():
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig

    mem = MemoryManager(project_root, CCRConfig())
    if not os.path.isdir(mem.ccr_root):
        return

    mem.log_ota(
        tool_name="session-end",
        observation="Claude Code session ending",
        thought="Persisting final state",
        action="Session ended cleanly",
    )


if __name__ == "__main__":
    main()
