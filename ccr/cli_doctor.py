"""CCR doctor command — extracted from cli.py to satisfy 800-line limit."""
from __future__ import annotations

import logging
import os
import re
import shlex
import sys

import click

logger = logging.getLogger(__name__)


def _check_stale_hook_paths(hooks: dict) -> list[str]:
    """Inspect CCR hook commands for stale python/script paths after pip upgrade.

    For each CCR hook command of the form:
        CCR_PROJECT_ROOT=/path  python_exe  script_path  [extra args...]
    checks that python_exe (index 1) and script_path (index 2) both exist on disk.

    Returns a list of [WARN] message strings for any missing paths, or [] if all valid.
    """
    _CCR_HOOK_SCRIPTS = {
        "on_session_start.py",
        "on_tool_use.py",
        "on_stop.py",
        "on_compact.py",
    }
    warnings: list[str] = []

    for _event, cmds in hooks.items():
        for entry in cmds:
            if not isinstance(entry, dict):
                continue
            cmd = entry.get("command", "")
            # Only inspect CCR hook commands (contain a known hook script name)
            if not any(script in cmd for script in _CCR_HOOK_SCRIPTS):
                continue

            try:
                parts = shlex.split(cmd)
            except ValueError:
                parts = cmd.split()
            # Format: CCR_PROJECT_ROOT=... python_exe script_path [args...]
            if len(parts) < 3:
                continue

            python_exe = parts[1]
            hook_script = parts[2]

            missing: list[str] = []
            if not os.path.isfile(python_exe):
                missing.append(f"python: {python_exe} → missing")
            if not os.path.isfile(hook_script):
                missing.append(f"script: {hook_script} → missing")

            if missing:
                detail = ", ".join(missing)
                warnings.append(
                    f"[WARN] Hook path stale — re-run `ccr install` to update paths "
                    f"after pip upgrade\n         {detail}"
                )

    return warnings


def _run_doctor_checks(project: str) -> tuple[list[str], list[str], list[str]]:
    """Run all doctor checks and return (ok_items, issues, notices)."""
    import json
    import datetime

    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")
    issues: list[str] = []
    notices: list[str] = []
    ok_items: list[str] = []

    # 1. Python version
    py_ver = sys.version.split()[0]
    if sys.version_info >= (3, 11):
        ok_items.append(f"Python {py_ver}")
    else:
        issues.append(f"Python {py_ver} — requires 3.11+")

    # 1b. Detect installed agents
    from ccr.adapters import detect_installed, get_adapters
    installed_agents = detect_installed()
    all_adapters = get_adapters()
    if installed_agents:
        names = ", ".join(a.display_name for a in installed_agents)
        ok_items.append(f"Agents detected: {names}")
    else:
        agent_names = ", ".join(a.display_name for a in all_adapters)
        notices.append(
            f"No supported agents detected. Supported: {agent_names}. "
            "Install at least one agent for CCR to integrate with."
        )

    # 2. .ccr/ directory
    if os.path.isdir(ccr_dir):
        ok_items.append(f".ccr/ exists at {ccr_dir}")
    else:
        issues.append(".ccr/ not found — run 'ccr init'")

    # 3. .mcp.json configuration
    mcp_json = os.path.join(project, ".mcp.json")
    if os.path.isfile(mcp_json):
        ok_items.append(".mcp.json configured")
    else:
        issues.append(".mcp.json not found — CCR MCP server not registered")

    # 4. ONNX semantic search (optional feature — demoted to notice)
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        import numpy  # noqa: F401
        ok_items.append("Semantic search available (onnxruntime + tokenizers)")
    except ImportError:
        notices.append("Semantic search not installed (optional) — pip install 'ccr-memory[semantic]'")

    # 5. ONNX model cached (optional — notice only)
    model_dir = os.path.expanduser("~/.cache/ccr/models/all-MiniLM-L6-v2")
    if os.path.isfile(os.path.join(model_dir, "model.onnx")):
        ok_items.append("ONNX model cached (~90 MB)")
    else:
        notices.append("ONNX model not cached — will download on first semantic search use")

    # 6. sqlite-vec (optional feature — demoted to notice)
    try:
        import sqlite_vec  # noqa: F401
        ok_items.append("Vector store available (sqlite-vec)")
    except ImportError:
        notices.append("Vector store not installed (optional) — pip install 'ccr-memory[vector]'")

    # 7. Adapter status checks — per-agent CCR configuration
    any_hooks_found = False
    for adapter in all_adapters:
        status = adapter.status()
        if not status.installed:
            continue  # Skip agents not installed
        if status.ccr_enabled:
            level_name = {5: "MCP+Hooks", 4: "MCP", 3: "FileWatcher", 2: "SDK", 1: "Manual"}.get(
                status.integration_level, "Unknown"
            )
            ok_items.append(f"{adapter.display_name}: CCR enabled ({level_name})")
            if status.hooks_configured:
                any_hooks_found = True
        else:
            notices.append(
                f"{adapter.display_name} installed but CCR not configured — "
                f"run 'ccr install-global --agents {adapter.name}'"
            )

    # Session logging summary
    if any_hooks_found:
        ok_items.append("Session logging enabled (Q&A turns saved to sessions.db)")
    else:
        if installed_agents:
            issues.append(
                "No auto-commit hooks found — run 'ccr install-global' to enable automatic memory"
            )

    # 8. Hook error log
    hook_errors_log = os.path.join(ccr_dir, ".hook_errors.log")
    if os.path.isfile(hook_errors_log):
        try:
            mtime = os.path.getmtime(hook_errors_log)
            age_h = (datetime.datetime.now().timestamp() - mtime) / 3600
            with open(hook_errors_log, encoding="utf-8") as f:
                content = f.read()
            n_errors = content.count("\n---")
            if age_h < 2:
                issues.append(
                    f"Recent hook errors logged ({n_errors} total) — "
                    f"run: tail -60 {hook_errors_log}"
                )
            elif age_h < 48:
                notices.append(
                    f"Hook errors logged ({n_errors} total, last {age_h:.0f}h ago) — "
                    f"run: tail -60 {hook_errors_log}"
                )
            else:
                ok_items.append(f"Hook errors: {n_errors} old (last {age_h:.0f}h ago)")
        except (OSError, UnicodeDecodeError):
            ok_items.append("Hook error log exists but unreadable")
    else:
        ok_items.append("No hook errors logged")

    # 9. Disk usage
    if os.path.isdir(ccr_dir):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(ccr_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
        size_kb = total_size / 1024
        ok_items.append(f".ccr/ disk usage: {size_kb:.0f} KB")

    # 10. Version parity: installed package vs __init__.__version__
    try:
        import importlib.metadata
        installed_ver = importlib.metadata.version("ccr-memory")
        from ccr import __version__ as src_ver
        if installed_ver == src_ver:
            ok_items.append(f"Version parity: {src_ver}")
        else:
            issues.append(
                f"Version skew: installed={installed_ver}, source={src_ver} "
                "— run: pip install -e . (dev) or pip install --upgrade ccr-memory (user)"
            )
    except Exception:
        notices.append("Could not verify version parity (ccr-memory not installed via pip)")

    # 11. Stale .session_active marker (from force-killed session)
    if os.path.isdir(ccr_dir):
        session_active = os.path.join(ccr_dir, ".session_active")
        if os.path.isfile(session_active):
            # Check if the PID inside it is still alive
            try:
                with open(session_active, "r") as f:
                    stored_pid = int(f.read().strip())
                os.kill(stored_pid, 0)  # Raises if process is dead
                ok_items.append("Session active (current session running)")
            except (ValueError, OSError):
                # Process is dead — stale marker
                import time as _time
                try:
                    age_h = (_time.time() - os.path.getmtime(session_active)) / 3600
                    notices.append(
                        f"Stale .session_active marker found (process dead, "
                        f"age {age_h:.1f}h) — it will be auto-cleaned on next "
                        f"session start, or delete it manually: "
                        f"rm {session_active}"
                    )
                except OSError:
                    notices.append(
                        "Stale .session_active marker found — delete it: "
                        f"rm {session_active}"
                    )

    return ok_items, issues, notices


def _fix_stale_hooks(project: str) -> list[str]:
    """Re-run ccr install to fix stale hook paths."""
    fixed: list[str] = []
    try:
        from ccr.cli import install  # noqa: PLC0415
        ctx = click.Context(install, info_name="install")
        ctx.params = {"project": project, "preset": "default"}
        ctx.invoke(install, project=project, preset="default")
        fixed.append("Re-ran ccr install to refresh hook paths")
    except Exception as exc:
        fixed.append(f"Failed to fix hooks: {exc}")
    return fixed


def _fix_stale_session_marker(ccr_dir: str) -> list[str]:
    """Remove stale .session_active marker if process is dead."""
    fixed: list[str] = []
    marker = os.path.join(ccr_dir, ".session_active")
    if not os.path.isfile(marker):
        return fixed
    try:
        with open(marker, "r") as f:
            stored_pid = int(f.read().strip())
        os.kill(stored_pid, 0)
    except (ValueError, OSError):
        os.remove(marker)
        fixed.append("Removed stale .session_active marker")
    return fixed


def _fix_sqlite_integrity(ccr_dir: str) -> list[str]:
    """Run PRAGMA integrity_check on memory.db and rebuild FTS if needed."""
    import sqlite3

    fixed: list[str] = []
    db_path = os.path.join(ccr_dir, "memory.db")
    if not os.path.isfile(db_path):
        return fixed

    try:
        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result and result[0] == "ok":
            fixed.append("memory.db integrity check passed")
        else:
            fixed.append(f"memory.db integrity check: {result[0] if result else 'unknown'}")
        conn.close()
    except sqlite3.DatabaseError as exc:
        fixed.append(f"memory.db integrity check failed: {exc}")

    index_db = os.path.join(ccr_dir, "index.db")
    if os.path.isfile(index_db):
        try:
            conn = sqlite3.connect(index_db)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%'",
            ).fetchall()]
            for table in tables:
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
                    continue
                try:
                    conn.execute(f"INSERT INTO {table}({table}, rank) VALUES('rebuild', 1)")
                    conn.commit()
                    fixed.append(f"Rebuilt FTS index: {table}")
                except sqlite3.OperationalError:
                    pass
            conn.close()
        except sqlite3.DatabaseError as exc:
            fixed.append(f"index.db FTS rebuild failed: {exc}")

    return fixed


def _fix_duplicate_hooks(project: str) -> list[str]:
    """Deduplicate hook commands in settings.local.json."""
    import json

    fixed: list[str] = []
    settings_path = os.path.join(project, ".claude", "settings.local.json")
    if not os.path.isfile(settings_path):
        return fixed

    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
    except (json.JSONDecodeError, OSError):
        return fixed

    hooks = settings.get("hooks", {})
    deduped = False
    for event in list(hooks.keys()):
        cmds = hooks[event]
        seen: set[str] = set()
        unique: list[dict] = []
        for c in cmds:
            cmd_str = c.get("command", "")
            if cmd_str not in seen:
                seen.add(cmd_str)
                unique.append(c)
            else:
                deduped = True
        hooks[event] = unique

    if deduped:
        settings["hooks"] = hooks
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        fixed.append("Removed duplicate hook commands")

    return fixed


@click.command()
@click.argument("project", default=".")
@click.option("--fix", is_flag=True, help="Auto-fix detected issues (stale hooks, corrupt DB, etc.)")
def doctor(project: str, fix: bool) -> None:
    """Diagnose CCR health and configuration issues.

    Use --fix to auto-repair: stale hook paths, duplicate hooks,
    stale session markers, and SQLite integrity/FTS issues.
    """
    ok_items, issues, notices = _run_doctor_checks(project)

    click.echo("CCR Doctor\n")
    for item in ok_items:
        click.echo(f"  [OK] {item}")
    for item in issues:
        click.echo(f"  [!!] {item}")
    for item in notices:
        if item.startswith("[WARN]"):
            click.echo(f"  {item}")
        else:
            click.echo(f"  [--] {item}")
    click.echo(f"\n  {len(ok_items)} OK, {len(issues)} issue(s), {len(notices)} notice(s)")

    if not fix:
        if issues:
            click.echo("\n  Run 'ccr doctor --fix' to attempt auto-repair.")
        else:
            click.echo("  All checks passed.")
        return

    click.echo("\n=== Auto-fix ===\n")
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")
    all_fixes: list[str] = []

    has_stale_hooks = any("stale" in i.lower() or "hook path" in i.lower() for i in issues)
    has_duplicate_hooks = any("duplicate hook" in i.lower() for i in issues)
    has_stale_marker = any("session_active" in n.lower() for n in notices)

    if has_duplicate_hooks:
        all_fixes.extend(_fix_duplicate_hooks(project))
    if has_stale_hooks:
        all_fixes.extend(_fix_stale_hooks(project))
    if has_stale_marker:
        all_fixes.extend(_fix_stale_session_marker(ccr_dir))
    if os.path.isdir(ccr_dir):
        all_fixes.extend(_fix_sqlite_integrity(ccr_dir))

    if all_fixes:
        for f_msg in all_fixes:
            click.echo(f"  [FIX] {f_msg}")
        click.echo(f"\n  {len(all_fixes)} fix(es) applied. Run 'ccr doctor' again to verify.")
    else:
        click.echo("  No auto-fixable issues found.")
