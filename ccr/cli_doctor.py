"""CCR doctor command — extracted from cli.py to satisfy 800-line limit."""
from __future__ import annotations

import os
import sys

import click


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

    # 7. Hooks configured
    settings_path = os.path.join(project, ".claude", "settings.local.json")
    if os.path.isfile(settings_path):
        try:
            with open(settings_path) as f:
                settings = json.loads(f.read())
            hooks = settings.get("hooks", {})
            if "Stop" in hooks or "UserPromptSubmit" in hooks:
                ok_items.append("Auto-commit hooks configured")
            else:
                issues.append("Hooks not configured — run 'ccr install'")
            # Check for duplicate hook commands (from running ccr install twice before fix)
            for event, cmds in hooks.items():
                commands = [c.get("command", "") for c in cmds if isinstance(c, dict)]
                if len(commands) != len(set(commands)):
                    issues.append(
                        f"Duplicate hook command in {event} — run: ccr uninstall && ccr install"
                    )
        except (json.JSONDecodeError, OSError):
            issues.append("settings.local.json exists but unreadable")
    else:
        issues.append("No hooks configured — run 'ccr install' for auto-commit")

    # 8. Hook error log
    hook_errors_log = os.path.join(ccr_dir, ".hook_errors.log")
    if os.path.isfile(hook_errors_log):
        try:
            mtime = os.path.getmtime(hook_errors_log)
            age_h = (datetime.datetime.now().timestamp() - mtime) / 3600
            with open(hook_errors_log, encoding="utf-8") as f:
                content = f.read()
            n_errors = content.count("\n---")
            if age_h < 24:
                issues.append(
                    f"Recent hook errors logged ({n_errors} total) — "
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

    return ok_items, issues, notices


@click.command()
@click.argument("project", default=".")
def doctor(project: str) -> None:
    """Diagnose CCR health and configuration issues."""
    ok_items, issues, notices = _run_doctor_checks(project)

    # Print results
    click.echo("CCR Doctor\n")
    for item in ok_items:
        click.echo(f"  [OK] {item}")
    for item in issues:
        click.echo(f"  [!!] {item}")
    for item in notices:
        click.echo(f"  [--] {item}")
    click.echo(f"\n  {len(ok_items)} OK, {len(issues)} issue(s), {len(notices)} notice(s)")
    if not issues:
        click.echo("  All checks passed.")
