"""CCR CLI — entry point for the ccr command."""

from __future__ import annotations

import logging
import os
import signal
import sys

import click
import yaml

from ccr.core.types import CCREngineConfig, CCRConfig, RouterConfig


def _load_config(config_path: str | None, overrides: dict) -> CCREngineConfig:
    """Load config from YAML file with CLI overrides."""
    base = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}

    # Build config from defaults + file + overrides
    config = CCREngineConfig()

    # API key from env or config
    config.anthropic_api_key = (
        overrides.get("anthropic_api_key")
        or os.environ.get("ANTHROPIC_API_KEY")
        or base.get("models", {}).get("primary", {}).get("api_key", "")
    )

    if overrides.get("sub_model"):
        config.sub_model = overrides["sub_model"]
    elif "models" in base and "sub" in base["models"]:
        config.sub_model = base["models"]["sub"].get("model", config.sub_model)

    if overrides.get("sub_model_url"):
        config.sub_model_base_url = overrides["sub_model_url"]
    elif "models" in base and "sub" in base["models"]:
        config.sub_model_base_url = base["models"]["sub"].get(
            "base_url", config.sub_model_base_url
        )

    config.sub_model_api_key = os.environ.get("CCR_SUB_MODEL_API_KEY") or base.get(
        "models", {}
    ).get("sub", {}).get("api_key")

    if overrides.get("port"):
        config.gateway_port = overrides["port"]
    elif "gateway" in base:
        config.gateway_port = base["gateway"].get("port", config.gateway_port)

    if "context" in base:
        config.pack_token_budget = base["context"].get(
            "token_budget", config.pack_token_budget
        )

    return config


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """CCR — Claude Context Reducer. Token-efficient middleware for Claude Code."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--project", "-p", default=".", help="Project root directory")
@click.option("--port", default=7447, type=int, help="Proxy port")
@click.option("--sub-model", default=None, help="Sub-model for cheap calls")
@click.option("--sub-model-url", default=None, help="Sub-model API URL")
@click.option("--config", "config_path", default=None, help="Path to config YAML")
def start(
    project: str,
    port: int,
    sub_model: str | None,
    sub_model_url: str | None,
    config_path: str | None,
) -> None:
    """[DEPRECATED — use 'ccr install' for Claude Code] Start the CCR gateway proxy."""
    click.echo(
        "WARNING: 'ccr start' runs the legacy HTTP proxy architecture (requires ANTHROPIC_API_KEY).\n"
        "   For Claude Code users, use 'ccr install' instead — it sets up MCP + hooks with no API key.\n"
        "   See: https://github.com/qbit-glitch/ccr#quick-start\n",
        err=True,
    )

    from ccr.core.engine import CCREngine
    from ccr.gateway import CCRGateway

    project = os.path.abspath(project)

    # Try default config location
    if not config_path:
        default_cfg = os.path.join(project, ".ccr", "config.yaml")
        if os.path.isfile(default_cfg):
            config_path = default_cfg
        else:
            default_cfg = os.path.join(
                os.path.dirname(__file__), "..", "config", "default.yaml"
            )
            if os.path.isfile(default_cfg):
                config_path = default_cfg

    config = _load_config(
        config_path,
        {
            "port": port,
            "sub_model": sub_model,
            "sub_model_url": sub_model_url,
        },
    )

    if not config.anthropic_api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set. Set it as environment variable.", err=True)
        sys.exit(1)

    engine = CCREngine(project, config)
    gateway = CCRGateway(engine, port=config.gateway_port)

    # Handle graceful shutdown
    def _shutdown(sig, frame):
        click.echo("\nShutting down CCR...")
        gateway.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    base_url = gateway.start()

    click.echo(f"""
╔══════════════════════════════════════════════╗
║         CCR — Claude Context Reducer         ║
╠══════════════════════════════════════════════╣
║  Gateway:  {base_url:<33}║
║  Project:  {project[:33]:<33}║
║  Model:    {config.claude_model[:33]:<33}║
║  Sub-model: {config.sub_model[:32]:<32}║
╠══════════════════════════════════════════════╣
║  Set this before running Claude Code:        ║
║                                              ║
║  export ANTHROPIC_BASE_URL={base_url:<17}║
║                                              ║
║  Press Ctrl+C to stop                        ║
╚══════════════════════════════════════════════╝
""")

    # Block until shutdown
    try:
        signal.pause()
    except AttributeError:
        # Windows doesn't have signal.pause
        import time
        while True:
            time.sleep(1)


@cli.command()
@click.argument("project", default=".")
def init(project: str) -> None:
    """Initialize .ccr/ in a project directory."""
    from ccr.core.memory import MemoryManager

    project = os.path.abspath(project)
    mem = MemoryManager(project)
    created = mem.ensure_structure()
    _register_project(project)
    if created:
        click.echo(f"Initialized .ccr/ in {project}")
    else:
        click.echo(f".ccr/ already exists in {project}")


@cli.command()
@click.argument("project", default=".")
@click.option("--level", "-l", default=2, type=int, help="Context detail level (1-5)")
def context(project: str, level: int) -> None:
    """Print current project context at a given detail level."""
    from ccr.core.memory import MemoryManager

    project = os.path.abspath(project)
    mem = MemoryManager(project)
    if not os.path.isdir(os.path.join(project, ".ccr")):
        click.echo("No .ccr/ found. Run 'ccr init' first.", err=True)
        sys.exit(1)
    click.echo(mem.get_context(level=level))


@cli.command()
@click.argument("project", default=".")
def index(project: str) -> None:
    """Build or rebuild the repo index."""
    from ccr.context.indexer import RepoIndex
    from ccr.core.memory import MemoryManager

    project = os.path.abspath(project)
    click.echo(f"Indexing {project}...")
    idx = RepoIndex.build(project)
    click.echo(idx.get_summary())

    mem = MemoryManager(project)
    mem.ensure_structure()
    mem.save_index(idx.to_json())
    click.echo(f"Index saved to .ccr/index.json ({len(idx.files)} files)")


@cli.command()
@click.argument("project", default=".")
def install(project: str) -> None:
    """Install CCR hooks into a project for automatic memory management.

    Generates .claude/settings.local.json hook configuration so Claude Code
    auto-commits session progress without manual gcc_commit calls.
    Also initializes .ccr/ if not present.
    """
    import json

    from ccr.core.memory import MemoryManager

    project = os.path.abspath(project)

    # Ensure .ccr/ exists
    mem = MemoryManager(project)
    mem.ensure_structure()

    # Find the CCR package root for hook paths
    ccr_pkg = os.path.dirname(os.path.abspath(__file__))
    hooks_dir = os.path.join(ccr_pkg, "hooks")

    # Determine python executable (prefer venv)
    python_exe = sys.executable

    # Build hook commands (all 4 hooks required for full auto-commit chain)
    hook_config = {
        "UserPromptSubmit": [{
            "type": "command",
            "command": f"{python_exe} {os.path.join(hooks_dir, 'on_session_start.py')}",
        }],
        "PostToolUse": [{
            "type": "command",
            "command": f"{python_exe} {os.path.join(hooks_dir, 'on_tool_use.py')}",
        }],
        "Stop": [{
            "type": "command",
            "command": f"{python_exe} {os.path.join(hooks_dir, 'on_stop.py')}",
        }],
        "PreCompact": [{
            "type": "command",
            "command": f"{python_exe} {os.path.join(hooks_dir, 'on_compact.py')}",
        }],
    }

    # Write to .claude/settings.local.json (merge with existing)
    claude_dir = os.path.join(project, ".claude")
    os.makedirs(claude_dir, exist_ok=True)
    settings_path = os.path.join(claude_dir, "settings.local.json")

    existing = {}
    if os.path.isfile(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                existing = json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            pass

    existing.setdefault("hooks", {})
    existing["hooks"].update(hook_config)

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")

    # Write .mcp.json — registers the CCR MCP server with Claude Code
    mcp_json_path = os.path.join(project, ".mcp.json")
    existing_mcp: dict = {}
    if os.path.isfile(mcp_json_path):
        try:
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                existing_mcp = json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            pass

    existing_mcp.setdefault("mcpServers", {})
    existing_mcp["mcpServers"]["ccr"] = {
        "command": python_exe,
        "args": ["-m", "ccr.mcp_server", "--project", project],
    }

    with open(mcp_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_mcp, f, indent=2)
        f.write("\n")

    click.echo(f"\n\u2705 CCR installed in {project}")
    click.echo(f"\n  Hooks ({claude_dir}/settings.local.json):")
    click.echo(f"    UserPromptSubmit \u2192 injects memory context at session start")
    click.echo(f"    PostToolUse      \u2192 tracks tool calls for auto-commit")
    click.echo(f"    Stop             \u2192 auto-commits session progress on exit")
    click.echo(f"    PreCompact       \u2192 saves state before context resets")
    click.echo(f"\n  MCP server ({mcp_json_path}):")
    click.echo(f"    ccr \u2192 {python_exe} -m ccr.mcp_server")
    abs_project = os.path.abspath(project)
    click.echo(f"\n=== Next step ===")
    click.echo(f"  Open Claude Code from this exact directory:")
    click.echo(f"    cd {abs_project} && claude")
    click.echo(f"")
    click.echo(f"  \u26a0\ufe0f  Claude Code must be launched from {abs_project}")
    click.echo(f"     Hooks only fire when CWD matches the project root.")
    click.echo(f"")
    click.echo(f"  Quick test — ask Claude:")
    click.echo(f'    "What do you remember about this project?"')
    click.echo(f"    Then call: gcc_status()")


@cli.command()
@click.argument("project", default=".")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def uninstall(project: str, yes: bool) -> None:
    """Remove CCR hooks and MCP registration. Memory (.ccr/) is preserved."""
    import json as _json

    project = os.path.abspath(project)
    claude_dir = os.path.join(project, ".claude")   # matches install's write path
    settings_path = os.path.join(claude_dir, "settings.local.json")
    mcp_json_path = os.path.join(project, ".mcp.json")

    changes = []

    # Check what will be removed
    hooks_to_remove = []
    if os.path.isfile(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                existing = _json.loads(f.read())
            for event, cmds in existing.get("hooks", {}).items():
                kept = [c for c in cmds if "ccr" not in c.get("command", "")]
                removed = [c for c in cmds if "ccr" in c.get("command", "")]
                if removed:
                    hooks_to_remove.extend([(event, c) for c in removed])
        except (OSError, _json.JSONDecodeError):
            pass

    mcp_entry_exists = False
    if os.path.isfile(mcp_json_path):
        try:
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                existing_mcp = _json.loads(f.read())
            mcp_entry_exists = "ccr" in existing_mcp.get("mcpServers", {})
        except (OSError, _json.JSONDecodeError):
            pass

    if not hooks_to_remove and not mcp_entry_exists:
        click.echo("CCR is not installed in this project — nothing to remove.")
        return

    # Show what will be removed
    click.echo(f"Will remove from {project}:")
    for event, cmd in hooks_to_remove:
        click.echo(f"  Hook [{event}]: {cmd.get('command', '')}")
    if mcp_entry_exists:
        click.echo(f"  MCP server: ccr entry from {mcp_json_path}")
    click.echo(f"  Memory (.ccr/) will NOT be touched.")

    if not yes:
        click.confirm("Proceed?", abort=True)

    # Remove CCR hooks from settings.local.json
    if hooks_to_remove and os.path.isfile(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                existing = _json.loads(f.read())
            for event in list(existing.get("hooks", {}).keys()):
                existing["hooks"][event] = [
                    c for c in existing["hooks"][event]
                    if "ccr" not in c.get("command", "")
                ]
                if not existing["hooks"][event]:
                    del existing["hooks"][event]
            if not existing.get("hooks"):
                existing.pop("hooks", None)
            with open(settings_path, "w", encoding="utf-8") as f:
                _json.dump(existing, f, indent=2)
                f.write("\n")
            click.echo(f"✓ Removed {len(hooks_to_remove)} hook(s) from {settings_path}")
        except (OSError, _json.JSONDecodeError) as exc:
            click.echo(f"⚠️  Failed to update hooks: {exc}", err=True)

    # Remove ccr entry from .mcp.json
    if mcp_entry_exists and os.path.isfile(mcp_json_path):
        try:
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                existing_mcp = _json.loads(f.read())
            existing_mcp.get("mcpServers", {}).pop("ccr", None)
            with open(mcp_json_path, "w", encoding="utf-8") as f:
                _json.dump(existing_mcp, f, indent=2)
                f.write("\n")
            click.echo(f"✓ Removed ccr MCP server from {mcp_json_path}")
        except (OSError, _json.JSONDecodeError) as exc:
            click.echo(f"⚠️  Failed to update .mcp.json: {exc}", err=True)

    click.echo("\nCCR uninstalled. Your .ccr/ memory is intact.")
    click.echo("To reinstall: ccr install")


@cli.command()
@click.argument("project", default=".")
@click.option("--days", "-d", default=90, type=int,
              help="Archive commits older than N days (default 90).")
@click.option("--dry-run", is_flag=True,
              help="Show what would be archived without making changes.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def clean(project: str, days: int, dry_run: bool, yes: bool) -> None:
    """Prune old commits to .ccr/archive/ (rolling summary preserved)."""
    import re
    from datetime import datetime, timedelta, timezone

    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")

    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ directory found in {project}. Run 'ccr init' first.", err=True)
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ccr.core.memory import MemoryManager
    from ccr.core.types import CCRConfig

    mem = MemoryManager(project, CCRConfig())
    branch = mem.get_active_branch()

    commits_path = os.path.join(mem.ccr_root, "branches", branch, "commits.md")
    if not os.path.isfile(commits_path):
        click.echo("No commits to clean.")
        return

    with open(commits_path, "r", encoding="utf-8") as f:
        content = f.read()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    commit_pattern = re.compile(
        r"(## \[C\d{3,}\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}).*?)(?=## \[C\d{3,}\]|\Z)",
        re.DOTALL,
    )

    to_archive = []
    to_keep = []
    for m in commit_pattern.finditer(content):
        block = m.group(1).strip()
        date_str = m.group(2)
        try:
            commit_dt = datetime.fromisoformat(date_str.replace(" ", "T") + ":00+00:00")
            if commit_dt < cutoff:
                to_archive.append(block)
            else:
                to_keep.append(block)
        except ValueError:
            to_keep.append(block)  # Keep if date unparseable

    if not to_archive:
        click.echo(f"No commits older than {days} days found — nothing to archive.")
        return

    click.echo(f"{'[dry-run] ' if dry_run else ''}Found {len(to_archive)} commit(s) older than {days} days to archive, {len(to_keep)} to keep.")

    if dry_run:
        click.echo("\nCommits that would be archived:")
        for block in to_archive:
            first_line = block.splitlines()[0]
            click.echo(f"  {first_line}")
        return

    if not yes:
        click.confirm(f"Archive {len(to_archive)} commit(s)? (Rolling summary is preserved)", abort=True)

    # Write archive file
    archive_dir = os.path.join(ccr_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_path = os.path.join(archive_dir, f"{today}_{branch}.md")
    with open(archive_path, "a", encoding="utf-8") as f:
        f.write(f"# Archived {today} — branch: {branch} (commits older than {days} days)\n\n")
        for block in to_archive:
            f.write(block + "\n\n---\n\n")

    # Rewrite commits.md keeping rolling summary + non-archived commits
    summary_match = re.search(r"^## Rolling Summary.*?(?=## \[C\d{3,}\]|\Z)", content, re.DOTALL | re.MULTILINE)
    summary_section = summary_match.group(0) if summary_match else ""

    new_content = summary_section
    if to_keep:
        new_content += "\n\n".join(to_keep) + "\n\n---\n\n"

    with open(commits_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    click.echo(f"✓ Archived {len(to_archive)} commit(s) → {archive_path}")
    click.echo(f"✓ {len(to_keep)} commit(s) retained in {commits_path}")
    click.echo("  Rolling summary preserved.")


@cli.command()
@click.argument("project", default=".")
def status(project: str) -> None:
    """Show project CCR status."""
    from ccr.core.memory import MemoryManager

    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")

    if not os.path.isdir(ccr_dir):
        click.echo("No .ccr/ found. Run 'ccr init' first.", err=True)
        sys.exit(1)

    mem = MemoryManager(project)
    branch = mem.get_active_branch()

    click.echo(f"Project: {project}")
    click.echo(f"Active branch: {branch}")

    # Show recent commits
    recent = mem._read_recent_commits(branch, 3)
    if recent:
        click.echo(f"\nRecent commits ({branch}):")
        click.echo(recent)
    else:
        click.echo("\nNo commits yet.")


@cli.command()
@click.argument("project", default=".")
def doctor(project: str) -> None:
    """Diagnose CCR health and configuration issues."""
    import importlib

    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")
    issues = []
    ok_items = []

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

    # 4. ONNX semantic search
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        import numpy  # noqa: F401
        ok_items.append("Semantic search available (onnxruntime + tokenizers)")
    except ImportError:
        issues.append("Semantic search unavailable — install ccr[semantic]")

    # 5. ONNX model cached
    model_dir = os.path.expanduser("~/.cache/ccr/models/all-MiniLM-L6-v2")
    if os.path.isfile(os.path.join(model_dir, "model.onnx")):
        ok_items.append("ONNX model cached (~90 MB)")
    else:
        issues.append("ONNX model not cached — will download on first use")

    # 6. sqlite-vec
    try:
        import sqlite_vec  # noqa: F401
        ok_items.append("Vector store available (sqlite-vec)")
    except ImportError:
        issues.append("Vector store unavailable — install ccr[vector] (optional)")

    # 7. Hooks configured
    settings_path = os.path.join(project, ".claude", "settings.local.json")
    if os.path.isfile(settings_path):
        try:
            import json
            with open(settings_path) as f:
                settings = json.loads(f.read())
            hooks = settings.get("hooks", {})
            if "Stop" in hooks or "UserPromptSubmit" in hooks:
                ok_items.append("Auto-commit hooks configured")
            else:
                issues.append("Hooks not configured — run 'ccr install'")
        except (json.JSONDecodeError, OSError):
            issues.append("settings.local.json exists but unreadable")
    else:
        issues.append("No hooks configured — run 'ccr install' for auto-commit")

    # 8. Hook error log
    hook_errors_log = os.path.join(ccr_dir, ".hook_errors.log")
    if os.path.isfile(hook_errors_log):
        import datetime
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

    # Print results
    click.echo("CCR Doctor\n")
    for item in ok_items:
        click.echo(f"  [OK] {item}")
    for item in issues:
        click.echo(f"  [!!] {item}")
    click.echo(f"\n  {len(ok_items)} OK, {len(issues)} issue(s)")
    if not issues:
        click.echo("  All checks passed.")


@cli.command()
@click.argument("project", default=".")
@click.option("--multiplier", "-m", default=4.0, type=float,
              help="Token re-typing multiplier (default 4). Use 2 for conservative, 6 for optimistic.")
@click.option("--last", default=30, type=int,
              help="Number of recent sessions to include in stats (default 30).")
def stats(project: str, multiplier: float, last: int) -> None:
    """Show CCR ROI dashboard: token savings, memory health, session history."""
    import json as _json
    import sys as _sys
    from datetime import datetime

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")

    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ directory found in {project}. Run 'ccr init' first.", err=True)
        return

    # --- Load session records ---
    jsonl_path = os.path.join(ccr_dir, "metrics", "sessions.jsonl")
    sessions = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                    if rec.get("context_tokens", 0) > 0:
                        sessions.append(rec)
                except _json.JSONDecodeError:
                    continue  # Skip corrupt lines
    sessions = sessions[-last:]  # Most recent N sessions

    # --- Load commit count from GCC ---
    try:
        from ccr.core.memory import MemoryManager
        from ccr.core.types import CCRConfig
        mem = MemoryManager(project, CCRConfig())
        branch = mem.get_active_branch()
        commit_index = mem._build_commit_index(branch)
        commit_count = len(commit_index)
        summary_path = os.path.join(mem.ccr_root, "branches", branch, "commits.md")
        summary_len = 0
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            m = re.search(r"## Rolling Summary\s*\n(.*?)\n---", content, re.DOTALL)
            if m:
                summary_len = len(m.group(1).strip())
    except Exception:
        commit_count = 0
        branch = "main"
        summary_len = 0

    # --- Load playbook ---
    try:
        playbook_path = os.path.join(ccr_dir, "playbook.txt")
        playbook_text = ""
        if os.path.isfile(playbook_path):
            with open(playbook_path, "r", encoding="utf-8") as f:
                playbook_text = f.read()
        playbook_tokens = max(1, len(playbook_text) // 4)
        bullet_count = playbook_text.count("] helpful=")
    except Exception:
        playbook_tokens = 0
        bullet_count = 0

    click.echo("\n=== CCR Stats Dashboard ===\n")

    click.echo(f"Project memory ({branch})")
    click.echo(f"  Commits:          {commit_count}")
    if summary_len:
        click.echo(f"  Rolling summary:  {summary_len}/1500 chars ({summary_len * 100 // 1500}%)")
        if summary_len > 1200:
            click.echo(f"  ⚠️  Summary is getting long — run gcc_consolidate() or gcc_context(level=5)")
    if bullet_count:
        click.echo(f"  Playbook:         {playbook_tokens:,} tokens ({bullet_count} bullets)")

    click.echo()

    if not sessions:
        click.echo("Session history")
        click.echo("  No session history yet.")
        click.echo("  (CCR stats are recorded after each session ends with 'ccr install' hooks active.)")
        return

    # --- Compute aggregates ---
    token_list = [s["context_tokens"] for s in sessions]
    dur_list = [s.get("duration_min", 0) for s in sessions]
    total_sessions = len(sessions)
    avg_tokens = sum(token_list) / total_sessions
    total_tokens = sum(token_list)
    avoided = total_tokens * multiplier
    avg_dur = sum(dur_list) / total_sessions if dur_list else 0
    breakeven = int(1 / multiplier * 10) + 1 if multiplier else 6

    click.echo(f"Session history (last {total_sessions} sessions)")
    click.echo(f"  Avg context injected:  {avg_tokens:,.0f} tokens/session")
    click.echo(f"  Total injected:        {total_tokens:,.0f} tokens")
    click.echo(f"  Est. tokens avoided:   {avoided:,.0f} tokens (×{multiplier:.0f} re-typing heuristic*)")
    if avg_dur:
        click.echo(f"  Avg session duration:  {avg_dur:.0f} min")

    click.echo()
    click.echo(f"30-session projection")
    projected_avoided = avg_tokens * 30 * multiplier
    click.echo(f"  Est. savings:  ~{projected_avoided:,.0f} tokens")
    if total_sessions >= breakeven:
        click.echo(f"  Break-even:    {breakeven} sessions ✓ (already past)")
    else:
        remaining = breakeven - total_sessions
        click.echo(f"  Break-even:    {breakeven} sessions ({remaining} more to go)")

    click.echo()
    click.echo(f"Recent sessions")
    click.echo(f"  {'Date':<12} {'Context':>8}  {'Avoided':>9}  {'Duration':>10}")
    click.echo(f"  {'-'*12} {'-'*8}  {'-'*9}  {'-'*10}")
    for s in reversed(sessions[-10:]):
        try:
            dt = datetime.fromisoformat(s["start"]).strftime("%Y-%m-%d")
        except Exception:
            dt = "unknown"
        ctx = s["context_tokens"]
        avd = int(ctx * multiplier)
        dur = s.get("duration_min", 0)
        dur_str = f"{dur:.0f} min" if dur else "  —  "
        click.echo(f"  {dt:<12} {ctx:>8,}  {avd:>9,}  {dur_str:>10}")

    click.echo()
    click.echo(
        f"* Rough estimate: assumes you'd re-type ~{multiplier:.0f}× the context without CCR."
    )
    click.echo(
        f"  Pass --multiplier N to adjust (e.g., --multiplier 2 for conservative estimate)."
    )


@cli.command()
def projects() -> None:
    """List all known CCR projects from the global registry."""
    import json

    registry_path = os.path.expanduser("~/.ccr/projects.json")
    if not os.path.isfile(registry_path):
        click.echo("No projects registered yet. Run 'ccr init' in a project to register it.")
        return

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            projects_list = json.loads(f.read())
    except (json.JSONDecodeError, OSError):
        click.echo("Registry file unreadable.", err=True)
        return

    if not projects_list:
        click.echo("No projects registered.")
        return

    click.echo("CCR Projects:\n")
    for p in projects_list:
        path = p.get("path", "?")
        name = p.get("name", os.path.basename(path))
        last_used = p.get("last_used", "never")
        commits = p.get("commit_count", "?")
        click.echo(f"  {name}")
        click.echo(f"    Path: {path}")
        click.echo(f"    Commits: {commits}, Last used: {last_used}")
        click.echo()


def _register_project(project_root: str) -> None:
    """Register a project in the global registry (~/.ccr/projects.json)."""
    import json
    from datetime import datetime, timezone

    global_ccr = os.path.expanduser("~/.ccr")
    os.makedirs(global_ccr, exist_ok=True)
    registry_path = os.path.join(global_ccr, "projects.json")

    projects_list: list[dict] = []
    if os.path.isfile(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                projects_list = json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            projects_list = []

    abs_path = os.path.abspath(project_root)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Update existing or add new
    for p in projects_list:
        if p.get("path") == abs_path:
            p["last_used"] = now
            break
    else:
        projects_list.append({
            "path": abs_path,
            "name": os.path.basename(abs_path),
            "last_used": now,
            "commit_count": 0,
        })

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(projects_list, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    cli()
