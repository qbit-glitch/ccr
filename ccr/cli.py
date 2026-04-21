"""CCR CLI — entry point for the ccr command."""

from __future__ import annotations

import logging
import os
import shlex
import signal
import sys

import click

from ccr.core.types import CCRConfig


def _load_config(config_path: str | None, overrides: dict):
    """Load config from YAML file with CLI overrides (legacy ccr start only)."""
    # Lazy imports — only needed for deprecated ccr start command
    import yaml  # noqa: PLC0415
    from ccr.core.types import CCREngineConfig  # noqa: PLC0415

    base = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
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

    # Create SQLite tables for fresh projects
    ccr_dir = os.path.join(project, ".ccr")
    try:
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        backend = SqliteStorageBackend(ccr_dir)
        backend.close()
    except Exception:
        pass

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
@click.option("--preset", default="default",
              type=click.Choice(["default", "ml", "academic"]),
              help="Workflow preset: ml (experiments+code), academic (writing+analysis), default")
def install(project: str, preset: str) -> None:
    """Install CCR hooks into a project for automatic memory management.

    Generates .claude/settings.local.json hook configuration so Claude Code
    auto-commits session progress without manual gcc_commit calls.
    Also initializes .ccr/ if not present.

    \b
    Presets:
        default   Generic mode — suitable for any project
        ml        ML researcher mode — experiment tracking, hypothesis branching
        academic  Academic researcher mode — decision logging, writing workflows
    """
    import json

    from ccr.core.memory import MemoryManager

    project = os.path.abspath(project)

    # Ensure .ccr/ exists
    mem = MemoryManager(project)
    mem.ensure_structure()

    # Write preset to metadata.yaml if non-default
    if preset != "default":
        try:
            from ccr.cli_presets import _write_preset_to_metadata  # noqa: PLC0415
            _write_preset_to_metadata(project, preset)
        except Exception as exc:
            click.echo(f"  [warn] Could not write preset: {exc}", err=True)

    # Find the CCR package root for hook paths
    ccr_pkg = os.path.dirname(os.path.abspath(__file__))
    hooks_dir = os.path.join(ccr_pkg, "hooks")

    # Determine python executable (prefer venv)
    python_exe = sys.executable

    # Build hook commands (all 4 hooks required for full auto-commit chain).
    # CCR_PROJECT_ROOT is baked in at install time so hooks work regardless of
    # which directory Claude Code is launched from.
    # Paths are shell-quoted so spaces in project/python paths work correctly.
    _q = shlex.quote
    _project_q = _q(project)
    _python_q = _q(python_exe)
    hook_config = {
        "UserPromptSubmit": [{
            "type": "command",
            "command": f"CCR_PROJECT_ROOT={_project_q} {_python_q} {_q(os.path.join(hooks_dir, 'on_session_start.py'))}",
        }],
        "PostToolUse": [{
            "type": "command",
            "command": f"CCR_PROJECT_ROOT={_project_q} {_python_q} {_q(os.path.join(hooks_dir, 'on_tool_use.py'))}",
        }],
        "Stop": [{
            "type": "command",
            "command": f"CCR_PROJECT_ROOT={_project_q} {_python_q} {_q(os.path.join(hooks_dir, 'on_stop.py'))}",
        }],
        "PreCompact": [{
            "type": "command",
            "command": f"CCR_PROJECT_ROOT={_project_q} {_python_q} {_q(os.path.join(hooks_dir, 'on_compact.py'))}",
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
    for event, commands in hook_config.items():
        # Identify CCR hook script for this event (last shell token of command string).
        # Use shlex.split to correctly handle quoted paths with spaces.
        try:
            hook_script = shlex.split(commands[0]["command"])[-1]
        except (ValueError, IndexError):
            hook_script = commands[0]["command"].split()[-1]
        # Remove any pre-existing CCR hook for this event (any version/prefix),
        # keeping non-CCR hooks from other tools intact.
        existing_cmds = [
            c for c in existing["hooks"].get(event, [])
            if hook_script not in c.get("command", "")
        ]
        existing["hooks"][event] = existing_cmds + commands

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

    # Warn early if Claude Code CLI is not installed
    import shutil as _shutil
    if not _shutil.which("claude"):
        click.echo(
            "\n[WARN] Claude Code CLI not found in PATH.\n"
            "  CCR is installed but hooks won't fire until Claude Code is available.\n"
            "  Install Claude Code: npm install -g @anthropic-ai/claude-code\n"
            "  Then re-run: ccr install",
            err=True,
        )

    click.echo(f"\n\u2705 CCR installed in {project}  [preset: {preset}]")
    click.echo(f"\n  Hooks ({claude_dir}/settings.local.json):")
    click.echo(f"    UserPromptSubmit \u2192 injects memory context at session start")
    click.echo(f"    PostToolUse      \u2192 tracks tool calls for auto-commit")
    click.echo(f"    Stop             \u2192 auto-commits session progress on exit")
    click.echo(f"    PreCompact       \u2192 saves state before context resets")
    click.echo(f"\n  MCP server ({mcp_json_path}):")
    click.echo(f"    ccr \u2192 {python_exe} -m ccr.mcp_server")
    # Run doctor inline to confirm install succeeded
    click.echo(f"\n=== Install verification ===")
    click.echo(f"  \u2705 CCR_PROJECT_ROOT={project} baked into hook commands.")
    click.echo(f"     Hooks will work regardless of which directory Claude Code is launched from.")
    try:
        from ccr.cli_doctor import _run_doctor_checks  # noqa: PLC0415
        ok_items, issues, notices = _run_doctor_checks(project)
        for item in ok_items:
            click.echo(f"  [OK] {item}")
        for item in notices:
            click.echo(f"  [--] {item}")
        for item in issues:
            click.echo(f"  [!!] {item}")
        if issues:
            click.echo(f"\n  {len(issues)} issue(s) found — fix before opening Claude Code.")
        else:
            click.echo(f"\n  All checks passed.")
    except Exception:
        click.echo(f"  (Run 'ccr doctor' to verify installation)")

    click.echo(f"\n=== Next step ===")
    click.echo(f"  Open Claude Code in any terminal:")
    click.echo(f"    claude")
    click.echo(f"  or from the project directory:")
    click.echo(f"    cd {project} && claude")
    click.echo(f"")
    click.echo(f"  Quick test — ask Claude:")
    click.echo(f'    "What do you remember about this project?"')
    click.echo(f"    Then call: gcc_status()")
    click.echo(f"")
    click.echo(f"  Other useful commands:")
    click.echo(f"    ccr export-context   — export memory for claude.ai (web) sessions")


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


# ---------------------------------------------------------------------------
# Agent management commands
# ---------------------------------------------------------------------------

@cli.group()
def agents():
    """Manage CCR integrations for AI agents."""


@agents.command("list")
def agents_list():
    """List all supported agents and their status."""
    from ccr.adapters import get_adapters

    all_adapters = get_adapters()
    if not all_adapters:
        click.echo("No agent adapters registered.")
        return

    click.echo("Supported AI Agents:\n")
    for adapter in all_adapters:
        status = adapter.status()
        installed = "✓ installed" if status.installed else "  not installed"
        ccr = "CCR enabled" if status.ccr_enabled else "CCR not configured"
        level_name = {5: "MCP+Hooks", 4: "MCP", 3: "FileWatcher", 2: "SDK", 1: "Manual"}.get(
            status.integration_level, "Unknown"
        )
        click.echo(f"  {adapter.display_name:<20} {installed:<18} {ccr} ({level_name})")

    click.echo(f"\n  Run 'ccr agents info <name>' for details on a specific agent.")


@agents.command("info")
@click.argument("name")
def agents_info(name: str):
    """Show detailed info about a specific agent's CCR integration."""
    from ccr.adapters import get_adapter

    adapter = get_adapter(name)
    if not adapter:
        click.echo(f"Unknown agent: {name}", err=True)
        return

    status = adapter.status()
    click.echo(f"Agent: {adapter.display_name} ({adapter.name})")
    click.echo(f"  Installed:      {status.installed}")
    click.echo(f"  CCR enabled:    {status.ccr_enabled}")
    click.echo(f"  MCP support:    {adapter.supports_mcp()}")
    click.echo(f"  Hooks support:  {adapter.supports_hooks()}")
    click.echo(f"  Integration:    {status.integration_level}")
    click.echo(f"  Context format: {adapter.context_format()}")
    click.echo(f"\n  Install command:")
    click.echo(f"    ccr install-global --agents {adapter.name}")


# doctor and stats live in submodules (split for 800-line compliance)
# Re-exported here so `from ccr.cli import doctor/stats` continues to work
from ccr.cli_doctor import doctor  # noqa: E402
from ccr.cli_stats import stats  # noqa: E402
from ccr.cli_global import install_global, uninstall_global  # noqa: E402
cli.add_command(doctor)
cli.add_command(stats)
cli.add_command(install_global)
cli.add_command(uninstall_global)
cli.add_command(agents)


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


# Register export/import commands
from ccr.cli_export import export as export_cmd  # noqa: E402
from ccr.cli_import import import_cmd  # noqa: E402
cli.add_command(export_cmd)
cli.add_command(import_cmd)

# Register preset commands from cli_presets.py
try:
    from ccr.cli_presets import export_context, set_preset  # noqa: PLC0415
    cli.add_command(set_preset)
    cli.add_command(export_context)
except Exception:
    pass  # Non-fatal — commands simply won't appear if import fails


if __name__ == "__main__":
    cli()
