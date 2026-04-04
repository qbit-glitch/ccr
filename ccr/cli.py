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
    """Start the CCR gateway proxy."""
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

    # Build hook commands
    hook_config = {
        "UserPromptSubmit": [{
            "type": "command",
            "command": f"{python_exe} {os.path.join(hooks_dir, 'on_session_start.py')}",
        }],
        "Stop": [{
            "type": "command",
            "command": f"{python_exe} {os.path.join(hooks_dir, 'on_stop.py')}",
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

    click.echo(f"Installed CCR hooks in {claude_dir}/settings.local.json")
    click.echo(f"  - UserPromptSubmit: session start + context injection")
    click.echo(f"  - Stop: auto-commit session progress")
    click.echo(f"\n.ccr/ initialized in {project}")
    click.echo("CCR will now auto-commit your session progress. No manual gcc_commit needed.")


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

    # 8. Disk usage
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
