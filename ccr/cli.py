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


if __name__ == "__main__":
    cli()
