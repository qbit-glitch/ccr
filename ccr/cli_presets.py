"""CCR CLI preset commands — set-preset and export-context."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import click

VALID_PRESETS = {"default", "ml", "academic"}


def _write_preset_to_metadata(project_root: str, preset: str) -> None:
    """Write preset field to .ccr/metadata.yaml (idempotent, preserves other keys).

    Args:
        project_root: Path to the project directory containing .ccr/.
        preset: One of "default", "ml", "academic".

    Raises:
        ValueError: If preset is not a recognised value.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(f"Unknown preset: {preset!r}. Valid: {VALID_PRESETS}")

    import yaml  # noqa: PLC0415 — lazy to keep CLI import fast

    meta_path = os.path.join(project_root, ".ccr", "metadata.yaml")
    if not os.path.isfile(meta_path):
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    meta["preset"] = preset
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)


@click.command("set-preset")
@click.argument("preset", type=click.Choice(sorted(VALID_PRESETS)))
@click.argument("project", default=".")
def set_preset(preset: str, project: str) -> None:
    """Change the workflow preset for an existing CCR installation.

    PRESET is one of: default, ml, academic.

    \b
    Examples:
        ccr set-preset ml            # ML researcher mode (experiment tracking)
        ccr set-preset academic .    # Academic researcher mode (writing/analysis)
        ccr set-preset default .     # Reset to generic mode

    Restart Claude Code after changing preset to pick up the new directive.
    """
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")
    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ found in {project}. Run 'ccr install' first.", err=True)
        sys.exit(1)

    try:
        _write_preset_to_metadata(project, preset)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"Preset set to '{preset}' in {project}/.ccr/metadata.yaml")
    click.echo("Restart Claude Code to apply the new workflow directive.")


@click.command("export-context")
@click.argument("project", default=".")
@click.option("--level", "-l", default=2, type=click.IntRange(1, 5),
              help="Context depth 1-5 (default 2: summary + recent commits)")
@click.option("--output", "-o", default=None,
              help="Write to FILE instead of stdout (e.g. --output CONTEXT.md)")
def export_context(project: str, level: int, output: str | None) -> None:
    """Export project memory as markdown — paste into any Claude session.

    \b
    Usage:
        ccr export-context > CONTEXT.md
        ccr export-context --output CONTEXT.md
        ccr export-context --level 3 --output CONTEXT.md

    The output can be pasted directly into claude.ai or any Claude interface
    to restore project memory without running Claude Code.
    """
    import sys  # noqa: PLC0415

    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")
    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ found in {project}. Run 'ccr install' first.", err=True)
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ccr.core.memory import MemoryManager  # noqa: PLC0415
    from ccr.core.types import CCRConfig  # noqa: PLC0415

    mem = MemoryManager(project, CCRConfig())
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = (
        f"# CCR Context Export\n"
        f"# Generated: {generated_at}\n"
        f"# Project:   {project}\n"
        f"# Level:     {level}\n\n"
        "---\n\n"
        "<!-- Paste this block at the start of any Claude session to restore project memory -->\n\n"
    )

    context_text = mem.get_context(level=level)
    content = header + context_text + "\n"

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"Context (level {level}) written to {output}")
    else:
        click.echo(content, nl=False)
