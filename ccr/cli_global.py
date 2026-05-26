"""Global CCR installation commands — now provider-agnostic via adapters.

Extracted from cli.py to satisfy 800-line limit.
"""
from __future__ import annotations

import os
import sys

import click

from ccr.adapters import (
    BaseAgentAdapter,
    get_adapter,
    get_adapters,
    list_adapter_names,
)


def _get_ccr_paths() -> tuple[str, str, str]:
    """Return (ccr_pkg_dir, python_executable, hooks_dir)."""
    ccr_pkg = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable
    hooks_dir = os.path.join(ccr_pkg, "hooks")
    return ccr_pkg, python_exe, hooks_dir


def _resolve_agents(agents_flag: str) -> list[BaseAgentAdapter]:
    """Resolve --agents flag to a list of adapter instances.

    Values:
        auto   → all adapters whose agent is_installed()
        all    → all registered adapters
        name1,name2  → specific adapters by name
    """
    all_adapters = get_adapters()

    if agents_flag == "auto":
        return [a for a in all_adapters if a.is_installed()]

    if agents_flag == "all":
        return all_adapters

    names = [n.strip() for n in agents_flag.split(",") if n.strip()]
    resolved = []
    for name in names:
        adapter = get_adapter(name)
        if adapter:
            resolved.append(adapter)
        else:
            known = ", ".join(list_adapter_names())
            click.echo(f"[WARN] Unknown agent '{name}'. Known agents: {known}", err=True)
    return resolved


def _create_global_helpers(python_exe: str, hooks_dir: str, ccr_pkg: str, adapters: list[BaseAgentAdapter]) -> str:
    """Create ~/.ccr/bin/ helper scripts. Returns the bin directory path."""
    bin_dir = os.path.expanduser("~/.ccr/bin")
    os.makedirs(bin_dir, exist_ok=True)

    # ccr-global — runs ccr CLI from the dev venv
    ccr_global = os.path.join(bin_dir, "ccr-global")
    with open(ccr_global, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/bash\n# Global CCR CLI wrapper\nexec {__import__('shlex').quote(python_exe)} -m ccr.cli \"$@\"\n")
    os.chmod(ccr_global, 0o755)

    # Per-agent wrapper scripts
    for adapter in adapters:
        if adapter.name == "claude-code":
            _write_claude_ccr(bin_dir, python_exe, ccr_pkg)
        elif adapter.name == "kimi":
            _write_kimi_ccr(bin_dir, python_exe, ccr_pkg)
        elif adapter.name == "codex":
            _write_codex_ccr(bin_dir, python_exe)

    return bin_dir


def _write_claude_ccr(bin_dir: str, python_exe: str, ccr_pkg: str) -> None:
    script = os.path.join(bin_dir, "claude-ccr")
    with open(script, "w", encoding="utf-8") as f:
        f.write(
            "#!/bin/bash\n"
            "set -e\n"
            'PROJECT_ROOT="$(pwd)"\n'
            'if [ ! -d "$PROJECT_ROOT/.ccr" ]; then\n'
            '  echo "[CCR] Initializing memory in $PROJECT_ROOT..."\n'
            f'  CCR_PROJECT_ROOT="$PROJECT_ROOT" {__import__("shlex").quote(python_exe)} -m ccr.cli init "$PROJECT_ROOT" 2>/dev/null || true\n'
            "fi\n"
            'exec claude "$@"\n'
        )
    os.chmod(script, 0o755)


def _write_kimi_ccr(bin_dir: str, python_exe: str, ccr_pkg: str) -> None:
    import shlex

    script = os.path.join(bin_dir, "kimi-ccr")
    with open(script, "w", encoding="utf-8") as f:
        f.write(
            "#!/bin/bash\n"
            "set -e\n"
            'PROJECT_ROOT="$(pwd)"\n'
            'if [ ! -d "$PROJECT_ROOT/.ccr" ]; then\n'
            '  echo "[CCR] Initializing memory in $PROJECT_ROOT..."\n'
            f'  CCR_PROJECT_ROOT="$PROJECT_ROOT" {shlex.quote(python_exe)} -m ccr.cli init "$PROJECT_ROOT" 2>/dev/null || true\n'
            "fi\n"
            'KIMI_MCP_CONFIG="$HOME/.kimi/mcp.json"\n'
            "if [ -f \"$KIMI_MCP_CONFIG\" ]; then\n"
            '  EXISTING=$(cat "$KIMI_MCP_CONFIG")\n'
            "else\n"
            '  EXISTING=\'{"mcpServers":{}}\'\n'
            "fi\n"
            'MERGED=$(python3 -c "\n'
            "import json, sys\n"
            "existing = json.loads(sys.argv[1])\n"
            "existing.setdefault('mcpServers', {})\n"
            "existing['mcpServers']['ccr'] = {\n"
            f"    'command': '{python_exe}',\n"
            "    'args': ['-m', 'ccr.mcp_server', '--project', '$PROJECT_ROOT'],\n"
            "}\n"
            "print(json.dumps(existing))\n"
            '" "$EXISTING")\n'
            'exec kimi --mcp-config "$MERGED" "$@"\n'
        )
    os.chmod(script, 0o755)


def _write_codex_ccr(bin_dir: str, python_exe: str) -> None:
    import shlex

    script = os.path.join(bin_dir, "codex-ccr")
    with open(script, "w", encoding="utf-8") as f:
        f.write(
            "#!/bin/bash\n"
            "set -e\n"
            f"exec {shlex.quote(python_exe)} -m ccr.cli_codex \"$@\"\n"
        )
    os.chmod(script, 0o755)


def _update_shell_aliases() -> list[str]:
    """Add CCR aliases to shell profiles. Returns list of modified files."""
    alias_block = (
        '\n# CCR global tools (auto-init memory + launch AI agents)\n'
        'export PATH="$HOME/.ccr/bin:$PATH"\n'
        "alias ccr='ccr-global'\n"
        "alias cclaude='claude-ccr'\n"
        "alias ckimi='kimi-ccr'\n"
        "alias ccodex='codex-ccr'\n"
    )
    modified: list[str] = []
    for rc_file in (os.path.expanduser("~/.zshrc"), os.path.expanduser("~/.bashrc")):
        existing = ""
        if os.path.isfile(rc_file):
            with open(rc_file, "r", encoding="utf-8") as f:
                existing = f.read()
        if ".ccr/bin" not in existing:
            with open(rc_file, "a", encoding="utf-8") as f:
                f.write(alias_block)
            modified.append(rc_file)
    return modified


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.option(
    "--agents",
    default="claude-code,kimi,codex",
    show_default=True,
    help="Comma-separated agent names, or 'auto' (detect installed), or 'all'",
)
def install_global(yes: bool, agents: str) -> None:
    """Install CCR globally for AI agents.

    Sets up global MCP servers and hooks so CCR works automatically
    in any directory you launch your agent from. Memory is stored
    per-project in ./.ccr/ and auto-initialized on first use.

    \b
    Examples:
        ccr install-global                    # Claude Code + Kimi + Codex (default)
        ccr install-global --agents auto      # Auto-detect installed agents
        ccr install-global --agents all       # All supported agents
        ccr install-global --agents ollama    # Ollama only
    """
    ccr_pkg, python_exe, hooks_dir = _get_ccr_paths()

    selected = _resolve_agents(agents)
    if not selected:
        click.echo("No agents selected or found. Run 'ccr agents list' to see supported agents.", err=True)
        return

    names = ", ".join(a.display_name for a in selected)
    click.echo("=== CCR Global Installation ===\n")
    click.echo(f"Target agents: {names}")
    click.echo("Memory will be auto-created in ./.ccr/ on first use.\n")

    if not yes:
        click.confirm("Proceed?", abort=True)

    # Run adapter installs
    all_created: list[str] = []
    all_modified: list[str] = []
    failures: list[str] = []

    for adapter in selected:
        click.echo(f"\n  → Setting up {adapter.display_name}...")
        try:
            result = adapter.install(python_exe, hooks_dir, ccr_pkg)
            if result.success:
                for f in result.files_created:
                    click.echo(f"    [OK] Created: {f}")
                for f in result.files_modified:
                    click.echo(f"    [OK] Updated: {f}")
                all_created.extend(result.files_created)
                all_modified.extend(result.files_modified)
            else:
                failures.append(f"{adapter.display_name}: {result.message}")
        except Exception as exc:
            failures.append(f"{adapter.display_name}: {exc}")

    # Helper scripts
    try:
        bin_dir = _create_global_helpers(python_exe, hooks_dir, ccr_pkg, selected)
        click.echo(f"\n  [OK] Helper scripts: {bin_dir}")
    except Exception as exc:
        failures.append(f"Helper scripts: {exc}")

    # Shell aliases
    modified = _update_shell_aliases()
    if modified:
        for rc in modified:
            click.echo(f"  [OK] Added aliases to {rc}")
        click.echo("\n  Run 'source ~/.zshrc' (or ~/.bashrc) to activate aliases.")
    else:
        click.echo("  [OK] Aliases already present in shell profiles")

    # Summary
    click.echo("\n=== Summary ===")
    if all_created:
        click.echo(f"  Created: {len(all_created)} file(s)")
    if all_modified:
        click.echo(f"  Modified: {len(all_modified)} file(s)")
    if failures:
        click.echo(f"  Failures: {len(failures)}")
        for f in failures:
            click.echo(f"    [!!] {f}")

    click.echo("\n=== Next steps ===")
    click.echo("  Open a NEW terminal, then:")
    for adapter in selected:
        if adapter.name == "claude-code":
            click.echo("    cd ~/any-project && claude     # Claude Code with auto-memory")
        elif adapter.name == "kimi":
            click.echo("    cd ~/any-project && kimi       # Kimi with auto-memory")
        elif adapter.name == "codex":
            click.echo("    cd ~/any-project && codex-ccr  # Codex with CCR lifecycle")
        else:
            click.echo(f"    cd ~/any-project && <{adapter.name}>   # {adapter.display_name}")


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.option(
    "--agents",
    default="claude-code,kimi,codex",
    show_default=True,
    help="Comma-separated agent names, or 'auto' (detect installed), or 'all'",
)
def uninstall_global(yes: bool, agents: str) -> None:
    """Remove global CCR configuration. Per-project .ccr/ memory is preserved."""
    selected = _resolve_agents(agents)
    if not selected:
        click.echo("No agents selected. Run 'ccr agents list' to see supported agents.", err=True)
        return

    names = ", ".join(a.display_name for a in selected)
    click.echo(f"Will remove global CCR configuration for: {names}")
    click.echo("  Per-project .ccr/ memory will NOT be touched.")

    if not yes:
        click.confirm("Proceed?", abort=True)

    failures: list[str] = []
    for adapter in selected:
        click.echo(f"\n  → Removing {adapter.display_name}...")
        try:
            result = adapter.uninstall()
            if result.success:
                for f in result.files_removed:
                    click.echo(f"    [OK] Removed: {f}")
                for f in result.files_modified:
                    click.echo(f"    [OK] Updated: {f}")
            else:
                failures.append(f"{adapter.display_name}: {result.message}")
        except Exception as exc:
            failures.append(f"{adapter.display_name}: {exc}")

    if failures:
        click.echo(f"\n  {len(failures)} failure(s):")
        for f in failures:
            click.echo(f"    [!!] {f}")

    click.echo("\nCCR globally uninstalled for selected agents. Your .ccr/ memory is intact.")
    click.echo("To reinstall: ccr install-global")
