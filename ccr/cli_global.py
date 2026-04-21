"""Global CCR installation commands for Claude Code and Kimi Code CLI.

Extracted from cli.py to satisfy 800-line limit.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys

import click


def _get_ccr_paths() -> tuple[str, str, str]:
    """Return (ccr_pkg_dir, python_executable, hooks_dir)."""
    ccr_pkg = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable
    hooks_dir = os.path.join(ccr_pkg, "hooks")
    return ccr_pkg, python_exe, hooks_dir


def _ensure_claude_global_mcp(python_exe: str) -> str:
    """Write CCR MCP server to ~/.claude/.mcp.json (merge with existing)."""
    claude_dir = os.path.expanduser("~/.claude")
    os.makedirs(claude_dir, exist_ok=True)
    mcp_path = os.path.join(claude_dir, ".mcp.json")

    existing: dict = {}
    if os.path.isfile(mcp_path):
        try:
            with open(mcp_path, "r", encoding="utf-8") as f:
                existing = json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["ccr"] = {
        "command": python_exe,
        "args": ["-m", "ccr.mcp_server", "--project", "."],
        "env": {"CCR_OLLAMA_MODEL": os.environ.get("CCR_OLLAMA_MODEL", "")},
    }
    # Remove empty env values for cleanliness
    if not existing["mcpServers"]["ccr"]["env"].get("CCR_OLLAMA_MODEL"):
        del existing["mcpServers"]["ccr"]["env"]

    with open(mcp_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return mcp_path


def _ensure_claude_global_hooks(python_exe: str, hooks_dir: str) -> str:
    """Write CCR hooks to ~/.claude/settings.json (merge with existing)."""
    claude_dir = os.path.expanduser("~/.claude")
    os.makedirs(claude_dir, exist_ok=True)
    settings_path = os.path.join(claude_dir, "settings.json")

    existing: dict = {}
    if os.path.isfile(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                existing = json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            pass

    _q = shlex.quote
    _py = _q(python_exe)

    ccr_hooks = {
        "UserPromptSubmit": [{
            "type": "command",
            "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_session_start.py'))}",
        }],
        "PostToolUse": [{
            "type": "command",
            "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_tool_use.py'))}",
        }],
        "Stop": [{
            "type": "command",
            "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_stop.py'))}",
        }],
        "PreCompact": [{
            "type": "command",
            "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_compact.py'))}",
        }],
    }

    existing.setdefault("hooks", {})
    for event, commands in ccr_hooks.items():
        # Identify CCR hook script for this event
        try:
            hook_script = shlex.split(commands[0]["command"])[-1]
        except (ValueError, IndexError):
            hook_script = commands[0]["command"].split()[-1]
        # Remove any pre-existing CCR hook for this event
        existing_cmds = [
            c for c in existing["hooks"].get(event, [])
            if hook_script not in c.get("command", "")
        ]
        existing["hooks"][event] = existing_cmds + commands

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return settings_path


def _ensure_kimi_global_mcp(python_exe: str) -> str:
    """Write CCR MCP server to ~/.kimi/mcp.json (merge with existing)."""
    kimi_dir = os.path.expanduser("~/.kimi")
    os.makedirs(kimi_dir, exist_ok=True)
    mcp_path = os.path.join(kimi_dir, "mcp.json")

    existing: dict = {}
    if os.path.isfile(mcp_path):
        try:
            with open(mcp_path, "r", encoding="utf-8") as f:
                existing = json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["ccr"] = {
        "command": python_exe,
        "args": ["-m", "ccr.mcp_server", "--project", "."],
        "env": {"CCR_OLLAMA_MODEL": os.environ.get("CCR_OLLAMA_MODEL", "")},
    }
    if not existing["mcpServers"]["ccr"]["env"].get("CCR_OLLAMA_MODEL"):
        del existing["mcpServers"]["ccr"]["env"]

    with open(mcp_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    return mcp_path


def _ensure_kimi_global_hooks(python_exe: str, hooks_dir: str) -> str:
    """Write CCR hooks to ~/.kimi/config.toml (merge with existing non-CCR hooks)."""
    kimi_dir = os.path.expanduser("~/.kimi")
    os.makedirs(kimi_dir, exist_ok=True)
    config_path = os.path.join(kimi_dir, "config.toml")

    content = ""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

    # Remove existing CCR hooks
    content = re.sub(r"^hooks\s*=\s*\[\]\s*\n?", "", content, flags=re.MULTILINE)
    hook_blocks = []
    rest = content
    while True:
        match = re.search(r"\[\[hooks\]\]\n", rest)
        if not match:
            break
        start = match.start()
        next_match = re.search(r"\[\[hooks\]\]\n", rest[start + 1 :])
        if next_match:
            end = start + 1 + next_match.start()
            block = rest[start:end]
            if "ccr" not in block.lower():
                hook_blocks.append(block)
            rest = rest[:start] + rest[end:]
        else:
            block = rest[start:]
            if "ccr" not in block.lower():
                hook_blocks.append(block)
            rest = rest[:start]
            break

    ccr_hooks_toml = f"""
[[hooks]]
event = "UserPromptSubmit"
command = "CCR_AUTO_INIT=1 {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_session_start.py'))}"
timeout = 30

[[hooks]]
event = "PostToolUse"
command = "CCR_AUTO_INIT=1 {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_tool_use.py'))}"
timeout = 30

[[hooks]]
event = "PreCompact"
command = "CCR_AUTO_INIT=1 {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_compact.py'))}"
timeout = 30

[[hooks]]
event = "Stop"
command = "CCR_AUTO_INIT=1 {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_stop.py'))}"
timeout = 60
"""

    new_content = rest.rstrip() + "\n" + ccr_hooks_toml.lstrip("\n")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return config_path


def _create_global_helpers(python_exe: str, hooks_dir: str, ccr_pkg: str) -> str:
    """Create ~/.ccr/bin/ helper scripts. Returns the bin directory path."""
    bin_dir = os.path.expanduser("~/.ccr/bin")
    os.makedirs(bin_dir, exist_ok=True)

    # ccr-global — runs ccr CLI from the dev venv
    ccr_global = os.path.join(bin_dir, "ccr-global")
    with open(ccr_global, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/bash\n# Global CCR CLI wrapper\nexec {shlex.quote(python_exe)} -m ccr.cli \"$@\"\n")
    os.chmod(ccr_global, 0o755)

    # claude-ccr — auto-inits .ccr/ then launches claude
    claude_ccr = os.path.join(bin_dir, "claude-ccr")
    with open(claude_ccr, "w", encoding="utf-8") as f:
        f.write(
            f"#!/bin/bash\n"
            f"set -e\n"
            f'PROJECT_ROOT="$(pwd)"\n'
            f"if [ ! -d \"$PROJECT_ROOT/.ccr\" ]; then\n"
            f'  echo "[CCR] Initializing memory in $PROJECT_ROOT..."\n'
            f'  CCR_PROJECT_ROOT="$PROJECT_ROOT" {shlex.quote(python_exe)} -m ccr.cli init "$PROJECT_ROOT" 2>/dev/null || true\n'
            f"fi\n"
            f'exec claude "$@"\n'
        )
    os.chmod(claude_ccr, 0o755)

    # kimi-ccr — auto-inits .ccr/ then launches kimi with merged MCP config
    kimi_ccr = os.path.join(bin_dir, "kimi-ccr")
    ollama_model = os.environ.get("CCR_OLLAMA_MODEL", "")
    with open(kimi_ccr, "w", encoding="utf-8") as f:
        f.write(
            "#!/bin/bash\n"
            "set -e\n"
            'PROJECT_ROOT="$(pwd)"\n'
            "if [ ! -d \"$PROJECT_ROOT/.ccr\" ]; then\n"
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
        )
        if ollama_model:
            f.write(
                f"    'env': {{'CCR_OLLAMA_MODEL': '{ollama_model}'}}\n"
            )
        else:
            f.write("    'env': {}\n")
        f.write(
            "}\n"
            "if not existing['mcpServers']['ccr'].get('env', {}).get('CCR_OLLAMA_MODEL'):\n"
            "    existing['mcpServers']['ccr'].pop('env', None)\n"
            "print(json.dumps(existing))\n"
            '" "$EXISTING")\n'
            'exec kimi --mcp-config "$MERGED" "$@"\n'
        )
    os.chmod(kimi_ccr, 0o755)

    return bin_dir


def _update_shell_aliases() -> list[str]:
    """Add CCR aliases to shell profiles. Returns list of modified files."""
    alias_block = (
        '\n# CCR global tools (auto-init memory + launch Claude Code / Kimi)\n'
        'export PATH="$HOME/.ccr/bin:$PATH"\n'
        "alias ccr='ccr-global'\n"
        "alias cclaude='claude-ccr'\n"
        "alias kimi-ccr='kimi-ccr'\n"
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
def install_global(yes: bool) -> None:
    """Install CCR globally for Claude Code and Kimi Code CLI.

    Sets up global MCP servers and hooks so CCR works automatically
    in any directory you launch claude or kimi from. Memory is stored
    per-project in ./.ccr/ and auto-initialized on first use.
    """
    ccr_pkg, python_exe, hooks_dir = _get_ccr_paths()

    click.echo("=== CCR Global Installation ===\n")
    click.echo("This will configure CCR to work across ALL projects:")
    click.echo("  - Claude Code: ~/.claude/.mcp.json + ~/.claude/settings.json")
    click.echo("  - Kimi Code CLI: ~/.kimi/mcp.json + ~/.kimi/config.toml")
    click.echo("  - Helper scripts: ~/.ccr/bin/")
    click.echo("  - Shell aliases: ~/.zshrc + ~/.bashrc")
    click.echo("")
    click.echo("Memory will be auto-created in ./.ccr/ on first use.")
    click.echo("")

    if not yes:
        click.confirm("Proceed?", abort=True)

    # 1. Claude Code global
    try:
        mcp_path = _ensure_claude_global_mcp(python_exe)
        settings_path = _ensure_claude_global_hooks(python_exe, hooks_dir)
        click.echo(f"  [OK] Claude Code global MCP: {mcp_path}")
        click.echo(f"  [OK] Claude Code global hooks: {settings_path}")
    except Exception as exc:
        click.echo(f"  [!!] Claude Code setup failed: {exc}", err=True)

    # 2. Kimi global
    try:
        mcp_path = _ensure_kimi_global_mcp(python_exe)
        config_path = _ensure_kimi_global_hooks(python_exe, hooks_dir)
        click.echo(f"  [OK] Kimi global MCP: {mcp_path}")
        click.echo(f"  [OK] Kimi global hooks: {config_path}")
    except Exception as exc:
        click.echo(f"  [!!] Kimi setup failed: {exc}", err=True)

    # 3. Helper scripts
    try:
        bin_dir = _create_global_helpers(python_exe, hooks_dir, ccr_pkg)
        click.echo(f"  [OK] Helper scripts: {bin_dir}")
    except Exception as exc:
        click.echo(f"  [!!] Helper scripts failed: {exc}", err=True)

    # 4. Shell aliases
    modified = _update_shell_aliases()
    if modified:
        for rc in modified:
            click.echo(f"  [OK] Added aliases to {rc}")
        click.echo("\n  Run 'source ~/.zshrc' (or ~/.bashrc) to activate aliases.")
    else:
        click.echo("  [OK] Aliases already present in shell profiles")

    click.echo("\n=== Next steps ===")
    click.echo("  Open a NEW terminal, then:")
    click.echo("    cd ~/any-project && claude     # Claude Code with auto-memory")
    click.echo("    cd ~/any-project && kimi       # Kimi with auto-memory")
    click.echo("")
    click.echo("  Or use wrappers before reloading shell:")
    click.echo("    ~/.ccr/bin/claude-ccr")
    click.echo("    ~/.ccr/bin/kimi-ccr")


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def uninstall_global(yes: bool) -> None:
    """Remove global CCR configuration. Per-project .ccr/ memory is preserved."""
    import json as _json

    paths_to_clean = []

    # Claude
    claude_mcp = os.path.expanduser("~/.claude/.mcp.json")
    claude_settings = os.path.expanduser("~/.claude/settings.json")
    if os.path.isfile(claude_mcp):
        paths_to_clean.append(("Claude MCP", claude_mcp))
    if os.path.isfile(claude_settings):
        paths_to_clean.append(("Claude hooks", claude_settings))

    # Kimi
    kimi_mcp = os.path.expanduser("~/.kimi/mcp.json")
    kimi_config = os.path.expanduser("~/.kimi/config.toml")
    if os.path.isfile(kimi_mcp):
        paths_to_clean.append(("Kimi MCP", kimi_mcp))
    if os.path.isfile(kimi_config):
        paths_to_clean.append(("Kimi hooks", kimi_config))

    # Helpers
    bin_dir = os.path.expanduser("~/.ccr/bin")
    if os.path.isdir(bin_dir):
        paths_to_clean.append(("Helper scripts", bin_dir))

    if not paths_to_clean:
        click.echo("CCR is not installed globally — nothing to remove.")
        return

    click.echo("Will remove global CCR configuration:")
    for label, path in paths_to_clean:
        click.echo(f"  {label}: {path}")
    click.echo("  Per-project .ccr/ memory will NOT be touched.")

    if not yes:
        click.confirm("Proceed?", abort=True)

    # Claude MCP
    if os.path.isfile(claude_mcp):
        try:
            with open(claude_mcp, "r", encoding="utf-8") as f:
                d = _json.loads(f.read())
            d.get("mcpServers", {}).pop("ccr", None)
            with open(claude_mcp, "w", encoding="utf-8") as f:
                _json.dump(d, f, indent=2)
                f.write("\n")
            click.echo(f"  ✓ Removed CCR from {claude_mcp}")
        except Exception as exc:
            click.echo(f"  ⚠️  Failed to update {claude_mcp}: {exc}", err=True)

    # Claude settings
    if os.path.isfile(claude_settings):
        try:
            with open(claude_settings, "r", encoding="utf-8") as f:
                d = _json.loads(f.read())
            for event in list(d.get("hooks", {}).keys()):
                d["hooks"][event] = [
                    c for c in d["hooks"][event]
                    if "ccr" not in c.get("command", "").lower()
                ]
                if not d["hooks"][event]:
                    del d["hooks"][event]
            if not d.get("hooks"):
                d.pop("hooks", None)
            with open(claude_settings, "w", encoding="utf-8") as f:
                _json.dump(d, f, indent=2)
                f.write("\n")
            click.echo(f"  ✓ Removed CCR hooks from {claude_settings}")
        except Exception as exc:
            click.echo(f"  ⚠️  Failed to update {claude_settings}: {exc}", err=True)

    # Kimi MCP
    if os.path.isfile(kimi_mcp):
        try:
            with open(kimi_mcp, "r", encoding="utf-8") as f:
                d = _json.loads(f.read())
            d.get("mcpServers", {}).pop("ccr", None)
            with open(kimi_mcp, "w", encoding="utf-8") as f:
                _json.dump(d, f, indent=2)
                f.write("\n")
            click.echo(f"  ✓ Removed CCR from {kimi_mcp}")
        except Exception as exc:
            click.echo(f"  ⚠️  Failed to update {kimi_mcp}: {exc}", err=True)

    # Kimi config
    if os.path.isfile(kimi_config):
        try:
            with open(kimi_config, "r", encoding="utf-8") as f:
                content = f.read()
            content = re.sub(r"^hooks\s*=\s*\[\]\s*\n?", "", content, flags=re.MULTILINE)
            rest = content
            cleaned = ""
            while True:
                match = re.search(r"\[\[hooks\]\]\n", rest)
                if not match:
                    cleaned += rest
                    break
                start = match.start()
                next_match = re.search(r"\[\[hooks\]\]\n", rest[start + 1 :])
                if next_match:
                    end = start + 1 + next_match.start()
                    block = rest[start:end]
                    if "ccr" not in block.lower():
                        cleaned += block
                    rest = rest[end:]
                else:
                    block = rest[start:]
                    if "ccr" not in block.lower():
                        cleaned += block
                    rest = rest[:start]
                    break
            with open(kimi_config, "w", encoding="utf-8") as f:
                f.write(cleaned)
            click.echo(f"  ✓ Removed CCR hooks from {kimi_config}")
        except Exception as exc:
            click.echo(f"  ⚠️  Failed to update {kimi_config}: {exc}", err=True)

    # Helpers
    if os.path.isdir(bin_dir):
        import shutil as _shutil
        try:
            _shutil.rmtree(bin_dir)
            click.echo(f"  ✓ Removed {bin_dir}")
        except Exception as exc:
            click.echo(f"  ⚠️  Failed to remove {bin_dir}: {exc}", err=True)

    click.echo("\nCCR globally uninstalled. Your .ccr/ memory is intact.")
    click.echo("To reinstall: ccr install-global")
