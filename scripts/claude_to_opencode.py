#!/usr/bin/env python3
"""Mirror Claude Code's MCP servers, skills, commands and agents into opencode's user config.

Writes ~/.config/opencode/{opencode.json,command/,agent/} (never the repo). The json holds copies of
the API keys from Claude's own config, so it is chmod 600. Re-run after changing Claude's setup.
Not portable (opencode has its own plugin system): Claude hooks, statusline, plugin LSP servers.
Usage: claude_to_opencode.py [--dry-run] [--enable NAME ...] [--skills user|all]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

HOME = os.path.expanduser("~")
CL, OC = f"{HOME}/.claude", f"{HOME}/.config/opencode"
# Disabled by default: an unattended agent with --auto must not hold these. Enable per name.
DENY = {"bit2me": "exchange trading keys", "zotero": "UNSAFE_OPERATIONS can delete library items"}
MODEL = "qwen3.8:27b"


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def latest_plugin_dirs() -> list[str]:
    """One dir per plugin: the most recently installed version."""
    out = []
    for plugin in glob.glob(f"{CL}/plugins/cache/*/*"):
        vers = [v for v in glob.glob(plugin + "/*") if os.path.isdir(v)]
        if vers:
            out.append(max(vers, key=os.path.getmtime))
    return out


def convert(name: str, s: dict, root: str = "") -> dict | None:
    sub = lambda x: x.replace("${CLAUDE_PLUGIN_ROOT}", root) if isinstance(x, str) else x  # noqa: E731
    if "command" in s:
        out = {"type": "local", "command": [sub(s["command"]), *map(sub, s.get("args", []))]}
        if s.get("env"):
            out["environment"] = {k: sub(v) for k, v in s["env"].items()}
    elif s.get("url"):
        hdr = s.get("headers", {})
        if any("${" in v for v in hdr.values()):  # unresolved token placeholder: needs manual auth
            return None
        out = {"type": "remote", "url": s["url"]}
        if hdr:
            out["headers"] = hdr
    else:
        return None
    out["enabled"] = name not in DENY
    return out


def collect_mcp() -> dict:
    servers: dict = {}
    sources = [(_load(f"{HOME}/.claude.json").get("mcpServers", {}), ""),
               (_load(f"{CL}/.mcp.json").get("mcpServers", {}), ""),
               (_load(f"{CL}/settings.json").get("mcpServers", {}), "")]
    for d in latest_plugin_dirs():
        j = _load(f"{d}/.mcp.json")
        sources.append((j.get("mcpServers", j), d))
    for group, root in sources:
        for name, s in group.items():
            if isinstance(s, dict) and name not in servers and (c := convert(name, s, root)):
                servers[name] = c
    return servers


def md_convert(text: str, keep: tuple[str, ...]) -> str:
    """Keep only whitelisted frontmatter keys (Claude and opencode frontmatter differ)."""
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        return text
    lines = [ln for ln in m.group(1).splitlines() if ln.split(":")[0] in keep]
    return "---\n" + "\n".join(lines) + "\n---\n" + m.group(2)


def sync_md(src_glob: str, dest: str, keep: tuple[str, ...], extra: str, dry: bool) -> int:
    files = glob.glob(src_glob)
    for f in files:
        if not dry:
            os.makedirs(dest, exist_ok=True)
            body = md_convert(open(f, encoding="utf-8").read(), keep)
            if extra and body.startswith("---\n"):
                body = body.replace("---\n", f"---\n{extra}\n", 1)
            open(f"{dest}/{os.path.basename(f)}", "w", encoding="utf-8").write(body)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--enable", nargs="*", default=[], help="enable servers disabled by default")
    ap.add_argument("--skills", choices=["user", "all"], default="all")
    a = ap.parse_args()

    mcp = collect_mcp()
    for n in a.enable:
        if n in mcp:
            mcp[n]["enabled"] = True
    skill_dirs = [f"{CL}/skills"] + ([f"{d}/skills" for d in latest_plugin_dirs()] if a.skills == "all" else [])
    skill_dirs = [d for d in skill_dirs if os.path.isdir(d)]

    cfg_path = f"{OC}/opencode.json"
    cfg = _load(cfg_path)
    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    cfg.setdefault("provider", {})["ollama"] = {
        "npm": "@ai-sdk/openai-compatible", "name": "Ollama",
        "options": {"baseURL": "http://localhost:11434/v1"}, "models": {MODEL: {"name": MODEL}}}
    cfg.setdefault("model", f"ollama/{MODEL}")
    cfg["mcp"] = mcp
    cfg["skills"] = skill_dirs

    n_cmd = sync_md(f"{CL}/commands/*.md", f"{OC}/command", ("description",), "", a.dry_run)
    n_agent = sync_md(f"{CL}/agents/*.md", f"{OC}/agent", ("description",), "mode: subagent", a.dry_run)
    on = sorted(k for k, v in mcp.items() if v["enabled"])
    print(f"mcp servers: {len(mcp)} ({len(on)} enabled); disabled: "
          f"{[f'{k} ({DENY[k]})' for k in mcp if not mcp[k]['enabled'] and k in DENY]}")
    print(f"skill dirs: {len(skill_dirs)}; commands: {n_cmd}; agents: {n_agent}")
    if a.dry_run:
        return
    os.makedirs(OC, exist_ok=True)
    if os.path.exists(cfg_path):
        shutil.copy(cfg_path, cfg_path + ".bak")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(cfg_path, 0o600)
    print(f"wrote {cfg_path} (chmod 600)")


if __name__ == "__main__":
    sys.exit(main())
