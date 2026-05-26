"""Optional Git-backed local-first sync for CCR memory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SyncResult:
    ok: bool
    message: str
    commands: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "commands": self.commands,
            "artifacts": self.artifacts,
        }


class GitMemorySync:
    def __init__(self, project: str):
        self.project = os.path.abspath(project)
        self.ccr_root = os.path.join(self.project, ".ccr")
        self.sync_root = os.path.join(self.ccr_root, "sync")
        self.repo = os.path.join(self.sync_root, "repo")
        self.config_path = os.path.join(self.sync_root, "config.json")

    def init(self, remote: str = "") -> SyncResult:
        if not os.path.isdir(self.ccr_root):
            return SyncResult(False, ".ccr directory not found")
        os.makedirs(self.repo, exist_ok=True)
        commands = [self._git(["init"])]
        if remote:
            existing = self._git(["remote"], check=False)
            if "origin" not in existing:
                commands.append(self._git(["remote", "add", "origin", remote]))
        self._write_config({"version": 1, "remote": remote, "repo": self.repo})
        return SyncResult(True, "sync repo initialized", commands=commands, artifacts=[self.repo])

    def push(self, message: str = "") -> SyncResult:
        if not os.path.isdir(os.path.join(self.repo, ".git")):
            init_result = self.init()
            if not init_result.ok:
                return init_result
        self._snapshot_to_repo()
        # Force-add: the snapshot lives under memory/.ccr/, which is matched by the
        # near-universal ".ccr/" gitignore pattern (global excludesFile or project).
        # Without -f the snapshot is silently ignored, nothing is committed, and the
        # push below fails with "src refspec HEAD does not match any".
        commands = [
            self._git(["add", "-f", "memory"]),
            self._git(["status", "--short"], check=False),
        ]
        if commands[-1].strip():
            msg = message or f"CCR memory sync {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
            commands.append(self._git([
                "-c", "user.email=ccr-sync@local",
                "-c", "user.name=CCR Sync",
                "commit", "-m", msg,
            ], check=False))
        cfg = self._read_config()
        if cfg.get("remote"):
            push_out = self._git(["push", "origin", "HEAD"], check=False)
            commands.append(push_out)
            if any(tok in push_out.lower() for tok in ("error:", "fatal:", "rejected", "failed to push")):
                return SyncResult(
                    False,
                    f"sync push to remote failed: {push_out.strip()[:200]}",
                    commands=commands,
                    artifacts=[self.repo],
                )
        return SyncResult(True, "sync push completed", commands=commands, artifacts=[self.repo])

    def pull(self, apply: bool = False) -> SyncResult:
        if not os.path.isdir(os.path.join(self.repo, ".git")):
            return SyncResult(False, "sync repo not initialized")
        commands = []
        cfg = self._read_config()
        if cfg.get("remote"):
            commands.append(self._git(["pull", "--ff-only", "origin", "HEAD"], check=False))
        if apply:
            source = os.path.join(self.repo, "memory", ".ccr")
            if not os.path.isdir(source):
                return SyncResult(False, "sync snapshot not found", commands=commands)
            self._copy_tree(source, self.ccr_root)
            return SyncResult(True, "sync snapshot applied", commands=commands, artifacts=[self.ccr_root])
        return SyncResult(True, "sync pull checked; pass --apply to restore snapshot", commands=commands)

    def resolve(self) -> SyncResult:
        conflict_files: list[str] = []
        for dirpath, _, filenames in os.walk(self.repo):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    with open(path, encoding="utf-8") as fh:
                        text = fh.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if "<<<<<<<" in text or ">>>>>>>" in text:
                    conflict_files.append(os.path.relpath(path, self.repo))
        if conflict_files:
            return SyncResult(False, "manual sync conflict resolution required", artifacts=conflict_files)
        return SyncResult(True, "no sync conflicts detected")

    def _snapshot_to_repo(self) -> None:
        dest = os.path.join(self.repo, "memory", ".ccr")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        self._copy_tree(self.ccr_root, dest)

    def _copy_tree(self, src: str, dest: str) -> None:
        ignore = shutil.ignore_patterns("backups", "sync", "*.tmp", "*.bak")
        shutil.copytree(src, dest, ignore=ignore, dirs_exist_ok=True)

    def _git(self, args: list[str], check: bool = True) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
        )
        output = (proc.stdout + proc.stderr).strip()
        if check and proc.returncode != 0:
            raise RuntimeError(output or f"git {' '.join(args)} failed")
        return output

    def _write_config(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _read_config(self) -> dict[str, Any]:
        try:
            with open(self.config_path, encoding="utf-8") as fh:
                return json.loads(fh.read() or "{}")
        except (OSError, json.JSONDecodeError):
            return {}
