"""Reliability and ops helpers: backup, restore, verify, repair, migrate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ccr.core.governance import signed_manifest


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class OpsReport:
    ok: bool
    checks: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "issues": self.issues,
            "artifacts": self.artifacts,
        }


def _ccr_dir(project: str) -> str:
    return os.path.join(os.path.abspath(project), ".ccr")


def backup_project(project: str, output: str = "") -> OpsReport:
    ccr_dir = _ccr_dir(project)
    if not os.path.isdir(ccr_dir):
        return OpsReport(False, issues=[".ccr directory not found"])
    backups_dir = os.path.join(ccr_dir, "backups")
    os.makedirs(backups_dir, exist_ok=True)
    out = output or os.path.join(backups_dir, f"ccr-backup-{utc_stamp()}.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(ccr_dir):
            dirnames[:] = [d for d in dirnames if d not in {"backups", "sync"}]
            for name in filenames:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, os.path.dirname(ccr_dir))
                zf.write(path, rel)
    manifest = signed_manifest(out)
    return OpsReport(True, checks=["backup created", f"sha256={manifest['sha256']}"], artifacts=[out, out + ".manifest.json"])


def restore_backup(project: str, backup: str, overwrite: bool = False) -> OpsReport:
    if not zipfile.is_zipfile(backup):
        return OpsReport(False, issues=["backup is not a zip file"])
    project = os.path.abspath(project)
    target = _ccr_dir(project)
    if os.path.exists(target) and not overwrite:
        target = os.path.join(project, f".ccr.restore-{utc_stamp()}")
    os.makedirs(project, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ccr-restore-") as tmp:
        with zipfile.ZipFile(backup) as zf:
            zf.extractall(tmp)
        extracted = os.path.join(tmp, ".ccr")
        if not os.path.isdir(extracted):
            return OpsReport(False, issues=["backup does not contain .ccr/"])
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(extracted, target)
    return OpsReport(True, checks=["backup restored"], artifacts=[target])


def verify_project(project: str) -> OpsReport:
    ccr_dir = _ccr_dir(project)
    report = OpsReport(True)
    if not os.path.isdir(ccr_dir):
        return OpsReport(False, issues=[".ccr directory not found"])
    report.checks.append(".ccr exists")
    for name in (
        "facts.json",
        "scratchpad.json",
        "agent_events.jsonl",
        "quarantine.json",
        "temporal_graph.json",
        "episodes.jsonl",
        "recall_traces.jsonl",
        "governance_audit.jsonl",
    ):
        path = os.path.join(ccr_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            if name.endswith(".json"):
                with open(path, encoding="utf-8") as fh:
                    json.loads(fh.read() or "{}")
            else:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            json.loads(line)
            report.checks.append(f"{name} parse ok")
        except (OSError, json.JSONDecodeError) as exc:
            report.ok = False
            report.issues.append(f"{name} parse failed: {exc}")
    try:
        from ccr.core.episodes import EpisodeStore
        episode_check = EpisodeStore(ccr_dir).verify_chain()
        if episode_check.ok:
            report.checks.append(f"episodes hash chain ok ({episode_check.checked} checked)")
        else:
            report.ok = False
            report.issues.extend(episode_check.errors)
    except Exception as exc:
        report.ok = False
        report.issues.append(f"episodes verification failed: {type(exc).__name__}: {exc}")
    try:
        from ccr.core.governance import verify_audit_chain
        audit_ok, audit_errors = verify_audit_chain(ccr_dir)
        if audit_ok:
            report.checks.append("governance audit chain ok")
        else:
            report.ok = False
            report.issues.extend(audit_errors)
    except Exception as exc:
        report.ok = False
        report.issues.append(f"governance audit verification failed: {type(exc).__name__}: {exc}")
    try:
        from ccr.core.replay import RecallTraceStore
        replay_ok, replay_errors = RecallTraceStore(ccr_dir).verify_chain()
        if replay_ok:
            report.checks.append("recall trace chain ok")
        else:
            report.ok = False
            report.issues.extend(replay_errors)
    except Exception as exc:
        report.ok = False
        report.issues.append(f"recall trace verification failed: {type(exc).__name__}: {exc}")
    enc_policy = os.path.join(ccr_dir, "encryption", "policy.json")
    if os.path.isfile(enc_policy):
        try:
            from ccr.core.encryption import EncryptionManager
            status = EncryptionManager(ccr_dir).status()
            report.checks.extend(f"encryption {check}" for check in status.checks)
            if not status.ok:
                report.ok = False
                report.issues.extend(status.issues)
        except Exception as exc:
            report.ok = False
            report.issues.append(f"encryption policy verification failed: {type(exc).__name__}: {exc}")
    for db_name in ("memory.db", "index.db", "sessions.db"):
        path = os.path.join(ccr_dir, db_name)
        if not os.path.isfile(path):
            continue
        try:
            conn = sqlite3.connect(path)
            row = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            if row and row[0] == "ok":
                report.checks.append(f"{db_name} integrity ok")
            else:
                report.ok = False
                report.issues.append(f"{db_name} integrity failed: {row[0] if row else 'unknown'}")
        except sqlite3.DatabaseError as exc:
            report.ok = False
            report.issues.append(f"{db_name} unreadable: {exc}")
    return report


def repair_project(project: str) -> OpsReport:
    ccr_dir = _ccr_dir(project)
    if not os.path.isdir(ccr_dir):
        os.makedirs(ccr_dir, exist_ok=True)
    report = verify_project(project)
    for subdir in ("branches", "summaries", "reports"):
        os.makedirs(os.path.join(ccr_dir, subdir), exist_ok=True)
    report.checks.append("required directories ensured")
    report.artifacts.append(ccr_dir)
    report.ok = not report.issues
    return report


def migrate_project(project: str) -> OpsReport:
    ccr_dir = _ccr_dir(project)
    if not os.path.isdir(ccr_dir):
        return OpsReport(False, issues=[".ccr directory not found"])
    db_path = os.path.join(ccr_dir, "memory.db")
    try:
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        from ccr.core.storage.migration import auto_migrate
        backend = SqliteStorageBackend(ccr_dir)
        backend.close()
        result = auto_migrate(ccr_dir, db_path)
        ok = not result.get("errors")
        return OpsReport(
            ok,
            checks=[f"migrated={result.get('total_migrated', 0)}"],
            issues=list(result.get("errors") or []),
            artifacts=[db_path],
        )
    except Exception as exc:
        return OpsReport(False, issues=[f"{type(exc).__name__}: {exc}"])


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
