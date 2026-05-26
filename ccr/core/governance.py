"""Local-first governance, redaction, and audit helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ccr.core.events import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)['\"]?\b([a-z0-9_ -]*(api[_-]?key|token|secret|password)[a-z0-9_ -]*)\b['\"]?\s*[:=]\s*['\"]?[^'\"\s]{8,}"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("us_phone", re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
]


@dataclass
class GovernanceFinding:
    kind: str
    path: str
    line: int
    column: int
    match: str
    severity: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GovernancePolicy:
    roles: dict[str, list[str]] = field(default_factory=lambda: {
        "reader": ["recall", "context", "export_redacted"],
        "writer": ["commit", "fact_write", "sync_push"],
        "admin": ["*"],
    })
    retention_days: int = 365
    redact_on_export: bool = True
    approved_tools: list[str] = field(default_factory=lambda: ["gcc_*", "ace_*", "rlm_*", "index_*"])
    project_boundary: str = ""
    require_approval_for: list[str] = field(default_factory=lambda: ["inferred", "speculative"])
    encryption_mode: str = "none"
    mcp_auth_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GovernancePolicy":
        return cls(
            roles=dict(data.get("roles") or cls().roles),
            retention_days=int(data.get("retention_days", 365)),
            redact_on_export=bool(data.get("redact_on_export", True)),
            approved_tools=list(data.get("approved_tools") or cls().approved_tools),
            project_boundary=str(data.get("project_boundary", "")),
            require_approval_for=list(data.get("require_approval_for") or cls().require_approval_for),
            encryption_mode=str(data.get("encryption_mode") or "none"),
            mcp_auth_required=bool(data.get("mcp_auth_required", False)),
        )


def policy_path(ccr_root: str) -> str:
    return os.path.join(ccr_root, "governance.json")


def load_policy(ccr_root: str) -> GovernancePolicy:
    path = policy_path(ccr_root)
    if not os.path.isfile(path):
        return GovernancePolicy(project_boundary=os.path.abspath(os.path.dirname(ccr_root)))
    try:
        with open(path, encoding="utf-8") as fh:
            return GovernancePolicy.from_dict(json.loads(fh.read() or "{}"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return GovernancePolicy(project_boundary=os.path.abspath(os.path.dirname(ccr_root)))


def save_policy(ccr_root: str, policy: GovernancePolicy) -> str:
    path = policy_path(ccr_root)
    atomic_write_json(path, {"version": 1, **policy.to_dict()})
    return path


def scan_text(text: str, path: str = "<memory>") -> list[GovernanceFinding]:
    findings: list[GovernanceFinding] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for kind, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(line):
                findings.append(GovernanceFinding(
                    kind=kind,
                    path=path,
                    line=line_no,
                    column=match.start() + 1,
                    match=match.group(0),
                    severity="medium" if kind in {"email", "us_phone"} else "high",
                ))
    return findings


def redact_text(text: str) -> str:
    redacted = text
    for kind, pattern in SECRET_PATTERNS:
        redacted = pattern.sub(f"[REDACTED:{kind}]", redacted)
    return redacted


def redact_data(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_data(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_data(v) for k, v in value.items()}
    return value


def scan_memory_tree(ccr_root: str) -> list[GovernanceFinding]:
    findings: list[GovernanceFinding] = []
    if not os.path.isdir(ccr_root):
        return findings
    allowed_ext = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".toml"}
    for dirpath, dirnames, filenames in os.walk(ccr_root):
        dirnames[:] = [d for d in dirnames if d not in {"backups", "sync"}]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in allowed_ext:
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            findings.extend(scan_text(text, os.path.relpath(path, ccr_root)))
    return findings


def append_audit(ccr_root: str, event_type: str, actor: str, details: dict[str, Any]) -> dict[str, Any]:
    """Append a hash-chained audit entry."""
    try:
        ccr_root = os.fspath(ccr_root)
    except TypeError:
        return {
            "timestamp": utc_now(),
            "event_type": event_type,
            "actor": actor or os.environ.get("USER", "unknown"),
            "details": details,
            "skipped": True,
            "reason": "invalid_ccr_root",
        }
    if not isinstance(ccr_root, str) or not ccr_root.strip():
        return {
            "timestamp": utc_now(),
            "event_type": event_type,
            "actor": actor or os.environ.get("USER", "unknown"),
            "details": details,
            "skipped": True,
            "reason": "invalid_ccr_root",
        }
    if os.path.basename(os.path.normpath(ccr_root)) != ".ccr":
        return {
            "timestamp": utc_now(),
            "event_type": event_type,
            "actor": actor or os.environ.get("USER", "unknown"),
            "details": details,
            "skipped": True,
            "reason": "non_ccr_root",
        }
    os.makedirs(ccr_root, exist_ok=True)
    path = os.path.join(ccr_root, "governance_audit.jsonl")
    prev_hash = ""
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        prev_hash = json.loads(line).get("hash", prev_hash)
        except (OSError, json.JSONDecodeError):
            prev_hash = ""
    entry = {
        "timestamp": utc_now(),
        "event_type": event_type,
        "actor": actor or os.environ.get("USER", "unknown"),
        "details": details,
        "prev_hash": prev_hash,
    }
    digest_input = json.dumps(entry, sort_keys=True).encode("utf-8")
    entry["hash"] = hashlib.sha256(digest_input).hexdigest()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True))
        fh.write("\n")
    return entry


def verify_audit_chain(ccr_root: str) -> tuple[bool, list[str]]:
    """Verify the local governance audit hash chain."""
    path = os.path.join(ccr_root, "governance_audit.jsonl")
    if not os.path.isfile(path):
        return True, []
    errors: list[str] = []
    prev_hash = ""
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"governance_audit.jsonl line {line_no}: invalid json: {exc}")
                continue
            if entry.get("prev_hash", "") != prev_hash:
                errors.append(f"governance_audit.jsonl line {line_no}: prev_hash mismatch")
            recorded_hash = str(entry.get("hash", ""))
            body = dict(entry)
            body.pop("hash", None)
            expected = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
            if recorded_hash != expected:
                errors.append(f"governance_audit.jsonl line {line_no}: hash mismatch")
            prev_hash = recorded_hash
    return not errors, errors


def signed_manifest(path: str) -> dict[str, str]:
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    manifest = {
        "path": os.path.abspath(path),
        "sha256": digest,
        "signed_at": utc_now(),
        "signature": digest,
        "signature_scheme": "sha256-local-manifest",
    }
    manifest_path = path + ".manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest
