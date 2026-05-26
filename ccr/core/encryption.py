"""Real at-rest encryption for local CCR memory artifacts.

CCR keeps memory local and inspectable.  Encryption therefore has explicit
lock/unlock semantics instead of pretending plaintext files can stay readable
and also be encrypted at rest.  A locked project stores selected memory files as
authenticated AES-256-GCM envelopes and removes their plaintext counterparts.

Key-management model:
- passphrase: derive a 256-bit key with PBKDF2-HMAC-SHA256.  The passphrase is
  supplied at operation time via CLI/env and is never stored.
- env: load a raw 32-byte base64 key from an explicit environment variable.
- keyfile: load a raw 32-byte base64 key from an explicit file path.

The policy stores only non-secret key metadata: key source, salt, KDF settings,
and a key fingerprint.  The key itself is never written to .ccr/.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ccr.core.events import atomic_write_json
from ccr.core.governance import append_audit


ALGORITHM = "AES-256-GCM"
KDF = "PBKDF2-HMAC-SHA256"
RAW_KEY_KDF = "raw-base64"
DEFAULT_ITERATIONS = 600_000
DEFAULT_PASSPHRASE_ENV = "CCR_ENCRYPTION_PASSPHRASE"
DEFAULT_RAW_KEY_ENV = "CCR_ENCRYPTION_KEY"
POLICY_REL_PATH = os.path.join("encryption", "policy.json")

DEFAULT_PROTECTED_FILES = [
    "memory.db",
    "memory.db-wal",
    "memory.db-shm",
    "index.db",
    "index.db-wal",
    "index.db-shm",
    "sessions.db",
    "sessions.db-wal",
    "sessions.db-shm",
    "facts.json",
    "episodes.jsonl",
    "recall_traces.jsonl",
    "quarantine.json",
    "temporal_graph.json",
    "scratchpad.json",
    "agent_events.jsonl",
    "playbook.txt",
    "failure_lessons.json",
    "metadata.yaml",
    "summary_meta.yaml",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key_id(key: bytes) -> str:
    return _sha256(key)[:24]


def _atomic_write_bytes(path: str, data: bytes, mode: int | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".enc.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@dataclass
class EncryptionPolicy:
    """Non-secret encryption policy and key metadata."""

    enabled: bool = True
    state: str = "unlocked"
    algorithm: str = ALGORITHM
    key_source: str = "passphrase"
    env_var: str = DEFAULT_PASSPHRASE_ENV
    key_file: str = ""
    kdf: str = KDF
    iterations: int = DEFAULT_ITERATIONS
    salt: str = ""
    key_id: str = ""
    protected_paths: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncryptionPolicy":
        return cls(
            enabled=bool(data.get("enabled", True)),
            state=str(data.get("state") or "unlocked"),
            algorithm=str(data.get("algorithm") or ALGORITHM),
            key_source=str(data.get("key_source") or "passphrase"),
            env_var=str(data.get("env_var") or DEFAULT_PASSPHRASE_ENV),
            key_file=str(data.get("key_file", "")),
            kdf=str(data.get("kdf") or KDF),
            iterations=int(data.get("iterations") or DEFAULT_ITERATIONS),
            salt=str(data.get("salt", "")),
            key_id=str(data.get("key_id", "")),
            protected_paths=list(data.get("protected_paths") or []),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            version=int(data.get("version") or 1),
        )


@dataclass
class EncryptionReport:
    """Outcome from an encryption operation."""

    ok: bool
    action: str
    checks: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EncryptionError(RuntimeError):
    """Raised for encryption/key-management failures."""


class EncryptionManager:
    """Manage CCR encryption policy and lock/unlock lifecycle."""

    def __init__(self, ccr_root: str):
        self.ccr_root = os.path.abspath(ccr_root)
        self.policy_path = os.path.join(self.ccr_root, POLICY_REL_PATH)

    def init_policy(
        self,
        *,
        key_source: str = "passphrase",
        env_var: str = "",
        key_file: str = "",
        passphrase: str = "",
        scope: str = "default",
        iterations: int = DEFAULT_ITERATIONS,
        overwrite: bool = False,
    ) -> EncryptionReport:
        """Create or replace encryption policy after proving key availability."""
        if os.path.isfile(self.policy_path) and not overwrite:
            return EncryptionReport(
                ok=False,
                action="init",
                issues=["encryption policy already exists; pass overwrite=True to replace it"],
                artifacts=[self.policy_path],
            )
        key_source = normalize_key_source(key_source)
        salt = secrets.token_bytes(16) if key_source == "passphrase" else b""
        policy = EncryptionPolicy(
            state="unlocked",
            key_source=key_source,
            env_var=resolve_env_var(key_source, env_var),
            key_file=os.path.abspath(key_file) if key_file else "",
            kdf=KDF if key_source == "passphrase" else RAW_KEY_KDF,
            iterations=max(100_000, int(iterations)),
            salt=_b64(salt),
            protected_paths=self.discover_protected_paths(scope=scope),
        )
        key = self._load_key(policy, passphrase=passphrase)
        policy.key_id = _key_id(key)
        policy.updated_at = utc_now()
        atomic_write_json(self.policy_path, policy.to_dict())
        self._sync_governance_policy(policy)
        append_audit(
            self.ccr_root,
            "encryption_policy_init",
            "cli",
            {
                "key_source": policy.key_source,
                "key_id": policy.key_id,
                "protected_paths": len(policy.protected_paths),
            },
        )
        return EncryptionReport(
            ok=True,
            action="init",
            checks=[
                f"policy initialized with {policy.algorithm}",
                f"key source: {policy.key_source}",
                f"protected paths: {len(policy.protected_paths)}",
            ],
            artifacts=[self.policy_path],
        )

    def load_policy(self) -> EncryptionPolicy:
        if not os.path.isfile(self.policy_path):
            raise EncryptionError("encryption policy not initialized")
        try:
            with open(self.policy_path, encoding="utf-8") as fh:
                data = json.loads(fh.read() or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            raise EncryptionError(f"could not load encryption policy: {exc}") from exc
        policy = EncryptionPolicy.from_dict(data)
        if policy.algorithm != ALGORITHM:
            raise EncryptionError(f"unsupported encryption algorithm: {policy.algorithm}")
        return policy

    def status(self) -> EncryptionReport:
        """Return policy status without requiring the key."""
        if not os.path.isfile(self.policy_path):
            return EncryptionReport(ok=False, action="status", issues=["encryption policy not initialized"])
        policy = self.load_policy()
        plaintext = []
        ciphertext = []
        missing = []
        for rel in policy.protected_paths:
            plain = self._abs(rel)
            enc = self._enc_path(rel)
            if os.path.isfile(plain):
                plaintext.append(rel)
            if os.path.isfile(enc):
                ciphertext.append(rel)
            if not os.path.isfile(plain) and not os.path.isfile(enc):
                missing.append(rel)
        issues = []
        if policy.state == "locked" and plaintext:
            issues.append(f"{len(plaintext)} protected plaintext file(s) exist while policy is locked")
        return EncryptionReport(
            ok=not issues,
            action="status",
            checks=[
                f"state: {policy.state}",
                f"algorithm: {policy.algorithm}",
                f"key source: {policy.key_source}",
                f"key id: {policy.key_id}",
                f"plaintext files: {len(plaintext)}",
                f"encrypted files: {len(ciphertext)}",
                f"missing optional files: {len(missing)}",
            ],
            issues=issues,
            artifacts=[self.policy_path],
        )

    def lock(self, *, passphrase: str = "") -> EncryptionReport:
        """Encrypt protected plaintext files and remove plaintext."""
        policy = self.load_policy()
        key = self._load_key(policy, passphrase=passphrase)
        self._assert_key_matches(policy, key)
        encrypted: list[str] = []
        skipped: list[str] = []
        for rel in policy.protected_paths:
            plain = self._abs(rel)
            enc = self._enc_path(rel)
            if not os.path.isfile(plain):
                if os.path.isfile(enc):
                    skipped.append(rel)
                continue
            envelope = encrypt_bytes(plain, rel, key)
            _atomic_write_bytes(enc, json.dumps(envelope, sort_keys=True).encode("utf-8") + b"\n")
            os.unlink(plain)
            encrypted.append(rel)
        policy.state = "locked"
        policy.updated_at = utc_now()
        self._save_policy(policy)
        append_audit(
            self.ccr_root,
            "encryption_lock",
            "cli",
            {"encrypted": encrypted, "skipped": skipped, "key_id": policy.key_id},
        )
        return EncryptionReport(
            ok=True,
            action="lock",
            checks=[f"encrypted {len(encrypted)} file(s)", f"already locked/skipped {len(skipped)} file(s)"],
            artifacts=[self._enc_path(rel) for rel in encrypted],
        )

    def unlock(self, *, passphrase: str = "", keep_encrypted: bool = False) -> EncryptionReport:
        """Decrypt protected envelope files back to plaintext."""
        policy = self.load_policy()
        key = self._load_key(policy, passphrase=passphrase)
        self._assert_key_matches(policy, key)
        restored: list[str] = []
        skipped: list[str] = []
        for rel in policy.protected_paths:
            enc = self._enc_path(rel)
            plain = self._abs(rel)
            if not os.path.isfile(enc):
                if os.path.isfile(plain):
                    skipped.append(rel)
                continue
            plaintext = decrypt_envelope_file(enc, rel, key)
            _atomic_write_bytes(plain, plaintext)
            if not keep_encrypted:
                os.unlink(enc)
            restored.append(rel)
        policy.state = "unlocked"
        policy.updated_at = utc_now()
        self._save_policy(policy)
        append_audit(
            self.ccr_root,
            "encryption_unlock",
            "cli",
            {"restored": restored, "skipped": skipped, "key_id": policy.key_id},
        )
        return EncryptionReport(
            ok=True,
            action="unlock",
            checks=[f"restored {len(restored)} file(s)", f"already plaintext/skipped {len(skipped)} file(s)"],
            artifacts=[self._abs(rel) for rel in restored],
        )

    def verify(self, *, passphrase: str = "") -> EncryptionReport:
        """Verify encrypted envelopes and locked-state invariants."""
        policy = self.load_policy()
        key = self._load_key(policy, passphrase=passphrase)
        self._assert_key_matches(policy, key)
        verified: list[str] = []
        plaintext_while_locked: list[str] = []
        issues: list[str] = []
        for rel in policy.protected_paths:
            plain = self._abs(rel)
            enc = self._enc_path(rel)
            if os.path.isfile(enc):
                try:
                    decrypt_envelope_file(enc, rel, key)
                    verified.append(rel)
                except EncryptionError as exc:
                    issues.append(f"{rel}: {exc}")
            if policy.state == "locked" and os.path.isfile(plain):
                plaintext_while_locked.append(rel)
        if plaintext_while_locked:
            issues.append(
                f"locked policy has plaintext files: {', '.join(plaintext_while_locked[:5])}"
            )
        append_audit(
            self.ccr_root,
            "encryption_verify",
            "cli",
            {"verified": len(verified), "issues": len(issues), "key_id": policy.key_id},
        )
        return EncryptionReport(
            ok=not issues,
            action="verify",
            checks=[f"verified {len(verified)} encrypted envelope(s)", f"state: {policy.state}"],
            issues=issues,
        )

    def rotate(
        self,
        *,
        new_key_source: str,
        new_env_var: str = "",
        new_key_file: str = "",
        old_passphrase: str = "",
        new_passphrase: str = "",
    ) -> EncryptionReport:
        """Rotate key metadata and re-encrypt locked envelopes with a new key."""
        old_policy = self.load_policy()
        old_key = self._load_key(old_policy, passphrase=old_passphrase)
        self._assert_key_matches(old_policy, old_key)
        plaintext_by_rel: dict[str, bytes] = {}
        for rel in old_policy.protected_paths:
            enc = self._enc_path(rel)
            plain = self._abs(rel)
            if os.path.isfile(enc):
                plaintext_by_rel[rel] = decrypt_envelope_file(enc, rel, old_key)
            elif os.path.isfile(plain):
                with open(plain, "rb") as fh:
                    plaintext_by_rel[rel] = fh.read()

        new_key_source = normalize_key_source(new_key_source)
        new_policy = EncryptionPolicy(
            state=old_policy.state,
            key_source=new_key_source,
            env_var=resolve_env_var(new_key_source, new_env_var),
            key_file=os.path.abspath(new_key_file) if new_key_file else "",
            kdf=KDF if new_key_source == "passphrase" else RAW_KEY_KDF,
            iterations=old_policy.iterations,
            salt=_b64(secrets.token_bytes(16)) if new_key_source == "passphrase" else "",
            protected_paths=list(old_policy.protected_paths),
            created_at=old_policy.created_at,
            updated_at=utc_now(),
        )
        new_key = self._load_key(new_policy, passphrase=new_passphrase)
        new_policy.key_id = _key_id(new_key)
        reencrypted: list[str] = []
        if old_policy.state == "locked":
            for rel, plaintext in plaintext_by_rel.items():
                enc = self._enc_path(rel)
                envelope = make_envelope(rel, plaintext, new_key)
                _atomic_write_bytes(enc, json.dumps(envelope, sort_keys=True).encode("utf-8") + b"\n")
                plain = self._abs(rel)
                if os.path.isfile(plain):
                    os.unlink(plain)
                reencrypted.append(rel)
        self._save_policy(new_policy)
        append_audit(
            self.ccr_root,
            "encryption_rotate",
            "cli",
            {
                "old_key_id": old_policy.key_id,
                "new_key_id": new_policy.key_id,
                "state": new_policy.state,
                "reencrypted": len(reencrypted),
            },
        )
        return EncryptionReport(
            ok=True,
            action="rotate",
            checks=[f"rotated {old_policy.key_id} -> {new_policy.key_id}", f"reencrypted {len(reencrypted)} file(s)"],
            artifacts=[self.policy_path],
        )

    def discover_protected_paths(self, scope: str = "default") -> list[str]:
        """Discover protected paths relative to .ccr/."""
        scope = scope.strip().lower() or "default"
        if scope not in {"default", "all"}:
            raise EncryptionError("scope must be one of: default, all")
        if scope == "default":
            return [rel for rel in DEFAULT_PROTECTED_FILES if os.path.isfile(self._abs(rel))]
        paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.ccr_root):
            dirnames[:] = [
                d for d in dirnames
                if d not in {"backups", "sync", "browser", "reports", "encryption"}
            ]
            for name in filenames:
                if name.endswith(".ccrenc"):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, self.ccr_root)
                if rel in {"governance.json", POLICY_REL_PATH}:
                    continue
                paths.append(rel)
        return sorted(paths)

    def _save_policy(self, policy: EncryptionPolicy) -> None:
        policy.updated_at = utc_now()
        atomic_write_json(self.policy_path, policy.to_dict())
        self._sync_governance_policy(policy)

    def _sync_governance_policy(self, policy: EncryptionPolicy) -> None:
        try:
            from ccr.core.governance import load_policy, save_policy
            gov = load_policy(self.ccr_root)
            gov.encryption_mode = f"{policy.algorithm}:{policy.state}"
            save_policy(self.ccr_root, gov)
        except Exception:
            pass

    def _load_key(self, policy: EncryptionPolicy, *, passphrase: str = "") -> bytes:
        source = normalize_key_source(policy.key_source)
        if source == "passphrase":
            secret = passphrase or os.environ.get(policy.env_var, "")
            if not secret:
                raise EncryptionError(
                    f"passphrase required; provide --passphrase or set {policy.env_var}"
                )
            salt = _unb64(policy.salt)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=policy.iterations,
            )
            return kdf.derive(secret.encode("utf-8"))
        if source == "env":
            raw = os.environ.get(policy.env_var, "")
            if not raw:
                raise EncryptionError(f"raw encryption key required in {policy.env_var}")
            return parse_raw_key(raw)
        if source == "keyfile":
            if not policy.key_file:
                raise EncryptionError("key_file is required for keyfile key source")
            try:
                with open(policy.key_file, encoding="utf-8") as fh:
                    raw = fh.read().strip()
            except OSError as exc:
                raise EncryptionError(f"could not read key file: {exc}") from exc
            return parse_raw_key(raw)
        raise EncryptionError(f"unsupported key source: {source}")

    @staticmethod
    def _assert_key_matches(policy: EncryptionPolicy, key: bytes) -> None:
        if policy.key_id and _key_id(key) != policy.key_id:
            raise EncryptionError("provided key does not match encryption policy key_id")

    def _abs(self, rel: str) -> str:
        rel = rel.strip().lstrip(os.sep)
        return os.path.join(self.ccr_root, rel)

    def _enc_path(self, rel: str) -> str:
        return self._abs(rel) + ".ccrenc"


def normalize_key_source(value: str) -> str:
    normalized = (value or "passphrase").strip().lower().replace("-", "_")
    aliases = {"raw": "env", "environment": "env", "file": "keyfile", "key_file": "keyfile"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"passphrase", "env", "keyfile"}:
        raise EncryptionError("key_source must be one of: passphrase, env, keyfile")
    return normalized


def resolve_env_var(key_source: str, env_var: str) -> str:
    key_source = normalize_key_source(key_source)
    if env_var.strip():
        return env_var.strip()
    if key_source == "env":
        return DEFAULT_RAW_KEY_ENV
    return DEFAULT_PASSPHRASE_ENV


def parse_raw_key(value: str) -> bytes:
    try:
        key = _unb64(value.strip())
    except Exception as exc:
        raise EncryptionError("raw key must be base64-encoded") from exc
    if len(key) != 32:
        raise EncryptionError("raw key must decode to exactly 32 bytes for AES-256-GCM")
    return key


def generate_raw_key() -> str:
    """Return a base64-encoded 256-bit random key."""
    return _b64(secrets.token_bytes(32))


def write_key_file(path: str, *, overwrite: bool = False) -> str:
    """Write a new base64 raw key to ``path`` with 0600 permissions."""
    path = os.path.abspath(path)
    if os.path.exists(path) and not overwrite:
        raise EncryptionError(f"key file already exists: {path}")
    _atomic_write_bytes(path, (generate_raw_key() + "\n").encode("ascii"), mode=0o600)
    return path


def encrypt_bytes(path: str, rel: str, key: bytes) -> dict[str, Any]:
    with open(path, "rb") as fh:
        plaintext = fh.read()
    return make_envelope(rel, plaintext, key)


def make_envelope(rel: str, plaintext: bytes, key: bytes) -> dict[str, Any]:
    nonce = secrets.token_bytes(12)
    aad = rel.encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    return {
        "version": 1,
        "algorithm": ALGORITHM,
        "key_id": _key_id(key),
        "nonce": _b64(nonce),
        "aad": rel,
        "plaintext_sha256": _sha256(plaintext),
        "ciphertext": _b64(ciphertext),
    }


def decrypt_envelope_file(path: str, rel: str, key: bytes) -> bytes:
    try:
        with open(path, encoding="utf-8") as fh:
            envelope = json.loads(fh.read() or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        raise EncryptionError(f"invalid encrypted envelope: {exc}") from exc
    return decrypt_envelope(envelope, rel, key)


def decrypt_envelope(envelope: dict[str, Any], rel: str, key: bytes) -> bytes:
    if envelope.get("algorithm") != ALGORITHM:
        raise EncryptionError(f"unsupported envelope algorithm: {envelope.get('algorithm')}")
    if str(envelope.get("aad", "")) != rel:
        raise EncryptionError("encrypted envelope path binding does not match")
    if str(envelope.get("key_id", "")) != _key_id(key):
        raise EncryptionError("provided key does not match encrypted envelope key_id")
    try:
        nonce = _unb64(str(envelope.get("nonce", "")))
        ciphertext = _unb64(str(envelope.get("ciphertext", "")))
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, rel.encode("utf-8"))
    except (InvalidTag, ValueError) as exc:
        raise EncryptionError("encrypted envelope authentication failed") from exc
    expected = str(envelope.get("plaintext_sha256", ""))
    if expected and _sha256(plaintext) != expected:
        raise EncryptionError("decrypted plaintext digest mismatch")
    return plaintext


def is_encrypted_artifact(path: str) -> bool:
    return bool(re.search(r"\.ccrenc$", path))
