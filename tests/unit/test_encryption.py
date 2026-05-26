"""Tests for explicit CCR at-rest encryption semantics."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from ccr.cli import cli
from ccr.core.encryption import (
    EncryptionError,
    EncryptionManager,
    generate_raw_key,
)
from ccr.core.governance import load_policy


def test_passphrase_lock_unlock_roundtrip_and_wrong_key(tmp_path):
    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    original = '{"version":1,"facts":[{"statement":"secret token value"}]}\n'
    (ccr_root / "facts.json").write_text(original)

    manager = EncryptionManager(str(ccr_root))
    init = manager.init_policy(passphrase="correct horse battery staple")
    locked = manager.lock(passphrase="correct horse battery staple")
    encrypted_path = ccr_root / "facts.json.ccrenc"

    assert init.ok
    assert locked.ok
    assert not (ccr_root / "facts.json").exists()
    assert encrypted_path.exists()
    assert "secret token value" not in encrypted_path.read_text()
    assert manager.verify(passphrase="correct horse battery staple").ok

    with pytest.raises(EncryptionError, match="provided key does not match"):
        manager.unlock(passphrase="wrong passphrase")
    assert not (ccr_root / "facts.json").exists()

    unlocked = manager.unlock(passphrase="correct horse battery staple")
    assert unlocked.ok
    assert (ccr_root / "facts.json").read_text() == original
    assert not encrypted_path.exists()
    assert manager.status().ok


def test_env_key_rotation_invalidates_old_key(tmp_path, monkeypatch):
    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    (ccr_root / "facts.json").write_text('{"version":1,"facts":[{"statement":"env secret"}]}\n')
    old_key = generate_raw_key()
    new_key = generate_raw_key()
    monkeypatch.setenv("CCR_OLD_KEY", old_key)
    monkeypatch.setenv("CCR_NEW_KEY", new_key)

    manager = EncryptionManager(str(ccr_root))
    assert manager.init_policy(key_source="env", env_var="CCR_OLD_KEY").ok
    assert manager.lock().ok
    old_policy = manager.load_policy()

    assert old_policy.key_source == "env"
    assert old_policy.env_var == "CCR_OLD_KEY"
    assert manager.verify().ok

    rotated = manager.rotate(new_key_source="env", new_env_var="CCR_NEW_KEY")
    new_policy = manager.load_policy()

    assert rotated.ok
    assert new_policy.env_var == "CCR_NEW_KEY"
    assert new_policy.key_id != old_policy.key_id
    assert manager.verify().ok

    monkeypatch.setenv("CCR_NEW_KEY", old_key)
    with pytest.raises(EncryptionError, match="provided key does not match"):
        manager.verify()


def test_encryption_cli_keyfile_lock_verify_unlock(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    project.mkdir()
    init = runner.invoke(cli, ["init", str(project)])
    assert init.exit_code == 0, init.output
    facts = project / ".ccr" / "facts.json"
    facts.write_text(json.dumps({"version": 1, "facts": [{"statement": "stored secret"}]}) + "\n")
    key_file = tmp_path / "ccr.key"

    keygen = runner.invoke(cli, ["encryption", "keygen", "--output", str(key_file)])
    enc_init = runner.invoke(
        cli,
        ["encryption", "init", str(project), "--key-source", "keyfile", "--key-file", str(key_file)],
    )
    lock = runner.invoke(cli, ["encryption", "lock", str(project)])
    verify = runner.invoke(cli, ["encryption", "verify", str(project)])

    assert keygen.exit_code == 0, keygen.output
    assert key_file.exists()
    assert oct(key_file.stat().st_mode & 0o777) == "0o600"
    assert enc_init.exit_code == 0, enc_init.output
    assert lock.exit_code == 0, lock.output
    assert "encrypted" in lock.output
    assert not facts.exists()
    assert (project / ".ccr" / "facts.json.ccrenc").exists()
    assert verify.exit_code == 0, verify.output
    unlock = runner.invoke(cli, ["encryption", "unlock", str(project)])
    assert unlock.exit_code == 0, unlock.output
    assert facts.exists()
    assert "stored secret" in facts.read_text()

    policy = load_policy(str(project / ".ccr"))
    assert policy.encryption_mode == "AES-256-GCM:unlocked"


def test_encryption_cli_init_requires_key_material(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    project.mkdir()
    assert runner.invoke(cli, ["init", str(project)]).exit_code == 0

    result = runner.invoke(cli, ["encryption", "init", str(project), "--key-source", "env"])

    assert result.exit_code == 1
    assert "raw encryption key required" in result.output
