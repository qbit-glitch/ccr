"""Industry-adoption CLI commands for CCR."""

from __future__ import annotations

import json
import os
import sys

import click

from ccr.core.types import CCRConfig


def _memory(project: str):
    from ccr.core.memory import MemoryManager
    project = os.path.abspath(project)
    storage_backend = os.environ.get("CCR_STORAGE_BACKEND", CCRConfig().storage_backend)
    return MemoryManager(project, CCRConfig(storage_backend=storage_backend))


def _require_ccr(project: str) -> str:
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")
    if not os.path.isdir(ccr_dir):
        click.echo("No .ccr/ found. Run 'ccr init' first.", err=True)
        sys.exit(1)
    return project


def _print_ops(report) -> None:
    for check in report.checks:
        click.echo(f"  [OK] {check}")
    for issue in report.issues:
        click.echo(f"  [!!] {issue}")
    for artifact in report.artifacts:
        click.echo(f"  [artifact] {artifact}")
    if not report.ok:
        sys.exit(1)


@click.group("episodes")
def episodes() -> None:
    """Inspect and verify immutable episode/evidence records."""


@episodes.command("list")
@click.argument("project", default=".")
@click.option("--query", default="", help="Optional query filter.")
@click.option("--limit", default=25, type=int, help="Maximum episodes to print.")
def episodes_list(project: str, query: str, limit: int) -> None:
    from ccr.core.episodes import EpisodeStore

    project = _require_ccr(project)
    store = EpisodeStore(os.path.join(project, ".ccr"))
    rows = store.list_episodes(query=query, limit=limit)
    if not rows:
        click.echo("No episodes found.")
        return
    for row in rows:
        click.echo(f"{row.id} {row.created_at} {row.event_type}: {row.summary or row.content[:120]}")


@episodes.command("verify")
@click.argument("project", default=".")
@click.option("--json-output", is_flag=True, help="Print JSON instead of text.")
def episodes_verify(project: str, json_output: bool) -> None:
    from ccr.core.episodes import EpisodeStore

    project = _require_ccr(project)
    result = EpisodeStore(os.path.join(project, ".ccr")).verify_chain()
    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(f"Episodes checked: {result.checked}")
        click.echo("Episode chain OK" if result.ok else "Episode chain FAILED")
        for error in result.errors:
            click.echo(f"[!!] {error}")
    if not result.ok:
        sys.exit(1)


@click.group("quarantine")
def quarantine() -> None:
    """Review, promote, or reject quarantined memory candidates."""


@quarantine.command("list")
@click.argument("project", default=".")
@click.option("--status", default="pending",
              type=click.Choice(["pending", "promoted", "rejected", "all"]),
              help="Quarantine status to list.")
@click.option("--json-output", is_flag=True, help="Print JSON instead of text.")
def quarantine_list(project: str, status: str, json_output: bool) -> None:
    from ccr.core.quarantine import MemoryQuarantine

    project = _require_ccr(project)
    rows = MemoryQuarantine(os.path.join(project, ".ccr")).list_items(status=status, limit=200)
    if json_output:
        click.echo(json.dumps([r.to_dict() for r in rows], indent=2))
        return
    if not rows:
        click.echo("No quarantine items found.")
        return
    for row in rows:
        click.echo(
            f"{row.id} [{row.status}] {row.classification} `{row.key}` "
            f"conf={row.confidence:.2f}: {row.statement}"
        )


@quarantine.command("promote")
@click.argument("project", default=".")
@click.argument("item_id")
def quarantine_promote(project: str, item_id: str) -> None:
    from ccr.core.episodes import EpisodeStore
    from ccr.core.facts import FactLedger
    from ccr.core.quarantine import MemoryQuarantine

    project = _require_ccr(project)
    ccr_root = os.path.join(project, ".ccr")
    item = MemoryQuarantine(ccr_root).promote(item_id, FactLedger(ccr_root))
    if not item:
        click.echo(f"Quarantine item {item_id} not found or not pending.", err=True)
        sys.exit(1)
    episode = EpisodeStore(ccr_root).append_episode(
        "quarantine_promoted",
        summary=f"Promoted quarantine item {item.id} to fact {item.promoted_fact_id}",
        content=item.statement,
        source_ids=[item.id, item.promoted_fact_id],
        tools=["ccr quarantine promote"],
        metadata={"classification": item.classification, "key": item.key},
    )
    click.echo(f"Promoted {item.id} -> {item.promoted_fact_id}; episode {episode.id}")


@quarantine.command("reject")
@click.argument("project", default=".")
@click.argument("item_id")
@click.option("--reason", default="", help="Reason for rejection.")
def quarantine_reject(project: str, item_id: str, reason: str) -> None:
    from ccr.core.episodes import EpisodeStore
    from ccr.core.quarantine import MemoryQuarantine

    project = _require_ccr(project)
    ccr_root = os.path.join(project, ".ccr")
    item = MemoryQuarantine(ccr_root).reject(item_id, reason=reason)
    if not item:
        click.echo(f"Quarantine item {item_id} not found or not pending.", err=True)
        sys.exit(1)
    episode = EpisodeStore(ccr_root).append_episode(
        "quarantine_rejected",
        summary=f"Rejected quarantine item {item.id}",
        content=reason,
        source_ids=[item.id],
        tools=["ccr quarantine reject"],
        metadata={"classification": item.classification, "key": item.key},
    )
    click.echo(f"Rejected {item.id}; episode {episode.id}")


@click.group("conflicts")
def conflicts() -> None:
    """List and resolve temporal fact conflicts."""


@conflicts.command("list")
@click.argument("project", default=".")
@click.option("--query", default="", help="Optional query/key filter.")
def conflicts_list(project: str, query: str) -> None:
    from ccr.core.facts import FactLedger

    project = _require_ccr(project)
    conflicts_found = FactLedger(os.path.join(project, ".ccr")).detect_conflicts(query=query, limit=200)
    if not conflicts_found:
        click.echo("No fact conflicts found.")
        return
    for conflict in conflicts_found:
        click.echo(
            f"{conflict.severity} `{conflict.key}`: "
            f"{conflict.fact_a} vs {conflict.fact_b} ({conflict.reason})"
        )


@conflicts.command("resolve")
@click.argument("project", default=".")
@click.option("--winner", required=True, help="Fact ID that should remain active.")
@click.option("--loser", required=True, help="Fact ID that should be superseded.")
@click.option("--reason", default="", help="Reason for conflict resolution.")
def conflicts_resolve(project: str, winner: str, loser: str, reason: str) -> None:
    from ccr.core.episodes import EpisodeStore
    from ccr.core.facts import FactLedger

    project = _require_ccr(project)
    ccr_root = os.path.join(project, ".ccr")
    ledger = FactLedger(ccr_root)
    winner_match = ledger.list_facts(query=winner, include_inactive=True, limit=1)
    loser_match = ledger.list_facts(query=loser, include_inactive=True, limit=1)
    if not winner_match:
        click.echo(f"Winner fact {winner} not found.", err=True)
        sys.exit(1)
    if not loser_match:
        click.echo(f"Loser fact {loser} not found.", err=True)
        sys.exit(1)
    if winner == loser:
        click.echo("Winner and loser must be different facts.", err=True)
        sys.exit(1)
    changed = ledger.supersede_fact(loser, superseded_by=winner)
    if not changed:
        click.echo(f"Could not supersede {loser}.", err=True)
        sys.exit(1)
    episode = EpisodeStore(ccr_root).append_episode(
        "conflict_resolution",
        summary=f"Resolved fact conflict: {winner} supersedes {loser}",
        content=reason,
        source_ids=[winner, loser],
        tools=["ccr conflicts resolve"],
        metadata={"winner": winner, "loser": loser},
    )
    click.echo(f"Resolved: {winner} supersedes {loser}; episode {episode.id}")


@click.group("replay")
def replay() -> None:
    """Inspect recall replay traces."""


@replay.command("list")
@click.argument("project", default=".")
@click.option("--query", default="", help="Optional query filter.")
@click.option("--limit", default=10, type=int, help="Maximum traces to print.")
@click.option("--json-output", is_flag=True, help="Print JSON instead of text.")
def replay_list(project: str, query: str, limit: int, json_output: bool) -> None:
    from ccr.core.replay import RecallTraceStore

    project = _require_ccr(project)
    traces = RecallTraceStore(os.path.join(project, ".ccr")).list_traces(query=query, limit=limit)
    if json_output:
        click.echo(json.dumps([t.to_dict() for t in traces], indent=2))
        return
    if not traces:
        click.echo("No recall traces found.")
        return
    for trace in traces:
        plan = trace.plan.get("intent", "recall")
        top = trace.evidence[0]["id"] if trace.evidence else "-"
        click.echo(f"{trace.id} {trace.created_at} {plan} conf={trace.confidence:.2f} top={top}: {trace.query}")


@replay.command("verify")
@click.argument("project", default=".")
def replay_verify(project: str) -> None:
    from ccr.core.replay import RecallTraceStore

    project = _require_ccr(project)
    ok, errors = RecallTraceStore(os.path.join(project, ".ccr")).verify_chain()
    click.echo("Recall trace chain OK" if ok else "Recall trace chain FAILED")
    for error in errors:
        click.echo(f"[!!] {error}")
    if not ok:
        sys.exit(1)


@click.command("tiers")
@click.argument("project", default=".")
@click.option("--json-output", is_flag=True, help="Print JSON instead of markdown.")
def tiers(project: str, json_output: bool) -> None:
    """Inspect explicit CCR memory tiers and promotion rules."""
    from ccr.core.memory_tiers import MemoryTierInspector

    project = _require_ccr(project)
    mem = _memory(project)
    inspector = MemoryTierInspector(mem)
    if json_output:
        click.echo(json.dumps([s.to_dict() for s in inspector.snapshot()], indent=2))
    else:
        click.echo(inspector.to_markdown())


@click.command("browser")
@click.argument("project", default=".")
@click.option("--output", "-o", default="", help="HTML output path.")
def browser(project: str, output: str) -> None:
    """Build a local static CCR memory browser HTML file."""
    from ccr.core.browser import MemoryBrowserBuilder

    project = _require_ccr(project)
    mem = _memory(project)
    out = output or os.path.join(project, ".ccr", "browser", "index.html")
    path = MemoryBrowserBuilder(mem).write(os.path.abspath(out))
    click.echo(f"Memory browser written to {path}")


@click.command("backup")
@click.argument("project", default=".")
@click.option("--output", "-o", default="", help="Backup zip path.")
def backup(project: str, output: str) -> None:
    """Create a signed zip backup of .ccr/."""
    from ccr.core.ops import backup_project

    project = _require_ccr(project)
    _print_ops(backup_project(project, output=output))


@click.command("restore")
@click.argument("backup_zip", type=click.Path(exists=True))
@click.argument("project", default=".")
@click.option("--overwrite", is_flag=True, help="Overwrite existing .ccr/.")
def restore(backup_zip: str, project: str, overwrite: bool) -> None:
    """Restore a CCR backup. Without --overwrite, restores to .ccr.restore-*."""
    from ccr.core.ops import restore_backup

    _print_ops(restore_backup(project, backup_zip, overwrite=overwrite))


@click.command("verify")
@click.argument("project", default=".")
@click.option("--json-output", is_flag=True, help="Print JSON report.")
def verify(project: str, json_output: bool) -> None:
    """Verify CCR memory files and SQLite databases."""
    from ccr.core.ops import verify_project

    report = verify_project(project)
    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _print_ops(report)


@click.command("repair")
@click.argument("project", default=".")
def repair(project: str) -> None:
    """Run safe local repairs: ensure dirs and verify databases."""
    from ccr.core.ops import repair_project

    _print_ops(repair_project(project))


@click.command("migrate")
@click.argument("project", default=".")
def migrate(project: str) -> None:
    """Run flat-file to SQLite migration checks for a CCR project."""
    from ccr.core.ops import migrate_project

    project = _require_ccr(project)
    _print_ops(migrate_project(project))


@click.group("governance")
def governance() -> None:
    """Security, redaction, policy, and audit helpers."""


@governance.command("init")
@click.argument("project", default=".")
def governance_init(project: str) -> None:
    """Create the default local governance policy."""
    from ccr.core.governance import GovernancePolicy, save_policy

    project = _require_ccr(project)
    path = save_policy(os.path.join(project, ".ccr"), GovernancePolicy(project_boundary=project))
    click.echo(f"Governance policy written to {path}")


@governance.command("scan")
@click.argument("project", default=".")
@click.option("--json-output", is_flag=True, help="Print JSON findings.")
def governance_scan(project: str, json_output: bool) -> None:
    """Scan .ccr memory files for secrets and PII."""
    from ccr.core.governance import append_audit, scan_memory_tree

    project = _require_ccr(project)
    ccr_root = os.path.join(project, ".ccr")
    findings = scan_memory_tree(ccr_root)
    append_audit(ccr_root, "governance_scan", "cli", {"findings": len(findings)})
    if json_output:
        click.echo(json.dumps([f.to_dict() for f in findings], indent=2))
        return
    if not findings:
        click.echo("No governance findings.")
        return
    for f in findings:
        click.echo(f"[{f.severity}] {f.kind} {f.path}:{f.line}:{f.column}")
    sys.exit(1)


@governance.command("events-validate")
@click.argument("event_file", type=click.Path(exists=True))
def governance_events_validate(event_file: str) -> None:
    """Validate a canonical AgentEvent JSONL file."""
    from ccr.core.events import validate_event_file

    valid, errors = validate_event_file(event_file)
    click.echo(f"Valid events: {valid}")
    for err in errors:
        click.echo(f"[!!] {err}")
    if errors:
        sys.exit(1)


@governance.command("audit-verify")
@click.argument("project", default=".")
def governance_audit_verify(project: str) -> None:
    """Verify local governance audit, episode, and recall trace chains."""
    from ccr.core.episodes import EpisodeStore
    from ccr.core.governance import verify_audit_chain
    from ccr.core.replay import RecallTraceStore

    project = _require_ccr(project)
    ccr_root = os.path.join(project, ".ccr")
    audit_ok, audit_errors = verify_audit_chain(ccr_root)
    episode_result = EpisodeStore(ccr_root).verify_chain()
    replay_ok, replay_errors = RecallTraceStore(ccr_root).verify_chain()
    click.echo(f"Governance audit: {'OK' if audit_ok else 'FAILED'}")
    click.echo(f"Episodes: {'OK' if episode_result.ok else 'FAILED'} ({episode_result.checked} checked)")
    click.echo(f"Recall traces: {'OK' if replay_ok else 'FAILED'}")
    for error in [*audit_errors, *episode_result.errors, *replay_errors]:
        click.echo(f"[!!] {error}")
    if not (audit_ok and episode_result.ok and replay_ok):
        sys.exit(1)


@click.group("encryption")
def encryption() -> None:
    """At-rest encryption for local CCR memory artifacts."""


@encryption.command("keygen")
@click.option("--output", "-o", default="", help="Write key to this file with 0600 permissions.")
@click.option("--show", is_flag=True, help="Print a base64 raw key for env-var based key management.")
@click.option("--overwrite", is_flag=True, help="Overwrite an existing key file.")
def encryption_keygen(output: str, show: bool, overwrite: bool) -> None:
    """Generate a 256-bit base64 key for env/keyfile encryption modes."""
    from ccr.core.encryption import generate_raw_key, write_key_file

    if not output and not show:
        click.echo("Use --output for a key file or --show for env-var key material.", err=True)
        sys.exit(1)
    if output:
        try:
            path = write_key_file(output, overwrite=overwrite)
        except Exception as exc:
            click.echo(f"[!!] {exc}", err=True)
            sys.exit(1)
        click.echo(f"Key written to {path}")
    if show:
        click.echo(generate_raw_key())


@encryption.command("init")
@click.argument("project", default=".")
@click.option("--key-source", default="passphrase",
              type=click.Choice(["passphrase", "env", "keyfile"]),
              help="Where CCR should load the encryption key from.")
@click.option("--env-var", default="", help="Environment variable containing passphrase or raw key.")
@click.option("--key-file", default="", help="Path to a base64 raw key file.")
@click.option("--passphrase", default="", hide_input=True,
              help="Passphrase for passphrase mode. Prefer the env var for automation.")
@click.option("--scope", default="default", type=click.Choice(["default", "all"]),
              help="Which .ccr files to protect.")
@click.option("--overwrite", is_flag=True, help="Replace an existing encryption policy.")
def encryption_init(
    project: str,
    key_source: str,
    env_var: str,
    key_file: str,
    passphrase: str,
    scope: str,
    overwrite: bool,
) -> None:
    """Initialize encryption policy without storing secret key material."""
    from ccr.core.encryption import EncryptionManager

    project = _require_ccr(project)
    try:
        report = EncryptionManager(os.path.join(project, ".ccr")).init_policy(
            key_source=key_source,
            env_var=env_var,
            key_file=key_file,
            passphrase=passphrase,
            scope=scope,
            overwrite=overwrite,
        )
    except Exception as exc:
        click.echo(f"[!!] {exc}", err=True)
        sys.exit(1)
    _print_ops(report)


@encryption.command("status")
@click.argument("project", default=".")
def encryption_status(project: str) -> None:
    """Show encryption policy status without loading the key."""
    from ccr.core.encryption import EncryptionManager

    project = _require_ccr(project)
    _print_ops(EncryptionManager(os.path.join(project, ".ccr")).status())


@encryption.command("lock")
@click.argument("project", default=".")
@click.option("--passphrase", default="", hide_input=True,
              help="Passphrase for passphrase mode. Env/keyfile modes ignore this.")
def encryption_lock(project: str, passphrase: str) -> None:
    """Encrypt protected .ccr artifacts and remove plaintext files."""
    from ccr.core.encryption import EncryptionManager

    project = _require_ccr(project)
    try:
        report = EncryptionManager(os.path.join(project, ".ccr")).lock(passphrase=passphrase)
    except Exception as exc:
        click.echo(f"[!!] {exc}", err=True)
        sys.exit(1)
    _print_ops(report)


@encryption.command("unlock")
@click.argument("project", default=".")
@click.option("--passphrase", default="", hide_input=True,
              help="Passphrase for passphrase mode. Env/keyfile modes ignore this.")
@click.option("--keep-encrypted", is_flag=True,
              help="Keep encrypted envelopes after restoring plaintext.")
def encryption_unlock(project: str, passphrase: str, keep_encrypted: bool) -> None:
    """Decrypt protected .ccr artifacts back to plaintext."""
    from ccr.core.encryption import EncryptionManager

    project = _require_ccr(project)
    try:
        report = EncryptionManager(os.path.join(project, ".ccr")).unlock(
            passphrase=passphrase,
            keep_encrypted=keep_encrypted,
        )
    except Exception as exc:
        click.echo(f"[!!] {exc}", err=True)
        sys.exit(1)
    _print_ops(report)


@encryption.command("verify")
@click.argument("project", default=".")
@click.option("--passphrase", default="", hide_input=True,
              help="Passphrase for passphrase mode. Env/keyfile modes ignore this.")
def encryption_verify(project: str, passphrase: str) -> None:
    """Verify encrypted envelopes and locked-state invariants."""
    from ccr.core.encryption import EncryptionManager

    project = _require_ccr(project)
    try:
        report = EncryptionManager(os.path.join(project, ".ccr")).verify(passphrase=passphrase)
    except Exception as exc:
        click.echo(f"[!!] {exc}", err=True)
        sys.exit(1)
    _print_ops(report)


@encryption.command("rotate")
@click.argument("project", default=".")
@click.option("--new-key-source", required=True,
              type=click.Choice(["passphrase", "env", "keyfile"]),
              help="New key source.")
@click.option("--new-env-var", default="", help="Environment variable for the new key/passphrase.")
@click.option("--new-key-file", default="", help="Path to the new base64 raw key file.")
@click.option("--old-passphrase", default="", hide_input=True,
              help="Current passphrase for passphrase mode.")
@click.option("--new-passphrase", default="", hide_input=True,
              help="New passphrase for passphrase mode.")
def encryption_rotate(
    project: str,
    new_key_source: str,
    new_env_var: str,
    new_key_file: str,
    old_passphrase: str,
    new_passphrase: str,
) -> None:
    """Rotate key metadata and re-encrypt locked envelopes when locked."""
    from ccr.core.encryption import EncryptionManager

    project = _require_ccr(project)
    try:
        report = EncryptionManager(os.path.join(project, ".ccr")).rotate(
            new_key_source=new_key_source,
            new_env_var=new_env_var,
            new_key_file=new_key_file,
            old_passphrase=old_passphrase,
            new_passphrase=new_passphrase,
        )
    except Exception as exc:
        click.echo(f"[!!] {exc}", err=True)
        sys.exit(1)
    _print_ops(report)


@click.command("rerankers")
def rerankers() -> None:
    """Show optional reranker provider availability."""
    from ccr.core.rerankers import provider_status

    for status in provider_status():
        mark = "OK" if status.available else "--"
        detail = f" ({status.detail})" if status.detail else ""
        click.echo(f"[{mark}] {status.provider}{detail}")


@click.group("sync")
def sync() -> None:
    """Git-backed local-first memory sync."""


@sync.command("init")
@click.argument("project", default=".")
@click.option("--remote", default="", help="Optional git remote URL.")
def sync_init(project: str, remote: str) -> None:
    from ccr.core.sync import GitMemorySync

    project = _require_ccr(project)
    result = GitMemorySync(project).init(remote=remote)
    click.echo(result.message)
    if not result.ok:
        sys.exit(1)


@sync.command("push")
@click.argument("project", default=".")
@click.option("--message", "-m", default="", help="Git commit message.")
def sync_push(project: str, message: str) -> None:
    from ccr.core.sync import GitMemorySync

    project = _require_ccr(project)
    result = GitMemorySync(project).push(message=message)
    click.echo(result.message)
    if not result.ok:
        sys.exit(1)


@sync.command("pull")
@click.argument("project", default=".")
@click.option("--apply", "apply_snapshot", is_flag=True, help="Apply the synced snapshot to .ccr/.")
def sync_pull(project: str, apply_snapshot: bool) -> None:
    from ccr.core.sync import GitMemorySync

    project = _require_ccr(project)
    result = GitMemorySync(project).pull(apply=apply_snapshot)
    click.echo(result.message)
    if not result.ok:
        sys.exit(1)


@sync.command("resolve")
@click.argument("project", default=".")
def sync_resolve(project: str) -> None:
    from ccr.core.sync import GitMemorySync

    project = _require_ccr(project)
    result = GitMemorySync(project).resolve()
    click.echo(result.message)
    for artifact in result.artifacts:
        click.echo(f"  {artifact}")
    if not result.ok:
        sys.exit(1)


@click.group("enterprise-gateway")
def enterprise_gateway() -> None:
    """Local enterprise policy gateway mode."""


@enterprise_gateway.command("init")
@click.argument("project", default=".")
def enterprise_gateway_init(project: str) -> None:
    from ccr.core.enterprise_gateway import EnterprisePolicyGateway

    project = _require_ccr(project)
    path = EnterprisePolicyGateway(project).init_policy()
    click.echo(f"Enterprise gateway policy initialized at {path}")


@enterprise_gateway.command("check")
@click.argument("project", default=".")
@click.option("--tool", default="gcc_commit", help="Tool name to check.")
@click.option("--role", default="writer", help="Actor role.")
@click.option("--text", default="", help="Optional proposed memory text to scan.")
def enterprise_gateway_check(project: str, tool: str, role: str, text: str) -> None:
    from ccr.core.enterprise_gateway import EnterprisePolicyGateway

    project = _require_ccr(project)
    gate = EnterprisePolicyGateway(project)
    decision = gate.check_memory_write(text, actor_role=role) if text else gate.check_tool(tool, role)
    if decision.allowed:
        click.echo("allowed")
        return
    for reason in decision.reasons:
        click.echo(f"[deny] {reason}")
    sys.exit(1)
