"""CCR import command — import memory data from exported files."""
from __future__ import annotations

import json
import logging
import os
import sys

import click

logger = logging.getLogger(__name__)


def _get_backend(ccr_dir: str):
    """Get a StorageBackend for the given .ccr/ directory."""
    from ccr.core.storage.sqlite_backend import SqliteStorageBackend

    db_path = os.path.join(ccr_dir, "memory.db")
    if os.path.isfile(db_path):
        return SqliteStorageBackend(ccr_dir), "sqlite"

    from ccr.core.storage.file_backend import FileStorageBackend
    return FileStorageBackend(ccr_dir), "files"


def _load_records(path: str) -> list[dict]:
    """Load records from JSON or JSONL file."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        return json.loads(content)

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _import_commits(backend, records: list[dict], branch: str, merge: bool) -> int:
    count = 0
    for rec in records:
        cid = rec.get("id", "")
        if not cid:
            continue
        if not merge:
            existing = backend.commit_get(branch, cid)
            if existing:
                continue
        backend.commit_insert(branch, rec)
        count += 1
    return count


def _import_triples(backend, records: list[dict]) -> int:
    if not records:
        return 0
    return backend.triple_insert_batch(records)


def _import_links(backend, records: list[dict]) -> int:
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        source = rec.get("source_id", "")
        if not source:
            continue
        grouped.setdefault(source, []).append(rec)
    count = 0
    for source_id, links in grouped.items():
        backend.link_insert_batch(source_id, links)
        count += len(links)
    return count


def _import_patterns(backend, records: list[dict]) -> int:
    if not records:
        return 0
    existing = backend.pattern_load_all()
    patterns = existing.get("patterns", {})
    next_id = existing.get("next_id", 1)
    count = 0
    for rec in records:
        pid = rec.get("id", f"P{next_id:03d}")
        if pid not in patterns:
            patterns[pid] = rec
            try:
                num = int(pid.lstrip("P"))
                next_id = max(next_id, num + 1)
            except ValueError:
                pass
            count += 1
    backend.pattern_save_all({
        "version": existing.get("version", 1),
        "patterns": patterns,
        "next_id": next_id,
    })
    return count


def _import_discussions(backend, records: list[dict], branch: str, merge: bool) -> int:
    count = 0
    for rec in records:
        if not rec.get("id"):
            continue
        backend.discussion_insert(branch, rec)
        count += 1
    return count


def _import_playbook(backend, records: list[dict]) -> int:
    count = 0
    for rec in records:
        if not rec.get("id"):
            continue
        existing = backend.bullet_get(rec["id"])
        if existing:
            continue
        backend.bullet_insert(rec)
        count += 1
    return count


_IMPORTERS = {
    "commits": _import_commits,
    "triples": _import_triples,
    "links": _import_links,
    "patterns": _import_patterns,
    "discussions": _import_discussions,
    "playbook": _import_playbook,
}

_BRANCH_TABLES = {"commits", "discussions"}


@click.command("import")
@click.argument("table", type=click.Choice(list(_IMPORTERS.keys())))
@click.argument("file", type=click.Path(exists=True))
@click.option("--project", "-p", default=".", help="Project root directory")
@click.option("--branch", "-b", default="main", help="Branch for commits/discussions")
@click.option("--merge", is_flag=True,
              help="Overwrite existing records on ID collision (default: skip)")
def import_cmd(table: str, file: str, project: str, branch: str, merge: bool) -> None:
    """Import CCR memory data from a JSON or JSONL file.

    \b
    Tables: commits, playbook, triples, links, patterns, discussions
    By default, existing records are skipped. Use --merge to overwrite.
    """
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")

    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ directory found in {project}. Run 'ccr init' first.", err=True)
        sys.exit(1)

    records = _load_records(file)
    if not records:
        click.echo("No records found in input file.")
        return

    backend, backend_type = _get_backend(ccr_dir)

    try:
        importer = _IMPORTERS[table]
        if table in _BRANCH_TABLES:
            count = importer(backend, records, branch, merge)
        elif table == "playbook":
            count = importer(backend, records)
        else:
            count = importer(backend, records)

        click.echo(
            f"Imported {count}/{len(records)} {table} records "
            f"[backend: {backend_type}, merge: {merge}]",
        )
    finally:
        backend.close()
