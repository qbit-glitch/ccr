"""CCR export command — export memory tables to portable formats."""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys

import click

logger = logging.getLogger(__name__)

_TABLES = ("commits", "playbook", "triples", "links", "patterns", "discussions", "facts")


def _get_backend(ccr_dir: str):
    """Get a StorageBackend for the given .ccr/ directory."""
    from ccr.core.storage.sqlite_backend import SqliteStorageBackend

    db_path = os.path.join(ccr_dir, "memory.db")
    if os.path.isfile(db_path):
        return SqliteStorageBackend(ccr_dir), "sqlite"

    from ccr.core.storage.file_backend import FileStorageBackend
    return FileStorageBackend(ccr_dir), "files"


def _export_commits(backend, branch: str) -> list[dict]:
    return backend.commit_list(branch, limit=100000, offset=0)


def _export_playbook(backend) -> list[dict]:
    return backend.bullet_list(scope="project")


def _export_triples(backend) -> list[dict]:
    return backend.triple_list(top_k=100000)


def _export_links(backend) -> list[dict]:
    all_links = backend.link_get_all()
    rows = []
    for source_id, type_map in all_links.items():
        for link_type, entries in type_map.items():
            for entry in entries:
                rows.append({
                    "source_id": source_id,
                    "link_type": link_type,
                    **entry,
                })
    return rows


def _export_patterns(backend) -> list[dict]:
    data = backend.pattern_load_all()
    rows = []
    for pid, pat in data.get("patterns", {}).items():
        rows.append({"id": pid, **pat})
    return rows


def _export_discussions(backend, branch: str) -> list[dict]:
    return backend.discussion_list(branch)


def _export_facts(backend) -> list[dict]:
    from ccr.core.facts import FactLedger

    return [f.to_dict() for f in FactLedger(backend.ccr_root).list_facts(
        include_inactive=True,
        limit=100000,
    )]


_EXPORTERS = {
    "commits": _export_commits,
    "playbook": _export_playbook,
    "triples": _export_triples,
    "links": _export_links,
    "patterns": _export_patterns,
    "discussions": _export_discussions,
    "facts": _export_facts,
}

_BRANCH_TABLES = {"commits", "discussions"}


def _format_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2, default=str)


def _format_jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, default=str) for r in rows)


def _format_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: str(v) if v is not None else "" for k, v in r.items()})
    return buf.getvalue()


def _format_markdown(rows: list[dict]) -> str:
    if not rows:
        return "*No data*\n"
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    lines = []
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            s = str(v) if v is not None else ""
            s = s.replace("|", "\\|").replace("\n", " ")
            if len(s) > 80:
                s = s[:77] + "..."
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


_FORMATTERS = {
    "json": _format_json,
    "jsonl": _format_jsonl,
    "csv": _format_csv,
    "markdown": _format_markdown,
}


@click.command("export")
@click.argument("table", type=click.Choice(_TABLES))
@click.argument("project", default=".")
@click.option("--format", "fmt", default="json",
              type=click.Choice(["json", "jsonl", "csv", "markdown"]),
              help="Output format (default: json)")
@click.option("--branch", "-b", default="main", help="Branch for commits/discussions")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
@click.option("--redacted", is_flag=True, help="Redact secrets/PII before exporting.")
def export(
    table: str,
    project: str,
    fmt: str,
    branch: str,
    output: str | None,
    redacted: bool,
) -> None:
    """Export a CCR memory table to a portable format.

    \b
    Tables: commits, playbook, triples, links, patterns, discussions
    Formats: json (default), jsonl, csv, markdown
    """
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")

    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ directory found in {project}. Run 'ccr init' first.", err=True)
        sys.exit(1)

    backend, backend_type = _get_backend(ccr_dir)

    try:
        exporter = _EXPORTERS[table]
        if table in _BRANCH_TABLES:
            rows = exporter(backend, branch)
        else:
            rows = exporter(backend)
        if redacted:
            from ccr.core.governance import append_audit, redact_data
            rows = redact_data(rows)
            append_audit(ccr_dir, "export_redacted", "cli", {"table": table, "rows": len(rows)})

        formatter = _FORMATTERS[fmt]
        text = formatter(rows)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(text)
            click.echo(
                f"Exported {len(rows)} {table} records ({fmt}) → {output} "
                f"[backend: {backend_type}]",
                err=True,
            )
        else:
            click.echo(text)
    finally:
        backend.close()
