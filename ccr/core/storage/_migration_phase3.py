"""Phase 3 migration: commits, links, patterns, summaries, discussions.

Sub-phases:
  3a — commits.md, rolling summaries, branch metadata
  3b — commit links, patterns, triples, evolved summaries, clusters
  3c — discussions, session summaries, phase summaries, summary meta, overview
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from typing import Any

from ccr.core.storage._migration_utils import _backup_file

logger = logging.getLogger(__name__)

_COMMIT_HEADER_RE = re.compile(
    r"## \[(C\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|[^|]*\|\s*(.*)",
)

_DISCUSSION_HEADER_RE = re.compile(
    r"^## \[(D\d{3,})\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \| (.+)$",
    re.MULTILINE,
)

_SESSION_HEADER_RE = re.compile(
    r"## \[(S\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*-\s*"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*(\S+)\s*\|\s*Session Summary",
)

_PHASE_HEADER_RE = re.compile(
    r"## \[(P\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*-\s*"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*Phase Summary",
)


# ── Phase 3a ─────────────────────────────────────────────────────────


def migrate_phase_3a(
    ccr_root: str, db_path: str,
) -> dict[str, Any]:
    """Migrate commits.md, rolling summaries, and branch metadata to SQLite."""
    result: dict[str, Any] = {"migrated": 0, "errors": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN")
        result["migrated"] += _migrate_commits(ccr_root, conn)
        result["migrated"] += _migrate_rolling_summaries(ccr_root, conn)
        result["migrated"] += _migrate_branches_metadata(ccr_root, conn)
        conn.commit()
        logger.info("Phase 3a migration: %d records migrated", result["migrated"])
    except Exception as exc:
        conn.rollback()
        result["errors"].append(f"Phase 3a failed: {exc}")
        logger.error("Phase 3a migration failed: %s", exc)
    finally:
        conn.close()

    return result


def _migrate_commits(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse all branches' commits.md → INSERT into commits table."""
    branches_dir = os.path.join(ccr_root, "branches")
    if not os.path.isdir(branches_dir):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for branch_name in os.listdir(branches_dir):
        if branch_name.startswith(("_", ".")):
            continue
        commits_path = os.path.join(branches_dir, branch_name, "commits.md")
        if not os.path.isfile(commits_path):
            continue

        try:
            with open(commits_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        for part in parts:
            part = part.strip()
            m = _COMMIT_HEADER_RE.match(part)
            if not m:
                continue

            commit_id = m.group(1)
            timestamp = m.group(2)
            title = m.group(3).strip()

            what_m = re.search(r"\*\*What\*\*(?:\s*\[evolved\])?\s*:\s*(.*)", part)
            why_m = re.search(r"\*\*Why\*\*:\s*(.*)", part)
            files_m = re.search(r"\*\*Files\*\*:\s*(.*)", part)
            next_m = re.search(r"\*\*Next\*\*:\s*(.*)", part)
            score_m = re.search(r"\*\*Score\*\*:\s*([\d.]+)", part)
            patterns_m = re.search(r"\*\*Patterns\*\*:\s*(.*)", part)
            author_m = re.search(r"\*\*Author\*\*:\s*(.*)", part)
            ota_m = re.search(r"\*\*OTA Trace\*\*:\s*(.*)", part)

            what = what_m.group(1).strip() if what_m else ""
            why = why_m.group(1).strip() if why_m else ""
            files_str = files_m.group(1).strip() if files_m else ""
            files = [f.strip() for f in files_str.split(",") if f.strip() and f.strip() != "(none)"]
            next_step = next_m.group(1).strip() if next_m else ""
            score = float(score_m.group(1)) if score_m else None
            patterns = [p.strip() for p in patterns_m.group(1).split("|") if p.strip()] if patterns_m else []
            author = author_m.group(1).strip() if author_m else ""
            ota_trace = ota_m.group(1).strip() if ota_m else None

            conn.execute(
                """INSERT OR IGNORE INTO commits
                   (id, branch, timestamp, title, what, why, files_json,
                    next_step, patterns_json, score, author,
                    ota_trace, raw_block, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    commit_id, branch_name, timestamp, title, what, why,
                    json.dumps(files), next_step,
                    json.dumps(patterns) if patterns else None,
                    score, author, ota_trace, part, now,
                ),
            )
            count += 1

    return count


def _migrate_rolling_summaries(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse Rolling Summary sections from commits.md → rolling_summaries table."""
    branches_dir = os.path.join(ccr_root, "branches")
    if not os.path.isdir(branches_dir):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for branch_name in os.listdir(branches_dir):
        if branch_name.startswith(("_", ".")):
            continue
        commits_path = os.path.join(branches_dir, branch_name, "commits.md")
        if not os.path.isfile(commits_path):
            continue

        try:
            with open(commits_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        m = re.search(r"## Rolling Summary\n(.*?)(?=\n---|\n# |\Z)", content, re.DOTALL)
        if m:
            summary = m.group(1).strip()
            if summary and summary != "(none yet)":
                conn.execute(
                    """INSERT OR IGNORE INTO rolling_summaries
                       (branch, summary, updated_at)
                       VALUES (?, ?, ?)""",
                    (branch_name, summary, now),
                )
                count += 1

    return count


def _migrate_branches_metadata(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse metadata.yaml branches → branches table."""
    meta_path = os.path.join(ccr_root, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return 0

    try:
        import yaml
        with open(meta_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Could not parse metadata.yaml for branches: %s", exc)
        return 0

    if not data or not isinstance(data.get("branches"), list):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for b in data["branches"]:
        name = b.get("name")
        if not name:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO branches
               (name, status, parent, purpose, hypothesis,
                linked_issue, team_owner, priority, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                b.get("status", "active"),
                b.get("parent"),
                b.get("purpose"),
                b.get("hypothesis"),
                b.get("linked_issue"),
                b.get("team_owner"),
                b.get("priority"),
                b.get("created", now),
            ),
        )
        count += 1

    return count


# ── Phase 3b ─────────────────────────────────────────────────────────


def migrate_phase_3b(
    ccr_root: str, db_path: str,
) -> dict[str, Any]:
    """Migrate links, patterns, triples, evolved summaries, clusters to SQLite."""
    result: dict[str, Any] = {"migrated": 0, "errors": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN")
        result["migrated"] += _migrate_commit_links(ccr_root, conn)
        result["migrated"] += _migrate_patterns(ccr_root, conn)
        result["migrated"] += _migrate_triples(ccr_root, conn)
        result["migrated"] += _migrate_evolved_summaries(ccr_root, conn)
        result["migrated"] += _migrate_clusters(ccr_root, conn)
        conn.commit()
        logger.info("Phase 3b migration: %d records migrated", result["migrated"])
    except Exception as exc:
        conn.rollback()
        result["errors"].append(f"Phase 3b failed: {exc}")
        logger.error("Phase 3b migration failed: %s", exc)
    finally:
        conn.close()

    return result


def _migrate_commit_links(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse commit_links.json → INSERT into commit_links table."""
    path = os.path.join(ccr_root, "commit_links.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse commit_links.json: %s", exc)
        return 0

    links = data.get("links", {})
    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for source_id, type_map in links.items():
        if not isinstance(type_map, dict):
            continue
        for link_type, entries in type_map.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                conn.execute(
                    """INSERT OR IGNORE INTO commit_links
                       (source_id, target_id, link_type, score,
                        shared_files_json, snippet, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_id,
                        entry.get("target", ""),
                        link_type,
                        entry.get("score", 0.0),
                        json.dumps(entry["shared_files"]) if "shared_files" in entry else None,
                        entry.get("snippet"),
                        entry.get("created_at", now),
                    ),
                )
                count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_patterns(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse patterns.json → INSERT into patterns + pattern_meta tables."""
    path = os.path.join(ccr_root, "patterns.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse patterns.json: %s", exc)
        return 0

    patterns = data.get("patterns", {})
    next_id = data.get("next_id", 1)
    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for pid, p in patterns.items():
        commit_ids = p.get("commit_ids", [])
        conn.execute(
            """INSERT OR IGNORE INTO patterns
               (id, text, first_seen, commit_ids_json, occurrence_count,
                promoted, success_count, failure_count, quality_score,
                last_seen, last_quality_update, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid,
                p.get("text", ""),
                p.get("first_seen"),
                json.dumps(commit_ids),
                p.get("occurrence_count", 1),
                int(p.get("promoted", False)),
                p.get("success_count", 0),
                p.get("failure_count", 0),
                p.get("quality_score", 0.5),
                p.get("last_seen"),
                p.get("last_quality_update"),
                p.get("created_at", now),
            ),
        )
        count += 1

    conn.execute(
        "INSERT OR REPLACE INTO pattern_meta (key, value) VALUES ('next_id', ?)",
        (str(next_id),),
    )

    if count:
        _backup_file(path)
    return count


def _migrate_triples(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse triples.json → INSERT into triples table."""
    path = os.path.join(ccr_root, "triples.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse triples.json: %s", exc)
        return 0

    triple_list = data.get("triples", [])
    count = 0

    for t in triple_list:
        conn.execute(
            """INSERT OR IGNORE INTO triples
               (subject, predicate, object, source_commit, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                t.get("subject", ""),
                t.get("predicate", ""),
                t.get("object", ""),
                t.get("source_commit", ""),
                t.get("confidence", 0.8),
                t.get("timestamp"),
            ),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_evolved_summaries(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse evolved_summaries.json → INSERT into evolved_summaries table."""
    path = os.path.join(ccr_root, "evolved_summaries.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse evolved_summaries.json: %s", exc)
        return 0

    evolved = data.get("evolved", {})
    count = 0

    for cid, e in evolved.items():
        conn.execute(
            """INSERT OR IGNORE INTO evolved_summaries
               (commit_id, evolved_what, evolution_reason, evolved_at,
                source_commit_id, original_what)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                cid,
                e.get("evolved_what", ""),
                e.get("evolution_reason", ""),
                e.get("evolved_at", ""),
                e.get("source_commit_id", ""),
                e.get("original_what", ""),
            ),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_clusters(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse commit_clusters.json → INSERT into clusters + cluster_mapping tables."""
    path = os.path.join(ccr_root, "commit_clusters.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse commit_clusters.json: %s", exc)
        return 0

    cluster_list = data.get("clusters", [])
    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for cl in cluster_list:
        commit_ids = cl.get("commit_ids", [])
        cursor = conn.execute(
            """INSERT INTO clusters (label, commit_ids_json, created_at)
               VALUES (?, ?, ?)""",
            (
                cl.get("name", cl.get("label", "")),
                json.dumps(commit_ids),
                now,
            ),
        )
        cluster_db_id = cursor.lastrowid
        for cid in commit_ids:
            conn.execute(
                "INSERT OR IGNORE INTO cluster_mapping (commit_id, cluster_id) VALUES (?, ?)",
                (cid, cluster_db_id),
            )
        count += 1

    if count:
        _backup_file(path)
    return count


# ── Phase 3c ─────────────────────────────────────────────────────────


def migrate_phase_3c(
    ccr_root: str, db_path: str,
) -> dict[str, Any]:
    """Migrate discussions, session summaries, phase summaries, meta, overview to SQLite."""
    result: dict[str, Any] = {"migrated": 0, "errors": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN")
        result["migrated"] += _migrate_discussions(ccr_root, conn)
        result["migrated"] += _migrate_session_summaries(ccr_root, conn)
        result["migrated"] += _migrate_phase_summaries(ccr_root, conn)
        result["migrated"] += _migrate_summary_meta(ccr_root, conn)
        result["migrated"] += _migrate_overview(ccr_root, conn)
        conn.commit()
        logger.info("Phase 3c migration: %d records migrated", result["migrated"])
    except Exception as exc:
        conn.rollback()
        result["errors"].append(f"Phase 3c failed: {exc}")
        logger.error("Phase 3c migration failed: %s", exc)
    finally:
        conn.close()

    return result


def _migrate_discussions(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse discussions.md from all branches → INSERT into discussions table."""
    branches_dir = os.path.join(ccr_root, "branches")
    if not os.path.isdir(branches_dir):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for branch_name in os.listdir(branches_dir):
        if branch_name.startswith(("_", ".")):
            continue
        disc_path = os.path.join(branches_dir, branch_name, "discussions.md")
        if not os.path.isfile(disc_path):
            continue

        try:
            with open(disc_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        blocks = re.split(r"\n---\n", content)
        for block in blocks:
            block = block.strip()
            m = _DISCUSSION_HEADER_RE.search(block)
            if not m:
                continue

            disc_id = m.group(1)
            timestamp = m.group(2)
            topic = m.group(3).strip()

            fields = {}
            for field, pat in [
                ("hypothesis", r"\*\*Hypothesis\*\*:\s*(.+)"),
                ("alternatives", r"\*\*Alternatives\*\*:\s*(.+)"),
                ("decision", r"\*\*Decision\*\*:\s*(.+)"),
                ("rationale", r"\*\*Rationale\*\*:\s*(.+)"),
                ("uncertainty", r"\*\*Uncertainty\*\*:\s*(.+)"),
                ("linked_commit", r"\*\*Linked Commit\*\*:\s*(.+)"),
            ]:
                fm = re.search(pat, block)
                fields[field] = fm.group(1).strip() if fm else ""

            conn.execute(
                """INSERT OR IGNORE INTO discussions
                   (id, branch, timestamp, topic, hypothesis, alternatives,
                    decision, rationale, uncertainty, linked_commit, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    disc_id, branch_name, timestamp, topic,
                    fields["hypothesis"], fields["alternatives"],
                    fields["decision"], fields["rationale"],
                    fields["uncertainty"], fields["linked_commit"] or None, now,
                ),
            )
            count += 1

    return count


def _migrate_session_summaries(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse summaries.md from all branches → INSERT into session_summaries table."""
    branches_dir = os.path.join(ccr_root, "branches")
    if not os.path.isdir(branches_dir):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for branch_name in os.listdir(branches_dir):
        if branch_name.startswith(("_", ".")):
            continue
        sum_path = os.path.join(branches_dir, branch_name, "summaries.md")
        if not os.path.isfile(sum_path):
            continue

        try:
            with open(sum_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        blocks = re.split(r"\n---\n", content)
        for block in blocks:
            block = block.strip()
            m = _SESSION_HEADER_RE.search(block)
            if not m:
                continue

            sid = m.group(1)
            start_date = m.group(2)
            end_date = m.group(3)

            fields = {}
            for field, pat in [
                ("commit_range", r"\*\*Commits\*\*:\s*(.*)"),
                ("accomplished", r"\*\*Accomplished\*\*:\s*(.*)"),
                ("files_touched", r"\*\*Files touched\*\*:\s*(.*)"),
                ("key_decisions", r"\*\*Key decisions\*\*:\s*(.*)"),
                ("direction", r"\*\*Direction\*\*:\s*(.*)"),
            ]:
                fm = re.search(pat, block)
                fields[field] = fm.group(1).strip() if fm else ""

            conn.execute(
                """INSERT OR IGNORE INTO session_summaries
                   (id, branch, start_date, end_date, commit_range,
                    accomplished, files_touched, key_decisions, direction, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sid, branch_name, start_date, end_date,
                    fields["commit_range"], fields["accomplished"],
                    fields["files_touched"], fields["key_decisions"],
                    fields["direction"], now,
                ),
            )
            count += 1

    return count


def _migrate_phase_summaries(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse phases.md → INSERT into phase_summaries table."""
    path = os.path.join(ccr_root, "summaries", "phases.md")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    blocks = re.split(r"\n---\n", content)
    for block in blocks:
        block = block.strip()
        m = _PHASE_HEADER_RE.search(block)
        if not m:
            continue

        pid = m.group(1)
        start_date = m.group(2)
        end_date = m.group(3)

        fields = {}
        for field, pat in [
            ("scope", r"\*\*Scope\*\*:\s*(.*)"),
            ("goal", r"\*\*Goal\*\*:\s*(.*)"),
            ("outcome", r"\*\*Outcome\*\*:\s*(.*)"),
            ("accomplishments", r"\*\*Key accomplishments\*\*:\s*(.*)"),
            ("files_changed", r"\*\*Files changed\*\*:\s*(.*)"),
            ("branch_summary", r"\*\*Branch summary\*\*:\s*(.*)"),
        ]:
            fm = re.search(pat, block)
            fields[field] = fm.group(1).strip() if fm else ""

        conn.execute(
            """INSERT OR IGNORE INTO phase_summaries
               (id, start_date, end_date, scope, goal, outcome,
                accomplishments, files_changed, branch_summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid, start_date, end_date,
                fields["scope"], fields["goal"], fields["outcome"],
                fields["accomplishments"], fields["files_changed"],
                fields["branch_summary"], now,
            ),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_summary_meta(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse summary_meta.yaml → INSERT into summary_meta table."""
    path = os.path.join(ccr_root, "summary_meta.yaml")
    if not os.path.isfile(path):
        return 0

    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Could not parse summary_meta.yaml: %s", exc)
        return 0

    if not data:
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    conn.execute(
        "INSERT OR REPLACE INTO summary_meta (key, value_json, updated_at) VALUES ('_root', ?, ?)",
        (json.dumps(data, default=str), _utcnow()),
    )

    _backup_file(path)
    return 1


def _migrate_overview(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse overview.md → INSERT into project_state table."""
    path = os.path.join(ccr_root, "overview.md")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return 0

    if not content.strip():
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    conn.execute(
        "INSERT OR REPLACE INTO project_state (key, value, updated_at) VALUES ('overview', ?, ?)",
        (content, _utcnow()),
    )
    _backup_file(path)
    return 1
