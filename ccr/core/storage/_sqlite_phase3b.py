"""Phase 3b SQLite mixin: links, patterns, triples, evolved summaries, clusters."""

from __future__ import annotations

import json
import re
from typing import Any

from ccr.core.storage._sqlite_utils import (
    _PATTERN_COLUMNS,
    _escape_like,
    _utcnow,
)


class Phase3bMixin:
    """Link, pattern, triple, evolved summary, and cluster methods.

    Requires self.memory_conn from SqliteStorageBackend.
    """

    # ── Links ──────────────────────────────────────────────────

    def link_insert_batch(self, source_id: str, links: list[dict]) -> None:
        conn = self.memory_conn
        for link in links:
            conn.execute(
                """INSERT OR REPLACE INTO commit_links
                   (source_id, target_id, link_type, score,
                    shared_files_json, snippet, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_id,
                    link["target"],
                    link["link_type"],
                    link.get("score", 0.0),
                    json.dumps(link.get("shared_files", []), default=str)
                    if link.get("shared_files") else None,
                    link.get("snippet"),
                    link.get("created_at", _utcnow()),
                ),
            )
        conn.commit()

    def link_get_for_commit(self, commit_id: str) -> dict:
        conn = self.memory_conn
        rows = conn.execute(
            """SELECT * FROM commit_links
               WHERE source_id = ? OR target_id = ?""",
            (commit_id, commit_id),
        ).fetchall()
        result: dict[str, list[dict]] = {}
        for r in rows:
            lt = r["link_type"]
            if lt not in result:
                result[lt] = []
            entry: dict[str, Any] = {
                "target": r["target_id"] if r["source_id"] == commit_id else r["source_id"],
                "score": r["score"],
                "created_at": r["created_at"],
            }
            if r["shared_files_json"]:
                try:
                    entry["shared_files"] = json.loads(r["shared_files_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if r["snippet"]:
                entry["snippet"] = r["snippet"]
            result[lt].append(entry)
        return result

    def link_get_all(self) -> dict:
        conn = self.memory_conn
        rows = conn.execute("SELECT * FROM commit_links").fetchall()
        result: dict[str, dict[str, list[dict]]] = {}
        for r in rows:
            sid = r["source_id"]
            lt = r["link_type"]
            if sid not in result:
                result[sid] = {}
            if lt not in result[sid]:
                result[sid][lt] = []
            entry: dict[str, Any] = {
                "target": r["target_id"],
                "score": r["score"],
                "created_at": r["created_at"],
            }
            if r["shared_files_json"]:
                try:
                    entry["shared_files"] = json.loads(r["shared_files_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if r["snippet"]:
                entry["snippet"] = r["snippet"]
            result[sid][lt].append(entry)
        return result

    def link_prune(self, max_nodes: int) -> int:
        conn = self.memory_conn
        rows = conn.execute(
            "SELECT DISTINCT source_id FROM commit_links",
        ).fetchall()
        source_ids = [r["source_id"] for r in rows]
        if len(source_ids) <= max_nodes:
            return 0

        def _commit_num(cid: str) -> int:
            m = re.search(r"\d+$", cid)
            return int(m.group()) if m else 0

        sorted_ids = sorted(source_ids, key=_commit_num)
        evict_count = len(source_ids) - max_nodes
        evict_set = sorted_ids[:evict_count]

        placeholders = ",".join("?" * len(evict_set))
        conn.execute(
            f"DELETE FROM commit_links WHERE source_id IN ({placeholders})",
            evict_set,
        )
        conn.execute(
            f"DELETE FROM commit_links WHERE target_id IN ({placeholders})",
            evict_set,
        )
        conn.commit()
        return evict_count

    # ── Patterns ───────────────────────────────────────────────

    def pattern_load_all(self) -> dict:
        conn = self.memory_conn
        rows = conn.execute("SELECT * FROM patterns").fetchall()
        patterns: dict[str, dict] = {}
        for r in rows:
            d = dict(r)
            try:
                d["commit_ids"] = json.loads(d.pop("commit_ids_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["commit_ids"] = []
                d.pop("commit_ids_json", None)
            d["promoted"] = bool(d.get("promoted", 0))
            patterns[d["id"]] = d

        meta_row = conn.execute(
            "SELECT value FROM pattern_meta WHERE key = 'next_id'",
        ).fetchone()
        next_id = int(meta_row["value"]) if meta_row else 1

        return {"version": 1, "patterns": patterns, "next_id": next_id}

    def pattern_save_all(self, data: dict) -> None:
        conn = self.memory_conn
        patterns = data.get("patterns", {})
        next_id = data.get("next_id", 1)

        conn.execute("DELETE FROM patterns")
        for pid, p in patterns.items():
            conn.execute(
                """INSERT INTO patterns
                   (id, text, first_seen, commit_ids_json, occurrence_count,
                    promoted, success_count, failure_count, quality_score,
                    last_seen, last_quality_update, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pid, p.get("text", ""), p.get("first_seen"),
                    json.dumps(p.get("commit_ids", []), default=str),
                    p.get("occurrence_count", 1),
                    int(p.get("promoted", False)),
                    p.get("success_count", 0), p.get("failure_count", 0),
                    p.get("quality_score", 0.5),
                    p.get("last_seen"), p.get("last_quality_update"),
                    p.get("created_at", _utcnow()),
                ),
            )
        conn.execute(
            """INSERT OR REPLACE INTO pattern_meta (key, value)
               VALUES ('next_id', ?)""",
            (str(next_id),),
        )
        conn.commit()

    def pattern_get(self, pattern_id: str) -> dict | None:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT * FROM patterns WHERE id = ?", (pattern_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["commit_ids"] = json.loads(d.pop("commit_ids_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["commit_ids"] = []
            d.pop("commit_ids_json", None)
        d["promoted"] = bool(d.get("promoted", 0))
        return d

    def pattern_update(self, pattern_id: str, updates: dict) -> bool:
        conn = self.memory_conn
        if not updates:
            return False
        set_parts = []
        values: list[Any] = []
        for key, val in updates.items():
            if key == "commit_ids":
                set_parts.append("commit_ids_json = ?")
                values.append(json.dumps(val, default=str))
            elif key == "promoted":
                set_parts.append("promoted = ?")
                values.append(int(val))
            elif key in _PATTERN_COLUMNS:
                set_parts.append(f"{key} = ?")
                values.append(val)
        if not set_parts:
            return False
        values.append(pattern_id)
        conn.execute(
            f"UPDATE patterns SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        changed = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        return changed > 0

    def pattern_get_next_id(self) -> int:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT value FROM pattern_meta WHERE key = 'next_id'",
        ).fetchone()
        return int(row["value"]) if row else 1

    # ── Triples ────────────────────────────────────────────────

    def triple_insert_batch(self, triples: list[dict]) -> int:
        conn = self.memory_conn
        added = 0
        for t in triples:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO triples
                       (subject, predicate, object, source_commit,
                        confidence, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        t["subject"], t["predicate"], t["object"],
                        t["source_commit"],
                        t.get("confidence", 0.8),
                        t.get("timestamp", _utcnow()),
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    added += 1
            except Exception:
                pass
        conn.commit()
        return added

    def triple_list(
        self, top_k: int = 10, commit_id: str | None = None,
        entity: str | None = None,
    ) -> list[dict]:
        conn = self.memory_conn
        conditions = []
        params: list[Any] = []
        if commit_id:
            conditions.append("source_commit = ?")
            params.append(commit_id)
        if entity:
            conditions.append("(subject = ? OR object = ?)")
            params.extend([entity, entity])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(top_k)
        rows = conn.execute(
            f"SELECT * FROM triples {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def triple_search(self, query: str, top_k: int = 10) -> list[dict]:
        conn = self.memory_conn
        like = f"%{_escape_like(query)}%"
        rows = conn.execute(
            """SELECT * FROM triples
               WHERE subject LIKE ? ESCAPE '\\' OR predicate LIKE ? ESCAPE '\\'
                     OR object LIKE ? ESCAPE '\\'
               ORDER BY id DESC LIMIT ?""",
            (like, like, like, top_k),
        ).fetchall()
        return [dict(r) for r in rows]

    def triple_count(self) -> int:
        conn = self.memory_conn
        return conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    # ── Evolved Summaries ──────────────────────────────────────

    def evolved_summary_get(self, commit_id: str) -> dict | None:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT * FROM evolved_summaries WHERE commit_id = ?",
            (commit_id,),
        ).fetchone()
        return dict(row) if row else None

    def evolved_summary_set(self, commit_id: str, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT OR REPLACE INTO evolved_summaries
               (commit_id, evolved_what, evolution_reason, evolved_at,
                source_commit_id, original_what)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                commit_id,
                data.get("evolved_what", ""),
                data.get("evolution_reason", ""),
                data.get("evolved_at", _utcnow()),
                data.get("source_commit_id", ""),
                data.get("original_what", ""),
            ),
        )
        conn.commit()

    def evolved_summary_all(self) -> dict:
        conn = self.memory_conn
        rows = conn.execute("SELECT * FROM evolved_summaries").fetchall()
        return {r["commit_id"]: dict(r) for r in rows}

    # ── Clusters ───────────────────────────────────────────────

    def cluster_save(self, clusters: list[dict]) -> None:
        conn = self.memory_conn
        conn.execute("DELETE FROM cluster_mapping")
        conn.execute("DELETE FROM clusters")
        for cl in clusters:
            cursor = conn.execute(
                """INSERT INTO clusters (label, commit_ids_json, created_at)
                   VALUES (?, ?, ?)""",
                (
                    cl.get("name", cl.get("label", "")),
                    json.dumps(cl.get("commit_ids", []), default=str),
                    cl.get("created_at", _utcnow()),
                ),
            )
            cluster_db_id = cursor.lastrowid
            for cid in cl.get("commit_ids", []):
                conn.execute(
                    """INSERT OR REPLACE INTO cluster_mapping
                       (commit_id, cluster_id) VALUES (?, ?)""",
                    (cid, cluster_db_id),
                )
        conn.commit()

    def cluster_load(self) -> dict:
        conn = self.memory_conn
        rows = conn.execute("SELECT * FROM clusters").fetchall()
        clusters = []
        for r in rows:
            try:
                commit_ids = json.loads(r["commit_ids_json"])
            except (json.JSONDecodeError, TypeError):
                commit_ids = []
            clusters.append({
                "id": f"CL{r['id']:03d}",
                "name": r["label"] or "",
                "commit_ids": commit_ids,
                "created_at": r["created_at"],
            })
        mapping_rows = conn.execute("SELECT * FROM cluster_mapping").fetchall()
        commit_to_cluster: dict[str, str] = {}
        for mr in mapping_rows:
            commit_to_cluster[mr["commit_id"]] = f"CL{mr['cluster_id']:03d}"
        return {
            "version": 1,
            "clusters": clusters,
            "commit_to_cluster": commit_to_cluster,
        }
