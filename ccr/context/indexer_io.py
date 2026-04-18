"""IO mixin for RepoIndex — serialization, deserialization, and summaries.

Extracted from indexer.py to keep files under 400 lines.
"""

from __future__ import annotations

import json
import os

from ccr.context.indexer_types import FileEntry


class IOMixin:
    """Mixin providing serialization, deserialization, and summary methods.

    Must be mixed into a class that has:
      - self.root: str
      - self.files: dict[str, FileEntry]
      - self._built_at: float | None
      - self._mtime_sig: str
      - self._file_hashes: dict[str, str]
      - self._embeddings: dict[str, list[float]]
      - self._chunk_embeddings: dict[str, list[float]]
    """

    def save_embeddings(self, path: str) -> None:
        """Save embeddings to gzip-compressed JSON."""
        from ccr.context.embeddings import save_embeddings

        save_embeddings(self._embeddings, path)

    def load_embeddings(self, path: str) -> bool:
        """Load pre-computed embeddings. Returns True if loaded."""
        from ccr.context.embeddings import load_embeddings

        loaded = load_embeddings(path)
        if loaded:
            self._embeddings = loaded
            return True
        return False

    def save_chunk_embeddings(self, path: str) -> None:
        """Save chunk embeddings to gzip-compressed JSON.

        Mirror of save_embeddings() for chunk-level data.
        Path: index_chunk_embeddings.json.gz in .ccr/.
        """
        from ccr.context.embeddings import save_embeddings

        save_embeddings(self._chunk_embeddings, path)

    def load_chunk_embeddings(self, path: str) -> bool:
        """Load pre-computed chunk embeddings. Returns True if loaded.

        Mirror of load_embeddings() for chunk-level data.
        Also restores chunk.embedding on each ChunkEntry by matching keys.
        """
        from ccr.context.embeddings import load_embeddings

        loaded = load_embeddings(path)
        if not loaded:
            return False

        self._chunk_embeddings = loaded

        # Restore embeddings back into ChunkEntry objects
        for rel, entry in self.files.items():
            for chunk in entry.chunks:
                key = f"{rel}::{chunk.chunk_idx}"
                if key in self._chunk_embeddings:
                    chunk.embedding = self._chunk_embeddings[key]

        return True

    def to_json(self) -> str:
        """Serialize to JSON for REPL loading. Excludes file contents for efficiency."""
        data = {
            "root": self.root,
            "built_at": self._built_at,
            "mtime_sig": self._mtime_sig or "",
            "file_count": len(self.files),
            "file_hashes": self._file_hashes,
            "files": {},
        }
        for rel, entry in self.files.items():
            data["files"][rel] = {
                "size": entry.size_bytes,
                "language": entry.language,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "lines": entry.line_count,
            }
        return json.dumps(data, indent=None, separators=(",", ":"))

    def to_full_json(self) -> str:
        """Serialize including file contents -- used for REPL context loading."""
        data = {
            "root": self.root,
            "built_at": self._built_at,
            "file_count": len(self.files),
            "files": {},
        }
        for rel, entry in self.files.items():
            data["files"][rel] = {
                "content": entry._content,
                "size": entry.size_bytes,
                "language": entry.language,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "lines": entry.line_count,
            }
        return json.dumps(data, indent=None, separators=(",", ":"))

    def to_context_dict(self) -> dict:
        """Convert to a dict suitable for loading as a REPL context variable.

        Returns metadata (path, symbols, imports, size) for each file -- NOT content.
        Content is fetched on-demand via get_file() to avoid loading everything.
        """
        files_meta = {}
        for rel, entry in self.files.items():
            files_meta[rel] = {
                "size": entry.size_bytes,
                "language": entry.language,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "lines": entry.line_count,
            }
        return {
            "root": self.root,
            "file_count": len(self.files),
            "files": files_meta,
        }

    @classmethod
    def from_cache(cls, root: str, cache_json: str):
        """Load from cache if mtime signatures match."""
        try:
            data = json.loads(cache_json)
            index = cls(root)
            # Backward compat: old caches won't have file_hashes
            index._file_hashes = data.get("file_hashes", {})
            for rel, fdata in data.get("files", {}).items():
                abs_path = os.path.join(root, rel)
                if not os.path.isfile(abs_path):
                    return None  # File deleted, cache invalid
                index.files[rel] = FileEntry(
                    rel_path=rel,
                    size_bytes=fdata.get("size", 0),
                    language=fdata.get("language", ""),
                    symbols=fdata.get("symbols", []),
                    imports=fdata.get("imports", []),
                    last_modified=0,
                    line_count=fdata.get("lines", 0),
                )
            index._built_at = data.get("built_at")
            index._mtime_sig = data.get("mtime_sig", "")
            return index
        except (json.JSONDecodeError, KeyError):
            return None

    def save_to_db(self, db) -> int:
        """Save index state to IndexDB. Returns count of files saved."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        files_data = []
        for rel, entry in self.files.items():
            files_data.append({
                "path": rel,
                "language": entry.language,
                "line_count": entry.line_count,
                "size_bytes": entry.size_bytes,
                "mtime": entry.last_modified,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "git_hash": self._file_hashes.get(rel, ""),
            })
        count = db.save_files_batch(files_data, now)

        chunks_data = []
        for rel, entry in self.files.items():
            for chunk in entry.chunks:
                chunks_data.append({
                    "file_path": rel,
                    "chunk_idx": chunk.chunk_idx,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                })
        if chunks_data:
            db.save_chunks_batch(chunks_data)

        if self._built_at:
            db.set_meta("built_at", str(self._built_at))
        if self._mtime_sig:
            db.set_meta("mtime_sig", self._mtime_sig)
        db.set_meta("root", self.root)
        db.set_meta("file_count", str(len(self.files)))

        return count

    @classmethod
    def from_db(cls, root: str, db):
        """Load index from IndexDB. Returns RepoIndex or None."""
        files_data = db.load_files()
        if not files_data:
            return None

        index = cls(root)
        for f in files_data:
            abs_path = os.path.join(root, f["path"])
            if not os.path.isfile(abs_path):
                continue
            index.files[f["path"]] = FileEntry(
                rel_path=f["path"],
                size_bytes=f["size_bytes"],
                language=f["language"],
                symbols=f["symbols"],
                imports=f["imports"],
                last_modified=f["mtime"],
                line_count=f["line_count"],
            )
            index._file_hashes[f["path"]] = f.get("git_hash", "")

        built_at = db.get_meta("built_at")
        if built_at:
            try:
                index._built_at = float(built_at)
            except (ValueError, TypeError):
                pass

        mtime_sig = db.get_meta("mtime_sig")
        if mtime_sig:
            index._mtime_sig = mtime_sig

        return index if index.files else None

    def get_summary(self) -> str:
        """Human-readable summary of the indexed repo."""
        langs: dict[str, int] = {}
        total_lines = 0
        for entry in self.files.values():
            langs[entry.language] = langs.get(entry.language, 0) + 1
            total_lines += entry.line_count

        lang_str = ", ".join(
            f"{lang}: {count}"
            for lang, count in sorted(langs.items(), key=lambda x: -x[1])[:5]
        )
        return (
            f"Repo: {self.root}\n"
            f"Files: {len(self.files)} | Lines: {total_lines:,}\n"
            f"Languages: {lang_str}\n"
        )
