"""Tests for Phase 5 CLI tools: doctor --fix, export, import, stats (SQLite)."""
from __future__ import annotations

import json
import os

import pytest
import yaml
from click.testing import CliRunner

from ccr.core.storage.sqlite_backend import SqliteStorageBackend


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def project_dir(tmp_path):
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    meta = {"version": 1, "created": "2026-04-18", "branches": []}
    with open(str(ccr / "metadata.yaml"), "w") as f:
        yaml.dump(meta, f)
    return tmp_path


@pytest.fixture
def sqlite_project(project_dir):
    """Project with SQLite backend and sample data."""
    ccr = project_dir / ".ccr"
    backend = SqliteStorageBackend(str(ccr))

    backend.commit_insert("main", {
        "id": "C001", "timestamp": "2026-04-18 10:00",
        "title": "Initial commit", "what": "Added storage layer",
        "why": "SQLite migration", "files_json": '["base.py"]',
        "next_step": "Add tests", "author": "test",
    })
    backend.commit_insert("main", {
        "id": "C002", "timestamp": "2026-04-18 11:00",
        "title": "Add tests", "what": "Test coverage",
        "why": "Quality", "files_json": '["test.py"]',
        "next_step": "Ship it", "author": "test",
    })

    backend.triple_insert_batch([
        {"subject": "StorageBackend", "predicate": "defines", "object": "interface",
         "source_commit": "C001", "confidence": 0.9},
        {"subject": "SqliteBackend", "predicate": "implements", "object": "StorageBackend",
         "source_commit": "C001", "confidence": 0.95},
    ])

    backend.link_insert_batch("C001", [
        {"target": "C002", "link_type": "causal", "score": 0.8},
    ])

    backend.pattern_save_all({
        "version": 1, "next_id": 2,
        "patterns": {
            "P001": {"text": "TDD workflow", "commit_ids": ["C001"],
                     "occurrence_count": 3, "promoted": True,
                     "quality_score": 0.85, "created_at": "2026-04-18"},
        },
    })

    backend.discussion_insert("main", {
        "id": "D001", "timestamp": "2026-04-18 10:00",
        "topic": "Storage design", "decision": "Use SQLite",
    })

    backend.close()
    return project_dir


# ── Doctor --fix ───────────────────────────────────────────────


class TestDoctorFix:
    def test_fix_flag_exists(self, runner, project_dir):
        from ccr.cli_doctor import doctor
        result = runner.invoke(doctor, [str(project_dir), "--fix"])
        assert result.exit_code == 0
        assert "Auto-fix" in result.output

    def test_fix_stale_session_marker(self, runner, project_dir):
        from ccr.cli_doctor import _fix_stale_session_marker
        ccr = str(project_dir / ".ccr")
        marker = os.path.join(ccr, ".session_active")
        with open(marker, "w") as f:
            f.write("99999999")
        fixes = _fix_stale_session_marker(ccr)
        assert len(fixes) == 1
        assert "Removed stale" in fixes[0]
        assert not os.path.isfile(marker)

    def test_fix_stale_marker_live_pid(self, runner, project_dir):
        from ccr.cli_doctor import _fix_stale_session_marker
        ccr = str(project_dir / ".ccr")
        marker = os.path.join(ccr, ".session_active")
        with open(marker, "w") as f:
            f.write(str(os.getpid()))
        fixes = _fix_stale_session_marker(ccr)
        assert fixes == []
        assert os.path.isfile(marker)

    def test_fix_sqlite_integrity(self, runner, sqlite_project):
        from ccr.cli_doctor import _fix_sqlite_integrity
        ccr = str(sqlite_project / ".ccr")
        fixes = _fix_sqlite_integrity(ccr)
        assert any("integrity check passed" in f for f in fixes)

    def test_fix_duplicate_hooks(self, runner, project_dir):
        from ccr.cli_doctor import _fix_duplicate_hooks
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir()
        settings = {
            "hooks": {
                "Stop": [
                    {"type": "command", "command": "python ccr/hooks/on_stop.py"},
                    {"type": "command", "command": "python ccr/hooks/on_stop.py"},
                ],
            },
        }
        with open(str(claude_dir / "settings.local.json"), "w") as f:
            json.dump(settings, f)

        fixes = _fix_duplicate_hooks(str(project_dir))
        assert len(fixes) == 1
        assert "Removed duplicate" in fixes[0]

        with open(str(claude_dir / "settings.local.json")) as f:
            result = json.load(f)
        assert len(result["hooks"]["Stop"]) == 1

    def test_fix_no_issues(self, runner, project_dir):
        from ccr.cli_doctor import doctor
        result = runner.invoke(doctor, [str(project_dir), "--fix"])
        assert result.exit_code == 0
        assert "No auto-fixable issues" in result.output or "fix" in result.output.lower()


# ── Export ─────────────────────────────────────────────────────


class TestExport:
    def test_export_commits_json(self, runner, sqlite_project):
        from ccr.cli_export import export
        result = runner.invoke(export, ["commits", str(sqlite_project), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["id"] in ("C001", "C002")

    def test_export_triples_jsonl(self, runner, sqlite_project):
        from ccr.cli_export import export
        result = runner.invoke(export, ["triples", str(sqlite_project), "--format", "jsonl"])
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert "subject" in first

    def test_export_links_csv(self, runner, sqlite_project):
        from ccr.cli_export import export
        result = runner.invoke(export, ["links", str(sqlite_project), "--format", "csv"])
        assert result.exit_code == 0
        assert "source_id" in result.output
        assert "C001" in result.output

    def test_export_patterns_markdown(self, runner, sqlite_project):
        from ccr.cli_export import export
        result = runner.invoke(export, ["patterns", str(sqlite_project), "--format", "markdown"])
        assert result.exit_code == 0
        assert "TDD workflow" in result.output
        assert "|" in result.output

    def test_export_discussions(self, runner, sqlite_project):
        from ccr.cli_export import export
        result = runner.invoke(export, ["discussions", str(sqlite_project)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["topic"] == "Storage design"

    def test_export_to_file(self, runner, sqlite_project):
        from ccr.cli_export import export
        out = str(sqlite_project / "commits.json")
        result = runner.invoke(export, [
            "commits", str(sqlite_project), "--output", out,
        ])
        assert result.exit_code == 0
        assert os.path.isfile(out)
        with open(out) as f:
            data = json.loads(f.read())
        assert len(data) == 2

    def test_export_no_ccr(self, runner, tmp_path):
        from ccr.cli_export import export
        result = runner.invoke(export, ["commits", str(tmp_path)])
        assert result.exit_code != 0

    def test_export_empty_table(self, runner, project_dir):
        from ccr.cli_export import export
        SqliteStorageBackend(str(project_dir / ".ccr")).close()
        result = runner.invoke(export, ["triples", str(project_dir)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    def test_export_playbook(self, runner, sqlite_project):
        from ccr.cli_export import export
        result = runner.invoke(export, ["playbook", str(sqlite_project)])
        assert result.exit_code == 0


# ── Import ─────────────────────────────────────────────────────


class TestImport:
    def test_import_triples(self, runner, project_dir):
        from ccr.cli_import import import_cmd
        ccr = str(project_dir / ".ccr")
        SqliteStorageBackend(ccr).close()

        data = [
            {"subject": "A", "predicate": "uses", "object": "B",
             "source_commit": "C001", "confidence": 0.9},
        ]
        input_file = str(project_dir / "triples.json")
        with open(input_file, "w") as f:
            json.dump(data, f)

        result = runner.invoke(import_cmd, [
            "triples", input_file, "--project", str(project_dir),
        ])
        assert result.exit_code == 0
        assert "1/1" in result.output

    def test_import_jsonl(self, runner, project_dir):
        from ccr.cli_import import import_cmd
        ccr = str(project_dir / ".ccr")
        SqliteStorageBackend(ccr).close()

        input_file = str(project_dir / "triples.jsonl")
        with open(input_file, "w") as f:
            f.write(json.dumps({"subject": "X", "predicate": "p", "object": "Y",
                                "source_commit": "C001", "confidence": 0.8}) + "\n")

        result = runner.invoke(import_cmd, [
            "triples", input_file, "--project", str(project_dir),
        ])
        assert result.exit_code == 0
        assert "1/1" in result.output

    def test_import_links(self, runner, project_dir):
        from ccr.cli_import import import_cmd
        ccr = str(project_dir / ".ccr")
        SqliteStorageBackend(ccr).close()

        data = [
            {"source_id": "C001", "target": "C002", "link_type": "entity", "score": 0.7},
        ]
        input_file = str(project_dir / "links.json")
        with open(input_file, "w") as f:
            json.dump(data, f)

        result = runner.invoke(import_cmd, [
            "links", input_file, "--project", str(project_dir),
        ])
        assert result.exit_code == 0
        assert "1/1" in result.output

    def test_import_patterns(self, runner, project_dir):
        from ccr.cli_import import import_cmd
        ccr = str(project_dir / ".ccr")
        SqliteStorageBackend(ccr).close()

        data = [
            {"id": "P001", "text": "test pattern", "commit_ids": [],
             "occurrence_count": 1, "promoted": False,
             "quality_score": 0.5, "created_at": "2026-04-18"},
        ]
        input_file = str(project_dir / "patterns.json")
        with open(input_file, "w") as f:
            json.dump(data, f)

        result = runner.invoke(import_cmd, [
            "patterns", input_file, "--project", str(project_dir),
        ])
        assert result.exit_code == 0
        assert "1/1" in result.output

    def test_export_import_roundtrip(self, runner, sqlite_project):
        from ccr.cli_export import export
        from ccr.cli_import import import_cmd

        out = str(sqlite_project / "triples_export.json")
        runner.invoke(export, [
            "triples", str(sqlite_project), "--output", out,
        ])

        new_proj = sqlite_project / "new_project"
        new_proj.mkdir()
        new_ccr = new_proj / ".ccr"
        new_ccr.mkdir()
        meta = {"version": 1, "created": "2026-04-18", "branches": []}
        with open(str(new_ccr / "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)
        SqliteStorageBackend(str(new_ccr)).close()

        result = runner.invoke(import_cmd, [
            "triples", out, "--project", str(new_proj),
        ])
        assert result.exit_code == 0
        assert "2/" in result.output

    def test_import_empty_file(self, runner, project_dir):
        from ccr.cli_import import import_cmd
        ccr = str(project_dir / ".ccr")
        SqliteStorageBackend(ccr).close()

        input_file = str(project_dir / "empty.json")
        with open(input_file, "w") as f:
            json.dump([], f)

        result = runner.invoke(import_cmd, [
            "triples", input_file, "--project", str(project_dir),
        ])
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_import_no_ccr(self, runner, tmp_path):
        from ccr.cli_import import import_cmd
        input_file = str(tmp_path / "data.json")
        with open(input_file, "w") as f:
            json.dump([{"subject": "A"}], f)
        result = runner.invoke(import_cmd, [
            "triples", input_file, "--project", str(tmp_path),
        ])
        assert result.exit_code != 0


# ── Stats (SQLite) ─────────────────────────────────────────────


class TestStatsSqlite:
    def test_stats_with_sqlite(self, runner, sqlite_project):
        from ccr.cli_stats import stats
        result = runner.invoke(stats, [str(sqlite_project)])
        assert result.exit_code == 0
        assert "sqlite" in result.output.lower()
        assert "Commits:" in result.output

    def test_stats_shows_counts(self, runner, sqlite_project):
        from ccr.cli_stats import stats
        result = runner.invoke(stats, [str(sqlite_project)])
        assert result.exit_code == 0
        assert "Triples:" in result.output
        assert "Links:" in result.output
        assert "Patterns:" in result.output

    def test_stats_no_ccr(self, runner, tmp_path):
        from ccr.cli_stats import stats
        result = runner.invoke(stats, [str(tmp_path)])
        assert result.exit_code == 0
        assert "No .ccr/" in result.output


# ── Formatters ─────────────────────────────────────────────────


class TestFormatters:
    def test_format_csv_empty(self):
        from ccr.cli_export import _format_csv
        assert _format_csv([]) == ""

    def test_format_csv_rows(self):
        from ccr.cli_export import _format_csv
        rows = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        out = _format_csv(rows)
        assert "a,b" in out
        assert "1,hello" in out

    def test_format_markdown_empty(self):
        from ccr.cli_export import _format_markdown
        assert "No data" in _format_markdown([])

    def test_format_markdown_truncates(self):
        from ccr.cli_export import _format_markdown
        rows = [{"text": "x" * 200}]
        out = _format_markdown(rows)
        assert "..." in out

    def test_format_jsonl(self):
        from ccr.cli_export import _format_jsonl
        rows = [{"a": 1}, {"b": 2}]
        out = _format_jsonl(rows)
        lines = out.strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_load_records_json(self, tmp_path):
        from ccr.cli_import import _load_records
        path = str(tmp_path / "data.json")
        with open(path, "w") as f:
            json.dump([{"x": 1}], f)
        assert _load_records(path) == [{"x": 1}]

    def test_load_records_jsonl(self, tmp_path):
        from ccr.cli_import import _load_records
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write('{"x": 1}\n{"x": 2}\n')
        records = _load_records(path)
        assert len(records) == 2
