"""Tests for the ``ccr bench`` CLI using a registered fake adapter."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from ccr.bench import datasets
from ccr.bench.types import Doc, Query, World
from ccr.cli import cli


def _fake_worlds(cfg) -> list[World]:
    docs = (Doc("d1", "the quokka lives in australia"), Doc("d2", "rain in spain"),
            Doc("d3", "weather is nice"))
    qs = (Query("q1", "quokka", frozenset({"d1"}), "australia", "cat1"),
          Query("q2", "no gold", frozenset(), None, "cat5"))
    return [World(f"w{i}", docs, qs, f"g{i}") for i in range(10)]


@pytest.fixture(autouse=True)
def fake_adapter(monkeypatch):
    monkeypatch.setitem(datasets.ADAPTERS, "fake", _fake_worlds)


def _run(tmp_path, *args):
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli, ["bench", "run", "fake", "--output", str(out), "--cache-dir",
              str(tmp_path / "cache"), *args])
    return result, out


def test_list_shows_datasets_and_systems() -> None:
    result = CliRunner().invoke(cli, ["bench", "list"])
    assert result.exit_code == 0
    assert "fake" in result.output and "bm25" in result.output


def test_pilot_run_writes_artifacts_with_pilot_label(tmp_path) -> None:
    result, out = _run(tmp_path, "--systems", "bm25", "--pilot", "--no-reader")
    assert result.exit_code == 0, result.output
    for name in ("results.jsonl", "report.json", "report.md", "meta.json", "sample.json"):
        assert (out / name).exists(), name
    assert "pilot (10%)" in (out / "report.md").read_text()
    assert len(json.loads((out / "sample.json").read_text())["world_ids"]) == 1


def test_full_run_is_not_labeled_pilot(tmp_path) -> None:
    result, out = _run(tmp_path, "--systems", "bm25", "--full", "--no-reader")
    assert result.exit_code == 0, result.output
    assert json.loads((out / "meta.json").read_text())["pilot"] == "false"
    assert len(json.loads((out / "sample.json").read_text())["world_ids"]) == 10


def test_rerun_resumes_without_duplicating_results(tmp_path) -> None:
    _, out = _run(tmp_path, "--systems", "bm25", "--no-reader")
    first = (out / "results.jsonl").read_text().count("\n")
    _run(tmp_path, "--systems", "bm25", "--no-reader")
    assert (out / "results.jsonl").read_text().count("\n") == first


def test_unknown_dataset_fails_and_lists_known() -> None:
    result = CliRunner().invoke(cli, ["bench", "run", "nope"])
    assert result.exit_code != 0
    assert "fake" in result.output


def test_unreachable_reader_exits_nonzero_naming_host(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")
    result, _ = _run(tmp_path, "--systems", "bm25", "--reader")
    assert result.exit_code != 0
    assert "127.0.0.1:1" in result.output


def test_ks_above_system_cap_are_dropped_and_recorded(tmp_path) -> None:
    result, out = _run(tmp_path, "--systems", "ccr_memory", "--no-reader",
                       "--ks", "1,5,100")
    assert result.exit_code == 0, result.output
    meta = json.loads((out / "meta.json").read_text())
    assert meta["dropped_ks.ccr_memory"] == "100"


def test_unavailable_system_is_skipped_and_recorded(tmp_path) -> None:
    result, out = _run(tmp_path, "--systems", "bm25,dense_minilm", "--no-reader")
    assert result.exit_code == 0, result.output
    meta = json.loads((out / "meta.json").read_text())
    assert "skipped.dense_minilm" in meta
    assert "bm25" in (out / "report.md").read_text()


def test_meta_records_reader_and_environment(tmp_path) -> None:
    _, out = _run(tmp_path, "--systems", "bm25", "--no-reader")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["dataset"] == "fake" and meta["reader"] == "none"
    assert meta["seed"] == "42" and "semantic_active" in meta and "git_sha" in meta


def test_dataset_without_evidence_labels_requires_reader(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(datasets.ADAPTERS, "fake", _fake_worlds)
    monkeypatch.setattr(datasets, "NEEDS_READER", {"fake"})
    result, _ = _run(tmp_path, "--systems", "bm25", "--no-reader")
    assert result.exit_code != 0
    assert "--reader" in result.output


def test_compare_writes_ranking_with_confidence_intervals(tmp_path) -> None:
    result, out = _run(tmp_path, "--systems", "bm25,full_context", "--full", "--no-reader")
    assert result.exit_code == 0, result.output
    cmp = CliRunner().invoke(cli, ["bench", "compare", str(out), "--boot", "50"])
    assert cmp.exit_code == 0, cmp.output
    text = (out / "ranking.md").read_text()
    assert "### ndcg@10" in text and "95% CI" in text and "bm25" in text
    assert "full_context" not in text  # it ranks nothing, so it has no retrieval row


def test_compare_rejects_a_directory_without_results(tmp_path) -> None:
    cmp = CliRunner().invoke(cli, ["bench", "compare", str(tmp_path)])
    assert cmp.exit_code != 0 and "results.jsonl" in cmp.output
