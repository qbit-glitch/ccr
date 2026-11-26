"""``ccr bench``: judge-free benchmark commands."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess

import click

import ccr.context.embeddings as embeddings
from ccr.bench import datasets
from ccr.bench.metrics import tokenizer_name
from ccr.bench.reader import DEFAULT_MODEL, OllamaReader, ReaderUnavailable
from ccr.bench.report import build_report, load_published
from ccr.bench.results import ResultStore
from ccr.bench.retrievers import REGISTRY, RetrieverUnavailable
from ccr.bench.runner import run_world
from ccr.bench.stats import DEFAULT_BOOT, metrics_present, ranking_markdown
from ccr.bench.types import BenchConfig

logger = logging.getLogger(__name__)

PILOT_FRACTION = 0.1
DEFAULT_KS = "1,5,10,25,50,100"
DEFAULT_CACHE = os.path.join("~", ".cache", "ccr", "bench")
PUBLISHED = os.path.join(os.path.dirname(__file__), "..", "docs", "bench", "published.yaml")


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=os.path.dirname(__file__), timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _published() -> list:
    return load_published(PUBLISHED) if os.path.isfile(PUBLISHED) else []


_FINGERPRINT_KEYS = ("dataset", "pilot", "fraction", "seed", "reader", "ks", "reader_k", "sample_size",
                     "num_ctx", "tokenizer", "semantic_active")


def _pin_configuration(output: str, meta: dict[str, str]) -> None:
    """Refuse to mix results from different settings in one output directory."""
    pinned = {k: meta[k] for k in _FINGERPRINT_KEYS}
    path = os.path.join(output, "fingerprint.json")
    results = os.path.join(output, "results.jsonl")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != pinned:
            diff = ", ".join(f"{k}: {old.get(k)} -> {pinned[k]}"
                             for k in _FINGERPRINT_KEYS if old.get(k) != pinned[k])
            raise click.ClickException(
                f"{output} holds results from a different configuration ({diff}); "
                "use another --output directory")
    elif os.path.isfile(results) and os.path.getsize(results) > 0:
        raise click.ClickException(
            f"{output} has results but no recorded configuration; use another --output directory")
    _write(path, json.dumps(pinned, indent=2))


def _write_report(output: str, store: ResultStore, meta: dict[str, str]) -> None:
    report = build_report(store.all(), meta)
    _write(os.path.join(output, "report.json"), report.to_json())
    _write(os.path.join(output, "report.md"), report.to_markdown(_published()))


@click.group()
def bench() -> None:
    """Judge-free retrieval and memory benchmarks on public datasets."""


@bench.command("list")
def list_cmd() -> None:
    """List available datasets and systems."""
    click.echo("datasets: " + ", ".join(sorted(datasets.ADAPTERS)))
    click.echo("systems: " + ", ".join(sorted(REGISTRY)))


@bench.command("run")
@click.argument("dataset")
@click.option("--systems", default="", help="Comma-separated systems (default: all).")
@click.option("--pilot/--full", default=True, help="10% sample (default) or everything.")
@click.option("--fraction", default=0.0, type=float,
              help="Sample this share of worlds instead of 10% (development runs).")
@click.option("--reader/--no-reader", default=False, help="Answer with the local Ollama reader.")
@click.option("--output", "-o", default="", help="Output dir (default bench_out/<dataset>).")
@click.option("--cache-dir", default=DEFAULT_CACHE, help="Dataset download cache.")
@click.option("--seed", default=42, type=int)
@click.option("--ks", default=DEFAULT_KS, help="Comma-separated cutoffs.")
@click.option("--reader-k", default=5, type=int, help="Docs given to the reader.")
@click.option("--model", default=DEFAULT_MODEL, help="Ollama model tag for the reader.")
@click.option("--num-ctx", default=16384, type=int, help="Reader context window.")
def run_cmd(dataset: str, systems: str, pilot: bool, fraction: float, reader: bool, output: str,
            cache_dir: str, seed: int, ks: str, reader_k: int, model: str,
            num_ctx: int) -> None:
    """Run DATASET against the selected systems and write a report."""
    if dataset not in datasets.ADAPTERS:
        known = ", ".join(sorted(datasets.ADAPTERS)) or "(none)"
        raise click.ClickException(f"unknown dataset {dataset!r}; known: {known}")
    if dataset in datasets.NEEDS_READER and not reader:
        raise click.ClickException(
            f"{dataset} has no evidence labels, so only answer metrics exist: rerun with --reader"
        )
    if fraction and (not pilot or not 0.0 < fraction < 1.0):
        raise click.ClickException("--fraction must be between 0 and 1 and cannot be used with --full")
    share = fraction or (PILOT_FRACTION if pilot else 1.0)
    names = [s for s in systems.split(",") if s] or sorted(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise click.ClickException(f"unknown systems: {', '.join(unknown)}")
    output = output or os.path.join("bench_out", dataset)
    os.makedirs(output, exist_ok=True)
    cfg = BenchConfig(
        ks=tuple(int(k) for k in ks.split(",")), seed=seed,
        sample_fraction=share, chunk_tokens=512,
        cache_dir=os.path.expanduser(cache_dir), reader_k=reader_k,
    )
    worlds = datasets.ADAPTERS[dataset](cfg)
    sampler = datasets.SAMPLERS.get(dataset, datasets.default_sampler)
    sample = sampler(worlds, cfg)
    ollama = OllamaReader(model=model, num_ctx=num_ctx) if reader else None
    meta = {
        "dataset": dataset, "pilot": "true" if pilot else "false", "fraction": f"{share:g}",
        "seed": str(seed),
        "reader": ollama.tag if ollama else "none", "git_sha": _git_sha(),
        "semantic_active": str(embeddings.SEMANTIC_AVAILABLE), "ks": ks,
        "reader_k": str(reader_k), "sample_size": str(len(sample)),
        "num_ctx": str(num_ctx) if ollama else "-", "tokenizer": tokenizer_name(),
    }
    for stat, value in datasets.STATS.get(dataset, {}).items():
        meta[f"load.{stat}"] = str(value)
    _pin_configuration(output, meta)
    _write(os.path.join(output, "sample.json"),
           json.dumps({"world_ids": [w.id for w in sample]}))
    store = ResultStore(os.path.join(output, "results.jsonl"))
    limit = ollama.context_limit_tokens if ollama else None
    for name in names:
        _run_system(name, sample, cfg, ollama, store, limit, dataset, meta)
    _write(os.path.join(output, "meta.json"), json.dumps(meta, indent=2))
    _write_report(output, store, meta)
    click.echo(f"report written to {output}/report.md")


def _run_system(name: str, sample: list, cfg: BenchConfig, ollama: OllamaReader | None,
                store: ResultStore, limit: int | None, dataset: str,
                meta: dict[str, str]) -> None:
    factory = REGISTRY[name]
    cap = factory().max_k
    kept = tuple(k for k in cfg.ks if cap is None or k <= cap)
    dropped = [str(k) for k in cfg.ks if k not in kept]
    if dropped:
        meta[f"dropped_ks.{name}"] = ",".join(dropped)
    if not kept:
        meta[f"skipped.{name}"] = f"every k exceeds its cap of {cap}"
        return
    sys_cfg = dataclasses.replace(cfg, ks=kept)
    gains = datasets.GAINS.get(dataset)
    for i, world in enumerate(sample, start=1):
        click.echo(f"[{name}] world {i}/{len(sample)}: {world.id}")
        try:
            run_world(world, factory(), sys_cfg, ollama, store, limit, gains)
        except RetrieverUnavailable as exc:
            meta[f"skipped.{name}"] = str(exc)
            click.echo(f"[{name}] skipped: {exc}", err=True)
            return
        except ReaderUnavailable as exc:
            raise click.ClickException(str(exc)) from exc


@bench.command("report")
@click.argument("directory")
def report_cmd(directory: str) -> None:
    """Regenerate report files from a run directory."""
    with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    _write_report(directory, ResultStore(os.path.join(directory, "results.jsonl")), meta)
    click.echo(f"report rewritten in {directory}")


@bench.command("compare")
@click.argument("directory")
@click.option("--metrics", default="", help="Comma-separated metrics (default: headline ones).")
@click.option("--boot", default=DEFAULT_BOOT, type=int, help="Bootstrap resamples.")
@click.option("--seed", default=0, type=int)
def compare_cmd(directory: str, metrics: str, boot: int, seed: int) -> None:
    """Rank systems with 95% bootstrap CIs and paired tests; writes ranking.md."""
    path = os.path.join(directory, "results.jsonl")
    if not os.path.isfile(path):
        raise click.ClickException(f"no results.jsonl in {directory}")
    rows = ResultStore(path).all()
    chosen = [m for m in metrics.split(",") if m] or metrics_present(rows)
    text = f"# Ranking: {os.path.basename(os.path.normpath(directory))}\n\n"
    text += ranking_markdown(rows, chosen, boot, seed)
    _write(os.path.join(directory, "ranking.md"), text)
    click.echo(f"ranking written to {directory}/ranking.md")
