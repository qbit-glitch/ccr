"""Tests for Memori-inspired semantic triple extraction."""

import json
import os
import tempfile
import threading

import pytest

from ccr.core.triples import Triple, TripleStore, _clean_entity


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "triples.json")


@pytest.fixture
def store(store_path):
    return TripleStore(store_path)


class TestTriple:
    def test_to_dict(self):
        t = Triple("auth module", "added_to", "API server", "C001", 0.8, "2026-01-01")
        d = t.to_dict()
        assert d["subject"] == "auth module"
        assert d["predicate"] == "added_to"
        assert d["object"] == "API server"

    def test_from_dict(self):
        d = {"subject": "X", "predicate": "fixed_in", "object": "Y", "source_commit": "C002"}
        t = Triple.from_dict(d)
        assert t.subject == "X"
        assert t.confidence == 0.8  # default

    def test_from_dict_with_confidence(self):
        d = {
            "subject": "X", "predicate": "fixed_in", "object": "Y",
            "source_commit": "C002", "confidence": 0.9, "timestamp": "2026-01-01",
        }
        t = Triple.from_dict(d)
        assert t.confidence == 0.9
        assert t.timestamp == "2026-01-01"

    def test_format_compact(self):
        t = Triple("auth", "added_to", "server", "C001")
        assert t.format_compact() == "auth --added_to--> server (C001)"

    def test_roundtrip(self):
        t = Triple("auth module", "added_to", "API server", "C001", 0.8, "2026-01-01T00:00:00")
        d = t.to_dict()
        t2 = Triple.from_dict(d)
        assert t.subject == t2.subject
        assert t.predicate == t2.predicate
        assert t.object == t2.object
        assert t.source_commit == t2.source_commit
        assert t.confidence == t2.confidence
        assert t.timestamp == t2.timestamp


class TestCleanEntity:
    def test_removes_articles(self):
        assert _clean_entity("the auth module") == "auth module"
        assert _clean_entity("a new feature") == "feature"
        assert _clean_entity("an API endpoint") == "API endpoint"

    def test_preserves_normal_text(self):
        assert _clean_entity("auth module") == "auth module"

    def test_strips_whitespace(self):
        assert _clean_entity("  spaced  ") == "spaced"

    def test_removes_all_prefix(self):
        assert _clean_entity("all tests") == "tests"

    def test_removes_some_prefix(self):
        assert _clean_entity("some modules") == "modules"

    def test_removes_new_prefix(self):
        assert _clean_entity("new parser") == "parser"

    def test_removes_old_prefix(self):
        assert _clean_entity("old implementation") == "implementation"

    def test_empty_string(self):
        assert _clean_entity("") == ""

    def test_only_whitespace(self):
        assert _clean_entity("   ") == ""


class TestExtractFromCommit:
    def test_added_to(self, store):
        triples = store.extract_from_commit(
            "C001", "Add auth to API", "Added JWT authentication to the login endpoint.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "added_to" in preds

    def test_refactored_into(self, store):
        triples = store.extract_from_commit(
            "C002", "", "Refactored database layer into repository pattern.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "refactored_into" in preds

    def test_fixed_in(self, store):
        triples = store.extract_from_commit(
            "C003", "", "Fixed rate limiting in login endpoint.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "fixed_in" in preds

    def test_replaced_by(self, store):
        triples = store.extract_from_commit(
            "C004", "", "Replaced old auth with new JWT system.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "replaced_by" in preds

    def test_created_for(self, store):
        triples = store.extract_from_commit(
            "C005", "", "Created migration script for database upgrade.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "created_for" in preds

    def test_implemented_in(self, store):
        triples = store.extract_from_commit(
            "C006", "", "Implemented caching in the API layer.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "implemented_in" in preds

    def test_updated_to(self, store):
        triples = store.extract_from_commit(
            "C007", "", "Updated dependencies to latest versions.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "updated_to" in preds

    def test_removed_from(self, store):
        triples = store.extract_from_commit(
            "C008", "", "Removed legacy code from utils module.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "removed_from" in preds

    def test_migrated_to(self, store):
        triples = store.extract_from_commit(
            "C009", "", "Migrated auth service to microservice architecture.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "migrated_to" in preds

    def test_split_into(self, store):
        triples = store.extract_from_commit(
            "C010", "", "Split monolith into separate services.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "split_into" in preds

    def test_merged_into(self, store):
        triples = store.extract_from_commit(
            "C011", "", "Merged feature branch into main.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "merged_into" in preds

    def test_renamed_to(self, store):
        triples = store.extract_from_commit(
            "C012", "", "Renamed config module to settings.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "renamed_to" in preds

    def test_moved_to(self, store):
        triples = store.extract_from_commit(
            "C013", "", "Moved test helpers to shared utils.", "", []
        )
        preds = [t.predicate for t in triples]
        assert "moved_to" in preds

    def test_file_level_triples(self, store):
        triples = store.extract_from_commit(
            "C005", "Update files", "", "", ["auth.py", "routes/login.py"]
        )
        file_triples = [t for t in triples if t.predicate == "modified_in"]
        assert len(file_triples) == 2
        subjects = {t.subject for t in file_triples}
        assert "auth.py" in subjects
        assert "routes/login.py" in subjects

    def test_file_triples_have_confidence_1(self, store):
        triples = store.extract_from_commit("C005", "", "", "", ["auth.py"])
        file_triples = [t for t in triples if t.predicate == "modified_in"]
        assert all(t.confidence == 1.0 for t in file_triples)

    def test_no_extraction_from_empty(self, store):
        triples = store.extract_from_commit("C006", "", "", "", [])
        assert triples == []

    def test_deduplication(self, store):
        store.extract_from_commit("C007", "Added auth to server.", "Added auth to server.", "", [])
        assert store.size <= 2  # Should not duplicate

    def test_multiple_extractions(self, store):
        triples = store.extract_from_commit(
            "C008", "",
            "Added rate limiting to API. Replaced old auth with JWT.",
            "", []
        )
        preds = [t.predicate for t in triples]
        assert len(preds) >= 2

    def test_short_entities_filtered(self, store):
        triples = store.extract_from_commit("C009", "", "Added a to b.", "", [])
        # "a" and "b" are too short (len <= 1), should be filtered
        non_file = [t for t in triples if t.predicate != "modified_in"]
        for t in non_file:
            assert len(t.subject) > 1
            assert len(t.object) > 1

    def test_extracts_from_title(self, store):
        triples = store.extract_from_commit(
            "C010", "Added authentication to server", "", "", []
        )
        preds = [t.predicate for t in triples]
        assert "added_to" in preds

    def test_extracts_from_why(self, store):
        triples = store.extract_from_commit(
            "C011", "", "", "Fixed performance issues in database queries.", []
        )
        preds = [t.predicate for t in triples]
        assert "fixed_in" in preds

    def test_timestamp_set(self, store):
        triples = store.extract_from_commit(
            "C012", "", "Added auth to server.", "", []
        )
        assert all(t.timestamp != "" for t in triples)

    def test_commit_id_in_triple(self, store):
        triples = store.extract_from_commit(
            "C012", "", "Added auth to server.", "", []
        )
        assert all(t.source_commit == "C012" for t in triples)


class TestSearch:
    def test_search_by_subject(self, store):
        store.extract_from_commit("C001", "", "Added authentication to API server.", "", [])
        results = store.search("authentication")
        assert len(results) > 0
        assert any(
            "authentication" in t.subject.lower() or "authentication" in t.object.lower()
            for t in results
        )

    def test_search_empty_query(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", [])
        results = store.search("")
        assert results == []

    def test_search_no_match(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", [])
        results = store.search("completely unrelated xyz query")
        assert results == []

    def test_search_respects_top_k(self, store):
        for i in range(10):
            store.extract_from_commit(f"C{i:03d}", f"Added feature{i} to module{i}.", "", "", [])
        results = store.search("feature", top_k=3)
        assert len(results) <= 3

    def test_search_ranks_by_relevance(self, store):
        store.extract_from_commit("C001", "", "Added authentication to API server.", "", [])
        store.extract_from_commit("C002", "", "Added caching to database layer.", "", [])
        results = store.search("authentication API")
        # The first result should be more relevant to "authentication API"
        if len(results) >= 2:
            assert "authentication" in results[0].subject.lower() or "api" in results[0].object.lower()


class TestGetByCommit:
    def test_get_by_commit(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", ["auth.py"])
        store.extract_from_commit("C002", "", "Fixed bug in parser.", "", ["parser.py"])
        results = store.get_by_commit("C001")
        assert all(t.source_commit == "C001" for t in results)

    def test_get_by_commit_no_match(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", [])
        results = store.get_by_commit("C999")
        assert results == []


class TestGetByEntity:
    def test_get_by_entity_subject(self, store):
        store.extract_from_commit("C001", "", "Added authentication to server.", "", [])
        results = store.get_by_entity("authentication")
        assert len(results) > 0

    def test_get_by_entity_object(self, store):
        store.extract_from_commit("C001", "", "Added auth to API server.", "", [])
        results = store.get_by_entity("server")
        assert len(results) > 0

    def test_get_by_entity_case_insensitive(self, store):
        store.extract_from_commit("C001", "", "Added Authentication to Server.", "", [])
        results = store.get_by_entity("authentication")
        assert len(results) > 0

    def test_get_by_entity_no_match(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", [])
        results = store.get_by_entity("nonexistent_entity")
        assert results == []


class TestPersistence:
    def test_save_and_load(self, store_path):
        s1 = TripleStore(store_path)
        s1.extract_from_commit("C001", "", "Added auth to server.", "", [])
        count = s1.size

        s2 = TripleStore(store_path)
        assert s2.size == count

    def test_corrupt_file(self, store_path):
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        with open(store_path, "w") as f:
            f.write("not json{{{")
        s = TripleStore(store_path)
        assert s.size == 0

    def test_nonexistent_path(self, tmp_path):
        path = str(tmp_path / "nonexistent" / "triples.json")
        s = TripleStore(path)
        assert s.size == 0

    def test_persistence_across_extractions(self, store_path):
        s1 = TripleStore(store_path)
        s1.extract_from_commit("C001", "", "Added auth to server.", "", [])
        s1.extract_from_commit("C002", "", "Fixed bug in parser.", "", ["parser.py"])
        count = s1.size

        s2 = TripleStore(store_path)
        assert s2.size == count

    def test_file_format(self, store_path):
        s = TripleStore(store_path)
        s.extract_from_commit("C001", "", "Added auth to server.", "", [])
        with open(store_path, "r") as f:
            data = json.load(f)
        assert "version" in data
        assert data["version"] == 1
        assert "triples" in data
        assert isinstance(data["triples"], list)


class TestFormatForContext:
    def test_empty_store(self, store):
        assert store.format_for_context() == ""

    def test_format(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", [])
        text = store.format_for_context()
        assert "Knowledge Graph" in text
        assert "added_to" in text

    def test_format_respects_top_k(self, store):
        for i in range(20):
            store.extract_from_commit(f"C{i:03d}", f"Added feature{i} to module{i}.", "", "", [])
        text = store.format_for_context(top_k=3)
        # Count lines starting with "- " (triple entries)
        triple_lines = [line for line in text.split("\n") if line.startswith("- ")]
        assert len(triple_lines) <= 3


class TestSize:
    def test_empty_size(self, store):
        assert store.size == 0

    def test_size_after_extraction(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", ["auth.py"])
        assert store.size > 0


class TestThreadSafety:
    def test_concurrent_extract(self, store):
        errors = []

        def extract(n):
            try:
                for i in range(5):
                    store.extract_from_commit(
                        f"C{n}{i}", f"Added feature{n}{i} to module{n}.", "", "", []
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=extract, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert store.size > 0

    def test_concurrent_search(self, store):
        store.extract_from_commit("C001", "", "Added auth to server.", "", [])
        errors = []

        def search_loop():
            try:
                for _ in range(10):
                    store.search("auth")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=search_loop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestBufferSizeEnforcement:
    """Tests for triple buffer size cap and eviction."""

    def test_enforce_buffer_size_evicts_over_limit(self, tmp_path):
        store = TripleStore(str(tmp_path / "triples.json"), max_buffer_size=5)
        for i in range(10):
            store.extract_from_commit(
                f"C{i:03d}", f"Title {i}", f"Added x{i} to y{i}", "reason", []
            )
        with store._lock:
            assert len(store._triples) <= 5

    def test_enforce_buffer_size_noop_under_limit(self, tmp_path):
        store = TripleStore(str(tmp_path / "triples.json"), max_buffer_size=100)
        store.extract_from_commit("C001", "Title", "Added foo to bar", "reason", [])
        with store._lock:
            count = len(store._triples)
        assert count >= 1
        assert count <= 100

    def test_enforce_keeps_highest_value(self, tmp_path):
        store = TripleStore(str(tmp_path / "triples.json"), max_buffer_size=3)
        # Add file triples (confidence=1.0) and text triples (confidence<1.0)
        store.extract_from_commit(
            "C001", "Title", "Added a to b", "reason",
            ["file1.py", "file2.py", "file3.py", "file4.py"],
        )
        with store._lock:
            remaining = store._triples
        # All remaining should be the highest-confidence ones
        assert len(remaining) == 3
        assert all(t.confidence >= 0.8 for t in remaining)


class TestGetRecent:
    """Tests for get_recent() public API."""

    def test_get_recent_returns_sorted(self, tmp_path):
        store = TripleStore(str(tmp_path / "triples.json"), max_buffer_size=100)
        store.extract_from_commit("C001", "First", "Added a to b", "r", ["f1.py"])
        store.extract_from_commit("C002", "Second", "Added c to d", "r", ["f2.py"])
        recent = store.get_recent(top_k=3)
        assert len(recent) >= 2
        # Verify descending timestamp order
        for i in range(len(recent) - 1):
            assert recent[i].timestamp >= recent[i + 1].timestamp

    def test_get_recent_respects_top_k(self, tmp_path):
        store = TripleStore(str(tmp_path / "triples.json"), max_buffer_size=100)
        for i in range(10):
            store.extract_from_commit(f"C{i:03d}", f"T{i}", f"Added x{i} to y{i}", "r", [])
        recent = store.get_recent(top_k=3)
        assert len(recent) <= 3


class TestFormatCompactEdgeCases:
    """Edge cases for Triple.format_compact()."""

    def test_special_chars_in_fields(self):
        """Arrows and parens in subject/object don't break format."""
        t = Triple(
            subject="file (v2)",
            predicate="replaced_by",
            object="file->v3",
            source_commit="C001",
            confidence=0.9,
            timestamp="2026-01-01T00:00:00Z",
        )
        result = t.format_compact()
        assert "--replaced_by-->" in result
        assert "file (v2)" in result
        assert "file->v3" in result

    def test_empty_subject_object(self):
        """Empty strings don't crash format_compact."""
        t = Triple(
            subject="",
            predicate="added_to",
            object="",
            source_commit="C001",
            confidence=0.5,
            timestamp="",
        )
        result = t.format_compact()
        assert "--added_to-->" in result

    def test_unicode_in_triple(self):
        """Unicode characters in triple fields are preserved."""
        t = Triple(
            subject="модуль",
            predicate="added_to",
            object="проект",
            source_commit="C001",
            confidence=0.8,
            timestamp="",
        )
        result = t.format_compact()
        assert "модуль" in result
        assert "проект" in result


class TestExtractionPatternEdgeCases:
    """Edge cases for regex extraction patterns."""

    def test_case_insensitive_verb_start(self, tmp_path):
        """Both 'Added' and 'added' should extract triples."""
        store = TripleStore(str(tmp_path / "t.json"))
        store.extract_from_commit("C001", "T", "added foo to bar", "", [])
        with store._lock:
            triples = list(store._triples)
        predicates = [t.predicate for t in triples]
        assert "added_to" in predicates

    def test_sentence_with_period_terminator(self, tmp_path):
        """Extraction should stop at period."""
        store = TripleStore(str(tmp_path / "t.json"))
        store.extract_from_commit("C001", "T", "Added auth to server. Done.", "", [])
        with store._lock:
            triples = [t for t in store._triples if t.predicate == "added_to"]
        assert len(triples) >= 1
        # Object should not include "Done"
        for t in triples:
            assert "Done" not in t.object

    def test_no_match_returns_file_triples_only(self, tmp_path):
        """When text has no matching verbs, only file triples are created."""
        store = TripleStore(str(tmp_path / "t.json"))
        store.extract_from_commit("C001", "T", "did nothing special", "", ["main.py"])
        with store._lock:
            triples = list(store._triples)
        # Should have at least the file triple (predicate is "modified_in")
        file_triples = [t for t in triples if t.predicate == "modified_in"]
        assert len(file_triples) == 1
        assert file_triples[0].subject == "main.py"

    def test_multiple_verbs_in_one_text(self, tmp_path):
        """Text with multiple matching verbs creates multiple triples."""
        store = TripleStore(str(tmp_path / "t.json"))
        store.extract_from_commit(
            "C001", "T",
            "Added auth to server. Removed debug from utils.",
            "", [],
        )
        with store._lock:
            predicates = [t.predicate for t in store._triples]
        assert "added_to" in predicates
        assert "removed_from" in predicates

    def test_very_long_text_no_redos(self, tmp_path):
        """Long repetitive text should not cause catastrophic backtracking."""
        import time
        store = TripleStore(str(tmp_path / "t.json"))
        # Pathological input for non-greedy patterns
        text = "Added " + "a " * 5000 + "to target"
        start = time.monotonic()
        store.extract_from_commit("C001", "T", text, "", [])
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"Extraction took {elapsed:.1f}s — possible ReDoS"


class TestCleanEntityEdgeCases:
    """Edge cases for _clean_entity() stop-prefix stripping."""

    def test_strips_whitespace(self):
        assert _clean_entity("  hello  ") == "hello"

    def test_strips_article_the(self):
        assert _clean_entity("the module") == "module"

    def test_strips_article_a(self):
        assert _clean_entity("a function") == "function"

    def test_strips_article_an(self):
        assert _clean_entity("an object") == "object"

    def test_strips_multiple_prefixes(self):
        """Chained prefixes are stripped iteratively."""
        assert _clean_entity("the new module") == "module"

    def test_case_insensitive_prefix(self):
        assert _clean_entity("The Module") == "Module"

    def test_no_prefix_passthrough(self):
        assert _clean_entity("server") == "server"

    def test_empty_string(self):
        assert _clean_entity("") == ""

    def test_only_prefix_word(self):
        """'the ' alone → stripped to 'the', prefix 'the ' won't match (needs trailing space)."""
        # After strip(), "the " becomes "the" which doesn't start with "the " (with space)
        assert _clean_entity("the ") == "the"

    def test_prefix_requires_trailing_space(self):
        """Prefix stripping requires the space (won't strip 'the' from 'theorem')."""
        assert _clean_entity("theorem") == "theorem"


class TestTripleSerializationEdgeCases:
    """Edge cases for Triple serialization round-trip."""

    def test_round_trip_preserves_fields(self, tmp_path):
        """Save and reload preserves all fields."""
        store = TripleStore(str(tmp_path / "t.json"))
        store.extract_from_commit("C001", "Title", "Added foo to bar", "why", ["f.py"])
        with store._lock:
            original = [t.to_dict() for t in store._triples]
        # Reload from disk
        store2 = TripleStore(str(tmp_path / "t.json"))
        with store2._lock:
            reloaded = [t.to_dict() for t in store2._triples]
        assert len(reloaded) == len(original)
        for orig, rel in zip(original, reloaded):
            assert orig["subject"] == rel["subject"]
            assert orig["predicate"] == rel["predicate"]

    def test_corrupted_json_handled(self, tmp_path):
        """Corrupted JSON file → empty store, no crash."""
        path = str(tmp_path / "t.json")
        with open(path, "w") as f:
            f.write("{invalid json!!!}")
        store = TripleStore(path)
        with store._lock:
            assert len(store._triples) == 0


# ── Round-5: Concurrency Edge Cases ──────────────────────────────────


class TestTripleConcurrency:
    """Thread safety for concurrent triple operations."""

    def test_concurrent_extraction(self, tmp_path):
        """4 threads extracting simultaneously — no errors or corruption."""
        import threading
        store = TripleStore(str(tmp_path / "t.json"), max_buffer_size=500)
        errors = []

        def extractor(thread_id):
            try:
                for i in range(20):
                    store.extract_from_commit(
                        f"C{thread_id}{i:02d}", f"Title-{thread_id}-{i}",
                        f"Added module_{thread_id}_{i} to system_{i}",
                        "reason", [f"file_{thread_id}_{i}.py"],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=extractor, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"Concurrent extraction failed: {errors}"
        with store._lock:
            assert len(store._triples) > 0

    def test_extract_dedup(self, tmp_path):
        """Same triple from 2 commits — dedup behavior verified."""
        store = TripleStore(str(tmp_path / "t.json"), max_buffer_size=100)
        store.extract_from_commit("C001", "T1", "Added auth to server", "r", [])
        store.extract_from_commit("C002", "T2", "Added auth to server", "r", [])
        with store._lock:
            auth_triples = [
                t for t in store._triples
                if t.predicate == "added_to" and "auth" in t.subject.lower()
            ]
        # Should have at most 2 (one per commit) — dedup removes exact duplicates
        assert len(auth_triples) <= 2
