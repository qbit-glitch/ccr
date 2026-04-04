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
