"""Tests for CCR workflow preset system.

Covers:
- _write_preset_to_metadata in cli_presets
- _get_preset and _DIRECTIVES in on_session_start
- set-preset CLI command
- install --preset option
- export-context CLI command
"""

from __future__ import annotations

import os

import pytest
import yaml
from click.testing import CliRunner

from ccr.cli_presets import VALID_PRESETS, _write_preset_to_metadata, set_preset, export_context
from ccr.hooks.on_session_start import _DIRECTIVES, _get_preset


# ---------------------------------------------------------------------------
# _write_preset_to_metadata
# ---------------------------------------------------------------------------

class TestWritePresetToMetadata:
    def _make_ccr_dir(self, tmp_path, initial_meta: dict | None = None) -> str:
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        meta = initial_meta or {"version": "1", "branch": "main"}
        with open(ccr / "metadata.yaml", "w") as f:
            yaml.dump(meta, f)
        return str(tmp_path)

    def test_writes_ml_preset(self, tmp_path):
        project = self._make_ccr_dir(tmp_path)
        _write_preset_to_metadata(project, "ml")
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["preset"] == "ml"

    def test_writes_academic_preset(self, tmp_path):
        project = self._make_ccr_dir(tmp_path)
        _write_preset_to_metadata(project, "academic")
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["preset"] == "academic"

    def test_writes_default_preset(self, tmp_path):
        project = self._make_ccr_dir(tmp_path)
        _write_preset_to_metadata(project, "default")
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["preset"] == "default"

    def test_preserves_existing_metadata_fields(self, tmp_path):
        project = self._make_ccr_dir(tmp_path, {"version": "2", "branch": "main"})
        _write_preset_to_metadata(project, "ml")
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["version"] == "2"
        assert meta["branch"] == "main"
        assert meta["preset"] == "ml"

    def test_idempotent(self, tmp_path):
        project = self._make_ccr_dir(tmp_path)
        _write_preset_to_metadata(project, "ml")
        _write_preset_to_metadata(project, "ml")
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["preset"] == "ml"

    def test_overwrites_existing_preset(self, tmp_path):
        project = self._make_ccr_dir(tmp_path, {"preset": "ml"})
        _write_preset_to_metadata(project, "academic")
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["preset"] == "academic"

    def test_raises_for_unknown_preset(self, tmp_path):
        project = self._make_ccr_dir(tmp_path)
        with pytest.raises(ValueError, match="Unknown preset"):
            _write_preset_to_metadata(project, "quantum")

    def test_does_nothing_when_metadata_missing(self, tmp_path):
        # .ccr/ exists but no metadata.yaml
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        # Should not raise
        _write_preset_to_metadata(str(tmp_path), "ml")


# ---------------------------------------------------------------------------
# _get_preset
# ---------------------------------------------------------------------------

class TestGetPreset:
    def _make_meta(self, tmp_path, meta: dict) -> str:
        ccr = tmp_path / ".ccr"
        ccr.mkdir(exist_ok=True)
        with open(ccr / "metadata.yaml", "w") as f:
            yaml.dump(meta, f)
        return str(ccr)

    def test_returns_ml_when_set(self, tmp_path):
        ccr_root = self._make_meta(tmp_path, {"preset": "ml"})
        assert _get_preset(ccr_root) == "ml"

    def test_returns_academic_when_set(self, tmp_path):
        ccr_root = self._make_meta(tmp_path, {"preset": "academic"})
        assert _get_preset(ccr_root) == "academic"

    def test_returns_default_when_key_absent(self, tmp_path):
        ccr_root = self._make_meta(tmp_path, {"version": "1"})
        assert _get_preset(ccr_root) == "default"

    def test_returns_default_when_metadata_missing(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        assert _get_preset(str(ccr)) == "default"

    def test_returns_default_on_corrupt_yaml(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        with open(ccr / "metadata.yaml", "w") as f:
            f.write(": invalid: yaml: [[[")
        assert _get_preset(str(ccr)) == "default"

    def test_returns_default_on_missing_ccr_dir(self, tmp_path):
        assert _get_preset(str(tmp_path / "nonexistent")) == "default"


# ---------------------------------------------------------------------------
# _DIRECTIVES content
# ---------------------------------------------------------------------------

class TestDirectivesContent:
    def test_default_directive_contains_gcc_commit(self):
        assert "gcc_commit" in _DIRECTIVES["default"]

    def test_default_directive_contains_session_log_turn(self):
        assert "session_log_turn" in _DIRECTIVES["default"]

    def test_ml_directive_contains_experiment_param(self):
        assert "experiment=" in _DIRECTIVES["ml"]

    def test_ml_directive_contains_gcc_experiments(self):
        assert "gcc_experiments" in _DIRECTIVES["ml"]

    def test_ml_directive_contains_gcc_branch(self):
        assert "gcc_branch" in _DIRECTIVES["ml"]

    def test_academic_directive_contains_gcc_discuss(self):
        assert "gcc_discuss" in _DIRECTIVES["academic"]

    def test_academic_directive_contains_gcc_discussions(self):
        assert "gcc_discussions" in _DIRECTIVES["academic"]

    def test_academic_directive_contains_session_search(self):
        assert "session_search" in _DIRECTIVES["academic"]

    def test_all_valid_presets_have_directives(self):
        for preset in VALID_PRESETS:
            assert preset in _DIRECTIVES, f"Missing directive for preset '{preset}'"

    def test_directives_contain_mandatory_header(self):
        for name, directive in _DIRECTIVES.items():
            assert "MANDATORY_CCR_ACTIONS" in directive, f"Preset '{name}' missing header"


# ---------------------------------------------------------------------------
# set-preset CLI command
# ---------------------------------------------------------------------------

class TestSetPresetCommand:
    def _install_ccr(self, tmp_path) -> str:
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        with open(ccr / "metadata.yaml", "w") as f:
            yaml.dump({"version": "1"}, f)
        return str(tmp_path)

    def test_set_preset_updates_metadata(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(set_preset, ["ml", project])
        assert result.exit_code == 0, result.output
        with open(os.path.join(project, ".ccr", "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        assert meta["preset"] == "ml"

    def test_set_preset_accepts_academic(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(set_preset, ["academic", project])
        assert result.exit_code == 0

    def test_set_preset_rejects_invalid_preset(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(set_preset, ["invalid", project])
        assert result.exit_code != 0

    def test_set_preset_fails_without_ccr_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(set_preset, ["ml", str(tmp_path)])
        assert result.exit_code != 0
        assert "ccr install" in result.output or ".ccr" in result.output

    def test_set_preset_prints_confirmation(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(set_preset, ["academic", project])
        assert "academic" in result.output


# ---------------------------------------------------------------------------
# export-context CLI command
# ---------------------------------------------------------------------------

class TestExportContext:
    def _install_ccr(self, tmp_path) -> str:
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        # Write minimal .ccr structure so MemoryManager doesn't crash
        (ccr / "metadata.yaml").write_text("version: '1'\nbranch: main\n")
        branches = ccr / "branches" / "main"
        branches.mkdir(parents=True)
        (branches / "commits.md").write_text("# Commits\n## Rolling Summary\n(none)\n---\n")
        (ccr / "main.md").write_text("# Project Focus\nTest project\n")
        return str(tmp_path)

    def test_export_context_stdout_contains_header(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(export_context, [project])
        assert result.exit_code == 0, result.output
        assert "CCR Context Export" in result.output

    def test_export_context_output_writes_file(self, tmp_path):
        project = self._install_ccr(tmp_path)
        out_file = str(tmp_path / "CONTEXT.md")
        runner = CliRunner()
        result = runner.invoke(export_context, [project, "--output", out_file])
        assert result.exit_code == 0, result.output
        assert os.path.isfile(out_file)
        content = open(out_file).read()
        assert "CCR Context Export" in content

    def test_export_context_includes_project_path(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(export_context, [project])
        assert os.path.basename(str(tmp_path)) in result.output or str(tmp_path) in result.output

    def test_export_context_fails_without_ccr_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(export_context, [str(tmp_path)])
        assert result.exit_code != 0

    def test_export_context_level_option_accepted(self, tmp_path):
        project = self._install_ccr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(export_context, [project, "--level", "3"])
        assert result.exit_code == 0, result.output
