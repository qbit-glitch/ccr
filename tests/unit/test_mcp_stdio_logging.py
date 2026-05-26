"""Tests for MCP stdio logging defaults."""

from __future__ import annotations

import io
import logging

from ccr.mcp.server import _configure_stdio_logging


def test_stdio_logging_quiet_by_default():
    """Default stdio logging suppresses INFO noise but keeps warnings."""
    root = logging.getLogger()
    old_root_level = root.level
    old_handlers = list(root.handlers)
    old_ccr_level = logging.getLogger("ccr").level
    old_mcp_level = logging.getLogger("mcp").level
    old_anyio_level = logging.getLogger("anyio").level

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    for existing in old_handlers:
        root.removeHandler(existing)
    root.addHandler(handler)

    try:
        _configure_stdio_logging(verbose=False)
        logger = logging.getLogger("ccr.core.storage.migration")
        logger.info("migration info noise")
        logger.warning("migration warning")

        output = stream.getvalue()
        assert "migration info noise" not in output
        assert "migration warning" in output
    finally:
        root.removeHandler(handler)
        for existing in old_handlers:
            root.addHandler(existing)
        root.setLevel(old_root_level)
        logging.getLogger("ccr").setLevel(old_ccr_level)
        logging.getLogger("mcp").setLevel(old_mcp_level)
        logging.getLogger("anyio").setLevel(old_anyio_level)


def test_stdio_logging_verbose_allows_info():
    """Verbose stdio logging keeps diagnostics available."""
    root = logging.getLogger()
    old_root_level = root.level
    old_handlers = list(root.handlers)
    old_ccr_level = logging.getLogger("ccr").level
    old_mcp_level = logging.getLogger("mcp").level
    old_anyio_level = logging.getLogger("anyio").level

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    for existing in old_handlers:
        root.removeHandler(existing)
    root.addHandler(handler)

    try:
        _configure_stdio_logging(verbose=True)
        logging.getLogger("ccr.core.storage.migration").info("migration info")

        assert "migration info" in stream.getvalue()
    finally:
        root.removeHandler(handler)
        for existing in old_handlers:
            root.addHandler(existing)
        root.setLevel(old_root_level)
        logging.getLogger("ccr").setLevel(old_ccr_level)
        logging.getLogger("mcp").setLevel(old_mcp_level)
        logging.getLogger("anyio").setLevel(old_anyio_level)
