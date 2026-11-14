"""Benchmarked retrieval systems. Importing this package registers them."""

from ccr.bench.retrievers.base import REGISTRY, Retriever, RetrieverUnavailable, register
from ccr.bench.retrievers import bm25 as _bm25  # noqa: F401  (registers "bm25")
from ccr.bench.retrievers import fullcontext as _fullcontext  # noqa: F401
from ccr.bench.retrievers import ccr_memory as _ccr_memory  # noqa: F401

__all__ = ["REGISTRY", "Retriever", "RetrieverUnavailable", "register"]
