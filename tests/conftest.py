"""
Shared fixtures.

`init_native_libs()` runs at import, before pytest collects anything that could
pull in chromadb. Collection order is not something a test file controls, so
doing it here is the only reliable place — see `core.init_native_libs` for why
the wrong import order kills the interpreter outright.
"""

from __future__ import annotations

import pytest

from core import init_native_libs

init_native_libs()

from core.config import cfg  # noqa: E402


def _index_available() -> bool:
    return cfg.manifest_path(cfg.active_index).exists()


# Most tests are pure logic and need nothing on disk. The few that need a built
# index skip rather than fail, so `pytest` is still useful on a fresh clone
# before `ingest.run` has been executed.
needs_index = pytest.mark.skipif(
    not _index_available(),
    reason="no index published — run: python -m ingest.run --from-csv",
)

needs_models = pytest.mark.skipif(
    not cfg.claim_detector_path.exists(),
    reason="trained models not present — see README for how to obtain them",
)


@pytest.fixture(scope="session")
def retriever():
    """The assembled read path. Session-scoped: building it loads three models."""
    from serve.retriever import get_retriever

    return get_retriever()


@pytest.fixture
def doc():
    """Build a Document with arbitrary metadata, briefly."""
    from langchain_core.documents import Document

    def _make(text: str = "text", **metadata):
        return Document(page_content=text, metadata=metadata)

    return _make
