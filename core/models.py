"""
Lazy singletons for the four local CPU models.

Replaces `src/nlp/model_manager.py`. Three things changed:

* **No YAML config, no placeholder fallback.** The old manager could silently
  swap a zero-shot model in for the trained claim detector and carry on, which
  meant a broken model path produced *different results* rather than an error.
  Both trained models are in the repo; if one is missing, that is a startup
  failure with a path in the message.
* **Loaded on first use, cached forever.** Importing this module costs nothing.
  A CLI run that rejects at the gate never pays to load the stance model.
* **`init_native_libs()` is called here**, once, before any model touches disk.

Every model is CPU-only and loads from the local HF cache (`HF_HUB_OFFLINE`).
Combined footprint is roughly 1 GB and ~10 s on first touch.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

# Must precede anything that can pull in chromadb — importing chromadb before
# sentence_transformers kills the interpreter with an access violation on
# Windows. See core.init_native_libs for the full diagnosis.
from core import init_native_libs

init_native_libs()

from core.config import cfg  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - import-cost avoidance
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def _require_dir(path, what: str):
    """Fail with a path, not an AttributeError six frames deep."""
    if not path.exists():
        raise RuntimeError(
            f"{what} not found at {path}\n"
            f"  The trained models ship with the repo — check the path in .env."
        )
    return path


@lru_cache(maxsize=1)
def embedder() -> HuggingFaceEmbeddings:
    """
    The query-side embedder.

    Deliberately **not** wrapped in `CacheBackedEmbeddings` the way the ingest
    side is: that cache is keyed on text, and queries are near-unique, so it
    would grow without ever hitting. `normalize_embeddings` must match what the
    index was built with or every distance is wrong — `serve.index` checks the
    model name against the manifest for exactly this reason.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.debug("Loading embedder %s", cfg.embed_model)
    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def cross_encoder() -> HuggingFaceCrossEncoder:
    """
    Reranker. Scores (query, passage) pairs jointly, which is why it beats the
    bi-encoder that produced the candidates — and why it is far too slow to run
    over the corpus, so it only ever sees the ~20 candidates retrieval returns.

    Output is a raw logit, roughly [-11, +7]. `core.compressors` squashes it.
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    logger.debug("Loading cross-encoder %s", cfg.rerank_model)
    return HuggingFaceCrossEncoder(
        model_name=cfg.rerank_model,
        model_kwargs={"device": cfg.device},
    )


@lru_cache(maxsize=1)
def claim_detector():
    """
    Trained binary claim detector (bert-base-uncased, 280k FEVER examples).

    Returns `(tokenizer, model)`. `LABEL_1` = is a claim.

    KNOWN LIMITATION — do not oversell this model. Its reported test accuracy is
    1.00, which is a red flag, not a triumph: probed by hand it separates
    *declarative sentences* from *questions and opinions*, which is what its
    training negatives actually were. It does not judge verifiability —
    `"asdkjh askjdh"` scores 0.997 as a claim. It is a cheap filter that keeps
    questions and chit-chat out of the retrieval path, and that is all it is
    used for here.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = _require_dir(cfg.claim_detector_path, "Claim detector")
    logger.debug("Loading claim detector from %s", path)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def stance_detector():
    """
    Trained FEVER stance model (bert-base-cased, 208k pairs).

    Returns `(tokenizer, model, id2label)` with labels
    `SUPPORTS / REFUTES / NOT ENOUGH INFO`.

    KNOWN LIMITATION: 73.6% test accuracy, and REFUTES is its weakest class
    (F1 0.74). Probed by hand it called a directly contradicting sentence
    SUPPORTS at 0.52 confidence. Its output is therefore fed to the LLM as a
    *signal with a confidence attached*, never as the verdict — see
    `core.prompts`, which instructs the model to weigh the evidence text over
    the stance label when the two disagree.
    """
    import json

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = _require_dir(cfg.stance_detector_path, "Stance detector")
    logger.debug("Loading stance detector from %s", path)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()

    # Prefer the sidecar labels.json: the checkpoint's own config.json was
    # written without an id2label map for the claim model, and relying on
    # config alone would silently give LABEL_0/1/2 here too.
    labels_file = path / "labels.json"
    if labels_file.exists():
        raw = json.loads(labels_file.read_text(encoding="utf-8"))["id2label"]
        id2label = {int(k): v for k, v in raw.items()}
    else:
        id2label = dict(model.config.id2label)

    return tokenizer, model, id2label


def warm_up() -> None:
    """Load everything now rather than mid-request. Used by the CLI and API."""
    embedder()
    cross_encoder()
    claim_detector()
    stance_detector()
    logger.info("All local models loaded")
