# Models

Two BERT classifiers fine-tuned for this project. Both run on CPU and are
loaded lazily, once per process (`core/models.py`, `@lru_cache(maxsize=1)`).

| Model | Base | Training data | Size | Loaded by |
|---|---|---|---|---|
| `claim_detector/final/` | bert-base-uncased | ~280k FEVER examples | 418 MB | `core/models.py:91` |
| `stance_detector/final/` | bert-base-cased | ~208k FEVER pairs | 413 MB | `core/models.py:116` |

Training notebooks are in [`notebooks/`](../notebooks/).

## They are not in this repository

Both directories are gitignored — 831 MB of weights does not belong in git.
Publishing them to the HuggingFace Hub is still outstanding; until that is
done, a fresh clone cannot run Tier 2.

**There is no silent fallback.** `core/models.py:41` raises with the missing
path rather than quietly substituting an off-the-shelf checkpoint, because a
system that reports a verdict from a model you did not train is worse than one
that refuses to start.

Point `.env` at wherever the weights live:

```bash
CLAIM_DETECTOR_PATH=models/claim_detector/final
STANCE_DETECTOR_PATH=models/stance_detector/final
```

## Stated limitations

Neither model is oversold anywhere in this codebase, and the docstrings carry
the same warnings:

- **Claim detector** reports 1.00 test accuracy, which is a red flag rather
  than a result. Probed by hand it separates *declarative sentences* from
  *questions and opinions* — which is what its training negatives actually
  were. It does not judge verifiability: `"asdkjh askjdh"` scores 0.997. It is
  used only as a cheap filter keeping questions out of the retrieval path.

- **Stance detector** scores 73.6% test accuracy, with REFUTES its weakest
  class (F1 0.74). Probed by hand it called a directly contradicting sentence
  SUPPORTS at 0.52 confidence. Its output therefore reaches the LLM as *a
  signal with a confidence attached*, never as the verdict — `core/prompts.py`
  instructs the model to weigh the evidence text over the stance label when
  the two disagree.

## Other directories here

`ner_model/`, `llm_finetuned/` and `language_detection/` are v1 artifacts. Tier 2
does not load them. The Mistral LoRA adapter was archived out of the tree to
`D:\verifai-archive\mistral_fv_adapter\`.
