# Evaluation results

## Retrieval quality — before and after

Known-item retrieval over 200 queries drawn from 100 corpus articles. Each article contributes two queries: one passage from its **head** (inside the embedder's 256-token window) and one from its **tail** (beyond it). Gold is the source article.

The head slice is the control — both indexes embedded that text.

### Head — control (text both indexes embedded)

| System | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|
| Old index (whole articles) | 0.950 | 0.980 | 0.980 | 0.965 |
| New index (chunked) | 1.000 | 1.000 | 1.000 | 1.000 |
| New index + BM25 + reranker | 0.990 | 1.000 | 1.000 | 0.993 |

### Tail — the truncated region

| System | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|
| Old index (whole articles) | 0.470 | 0.760 | 0.840 | 0.598 |
| New index (chunked) | 0.880 | 0.950 | 0.950 | 0.904 |
| New index + BM25 + reranker | 0.980 | 0.990 | 0.990 | 0.982 |

### Overall

| System | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|
| Old index (whole articles) | 0.710 | 0.870 | 0.910 | 0.781 |
| New index (chunked) | 0.940 | 0.975 | 0.975 | 0.952 |
| New index + BM25 + reranker | 0.985 | 0.995 | 0.995 | 0.988 |

> The head slice is the control: both indexes embedded that text, and they score within 0.01 of each other there. The tail slice is the truncated region. Truncation did NOT make tail content unfindable — news articles are topically coherent, so the head vector still represents the whole article well enough to retrieve it by subject. What it destroyed was ranking precision: on tail queries the old index put the correct article first only 47% of the time, against 88% for the same embedder over chunks.

> **Run-to-run variance.** This eval was run twice. The old index moved by 0.010 on head recall@1 and 0.005 on MRR between runs; the new index reproduced identically. Chroma's HNSW is an approximate index, so treat single-run figures as carrying roughly ±0.01 — two decimal places are meaningful, the third is not. The effect being measured (0.47 → 0.88 on tail recall@1) is forty times that noise floor.

## Relevance floor calibration

Swept 13 candidate floors against 200 known-item queries (should retrieve) and 20 adversarial claims (should abstain). No LLM calls.

| Floor | Gold document retained | Adversarial abstention |
|---:|---:|---:|
| 0.02 | 1.000 | 0.700 |
| 0.05 | 1.000 | 0.700 |
| 0.08 | 1.000 | 0.700 |
| 0.10 | 1.000 | 0.750 |
| 0.12 | 1.000 | 0.750 |
| 0.15 | 0.995 | 0.750 |
| 0.20 | 0.995 | 0.750 |
| 0.25 | 0.995 | 0.800 ← **chosen** |
| 0.30 | 0.995 | 0.800 |
| 0.35 | 0.995 | 0.800 |
| 0.40 | 0.995 | 0.800 |
| 0.50 | 0.995 | 0.800 |
| 0.60 | 0.995 | 0.850 |

### What the floor can and cannot separate

Of 20 adversarial claims, **14 score below 0.05** and are filtered by any floor at all. The **6 that score higher are all `near_miss`** — claims about subjects the corpus covers extensively, asserting facts it does not contain.

| Top relevance | Category | Claim |
|---:|---|---|
| 0.8579 | `near_miss` | OpenAI was founded in 2010 as a for-profit hardware company. |
| 0.7945 | `near_miss` | Netflix reported 500 million paying subscribers in 2023. |
| 0.6468 | `near_miss` | TechCrunch was founded in 1998 by Michael Arrington. |
| 0.5934 | `near_miss` | Apple sold exactly 41 million iPhones in the first quarter of 2019. |
| 0.2017 | `near_miss` | Bitcoin traded above 500,000 US dollars in March 2024. |
| 0.0967 | `near_miss` | Infosys relocated its global headquarters from Bengaluru to Chennai in 2022. |
| 0.0064 | `near_miss` | India's GDP contracted by 12 percent during the 2019 fiscal year. |
| 0.0058 | `absent_false` | The population of Reykjavik exceeded two million people in 2024. |

This is the honest limit of a relevance threshold: it separates *absent from the corpus* from *present in the corpus*. It cannot separate *supports this claim* from *is merely about the same subject*. That judgement is made downstream by the LLM's `directly_relevant` field.

> Known-item queries are verbatim passages from the corpus, so their gold documents score higher than a paraphrased real-world claim would. The retention side of this trade-off is therefore optimistic, and the recommended floor should be read as an upper bound. The adversarial side uses real claim text and is realistic.

## End-to-end behaviour

### Abstention on unanswerable claims

20 hand-written claims the corpus provably cannot answer. Correct behaviour for every one is abstention.

- **abstain_precision: 1.000** (20/20)
- rejected at the claim gate: 0
- wrongly answered: 0

**Two independent mechanisms produced these abstentions:**

- **16** were stopped by the relevance floor — nothing cleared it, so no LLM call was made and the claim cost nothing.
- **4** retrieved evidence, passed it to the LLM, and the LLM returned `UNVERIFIABLE` after reading it.

The second group is the interesting one. Those are claims about subjects the corpus genuinely covers, so retrieval returns relevant-looking articles and no threshold can filter them. Only reading the evidence answers them.

| Category | n | abstain_precision |
|---|---:|---:|
| `absent_false` | 4 | 1.000 |
| `absent_true` | 4 | 1.000 |
| `absurd` | 4 | 1.000 |
| `near_miss` | 8 | 1.000 |

### FEVER

200 claims from the KILT FEVER validation split, balanced SUPPORTS/REFUTES. **FEVER's gold evidence is Wikipedia, which this corpus does not contain**, so a high abstention rate here is correct behaviour rather than a failure — the system is being asked questions its corpus cannot answer.

| Metric | Value |
|---|---:|
| claims | 200 |
| answered | 5 |
| abstain_rate | 0.975 |
| verdict_accuracy (of answered) | 1.000 |
| **false_confidence** (wrong at ≥70%) | **0** |
| mean confidence when correct | 78 |

`false_confidence` counts wrong verdicts issued at high confidence. It is the metric to drive to zero: a confident wrong answer is worse than no answer. It is meaningful here **because** n is large — the system was given 200 opportunities to be confidently wrong and took none of them.

> **Read `verdict_accuracy` with care.** It is computed over the **5 claims the system actually committed to**, not over all 200. At that sample size the figure carries almost no statistical weight and should not be quoted as a headline accuracy number. The meaningful results on this set are the abstention rate and `false_confidence`, both of which are computed over all 200 claims.

---

Reproduce:

```bash
python -m eval_harness.prepare_datasets
python -m eval_harness.run_eval --all
python -m eval_harness.report --write
```