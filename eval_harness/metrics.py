"""
Metric definitions.

Kept separate from the runners so each definition is one readable function that
can be checked against what the README will claim. A metric whose definition is
buried inside a loop is a metric nobody can audit — including the person who
has to defend the number in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


# ── Retrieval ────────────────────────────────────────────────────────────────

def recall_at_k(ranked_urls: list[str], gold_url: str, k: int) -> float:
    """1.0 if the gold document appears anywhere in the top k, else 0.0."""
    return 1.0 if gold_url in ranked_urls[:k] else 0.0


def reciprocal_rank(ranked_urls: list[str], gold_url: str) -> float:
    """
    1/rank of the gold document, 0 if absent.

    Reported alongside recall because recall@5 is a step function: it cannot
    tell "ranked first" from "ranked fifth", and a change that moves gold from
    rank 5 to rank 1 is a real improvement recall@5 scores as zero change.
    """
    for i, url in enumerate(ranked_urls, start=1):
        if url == gold_url:
            return 1.0 / i
    return 0.0


@dataclass
class RetrievalScore:
    """Aggregate retrieval quality for one system over one slice of queries."""

    system: str
    slice_name: str
    n: int = 0
    hits_at_1: list[float] = field(default_factory=list)
    hits_at_5: list[float] = field(default_factory=list)
    hits_at_10: list[float] = field(default_factory=list)
    rr: list[float] = field(default_factory=list)

    def add(self, ranked_urls: list[str], gold_url: str) -> None:
        self.n += 1
        self.hits_at_1.append(recall_at_k(ranked_urls, gold_url, 1))
        self.hits_at_5.append(recall_at_k(ranked_urls, gold_url, 5))
        self.hits_at_10.append(recall_at_k(ranked_urls, gold_url, 10))
        self.rr.append(reciprocal_rank(ranked_urls, gold_url))

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "slice": self.slice_name,
            "n": self.n,
            "recall@1": round(mean(self.hits_at_1), 4) if self.n else 0.0,
            "recall@5": round(mean(self.hits_at_5), 4) if self.n else 0.0,
            "recall@10": round(mean(self.hits_at_10), 4) if self.n else 0.0,
            "mrr": round(mean(self.rr), 4) if self.n else 0.0,
        }


# ── Verdicts ─────────────────────────────────────────────────────────────────

# FEVER is two-class in KILT (NOT ENOUGH INFO is dropped). The system emits five
# verdict labels, so they are collapsed onto FEVER's vocabulary to be scored.
# UNVERIFIABLE deliberately maps to None: it is not a wrong answer, it is a
# refusal to answer, and folding it into either class would misrepresent both
# accuracy and abstention.
_VERDICT_TO_FEVER = {
    "TRUE":         "SUPPORTS",
    "MOSTLY_TRUE":  "SUPPORTS",
    "MOSTLY_FALSE": "REFUTES",
    "FALSE":        "REFUTES",
    "UNVERIFIABLE": None,
}


def to_fever_label(verdict: str) -> str | None:
    """Collapse a five-label verdict onto FEVER's two classes, or None."""
    return _VERDICT_TO_FEVER.get(verdict)


@dataclass
class VerdictScore:
    """
    System behaviour over a labelled claim set.

    Four numbers, and the relationship between them is the point:

      answered        cases where a verdict was actually committed to
      accuracy        of those, how many matched gold
      abstain_rate    how often it declined
      false_confidence    WRONG verdicts issued above the confidence threshold

    A system can reach high accuracy by answering only the easy cases, so
    accuracy is never reported without `answered` beside it. `false_confidence`
    is the one to drive to zero: a confident wrong answer is worse than none.
    """

    n: int = 0
    answered: int = 0
    correct: int = 0
    abstained: int = 0
    rejected: int = 0
    confident_wrong: int = 0
    confident_threshold: int = 70
    confidences_correct: list[int] = field(default_factory=list)
    confidences_wrong: list[int] = field(default_factory=list)

    def add(self, *, outcome: str, verdict: str | None,
            confidence: int, gold_label: str | None) -> None:
        self.n += 1

        if outcome == "rejected":
            self.rejected += 1
            return
        if outcome == "abstained":
            self.abstained += 1
            return

        predicted = to_fever_label(verdict or "")
        if predicted is None:
            # Read the evidence, still declined to commit. Counted as an
            # abstention in behaviour terms, because that is what it is.
            self.abstained += 1
            return

        self.answered += 1
        if gold_label is None:
            return

        if predicted == gold_label:
            self.correct += 1
            self.confidences_correct.append(confidence)
        else:
            self.confidences_wrong.append(confidence)
            if confidence >= self.confident_threshold:
                self.confident_wrong += 1

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "answered": self.answered,
            "abstained": self.abstained,
            "rejected": self.rejected,
            "abstain_rate": round(self.abstained / self.n, 4) if self.n else 0.0,
            "verdict_accuracy": (
                round(self.correct / self.answered, 4) if self.answered else None
            ),
            "false_confidence": self.confident_wrong,
            f"false_confidence_rate@{self.confident_threshold}": (
                round(self.confident_wrong / self.answered, 4)
                if self.answered else None
            ),
            "mean_confidence_when_correct": (
                round(mean(self.confidences_correct), 1)
                if self.confidences_correct else None
            ),
            "mean_confidence_when_wrong": (
                round(mean(self.confidences_wrong), 1)
                if self.confidences_wrong else None
            ),
        }


# ── Abstention ───────────────────────────────────────────────────────────────

@dataclass
class AbstainScore:
    """
    Behaviour on claims the corpus provably cannot answer.

    Every item's correct outcome is abstention, so this is a single rate. It is
    reported per category as well as overall, because the categories are graded
    by difficulty and the average hides that — `near_miss` failing while
    `absurd` passes is the interesting result, and it is invisible in the mean.
    """

    n: int = 0
    abstained: int = 0
    rejected: int = 0
    answered: int = 0
    # WHICH mechanism did the abstaining. The system has two independent ones,
    # and knowing the split is the difference between "abstention works" and
    # understanding why — the floor cannot catch a topically-present claim, so
    # if every abstention came from the floor, the near_miss cases were never
    # actually being handled.
    by_floor: int = 0
    by_llm: int = 0
    by_category: dict[str, list[bool]] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)

    def add(self, *, item: dict, outcome: str, verdict: str | None,
            confidence: int, evidence_count: int = 0) -> None:
        self.n += 1
        # A gate rejection also avoids the fabricated report, but it is a
        # different behaviour and is counted separately rather than as success.
        llm_abstained = outcome == "verified" and verdict == "UNVERIFIABLE"
        correct = outcome == "abstained" or llm_abstained

        mechanism = "answered"
        if outcome == "rejected":
            self.rejected += 1
            mechanism = "claim_gate"
        elif outcome == "abstained":
            # Never reached the LLM: nothing cleared the relevance floor.
            self.abstained += 1
            self.by_floor += 1
            mechanism = "relevance_floor"
        elif llm_abstained:
            # Evidence was retrieved and read, and judged not to support a
            # verdict. This is the path that handles near_miss.
            self.abstained += 1
            self.by_llm += 1
            mechanism = "llm_judgement"
        else:
            self.answered += 1
            self.failures.append({
                "id": item.get("id"),
                "claim": item.get("claim"),
                "category": item.get("category"),
                "verdict": verdict,
                "confidence": confidence,
            })

        self.details.append({
            "id": item.get("id"),
            "category": item.get("category"),
            "claim": item.get("claim"),
            "mechanism": mechanism,
            "evidence": evidence_count,
            "verdict": verdict,
            "confidence": confidence,
        })
        self.by_category.setdefault(item.get("category", "?"), []).append(correct)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "abstained": self.abstained,
            "rejected_at_gate": self.rejected,
            "wrongly_answered": self.answered,
            "abstain_precision": (
                round(self.abstained / self.n, 4) if self.n else 0.0
            ),
            "abstained_by_relevance_floor": self.by_floor,
            "abstained_by_llm_judgement": self.by_llm,
            "details": self.details,
            "by_category": {
                category: {
                    "n": len(results),
                    "abstain_precision": round(sum(results) / len(results), 4),
                }
                for category, results in sorted(self.by_category.items())
            },
            "failures": self.failures,
        }
