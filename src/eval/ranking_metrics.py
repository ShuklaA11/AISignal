"""Ranking metrics for offline recommendation evaluation.

`dcg_score`, `ndcg_score`, and `mrr_score` reproduce the official MIND scorer
(microsoft/recommenders, `newsrec_utils.py`) so that results computed here are
directly comparable to published MIND leaderboard numbers. Deviating from those
formulas — for example using linear rather than exponential gains, or taking the
first positive rather than averaging over all of them — silently produces
numbers that cannot be compared to the literature.

`auc_score` is implemented natively via the tie-corrected Mann-Whitney rank
identity rather than pulling in scikit-learn, which is not a runtime dependency
of this project. The test suite pins it against `sklearn.metrics.roc_auc_score`.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# MIND reports metrics at these two cutoffs alongside AUC and MRR.
NDCG_CUTOFFS = (5, 10)


@dataclass(frozen=True)
class ImpressionMetrics:
    """The four MIND metrics for a single impression's candidate list."""

    auc: float
    mrr: float
    ndcg_5: float
    ndcg_10: float


def dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Discounted cumulative gain over the top-k of a score-ordered list.

    Uses exponential gains (2^rel - 1) and a log2(rank + 1) discount, matching
    the official MIND scorer.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    ranked = np.take(y_true, order[:k])
    gains = 2**ranked - 1
    discounts = np.log2(np.arange(len(ranked)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Normalized DCG@k. Returns 0.0 when the list holds no positive label."""
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    return dcg_score(y_true, y_score, k) / best


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean reciprocal rank averaged over *every* positive in the list.

    This is MIND's definition, which differs from the more common "reciprocal
    rank of the first relevant item".
    """
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    ranked = np.take(y_true, order)
    reciprocal_ranks = ranked / (np.arange(len(ranked)) + 1)
    return float(np.sum(reciprocal_ranks) / total_positives)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve, with ties resolved by average ranking.

    Equivalent to `sklearn.metrics.roc_auc_score` for binary labels. Returns
    0.5 when either class is absent, since AUC is undefined there.
    """
    labels = np.asarray(y_true)
    n_positive = int(np.sum(labels == 1))
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return 0.5

    ranks = _average_ranks(np.asarray(y_score, dtype=float))
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks ascending, with tied values sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)

    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                tied = order[start:index]
                ranks[tied] = ranks[tied].mean()
            start = index
    return ranks


def evaluate_impression(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Optional[ImpressionMetrics]:
    """Score one impression, or return None when it carries no usable signal.

    An impression whose candidates are all positive or all negative cannot
    discriminate between rankers, so it is excluded rather than scored as 0 or
    1 — including such rows would bias every method toward the same constant.
    Callers are expected to report how many impressions were skipped.
    """
    labels = np.asarray(y_true)
    positives = int(np.sum(labels == 1))
    if positives == 0 or positives == labels.size:
        return None

    scores = np.asarray(y_score, dtype=float)
    return ImpressionMetrics(
        auc=auc_score(labels, scores),
        mrr=mrr_score(labels, scores),
        ndcg_5=ndcg_score(labels, scores, k=5),
        ndcg_10=ndcg_score(labels, scores, k=10),
    )


def mean_metrics(records: Sequence[ImpressionMetrics]) -> ImpressionMetrics:
    """Average each metric across impressions (MIND's aggregation)."""
    if not records:
        return ImpressionMetrics(auc=0.0, mrr=0.0, ndcg_5=0.0, ndcg_10=0.0)
    count = len(records)
    return ImpressionMetrics(
        auc=sum(r.auc for r in records) / count,
        mrr=sum(r.mrr for r in records) / count,
        ndcg_5=sum(r.ndcg_5 for r in records) / count,
        ndcg_10=sum(r.ndcg_10 for r in records) / count,
    )
