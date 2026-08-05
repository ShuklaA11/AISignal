"""Tests for ranking metrics used in the MIND benchmark evaluation.

The implementations must match the official MIND scorer (microsoft/recommenders
`newsrec_utils.py`) exactly, otherwise reported numbers are not comparable to
published baselines. AUC is cross-checked against sklearn's `roc_auc_score`.
"""

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from src.eval.ranking_metrics import (
    ImpressionMetrics,
    auc_score,
    dcg_score,
    evaluate_impression,
    mean_metrics,
    mrr_score,
    ndcg_score,
)

pytestmark = pytest.mark.unit


# ── nDCG ─────────────────────────────────────────────────────────────


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    # Arrange
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])

    # Act
    result = ndcg_score(y_true, y_score, k=10)

    # Assert
    assert result == pytest.approx(1.0)


def test_ndcg_discounts_a_positive_held_at_rank_two() -> None:
    # Arrange — single positive sitting second in the ranking
    y_true = np.array([0, 1, 0])
    y_score = np.array([3.0, 2.0, 1.0])

    # Act
    result = ndcg_score(y_true, y_score, k=10)

    # Assert — DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert result == pytest.approx(1.0 / math.log2(3))


def test_ndcg_is_zero_when_positives_fall_outside_k() -> None:
    # Arrange — the only positive is ranked last, beyond k=2
    y_true = np.array([0, 0, 0, 1])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])

    # Act
    result = ndcg_score(y_true, y_score, k=2)

    # Assert
    assert result == pytest.approx(0.0)


def test_ndcg_uses_exponential_gains_for_graded_relevance() -> None:
    # Arrange — grade 2 at rank 2, grade 1 at rank 1
    y_true = np.array([1, 2])
    y_score = np.array([2.0, 1.0])

    # Act
    result = ndcg_score(y_true, y_score, k=10)

    # Assert — gains are 2^rel - 1, so grade 2 contributes 3
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert result == pytest.approx(dcg / idcg)


def test_dcg_truncates_k_to_the_candidate_count() -> None:
    # Arrange — k larger than the list length must not error or pad
    y_true = np.array([1, 0])
    y_score = np.array([2.0, 1.0])

    # Act
    result = dcg_score(y_true, y_score, k=10)

    # Assert
    assert result == pytest.approx(1.0)


# ── MRR ──────────────────────────────────────────────────────────────


def test_mrr_is_one_when_a_positive_leads_the_ranking() -> None:
    # Arrange
    y_true = np.array([1, 0, 0])
    y_score = np.array([3.0, 2.0, 1.0])

    # Act
    result = mrr_score(y_true, y_score)

    # Assert
    assert result == pytest.approx(1.0)


def test_mrr_averages_reciprocal_ranks_across_all_positives() -> None:
    # Arrange — positives land at ranks 1 and 3
    y_true = np.array([1, 0, 1])
    y_score = np.array([3.0, 2.0, 1.0])

    # Act
    result = mrr_score(y_true, y_score)

    # Assert — MIND's MRR averages over every positive, not just the first
    assert result == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)


# ── AUC ──────────────────────────────────────────────────────────────


def test_auc_is_one_when_positives_outrank_every_negative() -> None:
    # Arrange
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])

    # Act
    result = auc_score(y_true, y_score)

    # Assert
    assert result == pytest.approx(1.0)


def test_auc_is_half_when_every_score_is_tied() -> None:
    # Arrange — a constant scorer carries no ranking information
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([2.0, 2.0, 2.0, 2.0])

    # Act
    result = auc_score(y_true, y_score)

    # Assert
    assert result == pytest.approx(0.5)


def test_auc_matches_sklearn_on_random_inputs_including_ties() -> None:
    # Arrange — coarse scores force ties, which is where naive AUC drifts
    rng = np.random.RandomState(0)

    for _ in range(50):
        n = rng.randint(4, 60)
        y_true = rng.randint(0, 2, size=n)
        if y_true.sum() in (0, n):
            continue
        y_score = rng.randint(0, 5, size=n).astype(float)

        # Act
        mine = auc_score(y_true, y_score)
        theirs = roc_auc_score(y_true, y_score)

        # Assert
        assert mine == pytest.approx(theirs)


# ── Impression-level evaluation ──────────────────────────────────────


def test_evaluate_impression_returns_none_when_all_labels_are_positive() -> None:
    # Arrange — AUC and nDCG are both undefined without a negative
    y_true = np.array([1, 1])
    y_score = np.array([2.0, 1.0])

    # Act
    result = evaluate_impression(y_true, y_score)

    # Assert
    assert result is None


def test_evaluate_impression_returns_none_when_no_positive_exists() -> None:
    # Arrange
    y_true = np.array([0, 0, 0])
    y_score = np.array([3.0, 2.0, 1.0])

    # Act
    result = evaluate_impression(y_true, y_score)

    # Assert
    assert result is None


def test_evaluate_impression_reports_all_four_mind_metrics() -> None:
    # Arrange
    y_true = np.array([0, 1, 0, 0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])

    # Act
    result = evaluate_impression(y_true, y_score)

    # Assert
    assert result is not None
    assert result.ndcg_10 == pytest.approx(1.0 / math.log2(3))
    assert result.ndcg_5 == pytest.approx(1.0 / math.log2(3))
    assert result.mrr == pytest.approx(0.5)
    assert result.auc == pytest.approx(2.0 / 3.0)


def test_mean_metrics_averages_each_field_over_impressions() -> None:
    # Arrange
    records = [
        ImpressionMetrics(auc=1.0, mrr=1.0, ndcg_5=1.0, ndcg_10=1.0),
        ImpressionMetrics(auc=0.0, mrr=0.0, ndcg_5=0.5, ndcg_10=0.0),
    ]

    # Act
    result = mean_metrics(records)

    # Assert
    assert result.auc == pytest.approx(0.5)
    assert result.ndcg_5 == pytest.approx(0.75)


def test_mean_metrics_of_an_empty_list_is_all_zeros() -> None:
    # Arrange / Act
    result = mean_metrics([])

    # Assert
    assert result.auc == 0.0
    assert result.ndcg_10 == 0.0
