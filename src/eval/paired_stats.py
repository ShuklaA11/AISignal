"""Paired significance testing for ranking-metric comparisons.

Two rankers scored on the same impressions produce highly correlated metric
vectors: both do well on easy slates and badly on hard ones. Comparing their
*means* therefore buries a small real effect under the variance of slate
difficulty, which is shared and irrelevant. Comparing the *per-impression
difference* cancels that shared term.

Concretely, on ~30k validation impressions the standard error of a single mean
nDCG@10 is ~0.0017 (per-impression std ~0.3), so an unpaired 95% interval on a
difference spans roughly ±0.005 — wider than most of the effects worth
measuring. The paired difference has a far smaller std because the two rankers
agree on the majority of slates, which is what makes a +0.003 change detectable
at all.

The interval is a percentile bootstrap over the differences rather than a
t-interval: per-impression nDCG is bounded, discrete-ish and skewed, so its
differences are not close to normal, and the bootstrap does not assume they are.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Resamples for the percentile bootstrap. 10k puts the Monte-Carlo error on the
# interval endpoints well below the fourth decimal, which is the resolution the
# nDCG comparisons in this project care about.
DEFAULT_RESAMPLES = 10_000

# Resamples drawn per numpy call. The full index matrix would be
# resamples x n int64s — 2.4GB at 10k x 30k — so it is built in slices.
RESAMPLE_CHUNK = 250


@dataclass(frozen=True)
class PairedDelta:
    """A per-impression difference between two methods, with an interval."""

    mean: float
    ci_low: float
    ci_high: float
    standard_error: float
    n: int
    confidence: float

    @property
    def is_significant(self) -> bool:
        """True when the interval excludes zero.

        This is the project's acceptance rule for keeping a ranking change.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def format(self, label: str = "delta") -> str:
        """One-line summary for evaluation output."""
        marker = "*" if self.is_significant else " "
        return (
            f"{label:<28} {self.mean:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"n={self.n}{marker}"
        )


def paired_delta(
    baseline: Sequence[float],
    variant: Sequence[float],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> PairedDelta:
    """Bootstrap a confidence interval on `variant - baseline`, impression-wise.

    Both sequences must hold one metric value per impression, in the same order,
    scored on the same impressions — otherwise the pairing is meaningless and
    the interval is wrong rather than merely wide.
    """
    baseline_values = np.asarray(baseline, dtype=float)
    variant_values = np.asarray(variant, dtype=float)

    if baseline_values.shape != variant_values.shape:
        raise ValueError(
            "baseline and variant must have the same length "
            f"(got {baseline_values.size} and {variant_values.size})"
        )
    if baseline_values.size < 2:
        raise ValueError("paired_delta needs at least 2 paired observations")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}")

    differences = variant_values - baseline_values
    count = differences.size

    means = _bootstrap_means(differences, resamples, seed)
    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(means, [tail, 1.0 - tail])

    return PairedDelta(
        mean=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        standard_error=float(np.sqrt(differences.var(ddof=1) / count)),
        n=count,
        confidence=confidence,
    )


def _bootstrap_means(
    differences: np.ndarray,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Means of `resamples` resamples-with-replacement, built in memory slices."""
    rng = np.random.RandomState(seed)
    count = differences.size
    means = np.empty(resamples, dtype=float)

    for start in range(0, resamples, RESAMPLE_CHUNK):
        width = min(RESAMPLE_CHUNK, resamples - start)
        indices = rng.randint(0, count, size=(width, count))
        means[start : start + width] = differences[indices].mean(axis=1)
    return means
