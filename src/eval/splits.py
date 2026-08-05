"""Deterministic user-level tri-split of MINDsmall_train.

Dev has already been scored, so every further design decision has to be made on
data held out from training. This module carves that holdout out of the train
split.

Three buckets rather than two, because a two-way split leaks:

  fit          model fitting (shared user tower, and any later GBDT)
  early_stop   LightGBM's own iteration selection
  validation   comparing designs across phases

If early stopping and design comparison shared a split, that split would be
driving both model selection inside the learner and selection across
experiments — the same overfitting the train/dev separation exists to prevent,
one level down.

**Split by user, not by impression.** MIND users recur across many impressions;
an impression-level split would put the same user's clicks on both sides, so a
model could memorize that user and validation would flatter it. It would also
misrepresent dev, where 88% of users never appear in train at all. Assigning
whole users keeps validation structurally like dev.

Assignment is a hash of the user id rather than a shuffled index, so it works on
a single streaming pass over behaviors.tsv without needing a row count, and is
stable no matter what order or subset of rows a caller happens to read.
"""

import hashlib
from datetime import date, datetime
from typing import Iterator, TypeVar

FIT = "fit"
EARLY_STOP = "early_stop"
VALIDATION = "validation"

SPLIT_NAMES = (FIT, EARLY_STOP, VALIDATION)

# Upper bound of each bucket on the unit interval. Validation gets 20% of users
# — roughly 30k of MINDsmall_train's 157k impressions, which puts the paired
# interval on a nDCG@10 difference at a few thousandths. A smaller holdout would
# leave the Phase 1 effects below the noise floor.
SPLIT_BOUNDS = ((FIT, 0.70), (EARLY_STOP, 0.80), (VALIDATION, 1.0))

DEFAULT_SEED = 42

# blake2b truncated to 8 bytes; the fraction is that integer over 2**64.
_DIGEST_BYTES = 8
_DIGEST_RANGE = float(1 << (_DIGEST_BYTES * 8))

T = TypeVar("T")


def user_fraction(user_id: str, seed: int = DEFAULT_SEED) -> float:
    """Map a user id onto a stable, uniformly distributed value in [0, 1)."""
    if not user_id:
        raise ValueError("user_id must be a non-empty string")
    digest = hashlib.blake2b(
        f"{seed}:{user_id}".encode("utf-8"), digest_size=_DIGEST_BYTES
    ).digest()
    return int.from_bytes(digest, "big") / _DIGEST_RANGE


def split_for_user(user_id: str, seed: int = DEFAULT_SEED) -> str:
    """Return which of `SPLIT_NAMES` this user belongs to."""
    fraction = user_fraction(user_id, seed)
    for name, upper_bound in SPLIT_BOUNDS:
        if fraction < upper_bound:
            return name
    return VALIDATION


def is_validation_user(user_id: str, seed: int = DEFAULT_SEED) -> bool:
    """True when this user is held out from every form of fitting."""
    return split_for_user(user_id, seed) == VALIDATION


def filter_impressions(
    impressions: Iterator[T],
    split: str,
    seed: int = DEFAULT_SEED,
) -> Iterator[T]:
    """Keep only the impressions whose user belongs to `split`.

    Accepts anything with a `user_id` attribute, which is what
    `mind_data.MindImpression` exposes.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    for impression in impressions:
        if split_for_user(impression.user_id, seed) == split:
            yield impression


# ── Temporal split ───────────────────────────────────────────────────
#
# The primary validation scheme. MINDsmall_train spans Nov 9-14 2019 and
# MINDsmall_dev is Nov 15 alone, so dev's defining property is that every
# quantity used to score it — popularity counts, trained weights — was computed
# on a *disjoint earlier period*.
#
# A user-level holdout does not reproduce that. Its impressions are drawn from
# the same six days as the data behind them, so popularity is contemporaneous
# rather than a day stale, and any recency-flavoured feature scores far better
# on validation than it will on dev. Measured: the popularity baseline reaches
# ~0.386 nDCG@10 on a user-split holdout against 0.3103 on dev — an artifact of
# the split, not a property of the method.
#
# Holding out the last train day instead mirrors dev exactly: fit on the
# preceding days, validate on the next one.

TEMPORAL_FIT = "days_before"
TEMPORAL_VALIDATION = "last_day"

TEMPORAL_SPLIT_NAMES = (TEMPORAL_FIT, TEMPORAL_VALIDATION)

# The final day of MINDsmall_train. Everything before it is fair game for
# fitting; this day is scored and nothing may be derived from it.
VALIDATION_DAY = date(2019, 11, 14)


def temporal_split_for(timestamp: datetime) -> str:
    """Return which side of the temporal holdout an impression falls on.

    Anything on or after `VALIDATION_DAY` is withheld from fitting. That
    includes the dev day itself, so dev rows can never be mistaken for training
    material if a caller points this at the wrong file.
    """
    return TEMPORAL_FIT if timestamp.date() < VALIDATION_DAY else TEMPORAL_VALIDATION


def filter_by_day(impressions: Iterator[T], split: str) -> Iterator[T]:
    """Keep only the impressions on one side of the temporal holdout.

    Accepts anything with a `timestamp` attribute, which is what
    `mind_data.MindImpression` exposes.
    """
    if split not in TEMPORAL_SPLIT_NAMES:
        raise ValueError(
            f"unknown temporal split {split!r}; expected one of {TEMPORAL_SPLIT_NAMES}"
        )
    for impression in impressions:
        if temporal_split_for(impression.timestamp) == split:
            yield impression
