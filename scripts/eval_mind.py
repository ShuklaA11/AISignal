#!/usr/bin/env python3
"""Evaluate the recommendation stack on MIND (Microsoft News Recommendation Dataset).

Protocol — matched to the published MIND baselines so the numbers are comparable:

  * Train on MINDsmall_train, evaluate on MINDsmall_dev (the standard reported
    split; MIND-large test labels are withheld behind the leaderboard).
  * Rank each dev impression's own candidate slate, score it, then average the
    four MIND metrics (AUC, MRR, nDCG@5, nDCG@10) across impressions.
  * The user representation for a dev impression is built only from that
    impression's `history` column. Dev labels never touch a user profile.

Two deliberate deviations from production, both documented in the report:

  1. Production fits one UserTower per user from that user's own engagement.
     Doing that here would leak dev labels into the model scoring them, and 88%
     of dev users never appear in train anyway. So a single shared tower is
     trained on the train split and applied to dev by pure inference.
  2. MIND exposes clicks but not skips in a user's history, so the EMA learner
     receives positive signal only. Skips still supply the tower's negatives
     during training, where the labels are legitimately available.

Usage:
    python scripts/eval_mind.py                    # full dev split
    python scripts/eval_mind.py --limit 5000       # quick iteration
    python scripts/eval_mind.py --skip-tower       # categorical methods only
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.user_tower import _PROJ, POOL_DIM, UserTower
from src.eval.mind_data import (
    MindImpression,
    build_user,
    load_behaviors,
    load_news,
    news_to_article,
    train_click_counts,
)
from src.eval.paired_stats import paired_delta
from src.eval.ranking_metrics import (
    ImpressionMetrics,
    evaluate_impression,
    mean_metrics,
)
from src.eval.splits import (
    FIT,
    SPLIT_NAMES,
    TEMPORAL_FIT,
    TEMPORAL_SPLIT_NAMES,
    TEMPORAL_VALIDATION,
    VALIDATION,
    filter_by_day,
    filter_impressions,
)
from src.personalization.learner import _apply_signal, _update_alpha
from src.personalization.scorer import (
    score_article_for_user,
    score_article_for_user_ml,
)
from src.storage.models import User, UserMLProfile

# Mirrors learner.update_on_click, which applies a raw signal of 0.5 per click.
CLICK_SIGNAL = 0.5

# Mirrors user_tower.build_user_features, which reads the 50 most recent
# engagements and mean-pools the first 20 of them.
MAX_HISTORY_FOR_TOWER = 50

# _compute_learned_score expects embedding_factor in [0.5, 1.5]; both vectors are
# unit-norm, so cosine in [-1, 1] maps onto that range at half weight.
EMBEDDING_FACTOR_WEIGHT = 0.5

# Content-similarity settings, selected on a 4,000-impression slice of
# MINDsmall_train and then frozen before dev was scored. Top-3 rather than
# mean-pooling because averaging a whole history into one vector washes out
# users with several distinct interests; the factor multiplies the full blended
# score rather than only the learned half because semantic similarity is the
# one continuous signal defined for every candidate, and burying it inside the
# (1 - alpha) term needlessly dilutes it.
CONTENT_TOP_K = 3
CONTENT_SIM_WEIGHT = 1.0
CONTENT_FACTOR_MIN = 0.3
CONTENT_FACTOR_MAX = 2.0

METHODS = (
    "random",
    "popularity",
    "rules_only",
    "rules+ml_learner",
    "rules+ml+tower",
    "rules+ml+content",
)


def content_similarity_factors(
    history,
    candidates,
    embedding_index,
    embeddings,
) -> np.ndarray | None:
    """Multiplicative semantic factor per candidate, or None without history.

    Scores each candidate by the mean of its top-K cosine similarities against
    the user's recent history items, then maps that onto the same
    [0.3, 2.0] factor range production's compute_embedding_factor uses.
    """
    history_rows = [
        embedding_index[n] for n in reversed(history) if n in embedding_index
    ]
    history_rows = history_rows[:MAX_HISTORY_FOR_TOWER]
    if not history_rows:
        return None

    candidate_rows = [embedding_index.get(n) for n in candidates]
    if any(row is None for row in candidate_rows):
        return None

    # Rows are unit-norm, so the dot product is already cosine similarity.
    similarities = embeddings[candidate_rows] @ embeddings[history_rows].T
    top_k = min(CONTENT_TOP_K, similarities.shape[1])
    pooled = np.sort(similarities, axis=1)[:, -top_k:].mean(axis=1)
    return np.clip(
        1.0 + pooled * CONTENT_SIM_WEIGHT, CONTENT_FACTOR_MIN, CONTENT_FACTOR_MAX
    )


# Every user attribute the scorer reads (role, level, topics, source
# preferences) is derived from the click history alone; `user_id` only fills in
# a display email that no scoring path touches. So both the User and the EMA
# profile are pure functions of the history tuple and can be memoised on it.
# On the validation split 31,610 impressions carry only 9,751 distinct
# histories, so this removes ~69% of the profile-replay work — which is what
# makes a hyperparameter sweep affordable, since the profile does not depend on
# any hyperparameter being swept.
ProfileCache = dict[tuple[str, ...], tuple[User, UserMLProfile]]


@dataclass
class MethodAccumulator:
    """Collects per-impression metrics for one scoring method."""

    name: str
    overall: list[ImpressionMetrics]
    warm: list[ImpressionMetrics]
    cold: list[ImpressionMetrics]


# ── Profile construction ─────────────────────────────────────────────


def build_ml_profile(history, articles) -> UserMLProfile:
    """Replay a click history through the production EMA learner.

    Calls the learner's own `_apply_signal` / `_update_alpha` rather than
    reimplementing them, so this cannot drift from shipped behaviour.
    """
    profile = UserMLProfile(user_id=1)
    for news_id in history:
        article = articles.get(news_id)
        if article is None:
            continue
        # Production applies the signal before incrementing the counter, so the
        # learning-rate tier reflects the count prior to this click.
        _apply_signal(profile, article, signal=CLICK_SIGNAL)
        profile.total_clicks += 1
        _update_alpha(profile)
    return profile


def build_tower_features(history, embedding_index, embeddings) -> np.ndarray:
    """Build the 128-dim user feature vector the UserTower expects.

    Layout matches user_tower.build_user_features:
      [0:32]   saved pool    — always zero, MIND has no save action
      [32:64]  clicked pool  — mean of the 20 most recent history embeddings
      [64:96]  skipped pool  — always zero, MIND histories are clicks only
      [96:128] engagement statistics

    The stat slots that need impression counts (CTR, save rate, skip rate) are
    not derivable from a MIND history and stay zero rather than being invented.
    Keeping this identical across train and dev is what makes the trained tower
    valid at inference time.
    """
    recent = [n for n in reversed(history) if n in embedding_index]
    recent = recent[:MAX_HISTORY_FOR_TOWER]

    features = np.zeros(POOL_DIM * 4, dtype=np.float32)
    if recent:
        vectors = embeddings[[embedding_index[n] for n in recent[:20]]]
        features[POOL_DIM : POOL_DIM * 2] = (np.mean(vectors, axis=0) @ _PROJ).astype(
            np.float32
        )

    stats_base = POOL_DIM * 3
    features[stats_base + 2] = min(len(history) / 100, 1.0)
    if len(recent) > 1:
        projected = embeddings[[embedding_index[n] for n in recent]] @ _PROJ
        features[stats_base + 6] = float(np.mean(np.std(projected, axis=0)))
    return features


# ── Tower training ───────────────────────────────────────────────────


def train_shared_tower(
    train_path: Path,
    articles,
    embedding_index,
    embeddings,
    num_impressions: int,
    epochs: int,
    batch_size: int,
    seed: int,
    train_split: str | None = None,
    train_period: str | None = None,
) -> UserTower | None:
    """Train one shared UserTower on the train split.

    Uses the production loss (cosine embedding loss, margin 0.2) and the same
    3:1 negative sampling ratio, but minibatched across users instead of
    full-batch per user.

    `train_split` restricts training to one user bucket of `src.eval.splits`,
    and `train_period` to one side of the temporal holdout. None for both trains
    on every impression in the file, which is what the published dev numbers
    were produced with.
    """
    rng = np.random.RandomState(seed)
    feature_rows: list[np.ndarray] = []
    pair_user: list[int] = []
    pair_article: list[int] = []
    pair_label: list[int] = []

    train_impressions = load_behaviors(train_path, limit=num_impressions)
    if train_period is not None:
        train_impressions = filter_by_day(train_impressions, train_period)
    if train_split is not None:
        train_impressions = filter_impressions(train_impressions, train_split)

    for impression in train_impressions:
        if not impression.history:
            continue
        positives = [
            n
            for n, label in zip(impression.candidates, impression.labels)
            if label == 1 and n in embedding_index
        ]
        if not positives:
            continue
        negatives = [
            n
            for n, label in zip(impression.candidates, impression.labels)
            if label == 0 and n in embedding_index
        ]
        if not negatives:
            continue

        features = build_tower_features(impression.history, embedding_index, embeddings)
        user_row = len(feature_rows)
        feature_rows.append(features)

        for news_id in positives:
            pair_user.append(user_row)
            pair_article.append(embedding_index[news_id])
            pair_label.append(1)
            for negative_id in rng.choice(
                negatives, size=min(3, len(negatives)), replace=False
            ):
                pair_user.append(user_row)
                pair_article.append(embedding_index[negative_id])
                pair_label.append(-1)

    if len(feature_rows) < 100:
        print(f"  Not enough training users ({len(feature_rows)}); skipping tower")
        return None

    features_all = torch.tensor(np.stack(feature_rows), dtype=torch.float32)
    embeddings_all = torch.tensor(embeddings, dtype=torch.float32)
    user_idx = torch.tensor(pair_user, dtype=torch.long)
    article_idx = torch.tensor(pair_article, dtype=torch.long)
    labels = torch.tensor(pair_label, dtype=torch.float32)

    print(
        f"  Tower training set: {len(feature_rows)} users, {len(pair_label)} pairs "
        f"({int((labels == 1).sum())} pos / {int((labels == -1).sum())} neg)"
    )

    torch.manual_seed(seed)
    model = UserTower()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.2)

    model.train()
    num_pairs = len(pair_label)
    for epoch in range(epochs):
        order = torch.randperm(num_pairs)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, num_pairs, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            user_emb = model(features_all[user_idx[batch]])
            loss = loss_fn(user_emb, embeddings_all[article_idx[batch]], labels[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        print(f"    epoch {epoch + 1}/{epochs}  loss={total_loss / num_batches:.4f}")

    model.eval()
    return model


# ── Evaluation ───────────────────────────────────────────────────────


def score_impression(
    impression: MindImpression,
    articles,
    user,
    profile: UserMLProfile,
    popularity,
    user_embedding,
    embedding_index,
    embeddings,
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """Produce one score vector per method for this impression's candidates."""
    candidate_articles = [articles[n] for n in impression.candidates]

    scores = {
        "random": rng.random_sample(len(candidate_articles)),
        "popularity": np.array(
            [float(popularity.get(n, 0)) for n in impression.candidates]
        ),
        "rules_only": np.array(
            [score_article_for_user(a, user) for a in candidate_articles]
        ),
        "rules+ml_learner": np.array(
            [score_article_for_user_ml(a, user, profile) for a in candidate_articles]
        ),
    }

    # A missing user embedding yields a neutral 1.0 factor, mirroring
    # production's compute_embedding_factor. This keeps every method scored on
    # exactly the same impressions, so the comparison stays like-for-like
    # instead of quietly excluding cold-start users from this method alone.
    content_factors = content_similarity_factors(
        impression.history, impression.candidates, embedding_index, embeddings
    )
    if content_factors is None:
        content_factors = np.ones(len(candidate_articles))
    scores["rules+ml+content"] = scores["rules+ml_learner"] * content_factors

    if user_embedding is not None:
        factors = []
        for news_id in impression.candidates:
            row = embedding_index.get(news_id)
            if row is None:
                factors.append(1.0)
            else:
                similarity = float(np.dot(user_embedding, embeddings[row]))
                factors.append(1.0 + similarity * EMBEDDING_FACTOR_WEIGHT)
        scores["rules+ml+tower"] = np.array(
            [
                score_article_for_user_ml(a, user, profile, embedding_factor=f)
                for a, f in zip(candidate_articles, factors)
            ]
        )
    return scores


def evaluate(
    dev_path: Path,
    articles,
    news,
    popularity,
    tower: UserTower | None,
    embedding_index,
    embeddings,
    limit: int | None,
    chunk_size: int,
    seed: int,
    eval_split: str | None = None,
    eval_period: str | None = None,
) -> tuple[dict[str, MethodAccumulator], int]:
    """Run the evaluation over a behaviors file, chunked so tower inference batches.

    `eval_split` restricts scoring to one user bucket of `src.eval.splits` and
    `eval_period` to one side of the temporal holdout. Applying both yields the
    strict validation set — held-out users on the held-out day.
    """
    accumulators = {name: MethodAccumulator(name, [], [], []) for name in METHODS}
    rng = np.random.RandomState(seed)
    profile_cache: ProfileCache = {}
    skipped = 0
    processed = 0
    started = time.time()

    impressions = load_behaviors(dev_path, limit=limit)
    if eval_period is not None:
        impressions = filter_by_day(impressions, eval_period)
    if eval_split is not None:
        impressions = filter_impressions(impressions, eval_split)

    chunk: list[MindImpression] = []
    for impression in impressions:
        chunk.append(impression)
        if len(chunk) < chunk_size:
            continue
        skipped += _process_chunk(
            chunk,
            accumulators,
            articles,
            news,
            popularity,
            tower,
            embedding_index,
            embeddings,
            rng,
            profile_cache,
        )
        processed += len(chunk)
        chunk = []
        rate = processed / (time.time() - started)
        print(f"  {processed} impressions ({rate:.0f}/s)", flush=True)

    if chunk:
        skipped += _process_chunk(
            chunk,
            accumulators,
            articles,
            news,
            popularity,
            tower,
            embedding_index,
            embeddings,
            rng,
            profile_cache,
        )
        processed += len(chunk)

    print(f"  {processed} impressions scored in {(time.time() - started) / 60:.1f} min")
    return accumulators, skipped


def _process_chunk(
    chunk,
    accumulators,
    articles,
    news,
    popularity,
    tower,
    embedding_index,
    embeddings,
    rng,
    profile_cache: ProfileCache,
) -> int:
    """Score one chunk of impressions; returns how many were unusable."""
    user_embeddings: list[np.ndarray | None] = [None] * len(chunk)
    if tower is not None:
        features = np.stack(
            [
                build_tower_features(imp.history, embedding_index, embeddings)
                for imp in chunk
            ]
        )
        with torch.no_grad():
            computed = tower(torch.tensor(features, dtype=torch.float32)).numpy()
        user_embeddings = list(computed)

    skipped = 0
    for impression, user_embedding in zip(chunk, user_embeddings):
        cached = profile_cache.get(impression.history)
        if cached is None:
            cached = (
                build_user(impression.user_id, impression.history, news),
                build_ml_profile(impression.history, articles),
            )
            profile_cache[impression.history] = cached
        user, profile = cached

        scores = score_impression(
            impression,
            articles,
            user,
            profile,
            popularity,
            user_embedding,
            embedding_index,
            embeddings,
            rng,
        )

        labels = np.array(impression.labels)
        # Shuffle once per impression and apply the same permutation to every
        # method, so tied scores cannot inherit an advantage from the order the
        # candidates happen to appear in the file.
        permutation = rng.permutation(len(labels))
        shuffled_labels = labels[permutation]

        usable = None
        for name, score_vector in scores.items():
            metrics = evaluate_impression(shuffled_labels, score_vector[permutation])
            if metrics is None:
                usable = False
                break
            usable = True
            accumulators[name].overall.append(metrics)
            if impression.history:
                accumulators[name].warm.append(metrics)
            else:
                accumulators[name].cold.append(metrics)
        if usable is False:
            skipped += 1
    return skipped


# ── Reporting ────────────────────────────────────────────────────────


def print_table(title: str, rows: list[tuple[str, ImpressionMetrics, int]]) -> None:
    print(f"\n{title}")
    print(
        f"  {'Method':<20} {'AUC':>7} {'MRR':>7} {'nDCG@5':>8} {'nDCG@10':>8}  {'n':>7}"
    )
    print(f"  {'-' * 20} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8}  {'-' * 7}")
    for name, metrics, count in rows:
        print(
            f"  {name:<20} {metrics.auc:7.4f} {metrics.mrr:7.4f} "
            f"{metrics.ndcg_5:8.4f} {metrics.ndcg_10:8.4f}  {count:7d}"
        )


def print_paired_table(
    accumulators: dict[str, MethodAccumulator],
    active: list[str],
    baseline: str,
    confidence: float,
) -> None:
    """Report each method's nDCG@10 against `baseline`, impression by impression.

    Comparing mean nDCG@10 across methods buries small effects under the
    variance of slate difficulty, which both methods share and neither causes.
    Every accumulator here holds one entry per impression in the same order —
    an impression that cannot be scored is dropped for all methods at once — so
    the lists are aligned and the difference can be taken pairwise, cancelling
    that shared term.

    A change is kept only when its interval excludes zero.
    """
    reference = [m.ndcg_10 for m in accumulators[baseline].overall]
    print(f"\nPAIRED ΔnDCG@10 vs {baseline} ({confidence:.0%} CI, * = significant)")
    for name in active:
        if name == baseline:
            continue
        variant = [m.ndcg_10 for m in accumulators[name].overall]
        delta = paired_delta(reference, variant, confidence=confidence)
        print(f"  {delta.format(name)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on MIND-small")
    parser.add_argument("--root", type=Path, default=Path("data/mind"))
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/mind/embeddings.npz")
    )
    parser.add_argument("--limit", type=int, default=None, help="cap dev impressions")
    parser.add_argument("--train-impressions", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tower", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="write results JSON")
    parser.add_argument(
        "--eval-on",
        choices=("dev", "train"),
        default="dev",
        help="which behaviors file to score (default: dev)",
    )
    parser.add_argument(
        "--eval-split",
        choices=SPLIT_NAMES,
        default=None,
        help="restrict scoring to one user bucket; omit to score every user",
    )
    parser.add_argument(
        "--paired-baseline",
        default="rules+ml+content",
        choices=METHODS,
        help="method the paired ΔnDCG@10 table is measured against",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="confidence level for the paired interval",
    )
    parser.add_argument(
        "--eval-period",
        choices=TEMPORAL_SPLIT_NAMES,
        default=None,
        help=(
            "restrict scoring to one side of the temporal holdout. "
            f"{TEMPORAL_VALIDATION!r} is the primary validation set: the last "
            "train day, mirroring dev's day-after structure"
        ),
    )
    args = parser.parse_args()

    # Anything the evaluation is scored with has to be derived from data the
    # evaluation cannot see. Both restrictions are inferred from the evaluation
    # target rather than left to separate flags, because forgetting either one
    # silently inflates the result instead of failing.
    train_split = FIT if args.eval_split == VALIDATION else None
    train_period = TEMPORAL_FIT if args.eval_period == TEMPORAL_VALIDATION else None
    if train_split is not None:
        print(f"Holding out {VALIDATION!r} users; tower restricted to {FIT!r}.")
    if train_period is not None:
        print(
            f"Holding out the last train day; popularity and tower restricted "
            f"to {TEMPORAL_FIT!r}."
        )

    eval_path = args.root / f"{args.eval_on}/behaviors.tsv"

    print("Loading MIND news...")
    news = {
        **load_news(args.root / "train/news.tsv"),
        **load_news(args.root / "dev/news.tsv"),
    }
    articles = {
        news_id: news_to_article(item, index)
        for index, (news_id, item) in enumerate(sorted(news.items()), start=1)
    }
    print(f"  {len(news)} unique news items")

    # The held-out users live inside MINDsmall_train, so when the evaluation
    # runs on one of its buckets the popularity table has to drop that bucket's
    # clicks. Counting them inflates popularity from 0.3103 nDCG@10 to 0.3874
    # purely by having seen the labels it is scored against.
    print("Counting train-split clicks for the popularity baseline...")
    popularity_exclusion = args.eval_split if args.eval_on == "train" else None
    popularity_period = train_period if args.eval_on == "train" else None
    if popularity_exclusion is not None:
        print(f"  excluding {popularity_exclusion!r} users to avoid label leakage")
    if popularity_period is not None:
        print(f"  counting only {popularity_period!r} clicks, as dev's table is")
    popularity = train_click_counts(
        args.root / "train/behaviors.tsv",
        exclude_split=popularity_exclusion,
        restrict_to_period=popularity_period,
    )
    print(f"  {len(popularity)} items received at least one train click")

    # Embeddings feed two independent methods: the content-similarity factor and
    # the trained tower. They are loaded whenever available so that --skip-tower
    # means "don't spend minutes training the tower" rather than "also disable
    # content similarity", which are very different experiments.
    embedding_index: dict[str, int] = {}
    embeddings = np.zeros((0, 0), dtype=np.float32)
    tower = None
    if not args.embeddings.exists():
        if not args.skip_tower:
            print(
                f"ERROR: {args.embeddings} not found. Run scripts/embed_mind.py first."
            )
            sys.exit(1)
        print(f"No embedding cache at {args.embeddings}; content method disabled.")
    else:
        print("Loading embedding cache...")
        with np.load(args.embeddings, allow_pickle=False) as data:
            news_ids = data["news_ids"]
            embeddings = data["embeddings"].astype(np.float32)
        embedding_index = {str(n): i for i, n in enumerate(news_ids)}
        coverage = sum(1 for n in news if n in embedding_index) / len(news)
        print(f"  {len(embedding_index)} vectors, {coverage:.1%} news coverage")

    if not args.skip_tower:
        print("Training shared user tower on the train split...")
        tower = train_shared_tower(
            args.root / "train/behaviors.tsv",
            articles,
            embedding_index,
            embeddings,
            args.train_impressions,
            args.epochs,
            args.batch_size,
            args.seed,
            train_split,
            train_period,
        )

    label = args.eval_split or "all"
    print(f"Evaluating on the {args.eval_on} split ({label} users)...")
    accumulators, skipped = evaluate(
        eval_path,
        articles,
        news,
        popularity,
        tower,
        embedding_index,
        embeddings,
        args.limit,
        args.chunk_size,
        args.seed,
        args.eval_split,
        args.eval_period,
    )

    active = [name for name in METHODS if accumulators[name].overall]
    print_table(
        f"OVERALL ({args.eval_on} split, {label} users)",
        [
            (n, mean_metrics(accumulators[n].overall), len(accumulators[n].overall))
            for n in active
        ],
    )
    if args.paired_baseline in active and len(active) > 1:
        print_paired_table(accumulators, active, args.paired_baseline, args.confidence)

    if any(accumulators[n].cold for n in active):
        print_table(
            "COLD START (empty click history)",
            [
                (n, mean_metrics(accumulators[n].cold), len(accumulators[n].cold))
                for n in active
            ],
        )
        print_table(
            "WARM (non-empty click history)",
            [
                (n, mean_metrics(accumulators[n].warm), len(accumulators[n].warm))
                for n in active
            ],
        )
    if skipped:
        print(f"\n{skipped} impressions excluded (no positive or no negative label)")

    if args.output:
        payload = {
            name: {
                "overall": vars(mean_metrics(accumulators[name].overall)),
                "cold": vars(mean_metrics(accumulators[name].cold)),
                "warm": vars(mean_metrics(accumulators[name].warm)),
                "n_overall": len(accumulators[name].overall),
                "n_cold": len(accumulators[name].cold),
            }
            for name in active
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
