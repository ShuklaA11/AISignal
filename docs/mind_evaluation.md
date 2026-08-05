# Benchmark evaluation on MIND-small

Evaluation of this project's ranking stack on the [MIND](https://msnews.github.io/)
(Microsoft News Recommendation Dataset) benchmark, against real click logs and
published baselines.

Reproduce with:

```bash
python scripts/embed_mind.py     # one-time, ~15 min, local Ollama
python scripts/eval_mind.py --train-impressions 80000 --epochs 3
```

## Why this exists

The prior evaluation (`scripts/offline_eval.py`) scored the ranker against
synthetic users whose preferences it generated itself. That is a smoke test, not
evidence: the ground-truth relevance function and the scorer share assumptions,
so the result is close to circular. MIND replaces synthetic labels with real
click logs and makes the numbers comparable to published work.

## Protocol

Matched to the published MIND baselines so the numbers mean the same thing:

- **Split.** Train on `MINDsmall_train` (156,965 impressions), evaluate on
  `MINDsmall_dev` (73,152 impressions, 50,000 users). MIND-large test labels are
  withheld behind the leaderboard, so dev is the standard reported split.
- **Unit of evaluation.** Each dev impression carries the candidate slate the
  user actually saw, labelled `-1` (clicked) or `-0` (skipped). Rank that slate,
  score it, average across impressions.
- **Metrics.** AUC, MRR, nDCG@5, nDCG@10 — computed by `src/eval/ranking_metrics.py`,
  which reproduces the official MIND scorer's formulas (exponential gains, and
  an MRR averaged over *every* positive rather than the first).
- **User representation.** Built only from the `history` column of the impression
  being scored. Dev labels never reach a user profile.
- **Tie handling.** Candidates are shuffled once per impression with a fixed
  seed and the same permutation is applied to every method, so tied scores can't
  inherit an advantage from file order.

### Verification

- `auc_score` is pinned against `sklearn.metrics.roc_auc_score` across 50 random
  tie-heavy inputs; `ndcg_score` against `sklearn.metrics.ndcg_score` across 600
  comparisons at both cutoffs. Zero mismatches.
- The random baseline scores AUC 0.4985 on 73,152 impressions — an end-to-end
  check that the harness is unbiased.
- The EMA learner is driven by importing the production `_apply_signal` and
  `_update_alpha` directly, so it cannot drift from shipped behaviour.

## Results — MINDsmall_dev, 73,152 impressions

| Method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| random | 0.4985 | 0.2182 | 0.2227 | 0.2860 |
| popularity (train clicks) | 0.5318 | 0.2388 | 0.2462 | 0.3103 |
| rules only | 0.5802 | 0.2758 | 0.2968 | 0.3576 |
| rules + EMA learner | 0.6177 | 0.2977 | 0.3242 | 0.3838 |
| rules + EMA + user tower | 0.6189 | 0.2974 | 0.3246 | 0.3839 |
| **rules + EMA + content similarity** | **0.6431** | **0.3121** | **0.3422** | **0.4022** |

Relative to random, the full system is **+40.6% nDCG@10**; relative to the
rules-only scorer, **+12.5%**.

### What moved the number

A diagnostic on the training split found the scorer was heavily
**quantized**: 38 candidates collapsed to ~11 distinct scores, with 53% of
candidates tied. The cause was structural rather than the final `round(x, 2)` —
removing the rounding recovered only +0.0003. Candidates whose category,
subcategory and entities are all absent from a user's history receive 1.0 on
every factor and are therefore *exactly* tied. What the score lacked was a
continuous signal defined for unseen items.

Semantic similarity is that signal, and two things were wrong with how it was
being used:

- **Mean-pooling the history into a single vector** washes out users with
  several distinct interests. Scoring each candidate by the mean of its top-3
  cosine similarities against individual history items beats mean-pooling
  (0.3831 vs 0.3782 nDCG@10 on the tuning slice) — the same multi-interest
  observation the MINS paper builds on.
- **The factor only reached the learned half of the blend.** Because
  `score_article_for_user_ml` returns `alpha * rule + (1 - alpha) * learned` and
  the embedding factor lives inside `_compute_learned_score`, the one continuous
  feature was down-weighted by `(1 - alpha)`. Applying it to the full blended
  score instead gave the largest single gain: 0.3831 → 0.3907.

Two hypotheses that did *not* pay off are worth recording, since both looked
plausible: removing score rounding (+0.0003) and retuning the rules/learned
blend weight alpha, which was nearly flat across its whole range (0.3782 at the
production schedule vs 0.3804 at the best fixed value).

**Tuning protocol.** Every configuration above was selected on a
4,000-impression slice of `MINDsmall_train` and frozen before dev was scored
once. Tuning on dev and reporting dev would overfit the test set and void the
comparison to published baselines, which is the entire point of running MIND.
The tuning slice predicted 0.3907 and dev returned 0.4022, so the choice
generalized rather than fitting noise.

### Against published baselines

Baselines from Wang et al., *Modeling Multi-interest News Sequence for News
Recommendation* ([arXiv:2207.07331](https://arxiv.org/abs/2207.07331)), which
states its protocol as "we split 10% samples from the train set as the
validation set, and take the released validation set as the test set" — the same
`MINDsmall_dev` used here.

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| BiasMF | 0.5108 | 0.2258 | 0.2318 | 0.2952 |
| DKN | 0.5726 | 0.2339 | 0.2418 | 0.3033 |
| LSTUR | 0.6021 | 0.2659 | 0.2873 | 0.3529 |
| NRMS | 0.6391 | 0.3017 | 0.3282 | 0.3937 |
| HiFi-Ark | 0.6403 | 0.2996 | 0.3272 | 0.3925 |
| **this system** | **0.6431** | **0.3121** | **0.3422** | **0.4022** |
| TANR | 0.6455 | 0.3107 | 0.3367 | 0.4017 |
| MINS | 0.6710 | 0.3171 | 0.3525 | 0.4150 |

The system lands above BiasMF, DKN, LSTUR, NRMS and HiFi-Ark, on par with TANR
(marginally ahead on nDCG@10 at 0.4022 vs 0.4017, marginally behind on AUC at
0.6431 vs 0.6455), and below MINS — from an exponential moving average over
categorical weights plus off-the-shelf embedding similarity, with no trained
neural text encoder.

**Read that with the cross-paper variance in mind.** Reported MIND-small numbers
for the *same* models vary substantially between papers; other reproductions put
LSTUR, NAML and NRMS at AUC 0.66–0.67, which would place this system below all
three. The comparison above is to one internally consistent table whose split
construction is stated explicitly. The defensible claim is "competitive with
published neural baselines on MIND-small," not a ranking against any specific
model.

## Honest limitations

**Only part of the system is under test.** MIND carries category, subcategory,
title/abstract and Wikidata entities. It carries no publication source, no
difficulty grading, no editorial importance score, and no user role or expertise
level. Those four signals sit at neutral defaults here, so what this measures is
the learned personalization layer, not the role/level cold-start heuristics.

**The trained tower contributes essentially nothing** (+0.0012 AUC, +0.0001
nDCG@10), and is comfortably beaten by plain top-3 cosine similarity
(+0.0254 AUC, +0.0184 nDCG@10) over the same embeddings. This is a negative
result and is reported as one. The likely cause is feature starvation rather
than a broken architecture: on MIND, two of the tower's three embedding pools
(saved, skipped) are structurally empty and three of its four engagement
statistics are underivable, so it sees little beyond a mean-pooled history
vector — and mean-pooling is exactly the representation the top-3 comparison
shows to be the weaker choice. A learned user encoder is not obviously earning
its complexity here.

**The cold-start result is a MIND artifact, not a system gap.** On the 2,214 dev
impressions with empty history (3.0%), rules-only and the EMA learner both score
AUC exactly 0.5000, and popularity beats them (nDCG@10 0.3165 vs 0.2969). That
looks like a cold-start failure, but the cause is missing data rather than a
missing fallback: MIND ships no importance score, so `base_importance_score`
falls back to a uniform `DEFAULT_IMPORTANCE` of 5.0 and — with role, level and
source also inert — every candidate receives an identical score.

Production does not behave this way. Scoring the real corpus for a synthetic
zero-history, zero-topic user yields 127 distinct scores spanning 2.25–16.20
(std 2.45, a single article tied at the maximum), because the LLM-assigned
`base_importance_score` (40 distinct values across 6,520 articles) and
`quality_weight` still separate candidates. The lesson is about the benchmark's
coverage, not the ranker: **any factor MIND cannot supply silently drops out of
the measurement**, and a metric can collapse for want of an input rather than
for want of a model.

**The popularity baseline is weak for a structural reason** worth noting: only
7,713 of 65,238 news items receive any train click, so most dev candidates tie
at zero and the metric is depressed by ties rather than by ranking error.

**Two deliberate deviations from production**, both required for a valid measurement:

1. Production fits one `UserTower` per user on that user's own engagement.
   Reproducing that would leak dev labels into the model scoring them, and 88%
   of dev users (44,057 of 50,000) never appear in train at all. A single shared
   tower is trained on the train split and applied to dev by pure inference —
   a stricter protocol than production, and the one neural baselines use.
2. MIND histories contain clicks but not skips, so the EMA learner receives
   positive signal only. Skips still supply the tower's training negatives,
   where labels are legitimately available.

## What this evaluation is not

This is **offline replay without off-policy correction**. MIND ships the slate
and the labels but no display positions and no propensities — the impression
list is deliberately shuffled — so IPS, SNIPS and doubly-robust estimators are
not merely hard here, they are undefined. The result therefore measures
full-slate reranking accuracy *conditional on MSN's candidate selection*, and
still carries selection bias (MSN pre-filtered thousands of candidates down to
~37), position and presentation bias (clicks arose under MSN's ordering), and no
counterfactual claim.

It answers "would this ranker have placed the clicked item highly?" It does not
estimate online lift. That requires propensity-logged interactions, then
interleaving, then A/B.
