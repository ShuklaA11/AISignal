#!/usr/bin/env python3
"""Offline evaluation of the recommendation system on synthetic users.

Generates synthetic user profiles with known preferences, simulates
realistic engagement (clicks/saves/skips with noise), trains the ML
learner and user tower, then measures whether the learned models
recover the ground-truth preferences better than the rules-only baseline.

Metrics reported:
  - nDCG@10: ranking quality (did relevant articles land near the top?)
  - Precision@10: fraction of top-10 that are truly relevant
  - MRR: mean reciprocal rank of first relevant item
  - CTR@10: simulated click-through rate in top-10

Usage:
    python scripts/offline_eval.py
    python scripts/offline_eval.py --articles 800 --extra-users 10 --seed 7
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.provider import EMBEDDING_DIM
from src.personalization.scorer import (
    DEFAULT_IMPORTANCE,
    LEVEL_DIFFICULTY_WEIGHTS,
    MAX_SCORE,
    MAX_TOPIC_FACTOR,
    ROLE_CATEGORY_WEIGHTS,
    SOURCE_WEIGHT_BASELINE,
    TOPIC_MATCH_BOOST,
    score_article_for_user,
    score_article_for_user_ml,
)
from src.storage.models import Article, User, UserMLProfile

# ── Constants ────────────────────────────────────────────────────────

ALL_TOPICS = [
    "NLP", "Computer Vision", "Reinforcement Learning", "ML Theory",
    "AI Safety", "Multimodal", "Robotics", "AI Agents",
    "LLM APIs", "AI Infrastructure", "AI Startups", "Enterprise AI",
    "AI Regulation", "Fundraising",
    "Open Source Models", "AI Art", "AI Coding Tools", "AI Hardware",
    "Tutorials", "General AI",
]
CATEGORIES = ["research", "product", "industry", "open_source", "opinion"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]
SOURCES = ["techcrunch_ai", "arxiv", "huggingface", "reddit", "github", "anthropic_blog"]
ROLES = ["student", "industry", "enthusiast"]
LEVELS = ["beginner", "intermediate", "advanced"]
ENTITIES = [
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Hugging Face",
    "NVIDIA", "Microsoft", "Mistral", "Stability AI", "Cohere",
    "GPT-4", "Claude", "Llama", "Gemini", "DALL-E", "Stable Diffusion",
]


# ── Synthetic user archetypes ────────────────────────────────────────

@dataclass
class UserArchetype:
    name: str
    role: str
    level: str
    preferred_topics: list[str]
    preferred_categories: list[str]
    preferred_sources: list[str]
    preferred_entities: list[str]
    source_weights: dict[str, int]
    noise_level: float = 0.15

ARCHETYPES = [
    UserArchetype(
        name="nlp_researcher", role="student", level="advanced",
        preferred_topics=["NLP", "ML Theory", "AI Safety"],
        preferred_categories=["research"],
        preferred_sources=["arxiv"],
        preferred_entities=["Anthropic", "Google DeepMind", "Claude"],
        source_weights={"arxiv": 10, "huggingface": 8, "rss": 4, "reddit": 5, "github": 6},
    ),
    UserArchetype(
        name="ml_engineer", role="industry", level="intermediate",
        preferred_topics=["LLM APIs", "AI Infrastructure", "Open Source Models", "AI Coding Tools"],
        preferred_categories=["product", "open_source"],
        preferred_sources=["huggingface", "github"],
        preferred_entities=["Hugging Face", "NVIDIA", "Llama", "Mistral"],
        source_weights={"github": 9, "huggingface": 9, "rss": 7, "arxiv": 4, "reddit": 6},
    ),
    UserArchetype(
        name="ai_enthusiast", role="enthusiast", level="beginner",
        preferred_topics=["AI Art", "General AI", "Tutorials", "AI Agents"],
        preferred_categories=["opinion", "product"],
        preferred_sources=["reddit", "techcrunch_ai"],
        preferred_entities=["OpenAI", "GPT-4", "DALL-E", "Stable Diffusion"],
        source_weights={"reddit": 9, "rss": 8, "huggingface": 5, "arxiv": 2, "github": 4},
    ),
    UserArchetype(
        name="startup_founder", role="industry", level="advanced",
        preferred_topics=["AI Startups", "Fundraising", "Enterprise AI", "AI Regulation"],
        preferred_categories=["industry", "product"],
        preferred_sources=["techcrunch_ai"],
        preferred_entities=["OpenAI", "Anthropic", "Microsoft", "Cohere"],
        source_weights={"rss": 10, "reddit": 5, "arxiv": 3, "github": 4, "huggingface": 4},
    ),
    UserArchetype(
        name="cv_student", role="student", level="intermediate",
        preferred_topics=["Computer Vision", "Multimodal", "Robotics", "ML Theory"],
        preferred_categories=["research", "open_source"],
        preferred_sources=["arxiv", "github"],
        preferred_entities=["Google DeepMind", "Meta AI", "NVIDIA", "Gemini"],
        source_weights={"arxiv": 10, "github": 8, "huggingface": 7, "rss": 4, "reddit": 6},
    ),
    UserArchetype(
        name="safety_researcher", role="student", level="advanced",
        preferred_topics=["AI Safety", "AI Regulation", "AI Agents"],
        preferred_categories=["research", "opinion"],
        preferred_sources=["arxiv", "anthropic_blog"],
        preferred_entities=["Anthropic", "Claude", "Google DeepMind", "OpenAI"],
        source_weights={"arxiv": 10, "rss": 7, "reddit": 5, "huggingface": 4, "github": 3},
    ),
    UserArchetype(
        name="hobbyist_tinkerer", role="enthusiast", level="intermediate",
        preferred_topics=["Open Source Models", "AI Coding Tools", "Tutorials", "AI Hardware"],
        preferred_categories=["open_source", "product"],
        preferred_sources=["reddit", "huggingface", "github"],
        preferred_entities=["Hugging Face", "Llama", "Mistral", "NVIDIA"],
        source_weights={"reddit": 8, "github": 8, "huggingface": 9, "rss": 5, "arxiv": 4},
    ),
]


# ── Embedding generation ─────────────────────────────────────────────

# Precompute stable per-topic and per-category direction vectors
_topic_rng = np.random.RandomState(seed=12345)
_TOPIC_VECS = {}
for _t in ALL_TOPICS:
    _v = _topic_rng.randn(EMBEDDING_DIM).astype(np.float32)
    _v /= np.linalg.norm(_v)
    _TOPIC_VECS[_t] = _v

_cat_rng = np.random.RandomState(seed=67890)
_CAT_VECS = {}
for _c in CATEGORIES:
    _v = _cat_rng.randn(EMBEDDING_DIM).astype(np.float32)
    _v /= np.linalg.norm(_v)
    _CAT_VECS[_c] = _v


def _make_embedding(topics: list[str], category: str, rng: np.random.RandomState) -> np.ndarray:
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for t in topics:
        if t in _TOPIC_VECS:
            vec += _TOPIC_VECS[t]
    if category in _CAT_VECS:
        vec += 0.5 * _CAT_VECS[category]
    vec += rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.3
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Article generation ───────────────────────────────────────────────

def generate_articles(n: int, rng: np.random.RandomState):
    articles = []
    embeddings = {}
    for i in range(n):
        n_topics = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        topics = list(rng.choice(ALL_TOPICS, size=n_topics, replace=False))
        category = rng.choice(CATEGORIES)
        difficulty = rng.choice(DIFFICULTIES, p=[0.25, 0.5, 0.25])
        source = rng.choice(SOURCES)
        importance = round(rng.uniform(3.0, 9.0), 1)
        n_ent = rng.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        ents = list(rng.choice(ENTITIES, size=n_ent, replace=False)) if n_ent else []

        a = Article(
            id=i + 1, url=f"https://example.com/a-{i}", content_hash=f"h{i}",
            title=f"Article {i}: {', '.join(topics)}", source_name=source,
            source_type="rss", category=category,
            base_importance_score=importance, topics_json=json.dumps(topics),
            difficulty_level=difficulty, key_entities_json=json.dumps(ents),
        )
        articles.append(a)
        embeddings[a.id] = _make_embedding(topics, category, rng)

    return articles, embeddings


# ── Ground-truth relevance ───────────────────────────────────────────

def ground_truth_relevance(article: Article, arch: UserArchetype) -> float:
    """How much this user would genuinely like this article. Returns [0, 1]."""
    score = 0.0

    # Topic match — strongest signal
    a_topics = set(article.topics)
    p_topics = set(arch.preferred_topics)
    if a_topics:
        score += 0.4 * (len(a_topics & p_topics) / len(a_topics))

    # Category
    if article.category in arch.preferred_categories:
        score += 0.2

    # Source
    if article.source_name in arch.preferred_sources:
        score += 0.15

    # Entities
    a_ents = set(article.key_entities)
    if a_ents:
        score += 0.1 * (len(a_ents & set(arch.preferred_entities)) / len(a_ents))

    # Difficulty alignment
    level_w = LEVEL_DIFFICULTY_WEIGHTS.get(arch.level, {})
    diff = article.difficulty_level or "intermediate"
    score += 0.15 * min(level_w.get(diff, 1.0) / 1.4, 1.0)

    return min(score, 1.0)


# ── Engagement simulation ────────────────────────────────────────────

@dataclass
class EngagementRecord:
    article_id: int
    clicked: bool
    saved: bool
    relevance: float


def simulate_engagement(articles, arch, rng, n_impressions=200):
    shown = rng.choice(articles, size=min(n_impressions, len(articles)), replace=False)
    records = []
    for article in shown:
        rel = ground_truth_relevance(article, arch)
        noise = rng.uniform(-arch.noise_level, arch.noise_level)
        click_prob = max(0, min(1, rel + noise))
        clicked = rng.random() < click_prob
        saved = clicked and rel > 0.4 and rng.random() < (rel * 0.5)
        records.append(EngagementRecord(article.id, clicked, saved, rel))
    return records


# ── EMA learner (mirrors src/personalization/learner.py) ─────────────

def _ema(old, signal, lr):
    return round(max(0.1, min(3.0, (1 - lr) * old + lr * (1.0 + signal))), 4)


def train_ml_profile(articles_by_id, engagement):
    profile = UserMLProfile(user_id=0)
    for rec in engagement:
        art = articles_by_id.get(rec.article_id)
        if not art:
            continue
        if rec.saved:
            signal, profile.total_saves = 1.0, profile.total_saves + 1
        elif rec.clicked:
            signal, profile.total_clicks = 0.5, profile.total_clicks + 1
        else:
            signal = -0.1

        total = profile.total_clicks + profile.total_saves
        lr = 0.3 if total < 10 else (0.15 if total < 50 else 0.05)

        sw = profile.source_weights
        sw[art.source_name] = _ema(sw.get(art.source_name, 1.0), signal, lr)
        profile.source_weights = sw

        if art.category:
            cw = profile.category_weights
            cw[art.category] = _ema(cw.get(art.category, 1.0), signal, lr)
            profile.category_weights = cw

        for topic in art.topics:
            tw = profile.topic_weights
            tw[topic] = _ema(tw.get(topic, 1.0), signal, lr)
            profile.topic_weights = tw

        if art.difficulty_level:
            dw = profile.difficulty_weights
            dw[art.difficulty_level] = _ema(dw.get(art.difficulty_level, 1.0), signal, lr)
            profile.difficulty_weights = dw

        for ent in art.key_entities[:5]:
            ew = profile.entity_weights
            ew[ent] = _ema(ew.get(ent, 1.0), signal, lr)
            profile.entity_weights = ew

    total = profile.total_clicks + profile.total_saves
    profile.alpha = round(max(0.3, 1.0 - (total / 100)), 4)
    return profile


# ── User tower training ──────────────────────────────────────────────

def train_tower(engagement, embeddings):
    try:
        import torch
        from src.embeddings.user_tower import UserTower, POOL_DIM
    except ImportError:
        return None, 0.0

    positive_ids = {r.article_id for r in engagement if r.clicked or r.saved}
    negative_ids = {r.article_id for r in engagement if not r.clicked and not r.saved}
    pos_with = [a for a in positive_ids if a in embeddings]
    neg_with = [a for a in negative_ids if a in embeddings]

    if len(pos_with) < 5:
        return None, 0.0

    art_embs, labels = [], []
    _rng = random.Random(42)
    for aid in pos_with:
        art_embs.append(embeddings[aid])
        labels.append(1)
        for nid in _rng.sample(neg_with, min(3, len(neg_with))):
            art_embs.append(embeddings[nid])
            labels.append(-1)

    # Projection matrix (same seed as user_tower.py)
    proj_rng = np.random.RandomState(seed=EMBEDDING_DIM * 1000 + POOL_DIM)
    proj = proj_rng.randn(EMBEDDING_DIM, POOL_DIM).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    def pool(ids):
        vecs = [embeddings[a] for a in ids if a in embeddings]
        if not vecs:
            return np.zeros(POOL_DIM, dtype=np.float32)
        return (np.mean(vecs[:20], axis=0) @ proj).astype(np.float32)

    saved_ids = {r.article_id for r in engagement if r.saved}
    clicked_ids = positive_ids - saved_ids
    stats = np.zeros(POOL_DIM, dtype=np.float32)
    n = len(engagement)
    stats[0] = len(clicked_ids) / max(n, 1)
    stats[1] = len(saved_ids) / max(n, 1)
    stats[2] = min(n / 100, 1.0)
    stats[3] = len(negative_ids) / max(n, 1)

    features = np.concatenate([pool(saved_ids), pool(clicked_ids), pool(negative_ids), stats])

    device = torch.device("cpu")
    feat_t = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    art_t = torch.tensor(np.stack(art_embs), dtype=torch.float32, device=device)
    lab_t = torch.tensor(labels, dtype=torch.float32, device=device)

    model = UserTower().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.2)

    model.train()
    final_loss = 0.0
    for _ in range(50):
        opt.zero_grad()
        u = model(feat_t).expand(len(labels), -1)
        loss = loss_fn(u, art_t, lab_t)
        loss.backward()
        opt.step()
        final_loss = loss.item()

    model.eval()
    with torch.no_grad():
        emb = model(feat_t).squeeze(0).numpy()
    return emb, final_loss


# ── Ranking metrics ──────────────────────────────────────────────────

def ndcg_at_k(rels, k=10):
    top = rels[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(relevant, k=10):
    return sum(relevant[:k]) / k if k > 0 else 0.0

def mrr(relevant):
    for i, r in enumerate(relevant):
        if r:
            return 1.0 / (i + 1)
    return 0.0

def ctr_top_k(engagement, top_ids):
    shown = [r for r in engagement if r.article_id in top_ids]
    return sum(1 for r in shown if r.clicked) / len(shown) if shown else 0.0


# ── Per-user evaluation ──────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.35

@dataclass
class EvalResult:
    archetype: str
    method: str
    ndcg_10: float
    precision_10: float
    mrr_val: float
    ctr_10: float


def evaluate_user(arch, articles, embeddings, rng):
    articles_by_id = {a.id: a for a in articles}
    all_eng = simulate_engagement(articles, arch, rng, n_impressions=len(articles))
    split = int(len(all_eng) * 0.6)
    train_eng = all_eng[:split]
    eval_eng = all_eng[split:]

    eval_ids = {r.article_id for r in eval_eng}
    eval_arts = [articles_by_id[a] for a in eval_ids if a in articles_by_id]
    if len(eval_arts) < 10:
        return []

    rel_map = {r.article_id: r.relevance for r in eval_eng}

    user = User(
        id=1, email=f"{arch.name}@test.com", role=arch.role, level=arch.level,
        topics_json=json.dumps(arch.preferred_topics),
        source_preferences_json=json.dumps(arch.source_weights),
    )
    ml_profile = train_ml_profile(articles_by_id, train_eng)
    learned_emb, tower_loss = train_tower(train_eng, embeddings)

    results = []

    def _eval(method_name, scored_articles):
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        rels = [rel_map.get(a.id, 0) for a, _ in scored_articles]
        relevant = [rel_map.get(a.id, 0) >= RELEVANCE_THRESHOLD for a, _ in scored_articles]
        top10 = {a.id for a, _ in scored_articles[:10]}
        results.append(EvalResult(
            arch.name, method_name,
            ndcg_at_k(rels, 10), precision_at_k(relevant, 10),
            mrr(relevant), ctr_top_k(eval_eng, top10),
        ))

    # Random
    rand_arts = list(eval_arts)
    rng.shuffle(rand_arts)
    _eval("random", [(a, rng.random()) for a in rand_arts])

    # Rules only
    _eval("rules_only", [(a, score_article_for_user(a, user)) for a in eval_arts])

    # Rules + ML learner
    _eval("rules+ml_learner", [(a, score_article_for_user_ml(a, user, ml_profile)) for a in eval_arts])

    # Rules + ML + tower embeddings
    if learned_emb is not None:
        def emb_factor(aid):
            if aid not in embeddings:
                return 1.0
            return 1.0 + float(np.dot(learned_emb, embeddings[aid])) * 0.5

        _eval("rules+ml+tower", [
            (a, score_article_for_user_ml(a, user, ml_profile, embedding_factor=emb_factor(a.id)))
            for a in eval_arts
        ])

    return results


# ── Main ─────────────────────────────────────────────────────────────

def run(n_articles=500, n_extra=0, seed=42):
    rng = np.random.RandomState(seed)

    print(f"Generating {n_articles} synthetic articles...")
    articles, embeddings = generate_articles(n_articles, rng)
    topic_dist = {}
    for a in articles:
        for t in a.topics:
            topic_dist[t] = topic_dist.get(t, 0) + 1
    print(f"  {len(articles)} articles across {len(topic_dist)} topics, "
          f"{len(set(a.category for a in articles))} categories, "
          f"{len(set(a.source_name for a in articles))} sources")

    users = list(ARCHETYPES)
    for i in range(n_extra):
        nt = rng.choice([2, 3, 4, 5])
        topics = list(rng.choice(ALL_TOPICS, size=nt, replace=False))
        nc = rng.choice([1, 2])
        cats = list(rng.choice(CATEGORIES, size=nc, replace=False))
        ns = rng.choice([1, 2, 3])
        srcs = list(rng.choice(SOURCES, size=ns, replace=False))
        ne = rng.choice([2, 3, 4])
        ents = list(rng.choice(ENTITIES, size=ne, replace=False))
        sw = {s: int(rng.choice([3, 5, 7, 9])) for s in ["arxiv", "huggingface", "rss", "reddit", "github"]}
        users.append(UserArchetype(
            name=f"random_{i}", role=rng.choice(ROLES), level=rng.choice(LEVELS),
            preferred_topics=topics, preferred_categories=cats,
            preferred_sources=srcs, preferred_entities=ents,
            source_weights=sw, noise_level=rng.uniform(0.1, 0.25),
        ))

    all_results = []
    for arch in users:
        print(f"\n{'='*60}")
        print(f"  {arch.name} ({arch.role}/{arch.level})")
        print(f"  topics: {arch.preferred_topics}")
        print(f"  categories: {arch.preferred_categories}")
        u_rng = np.random.RandomState(seed + hash(arch.name) % 10000)
        results = evaluate_user(arch, articles, embeddings, u_rng)
        all_results.extend(results)
        for r in results:
            print(f"    {r.method:22s}  nDCG@10={r.ndcg_10:.3f}  P@10={r.precision_10:.2f}  MRR={r.mrr_val:.3f}  CTR@10={r.ctr_10:.3f}")

    # Aggregates
    methods = sorted(set(r.method for r in all_results))
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"  {'Method':22s}  {'nDCG@10':>8}  {'P@10':>6}  {'MRR':>6}  {'CTR@10':>7}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}")

    agg = {}
    for m in methods:
        mrs = [r for r in all_results if r.method == m]
        agg[m] = {
            "ndcg": sum(r.ndcg_10 for r in mrs) / len(mrs),
            "p10": sum(r.precision_10 for r in mrs) / len(mrs),
            "mrr": sum(r.mrr_val for r in mrs) / len(mrs),
            "ctr": sum(r.ctr_10 for r in mrs) / len(mrs),
        }
        a = agg[m]
        print(f"  {m:22s}  {a['ndcg']:8.3f}  {a['p10']:6.2f}  {a['mrr']:6.3f}  {a['ctr']:7.3f}")

    # Lift calculations
    rand_ndcg = agg.get("random", {}).get("ndcg", 0)
    rules_ndcg = agg.get("rules_only", {}).get("ndcg", 0)

    print(f"\nLIFT OVER RANDOM:")
    for m in methods:
        if m == "random":
            continue
        lift = (agg[m]["ndcg"] / rand_ndcg - 1) * 100 if rand_ndcg > 0 else 0
        print(f"  {m:22s}  +{lift:5.1f}% nDCG@10")

    print(f"\nLIFT OVER RULES-ONLY:")
    for m in methods:
        if m in ("random", "rules_only"):
            continue
        lift = (agg[m]["ndcg"] / rules_ndcg - 1) * 100 if rules_ndcg > 0 else 0
        print(f"  {m:22s}  {'+' if lift >= 0 else ''}{lift:5.1f}% nDCG@10")

    # Cold-start analysis: compare users with few vs many interactions
    print(f"\nCOLD-START ANALYSIS:")
    for arch in users[:len(ARCHETYPES)]:
        u_results = [r for r in all_results if r.archetype == arch.name]
        rules_r = next((r for r in u_results if r.method == "rules_only"), None)
        ml_r = next((r for r in u_results if r.method == "rules+ml_learner"), None)
        tower_r = next((r for r in u_results if r.method == "rules+ml+tower"), None)
        if rules_r and ml_r:
            delta = ml_r.ndcg_10 - rules_r.ndcg_10
            tower_delta = (tower_r.ndcg_10 - rules_r.ndcg_10) if tower_r else 0
            print(f"  {arch.name:22s}  rules={rules_r.ndcg_10:.3f}  "
                  f"ml={'+' if delta >= 0 else ''}{delta:.3f}  "
                  f"tower={'+' if tower_delta >= 0 else ''}{tower_delta:.3f}")

    # Final summary
    best_method = max(methods, key=lambda m: agg[m]["ndcg"])
    best_ndcg = agg[best_method]["ndcg"]
    rand_lift = (best_ndcg / rand_ndcg - 1) * 100 if rand_ndcg > 0 else 0
    rules_lift = (best_ndcg / rules_ndcg - 1) * 100 if rules_ndcg > 0 else 0

    print(f"\nSUMMARY:")
    print(f"  Best method: {best_method}")
    print(f"  nDCG@10: {best_ndcg:.3f} (+{rand_lift:.0f}% over random, {'+' if rules_lift >= 0 else ''}{rules_lift:.1f}% over rules)")
    print(f"  Users evaluated: {len(users)}")
    print(f"  Articles: {n_articles}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline recommendation evaluation")
    parser.add_argument("--articles", type=int, default=500)
    parser.add_argument("--extra-users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n_articles=args.articles, n_extra=args.extra_users, seed=args.seed)
