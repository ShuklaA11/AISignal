"""Per-user article scoring based on role, level, and topic preferences."""

from __future__ import annotations

from typing import Optional

from src.config import load_settings
from src.storage.models import Article, User, UserMLProfile

# Scoring constants
MAX_SCORE = 20.0  # Upper bound to prevent extreme outliers
DEFAULT_IMPORTANCE = 5.0  # Fallback when base_importance_score is None
TOPIC_MATCH_BOOST = 0.3  # +30% per matching topic
MAX_TOPIC_FACTOR = 2.0  # Cap topic boost (prevents runaway scoring with 4+ matches)
SOURCE_WEIGHT_BASELINE = 5  # Source weight that maps to 1.0x factor
ALPHA_PURE_RULES_THRESHOLD = 0.99  # Above this, skip ML blending entirely

# Role -> category affinity weights
ROLE_CATEGORY_WEIGHTS = {
    "student": {
        "research": 1.5, "open_source": 1.3, "opinion": 0.8,
        "product": 0.7, "industry": 0.6,
    },
    "industry": {
        "product": 1.5, "industry": 1.4, "open_source": 0.9,
        "research": 0.7, "opinion": 0.8,
    },
    "enthusiast": {
        "research": 1.1, "product": 1.1, "open_source": 1.2,
        "industry": 0.9, "opinion": 1.0,
    },
}

# Level -> difficulty preference (how much to boost/penalize mismatched difficulty)
LEVEL_DIFFICULTY_WEIGHTS = {
    "beginner": {"beginner": 1.3, "intermediate": 1.0, "advanced": 0.5},
    "intermediate": {"beginner": 0.8, "intermediate": 1.2, "advanced": 1.0},
    "advanced": {"beginner": 0.4, "intermediate": 0.8, "advanced": 1.4},
}


_RSS_PREFIXES = [
    "techcrunch", "venturebeat", "mit_tech", "the_verge",
    "openai", "anthropic", "deepmind", "huggingface_blog", "meta_ai",
]


def _clamp_factor(value: float, low: float = 0.5, high: float = 2.0) -> float:
    """Clamp a multiplicative scoring factor to [low, high].

    Prevents any single factor from collapsing the final score to near-zero
    or inflating it unboundedly.

    Args:
        value: The raw factor value.
        low: Minimum allowed value (default 0.5).
        high: Maximum allowed value (default 2.0).

    Returns:
        The clamped factor.
    """
    return max(low, min(high, value))


def normalize_source_key(source_name: str) -> str:
    """Normalize raw source_name to the scoring key used in source preferences."""
    if source_name.startswith("r/"):
        return "reddit"
    for prefix in _RSS_PREFIXES:
        if source_name.startswith(prefix):
            return "rss"
    return source_name


def score_article_for_user(
    article: Article,
    user: User,
    ml_profile: Optional[UserMLProfile] = None,
) -> float:
    """Compute a personalized score for an article based on user preferences.

    All multiplicative factors are clamped to [0.5, 2.0] so that no single
    signal can collapse or dominate the final score.

    Args:
        article: The article to score.
        user: The user whose preferences drive the score.
        ml_profile: Optional learned profile; when present and it carries
            topic_weights, those weights replace the uniform +0.3 boost.

    Returns:
        A personalized score in the range (0, MAX_SCORE].
    """
    base = article.base_importance_score or DEFAULT_IMPORTANCE

    # 1. Role-category weight
    category = article.category or "opinion"
    role_weights = ROLE_CATEGORY_WEIGHTS.get(user.role, {})
    role_factor = _clamp_factor(role_weights.get(category, 1.0))

    # 2. Topic match boost
    article_topics = set(article.topics)
    user_topics = set(user.topics)
    if article_topics and user_topics:
        if ml_profile and ml_profile.topic_weights:
            tw = ml_profile.topic_weights
            weighted_boost = sum(
                tw.get(t, 1.0) * TOPIC_MATCH_BOOST
                for t in (article_topics & user_topics)
            )
            topic_factor = min(1.0 + weighted_boost, MAX_TOPIC_FACTOR)
        else:
            overlap = len(article_topics & user_topics)
            topic_factor = min(1.0 + (overlap * TOPIC_MATCH_BOOST), MAX_TOPIC_FACTOR)
    else:
        topic_factor = 1.0
    topic_factor = _clamp_factor(topic_factor)

    # 3. Level-difficulty filter
    difficulty = article.difficulty_level or "intermediate"
    level_weights = LEVEL_DIFFICULTY_WEIGHTS.get(user.level, {})
    level_factor = _clamp_factor(level_weights.get(difficulty, 1.0))

    # 4. Source preference weight
    source_prefs = user.source_preferences
    source_key = normalize_source_key(article.source_name)
    source_weight = source_prefs.get(source_key, SOURCE_WEIGHT_BASELINE)
    source_factor = _clamp_factor(source_weight / SOURCE_WEIGHT_BASELINE)

    final_score = base * role_factor * topic_factor * level_factor * source_factor
    return round(min(final_score, MAX_SCORE), 2)


def score_article_for_user_ml(
    article: Article, user: User, ml_profile: Optional[UserMLProfile] = None,
    embedding_factor: float = 1.0,
) -> float:
    """Compute blended rule-based + ML-learned score.

    When ml_profile is None or has no interaction data, falls back to
    pure rule-based scoring. As users interact, the learned component
    gradually blends in via the alpha parameter.

    embedding_factor: multiplicative semantic similarity signal [0.5, 1.5].
    """
    rule_score = score_article_for_user(article, user, ml_profile=ml_profile)

    if ml_profile is None or ml_profile.alpha >= ALPHA_PURE_RULES_THRESHOLD:
        return rule_score

    learned_score = _compute_learned_score(article, ml_profile, embedding_factor)
    alpha = ml_profile.alpha
    blended = alpha * rule_score + (1 - alpha) * learned_score
    return round(min(blended, MAX_SCORE), 2)


def _compute_learned_score(
    article: Article, profile: UserMLProfile, embedding_factor: float = 1.0,
) -> float:
    """Score an article using only learned behavioral weights."""
    base = article.base_importance_score or DEFAULT_IMPORTANCE

    # Source factor (use normalized key for consistency with rule-based scoring)
    source_w = profile.source_weights
    source_key = normalize_source_key(article.source_name)
    source_factor = _clamp_factor(source_w.get(source_key, 1.0))

    # Category factor
    cat_w = profile.category_weights
    category_factor = _clamp_factor(cat_w.get(article.category or "opinion", 1.0))

    # Topic factor (best matching topic drives the score)
    topic_w = profile.topic_weights
    article_topics = article.topics
    if article_topics:
        topic_scores = [topic_w.get(t, 1.0) for t in article_topics]
        topic_factor = _clamp_factor(max(topic_scores) if topic_scores else 1.0)
    else:
        topic_factor = 1.0

    # Difficulty factor
    diff_w = profile.difficulty_weights
    difficulty_factor = _clamp_factor(
        diff_w.get(article.difficulty_level or "intermediate", 1.0)
    )

    # Entity factor (best matching entity)
    ent_w = profile.entity_weights
    article_entities = article.key_entities[:5]
    if article_entities:
        entity_scores = [ent_w.get(e, 1.0) for e in article_entities]
        entity_factor = _clamp_factor(max(entity_scores) if entity_scores else 1.0)
    else:
        entity_factor = 1.0

    score = (
        base
        * source_factor
        * category_factor
        * topic_factor
        * difficulty_factor
        * entity_factor
        * embedding_factor
    )
    return min(score, MAX_SCORE)
