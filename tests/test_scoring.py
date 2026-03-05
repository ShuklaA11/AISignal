"""Tests for the personalization scoring, learning, and exploration systems.

Covers:
- Rule-based scoring (role/level/topic/source factors)
- ML-blended scoring (alpha blending, learned weights)
- Behavioral learning (EMA updates, signal strengths, learning rate tiers)
- Embedding similarity (cosine similarity, embedding factor mapping)
- Thompson exploration (Beta sampling, score modulation)
- Metrics-driven adaptation (alpha retreat, LR adjustment)
"""

import json
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine

from src.storage.models import (
    Article, FeedImpression, ScoringMetric, User, UserMLProfile, utcnow,
)
from src.personalization.scorer import (
    ALPHA_PURE_RULES_THRESHOLD,
    DEFAULT_IMPORTANCE,
    LEVEL_DIFFICULTY_WEIGHTS,
    MAX_SCORE,
    MAX_TOPIC_FACTOR,
    ROLE_CATEGORY_WEIGHTS,
    SOURCE_WEIGHT_BASELINE,
    TOPIC_MATCH_BOOST,
    _compute_learned_score,
    score_article_for_user,
    score_article_for_user_ml,
)
from src.personalization.learner import (
    _ema,
    _get_base_learning_rate,
    _update_alpha,
    adapt_from_metrics,
    process_skips,
    update_on_click,
    update_on_dislike,
    update_on_like,
    update_on_save,
)
from src.embeddings.similarity import (
    compute_embedding_factor,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """In-memory SQLite for testing."""
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def make_user(
    role="enthusiast", level="intermediate", topics=None,
    source_prefs=None, **kwargs,
) -> User:
    topics = topics or []
    source_prefs = source_prefs or {}
    return User(
        id=kwargs.get("id", 1),
        email=kwargs.get("email", "test@example.com"),
        role=role,
        level=level,
        topics_json=json.dumps(topics),
        source_preferences_json=json.dumps(source_prefs),
    )


def make_article(
    source_name="techcrunch", category="product", topics=None,
    difficulty="intermediate", base_importance=7.0, **kwargs,
) -> Article:
    topics = topics or []
    return Article(
        id=kwargs.get("id", 1),
        url=kwargs.get("url", f"https://example.com/{kwargs.get('id', 1)}"),
        content_hash="abc123",
        title=kwargs.get("title", "Test Article"),
        source_name=source_name,
        source_type="rss",
        category=category,
        base_importance_score=base_importance,
        topics_json=json.dumps(topics),
        difficulty_level=difficulty,
        key_entities_json=json.dumps(kwargs.get("entities", [])),
        status="processed",
    )


def make_ml_profile(
    user_id=1, alpha=0.5, clicks=50, saves=10,
    source_weights=None, category_weights=None, topic_weights=None,
    difficulty_weights=None, entity_weights=None, lr_override=None,
) -> UserMLProfile:
    return UserMLProfile(
        user_id=user_id,
        alpha=alpha,
        total_clicks=clicks,
        total_saves=saves,
        source_weights_json=json.dumps(source_weights or {}),
        category_weights_json=json.dumps(category_weights or {}),
        topic_weights_json=json.dumps(topic_weights or {}),
        difficulty_weights_json=json.dumps(difficulty_weights or {}),
        entity_weights_json=json.dumps(entity_weights or {}),
        learning_rate_override=lr_override,
    )


# ===========================================================================
# 1. Rule-based scoring
# ===========================================================================

class TestRuleBasedScoring:
    """Test score_article_for_user (pure rule-based)."""

    def test_baseline_score_no_preferences(self):
        """Enthusiast with no topics/source prefs → factors are all ~1.0."""
        user = make_user(role="enthusiast", level="intermediate")
        article = make_article(
            category="opinion", difficulty="intermediate", base_importance=5.0,
        )
        score = score_article_for_user(article, user)
        # opinion weight for enthusiast = 1.0, level match = 1.2,
        # source defaults to baseline = 1.0
        assert score == round(5.0 * 1.0 * 1.0 * 1.2 * 1.0, 2)

    def test_student_research_boost(self):
        """Students get 1.5x boost on research articles."""
        user = make_user(role="student", level="intermediate")
        article = make_article(category="research", base_importance=8.0)
        score = score_article_for_user(article, user)
        assert score > 8.0  # Should be boosted
        expected_role_factor = ROLE_CATEGORY_WEIGHTS["student"]["research"]
        assert expected_role_factor == 1.5

    def test_student_product_penalty(self):
        """Students get 0.7x on product articles."""
        user = make_user(role="student", level="intermediate")
        article = make_article(category="product", base_importance=8.0)
        score = score_article_for_user(article, user)
        # 8.0 * 0.7 * 1.2 = 6.72
        assert score < 8.0

    def test_industry_product_boost(self):
        """Industry users get 1.5x on product articles."""
        user = make_user(role="industry", level="intermediate")
        article = make_article(category="product", base_importance=6.0)
        score = score_article_for_user(article, user)
        role_factor = ROLE_CATEGORY_WEIGHTS["industry"]["product"]
        assert role_factor == 1.5
        assert score > 6.0

    def test_topic_overlap_boost(self):
        """Each matching topic adds +30%."""
        user = make_user(topics=["NLP", "Computer Vision", "RL"])
        article = make_article(topics=["NLP", "Computer Vision"])
        score_with_overlap = score_article_for_user(article, user)

        article_no_match = make_article(topics=["AI Safety"])
        score_no_overlap = score_article_for_user(article_no_match, user)

        # 2 matching topics → 1.0 + 2*0.3 = 1.6x vs 1.0x
        assert score_with_overlap > score_no_overlap
        ratio = score_with_overlap / score_no_overlap
        assert abs(ratio - 1.6) < 0.01

    def test_topic_factor_capped_at_max(self):
        """4+ matching topics should not exceed MAX_TOPIC_FACTOR (2.0x)."""
        many_topics = ["NLP", "Computer Vision", "RL", "AI Safety", "Robotics"]
        user = make_user(topics=many_topics)
        article = make_article(topics=many_topics, base_importance=5.0)
        score = score_article_for_user(article, user)

        # Without cap: 1.0 + 5*0.3 = 2.5x → 5.0 * 2.5 = 12.5 (before other factors)
        # With cap: topic_factor = min(2.5, 2.0) = 2.0x
        article_no_topics = make_article(topics=["AI Safety"], base_importance=5.0)
        score_one = score_article_for_user(article_no_topics, user)
        # 1 match → factor 1.3x, 5 matches → capped at 2.0x, ratio = 2.0/1.3
        ratio = score / score_one
        assert abs(ratio - MAX_TOPIC_FACTOR / 1.3) < 0.01

    def test_beginner_advanced_penalty(self):
        """Beginners get 0.5x on advanced articles."""
        user = make_user(level="beginner")
        article = make_article(difficulty="advanced", base_importance=10.0)
        score = score_article_for_user(article, user)
        level_factor = LEVEL_DIFFICULTY_WEIGHTS["beginner"]["advanced"]
        assert level_factor == 0.5
        # Score should be significantly reduced
        assert score < 10.0

    def test_advanced_beginner_penalty(self):
        """Advanced users get 0.4x on beginner articles."""
        user = make_user(level="advanced")
        article = make_article(difficulty="beginner", base_importance=10.0)
        score = score_article_for_user(article, user)
        level_factor = LEVEL_DIFFICULTY_WEIGHTS["advanced"]["beginner"]
        assert level_factor == 0.4

    def test_source_preference_boost(self):
        """Source weight 10 (out of baseline 5) → 2.0x factor."""
        user = make_user(source_prefs={"rss": 10})
        article = make_article(source_name="techcrunch", base_importance=5.0)
        score = score_article_for_user(article, user)
        # techcrunch → normalized to "rss", weight 10/5 = 2.0x
        assert score > 5.0

    def test_source_preference_penalty(self):
        """Source weight 1 (out of baseline 5) → 0.2x factor."""
        user = make_user(source_prefs={"rss": 1})
        article = make_article(source_name="techcrunch", base_importance=5.0)
        score = score_article_for_user(article, user)
        assert score < 5.0

    def test_reddit_source_normalization(self):
        """r/MachineLearning → 'reddit' for source weight lookup."""
        user = make_user(source_prefs={"reddit": 10})
        article = make_article(source_name="r/MachineLearning", base_importance=5.0)
        score = score_article_for_user(article, user)
        # reddit weight = 10/5 = 2.0x
        assert score > 5.0

    def test_score_capped_at_max(self):
        """Score never exceeds MAX_SCORE (20.0)."""
        user = make_user(
            role="student", level="advanced",
            topics=["NLP", "Computer Vision", "RL"],
            source_prefs={"arxiv": 10},
        )
        article = make_article(
            source_name="arxiv", category="research",
            difficulty="advanced", base_importance=10.0,
            topics=["NLP", "Computer Vision", "RL"],
        )
        score = score_article_for_user(article, user)
        assert score <= MAX_SCORE

    def test_missing_importance_uses_default(self):
        """None base_importance_score falls back to 5.0."""
        user = make_user()
        article = make_article(base_importance=None)
        score = score_article_for_user(article, user)
        assert score > 0
        # Should use DEFAULT_IMPORTANCE (5.0) as base
        user2 = make_user()
        article2 = make_article(base_importance=DEFAULT_IMPORTANCE)
        assert score == score_article_for_user(article2, user2)

    def test_unknown_category_defaults_to_1(self):
        """Unknown category gives 1.0x role factor."""
        user = make_user(role="student")
        article = make_article(category="unknown_category", base_importance=5.0)
        score = score_article_for_user(article, user)
        # role_factor = 1.0 (default for unknown), level_factor = 1.2
        assert score == round(5.0 * 1.0 * 1.0 * 1.2 * 1.0, 2)


# ===========================================================================
# 2. ML-blended scoring
# ===========================================================================

class TestMLBlendedScoring:
    """Test score_article_for_user_ml (alpha blending)."""

    def test_cold_start_no_profile(self):
        """No ML profile → pure rule-based score."""
        user = make_user()
        article = make_article()
        ml_score = score_article_for_user_ml(article, user, ml_profile=None)
        rule_score = score_article_for_user(article, user)
        assert ml_score == rule_score

    def test_alpha_1_equals_rules(self):
        """alpha=1.0 → 100% rule-based (cold start)."""
        user = make_user()
        article = make_article()
        profile = make_ml_profile(alpha=1.0)
        ml_score = score_article_for_user_ml(article, user, ml_profile=profile)
        rule_score = score_article_for_user(article, user)
        assert ml_score == rule_score

    def test_alpha_above_threshold_skips_ml(self):
        """alpha >= 0.99 → skip ML entirely (avoids noise from sparse data)."""
        user = make_user()
        article = make_article()
        profile = make_ml_profile(alpha=ALPHA_PURE_RULES_THRESHOLD)
        ml_score = score_article_for_user_ml(article, user, ml_profile=profile)
        rule_score = score_article_for_user(article, user)
        assert ml_score == rule_score

    def test_blending_at_half_alpha(self):
        """alpha=0.5 → 50/50 blend of rule and learned scores."""
        user = make_user()
        article = make_article(source_name="arxiv", category="research")
        profile = make_ml_profile(
            alpha=0.5,
            source_weights={"arxiv": 2.0},
            category_weights={"research": 2.0},
        )
        ml_score = score_article_for_user_ml(article, user, ml_profile=profile)
        rule_score = score_article_for_user(article, user)
        # ML score should be different from pure rules due to learned weights
        # (unless they coincidentally match)
        learned = _compute_learned_score(article, profile)
        expected = 0.5 * rule_score + 0.5 * learned
        assert ml_score == round(min(expected, MAX_SCORE), 2)

    def test_embedding_factor_modulates_learned_score(self):
        """Embedding factor > 1.0 boosts the learned component."""
        user = make_user()
        article = make_article(source_name="arxiv")
        profile = make_ml_profile(
            alpha=0.5,
            source_weights={"arxiv": 1.5},
        )
        score_neutral = score_article_for_user_ml(article, user, profile, embedding_factor=1.0)
        score_boosted = score_article_for_user_ml(article, user, profile, embedding_factor=1.5)
        assert score_boosted > score_neutral

    def test_learned_score_uses_max_topic_weight(self):
        """Learned score picks the best matching topic, not average."""
        article = make_article(topics=["NLP", "RL"])
        profile = make_ml_profile(
            topic_weights={"NLP": 2.5, "RL": 0.5},
        )
        score = _compute_learned_score(article, profile)
        # Should use NLP weight (2.5), not average (1.5) or RL (0.5)
        profile_nlp_only = make_ml_profile(topic_weights={"NLP": 2.5})
        score_nlp = _compute_learned_score(
            make_article(topics=["NLP"]), profile_nlp_only,
        )
        # Both should use 2.5 as the topic factor
        assert score == score_nlp


# ===========================================================================
# 3. EMA learning
# ===========================================================================

class TestEMALearning:
    """Test the exponential moving average update mechanics."""

    def test_ema_positive_click(self):
        """Click signal (+0.5) moves weight toward 1.5."""
        result = _ema(old_weight=1.0, signal=0.5, lr=0.3)
        # target = 1.5, new = 0.7*1.0 + 0.3*1.5 = 1.15
        assert abs(result - 1.15) < 0.001

    def test_ema_positive_like(self):
        """Like signal (+1.5) moves weight toward 2.5."""
        result = _ema(old_weight=1.0, signal=1.5, lr=0.3)
        # target = 2.5, new = 0.7*1.0 + 0.3*2.5 = 1.45
        assert abs(result - 1.45) < 0.001

    def test_ema_negative_dislike(self):
        """Dislike signal (-0.5) moves weight toward 0.5."""
        result = _ema(old_weight=1.0, signal=-0.5, lr=0.3)
        # target = 0.5, new = 0.7*1.0 + 0.3*0.5 = 0.85
        assert abs(result - 0.85) < 0.001

    def test_ema_negative_skip(self):
        """Skip signal (-0.25) moves weight toward 0.75."""
        result = _ema(old_weight=1.0, signal=-0.25, lr=0.3)
        # target = 0.75, new = 0.7*1.0 + 0.3*0.75 = 0.925
        assert abs(result - 0.925) < 0.001

    def test_ema_clamped_floor(self):
        """Weight never drops below 0.1."""
        result = _ema(old_weight=0.1, signal=-0.5, lr=0.3)
        assert result >= 0.1

    def test_ema_clamped_ceiling(self):
        """Weight never exceeds 3.0."""
        result = _ema(old_weight=3.0, signal=1.5, lr=0.3)
        assert result <= 3.0

    def test_repeated_likes_converge(self):
        """Repeated likes converge weight toward 2.5 (capped at 3.0)."""
        weight = 1.0
        for _ in range(100):
            weight = _ema(weight, signal=1.5, lr=0.15)
        # Should converge near target 2.5
        assert abs(weight - 2.5) < 0.01

    def test_repeated_dislikes_converge(self):
        """Repeated dislikes converge weight toward 0.5."""
        weight = 1.0
        for _ in range(100):
            weight = _ema(weight, signal=-0.5, lr=0.15)
        assert abs(weight - 0.5) < 0.01


# ===========================================================================
# 4. Learning rate tiers
# ===========================================================================

class TestLearningRateTiers:
    """Test interaction-count-based learning rate schedule."""

    def test_new_user_fast_lr(self):
        """0-9 interactions → LR=0.3 (fast learning)."""
        profile = make_ml_profile(clicks=3, saves=2)
        assert _get_base_learning_rate(profile) == 0.3

    def test_moderate_user_medium_lr(self):
        """10-49 interactions → LR=0.15."""
        profile = make_ml_profile(clicks=20, saves=5)
        assert _get_base_learning_rate(profile) == 0.15

    def test_mature_user_slow_lr(self):
        """50+ interactions → LR=0.05 (stable refinement)."""
        profile = make_ml_profile(clicks=40, saves=15)
        assert _get_base_learning_rate(profile) == 0.05

    def test_lr_boundary_at_10(self):
        """Exactly 10 interactions → 0.15."""
        profile = make_ml_profile(clicks=8, saves=2)
        assert _get_base_learning_rate(profile) == 0.15

    def test_lr_boundary_at_50(self):
        """Exactly 50 interactions → 0.05."""
        profile = make_ml_profile(clicks=40, saves=10)
        assert _get_base_learning_rate(profile) == 0.05


# ===========================================================================
# 5. Alpha decay
# ===========================================================================

class TestAlphaDecay:
    """Test the rule-vs-learned blending weight decay."""

    def test_alpha_fresh_user(self):
        """0 interactions → alpha=1.0 (pure rules)."""
        profile = make_ml_profile(clicks=0, saves=0, alpha=1.0)
        _update_alpha(profile)
        assert profile.alpha == 1.0

    def test_alpha_50_interactions(self):
        """50 interactions → alpha ~0.50 (smooth exponential decay)."""
        profile = make_ml_profile(clicks=40, saves=10)
        _update_alpha(profile)
        assert profile.alpha == 0.5006

    def test_alpha_floor_at_03(self):
        """150 interactions → alpha approaches floor of 0.3."""
        profile = make_ml_profile(clicks=100, saves=50)
        _update_alpha(profile)
        assert profile.alpha == 0.3165

    def test_alpha_exactly_100(self):
        """100 interactions → alpha still decaying toward 0.3."""
        profile = make_ml_profile(clicks=80, saves=20)
        _update_alpha(profile)
        assert profile.alpha == 0.3575


# ===========================================================================
# 6. Full learning integration (with DB)
# ===========================================================================

class TestLearningWithDB:
    """Test update_on_click/save/like/dislike with real DB sessions."""

    def test_click_creates_profile(self, session):
        """First click auto-creates ML profile."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="arxiv", category="research")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1)

        profile = session.get(UserMLProfile, 1)
        assert profile is not None
        assert profile.total_clicks == 1
        assert profile.source_weights.get("arxiv", 1.0) > 1.0

    def test_click_updates_source_weight(self, session):
        """Clicking an arxiv article increases arxiv source weight."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="arxiv", category="research")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1)
        profile = session.get(UserMLProfile, 1)
        arxiv_weight = profile.source_weights.get("arxiv", 1.0)
        assert arxiv_weight > 1.0  # Moved toward 1.5

    def test_dislike_decreases_weight(self, session):
        """Disliking an article decreases its feature weights."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="rss", category="product")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_dislike(session, user_id=1, article_id=1)
        profile = session.get(UserMLProfile, 1)
        product_weight = profile.category_weights.get("product", 1.0)
        assert product_weight < 1.0  # Moved toward 0.5

    def test_like_is_strongest_positive(self, session):
        """Like signal (+1.5) produces a bigger weight shift than click (+0.5)."""
        user = make_user(id=1)
        a1 = make_article(id=1, source_name="src_a", category="research")
        a2 = make_article(id=2, source_name="src_b", category="research",
                          url="https://example.com/2")
        session.add(user)
        session.add(a1)
        session.add(a2)
        session.commit()

        update_on_click(session, user_id=1, article_id=1)
        profile = session.get(UserMLProfile, 1)
        click_weight = profile.source_weights.get("src_a", 1.0)

        update_on_like(session, user_id=1, article_id=2)
        session.refresh(profile)
        like_weight = profile.source_weights.get("src_b", 1.0)

        # Like should produce a bigger shift from 1.0 than click
        assert (like_weight - 1.0) > (click_weight - 1.0)

    def test_save_updates_save_counter(self, session):
        """Save increments total_saves, not total_clicks."""
        user = make_user(id=1)
        article = make_article(id=1)
        session.add(user)
        session.add(article)
        session.commit()

        update_on_save(session, user_id=1, article_id=1)
        profile = session.get(UserMLProfile, 1)
        assert profile.total_saves == 1
        assert profile.total_clicks == 0

    def test_process_skips_negative_with_position(self, session):
        """Skipped impressions (>6h, not clicked) apply -0.25 signal scaled by position."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="rss", category="product")
        session.add(user)
        session.add(article)
        session.commit()

        # Create an old impression that was shown but not clicked
        imp = FeedImpression(
            user_id=1, article_id=1,
            shown_at=utcnow() - timedelta(hours=48),
            clicked=False, saved=False, processed=False,
            position=0,
        )
        session.add(imp)
        session.commit()

        count = process_skips(session, user_id=1)
        assert count == 1

        profile = session.get(UserMLProfile, 1)
        rss_weight = profile.source_weights.get("rss", 1.0)
        assert rss_weight < 1.0  # Decreased

    def test_process_skips_ignores_recent(self, session):
        """Impressions < 6h old are not processed as skips."""
        user = make_user(id=1)
        article = make_article(id=1)
        session.add(user)
        session.add(article)
        session.commit()

        imp = FeedImpression(
            user_id=1, article_id=1,
            shown_at=utcnow() - timedelta(hours=2),  # Recent (under 6h)
            clicked=False, saved=False, processed=False,
        )
        session.add(imp)
        session.commit()

        count = process_skips(session, user_id=1)
        assert count == 0


# ===========================================================================
# 7. Embedding similarity
# ===========================================================================

class TestEmbeddingSimilarity:
    """Test cosine similarity and embedding factor mapping."""

    def test_identical_vectors(self):
        """Identical vectors → similarity 1.0."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors → similarity 0.0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors → similarity -1.0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_0(self):
        """Zero vector → similarity 0.0 (not NaN)."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_embedding_factor_identical(self):
        """Similarity 1.0 → factor 2.0 (maximum boost, widened range)."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        factor = compute_embedding_factor(v, v)
        assert factor == pytest.approx(2.0)

    def test_embedding_factor_orthogonal(self):
        """Similarity 0.0 → factor 1.0 (neutral)."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        factor = compute_embedding_factor(a, b)
        assert factor == pytest.approx(1.0)

    def test_embedding_factor_opposite(self):
        """Similarity -1.0 → factor 0.3 (maximum penalty, clamped from 0.0)."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        factor = compute_embedding_factor(a, b)
        assert factor == pytest.approx(0.3)

    def test_embedding_factor_missing_returns_neutral(self):
        """Missing embedding → factor 1.0."""
        v = np.array([1.0, 0.0], dtype=np.float32)
        assert compute_embedding_factor(None, v) == 1.0
        assert compute_embedding_factor(v, None) == 1.0
        assert compute_embedding_factor(None, None) == 1.0


# ===========================================================================
# 8. Metrics-driven adaptation
# ===========================================================================

class TestMetricsAdaptation:
    """Test adapt_from_metrics (nightly alpha/LR adjustment)."""

    def _seed_metrics(self, session, user_id, days_data):
        """Helper: seed ScoringMetric rows. days_data is list of (days_ago, ndcg, lift)."""
        from datetime import date
        today = date.today()
        for days_ago, ndcg, lift in days_data:
            m = ScoringMetric(
                user_id=user_id,
                metric_date=today - timedelta(days=days_ago),
                ndcg_at_10=ndcg,
                personalization_lift=lift,
                total_impressions=10,
            )
            session.add(m)
        session.commit()

    def test_low_lift_retreats_to_rules(self, session):
        """Personalization lift < 1.0 → alpha increases (more rule-based)."""
        user = make_user(id=1)
        session.add(user)
        profile = UserMLProfile(
            user_id=1, total_clicks=30, total_saves=10, alpha=0.6,
        )
        session.add(profile)
        session.commit()

        # 3 days of bad lift
        self._seed_metrics(session, 1, [
            (0, 0.5, 0.8),  # lift < 1.0
            (1, 0.5, 0.7),
            (2, 0.5, 0.9),
        ])

        adapt_from_metrics(session, user_id=1)
        session.refresh(profile)

        # Alpha should have increased (retreated toward rules)
        # interaction_alpha = 0.3 + 0.7*exp(-40/40) = 0.5575
        # low lift → 0.5575 + 0.15 = 0.7075
        assert profile.alpha > 0.55

    def test_good_lift_uses_interaction_alpha(self, session):
        """Personalization lift >= 1.0 → alpha follows interaction decay."""
        user = make_user(id=1)
        session.add(user)
        profile = UserMLProfile(
            user_id=1, total_clicks=60, total_saves=20, alpha=0.9,
        )
        session.add(profile)
        session.commit()

        # 3 days of good lift
        self._seed_metrics(session, 1, [
            (0, 0.7, 1.5),
            (1, 0.7, 1.3),
            (2, 0.7, 1.2),
        ])

        adapt_from_metrics(session, user_id=1)
        session.refresh(profile)

        # interaction_alpha = 0.3 + 0.7*exp(-80/40) = 0.3947
        # Good lift → use interaction alpha
        assert profile.alpha <= 0.4

    def test_declining_ndcg_increases_lr(self, session):
        """nDCG declining → LR bumped 1.5x to escape local optimum."""
        user = make_user(id=1)
        session.add(user)
        profile = UserMLProfile(
            user_id=1, total_clicks=30, total_saves=10, alpha=0.6,
        )
        session.add(profile)
        session.commit()

        # 7 days: recent nDCG much lower than prior
        self._seed_metrics(session, 1, [
            (0, 0.3, 1.1),  # recent (low nDCG)
            (1, 0.3, 1.1),
            (2, 0.3, 1.1),
            (3, 0.7, 1.1),  # prior (high nDCG)
            (4, 0.7, 1.1),
            (5, 0.7, 1.1),
            (6, 0.7, 1.1),
        ])

        adapt_from_metrics(session, user_id=1)
        session.refresh(profile)

        # LR should be overridden upward
        assert profile.learning_rate_override is not None
        base_lr = _get_base_learning_rate(profile)
        assert profile.learning_rate_override > base_lr

    def test_too_few_interactions_skips(self, session):
        """< 5 interactions → no adaptation."""
        user = make_user(id=1)
        session.add(user)
        profile = UserMLProfile(
            user_id=1, total_clicks=2, total_saves=1, alpha=0.97,
        )
        session.add(profile)
        session.commit()

        self._seed_metrics(session, 1, [
            (0, 0.5, 0.5), (1, 0.5, 0.5), (2, 0.5, 0.5),
        ])

        adapt_from_metrics(session, user_id=1)
        session.refresh(profile)

        # Should not have changed
        assert profile.alpha == 0.97


# ===========================================================================
# 9. Cross-role scoring comparison
# ===========================================================================

class TestCrossRoleComparison:
    """Verify that the same article scores differently across user roles."""

    def test_research_article_ranking_across_roles(self):
        """Research article: student > enthusiast > industry."""
        article = make_article(
            category="research", base_importance=7.0,
            difficulty="intermediate",
        )
        student = make_user(role="student", level="intermediate")
        industry = make_user(role="industry", level="intermediate")
        enthusiast = make_user(role="enthusiast", level="intermediate")

        s_score = score_article_for_user(article, student)
        i_score = score_article_for_user(article, industry)
        e_score = score_article_for_user(article, enthusiast)

        assert s_score > e_score > i_score

    def test_product_article_ranking_across_roles(self):
        """Product article: industry > enthusiast > student."""
        article = make_article(
            category="product", base_importance=7.0,
            difficulty="intermediate",
        )
        student = make_user(role="student", level="intermediate")
        industry = make_user(role="industry", level="intermediate")
        enthusiast = make_user(role="enthusiast", level="intermediate")

        s_score = score_article_for_user(article, student)
        i_score = score_article_for_user(article, industry)
        e_score = score_article_for_user(article, enthusiast)

        assert i_score > e_score > s_score
