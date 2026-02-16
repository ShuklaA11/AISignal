"""Stress tests for personalization fixes (phases 1-5).

Covers:
- Factor clamping: no single factor can crush or dominate scores
- Weighted topic matching: learned topic weights modulate rule-based boost
- Wider embedding range: [0.3, 2.0] gives semantic signal real leverage
- Skip signal strength: -0.25 base, 6h cutoff, position-aware scaling
- Position bias correction: IPS weighting on learning signals
- Confidence-aware weight decay: high-signal weights decay slower
- Signal count tracking: _apply_signal increments per-feature counts
"""

import json
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine

from src.storage.models import (
    Article, FeedImpression, User, UserMLProfile, utcnow,
)
from src.personalization.scorer import (
    _clamp_factor,
    _compute_learned_score,
    score_article_for_user,
    score_article_for_user_ml,
    MAX_SCORE,
)
from src.personalization.learner import (
    _apply_signal,
    _ema,
    _get_or_create_profile,
    decay_weights,
    process_skips,
    update_on_click,
    update_on_save,
    update_on_like,
    update_on_dislike,
)
from src.embeddings.similarity import compute_embedding_factor
from src.metrics.calculator import compute_position_ctr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
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
    return User(
        id=kwargs.get("id", 1),
        email=kwargs.get("email", "test@example.com"),
        role=role,
        level=level,
        topics_json=json.dumps(topics or []),
        source_preferences_json=json.dumps(source_prefs or {}),
    )


def make_article(
    source_name="techcrunch", category="product", topics=None,
    difficulty="intermediate", base_importance=7.0, **kwargs,
) -> Article:
    return Article(
        id=kwargs.get("id", 1),
        url=kwargs.get("url", f"https://example.com/{kwargs.get('id', 1)}"),
        content_hash=f"hash{kwargs.get('id', 1)}",
        title=kwargs.get("title", "Test Article"),
        source_name=source_name,
        source_type="rss",
        category=category,
        base_importance_score=base_importance,
        topics_json=json.dumps(topics or []),
        difficulty_level=difficulty,
        key_entities_json=json.dumps(kwargs.get("entities", [])),
        status="processed",
    )


def make_ml_profile(
    user_id=1, alpha=0.5, clicks=50, saves=10,
    source_weights=None, category_weights=None, topic_weights=None,
    difficulty_weights=None, entity_weights=None, signal_counts=None,
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
        signal_counts_json=json.dumps(signal_counts or {}),
    )


# ===========================================================================
# 1. Factor clamping stress tests
# ===========================================================================

class TestFactorClamping:
    """Verify clamping prevents score collapse and domination."""

    def test_clamp_lower_bound(self):
        assert _clamp_factor(0.1) == 0.5
        assert _clamp_factor(0.0) == 0.5
        assert _clamp_factor(-1.0) == 0.5

    def test_clamp_upper_bound(self):
        assert _clamp_factor(3.0) == 2.0
        assert _clamp_factor(10.0) == 2.0

    def test_clamp_passthrough(self):
        assert _clamp_factor(1.0) == 1.0
        assert _clamp_factor(0.5) == 0.5
        assert _clamp_factor(2.0) == 2.0
        assert _clamp_factor(1.37) == 1.37

    def test_beginner_advanced_no_longer_crushed(self):
        """Before fix: beginner + advanced = 0.5x factor (unclamped).
        After fix: clamped to 0.5, still penalized but not devastating."""
        user = make_user(level="beginner")
        article = make_article(difficulty="advanced", base_importance=10.0)
        score = score_article_for_user(article, user)
        # Level factor 0.5 (at clamp floor), role ~1.1, score should be reasonable
        assert score >= 4.0  # Not crushed below usability

    def test_advanced_beginner_clamped(self):
        """advanced user + beginner article: 0.4 -> clamped to 0.5."""
        user = make_user(level="advanced")
        article = make_article(difficulty="beginner", base_importance=10.0)
        score = score_article_for_user(article, user)
        # Without clamp: 10 * 1.1 * 1.0 * 0.4 * 1.0 = 4.4
        # With clamp: 10 * 1.1 * 1.0 * 0.5 * 1.0 = 5.5
        assert score >= 5.0

    def test_source_weight_extreme_low_clamped(self):
        """Source weight 1/5 = 0.2 -> clamped to 0.5."""
        user = make_user(source_prefs={"rss": 1})
        article = make_article(source_name="techcrunch", base_importance=10.0)
        score = score_article_for_user(article, user)
        # Without clamp: factor = 0.2x -> 2.0
        # With clamp: factor = 0.5x -> 5.0+
        assert score >= 4.5

    def test_source_weight_extreme_high_clamped(self):
        """Source weight 20/5 = 4.0 -> clamped to 2.0."""
        user = make_user(source_prefs={"rss": 20})
        article = make_article(source_name="techcrunch", base_importance=10.0)
        score = score_article_for_user(article, user)
        # Without clamp: factor = 4.0x -> potentially > MAX_SCORE
        # With clamp: factor = 2.0x -> bounded
        assert score <= MAX_SCORE

    def test_all_worst_case_factors_still_produce_reasonable_score(self):
        """Worst case: all factors at floor. Score should still be usable."""
        # Student + product (0.7->0.7 clamped ok) + beginner-advanced (0.5) + low source
        user = make_user(role="student", level="beginner", source_prefs={"rss": 1})
        article = make_article(
            source_name="techcrunch", category="industry",
            difficulty="advanced", base_importance=5.0,
        )
        score = score_article_for_user(article, user)
        # All factors at/near 0.5 floor: 5.0 * 0.5^4 = 0.3125 is worst without clamp
        # With clamping: 5.0 * 0.6 * 1.0 * 0.5 * 0.5 = 0.75 minimum
        assert score > 0.5

    def test_learned_score_factors_also_clamped(self):
        """_compute_learned_score also clamps factors."""
        article = make_article(
            source_name="arxiv", category="research",
            topics=["NLP"], difficulty="advanced",
            entities=["GPT"],
        )
        # Set extreme weights
        profile = make_ml_profile(
            source_weights={"arxiv": 0.05},  # would be 0.05 unclamped
            category_weights={"research": 5.0},  # would be 5.0 unclamped
            topic_weights={"NLP": 0.01},
            difficulty_weights={"advanced": 4.0},
            entity_weights={"GPT": 0.02},
        )
        score = _compute_learned_score(article, profile)
        # All should be clamped to [0.5, 2.0]
        # base=7.0 * 0.5 * 2.0 * 0.5 * 2.0 * 0.5 * 1.0 = 3.5
        assert 1.0 <= score <= MAX_SCORE


# ===========================================================================
# 2. Weighted topic matching
# ===========================================================================

class TestWeightedTopicMatching:
    """Verify learned topic weights influence rule-based scoring."""

    def test_without_ml_profile_uses_uniform_boost(self):
        """No ml_profile -> each match = +0.3 (old behavior)."""
        user = make_user(topics=["NLP", "CV"])
        article = make_article(topics=["NLP", "CV"])
        score = score_article_for_user(article, user, ml_profile=None)
        # topic_factor = 1.0 + 2*0.3 = 1.6
        user2 = make_user(topics=["NLP", "CV"])
        article2 = make_article(topics=["NLP"])
        score2 = score_article_for_user(article2, user2, ml_profile=None)
        # topic_factor = 1.0 + 1*0.3 = 1.3
        ratio = score / score2
        assert abs(ratio - 1.6 / 1.3) < 0.02

    def test_with_ml_profile_weights_boost_by_topic(self):
        """ML profile with strong NLP weight should boost NLP matches more."""
        user = make_user(topics=["NLP", "Ethics"])
        article_nlp = make_article(topics=["NLP"], id=1, url="http://a.com/1")
        article_ethics = make_article(topics=["Ethics"], id=2, url="http://a.com/2")

        profile = make_ml_profile(
            topic_weights={"NLP": 2.5, "Ethics": 0.5},
        )

        score_nlp = score_article_for_user(article_nlp, user, ml_profile=profile)
        score_ethics = score_article_for_user(article_ethics, user, ml_profile=profile)

        # NLP boost: 1.0 + 2.5*0.3 = 1.75
        # Ethics boost: 1.0 + 0.5*0.3 = 1.15
        assert score_nlp > score_ethics

    def test_weighted_topic_respects_cap(self):
        """Even with high weights, topic factor stays <= MAX_TOPIC_FACTOR (2.0)."""
        user = make_user(topics=["NLP", "CV", "RL", "Safety"])
        article = make_article(topics=["NLP", "CV", "RL", "Safety"])

        profile = make_ml_profile(
            topic_weights={"NLP": 3.0, "CV": 3.0, "RL": 3.0, "Safety": 3.0},
        )

        score = score_article_for_user(article, user, ml_profile=profile)
        # sum = 4 * 3.0 * 0.3 = 3.6, capped: min(1.0+3.6, 2.0) = 2.0
        assert score <= MAX_SCORE

    def test_empty_topic_weights_falls_back_to_uniform(self):
        """ml_profile with empty topic_weights -> uniform boost."""
        user = make_user(topics=["NLP"])
        article = make_article(topics=["NLP"])
        profile = make_ml_profile(topic_weights={})

        score_with = score_article_for_user(article, user, ml_profile=profile)
        score_without = score_article_for_user(article, user, ml_profile=None)
        assert score_with == score_without


# ===========================================================================
# 3. Embedding range stress tests
# ===========================================================================

class TestEmbeddingRange:
    """Verify widened [0.3, 2.0] range gives embeddings real leverage."""

    def test_range_extremes(self):
        """Verify the full [0.3, 2.0] range."""
        identical = np.array([1.0, 0.0], dtype=np.float32)
        opposite = np.array([-1.0, 0.0], dtype=np.float32)
        orthogonal = np.array([0.0, 1.0], dtype=np.float32)

        assert compute_embedding_factor(identical, identical) == pytest.approx(2.0)
        assert compute_embedding_factor(identical, orthogonal) == pytest.approx(1.0)
        assert compute_embedding_factor(identical, opposite) == pytest.approx(0.3)

    def test_moderate_similarity_has_meaningful_effect(self):
        """0.6 similarity -> factor 1.6, a 60% boost (was only 30% before)."""
        a = np.array([1.0, 0.3], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        sim = float(np.dot(a, b))
        factor = compute_embedding_factor(a, b)
        expected = 1.0 + sim
        assert factor == pytest.approx(expected, abs=0.01)
        assert factor > 1.3  # Meaningful boost

    def test_negative_similarity_strong_penalty(self):
        """-0.8 similarity -> factor 0.3 (clamped from 0.2)."""
        a = np.array([1.0, 0.1], dtype=np.float32)
        b = np.array([-1.0, 0.1], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        factor = compute_embedding_factor(a, b)
        assert factor <= 0.3  # Strong penalty, hits clamp floor

    def test_embedding_factor_in_learned_scoring(self):
        """Embedding factor 2.0 should double the learned score component."""
        article = make_article(source_name="arxiv", base_importance=5.0)
        profile = make_ml_profile(source_weights={"arxiv": 1.0})

        score_neutral = _compute_learned_score(article, profile, embedding_factor=1.0)
        score_boost = _compute_learned_score(article, profile, embedding_factor=2.0)

        assert abs(score_boost / score_neutral - 2.0) < 0.01


# ===========================================================================
# 4. Skip signal strength
# ===========================================================================

class TestSkipSignalStrength:
    """Verify skip signals are stronger and faster."""

    def test_skip_ema_target_is_075(self):
        """Skip signal -0.25 -> target = 1.0 + (-0.25) = 0.75."""
        result = _ema(1.0, signal=-0.25, lr=0.3)
        # target=0.75, new = 0.7*1.0 + 0.3*0.75 = 0.925
        assert abs(result - 0.925) < 0.001

    def test_multiple_skips_converge_lower(self):
        """8 consecutive skips should drive weight notably below 1.0."""
        weight = 1.0
        for _ in range(8):
            weight = _ema(weight, signal=-0.25, lr=0.15)
        assert weight < 0.85  # Should have dropped meaningfully

    def test_skip_signal_stronger_than_before(self):
        """Old: -0.1 (target 0.9), New: -0.25 (target 0.75). New drops further."""
        old_result = _ema(1.0, signal=-0.1, lr=0.15)
        new_result = _ema(1.0, signal=-0.25, lr=0.15)
        assert new_result < old_result

    def test_6h_cutoff_processes_old_impressions(self, session):
        """Impressions older than 6h should be processed."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="rss")
        session.add(user)
        session.add(article)
        session.commit()

        imp = FeedImpression(
            user_id=1, article_id=1,
            shown_at=utcnow() - timedelta(hours=8),
            clicked=False, saved=False, processed=False,
            position=0,
        )
        session.add(imp)
        session.commit()

        count = process_skips(session, user_id=1)
        assert count == 1

    def test_6h_cutoff_ignores_recent(self, session):
        """Impressions < 6h old should NOT be processed."""
        user = make_user(id=1)
        article = make_article(id=1)
        session.add(user)
        session.add(article)
        session.commit()

        imp = FeedImpression(
            user_id=1, article_id=1,
            shown_at=utcnow() - timedelta(hours=4),
            clicked=False, saved=False, processed=False,
        )
        session.add(imp)
        session.commit()

        count = process_skips(session, user_id=1)
        assert count == 0

    def test_skip_at_position_8_stronger_than_position_0(self, session):
        """Skip at position 8 should produce a larger weight shift than position 0."""
        user = make_user(id=1)
        a1 = make_article(id=1, source_name="src_a", url="http://a.com/1")
        a2 = make_article(id=2, source_name="src_b", url="http://a.com/2")
        session.add(user)
        session.add(a1)
        session.add(a2)
        session.commit()

        old_time = utcnow() - timedelta(hours=10)

        # Skip at position 0
        imp0 = FeedImpression(
            user_id=1, article_id=1, shown_at=old_time,
            clicked=False, saved=False, processed=False, position=0,
        )
        session.add(imp0)
        session.commit()
        process_skips(session, user_id=1)
        profile = session.exec(
            __import__('sqlmodel', fromlist=['select']).select(UserMLProfile)
            .where(UserMLProfile.user_id == 1)
        ).first()
        weight_pos0 = profile.source_weights.get("src_a", 1.0)

        # Skip at position 8
        imp8 = FeedImpression(
            user_id=1, article_id=2, shown_at=old_time,
            clicked=False, saved=False, processed=False, position=8,
        )
        session.add(imp8)
        session.commit()
        process_skips(session, user_id=1)
        session.refresh(profile)
        weight_pos8 = profile.source_weights.get("src_b", 1.0)

        # Position 8 skip should cause a bigger drop
        assert (1.0 - weight_pos8) > (1.0 - weight_pos0)


# ===========================================================================
# 5. Position bias on explicit signals
# ===========================================================================

class TestPositionBiasOnSignals:
    """Verify position-aware IPS weighting on click/save/like/dislike."""

    def test_click_at_position_0_baseline(self, session):
        """Click at position 0 -> signal = 0.5 * min(2.0, 1.0) = 0.5 (unscaled)."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="src_x")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1, position=0)
        profile = session.get(UserMLProfile, 1)
        w = profile.source_weights.get("src_x", 1.0)
        # signal = 0.5 * 1.0 = 0.5, target = 1.5, LR=0.3 (new user)
        # new = 0.7*1.0 + 0.3*1.5 = 1.15
        assert abs(w - 1.15) < 0.01

    def test_click_at_position_5_stronger(self, session):
        """Click at position 5 -> signal = 0.5 * 1.5 = 0.75, bigger shift."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="src_y")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1, position=5)
        profile = session.get(UserMLProfile, 1)
        w = profile.source_weights.get("src_y", 1.0)
        # signal = 0.5 * 1.5 = 0.75, target = 1.75, LR=0.3
        # new = 0.7*1.0 + 0.3*1.75 = 1.225
        assert abs(w - 1.225) < 0.01

    def test_click_position_capped_at_2x(self, session):
        """Position >= 10 -> multiplier capped at 2.0."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="src_z")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1, position=15)
        profile = session.get(UserMLProfile, 1)
        w = profile.source_weights.get("src_z", 1.0)
        # signal = 0.5 * 2.0 = 1.0, target = 2.0, LR=0.3
        # new = 0.7*1.0 + 0.3*2.0 = 1.3
        assert abs(w - 1.3) < 0.01

    def test_no_position_backward_compatible(self, session):
        """No position arg -> same as old behavior."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="src_w")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1)  # no position
        profile = session.get(UserMLProfile, 1)
        w = profile.source_weights.get("src_w", 1.0)
        # signal = 0.5 (unscaled), target = 1.5, LR=0.3
        assert abs(w - 1.15) < 0.01

    def test_dislike_at_high_position_amplified(self, session):
        """Dislike at position 8 -> signal = -0.5 * 1.8 = -0.9."""
        user = make_user(id=1)
        article = make_article(id=1, source_name="src_d")
        session.add(user)
        session.add(article)
        session.commit()

        update_on_dislike(session, user_id=1, article_id=1, position=8)
        profile = session.get(UserMLProfile, 1)
        w = profile.source_weights.get("src_d", 1.0)
        # signal = -0.5 * 1.8 = -0.9, target = 0.1, LR=0.3
        # new = 0.7*1.0 + 0.3*0.1 = 0.73
        assert abs(w - 0.73) < 0.01


# ===========================================================================
# 6. Signal count tracking
# ===========================================================================

class TestSignalCountTracking:
    """Verify _apply_signal increments per-feature signal counts."""

    def test_click_increments_all_feature_counts(self, session):
        user = make_user(id=1)
        article = make_article(
            id=1, source_name="arxiv", category="research",
            topics=["NLP", "Transformers"], difficulty="advanced",
            entities=["GPT-4"],
        )
        session.add(user)
        session.add(article)
        session.commit()

        update_on_click(session, user_id=1, article_id=1)
        profile = session.get(UserMLProfile, 1)
        counts = profile.signal_counts

        assert counts.get("source:arxiv") == 1
        assert counts.get("category:research") == 1
        assert counts.get("topic:NLP") == 1
        assert counts.get("topic:Transformers") == 1
        assert counts.get("difficulty:advanced") == 1
        assert counts.get("entity:GPT-4") == 1

    def test_multiple_signals_accumulate(self, session):
        user = make_user(id=1)
        a1 = make_article(id=1, source_name="arxiv", category="research", topics=["NLP"])
        a2 = make_article(id=2, source_name="arxiv", category="research", topics=["NLP"],
                          url="http://a.com/2")
        session.add(user)
        session.add(a1)
        session.add(a2)
        session.commit()

        update_on_click(session, user_id=1, article_id=1)
        update_on_click(session, user_id=1, article_id=2)

        profile = session.get(UserMLProfile, 1)
        counts = profile.signal_counts
        assert counts.get("source:arxiv") == 2
        assert counts.get("topic:NLP") == 2


# ===========================================================================
# 7. Confidence-aware weight decay
# ===========================================================================

class TestConfidenceAwareDecay:
    """Verify high-signal weights decay slower than low-signal ones."""

    def test_zero_signals_uses_base_decay(self, session):
        """0 signals -> decay_rate = 0.95 (standard)."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        profile = make_ml_profile(
            user_id=1,
            source_weights={"arxiv": 2.0},
            signal_counts={},
        )
        session.add(profile)
        session.commit()

        decay_weights(session, user_id=1)
        session.refresh(profile)

        # 1.0 + (2.0 - 1.0) * 0.95 = 1.95
        assert profile.source_weights["arxiv"] == pytest.approx(1.95, abs=0.001)

    def test_50_signals_barely_decays(self, session):
        """50+ signals -> decay_rate = 0.99."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        profile = make_ml_profile(
            user_id=1,
            source_weights={"arxiv": 2.0},
            signal_counts={"source:arxiv": 60},
        )
        session.add(profile)
        session.commit()

        decay_weights(session, user_id=1)
        session.refresh(profile)

        # 1.0 + (2.0 - 1.0) * 0.99 = 1.99
        assert profile.source_weights["arxiv"] == pytest.approx(1.99, abs=0.001)

    def test_25_signals_moderate_decay(self, session):
        """25 signals -> decay_rate = 0.95 + 0.04 * 0.5 = 0.97."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        profile = make_ml_profile(
            user_id=1,
            topic_weights={"NLP": 2.0},
            signal_counts={"topic:NLP": 25},
        )
        session.add(profile)
        session.commit()

        decay_weights(session, user_id=1)
        session.refresh(profile)

        # 1.0 + (2.0 - 1.0) * 0.97 = 1.97
        assert profile.topic_weights["NLP"] == pytest.approx(1.97, abs=0.001)

    def test_high_confidence_preserves_over_many_nights(self, session):
        """50+ signals: after 30 nights of decay, weight still above 1.5."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        profile = make_ml_profile(
            user_id=1,
            source_weights={"arxiv": 2.0},
            signal_counts={"source:arxiv": 100},
        )
        session.add(profile)
        session.commit()

        for _ in range(30):
            decay_weights(session, user_id=1)
            session.refresh(profile)

        # decay_rate=0.99, after 30 nights: 1.0 + 1.0 * 0.99^30 = 1.74
        assert profile.source_weights["arxiv"] > 1.5

    def test_low_confidence_decays_to_neutral_quickly(self, session):
        """0 signals: after 30 nights of decay, weight near neutral."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        profile = make_ml_profile(
            user_id=1,
            source_weights={"arxiv": 2.0},
            signal_counts={},
        )
        session.add(profile)
        session.commit()

        for _ in range(30):
            decay_weights(session, user_id=1)
            session.refresh(profile)

        # decay_rate=0.95, after 30 nights: 1.0 + 1.0 * 0.95^30 = 1.215
        assert profile.source_weights["arxiv"] < 1.3

    def test_mixed_confidence_per_feature(self, session):
        """Different features decay at different rates based on their signal counts."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        profile = make_ml_profile(
            user_id=1,
            source_weights={"arxiv": 2.0, "rss": 2.0},
            signal_counts={"source:arxiv": 100, "source:rss": 2},
        )
        session.add(profile)
        session.commit()

        for _ in range(10):
            decay_weights(session, user_id=1)
            session.refresh(profile)

        # arxiv (high confidence) should retain more weight than rss (low confidence)
        assert profile.source_weights["arxiv"] > profile.source_weights["rss"]


# ===========================================================================
# 8. Position CTR baselines
# ===========================================================================

class TestPositionCTR:
    """Verify compute_position_ctr aggregates correctly."""

    def test_empty_returns_empty(self, session):
        result = compute_position_ctr(session)
        assert result == {}

    def test_basic_position_ctr(self, session):
        """Create impressions at different positions, verify per-position CTR."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        now = utcnow()
        # 15 impressions at position 0: 10 clicked
        for i in range(15):
            imp = FeedImpression(
                user_id=1, article_id=i + 1, shown_at=now,
                clicked=(i < 10), position=0,
            )
            session.add(imp)
        # 12 impressions at position 5: 3 clicked
        for i in range(12):
            imp = FeedImpression(
                user_id=1, article_id=100 + i, shown_at=now,
                clicked=(i < 3), position=5,
            )
            session.add(imp)
        session.commit()

        result = compute_position_ctr(session)
        assert abs(result[0] - 10 / 15) < 0.01  # ~0.667
        assert abs(result[5] - 3 / 12) < 0.01   # 0.25

    def test_low_count_uses_global_avg(self, session):
        """Position with < 10 impressions -> falls back to global average."""
        user = make_user(id=1)
        session.add(user)
        session.commit()

        now = utcnow()
        # Position 0: 20 impressions, 10 clicked (CTR=0.5)
        for i in range(20):
            imp = FeedImpression(
                user_id=1, article_id=i + 1, shown_at=now,
                clicked=(i < 10), position=0,
            )
            session.add(imp)
        # Position 9: only 5 impressions, 1 clicked
        for i in range(5):
            imp = FeedImpression(
                user_id=1, article_id=200 + i, shown_at=now,
                clicked=(i < 1), position=9,
            )
            session.add(imp)
        session.commit()

        result = compute_position_ctr(session)
        # Position 0 has enough data: exact CTR
        assert abs(result[0] - 0.5) < 0.01
        # Position 9 < 10 impressions: uses global avg
        # Global: 11/25 = 0.44
        assert abs(result[9] - 11 / 25) < 0.01


# ===========================================================================
# 9. End-to-end integration: all fixes working together
# ===========================================================================

class TestEndToEndIntegration:
    """Full pipeline: scoring -> signals -> decay -> re-scoring."""

    def test_full_cycle(self, session):
        """Simulate: new user -> clicks -> skips -> decay -> verify improvement."""
        user = make_user(id=1, topics=["NLP", "CV"])
        session.add(user)

        # Create articles
        liked_article = make_article(
            id=1, source_name="arxiv", category="research",
            topics=["NLP"], difficulty="intermediate",
        )
        skipped_article = make_article(
            id=2, source_name="techcrunch", category="product",
            topics=["Blockchain"], difficulty="beginner",
            url="http://a.com/2",
        )
        session.add(liked_article)
        session.add(skipped_article)
        session.commit()

        # Score before any signals (rule-based only)
        score_liked_before = score_article_for_user(liked_article, user)
        score_skipped_before = score_article_for_user(skipped_article, user)

        # Simulate interactions
        update_on_click(session, user_id=1, article_id=1, position=0)
        update_on_click(session, user_id=1, article_id=1, position=0)
        update_on_save(session, user_id=1, article_id=1, position=0)

        # Add skip impression
        imp = FeedImpression(
            user_id=1, article_id=2,
            shown_at=utcnow() - timedelta(hours=8),
            clicked=False, saved=False, processed=False, position=0,
        )
        session.add(imp)
        session.commit()
        process_skips(session, user_id=1)

        # Decay once
        decay_weights(session, user_id=1)

        # Re-score with ML profile
        profile = session.exec(
            __import__('sqlmodel', fromlist=['select']).select(UserMLProfile)
            .where(UserMLProfile.user_id == 1)
        ).first()

        score_liked_after = score_article_for_user_ml(
            liked_article, user, ml_profile=profile,
        )
        score_skipped_after = score_article_for_user_ml(
            skipped_article, user, ml_profile=profile,
        )

        # After learning, the liked article should be relatively stronger
        # compared to the skipped one
        gap_before = score_liked_before - score_skipped_before
        gap_after = score_liked_after - score_skipped_after
        assert gap_after > gap_before
