from __future__ import annotations

import random

from smartwash.ai.datasets import random_profile
from smartwash.ai.estimator import CycleEstimator
from smartwash.ai.feedback import FeedbackRecord, training_examples
from smartwash.ai.training import train_recommender
from smartwash.knowledge import enforce_care_constraints, program_suits
from smartwash.schemas import CycleSettings, LoadProfile, Program

ESTIMATOR = CycleEstimator()


def test_recommendations_are_always_safe_for_the_fabric(recommender):
    rng = random.Random(42)
    for _ in range(300):
        profile = random_profile(rng)
        rec = recommender.recommend(profile, estimator=ESTIMATOR)
        assert enforce_care_constraints(profile, rec.settings) == (rec.settings, [])


def test_model_learned_the_expert_rules(recommender):
    metrics = recommender.metrics
    assert metrics["program_agreement_with_expert"] > 0.9
    assert metrics["temperature_c_agreement_with_expert"] > 0.9
    assert metrics["exact_match_with_expert"] > 0.8
    assert metrics["guardrail_intervention_rate"] < 0.05


def test_recommendation_is_complete_and_explained(recommender):
    profile = LoadProfile(fabric="sportswear", soil="heavy", weight_kg=3.5, stains=["mud"])
    rec = recommender.recommend(profile, estimator=ESTIMATOR)
    assert rec.settings.program is Program.SPORTSWEAR
    assert rec.settings.prewash
    assert 0 < rec.confidence <= 1
    assert rec.estimate.duration_min > 30
    assert rec.estimate.energy_kwh > 0
    assert rec.estimate.water_l > 0
    assert rec.detergent_ml > 0
    assert len(rec.explanation) >= 4
    assert rec.model_version == recommender.version
    assert all(program_suits(profile, alt.program) for alt in rec.alternatives)


def test_warnings_for_overloading_and_time_budget(recommender):
    rec = recommender.recommend(
        LoadProfile(
            fabric="towels", color="whites", soil="heavy", weight_kg=9, max_duration_min=30
        ),
        estimator=ESTIMATOR,
        capacity_kg=8,
    )
    text = " ".join(rec.warnings)
    assert "capacity" in text
    assert "time budget" in text


def test_feedback_personalises_similar_loads_only():
    everyday = {"fabric": "cotton", "color": "colors", "soil": "normal", "eco_priority": 0.5}
    preference = CycleSettings(program="cotton", temperature_c=30, spin_rpm=1000)
    feedback = [
        FeedbackRecord(profile=LoadProfile(**everyday, weight_kg=w), settings=preference)
        for w in (3.5, 4.0, 4.5, 5.0, 5.5)
    ]
    before = train_recommender(n_samples=6000, seed=1)
    after = train_recommender(n_samples=6000, seed=1, feedback=feedback)
    similar = LoadProfile(**everyday, weight_kg=4.2)
    assert before.recommend(similar, estimator=ESTIMATOR).settings.temperature_c == 40
    personal = after.recommend(similar, estimator=ESTIMATOR).settings
    assert (personal.temperature_c, personal.spin_rpm) == (30, 1000)
    for unrelated in (
        LoadProfile(fabric="wool", color="darks", weight_kg=1.5),
        LoadProfile(fabric="towels", color="whites", weight_kg=5),
    ):
        assert (
            after.recommend(unrelated, estimator=ESTIMATOR).settings
            == before.recommend(unrelated, estimator=ESTIMATOR).settings
        )


def test_unsafe_feedback_is_never_learned():
    wool = LoadProfile(fabric="wool", weight_kg=1.5)
    hot = CycleSettings(program="cotton", temperature_c=60, spin_rpm=1400)
    examples = training_examples([FeedbackRecord(profile=wool, settings=hot)] * 3)
    assert {(s.program, s.temperature_c, s.spin_rpm) for _, s, _ in examples} == {
        (Program.WOOL, 30, 800)
    }
    model = train_recommender(
        n_samples=4000, feedback=[FeedbackRecord(profile=wool, settings=hot)] * 20
    )
    settings = model.recommend(wool, estimator=ESTIMATOR).settings
    assert settings.temperature_c <= 30
    assert settings.spin_rpm <= 800


def test_poorly_rated_feedback_is_ignored_and_average_counts_half():
    profile = LoadProfile(fabric="cotton")
    settings = CycleSettings(program="cotton", temperature_c=40, spin_rpm=1200)
    examples = training_examples(
        [
            FeedbackRecord(profile=profile, settings=settings, rating=1),
            FeedbackRecord(profile=profile, settings=settings, rating=3),
            FeedbackRecord(profile=profile, settings=settings, rating=5),
        ]
    )
    assert [weight for _, _, weight in examples] == [0.5, 1.0]
