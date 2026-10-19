from __future__ import annotations

import json

import pytest

from smartwash import physics
from smartwash.ai.estimator import CycleEstimator
from smartwash.ai.feedback import FeedbackRecord, FeedbackStore
from smartwash.ai.registry import METADATA_FILE, ModelBundle
from smartwash.schemas import CycleSettings, LoadProfile

from .conftest import copy_of

COTTON = CycleSettings(program="cotton", temperature_c=40, spin_rpm=1200)
WOOL = CycleSettings(program="wool", temperature_c=30, spin_rpm=600)


def test_uncalibrated_estimates_come_from_physics():
    assert CycleEstimator().estimate(COTTON, 4.0) == physics.estimate_cycle(COTTON, 4.0)


def test_calibration_learns_this_machine():
    estimator = CycleEstimator()
    base = physics.estimate_cycle(COTTON, 4.0)
    for _ in range(3):
        estimator.observe(COTTON, 4.0, duration_min=base.duration_min * 1.3, energy_kwh=None)
    cotton_factor = estimator.estimate(COTTON, 4.0).duration_min / base.duration_min
    assert 1.2 < cotton_factor < 1.3  # shrunk towards the prior after only three cycles
    wool_base = physics.estimate_cycle(WOOL, 2.0).duration_min
    wool_factor = estimator.estimate(WOOL, 2.0).duration_min / wool_base
    assert 1.0 < wool_factor < cotton_factor  # partly inherits the machine-wide trend
    assert estimator.estimate(COTTON, 4.0).energy_kwh == base.energy_kwh
    assert estimator.observations() == 3

    restored = CycleEstimator(json.loads(json.dumps(estimator.to_dict())))
    assert restored.estimate(COTTON, 4.0) == estimator.estimate(COTTON, 4.0)
    assert restored.summary()["cotton"]["cycles"] == 3


def test_outlier_observations_are_clipped():
    estimator = CycleEstimator()
    estimator.observe(COTTON, 4.0, duration_min=10_000)
    assert estimator.factor(COTTON.program, "duration") < 5


def test_feedback_store_round_trip_skips_bad_lines(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    record = FeedbackRecord(profile=LoadProfile(fabric="cotton"), settings=COTTON, rating=5)
    store.add(record)
    with store.path.open("a") as fh:
        fh.write("not json\n\n")
    store.add(record)
    assert [r.rating for r in store.load()] == [5, 5]
    assert len(store) == 2
    assert FeedbackStore(tmp_path / "missing.jsonl").load() == []


def test_bundle_save_and_load_round_trip(bundle, tmp_path):
    original = copy_of(bundle)
    original.estimator.observe(COTTON, 4.0, duration_min=150)
    original.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    profile = LoadProfile(fabric="denim", color="darks", weight_kg=4)
    assert loaded.recommender.predict([profile]) == original.recommender.predict([profile])
    assert loaded.estimator.to_dict() == original.estimator.to_dict()
    assert loaded.metadata["fault_detector"]["test_cycles"] > 0


def test_load_or_train_creates_models_and_retrains_incompatible_ones(tmp_path):
    created = ModelBundle.load_or_train(tmp_path, recommender_samples=1500, fault_cycles=24)
    assert (tmp_path / METADATA_FILE).exists()
    created.estimator.observe(WOOL, 2.0, duration_min=60)
    created.save_calibration(tmp_path)

    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    metadata["sklearn_version"] = "0.0"
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata))
    with pytest.raises(FileNotFoundError):
        ModelBundle.load_or_train(tmp_path / "empty", auto_train=False)
    retrained = ModelBundle.load_or_train(tmp_path, recommender_samples=1500, fault_cycles=24)
    assert retrained.metadata["sklearn_version"] != "0.0"
    assert retrained.estimator.observations() == 1  # calibration survives retraining


def test_retrain_recommender_with_feedback(bundle):
    retrained = copy_of(bundle)
    feedback = [FeedbackRecord(profile=LoadProfile(fabric="cotton"), settings=COTTON)]
    metrics = retrained.retrain_recommender(feedback, n_samples=2000)
    assert metrics["feedback_examples"] == 1
    assert retrained.metadata["recommender"]["version"] == retrained.recommender.version
    assert retrained.recommender is not bundle.recommender
