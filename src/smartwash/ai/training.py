"""Training and evaluation pipelines for the SmartWash models."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import classification_report

from ..schemas import FaultType
from .anomaly import FaultDebouncer, FaultDetector
from .datasets import (
    FaultDataset,
    generate_fault_data,
    generate_recommendation_data,
    healthy_cycle_features,
)
from .feedback import FeedbackRecord, training_examples
from .recommender import CycleRecommender

DEFAULT_FEEDBACK_WEIGHT = 30.0
"""How many synthetic examples one piece of the household's own feedback is worth."""


def train_recommender(
    *,
    n_samples: int = 20_000,
    seed: int = 0,
    feedback: Iterable[FeedbackRecord] = (),
    feedback_weight: float = DEFAULT_FEEDBACK_WEIGHT,
    n_estimators: int = 60,
    test_fraction: float = 0.2,
) -> CycleRecommender:
    """Train on expert-labelled synthetic loads plus (heavily weighted) user feedback."""
    started = time.perf_counter()
    profiles, labels = generate_recommendation_data(n_samples, seed)
    n_test = max(1, int(n_samples * test_fraction))
    train_profiles, train_labels = profiles[n_test:], labels[n_test:]
    weights = [1.0] * len(train_profiles)
    examples = training_examples(list(feedback))
    for profile, settings, weight in examples:
        train_profiles.append(profile)
        train_labels.append(settings)
        weights.append(weight * feedback_weight)

    model = CycleRecommender(n_estimators=n_estimators, random_state=seed)
    model.fit(train_profiles, train_labels, sample_weight=np.asarray(weights))
    model.metrics = {
        **model.evaluate(profiles[:n_test], labels[:n_test]),
        "train_samples": len(train_profiles),
        "feedback_examples": len(examples),
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    model.version = f"rec-{time.strftime('%Y%m%d-%H%M%S')}-n{n_samples}-fb{len(examples)}"
    return model


def train_fault_detector(
    *,
    n_cycles: int = 300,
    seed: int = 0,
    n_estimators: int = 150,
    consecutive: int = 2,
    min_confidence: float = 0.6,
    calibration_cycles: int = 40,
    holdout_cycles: int = 30,
) -> FaultDetector:
    """Train on simulated healthy and faulty cycles, holding out whole cycles for evaluation.

    The novelty thresholds are then calibrated on ``calibration_cycles`` complete real-length
    healthy cycles, and the false-alarm rate users would see is measured on ``holdout_cycles``
    more.
    """
    started = time.perf_counter()
    data = generate_fault_data(n_cycles, seed)
    healthy = data.cycle_fault == FaultType.NONE.value
    rng = np.random.default_rng(seed)
    test_ids: list[int] = []
    calibration_ids: list[int] = []
    for group in (np.unique(data.cycle[healthy]), np.unique(data.cycle[~healthy])):
        group = rng.permutation(group)
        n_test = max(1, int(len(group) * 0.2)) if len(group) > 2 else 0
        n_calibration = max(1, int(len(group) * 0.1)) if len(group) > 2 else 0
        test_ids += group[:n_test].tolist()
        calibration_ids += group[n_test : n_test + n_calibration].tolist()
    is_test = np.isin(data.cycle, test_ids)
    is_calibration = np.isin(data.cycle, calibration_ids)
    is_train = ~(is_test | is_calibration)
    if not (is_train & healthy).any():
        raise ValueError("fault training data needs some healthy cycles")

    detector = FaultDetector(n_estimators=n_estimators, random_state=seed)
    detector.fit(
        data.X[is_train],
        data.y[is_train],
        X_healthy=data.X[is_train & healthy],
        X_calibration=data.X[is_calibration & healthy]
        if (is_calibration & healthy).any()
        else None,
    )
    detector.calibrate_on_healthy_cycles(
        healthy_cycle_features(calibration_cycles, seed + 1), consecutive=consecutive
    )
    holdout = healthy_cycle_features(holdout_cycles, seed + 2)
    false_alarms = sum(
        bool(_confirmed_faults(detector, X, consecutive, min_confidence)) for X in holdout
    )
    detector.metrics = {
        **evaluate_fault_detector(
            detector, data, is_test, consecutive=consecutive, min_confidence=min_confidence
        ),
        "full_cycle_false_alarm_rate": round(false_alarms / max(len(holdout), 1), 3),
        "full_cycle_holdout": len(holdout),
        "train_windows": int(is_train.sum()),
        "training_cycles": int(len(np.unique(data.cycle[~is_test]))),
        "novelty_threshold": round(detector.novelty_threshold, 4),
        "envelope_tolerance": round(detector.envelope_tolerance, 4),
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    return detector


def _confirmed_faults(
    detector: FaultDetector, X: np.ndarray, consecutive: int, min_confidence: float
) -> set[FaultType]:
    debouncer = FaultDebouncer(consecutive, min_confidence)
    for fault, confidence, _ in detector.classify(X):
        debouncer.observe(fault, confidence)
    return debouncer.confirmed


def evaluate_fault_detector(
    detector: FaultDetector,
    data: FaultDataset,
    mask: np.ndarray,
    *,
    consecutive: int = 2,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Window-level precision/recall and cycle-level detection with the controller's debouncing."""
    results = detector.classify(data.X[mask])
    predicted = np.asarray([fault.value for fault, _, _ in results])
    truth = data.y[mask]
    report = classification_report(truth, predicted, output_dict=True, zero_division=0)
    per_class = {
        label: {k: round(float(v), 3) for k, v in scores.items()}
        for label, scores in report.items()
        if label in {f.value for f in FaultType}
    }
    healthy_windows = data.cycle_fault[mask] == FaultType.NONE.value

    manifested: Counter[str] = Counter()
    detected: Counter[str] = Counter()
    latency_windows: list[int] = []
    false_alarm_cycles = misdiagnosed_cycles = healthy_cycles = 0
    cycles = data.cycle[mask]
    for cycle in np.unique(cycles):
        rows = np.flatnonzero(cycles == cycle)
        injected = FaultType(data.cycle_fault[mask][rows[0]])
        debouncer = FaultDebouncer(consecutive, min_confidence)
        confirmed_at: dict[FaultType, int] = {}
        for position, row in enumerate(rows):
            fault, confidence, _ = results[row]
            if debouncer.observe(fault, confidence):
                confirmed_at[fault] = position
        if injected is FaultType.NONE:
            healthy_cycles += 1
            false_alarm_cycles += bool(confirmed_at)
            continue
        labelled = np.flatnonzero(truth[rows] == injected.value)
        if len(labelled) < consecutive:
            continue  # this cycle never exercised the fault (e.g. a heater fault on a cold wash)
        manifested[injected.value] += 1
        if injected in confirmed_at:
            detected[injected.value] += 1
            latency_windows.append(confirmed_at[injected] - int(labelled[0]))
        misdiagnosed_cycles += any(f not in (injected, FaultType.UNKNOWN) for f in confirmed_at)

    return {
        "window_accuracy": round(float((predicted == truth).mean()), 4),
        "window_false_alarm_rate": round(
            float((predicted[healthy_windows] != FaultType.NONE.value).mean()), 4
        ),
        "per_class": per_class,
        "cycle_detection_rate": {
            fault: round(detected[fault] / n, 3) for fault, n in sorted(manifested.items())
        },
        "cycle_false_alarm_rate": round(false_alarm_cycles / max(healthy_cycles, 1), 3),
        "cycle_misdiagnosis_rate": round(misdiagnosed_cycles / max(sum(manifested.values()), 1), 3),
        "median_detection_latency_windows": float(np.median(latency_windows))
        if latency_windows
        else None,
        "test_cycles": int(len(np.unique(cycles))),
    }
