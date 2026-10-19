from __future__ import annotations

import random

import numpy as np
import pytest

from smartwash.ai.anomaly import (
    FEATURE_NAMES,
    FaultDebouncer,
    TelemetryWindower,
    make_diagnosis,
    window_features,
)
from smartwash.ai.datasets import SIMULATED_FAULTS
from smartwash.devices.simulator import WasherSimulator
from smartwash.schemas import CyclePhase, CycleSettings, FaultType, TelemetrySample


def sample(t: float, phase: CyclePhase = CyclePhase.WASH, **values) -> TelemetrySample:
    fields = {
        "drum_rpm": 50.0,
        "target_rpm": 50.0,
        "motor_current_a": 1.0,
        "vibration_g": 0.04,
        "water_temp_c": 30.0,
        "water_level_l": 12.0,
        "power_w": 150.0,
    }
    return TelemetrySample(timestamp=t, phase=phase, **{**fields, **values})


def confirmed_faults(detector, records) -> set[FaultType]:
    windower = TelemetryWindower()
    debouncer = FaultDebouncer()
    for record in records:
        for window in windower.push(record.sample):
            debouncer.update(detector.assess(window))
    for window in windower.flush():
        debouncer.update(detector.assess(window))
    return debouncer.confirmed


def test_windower_splits_by_phase_and_length():
    windower = TelemetryWindower()
    stream = [sample(t) for t in range(0, 150, 2)] + [
        sample(t, CyclePhase.DRAIN) for t in range(150, 200, 2)
    ]
    windows = [w for s in stream for w in windower.push(s)] + windower.flush()
    assert [len(w) for w in windows] == [30, 30, 15, 25]
    assert all(len({s.phase for s in w}) == 1 for w in windows)


def test_windower_drops_short_windows_and_splits_at_gaps():
    windower = TelemetryWindower()
    short = [sample(t) for t in range(0, 20, 2)]
    assert [w for s in short for w in windower.push(s)] == []
    resumed = [sample(t) for t in range(300, 340, 2)]  # 280 s gap, e.g. a pause
    windows = [w for s in resumed for w in windower.push(s)] + windower.flush()
    assert [len(w) for w in windows] == [20]
    assert windower.push(sample(400, CyclePhase.DONE)) == []


def test_features_need_the_full_sensor_set(detector):
    window = [sample(t) for t in range(0, 60, 2)]
    assert window_features(window).shape == (len(FEATURE_NAMES),)
    window[3] = sample(6, vibration_g=None)
    with pytest.raises(ValueError):
        window_features(window)
    assert detector.assess(window) is None


@pytest.mark.parametrize("fault", SIMULATED_FAULTS)
def test_detects_each_fault_in_a_new_cycle(detector, fault):
    sim = WasherSimulator(seed=1234)
    settings = CycleSettings(program="cotton", temperature_c=60, spin_rpm=1400)
    sim.start(settings, 4.5, fault=fault, severity=1.0, duration_scale=0.5)
    assert fault in confirmed_faults(detector, sim.run())


def test_no_false_alarms_on_healthy_cycles(detector):
    rng = random.Random(99)
    programmes = [
        CycleSettings(program="cotton", temperature_c=40, spin_rpm=1200),
        CycleSettings(program="wool", temperature_c=30, spin_rpm=600),
        CycleSettings(program="eco_40_60", temperature_c=40, spin_rpm=1400, extra_rinse=True),
        CycleSettings(program="synthetics", temperature_c=40, spin_rpm=800, prewash=True),
        CycleSettings(program="quick", temperature_c=20, spin_rpm=1000),
    ]
    for settings in programmes:
        sim = WasherSimulator(seed=rng.randrange(10**6), inlet_temp_c=rng.uniform(10, 18))
        sim.start(settings, rng.uniform(2, 6), duration_scale=0.5)
        assert confirmed_faults(detector, sim.run()) == set(), settings.program


def test_physically_impossible_telemetry_is_flagged(detector):
    window = [
        sample(t, CyclePhase.RINSE, power_w=6000.0, vibration_g=3.0, water_level_l=60.0)
        for t in range(0, 60, 2)
    ]
    features = window_features(window)
    fault, _, _ = detector.classify(features.reshape(1, -1))[0]
    assert fault is not FaultType.NONE
    assert detector.envelope_excess(features.reshape(1, -1)).max() > detector.ENVELOPE_TOLERANCE
    assert detector.most_unusual_signal(features) is not None


def test_unknown_anomaly_names_the_unusual_signal(detector):
    window = [sample(t, CyclePhase.WASH) for t in range(0, 60, 2)]
    features = window_features(window)
    features[FEATURE_NAMES.index("level_mean")] = 500.0  # a sensor gone haywire
    excess = detector.envelope_excess(features.reshape(1, -1))[0]
    assert FEATURE_NAMES[int(np.argmax(excess))] == "level_mean"
    assert detector.most_unusual_signal(features) == "level_mean"


def test_debouncer_needs_consecutive_windows_and_confirms_once():
    debouncer = FaultDebouncer(consecutive=2, min_confidence=0.6)

    def diag(fault, confidence=0.9):
        return make_diagnosis(0.0, CyclePhase.SPIN, fault, confidence, 0.5)

    assert debouncer.update(diag(FaultType.UNBALANCED_LOAD)) is None
    assert debouncer.update(diag(FaultType.NONE)) is None
    assert debouncer.update(diag(FaultType.UNBALANCED_LOAD)) is None
    assert debouncer.update(diag(FaultType.UNBALANCED_LOAD, 0.5)) is None  # too unsure
    assert debouncer.update(diag(FaultType.UNBALANCED_LOAD)) is None
    confirmed = debouncer.update(diag(FaultType.UNBALANCED_LOAD))
    assert confirmed is not None and confirmed.fault is FaultType.UNBALANCED_LOAD
    assert debouncer.update(diag(FaultType.UNBALANCED_LOAD)) is None  # already confirmed
    debouncer.reset()
    debouncer.update(diag(FaultType.UNBALANCED_LOAD))
    assert debouncer.update(diag(FaultType.UNBALANCED_LOAD)) is not None


def test_training_metrics_are_reported(detector):
    metrics = detector.metrics
    assert metrics["cycle_false_alarm_rate"] == 0.0
    assert metrics["window_accuracy"] > 0.9
    assert set(metrics["cycle_detection_rate"]) <= {f.value for f in SIMULATED_FAULTS}
    assert np.isfinite(metrics["novelty_threshold"])
