"""AI fault detection from machine telemetry.

Telemetry is cut into windows of one phase and up to a minute long. Each window becomes a vector of
physically meaningful features (speed deficit, vibration normalised by speed, heater draw, fill and
drain rates, ...). Two models then look at it:

* a random forest trained on simulated faults diagnoses the known fault types;
* an isolation forest trained on healthy cycles flags anything else that looks unusual.

The controller confirms a fault only after several consecutive windows agree (``FaultDebouncer``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .. import physics
from ..knowledge import FAULT_GUIDE
from ..schemas import FAULT_DETECTION_FIELDS, CyclePhase, Diagnosis, FaultType, TelemetrySample

T = TypeVar("T")

WINDOW_S = 60.0
MIN_WINDOW_S = 30.0
MAX_GAP_S = 15.0
MIN_WINDOW_SAMPLES = 5
TYPICAL_LOAD_KG = 4.0
WINDOW_PHASES: tuple[CyclePhase, ...] = (
    CyclePhase.FILL,
    CyclePhase.WASH,
    CyclePhase.RINSE,
    CyclePhase.DRAIN,
    CyclePhase.SPIN,
)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"phase={p.value}" for p in WINDOW_PHASES),
    "target_rpm_mean",
    "rpm_mean",
    "rpm_std",
    "rpm_deficit",
    "rpm_deficit_rel",
    "rpm_slope",
    "vibration_mean",
    "vibration_p95",
    "vibration_std",
    "vibration_per_speed",
    "current_mean",
    "current_p95",
    "current_moving_mean",
    "temp_mean",
    "temp_slope",
    "heat_demand_frac",
    "temp_deficit",
    "heater_power_w",
    "heating_efficiency",
    "level_mean",
    "level_slope",
    "power_mean",
    "power_p95",
    "power_non_motor",
)


def _slope_per_min(minutes: np.ndarray, values: np.ndarray) -> float:
    x = minutes - minutes.mean()
    denom = float(x @ x)
    return float(x @ (values - values.mean()) / denom) if denom > 0 else 0.0


def _heating_efficiency(
    minutes: np.ndarray, temp: np.ndarray, level: np.ndarray, heat_demand: np.ndarray
) -> float:
    """Measured temperature rise relative to what a working heater achieves at this water level.

    About 1 for a healthy heater whatever the target temperature; near 0 when it has failed.
    """
    if heat_demand.mean() < 0.5:
        return 0.0
    mass = (
        float(level.mean()) * physics.WATER_J_PER_L_K + TYPICAL_LOAD_KG * physics.LAUNDRY_J_PER_KG_K
    )
    expected = physics.HEATER_POWER_W * 60.0 / max(mass, 1.0)
    return _slope_per_min(minutes, temp) / expected


def has_required_fields(sample: TelemetrySample) -> bool:
    return all(getattr(sample, name) is not None for name in FAULT_DETECTION_FIELDS)


def window_features(samples: Sequence[TelemetrySample]) -> np.ndarray:
    """Feature vector for one window of samples from a single phase."""
    if not samples or not all(has_required_fields(s) for s in samples):
        raise ValueError("window lacks the telemetry fields the fault detector needs")

    def column(name: str) -> np.ndarray:
        return np.fromiter((getattr(s, name) for s in samples), dtype=float, count=len(samples))

    minutes = (column("timestamp") - samples[0].timestamp) / 60.0
    rpm = column("drum_rpm")
    target_rpm = column("target_rpm")
    current = column("motor_current_a")
    vibration = column("vibration_g")
    temp = column("water_temp_c")
    level = column("water_level_l")
    power = column("power_w")
    target_temp = np.fromiter(
        (np.nan if s.target_temp_c is None else s.target_temp_c for s in samples),
        dtype=float,
        count=len(samples),
    )

    heating = ~np.isnan(target_temp)
    heat_demand = heating & (temp < np.where(heating, target_temp, 0.0) - 1.0)
    motor_power = physics.motor_power_w(1.0) * current
    non_motor = power - motor_power
    moving = rpm > 5.0
    target_mean = float(target_rpm.mean())
    rpm_mean = float(rpm.mean())
    deficit = float((target_rpm - rpm).mean())
    phase = samples[0].phase

    return np.array(
        [
            *(float(phase is p) for p in WINDOW_PHASES),
            target_mean,
            rpm_mean,
            float(rpm.std()),
            deficit,
            deficit / max(target_mean, 20.0),
            _slope_per_min(minutes, rpm),
            float(vibration.mean()),
            float(np.percentile(vibration, 95)),
            float(vibration.std()),
            float(vibration.mean()) / (0.05 + (rpm_mean / 1000.0) ** 2),
            float(current.mean()),
            float(np.percentile(current, 95)),
            float(current[moving].mean()) if moving.any() else 0.0,
            float(temp.mean()),
            _slope_per_min(minutes, temp),
            float(heat_demand.mean()),
            float((target_temp[heating] - temp[heating]).mean()) if heating.any() else 0.0,
            float(non_motor[heat_demand].mean()) if heat_demand.any() else 0.0,
            _heating_efficiency(minutes, temp, level, heat_demand),
            float(level.mean()),
            _slope_per_min(minutes, level),
            float(power.mean()),
            float(np.percentile(power, 95)),
            float(non_motor.mean()),
        ]
    )


def _as_sample(item: Any) -> TelemetrySample:
    return item


class TelemetryWindower(Generic[T]):
    """Cuts a stream of telemetry into windows of one phase, at most ``window_s`` long.

    Works on samples directly or, with ``key``, on any items that carry one (such as labelled
    simulator records). A window closes when the phase changes, it reaches ``window_s``, or the
    stream has a gap (for example while the machine was paused); windows shorter than
    ``min_window_s`` are dropped.
    """

    def __init__(
        self,
        *,
        window_s: float = WINDOW_S,
        min_window_s: float = MIN_WINDOW_S,
        max_gap_s: float = MAX_GAP_S,
        key: Callable[[T], TelemetrySample] | None = None,
    ) -> None:
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.max_gap_s = max_gap_s
        self._key: Callable[[T], TelemetrySample] = key if key is not None else _as_sample
        self._items: list[T] = []

    def push(self, item: T) -> list[list[T]]:
        """Add one item; return any windows that are now complete."""
        sample = self._key(item)
        if sample.phase not in WINDOW_PHASES:
            return self.flush()
        completed: list[list[T]] = []
        if self._items:
            first, last = self._key(self._items[0]), self._key(self._items[-1])
            if (
                sample.phase != first.phase
                or sample.timestamp - first.timestamp >= self.window_s
                or not 0 <= sample.timestamp - last.timestamp <= self.max_gap_s
            ):
                completed = self.flush()
        self._items.append(item)
        return completed

    def flush(self) -> list[list[T]]:
        """Close the current window, returning it if it is long enough."""
        items, self._items = self._items, []
        if len(items) < MIN_WINDOW_SAMPLES:
            return []
        first, last = self._key(items[0]), self._key(items[-1])
        period = (last.timestamp - first.timestamp) / (len(items) - 1)
        return [items] if last.timestamp - first.timestamp + period >= self.min_window_s else []

    def reset(self) -> None:
        self._items = []


class FaultDetector:
    """Diagnoses telemetry windows: known faults by classifier, anything else by novelty checks.

    Two novelty checks complement each other: an isolation forest catches unusual *combinations*
    of signals, and a per-phase envelope catches single signals far outside anything a healthy
    machine produced in that phase (which an isolation forest dilutes across many features).
    """

    MIN_FAULT_PROBABILITY = 0.5
    ENVELOPE_TOLERANCE = 0.5
    """How far beyond the healthy range, in multiples of its width, a signal may stray."""

    def __init__(self, *, n_estimators: int = 150, random_state: int = 0) -> None:
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
        self.novelty = IsolationForest(n_estimators=200, random_state=random_state)
        self.novelty_threshold = 0.6
        self.envelope_tolerance = self.ENVELOPE_TOLERANCE
        shape = (len(WINDOW_PHASES), len(FEATURE_NAMES))
        self._envelope_low = np.full(shape, -np.inf)
        self._envelope_high = np.full(shape, np.inf)
        self._envelope_scale = np.ones(shape)
        self.metrics: dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_healthy: np.ndarray,
        X_calibration: np.ndarray | None = None,
        *,
        false_alarm_quantile: float = 0.995,
    ) -> FaultDetector:
        """Train on labelled windows; learn what healthy looks like from healthy-cycle windows."""
        self.classifier.fit(X, y)
        self.classifier.n_jobs = 1  # live diagnosis is one window at a time
        self.novelty.fit(X_healthy)
        self._fit_envelope(X_healthy)
        calibration = X_calibration if X_calibration is not None else X_healthy
        self.novelty_threshold = float(
            np.quantile(self.anomaly_scores(calibration), false_alarm_quantile)
        )
        return self

    def _fit_envelope(self, X_healthy: np.ndarray) -> None:
        phases = X_healthy[:, : len(WINDOW_PHASES)].argmax(axis=1)
        overall = np.quantile(X_healthy, 0.999, axis=0) - np.quantile(X_healthy, 0.001, axis=0)
        floor = np.maximum(0.1 * overall, 1e-3)
        for p in range(len(WINDOW_PHASES)):
            rows = X_healthy[phases == p]
            if len(rows) < 20:
                continue  # too little healthy data for this phase to judge it
            self._envelope_low[p] = np.quantile(rows, 0.001, axis=0)
            self._envelope_high[p] = np.quantile(rows, 0.999, axis=0)
            self._envelope_scale[p] = np.maximum(
                self._envelope_high[p] - self._envelope_low[p], floor
            )

    def calibrate_on_healthy_cycles(
        self, cycles: Sequence[np.ndarray], *, consecutive: int, margin: float = 0.01
    ) -> None:
        """Raise the novelty thresholds until none of these healthy cycles would raise an alarm.

        ``cycles`` holds the window features of complete healthy cycles, in time order. Windows of
        one cycle are strongly correlated, so a per-window false-alarm rate understates how often
        a debounced alarm fires; this calibrates on what users actually see.
        """
        for X in cycles:
            if len(X) < consecutive:
                continue
            peak = _sustained_peak(self.anomaly_scores(X), consecutive)
            self.novelty_threshold = max(self.novelty_threshold, peak + margin)
            peak = _sustained_peak(self.envelope_excess(X).max(axis=1), consecutive)
            self.envelope_tolerance = max(self.envelope_tolerance, peak + margin)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Isolation-forest anomaly scores in (0, 1]; healthy windows score around 0.4-0.5."""
        return -self.novelty.score_samples(X)

    def envelope_excess(self, X: np.ndarray) -> np.ndarray:
        """Per feature, how far outside the healthy range for its phase, in range widths."""
        phases = X[:, : len(WINDOW_PHASES)].argmax(axis=1)
        low, high = self._envelope_low[phases], self._envelope_high[phases]
        excess = np.maximum(X - high, low - X) / self._envelope_scale[phases]
        return np.maximum(excess, 0.0)

    def classify(self, X: np.ndarray) -> list[tuple[FaultType, float, float]]:
        """``(fault, confidence, anomaly_score)`` for each row of window features."""
        probabilities = self.classifier.predict_proba(X)
        scores = self.anomaly_scores(X)
        excess = self.envelope_excess(X).max(axis=1)
        classes = [FaultType(c) for c in self.classifier.classes_]
        none_index = classes.index(FaultType.NONE)
        results = []
        for probs, score, out in zip(probabilities, scores, excess, strict=True):
            best = int(np.argmax(probs))
            if classes[best] is not FaultType.NONE and probs[best] >= self.MIN_FAULT_PROBABILITY:
                results.append((classes[best], float(probs[best]), float(score)))
            elif score > self.novelty_threshold or out > self.envelope_tolerance:
                confidence = max(
                    0.5 + 10.0 * (score - self.novelty_threshold),
                    0.5 + 0.25 * (out - self.envelope_tolerance),
                )
                results.append((FaultType.UNKNOWN, min(0.99, confidence), float(score)))
            else:
                results.append((FaultType.NONE, float(probs[none_index]), float(score)))
        return results

    def most_unusual_signal(self, features: np.ndarray) -> str | None:
        """The feature furthest outside its healthy range, if any is outside it."""
        excess = self.envelope_excess(features.reshape(1, -1))[0]
        worst = int(np.argmax(excess))
        return FEATURE_NAMES[worst] if excess[worst] > 0 else None

    def assess(self, window: Sequence[TelemetrySample]) -> Diagnosis | None:
        """Diagnose one window, or ``None`` if it lacks the telemetry needed."""
        try:
            features = window_features(window)
        except ValueError:
            return None
        fault, confidence, score = self.classify(features.reshape(1, -1))[0]
        detail = self.most_unusual_signal(features) if fault is FaultType.UNKNOWN else None
        return make_diagnosis(
            window[-1].timestamp, window[0].phase, fault, confidence, score, detail=detail
        )


def _sustained_peak(values: np.ndarray, consecutive: int) -> float:
    """The highest level that ``consecutive`` values in a row all reach."""
    windows = np.lib.stride_tricks.sliding_window_view(values, consecutive)
    return float(windows.min(axis=1).max())


def make_diagnosis(
    timestamp: float,
    phase: CyclePhase,
    fault: FaultType,
    confidence: float,
    score: float,
    *,
    detail: str | None = None,
) -> Diagnosis:
    advice = FAULT_GUIDE[fault]
    message = advice.message
    if detail:
        message += f" Most unusual signal: {detail.replace('_', ' ')}."
    return Diagnosis(
        timestamp=timestamp,
        phase=phase,
        fault=fault,
        confidence=round(confidence, 3),
        anomaly_score=round(score, 3),
        severity=advice.severity,
        message=message,
        recommended_action=advice.action,
    )


class FaultDebouncer:
    """Confirms a fault once it has been diagnosed in ``consecutive`` windows in a row.

    Each fault is confirmed at most once until ``reset()`` (call it when a new cycle starts).
    """

    def __init__(self, consecutive: int = 2, min_confidence: float = 0.6) -> None:
        if consecutive < 1:
            raise ValueError("consecutive must be at least 1")
        self.consecutive = consecutive
        self.min_confidence = min_confidence
        self._streak_fault = FaultType.NONE
        self._streak = 0
        self.confirmed: set[FaultType] = set()

    def update(self, diagnosis: Diagnosis) -> Diagnosis | None:
        """Feed one window's diagnosis; return it if this confirms a new fault."""
        return diagnosis if self.observe(diagnosis.fault, diagnosis.confidence) else None

    def observe(self, fault: FaultType, confidence: float) -> bool:
        """Feed one window's verdict; ``True`` if it confirms a new fault."""
        if fault is FaultType.NONE or confidence < self.min_confidence:
            self._streak_fault, self._streak = FaultType.NONE, 0
            return False
        if fault is self._streak_fault:
            self._streak += 1
        else:
            self._streak_fault, self._streak = fault, 1
        if self._streak >= self.consecutive and fault not in self.confirmed:
            self.confirmed.add(fault)
            return True
        return False

    def reset(self) -> None:
        self._streak_fault, self._streak = FaultType.NONE, 0
        self.confirmed = set()
