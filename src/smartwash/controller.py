"""The controller: connects the AI to the washing machine.

It starts cycles with recommended (or user-chosen, guard-railed) settings, watches the machine
while it runs, feeds telemetry to the fault detector, pauses the cycle when a critical fault is
confirmed, sends notifications, and learns this machine's real cycle times and energy use.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from .ai.anomaly import FaultDebouncer, TelemetryWindower
from .ai.feedback import FeedbackRecord, FeedbackStore
from .ai.registry import ModelBundle
from .config import MonitoringConfig, Settings
from .devices.base import DeviceError, DeviceNotReady, UnsupportedOperation, WashingMachineAdapter
from .devices.factory import create_adapter
from .devices.simulator import SimulatedWasherAdapter
from .knowledge import enforce_care_constraints
from .notifications import Notifier
from .schemas import (
    ACTIVE_STATUSES,
    FAULT_DETECTION_FIELDS,
    CycleEstimate,
    CycleSettings,
    Diagnosis,
    FaultType,
    LoadProfile,
    MachineState,
    MachineStatus,
    Recommendation,
    Severity,
    TelemetrySample,
)

log = logging.getLogger(__name__)

MAX_SAMPLE_GAP_S = 15.0
"""Telemetry gaps longer than this (pauses, outages) are not counted as running time."""
START_GRACE_S = 120.0
"""Cloud APIs can report the old state for a while after a start command. Until the machine has
reported the new cycle as running, an idle or finished state is not taken as its end for this long.
"""


class StartCycleRequest(BaseModel):
    """Start from a load profile (the AI picks the settings), explicit settings, or both.

    With both, the settings are checked against the load's care limits (unless
    ``enforce_care_rules`` is false) and remembered as a preference for personalisation.
    """

    profile: LoadProfile | None = None
    settings: CycleSettings | None = None
    enforce_care_rules: bool = True

    @model_validator(mode="after")
    def _needs_profile_or_settings(self) -> StartCycleRequest:
        if self.profile is None and self.settings is None:
            raise ValueError("give a load profile, explicit settings, or both")
        return self


class StartCycleResult(BaseModel):
    settings: CycleSettings
    """The settings SmartWash chose; see ``applied`` for whether the machine could take them."""
    applied: bool = True
    """False when the connection cannot choose programme, temperature and spin, so the machine
    ran whatever was set on it."""
    recommendation: Recommendation | None = None
    adjustments: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimate: CycleEstimate
    state: MachineState


class FeedbackRequest(BaseModel):
    profile: LoadProfile
    settings: CycleSettings
    """The settings you used (or would have preferred) for this load."""
    rating: int | None = Field(default=None, ge=1, le=5)


class FaultInjection(BaseModel):
    fault: FaultType
    severity: float = Field(default=1.0, ge=0, le=2)


class CycleRecord(BaseModel):
    started_at: float
    ended_at: float | None = None
    outcome: str = "running"
    """``running``, ``finished``, ``stopped`` or ``error``."""
    source: str = "smartwash"
    """``smartwash`` when started through SmartWash, ``machine`` when started on the machine."""
    settings: CycleSettings | None = None
    vendor_program: str | None = None
    profile: LoadProfile | None = None
    estimate: CycleEstimate | None = None
    duration_min: float | None = None
    energy_kwh: float | None = None
    faults: list[Diagnosis] = Field(default_factory=list)
    error_code: str | None = None


class MachineSnapshot(BaseModel):
    adapter: str
    device: dict[str, Any]
    capabilities: dict[str, Any]
    fault_detection: bool
    monitoring: bool
    state: MachineState
    cycle: CycleRecord | None = None
    last_error: str | None = None


class DiagnosticsReport(BaseModel):
    fault_detection: bool
    unavailable_reason: str | None = None
    latest: Diagnosis | None = None
    """The detector's verdict on the most recent telemetry window."""
    cycle_faults: list[Diagnosis] = Field(default_factory=list)
    recent_alerts: list[Diagnosis] = Field(default_factory=list)
    windows_assessed: int = 0


class RetrainInProgress(RuntimeError):
    pass


class CycleHistory:
    """Completed cycles, as a JSON-lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: CycleRecord) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")

    def recent(self, limit: int = 20) -> list[CycleRecord]:
        """The latest ``limit`` cycles, newest first."""
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in reversed(lines):
            if len(records) >= limit:
                break
            try:
                records.append(CycleRecord.model_validate_json(line))
            except ValidationError:
                log.warning("Skipping malformed line in %s", self.path)
        return records


def _settings_from_state(state: MachineState) -> CycleSettings | None:
    if state.program is None or state.temperature_c is None or state.spin_rpm is None:
        return None
    try:
        return CycleSettings(
            program=state.program, temperature_c=state.temperature_c, spin_rpm=state.spin_rpm
        )
    except ValidationError:
        return None


class WasherController:
    def __init__(
        self,
        adapter: WashingMachineAdapter,
        models: ModelBundle,
        *,
        model_dir: str | Path,
        feedback: FeedbackStore,
        history: CycleHistory,
        notifier: Notifier | None = None,
        capacity_kg: float = 8.0,
        monitoring: MonitoringConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.adapter = adapter
        self.models = models
        self.model_dir = Path(model_dir)
        self.feedback = feedback
        self.history = history
        self.notifier = notifier or Notifier()
        self.capacity_kg = capacity_kg
        self.monitoring = monitoring or MonitoringConfig()
        self._clock = clock
        self._lock = threading.RLock()  # serialises machine access and controller state
        self._retrain_lock = threading.Lock()
        self._windower: TelemetryWindower[TelemetrySample] = TelemetryWindower()
        self._debouncer = FaultDebouncer(
            self.monitoring.consecutive_windows, self.monitoring.min_confidence
        )
        self._cycle: CycleRecord | None = None
        self._running_s = 0.0
        self._telemetry_s = 0.0
        self._energy_j = 0.0
        self._has_power = False
        self._last_sample: TelemetrySample | None = None
        self._last_state: MachineState | None = None
        self._last_seen: float | None = None
        self._seen_active = False
        self._latest: Diagnosis | None = None
        self._alerts: deque[Diagnosis] = deque(maxlen=50)
        self._windows_assessed = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_error: str | None = None

    # --------------------------------------------------------------------- AI

    @property
    def fault_detection(self) -> bool:
        return self.adapter.capabilities.fault_detection

    def recommend(self, profile: LoadProfile) -> Recommendation:
        return self.models.recommender.recommend(
            profile, estimator=self.models.estimator, capacity_kg=self.capacity_kg
        )

    # ---------------------------------------------------------------- machine

    def snapshot(self) -> MachineSnapshot:
        with self._lock:
            state = self._refresh_state()
            caps = self.adapter.capabilities
            return MachineSnapshot(
                adapter=self.adapter.name,
                device=self.adapter.describe(),
                capabilities={
                    "remote_start": caps.remote_start,
                    "select_program": caps.select_program,
                    "pause_resume": caps.pause_resume,
                    "stop": caps.stop,
                    "telemetry_fields": sorted(caps.telemetry_fields),
                },
                fault_detection=caps.fault_detection,
                monitoring=self.is_monitoring,
                state=state,
                cycle=self._cycle,
                last_error=self.last_error,
            )

    def start_cycle(self, request: StartCycleRequest) -> StartCycleResult:
        profile = request.profile
        recommendation: Recommendation | None = None
        adjustments: list[str] = []
        if request.settings is None:
            assert profile is not None
            recommendation = self.recommend(profile)
            settings = recommendation.settings
        else:
            settings = request.settings
            if profile is not None and request.enforce_care_rules:
                settings, adjustments = enforce_care_constraints(profile, settings)
        capabilities = self.adapter.capabilities
        if not capabilities.remote_start:
            raise UnsupportedOperation(f"the {self.adapter.name} connection cannot start cycles")
        applied = capabilities.select_program
        warnings = []
        if not applied:
            warnings.append(
                f"The {self.adapter.name} connection cannot choose the programme, temperature or "
                "spin, so the machine runs whatever is set on it. Make sure that is no harsher "
                f"than {settings.program.value} at {settings.temperature_c} °C and "
                f"{settings.spin_rpm} rpm."
            )
        load_kg = profile.weight_kg if profile else None
        estimate = (
            recommendation.estimate
            if recommendation
            else self.models.estimator.estimate(settings, load_kg or self.capacity_kg / 2)
        )
        with self._lock:
            state = self._refresh_state()
            if state.status in ACTIVE_STATUSES:
                raise DeviceNotReady(f"the machine is busy ({state.status.value})")
            if state.status is MachineStatus.OFFLINE:
                raise DeviceNotReady("the machine is offline")
            self.adapter.start_cycle(settings, load_kg=load_kg)
            # Settings the machine could not take must not calibrate the estimates later.
            self._begin_cycle(
                CycleRecord(
                    started_at=self._clock(),
                    settings=settings if applied else None,
                    profile=profile,
                    estimate=estimate if applied else None,
                )
            )
            try:
                state = self._refresh_state()
            except DeviceError as exc:
                # The machine accepted the start; don't report failure (and invite a retry).
                log.warning("Cycle started, but reading the machine's state failed: %s", exc)
                state = MachineState(status=MachineStatus.RUNNING, program=settings.program)
        chose_own = profile is not None and recommendation is None
        if chose_own and settings != self.recommend(profile).settings:
            # Remember the household's preference; it personalises the next retraining.
            self.feedback.add(FeedbackRecord(profile=profile, settings=settings, source="override"))
        if applied:
            message = (
                f"{settings.program.value.replace('_', ' ')} at {settings.temperature_c} °C, "
                f"{settings.spin_rpm} rpm; about {estimate.duration_min:.0f} min."
            )
        else:
            message = "The programme set on the machine has started."
        self.notifier.notify(
            "cycle_started",
            "Cycle started",
            message,
            settings=settings.model_dump(mode="json"),
            applied=applied,
        )
        return StartCycleResult(
            settings=settings,
            applied=applied,
            recommendation=recommendation,
            adjustments=adjustments,
            warnings=warnings,
            estimate=estimate,
            state=state,
        )

    def pause(self) -> MachineState:
        if not self.adapter.capabilities.pause_resume:
            raise UnsupportedOperation(f"the {self.adapter.name} connection cannot pause")
        with self._lock:
            self.adapter.pause()
            return self._refresh_state()

    def resume(self) -> MachineState:
        if not self.adapter.capabilities.pause_resume:
            raise UnsupportedOperation(f"the {self.adapter.name} connection cannot resume")
        with self._lock:
            self.adapter.resume()
            return self._refresh_state()

    def stop(self) -> MachineState:
        if not self.adapter.capabilities.stop:
            raise UnsupportedOperation(f"the {self.adapter.name} connection cannot stop cycles")
        with self._lock:
            self.adapter.stop()
            return self._refresh_state()

    def inject_fault(self, fault: FaultType, severity: float = 1.0) -> dict[str, Any]:
        """Give the virtual machine a fault, to see the AI catch it."""
        if not isinstance(self.adapter, SimulatedWasherAdapter):
            raise UnsupportedOperation("fault injection only works with the simulator")
        with self._lock:
            self.adapter.inject_fault(fault, severity)
            return self.adapter.describe()

    # ------------------------------------------------------------- monitoring

    def poll(self) -> list[Diagnosis]:
        """One monitoring step: ingest telemetry, run fault detection, track the cycle.

        Returns the faults newly confirmed by this step.
        """
        with self._lock:
            samples = self.adapter.read_telemetry()
            confirmed: list[Diagnosis] = []
            if samples and self._cycle is not None:
                self._ingest(samples)
                if self.fault_detection:
                    confirmed = self._detect(samples)
            state = self._refresh_state()
            if state.status is not MachineStatus.RUNNING:
                self._windower.reset()
            return confirmed

    @property
    def is_monitoring(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_monitoring(self) -> None:
        if self.is_monitoring:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, name="smartwash-monitor", daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _monitor(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
                self.last_error = None
            except DeviceError as exc:
                self.last_error = str(exc)
                log.warning("Monitoring: %s", exc)
            except Exception:  # keep watching the machine whatever happens
                log.exception("Unexpected error in the monitoring loop")
            self._stop.wait(self.monitoring.interval_s)

    def diagnostics(self) -> DiagnosticsReport:
        with self._lock:
            reason = None
            if not self.fault_detection:
                missing = sorted(
                    FAULT_DETECTION_FIELDS - self.adapter.capabilities.telemetry_fields
                )
                reason = (
                    f"The {self.adapter.name} connection does not report the sensor data fault "
                    f"detection needs (missing: {', '.join(missing)}). Recommendations and remote "
                    "control still work; use the MQTT adapter with sensors for fault detection."
                )
            return DiagnosticsReport(
                fault_detection=self.fault_detection,
                unavailable_reason=reason,
                latest=self._latest,
                cycle_faults=list(self._cycle.faults) if self._cycle else [],
                recent_alerts=list(reversed(self._alerts)),
                windows_assessed=self._windows_assessed,
            )

    # --------------------------------------------------------------- learning

    def add_feedback(self, request: FeedbackRequest) -> int:
        self.feedback.add(
            FeedbackRecord(
                profile=request.profile, settings=request.settings, rating=request.rating
            )
        )
        return len(self.feedback)

    def retrain(self) -> dict[str, Any]:
        """Retrain the recommender with all feedback so far, then save the models."""
        if not self._retrain_lock.acquire(blocking=False):
            raise RetrainInProgress("retraining is already in progress")
        try:
            metrics = self.models.retrain_recommender(self.feedback.load())
            with self._lock:  # a finishing cycle may be updating the calibration
                self.models.save(self.model_dir)
            return metrics
        finally:
            self._retrain_lock.release()

    def model_info(self) -> dict[str, Any]:
        return {
            "recommender_version": self.models.recommender.version,
            "feedback_records": len(self.feedback),
            "calibration": self.models.estimator.summary(),
            "metadata": self.models.metadata,
        }

    def close(self) -> None:
        self.stop_monitoring()
        self.adapter.close()
        self.notifier.close()

    # ---------------------------------------------------------------- internals

    def _refresh_state(self) -> MachineState:
        state = self.adapter.get_state()
        self._track(state)
        return state

    def _track(self, state: MachineState) -> None:
        """Follow cycle start/end transitions, wherever the cycle was started from."""
        now = self._clock()
        previous = self._last_state
        if (
            self._cycle is not None
            and previous is not None
            and previous.status is MachineStatus.RUNNING
            and self._last_seen is not None
        ):
            self._running_s += now - self._last_seen
        self._last_seen = now

        if state.status in ACTIVE_STATUSES and self._cycle is None:
            settings = _settings_from_state(state)
            self._begin_cycle(
                CycleRecord(
                    started_at=now,
                    source="machine",
                    settings=settings,
                    vendor_program=state.vendor_program,
                )
            )
            self.notifier.notify(
                "cycle_started",
                "Cycle started on the machine",
                f"Programme {state.vendor_program or 'unknown'} started from the machine itself.",
            )
        if self._cycle is not None:
            self._cycle.vendor_program = self._cycle.vendor_program or state.vendor_program
            if state.status in ACTIVE_STATUSES:
                self._seen_active = True
            elif state.status is MachineStatus.ERROR:
                self._end_cycle("error", state)
            elif state.status is not MachineStatus.OFFLINE:
                if self._seen_active:
                    finished = state.status is MachineStatus.FINISHED or self._was_nearly_done(
                        previous
                    )
                    self._end_cycle("finished" if finished else "stopped", state)
                elif now - self._cycle.started_at >= START_GRACE_S:
                    log.warning("The machine never reported the cycle as running.")
                    self._end_cycle("stopped", state)
        if state.status is MachineStatus.ERROR and (
            previous is None or previous.status is not MachineStatus.ERROR
        ):
            self.notifier.notify(
                "machine_error",
                "Machine error",
                f"The machine reports error {state.error_code or '(no code)'}.",
                error_code=state.error_code,
            )
        self._last_state = state

    @staticmethod
    def _was_nearly_done(previous: MachineState | None) -> bool:
        if previous is None:
            return False
        if previous.progress_pct is not None and previous.progress_pct >= 95:
            return True
        return previous.remaining_min is not None and previous.remaining_min <= 2

    def _begin_cycle(self, record: CycleRecord) -> None:
        self._cycle = record
        self._running_s = self._telemetry_s = self._energy_j = 0.0
        self._has_power = False
        self._seen_active = False
        self._last_sample = None
        self._windower.reset()
        self._debouncer.reset()

    def _end_cycle(self, outcome: str, state: MachineState) -> None:
        cycle = self._cycle
        assert cycle is not None
        self._cycle = None
        self._windower.reset()
        cycle.ended_at = self._clock()
        cycle.outcome = outcome
        cycle.error_code = state.error_code
        # Telemetry timestamps are machine time, so prefer them (they are also right for the
        # accelerated simulator); fall back to the wall-clock time spent running.
        duration_s = self._telemetry_s if self._telemetry_s >= 60 else self._running_s
        cycle.duration_min = round(duration_s / 60.0, 1) if duration_s > 0 else None
        if self._has_power and self._energy_j > 0:
            cycle.energy_kwh = round(self._energy_j / 3.6e6, 3)
        if outcome == "finished" and cycle.settings is not None and cycle.duration_min:
            load_kg = cycle.profile.weight_kg if cycle.profile else self.capacity_kg / 2
            self.models.estimator.observe(
                cycle.settings,
                load_kg,
                duration_min=cycle.duration_min,
                energy_kwh=cycle.energy_kwh,
            )
            try:
                self.models.save_calibration(self.model_dir)
            except OSError as exc:
                log.warning("Could not save the cycle calibration: %s", exc)
        self.history.append(cycle)
        if outcome == "finished":
            energy = f" and used {cycle.energy_kwh:.2f} kWh" if cycle.energy_kwh else ""
            took = f" It took {cycle.duration_min:.0f} min{energy}." if cycle.duration_min else ""
            self.notifier.notify(
                "cycle_finished",
                "Laundry is done",
                f"The cycle has finished.{took}",
                cycle=cycle.model_dump(mode="json"),
            )
        elif outcome == "stopped":
            self.notifier.notify(
                "cycle_stopped", "Cycle stopped", "The cycle was stopped before it finished."
            )

    def _ingest(self, samples: list[TelemetrySample]) -> None:
        for sample in samples:
            previous = self._last_sample
            if previous is not None:
                dt = sample.timestamp - previous.timestamp
                if 0 < dt <= MAX_SAMPLE_GAP_S:
                    self._telemetry_s += dt
                    if sample.power_w is not None:
                        self._energy_j += sample.power_w * dt
                        self._has_power = True
            self._last_sample = sample

    def _detect(self, samples: list[TelemetrySample]) -> list[Diagnosis]:
        confirmed = []
        for sample in samples:
            for window in self._windower.push(sample):
                diagnosis = self.models.detector.assess(window)
                if diagnosis is None:
                    continue
                self._windows_assessed += 1
                self._latest = diagnosis
                if self._debouncer.update(diagnosis) is not None:
                    confirmed.append(diagnosis)
                    self._on_fault(diagnosis)
        return confirmed

    def _on_fault(self, diagnosis: Diagnosis) -> None:
        if self._cycle is not None:
            self._cycle.faults.append(diagnosis)
        self._alerts.append(diagnosis)
        name = diagnosis.fault.value.replace("_", " ")
        self.notifier.notify(
            "fault_detected",
            f"Possible fault: {name}",
            f"{diagnosis.message} {diagnosis.recommended_action}",
            diagnosis=diagnosis.model_dump(mode="json"),
        )
        if (
            diagnosis.severity is Severity.CRITICAL
            and self.monitoring.auto_pause_on_critical
            and self.adapter.capabilities.pause_resume
        ):
            try:
                self.adapter.pause()
            except DeviceError as exc:
                log.warning("Could not pause the cycle after detecting %s: %s", name, exc)
            else:
                self.notifier.notify(
                    "auto_paused",
                    "Cycle paused for safety",
                    f"SmartWash paused the cycle because of a suspected {name}. "
                    f"{diagnosis.recommended_action}",
                )


def build_controller(
    settings: Settings,
    *,
    adapter: WashingMachineAdapter | None = None,
    models: ModelBundle | None = None,
) -> WasherController:
    """Load (or train) the models, connect to the configured machine and wire everything up."""
    feedback = FeedbackStore(settings.data_dir / "feedback.jsonl")
    if models is None:
        models = ModelBundle.load_or_train(
            settings.models.dir,
            auto_train=settings.models.auto_train,
            feedback=feedback.load(),
            recommender_samples=settings.models.recommender_samples,
            fault_cycles=settings.models.fault_cycles,
        )
    adapter = adapter or create_adapter(settings.machine)
    adapter.connect()
    notifications = settings.notifications
    return WasherController(
        adapter,
        models,
        model_dir=settings.models.dir,
        feedback=feedback,
        history=CycleHistory(settings.data_dir / "cycles.jsonl"),
        notifier=Notifier(
            notifications.webhook_url,
            events=notifications.events,
            timeout_s=notifications.timeout_s,
        ),
        capacity_kg=settings.machine.capacity_kg,
        monitoring=settings.monitoring,
    )
