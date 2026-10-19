from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from smartwash.ai.feedback import FeedbackStore
from smartwash.ai.registry import CALIBRATION_FILE
from smartwash.config import MonitoringConfig
from smartwash.controller import (
    CycleHistory,
    FeedbackRequest,
    StartCycleRequest,
    WasherController,
)
from smartwash.devices import (
    AdapterCapabilities,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
)
from smartwash.devices.base import WashingMachineAdapter
from smartwash.devices.simulator import SimulatedWasherAdapter
from smartwash.notifications import Notifier
from smartwash.schemas import (
    CycleSettings,
    FaultType,
    LoadProfile,
    MachineState,
    MachineStatus,
    Program,
)

COTTON = LoadProfile(fabric="cotton", color="colors", weight_kg=4)


def make_controller(models, tmp_path, clock, *, adapter=None, **monitoring):
    adapter = adapter or SimulatedWasherAdapter(time_scale=60, clock=clock, seed=11)
    return WasherController(
        adapter,
        models,
        model_dir=tmp_path / "models",
        feedback=FeedbackStore(tmp_path / "feedback.jsonl"),
        history=CycleHistory(tmp_path / "cycles.jsonl"),
        notifier=Notifier(),
        monitoring=MonitoringConfig(**monitoring),
        clock=clock,
    )


def run_until(controller, clock, predicate, *, step_s=5.0, max_steps=400):
    """Advance time (60x faster on the virtual machine) and poll until ``predicate()``."""
    confirmed = []
    for _ in range(max_steps):
        clock.now += step_s
        confirmed += controller.poll()
        if predicate():
            return confirmed
    raise AssertionError("condition not reached")


def events(controller) -> list[str]:
    return [e["event"] for e in controller.notifier.recent]


def test_start_from_a_load_uses_the_ai_recommendation(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    result = controller.start_cycle(StartCycleRequest(profile=COTTON))
    assert result.recommendation is not None
    assert result.settings == result.recommendation.settings
    assert result.state.status is MachineStatus.RUNNING
    assert result.state.program is result.settings.program
    assert events(controller) == ["cycle_started"]
    with pytest.raises(DeviceNotReady):
        controller.start_cycle(StartCycleRequest(profile=COTTON))


def test_own_settings_are_made_safe_and_remembered(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    wool = LoadProfile(fabric="wool", weight_kg=1.5)
    hot = CycleSettings(program="cotton", temperature_c=60, spin_rpm=1400)
    result = controller.start_cycle(StartCycleRequest(profile=wool, settings=hot))
    assert (result.settings.program, result.settings.temperature_c) == (Program.WOOL, 30)
    assert result.adjustments
    assert result.state.program is Program.WOOL
    [record] = controller.feedback.load()
    assert record.source == "override"
    assert record.settings == result.settings


def test_healthy_cycle_is_recorded_and_calibrates_the_estimates(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    start = controller.start_cycle(StartCycleRequest(profile=COTTON))
    confirmed = run_until(
        controller, clock, lambda: controller.snapshot().state.status is MachineStatus.FINISHED
    )
    assert confirmed == []
    [cycle] = controller.history.recent()
    assert cycle.outcome == "finished"
    assert cycle.source == "smartwash"
    assert cycle.faults == []
    assert cycle.duration_min == pytest.approx(start.estimate.duration_min, rel=0.05)
    assert cycle.energy_kwh == pytest.approx(start.estimate.energy_kwh, rel=0.15)
    assert models.estimator.observations() == 1
    assert (tmp_path / "models" / CALIBRATION_FILE).exists()
    assert events(controller)[-1] == "cycle_finished"
    assert controller.diagnostics().windows_assessed > 50


def test_critical_fault_is_detected_and_the_cycle_paused(models, tmp_path, clock):
    adapter = SimulatedWasherAdapter(
        time_scale=60, clock=clock, seed=4, fault=FaultType.DRAIN_BLOCKAGE, fault_severity=1.0
    )
    controller = make_controller(models, tmp_path, clock, adapter=adapter)
    controller.start_cycle(StartCycleRequest(profile=COTTON))
    confirmed = run_until(controller, clock, lambda: bool(controller.diagnostics().cycle_faults))
    assert [d.fault for d in confirmed] == [FaultType.DRAIN_BLOCKAGE]
    assert controller.snapshot().state.status is MachineStatus.PAUSED
    assert events(controller)[-2:] == ["fault_detected", "auto_paused"]
    report = controller.diagnostics()
    assert report.fault_detection
    assert report.recent_alerts[0].fault is FaultType.DRAIN_BLOCKAGE
    controller.stop()
    [cycle] = controller.history.recent()
    assert cycle.outcome == "stopped"
    assert [f.fault for f in cycle.faults] == [FaultType.DRAIN_BLOCKAGE]
    assert models.estimator.observations() == 0  # aborted cycles do not calibrate


def test_auto_pause_can_be_turned_off(models, tmp_path, clock):
    adapter = SimulatedWasherAdapter(
        time_scale=60, clock=clock, seed=4, fault=FaultType.DRAIN_BLOCKAGE, fault_severity=1.0
    )
    controller = make_controller(
        models, tmp_path, clock, adapter=adapter, auto_pause_on_critical=False
    )
    controller.start_cycle(StartCycleRequest(profile=COTTON))
    run_until(controller, clock, lambda: bool(controller.diagnostics().cycle_faults))
    assert controller.snapshot().state.status is MachineStatus.RUNNING
    assert "auto_paused" not in events(controller)


def test_cycles_started_on_the_machine_are_followed(models, tmp_path, clock):
    adapter = SimulatedWasherAdapter(time_scale=60, clock=clock, seed=2)
    controller = make_controller(models, tmp_path, clock, adapter=adapter)
    adapter.start_cycle(CycleSettings(program="quick", temperature_c=30, spin_rpm=1000))
    controller.poll()
    assert controller.snapshot().cycle.source == "machine"
    run_until(controller, clock, lambda: bool(controller.history.recent()))
    [cycle] = controller.history.recent()
    assert cycle.outcome == "finished"
    assert cycle.settings.program is Program.QUICK


def test_machine_errors_are_reported(models, tmp_path, clock):
    adapter = SimulatedWasherAdapter(
        time_scale=60, clock=clock, seed=4, fault=FaultType.DRAIN_BLOCKAGE, fault_severity=1.4
    )
    controller = make_controller(
        models, tmp_path, clock, adapter=adapter, auto_pause_on_critical=False
    )
    controller.start_cycle(StartCycleRequest(profile=COTTON))
    run_until(controller, clock, lambda: bool(controller.history.recent()))
    [cycle] = controller.history.recent()
    assert (cycle.outcome, cycle.error_code) == ("error", "E_DRAIN")
    assert "machine_error" in events(controller)


class CloudLikeAdapter(WashingMachineAdapter):
    """Remote control but no sensor telemetry, like the vendor cloud APIs."""

    name = "cloud"

    def __init__(self) -> None:
        self.status = MachineStatus.IDLE
        self.started: list[CycleSettings] = []

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(pause_resume=False)

    def get_state(self) -> MachineState:
        return MachineState(status=self.status)

    def start_cycle(self, settings, *, load_kg=None) -> None:
        self.started.append(settings)
        self.status = MachineStatus.RUNNING

    def pause(self) -> None:
        raise AssertionError("not supported")

    resume = pause

    def stop(self) -> None:
        self.status = MachineStatus.IDLE


class LaggingAdapter(CloudLikeAdapter):
    """Keeps reporting the previous state for a few reads after a start, like a slow cloud API."""

    def __init__(self, lag_reads: int = 3, starts: bool = True) -> None:
        super().__init__()
        self.status = self.reported = MachineStatus.FINISHED  # the previous cycle's state
        self.lag_reads = lag_reads
        self.starts = starts
        self._lag = 0

    def get_state(self) -> MachineState:
        if self._lag > 0:
            self._lag -= 1
        else:
            self.reported = self.status
        return MachineState(status=self.reported)

    def start_cycle(self, settings, *, load_kg=None) -> None:
        self.started.append(settings)
        if self.starts:
            self.status = MachineStatus.RUNNING
        self._lag = self.lag_reads


def test_stale_state_after_a_start_command_does_not_end_the_cycle(models, tmp_path, clock):
    adapter = LaggingAdapter(lag_reads=3)
    controller = make_controller(models, tmp_path, clock, adapter=adapter)
    controller.start_cycle(StartCycleRequest(profile=COTTON))
    for _ in range(4):
        clock.now += 2
        controller.poll()
    assert controller.history.recent() == []
    assert "cycle_stopped" not in events(controller)
    assert controller.snapshot().cycle.source == "smartwash"
    clock.now += 3600
    controller.poll()
    adapter.status = MachineStatus.FINISHED
    controller.poll()
    [cycle] = controller.history.recent()
    assert (cycle.outcome, cycle.source, cycle.profile) == ("finished", "smartwash", COTTON)
    assert events(controller).count("cycle_started") == 1


def test_a_cycle_the_machine_never_starts_is_closed_after_the_grace_period(models, tmp_path, clock):
    adapter = LaggingAdapter(lag_reads=0, starts=False)
    controller = make_controller(models, tmp_path, clock, adapter=adapter)
    controller.start_cycle(StartCycleRequest(profile=COTTON))
    clock.now += 60
    controller.poll()
    assert controller.history.recent() == []
    clock.now += 61
    controller.poll()
    [cycle] = controller.history.recent()
    assert cycle.outcome == "stopped"
    assert cycle.duration_min is None


def test_without_telemetry_control_works_but_fault_detection_explains_why_not(
    models, tmp_path, clock
):
    adapter = CloudLikeAdapter()
    controller = make_controller(models, tmp_path, clock, adapter=adapter)
    controller.start_cycle(StartCycleRequest(profile=COTTON))
    assert len(adapter.started) == 1
    with pytest.raises(UnsupportedOperation):
        controller.pause()
    report = controller.diagnostics()
    assert not report.fault_detection
    assert "vibration_g" in report.unavailable_reason
    clock.now += 3600  # an hour of wall-clock running time
    controller.poll()
    adapter.status = MachineStatus.FINISHED
    controller.poll()
    [cycle] = controller.history.recent()
    assert cycle.duration_min == pytest.approx(60, abs=0.1)
    assert cycle.energy_kwh is None


def test_feedback_and_retraining(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    before = models.recommender
    count = controller.add_feedback(
        FeedbackRequest(
            profile=COTTON,
            settings=CycleSettings(program="cotton", temperature_c=30, spin_rpm=1000),
            rating=5,
        )
    )
    assert count == 1
    metrics = controller.retrain()
    assert metrics["feedback_examples"] == 1
    assert models.recommender is not before
    assert (tmp_path / "models" / "recommender.joblib").exists()
    info = controller.model_info()
    assert info["feedback_records"] == 1
    assert info["recommender_version"] == models.recommender.version


def test_background_monitoring_thread(models, tmp_path):
    controller = make_controller(models, tmp_path, time.time, interval_s=0.05)
    controller.start_monitoring()
    assert controller.is_monitoring
    controller.start_monitoring()  # idempotent
    controller.close()
    assert not controller.is_monitoring


def test_fault_injection_needs_the_simulator(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    assert controller.inject_fault(FaultType.BEARING_WEAR, 0.8)["fault"] == "bearing_wear"
    cloud = make_controller(models, tmp_path, clock, adapter=CloudLikeAdapter())
    with pytest.raises(UnsupportedOperation):
        cloud.inject_fault(FaultType.BEARING_WEAR)


class FlakyAfterStartAdapter(CloudLikeAdapter):
    def get_state(self) -> MachineState:
        if self.started and not getattr(self, "_failed_once", False):
            self._failed_once = True
            raise DeviceError("cloud timeout")
        return super().get_state()


def test_a_failed_state_read_after_starting_still_reports_success(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock, adapter=FlakyAfterStartAdapter())
    result = controller.start_cycle(StartCycleRequest(profile=COTTON))
    assert result.state.status is MachineStatus.RUNNING
    controller.poll()
    assert controller.snapshot().cycle.source == "smartwash"


def test_webhook_receives_the_configured_events():
    received = []
    delivered = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        delivered.set()
        return httpx.Response(200)

    notifier = Notifier(
        "https://hooks.example/laundry",
        events=["cycle_finished"],
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    notifier.notify("cycle_started", "Cycle started", "not sent")
    notifier.notify("cycle_finished", "Laundry is done", "It took 120 min.", minutes=120)
    assert delivered.wait(5)
    assert [r["event"] for r in received] == ["cycle_finished"]
    assert received[0]["text"] == "Laundry is done: It took 120 min."
    assert received[0]["data"] == {"minutes": 120}
    assert [e["event"] for e in notifier.recent] == ["cycle_started", "cycle_finished"]
    notifier.close()
