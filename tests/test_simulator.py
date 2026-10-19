from __future__ import annotations

import pytest

from smartwash import physics
from smartwash.devices import DeviceNotReady
from smartwash.devices.simulator import SimulatedWasherAdapter, WasherSimulator
from smartwash.schemas import (
    FAULT_DETECTION_FIELDS,
    CyclePhase,
    CycleSettings,
    FaultType,
    MachineStatus,
)

from .conftest import FakeClock

COTTON_40 = CycleSettings(program="cotton", temperature_c=40, spin_rpm=1200)


def run(settings=COTTON_40, load_kg=4.0, **kwargs):
    sim = WasherSimulator(seed=7)
    sim.start(settings, load_kg, **kwargs)
    return sim, sim.run()


def test_healthy_cycle_runs_through_every_phase_and_finishes():
    sim, records = run()
    phases = [r.sample.phase for r in records]
    assert phases[0] is CyclePhase.FILL
    assert {CyclePhase.WASH, CyclePhase.RINSE, CyclePhase.DRAIN, CyclePhase.SPIN} <= set(phases)
    assert phases[-1] is CyclePhase.SPIN
    assert sim.status is MachineStatus.FINISHED
    assert sim.state().progress_pct == 100.0
    assert all(r.label is FaultType.NONE for r in records)
    assert all(getattr(r.sample, f) is not None for r in records for f in FAULT_DETECTION_FIELDS)


def test_simulated_cycle_matches_the_physics_estimate():
    sim, records = run()
    estimate = physics.estimate_cycle(COTTON_40, 4.0)
    duration_min = len(records) * sim.dt / 60
    energy_kwh = sum(r.sample.power_w for r in records) * sim.dt / 3.6e6
    assert duration_min == pytest.approx(estimate.duration_min, rel=0.03)
    assert energy_kwh == pytest.approx(estimate.energy_kwh, rel=0.05)
    peak_temp = max(r.sample.water_temp_c for r in records)
    assert 38 <= peak_temp <= 42


def test_heater_failure_keeps_the_water_cold():
    _, records = run(fault=FaultType.HEATER_FAILURE, severity=1.2)
    assert max(r.sample.water_temp_c for r in records) < 25
    assert any(r.label is FaultType.HEATER_FAILURE for r in records)


def test_severe_drain_blockage_ends_in_a_drain_error():
    sim, records = run(fault=FaultType.DRAIN_BLOCKAGE, severity=1.4)
    assert sim.status is MachineStatus.ERROR
    assert sim.error_code == "E_DRAIN"
    assert records[-1].sample.phase is CyclePhase.DRAIN


def test_unbalanced_load_cannot_reach_spin_speed():
    settings = CycleSettings(program="cotton", temperature_c=40, spin_rpm=1400)
    _, records = run(settings, fault=FaultType.UNBALANCED_LOAD, severity=1.0)
    spins = [r.sample for r in records if r.sample.phase is CyclePhase.SPIN]
    assert max(s.drum_rpm for s in spins) < 800
    assert any(r.label is FaultType.UNBALANCED_LOAD for r in records)


def test_pause_resume_and_stop():
    sim = WasherSimulator(seed=1)
    sim.start(COTTON_40, 4.0)
    for _ in range(50):
        sim.step()
    sim.pause()
    assert sim.step() is None
    sim.resume()
    assert sim.step() is not None
    sim.stop()
    assert sim.status is MachineStatus.IDLE
    with pytest.raises(RuntimeError):
        sim.resume()


def test_adapter_runs_faster_than_real_time():
    clock = FakeClock()
    washer = SimulatedWasherAdapter(time_scale=60, clock=clock, seed=3)
    washer.start_cycle(COTTON_40, load_kg=4)
    clock.now += 30  # 30 s wall-clock = 30 min of machine time
    state = washer.get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.program.value == "cotton"
    assert 0 < state.progress_pct < 100
    samples = washer.read_telemetry()
    assert len(samples) == pytest.approx(30 * 60 / 2, abs=2)
    assert samples[-1].timestamp - samples[0].timestamp == pytest.approx(1800, abs=5)
    assert washer.read_telemetry() == []
    clock.now += 3 * 3600  # long after the cycle ended: no hang, cycle finished
    assert washer.get_state().status is MachineStatus.FINISHED


def test_adapter_refuses_commands_that_do_not_fit_the_state():
    washer = SimulatedWasherAdapter(clock=FakeClock())
    with pytest.raises(DeviceNotReady):
        washer.pause()
    washer.start_cycle(COTTON_40)
    with pytest.raises(DeviceNotReady):
        washer.start_cycle(COTTON_40)
    with pytest.raises(DeviceNotReady):
        washer.resume()
    washer.pause()
    assert washer.get_state().status is MachineStatus.PAUSED
    washer.resume()
    washer.stop()
    assert washer.get_state().status is MachineStatus.IDLE


def test_injected_fault_affects_the_running_cycle():
    clock = FakeClock()
    washer = SimulatedWasherAdapter(time_scale=60, clock=clock, seed=5)
    washer.start_cycle(COTTON_40, load_kg=4)
    washer.inject_fault(FaultType.DRAIN_BLOCKAGE, severity=1.4)
    clock.now += 4 * 3600
    state = washer.get_state()
    assert state.status is MachineStatus.ERROR
    assert state.error_code == "E_DRAIN"
    assert washer.describe()["fault"] == "drain_blockage"
    with pytest.raises(DeviceNotReady):
        washer.start_cycle(COTTON_40)
    washer.stop()
    washer.inject_fault(FaultType.NONE)
    washer.start_cycle(COTTON_40)
    assert washer.get_state().status is MachineStatus.RUNNING
