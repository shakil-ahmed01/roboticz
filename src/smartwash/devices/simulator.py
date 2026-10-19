"""Physics-based washing-machine simulator.

Used three ways: as a virtual machine (``SimulatedWasherAdapter``) so the whole system runs without
hardware, as the device behind the MQTT virtual device, and as the source of labelled telemetry for
training the fault detector.
"""

from __future__ import annotations

import dataclasses
import math
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple

from .. import physics
from ..physics import PlanStep
from ..schemas import (
    ACTIVE_STATUSES,
    FAULT_DETECTION_FIELDS,
    CyclePhase,
    CycleSettings,
    FaultType,
    MachineState,
    MachineStatus,
    TelemetrySample,
)
from .base import AdapterCapabilities, DeviceNotReady, WashingMachineAdapter


class SimRecord(NamedTuple):
    sample: TelemetrySample
    label: FaultType
    """Ground truth: the fault visible in this sample (``FaultType.NONE`` when healthy)."""


@dataclasses.dataclass(frozen=True)
class _Effects:
    """Hardware parameters for the current cycle once the injected fault is applied."""

    fill_rate_lpm: float
    drain_rate_lpm: float
    heater_w: float
    imbalance: float
    imbalance_cap_rpm: float | None
    wear: float
    overload: float
    rpm_factor: float


class WasherSimulator:
    """Time-stepped model of a washing machine with optional fault injection.

    Each ``step()`` advances ``sample_period_s`` seconds of machine time and, while a cycle runs,
    returns one telemetry sample with its ground-truth fault label. Not thread-safe;
    ``SimulatedWasherAdapter`` adds locking and wall-clock pacing.
    """

    FILL_TIMEOUT_S = 20 * 60.0
    DRAIN_TIMEOUT_S = 10 * 60.0
    HEAT_TIMEOUT_S = 45 * 60.0
    MOTOR_TAU_S = 4.0

    def __init__(
        self,
        *,
        sample_period_s: float = 2.0,
        seed: int | None = None,
        inlet_temp_c: float = physics.INLET_TEMP_C,
        ambient_c: float = physics.AMBIENT_TEMP_C,
        epoch: float = 0.0,
    ) -> None:
        if sample_period_s <= 0:
            raise ValueError("sample_period_s must be positive")
        self.dt = float(sample_period_s)
        self.rng = random.Random(seed)
        self.inlet_temp_c = inlet_temp_c
        self.ambient_c = ambient_c
        self.epoch = epoch
        self.t = 0.0
        self.status = MachineStatus.IDLE
        self.phase = CyclePhase.IDLE
        self.settings: CycleSettings | None = None
        self.load_kg = 0.0
        self.error_code: str | None = None
        self.fault = FaultType.NONE
        self.severity = 1.0
        self.water_l = 0.0
        self.water_temp_c = ambient_c
        self.rpm = 0.0
        self._plan: list[PlanStep] = []
        self._index = 0
        self._step_elapsed = 0.0
        self._hold_elapsed = 0.0
        self._heated = False
        self._thermostat_on = False
        self._cycle_elapsed = 0.0
        self._fill_rate = physics.FILL_RATE_LPM
        self._drain_rate = physics.DRAIN_RATE_LPM
        self._heater_w = physics.HEATER_POWER_W
        self._imbalance = 0.2
        self._effects = self._compute_effects()

    # ------------------------------------------------------------------ control

    def start(
        self,
        settings: CycleSettings,
        load_kg: float,
        *,
        fault: FaultType = FaultType.NONE,
        severity: float = 1.0,
        duration_scale: float = 1.0,
    ) -> None:
        if self.status in ACTIVE_STATUSES:
            raise RuntimeError("a cycle is already running")
        if self.status is MachineStatus.ERROR:
            raise RuntimeError(f"machine is in error {self.error_code}; stop it to reset")
        self.settings = settings
        self.load_kg = float(load_kg)
        self._plan = physics.build_cycle_plan(settings, self.load_kg, duration_scale)
        self._index = 0
        self._reset_step()
        self._cycle_elapsed = 0.0
        self.water_l = 0.0
        self.error_code = None
        # Every machine and every load is a little different.
        self._fill_rate = self.rng.uniform(8.5, 11.5)
        self._drain_rate = self.rng.uniform(15.0, 20.0)
        self._heater_w = self.rng.uniform(1900.0, 2100.0)
        self._imbalance = self.rng.uniform(0.05, 0.4)
        self.set_fault(fault, severity)
        self.status = MachineStatus.RUNNING
        self.phase = self._plan[0].phase

    def set_fault(self, fault: FaultType | str, severity: float = 1.0) -> None:
        """Inject (or clear, with ``FaultType.NONE``) a fault; takes effect immediately."""
        self.fault = FaultType(fault)
        self.severity = float(severity)
        self._effects = self._compute_effects()

    def pause(self) -> None:
        if self.status is not MachineStatus.RUNNING:
            raise RuntimeError("no running cycle to pause")
        self.status = MachineStatus.PAUSED

    def resume(self) -> None:
        if self.status is not MachineStatus.PAUSED:
            raise RuntimeError("the cycle is not paused")
        self.status = MachineStatus.RUNNING

    def stop(self) -> None:
        """Abort the cycle (or clear an error) and drain the drum."""
        self.status = MachineStatus.IDLE
        self.phase = CyclePhase.IDLE
        self.settings = None
        self.error_code = None
        self._plan = []
        self.water_l = 0.0
        self.rpm = 0.0

    # ----------------------------------------------------------------- progress

    @property
    def remaining_s(self) -> float | None:
        if not self._plan or self.status not in ACTIVE_STATUSES:
            return None
        current = self._plan[self._index]
        if self._heated:
            current = dataclasses.replace(current, target_temp_c=None)
        steps = [current, *self._plan[self._index + 1 :]]
        estimates = physics.estimate_steps(
            steps,
            self.load_kg,
            inlet_temp_c=self.inlet_temp_c,
            ambient_c=self.ambient_c,
            level_l=self.water_l,
            temp_c=self.water_temp_c,
        )
        remaining = sum(e.duration_s for e in estimates)
        if current.kind == "tumble":
            remaining -= self._hold_elapsed
        elif current.kind == "spin":
            remaining -= self._step_elapsed
        return max(remaining, 0.0)

    @property
    def progress_pct(self) -> float | None:
        if self.status is MachineStatus.FINISHED:
            return 100.0
        remaining = self.remaining_s
        if remaining is None:
            return None
        total = self._cycle_elapsed + remaining
        return round(100.0 * self._cycle_elapsed / total, 1) if total > 0 else 0.0

    def state(self) -> MachineState:
        settings = self.settings
        remaining = self.remaining_s
        return MachineState(
            status=self.status,
            phase=self.phase,
            program=settings.program if settings else None,
            vendor_program=settings.program.value if settings else None,
            temperature_c=settings.temperature_c if settings else None,
            spin_rpm=settings.spin_rpm if settings else None,
            progress_pct=self.progress_pct,
            remaining_min=round(remaining / 60.0, 1) if remaining is not None else None,
            door_open=False,
            remote_start_allowed=True,
            error_code=self.error_code,
            raw={"fault": self.fault.value, "water_l": round(self.water_l, 2)},
        )

    # --------------------------------------------------------------- simulation

    def run(self, max_steps: int = 500_000) -> list[SimRecord]:
        """Step until the running cycle finishes, fails or is paused; return its telemetry."""
        records: list[SimRecord] = []
        for _ in range(max_steps):
            if self.status is not MachineStatus.RUNNING:
                break
            record = self.step()
            if record is not None:
                records.append(record)
        return records

    def advance_idle(self, seconds: float) -> None:
        """Fast-forward while no cycle is running (no telemetry is produced)."""
        if self.status is MachineStatus.RUNNING:
            raise RuntimeError("use step() while a cycle is running")
        self.t += seconds
        self.rpm = 0.0
        decay = math.exp(-physics.HEAT_LOSS_PER_S * seconds)
        self.water_temp_c = self.ambient_c + (self.water_temp_c - self.ambient_c) * decay

    def step(self) -> SimRecord | None:
        """Advance one sample period; return a telemetry record if a cycle is running."""
        dt = self.dt
        if self.status is not MachineStatus.RUNNING:
            self.advance_idle(dt)
            return None
        self.t += dt
        self._cycle_elapsed += dt
        self._step_elapsed += dt
        step = self._plan[self._index]
        effects = self._effects
        heater_on = pump_on = False
        label = FaultType.NONE
        setpoint = 0.0
        target_temp: float | None = None
        speed_cap: float | None = None

        if step.kind == "fill":
            setpoint = step.rpm
            added = effects.fill_rate_lpm * dt / 60.0
            level = self.water_l + added
            self.water_temp_c = (
                self.water_temp_c * self.water_l + self.inlet_temp_c * added
            ) / level
            self.water_l = level
            if self.fault is FaultType.WATER_SUPPLY:
                label = FaultType.WATER_SUPPLY
            if self.water_l >= step.water_l:
                self._advance()
            elif self._step_elapsed > self.FILL_TIMEOUT_S:
                self._fail("E_FILL")
        elif step.kind == "tumble":
            setpoint = step.rpm if self._tumbling(step) else 0.0
            target = step.target_temp_c
            if target is not None and not self._heated:
                target_temp = target
                if self.water_temp_c >= target - 0.5 or self._step_elapsed > self.HEAT_TIMEOUT_S:
                    self._heated = True
                else:
                    heater_on = True
            if self._heated or target is None:
                if target is not None and step.hold_heat:
                    target_temp = target
                    if self.water_temp_c < target - 1.5:
                        self._thermostat_on = True
                    elif self.water_temp_c >= target:
                        self._thermostat_on = False
                    heater_on = self._thermostat_on
                self._hold_elapsed += dt
                if self._hold_elapsed >= step.duration_s:
                    self._advance()
            if heater_on and self.fault is FaultType.HEATER_FAILURE:
                label = FaultType.HEATER_FAILURE
        elif step.kind == "drain":
            pump_on = True
            drained = effects.drain_rate_lpm * dt / 60.0
            self.water_l = max(physics.RESIDUAL_WATER_L, self.water_l - drained)
            if self.fault is FaultType.DRAIN_BLOCKAGE:
                label = FaultType.DRAIN_BLOCKAGE
            if self.water_l <= physics.RESIDUAL_WATER_L + 0.01:
                self._advance()
            elif self._step_elapsed > self.DRAIN_TIMEOUT_S:
                self._fail("E_DRAIN")
        else:  # spin
            pump_on = True
            setpoint = min(step.rpm, physics.SPIN_RAMP_RPM_PER_S * self._step_elapsed)
            speed_cap = effects.imbalance_cap_rpm
            if speed_cap is not None and setpoint > speed_cap:
                label = FaultType.UNBALANCED_LOAD
            if setpoint >= step.rpm:
                self._hold_elapsed += dt
                if self._hold_elapsed >= step.duration_s:
                    self._advance()

        driven = step.kind != "drain"
        if driven and self.fault in (FaultType.BEARING_WEAR, FaultType.MOTOR_OVERLOAD):
            label = self.fault

        # Drum motor: first-order lag towards the (possibly limited) speed.
        drive = setpoint if speed_cap is None else min(setpoint, speed_cap)
        previous_rpm = self.rpm
        self.rpm += (drive * effects.rpm_factor - self.rpm) * (
            1.0 - math.exp(-dt / self.MOTOR_TAU_S)
        )
        current = physics.motor_current_a(
            self.rpm,
            self.load_kg,
            accel_rpm_s=(self.rpm - previous_rpm) / dt,
            wear=effects.wear,
            overload=effects.overload,
        )

        heater_w = 0.0
        if heater_on and self.water_l >= physics.MIN_HEATING_WATER_L:
            heater_w = effects.heater_w
            mass = physics.thermal_mass_j_per_k(self.water_l, self.load_kg)
            self.water_temp_c += heater_w * dt / mass
        self.water_temp_c -= (self.water_temp_c - self.ambient_c) * physics.HEAT_LOSS_PER_S * dt

        power = (
            physics.STANDBY_POWER_W
            + heater_w
            + physics.motor_power_w(current)
            + (physics.PUMP_POWER_W if pump_on else 0.0)
        )
        gauss = self.rng.gauss
        sample = TelemetrySample.model_construct(
            timestamp=self.epoch + self.t,
            phase=step.phase,
            drum_rpm=max(0.0, self.rpm + gauss(0.0, 0.8 + 0.008 * self.rpm)),
            target_rpm=setpoint,
            motor_current_a=max(0.0, current + gauss(0.0, 0.04)),
            vibration_g=self._vibration(effects),
            water_temp_c=self.water_temp_c + gauss(0.0, 0.15),
            target_temp_c=target_temp,
            water_level_l=max(0.0, self.water_l + gauss(0.0, 0.15)),
            power_w=max(0.0, power + gauss(0.0, 8.0)),
        )
        return SimRecord(sample, label)

    # ------------------------------------------------------------------ helpers

    def _tumbling(self, step: PlanStep) -> bool:
        return (self._step_elapsed % physics.TUMBLE_PERIOD_S) < step.duty * physics.TUMBLE_PERIOD_S

    def _vibration(self, effects: _Effects) -> float:
        if self.rpm < 3.0:
            return abs(0.005 + self.rng.gauss(0.0, 0.001))
        krpm = self.rpm / 1000.0
        level = (
            0.012
            + 0.0005 * min(self.rpm, 60.0)
            + 0.3 * effects.imbalance * krpm**2
            + effects.wear * (0.03 + 0.18 * krpm)
        )
        return abs(level * (1.0 + self.rng.gauss(0.0, 0.06 + 0.25 * effects.wear)))

    def _compute_effects(self) -> _Effects:
        fault, s = self.fault, self.severity
        imbalance = 1.0 + 1.5 * s if fault is FaultType.UNBALANCED_LOAD else self._imbalance
        cap = min(900.0, max(300.0, 1500.0 / imbalance - 150.0)) if imbalance > 1.0 else None
        return _Effects(
            fill_rate_lpm=self._fill_rate
            * (max(0.1, 0.45 - 0.25 * s) if fault is FaultType.WATER_SUPPLY else 1.0),
            drain_rate_lpm=self._drain_rate
            * (max(0.03, 0.25 - 0.15 * s) if fault is FaultType.DRAIN_BLOCKAGE else 1.0),
            heater_w=self._heater_w
            * (max(0.0, 1.0 - 0.9 * s) if fault is FaultType.HEATER_FAILURE else 1.0),
            imbalance=imbalance,
            imbalance_cap_rpm=cap,
            wear=s if fault is FaultType.BEARING_WEAR else 0.0,
            overload=1.0 + 0.9 * s if fault is FaultType.MOTOR_OVERLOAD else 1.0,
            rpm_factor=1.0 - (0.02 + 0.06 * s) if fault is FaultType.MOTOR_OVERLOAD else 1.0,
        )

    def _reset_step(self) -> None:
        self._step_elapsed = 0.0
        self._hold_elapsed = 0.0
        self._heated = False
        self._thermostat_on = False

    def _advance(self) -> None:
        self._index += 1
        self._reset_step()
        if self._index >= len(self._plan):
            self.status = MachineStatus.FINISHED
            self.phase = CyclePhase.DONE
            self.water_l = 0.0
        else:
            self.phase = self._plan[self._index].phase

    def _fail(self, code: str) -> None:
        self.status = MachineStatus.ERROR
        self.error_code = code


class SimulatedWasherAdapter(WashingMachineAdapter):
    """A virtual washing machine that runs ``time_scale`` times faster than real time."""

    name = "simulator"

    def __init__(
        self,
        *,
        time_scale: float = 60.0,
        sample_period_s: float = 2.0,
        fault: FaultType | str = FaultType.NONE,
        fault_severity: float = 1.0,
        default_load_kg: float = 4.0,
        seed: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_buffered_samples: int = 20_000,
    ) -> None:
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.time_scale = time_scale
        self._clock = clock
        self._sim = WasherSimulator(sample_period_s=sample_period_s, seed=seed, epoch=time.time())
        self._fault = FaultType(fault)
        self._severity = fault_severity
        self._default_load_kg = default_load_kg
        self._buffer: deque[TelemetrySample] = deque(maxlen=max_buffered_samples)
        self._lock = threading.RLock()
        self._last = clock()
        self._carry_s = 0.0

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(telemetry_fields=FAULT_DETECTION_FIELDS | {"target_temp_c"})

    def get_state(self) -> MachineState:
        with self._lock:
            self._advance()
            return self._sim.state()

    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        with self._lock:
            self._advance()
            if self._sim.status in ACTIVE_STATUSES:
                raise DeviceNotReady("a cycle is already running")
            if self._sim.status is MachineStatus.ERROR:
                raise DeviceNotReady(
                    f"machine reports error {self._sim.error_code}; stop it to reset"
                )
            self._buffer.clear()
            self._sim.start(
                settings,
                load_kg if load_kg is not None else self._default_load_kg,
                fault=self._fault,
                severity=self._severity,
            )

    def pause(self) -> None:
        with self._lock:
            self._advance()
            if self._sim.status is not MachineStatus.RUNNING:
                raise DeviceNotReady("no running cycle to pause")
            self._sim.pause()

    def resume(self) -> None:
        with self._lock:
            self._advance()
            if self._sim.status is not MachineStatus.PAUSED:
                raise DeviceNotReady("the cycle is not paused")
            self._sim.resume()

    def stop(self) -> None:
        with self._lock:
            self._advance()
            self._sim.stop()

    def read_telemetry(self) -> list[TelemetrySample]:
        with self._lock:
            self._advance()
            samples = list(self._buffer)
            self._buffer.clear()
            return samples

    def inject_fault(self, fault: FaultType | str, severity: float = 1.0) -> None:
        """Give the virtual machine a fault (``none`` repairs it), effective immediately."""
        with self._lock:
            self._advance()
            self._fault = FaultType(fault)
            self._severity = severity
            self._sim.set_fault(self._fault, severity)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "model": "SmartWash virtual washer",
            "time_scale": self.time_scale,
            "fault": self._fault.value,
            "fault_severity": self._severity,
        }

    def _advance(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last) * self.time_scale + self._carry_s
        self._last = now
        dt = self._sim.dt
        remaining = int(elapsed // dt)
        self._carry_s = elapsed - remaining * dt
        while remaining > 0:
            if self._sim.status is not MachineStatus.RUNNING:
                self._sim.advance_idle(dt * remaining)
                break
            record = self._sim.step()
            if record is not None:
                self._buffer.append(record.sample)
            remaining -= 1
