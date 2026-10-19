"""First-principles model of a front-loading washing machine.

The step-by-step simulator and the analytic cycle estimate share these constants and the same cycle
plan, so a simulated cycle and its estimate agree. Numbers are typical for an 8 kg EU machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import CycleEstimate, CyclePhase, CycleSettings, Program

INLET_TEMP_C = 15.0
AMBIENT_TEMP_C = 20.0
HEATER_POWER_W = 2000.0
FILL_RATE_LPM = 10.0
DRAIN_RATE_LPM = 18.0
RESIDUAL_WATER_L = 0.3
"""Water left in the sump after draining."""
WATER_J_PER_L_K = 4186.0
LAUNDRY_J_PER_KG_K = 1300.0
HEAT_LOSS_PER_S = 2.0e-4
"""Fraction of the water-to-ambient temperature difference lost per second."""
MIN_HEATING_WATER_L = 3.0
"""The heater stays off below this level (dry-heating protection)."""
COLD_WASH_MAX_C = 20
"""Temperatures at or below this are washed without heating."""
SPIN_RAMP_RPM_PER_S = 13.0
FILL_TUMBLE_RPM = 25.0
INTERIM_SPIN_RPM = 800
INTERIM_SPIN_S = 60.0
PREWASH_S = 15 * 60.0
PREWASH_MAX_TEMP_C = 30.0
TUMBLE_PERIOD_S = 20.0
"""The drum turns for ``duty`` of every period while washing, then rests."""
PUMP_POWER_W = 35.0
STANDBY_POWER_W = 3.0
MAINS_V = 230.0
POWER_FACTOR = 0.65


@dataclass(frozen=True)
class ProgramProfile:
    wash_min: float
    """Main-wash time once the water is at temperature."""
    rinses: int
    rinse_min: float
    spin_min: float
    """Final spin hold time."""
    tumble_rpm: float
    tumble_duty: float
    """Fraction of the time the drum turns while washing (gentle programmes rest more)."""
    fill_base_l: float
    fill_per_kg_l: float
    rinse_base_l: float
    rinse_per_kg_l: float
    heat_factor: float = 1.0
    """Eco programmes heat to a lower actual temperature and wash longer instead."""
    hold_heat: bool = True
    """Whether the thermostat keeps the water hot for the whole main wash."""


PROGRAM_PROFILES: dict[Program, ProgramProfile] = {
    Program.COTTON: ProgramProfile(70, 3, 6, 8, 50, 0.80, 6.0, 1.2, 8.0, 1.0),
    Program.ECO_40_60: ProgramProfile(
        150, 3, 6, 8, 45, 0.45, 5.0, 1.0, 7.0, 0.9, heat_factor=0.75, hold_heat=False
    ),
    Program.SYNTHETICS: ProgramProfile(45, 2, 5, 6, 45, 0.70, 6.0, 1.5, 8.0, 1.2),
    Program.MIXED: ProgramProfile(40, 2, 5, 6, 45, 0.70, 6.0, 1.3, 8.0, 1.1),
    Program.WOOL: ProgramProfile(15, 2, 4, 3, 25, 0.30, 10.0, 3.0, 12.0, 3.0),
    Program.DELICATES: ProgramProfile(20, 2, 4, 3, 30, 0.35, 10.0, 2.5, 12.0, 2.5),
    Program.QUICK: ProgramProfile(12, 1, 4, 4, 50, 0.85, 5.0, 1.0, 7.0, 1.0),
    Program.SPORTSWEAR: ProgramProfile(30, 2, 5, 5, 40, 0.60, 6.0, 1.4, 8.0, 1.2),
    Program.BEDDING: ProgramProfile(55, 3, 6, 8, 45, 0.70, 8.0, 1.6, 10.0, 1.4),
    Program.TOWELS: ProgramProfile(65, 3, 6, 9, 50, 0.80, 7.0, 1.5, 9.0, 1.2),
    Program.DARK_WASH: ProgramProfile(40, 2, 5, 6, 40, 0.60, 6.0, 1.3, 8.0, 1.1),
    Program.HYGIENE: ProgramProfile(85, 3, 6, 8, 50, 0.80, 6.0, 1.2, 8.0, 1.0),
}


@dataclass(frozen=True)
class PlanStep:
    phase: CyclePhase
    kind: str
    """One of ``fill``, ``tumble``, ``drain`` or ``spin``."""
    rpm: float = 0.0
    duty: float = 1.0
    water_l: float = 0.0
    """fill: the water level to reach."""
    duration_s: float = 0.0
    """tumble/spin: how long to hold once at temperature/speed."""
    target_temp_c: float | None = None
    """tumble: heat the water to this temperature first."""
    hold_heat: bool = True


@dataclass(frozen=True)
class StepEstimate:
    duration_s: float
    energy_j: float
    water_l: float


def heating_target_c(settings: CycleSettings) -> float | None:
    """Water temperature the main wash heats to, or ``None`` for a cold wash."""
    if settings.temperature_c <= COLD_WASH_MAX_C:
        return None
    profile = PROGRAM_PROFILES[settings.program]
    return INLET_TEMP_C + (settings.temperature_c - INLET_TEMP_C) * profile.heat_factor


def build_cycle_plan(
    settings: CycleSettings, load_kg: float, duration_scale: float = 1.0
) -> list[PlanStep]:
    """Break a cycle into fill / wash / drain / spin steps.

    ``duration_scale`` shortens the timed holds (not fills, drains or heating) to make quick
    simulated cycles for training data.
    """
    profile = PROGRAM_PROFILES[settings.program]
    main_l = profile.fill_base_l + profile.fill_per_kg_l * load_kg
    rinse_l = profile.rinse_base_l + profile.rinse_per_kg_l * load_kg
    heat_to = heating_target_c(settings)
    interim_rpm = float(min(INTERIM_SPIN_RPM, settings.spin_rpm))
    steps: list[PlanStep] = []

    def block(
        phase: CyclePhase, water_l: float, minutes: float, temp: float | None, hold: bool
    ) -> None:
        steps.append(PlanStep(CyclePhase.FILL, "fill", rpm=FILL_TUMBLE_RPM, water_l=water_l))
        steps.append(
            PlanStep(
                phase,
                "tumble",
                rpm=profile.tumble_rpm,
                duty=profile.tumble_duty,
                duration_s=minutes * 60.0 * duration_scale,
                target_temp_c=temp,
                hold_heat=hold,
            )
        )
        steps.append(PlanStep(CyclePhase.DRAIN, "drain"))

    def spin(rpm: float, seconds: float) -> None:
        steps.append(
            PlanStep(CyclePhase.SPIN, "spin", rpm=rpm, duration_s=seconds * duration_scale)
        )

    if settings.prewash:
        prewash_temp = min(heat_to, PREWASH_MAX_TEMP_C) if heat_to is not None else None
        block(CyclePhase.WASH, main_l, PREWASH_S / 60.0, prewash_temp, False)
    block(CyclePhase.WASH, main_l, profile.wash_min, heat_to, profile.hold_heat)
    spin(interim_rpm, INTERIM_SPIN_S)
    rinses = profile.rinses + int(settings.extra_rinse)
    for i in range(rinses):
        block(CyclePhase.RINSE, rinse_l, profile.rinse_min, None, False)
        if i < rinses - 1:
            spin(interim_rpm, INTERIM_SPIN_S)
    spin(float(settings.spin_rpm), profile.spin_min * 60.0)
    return steps


def motor_current_a(
    rpm: float,
    load_kg: float,
    *,
    accel_rpm_s: float = 0.0,
    wear: float = 0.0,
    overload: float = 1.0,
) -> float:
    """Motor current for a drum turning at ``rpm`` with ``load_kg`` of laundry."""
    if rpm < 3.0:
        return 0.0
    current = (
        0.45
        + 0.09 * load_kg * min(rpm, 60.0) / 60.0
        + 0.0009 * rpm
        + 0.02 * max(accel_rpm_s, 0.0)
        + 0.15 * wear
    )
    return current * overload


def motor_power_w(current_a: float) -> float:
    return MAINS_V * current_a * POWER_FACTOR


def thermal_mass_j_per_k(water_l: float, load_kg: float) -> float:
    return water_l * WATER_J_PER_L_K + load_kg * LAUNDRY_J_PER_KG_K


def estimate_steps(
    steps: list[PlanStep],
    load_kg: float,
    *,
    inlet_temp_c: float = INLET_TEMP_C,
    ambient_c: float = AMBIENT_TEMP_C,
    level_l: float = 0.0,
    temp_c: float | None = None,
) -> list[StepEstimate]:
    """Analytic duration, energy and water use of each step, starting from the given drum state."""
    level = level_l
    temp = inlet_temp_c if temp_c is None else temp_c
    estimates: list[StepEstimate] = []
    for step in steps:
        if step.kind == "fill":
            added = max(step.water_l - level, 0.0)
            duration = added / FILL_RATE_LPM * 60.0
            if level + added > 0:
                temp = (temp * level + inlet_temp_c * added) / (level + added)
            level += added
            energy = motor_power_w(motor_current_a(step.rpm, load_kg)) * duration * step.duty
            estimates.append(StepEstimate(duration, energy, added))
        elif step.kind == "tumble":
            mass = thermal_mass_j_per_k(level, load_kg)
            heat_s = heat_j = 0.0
            target = step.target_temp_c
            if target is not None and temp < target and level >= MIN_HEATING_WATER_L:
                loss_w = HEAT_LOSS_PER_S * mass * ((temp + target) / 2.0 - ambient_c)
                heat_s = mass * (target - temp) / max(HEATER_POWER_W - loss_w, 1.0)
                heat_j = HEATER_POWER_W * heat_s
                temp = target
            if target is not None and step.hold_heat:
                heat_j += HEAT_LOSS_PER_S * mass * max(target - ambient_c, 0.0) * step.duration_s
            else:
                temp = ambient_c + (temp - ambient_c) * math.exp(-HEAT_LOSS_PER_S * step.duration_s)
            duration = heat_s + step.duration_s
            motor_j = motor_power_w(motor_current_a(step.rpm, load_kg)) * duration * step.duty
            estimates.append(StepEstimate(duration, heat_j + motor_j, 0.0))
        elif step.kind == "drain":
            duration = max(level - RESIDUAL_WATER_L, 0.0) / DRAIN_RATE_LPM * 60.0
            level = min(level, RESIDUAL_WATER_L)
            estimates.append(StepEstimate(duration, PUMP_POWER_W * duration, 0.0))
        elif step.kind == "spin":
            ramp_s = step.rpm / SPIN_RAMP_RPM_PER_S
            duration = ramp_s + step.duration_s
            ramp_current = motor_current_a(step.rpm / 2.0, load_kg, accel_rpm_s=SPIN_RAMP_RPM_PER_S)
            energy = (
                motor_power_w(motor_current_a(step.rpm, load_kg)) * step.duration_s
                + motor_power_w(ramp_current) * ramp_s
                + PUMP_POWER_W * duration
            )
            estimates.append(StepEstimate(duration, energy, 0.0))
        else:  # pragma: no cover - plans are built by build_cycle_plan
            raise ValueError(f"unknown plan step kind {step.kind!r}")
    return estimates


def estimate_cycle(
    settings: CycleSettings,
    load_kg: float,
    *,
    inlet_temp_c: float = INLET_TEMP_C,
    ambient_c: float = AMBIENT_TEMP_C,
) -> CycleEstimate:
    """Physics-based duration, energy and water estimate for a whole cycle."""
    steps = estimate_steps(
        build_cycle_plan(settings, load_kg), load_kg, inlet_temp_c=inlet_temp_c, ambient_c=ambient_c
    )
    duration_s = sum(s.duration_s for s in steps)
    energy_j = sum(s.energy_j for s in steps) + STANDBY_POWER_W * duration_s
    return CycleEstimate(
        duration_min=round(duration_s / 60.0, 1),
        energy_kwh=round(energy_j / 3.6e6, 3),
        water_l=round(sum(s.water_l for s in steps), 1),
    )
