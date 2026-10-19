"""Synthetic training data.

There is no usage history on day one, so both models are bootstrapped on synthetic data:

* load profiles drawn from a realistic household mix and labelled by the rule-based laundry expert
  (``knowledge.expert_settings``), for the cycle recommender;
* labelled telemetry from the physics simulator with injected faults, for the fault detector.

Real feedback and real cycles then refine them (see ``feedback.py`` and ``estimator.py``).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ..devices.simulator import SimRecord, WasherSimulator
from ..knowledge import PROGRAM_SPECS, expert_settings
from ..physics import COLD_WASH_MAX_C
from ..schemas import (
    SPIN_SPEEDS_RPM,
    TEMPERATURES_C,
    ColorGroup,
    CycleSettings,
    Fabric,
    FaultType,
    LoadProfile,
    Program,
    SoilLevel,
    StainType,
    WaterHardness,
)
from .anomaly import TelemetryWindower, window_features

T = TypeVar("T")

FABRIC_MIX: dict[Fabric, float] = {
    Fabric.COTTON: 0.22,
    Fabric.MIXED: 0.14,
    Fabric.SYNTHETIC: 0.11,
    Fabric.TOWELS: 0.08,
    Fabric.BEDDING: 0.08,
    Fabric.DENIM: 0.08,
    Fabric.SPORTSWEAR: 0.09,
    Fabric.WOOL: 0.06,
    Fabric.DELICATE: 0.06,
    Fabric.BABY: 0.08,
}
COLOR_MIX: dict[ColorGroup, float] = {
    ColorGroup.WHITES: 0.2,
    ColorGroup.LIGHTS: 0.2,
    ColorGroup.COLORS: 0.35,
    ColorGroup.DARKS: 0.25,
}
WEIGHT_RANGE_KG: dict[Fabric, tuple[float, float]] = {
    Fabric.WOOL: (0.5, 3.0),
    Fabric.DELICATE: (0.5, 3.0),
    Fabric.BEDDING: (2.0, 7.0),
    Fabric.TOWELS: (1.5, 7.0),
    Fabric.BABY: (1.0, 4.0),
}
TIME_BUDGETS_MIN = (30, 45, 60, 90, 120, 180, 240)

SIMULATED_FAULTS: tuple[FaultType, ...] = (
    FaultType.UNBALANCED_LOAD,
    FaultType.HEATER_FAILURE,
    FaultType.DRAIN_BLOCKAGE,
    FaultType.BEARING_WEAR,
    FaultType.WATER_SUPPLY,
    FaultType.MOTOR_OVERLOAD,
)


def _weighted(rng: random.Random, weights: Mapping[T, float]) -> T:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def random_profile(rng: random.Random) -> LoadProfile:
    fabric = _weighted(rng, FABRIC_MIX)
    if fabric is Fabric.DENIM:
        color = _weighted(
            rng, {ColorGroup.DARKS: 0.6, ColorGroup.COLORS: 0.3, ColorGroup.LIGHTS: 0.1}
        )
    elif fabric is Fabric.BABY:
        color = _weighted(
            rng,
            {
                ColorGroup.WHITES: 0.35,
                ColorGroup.LIGHTS: 0.35,
                ColorGroup.COLORS: 0.25,
                ColorGroup.DARKS: 0.05,
            },
        )
    else:
        color = _weighted(rng, COLOR_MIX)
    low, high = WEIGHT_RANGE_KG.get(fabric, (1.0, 8.0))
    stains = []
    for stain in StainType:
        probability = 0.12
        if stain is StainType.MUD and fabric in (Fabric.SPORTSWEAR, Fabric.DENIM):
            probability = 0.35
        elif stain is StainType.PROTEIN and fabric in (Fabric.SPORTSWEAR, Fabric.BABY):
            probability = 0.3
        if rng.random() < probability:
            stains.append(stain)
    return LoadProfile(
        fabric=fabric,
        color=color,
        soil=_weighted(rng, {SoilLevel.LIGHT: 0.25, SoilLevel.NORMAL: 0.55, SoilLevel.HEAVY: 0.2}),
        weight_kg=round(rng.uniform(low, high), 1),
        stains=stains,
        sensitive_skin=rng.random() < (0.5 if fabric is Fabric.BABY else 0.15),
        eco_priority=round(rng.random() * 20) / 20,
        max_duration_min=None if rng.random() < 0.7 else rng.choice(TIME_BUDGETS_MIN),
        water_hardness=_weighted(
            rng,
            {WaterHardness.SOFT: 0.3, WaterHardness.MEDIUM: 0.4, WaterHardness.HARD: 0.3},
        ),
    )


def generate_recommendation_data(
    n_samples: int, seed: int = 0
) -> tuple[list[LoadProfile], list[CycleSettings]]:
    """Random loads, each labelled with a sampled expert choice."""
    rng = random.Random(seed)
    profiles = [random_profile(rng) for _ in range(n_samples)]
    return profiles, [expert_settings(p, rng) for p in profiles]


@dataclass
class FaultDataset:
    X: np.ndarray
    """One row of ``window_features`` per telemetry window."""
    y: np.ndarray
    """Ground-truth fault visible in each window (``FaultType`` values)."""
    cycle: np.ndarray
    """Index of the simulated cycle each window came from."""
    cycle_fault: np.ndarray
    """Fault injected into that cycle."""


def random_settings(rng: random.Random) -> CycleSettings:
    """Any settings the machine accepts, as a person choosing by hand might pick them."""
    program = rng.choice(list(Program))
    spec = PROGRAM_SPECS[program]
    return CycleSettings(
        program=program,
        temperature_c=rng.choice(
            [t for t in TEMPERATURES_C if spec.min_temp_c <= t <= spec.max_temp_c]
        ),
        spin_rpm=rng.choice([s for s in SPIN_SPEEDS_RPM if s <= spec.max_spin_rpm]),
        prewash=spec.allows_prewash and rng.random() < 0.2,
        extra_rinse=rng.random() < 0.2,
    )


def exercises(fault: FaultType, settings: CycleSettings) -> bool:
    """Whether a cycle with these settings would show the fault at all.

    A broken heater is invisible in a cold wash, and an unbalanced load only shows at high spin.
    """
    if fault is FaultType.HEATER_FAILURE:
        return settings.temperature_c > COLD_WASH_MAX_C
    if fault is FaultType.UNBALANCED_LOAD:
        return settings.spin_rpm >= 1000
    return True


def simulate_cycle(
    rng: random.Random,
    fault: FaultType,
    *,
    severity: float,
    duration_scale: float = 0.35,
    sample_period_s: float = 2.0,
    manual_fraction: float = 0.3,
) -> list[SimRecord]:
    """Run one random household cycle, on a randomly varied machine, that exercises ``fault``.

    Most cycles use the settings an expert would pick; ``manual_fraction`` of them use whatever a
    person might choose, so the detector learns healthy behaviour across all real usage.
    """
    for _ in range(50):
        if rng.random() < manual_fraction:
            settings = random_settings(rng)
            load_kg = round(rng.uniform(0.5, 8.0), 1)
        else:
            profile = random_profile(rng)
            settings = expert_settings(profile, rng)
            load_kg = profile.weight_kg
        if exercises(fault, settings):
            break
    sim = WasherSimulator(
        sample_period_s=sample_period_s,
        seed=rng.randrange(2**32),
        inlet_temp_c=rng.uniform(8.0, 20.0),
        ambient_c=rng.uniform(15.0, 25.0),
    )
    sim.start(
        settings,
        load_kg,
        fault=fault,
        severity=severity,
        duration_scale=duration_scale,
    )
    return sim.run()


def label_windows(records: list[SimRecord]) -> list[tuple[list[SimRecord], FaultType]]:
    """Cut a cycle's telemetry into windows labelled with their majority ground truth."""
    windower: TelemetryWindower[SimRecord] = TelemetryWindower(key=lambda r: r.sample)
    windows = [w for record in records for w in windower.push(record)]
    windows += windower.flush()
    labelled = []
    for window in windows:
        faulty = [r.label for r in window if r.label is not FaultType.NONE]
        label = faulty[0] if len(faulty) * 2 >= len(window) else FaultType.NONE
        labelled.append((window, label))
    return labelled


def generate_fault_data(
    n_cycles: int = 300,
    seed: int = 0,
    *,
    healthy_fraction: float = 0.3,
    severity_range: tuple[float, float] = (0.3, 1.4),
    duration_scale: float = 0.35,
    sample_period_s: float = 2.0,
) -> FaultDataset:
    """Simulate healthy and faulty cycles and turn their telemetry into labelled windows.

    Severities below ~0.6 are early-stage faults that are hard to tell from normal variation.
    """
    rng = random.Random(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    cycles: list[int] = []
    cycle_faults: list[str] = []
    n_faulty = 0
    for i in range(n_cycles):
        if rng.random() < healthy_fraction:
            fault = FaultType.NONE
        else:
            fault = SIMULATED_FAULTS[n_faulty % len(SIMULATED_FAULTS)]
            n_faulty += 1
        records = simulate_cycle(
            rng,
            fault,
            severity=rng.uniform(severity_range[0], severity_range[1]),
            duration_scale=duration_scale,
            sample_period_s=sample_period_s,
        )
        for window, label in label_windows(records):
            rows.append(window_features([r.sample for r in window]))
            labels.append(label.value)
            cycles.append(i)
            cycle_faults.append(fault.value)
    return FaultDataset(
        X=np.vstack(rows),
        y=np.asarray(labels),
        cycle=np.asarray(cycles),
        cycle_fault=np.asarray(cycle_faults),
    )


def healthy_cycle_features(
    n_cycles: int, seed: int = 0, *, duration_scale: float = 1.0, sample_period_s: float = 2.0
) -> list[np.ndarray]:
    """Window features of complete, real-length healthy cycles: one array per cycle."""
    rng = random.Random(seed)
    cycles = []
    for _ in range(n_cycles):
        records = simulate_cycle(
            rng,
            FaultType.NONE,
            severity=1.0,
            duration_scale=duration_scale,
            sample_period_s=sample_period_s,
        )
        windows = label_windows(records)
        if windows:
            cycles.append(np.vstack([window_features([r.sample for r in w]) for w, _ in windows]))
    return cycles
