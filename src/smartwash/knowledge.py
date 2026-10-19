"""Laundry-care domain knowledge.

* ``expert_settings`` is a rule-based "teacher" distilled from textile-care guidance (care-label
  limits, colour fastness, soil and stain handling). It labels the synthetic training data the
  recommender learns from. Given an ``rng`` it samples the spread of choices different experts make;
  without one it returns the most common choice.
* ``enforce_care_constraints`` is a hard guardrail applied to every model output, and to user
  feedback before the model learns from it: never hotter, faster or harsher than the fabric allows.
* ``explain``, ``load_warnings`` and ``FAULT_GUIDE`` turn decisions into advice people can act on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .schemas import (
    SPIN_SPEEDS_RPM,
    TEMPERATURES_C,
    ColorGroup,
    CycleSettings,
    Fabric,
    FaultType,
    LoadProfile,
    Program,
    Severity,
    SoilLevel,
    StainType,
    WaterHardness,
)


class RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True)
class FabricCare:
    program: Program
    """Programme to use by default."""
    max_temp_c: int
    max_spin_rpm: int
    default_spin_rpm: int
    max_intensity: int
    """Harshest drum action tolerated, from 1 (hand-wash) to 4 (intensive)."""
    label: str


FABRIC_CARE: dict[Fabric, FabricCare] = {
    Fabric.COTTON: FabricCare(Program.COTTON, 90, 1400, 1200, 4, "cotton"),
    Fabric.SYNTHETIC: FabricCare(Program.SYNTHETICS, 60, 1200, 800, 3, "synthetics"),
    Fabric.WOOL: FabricCare(Program.WOOL, 30, 800, 600, 1, "wool"),
    Fabric.DELICATE: FabricCare(Program.DELICATES, 30, 600, 400, 1, "delicates"),
    Fabric.DENIM: FabricCare(Program.DARK_WASH, 40, 1200, 800, 4, "denim"),
    Fabric.BEDDING: FabricCare(Program.BEDDING, 90, 1400, 1000, 4, "bedding"),
    Fabric.TOWELS: FabricCare(Program.TOWELS, 90, 1400, 1200, 4, "towels"),
    Fabric.SPORTSWEAR: FabricCare(Program.SPORTSWEAR, 40, 1000, 800, 2, "sportswear"),
    Fabric.MIXED: FabricCare(Program.MIXED, 60, 1400, 1000, 3, "mixed fabrics"),
    Fabric.BABY: FabricCare(Program.HYGIENE, 60, 1200, 1000, 4, "baby clothes"),
}

COLOR_MAX_TEMP_C: dict[ColorGroup, int] = {
    ColorGroup.WHITES: 90,
    ColorGroup.LIGHTS: 60,
    ColorGroup.COLORS: 40,
    ColorGroup.DARKS: 40,
}

_COLOR_LIMIT_REASON: dict[ColorGroup, str] = {
    ColorGroup.WHITES: "whites",
    ColorGroup.LIGHTS: "light colours can discolour above 60 °C",
    ColorGroup.COLORS: "colours can bleed above 40 °C",
    ColorGroup.DARKS: "dark colours fade above 40 °C",
}


@dataclass(frozen=True)
class ProgramSpec:
    intensity: int
    min_temp_c: int
    max_temp_c: int
    max_spin_rpm: int
    allows_prewash: bool
    description: str


PROGRAM_SPECS: dict[Program, ProgramSpec] = {
    Program.COTTON: ProgramSpec(
        4, 20, 90, 1400, True, "Cotton programme: strong drum action for robust cotton"
    ),
    Program.ECO_40_60: ProgramSpec(
        3,
        40,
        60,
        1400,
        False,
        "Eco 40-60 programme: a longer wash at a lower actual temperature that cleans 40-60 °C "
        "cotton with much less energy",
    ),
    Program.SYNTHETICS: ProgramSpec(
        2, 20, 60, 1200, True, "Synthetics programme: gentler action that limits creasing"
    ),
    Program.MIXED: ProgramSpec(
        3, 20, 40, 1400, True, "Mixed programme: for cotton and synthetics washed together"
    ),
    Program.WOOL: ProgramSpec(
        1, 20, 30, 800, False, "Wool programme: gentle rocking so wool does not felt or shrink"
    ),
    Program.DELICATES: ProgramSpec(
        1,
        20,
        30,
        600,
        False,
        "Delicates programme: minimal drum action for silk, lace and lingerie",
    ),
    Program.QUICK: ProgramSpec(
        2, 20, 40, 1200, False, "Quick programme: a short cycle for a small, lightly soiled load"
    ),
    Program.SPORTSWEAR: ProgramSpec(
        2,
        20,
        40,
        1000,
        True,
        "Sportswear programme: lifts sweat and odour while protecting technical fibres",
    ),
    Program.BEDDING: ProgramSpec(
        3, 20, 90, 1200, True, "Bedding programme: extra water for bulky sheets and covers"
    ),
    Program.TOWELS: ProgramSpec(
        4,
        20,
        90,
        1400,
        True,
        "Towels programme: a thorough wash and high spin for absorbent cotton",
    ),
    Program.DARK_WASH: ProgramSpec(
        2,
        20,
        40,
        1200,
        True,
        "Dark wash programme: reduced drum action that keeps dark colours and denim from fading",
    ),
    Program.HYGIENE: ProgramSpec(
        4,
        60,
        90,
        1400,
        True,
        "Hygiene programme: holds a high temperature to kill germs and allergens",
    ),
}

BASE_TEMPERATURE_C: dict[Fabric, dict[ColorGroup, int]] = {
    Fabric.COTTON: {
        ColorGroup.WHITES: 60,
        ColorGroup.LIGHTS: 40,
        ColorGroup.COLORS: 40,
        ColorGroup.DARKS: 30,
    },
    Fabric.TOWELS: {
        ColorGroup.WHITES: 60,
        ColorGroup.LIGHTS: 60,
        ColorGroup.COLORS: 40,
        ColorGroup.DARKS: 40,
    },
    Fabric.BEDDING: {
        ColorGroup.WHITES: 60,
        ColorGroup.LIGHTS: 60,
        ColorGroup.COLORS: 40,
        ColorGroup.DARKS: 40,
    },
    Fabric.BABY: {
        ColorGroup.WHITES: 60,
        ColorGroup.LIGHTS: 60,
        ColorGroup.COLORS: 40,
        ColorGroup.DARKS: 40,
    },
    Fabric.SYNTHETIC: {
        ColorGroup.WHITES: 40,
        ColorGroup.LIGHTS: 40,
        ColorGroup.COLORS: 30,
        ColorGroup.DARKS: 30,
    },
    Fabric.MIXED: {
        ColorGroup.WHITES: 40,
        ColorGroup.LIGHTS: 40,
        ColorGroup.COLORS: 40,
        ColorGroup.DARKS: 30,
    },
    Fabric.DENIM: {
        ColorGroup.WHITES: 40,
        ColorGroup.LIGHTS: 30,
        ColorGroup.COLORS: 30,
        ColorGroup.DARKS: 30,
    },
    Fabric.SPORTSWEAR: dict.fromkeys(ColorGroup, 30),
    Fabric.WOOL: dict.fromkeys(ColorGroup, 30),
    Fabric.DELICATE: dict.fromkeys(ColorGroup, 30),
}

QUICK_FABRICS = frozenset(
    {Fabric.COTTON, Fabric.SYNTHETIC, Fabric.MIXED, Fabric.SPORTSWEAR, Fabric.DENIM}
)
ECO_REPLACEABLE = frozenset({Program.COTTON, Program.BEDDING, Program.TOWELS})
"""Programmes the Eco 40-60 programme can stand in for."""
QUICK_MAX_KG = 4.0
ECO_MIN_BUDGET_MIN = 210

DETERGENT_ML_PER_KG = 9.0
SOIL_DOSE_FACTOR = {SoilLevel.LIGHT: 0.75, SoilLevel.NORMAL: 1.0, SoilLevel.HEAVY: 1.35}
HARDNESS_DOSE_FACTOR = {
    WaterHardness.SOFT: 0.85,
    WaterHardness.MEDIUM: 1.0,
    WaterHardness.HARD: 1.25,
}
FABRIC_DOSE_FACTOR = {Fabric.WOOL: 0.6, Fabric.DELICATE: 0.6, Fabric.SPORTSWEAR: 0.8}


def temperature_step(temp_c: int, steps: int) -> int:
    """Move ``steps`` positions along ``TEMPERATURES_C``, staying within range."""
    i = TEMPERATURES_C.index(temp_c) + steps
    return TEMPERATURES_C[min(max(i, 0), len(TEMPERATURES_C) - 1)]


def spin_step(rpm: int, steps: int) -> int:
    i = SPIN_SPEEDS_RPM.index(rpm) + steps
    return SPIN_SPEEDS_RPM[min(max(i, 0), len(SPIN_SPEEDS_RPM) - 1)]


def _chance(probability: float, rng: RandomSource | None) -> bool:
    if rng is None:
        return probability >= 0.5
    return rng.random() < probability


def _eco_probability(eco_priority: float) -> float:
    return 1.0 / (1.0 + math.exp(-(eco_priority - 0.7) * 15.0))


def expert_settings(profile: LoadProfile, rng: RandomSource | None = None) -> CycleSettings:
    """What a laundry expert would choose for this load (sampled when ``rng`` is given)."""
    care = FABRIC_CARE[profile.fabric]
    stains = set(profile.stains)

    program = care.program
    if profile.color is ColorGroup.DARKS and profile.fabric in (Fabric.COTTON, Fabric.MIXED):
        program = Program.DARK_WASH
    if profile.fabric is Fabric.BABY and profile.color not in (
        ColorGroup.WHITES,
        ColorGroup.LIGHTS,
    ):
        program = Program.COTTON
    if (
        profile.sensitive_skin
        and profile.fabric in (Fabric.COTTON, Fabric.BEDDING, Fabric.TOWELS)
        and profile.color in (ColorGroup.WHITES, ColorGroup.LIGHTS)
    ):
        program = Program.HYGIENE

    temp = BASE_TEMPERATURE_C[profile.fabric][profile.color]
    cotton_like = (Fabric.COTTON, Fabric.MIXED, Fabric.TOWELS, Fabric.BEDDING)
    if profile.color is ColorGroup.COLORS and profile.fabric in cotton_like and _chance(0.25, rng):
        temp = 30
    if profile.soil is SoilLevel.LIGHT and _chance(0.8, rng):
        temp = temperature_step(temp, -1)
    elif profile.soil is SoilLevel.HEAVY and _chance(0.85, rng):
        temp = temperature_step(temp, +1)
    if stains & {StainType.GREASE, StainType.TANNIN}:
        temp = max(temp, 40)

    budget = profile.max_duration_min
    eco_mode = profile.soil is not SoilLevel.HEAVY and _chance(
        _eco_probability(profile.eco_priority), rng
    )
    quick_ok = (
        profile.fabric in QUICK_FABRICS
        and profile.soil is not SoilLevel.HEAVY
        and not stains & {StainType.MUD, StainType.PROTEIN}
        and profile.weight_kg <= QUICK_MAX_KG
        and program is not Program.HYGIENE
    )
    wants_quick = (budget is not None and budget <= 60) or (
        profile.soil is SoilLevel.LIGHT and profile.weight_kg <= 3.0 and _chance(0.8, rng)
    )
    if quick_ok and wants_quick:
        program = Program.QUICK
    elif (
        eco_mode
        and program in ECO_REPLACEABLE
        and temp >= 40
        and (budget is None or budget >= ECO_MIN_BUDGET_MIN)
    ):
        program = Program.ECO_40_60
        temp = 40
    elif eco_mode and temp >= 40 and program is not Program.HYGIENE:
        temp = temperature_step(temp, -1)
    if program is Program.HYGIENE:
        temp = max(temp, 60)

    spin = care.default_spin_rpm
    if care.max_intensity >= 3:
        if eco_mode:
            spin = spin_step(spin, +1)  # drier laundry needs less tumble-drying energy
        elif rng is not None:
            r = rng.random()
            if r < 0.12:
                spin = spin_step(spin, -1)
            elif r > 0.88:
                spin = spin_step(spin, +1)

    prewash = bool(stains & {StainType.MUD, StainType.PROTEIN}) or (
        profile.soil is SoilLevel.HEAVY
        and profile.fabric in (Fabric.COTTON, Fabric.TOWELS, Fabric.BEDDING, Fabric.DENIM)
        and _chance(0.7, rng)
    )
    extra_rinse = (
        profile.sensitive_skin
        or profile.fabric is Fabric.BABY
        or (profile.water_hardness is WaterHardness.HARD and profile.soil is SoilLevel.HEAVY)
    )
    settings = CycleSettings(
        program=program,
        temperature_c=temp,
        spin_rpm=spin,
        prewash=prewash,
        extra_rinse=extra_rinse,
    )
    return enforce_care_constraints(profile, settings)[0]


def _program_fits(program: Program, care: FabricCare, temp_cap: int) -> bool:
    spec = PROGRAM_SPECS[program]
    return spec.intensity <= care.max_intensity and spec.min_temp_c <= temp_cap


def program_suits(profile: LoadProfile, program: Program) -> bool:
    """Whether ``program`` can wash this load without breaking its care limits."""
    care = FABRIC_CARE[profile.fabric]
    return _program_fits(program, care, min(care.max_temp_c, COLOR_MAX_TEMP_C[profile.color]))


def enforce_care_constraints(
    profile: LoadProfile, settings: CycleSettings
) -> tuple[CycleSettings, list[str]]:
    """Make ``settings`` safe for the load; return the safe settings and what had to change."""
    care = FABRIC_CARE[profile.fabric]
    color_cap = COLOR_MAX_TEMP_C[profile.color]
    temp_cap = min(care.max_temp_c, color_cap)
    notes: list[str] = []

    program = settings.program
    spec = PROGRAM_SPECS[program]
    if not _program_fits(program, care, temp_cap):
        replacement = next(
            p
            for p in (care.program, Program.COTTON, Program.DELICATES)
            if _program_fits(p, care, temp_cap)
        )
        if spec.intensity > care.max_intensity:
            reason = f"its drum action is too harsh for {care.label}"
        else:
            reason = (
                f"it needs at least {spec.min_temp_c} °C but this load is limited to {temp_cap} °C"
            )
        notes.append(f"Programme changed from {program.value} to {replacement.value}: {reason}.")
        program = replacement
        spec = PROGRAM_SPECS[program]

    temp = settings.temperature_c
    max_temp = min(temp_cap, spec.max_temp_c)
    if temp > max_temp:
        new_temp = max(t for t in TEMPERATURES_C if t <= max_temp)
        if care.max_temp_c == max_temp:
            reason = f"{care.label} tolerates at most {care.max_temp_c} °C"
        elif color_cap == max_temp:
            reason = _COLOR_LIMIT_REASON[profile.color]
        else:
            reason = f"the {program.value} programme runs at up to {spec.max_temp_c} °C"
        notes.append(f"Temperature lowered from {temp} °C to {new_temp} °C: {reason}.")
        temp = new_temp
    if temp < spec.min_temp_c:
        new_temp = min(t for t in TEMPERATURES_C if t >= spec.min_temp_c)
        notes.append(
            f"Temperature raised from {temp} °C to {new_temp} °C: the {program.value} programme "
            f"runs at {spec.min_temp_c} °C or more."
        )
        temp = new_temp

    spin = settings.spin_rpm
    max_spin = min(care.max_spin_rpm, spec.max_spin_rpm)
    if spin > max_spin:
        new_spin = max(s for s in SPIN_SPEEDS_RPM if s <= max_spin)
        if care.max_spin_rpm == max_spin:
            reason = f"{care.label} should not be spun faster than {care.max_spin_rpm} rpm"
        else:
            reason = f"the {program.value} programme spins at up to {spec.max_spin_rpm} rpm"
        notes.append(f"Spin lowered from {spin} to {new_spin} rpm: {reason}.")
        spin = new_spin

    prewash = settings.prewash
    if prewash and (not spec.allows_prewash or care.max_intensity <= 1):
        notes.append(f"Pre-wash turned off: it is not suitable for {care.label} or this programme.")
        prewash = False

    extra_rinse = settings.extra_rinse
    if not extra_rinse and (profile.sensitive_skin or profile.fabric is Fabric.BABY):
        notes.append("Extra rinse turned on: it removes detergent residue for sensitive skin.")
        extra_rinse = True

    safe = CycleSettings(
        program=program,
        temperature_c=temp,
        spin_rpm=spin,
        prewash=prewash,
        extra_rinse=extra_rinse,
    )
    return safe, notes


def detergent_dose_ml(profile: LoadProfile) -> int:
    dose = (
        DETERGENT_ML_PER_KG
        * profile.weight_kg
        * SOIL_DOSE_FACTOR[profile.soil]
        * HARDNESS_DOSE_FACTOR[profile.water_hardness]
        * FABRIC_DOSE_FACTOR.get(profile.fabric, 1.0)
    )
    return int(5 * round(min(max(dose, 15.0), 120.0) / 5))


_SOIL_ADVERB = {
    SoilLevel.LIGHT: "lightly",
    SoilLevel.NORMAL: "normally",
    SoilLevel.HEAVY: "heavily",
}


def _temperature_reason(profile: LoadProfile, settings: CycleSettings) -> str:
    temp = settings.temperature_c
    label = FABRIC_CARE[profile.fabric].label
    if settings.program is Program.ECO_40_60:
        return "Eco 40-60 cleans laundry rated for 40-60 °C while heating the water less."
    if temp >= 90:
        return "90 °C gives a boil wash for heavily soiled whites."
    if temp >= 60:
        return "60 °C kills bacteria and dust mites and lifts heavy soiling."
    if profile.fabric is Fabric.WOOL:
        return f"{temp} °C keeps wool from shrinking and felting."
    if profile.fabric is Fabric.DELICATE:
        return f"{temp} °C stops delicate fibres shrinking or losing their shape."
    if profile.color is ColorGroup.DARKS:
        return f"{temp} °C keeps dark colours from fading."
    if temp <= 20:
        return "A cold wash uses the least energy and is gentlest on fabrics."
    if profile.eco_priority >= 0.7:
        return f"{temp} °C saves energy and still cleans everyday soiling."
    if profile.soil is SoilLevel.LIGHT:
        return f"Lightly soiled laundry gets clean at {temp} °C, saving energy."
    if temp == 40:
        return "40 °C cleans everyday laundry well without fading colours."
    return f"{temp} °C is enough for normally soiled {label} and protects the fibres."


def _spin_reason(profile: LoadProfile, settings: CycleSettings) -> str:
    rpm = settings.spin_rpm
    if rpm <= 600:
        return f"A gentle {rpm} rpm spin protects delicate fabrics."
    article = "An" if str(rpm).startswith("8") else "A"
    if rpm >= 1200:
        extra = " and less tumble-drying energy" if profile.eco_priority >= 0.7 else ""
        return f"{article} {rpm} rpm spin extracts more water, for faster drying{extra}."
    return f"{article} {rpm} rpm spin balances water extraction against creasing."


def explain(profile: LoadProfile, settings: CycleSettings) -> list[str]:
    """Human-readable reasons for each part of ``settings``."""
    reasons = [PROGRAM_SPECS[settings.program].description + "."]
    reasons.append(_temperature_reason(profile, settings))
    reasons.append(_spin_reason(profile, settings))
    stains = set(profile.stains)
    if settings.prewash:
        if stains & {StainType.MUD, StainType.PROTEIN}:
            names = " and ".join(
                s
                for s, t in (("mud", StainType.MUD), ("protein (blood, sweat)", StainType.PROTEIN))
                if t in stains
            )
            reasons.append(f"Pre-wash loosens {names} stains before the main wash.")
        else:
            reasons.append("Pre-wash tackles the heavy soiling before the main wash.")
    if settings.extra_rinse:
        if profile.sensitive_skin or profile.fabric is Fabric.BABY:
            reasons.append("Extra rinse removes detergent residue for sensitive skin.")
        elif profile.water_hardness is WaterHardness.HARD:
            reasons.append("Extra rinse flushes out the extra detergent hard water needs.")
        else:
            reasons.append("Extra rinse for a cleaner finish.")
    dose = detergent_dose_ml(profile)
    detergent = (
        " of wool/delicates detergent"
        if profile.fabric in (Fabric.WOOL, Fabric.DELICATE)
        else " of detergent"
    )
    reasons.append(
        f"Use about {dose} ml{detergent} for {profile.weight_kg:g} kg of "
        f"{_SOIL_ADVERB[profile.soil]} soiled laundry in {profile.water_hardness.value} water."
    )
    return reasons


def load_warnings(profile: LoadProfile, settings: CycleSettings, capacity_kg: float) -> list[str]:
    warnings: list[str] = []
    if profile.weight_kg > capacity_kg:
        warnings.append(
            f"{profile.weight_kg:g} kg is more than the machine's {capacity_kg:g} kg capacity; "
            "split the load."
        )
    elif (
        settings.program in (Program.WOOL, Program.DELICATES)
        and profile.weight_kg > capacity_kg / 2
    ):
        warnings.append(
            f"Wool and delicates programmes work best with the drum at most half full "
            f"(about {capacity_kg / 2:g} kg)."
        )
    elif settings.program is Program.QUICK and profile.weight_kg > QUICK_MAX_KG:
        warnings.append(f"Quick programmes are designed for loads up to about {QUICK_MAX_KG:g} kg.")
    return warnings


@dataclass(frozen=True)
class FaultAdvice:
    severity: Severity
    message: str
    action: str


FAULT_GUIDE: dict[FaultType, FaultAdvice] = {
    FaultType.NONE: FaultAdvice(Severity.INFO, "Machine operating normally.", "No action needed."),
    FaultType.UNBALANCED_LOAD: FaultAdvice(
        Severity.WARNING,
        "The load is unbalanced: the drum cannot reach its spin speed and vibrates strongly.",
        "Pause, spread the laundry evenly (mix large and small items, or add a towel), then "
        "resume.",
    ),
    FaultType.HEATER_FAILURE: FaultAdvice(
        Severity.WARNING,
        "The water is not heating up as expected; the heating element or temperature sensor may "
        "be failing (limescale is a common cause).",
        "This load may not be cleaned at the selected temperature. Run a descaling cycle; if it "
        "keeps happening, have the heater and temperature sensor checked.",
    ),
    FaultType.DRAIN_BLOCKAGE: FaultAdvice(
        Severity.CRITICAL,
        "Water is draining far too slowly; the pump filter or drain hose is probably blocked.",
        "Stop the cycle, clean the drain-pump filter and check the drain hose for kinks before "
        "restarting.",
    ),
    FaultType.BEARING_WEAR: FaultAdvice(
        Severity.WARNING,
        "Abnormal drum vibration at every speed, a typical sign of worn drum bearings.",
        "Book a service soon: worn bearings get noisy and can damage the drum and seals.",
    ),
    FaultType.WATER_SUPPLY: FaultAdvice(
        Severity.WARNING,
        "The machine is filling with water far too slowly.",
        "Check that the tap is fully open, the inlet hose is not kinked and its filter screen is "
        "clean.",
    ),
    FaultType.MOTOR_OVERLOAD: FaultAdvice(
        Severity.CRITICAL,
        "The motor is drawing excessive current and cannot hold its speed; the drum may be "
        "overloaded or the motor is failing.",
        "Stop the cycle and take some laundry out. If it happens with normal loads, have the motor "
        "checked.",
    ),
    FaultType.UNKNOWN: FaultAdvice(
        Severity.WARNING,
        "Unusual behaviour that does not match a known fault pattern.",
        "Keep an eye on the machine; if it happens again, check its error log and consider a "
        "service visit.",
    ),
}
