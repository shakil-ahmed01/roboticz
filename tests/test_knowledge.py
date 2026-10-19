from __future__ import annotations

import random

import pytest

from smartwash.ai.datasets import random_profile
from smartwash.knowledge import (
    COLOR_MAX_TEMP_C,
    FABRIC_CARE,
    detergent_dose_ml,
    enforce_care_constraints,
    expert_settings,
    explain,
    load_warnings,
    program_suits,
)
from smartwash.schemas import CycleSettings, LoadProfile, Program


def is_safe(profile: LoadProfile, settings: CycleSettings) -> bool:
    return enforce_care_constraints(profile, settings)[0] == settings


def test_expert_choices_always_respect_care_limits():
    rng = random.Random(1)
    for _ in range(2000):
        profile = random_profile(rng)
        care = FABRIC_CARE[profile.fabric]
        for settings in (expert_settings(profile), expert_settings(profile, rng)):
            assert settings.temperature_c <= min(care.max_temp_c, COLOR_MAX_TEMP_C[profile.color])
            assert settings.spin_rpm <= care.max_spin_rpm
            assert program_suits(profile, settings.program)
            assert is_safe(profile, settings)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (
            LoadProfile(fabric="wool", color="darks", weight_kg=1.5),
            {"program": Program.WOOL, "temperature_c": 30, "spin_rpm": 600, "prewash": False},
        ),
        (
            LoadProfile(fabric="cotton", color="whites", soil="heavy", weight_kg=5),
            {"program": Program.COTTON, "temperature_c": 90, "prewash": True},
        ),
        (
            LoadProfile(fabric="cotton", color="darks", weight_kg=4),
            {"program": Program.DARK_WASH, "temperature_c": 30},
        ),
        (
            LoadProfile(fabric="cotton", color="whites", weight_kg=5, eco_priority=0.95),
            {"program": Program.ECO_40_60, "temperature_c": 40, "spin_rpm": 1400},
        ),
        (
            LoadProfile(fabric="mixed", color="colors", weight_kg=2.5, max_duration_min=45),
            {"program": Program.QUICK},
        ),
        (
            LoadProfile(fabric="baby", color="whites", weight_kg=3),
            {"program": Program.HYGIENE, "temperature_c": 60, "extra_rinse": True},
        ),
        (
            LoadProfile(fabric="baby", color="colors", weight_kg=3),
            {"program": Program.COTTON, "temperature_c": 40, "extra_rinse": True},
        ),
        (
            LoadProfile(fabric="synthetic", color="colors", weight_kg=3, stains=["grease"]),
            {"program": Program.SYNTHETICS, "temperature_c": 40},
        ),
        (
            LoadProfile(fabric="sportswear", soil="heavy", weight_kg=3, stains=["mud"]),
            {"program": Program.SPORTSWEAR, "prewash": True},
        ),
    ],
)
def test_expert_consensus_choices(profile, expected):
    settings = expert_settings(profile)
    for field, value in expected.items():
        assert getattr(settings, field) == value, field


def test_guardrails_make_unsafe_wool_settings_safe():
    wool = LoadProfile(fabric="wool", weight_kg=1)
    unsafe = CycleSettings(program="cotton", temperature_c=60, spin_rpm=1400, prewash=True)
    safe, notes = enforce_care_constraints(wool, unsafe)
    assert (safe.program, safe.temperature_c, safe.spin_rpm, safe.prewash) == (
        Program.WOOL,
        30,
        800,
        False,
    )
    assert len(notes) == 4
    assert any("too harsh for wool" in note for note in notes)


def test_hygiene_programme_replaced_when_colours_cannot_take_60_degrees():
    profile = LoadProfile(fabric="baby", color="colors")
    unsafe = CycleSettings(program="hygiene", temperature_c=60, spin_rpm=1000)
    safe, notes = enforce_care_constraints(profile, unsafe)
    assert safe.program is Program.COTTON
    assert safe.temperature_c == 40
    assert safe.extra_rinse
    assert any("colours can bleed" in note for note in notes)


def test_eco_programme_temperature_raised_to_its_minimum():
    profile = LoadProfile(fabric="cotton", color="whites")
    safe, notes = enforce_care_constraints(
        profile, CycleSettings(program="eco_40_60", temperature_c=30, spin_rpm=1200)
    )
    assert safe.temperature_c == 40
    assert "raised" in notes[0]


def test_safe_settings_pass_unchanged():
    profile = LoadProfile(fabric="cotton", color="colors")
    settings = CycleSettings(program="cotton", temperature_c=40, spin_rpm=1200)
    assert enforce_care_constraints(profile, settings) == (settings, [])


def test_detergent_dose_scales_with_soil_hardness_and_weight():
    base = {"fabric": "cotton", "weight_kg": 5}
    light = detergent_dose_ml(LoadProfile(**base, soil="light"))
    normal = detergent_dose_ml(LoadProfile(**base))
    heavy = detergent_dose_ml(LoadProfile(**base, soil="heavy"))
    hard = detergent_dose_ml(LoadProfile(**base, water_hardness="hard"))
    assert light < normal < heavy
    assert hard > normal
    assert detergent_dose_ml(LoadProfile(fabric="wool", weight_kg=0.5)) == 15
    assert detergent_dose_ml(LoadProfile(fabric="towels", weight_kg=15, soil="heavy")) == 120
    assert all(dose % 5 == 0 for dose in (light, normal, heavy, hard))


def test_explanations_mention_every_option():
    profile = LoadProfile(fabric="baby", color="whites", soil="normal", stains=["protein"])
    settings = expert_settings(profile)
    text = " ".join(explain(profile, settings))
    assert settings.prewash and "Pre-wash" in text
    assert settings.extra_rinse and "Extra rinse" in text
    assert "ml of detergent" in text


def test_load_warnings():
    heavy = LoadProfile(fabric="cotton", weight_kg=10)
    settings = expert_settings(heavy)
    assert "capacity" in load_warnings(heavy, settings, capacity_kg=8)[0]
    delicates = LoadProfile(fabric="delicate", weight_kg=5)
    assert "half full" in load_warnings(delicates, expert_settings(delicates), capacity_kg=8)[0]
    assert load_warnings(LoadProfile(fabric="cotton"), settings, capacity_kg=8) == []
