"""Numeric encoding of load profiles for the recommender."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..schemas import ColorGroup, Fabric, LoadProfile, SoilLevel, StainType, WaterHardness

NO_TIME_BUDGET_MIN = 600.0

_SOIL = {SoilLevel.LIGHT: 0.0, SoilLevel.NORMAL: 1.0, SoilLevel.HEAVY: 2.0}
_HARDNESS = {WaterHardness.SOFT: 0.0, WaterHardness.MEDIUM: 1.0, WaterHardness.HARD: 2.0}

PROFILE_FEATURES: tuple[str, ...] = (
    *(f"fabric={f.value}" for f in Fabric),
    *(f"color={c.value}" for c in ColorGroup),
    "soil",
    "weight_kg",
    *(f"stain={s.value}" for s in StainType),
    "sensitive_skin",
    "eco_priority",
    "time_budget_min",
    "has_time_budget",
    "water_hardness",
)


def encode_profile(profile: LoadProfile) -> list[float]:
    stains = set(profile.stains)
    budget = profile.max_duration_min
    return [
        *(float(profile.fabric is f) for f in Fabric),
        *(float(profile.color is c) for c in ColorGroup),
        _SOIL[profile.soil],
        profile.weight_kg,
        *(float(s in stains) for s in StainType),
        float(profile.sensitive_skin),
        profile.eco_priority,
        min(float(budget), NO_TIME_BUDGET_MIN) if budget is not None else NO_TIME_BUDGET_MIN,
        float(budget is not None),
        _HARDNESS[profile.water_hardness],
    ]


def encode_profiles(profiles: Sequence[LoadProfile]) -> np.ndarray:
    return np.asarray([encode_profile(p) for p in profiles], dtype=np.float64).reshape(
        len(profiles), len(PROFILE_FEATURES)
    )
