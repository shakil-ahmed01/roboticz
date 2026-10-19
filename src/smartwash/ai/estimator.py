"""Cycle duration, energy and water estimates that learn the user's own machine.

Estimates start from the physics model and are corrected by per-programme factors learned from
cycles the machine has actually completed. The factors shrink towards a machine-wide factor, which
in turn shrinks towards 1.0, so one odd cycle cannot throw the estimates off.
"""

from __future__ import annotations

from typing import Any

from .. import physics
from ..schemas import CycleEstimate, CycleSettings, Program

MEASURES: tuple[str, ...] = ("duration", "energy", "water")
PRIOR_STRENGTH = 2.0
"""How many observed cycles the prior (physics model / machine-wide factor) is worth."""
RATIO_LIMITS = (0.2, 5.0)


class CycleEstimator:
    def __init__(self, calibration: dict[str, Any] | None = None) -> None:
        # {program: {measure: [count, sum_of_actual_over_physics_ratios]}}
        self._stats: dict[str, dict[str, list[float]]] = {
            program: {measure: [float(v[0]), float(v[1])] for measure, v in measures.items()}
            for program, measures in (calibration or {}).items()
        }

    def factor(self, program: Program, measure: str) -> float:
        total_n = sum(m.get(measure, [0, 0])[0] for m in self._stats.values())
        total_sum = sum(m.get(measure, [0, 0])[1] for m in self._stats.values())
        machine = (total_sum + PRIOR_STRENGTH) / (total_n + PRIOR_STRENGTH)
        n, ratio_sum = self._stats.get(program.value, {}).get(measure, [0.0, 0.0])
        return (ratio_sum + PRIOR_STRENGTH * machine) / (n + PRIOR_STRENGTH)

    def estimate(self, settings: CycleSettings, load_kg: float) -> CycleEstimate:
        base = physics.estimate_cycle(settings, load_kg)
        program = settings.program
        return CycleEstimate(
            duration_min=round(base.duration_min * self.factor(program, "duration"), 1),
            energy_kwh=round(base.energy_kwh * self.factor(program, "energy"), 3),
            water_l=round(base.water_l * self.factor(program, "water"), 1),
        )

    def observe(
        self,
        settings: CycleSettings,
        load_kg: float,
        *,
        duration_min: float | None = None,
        energy_kwh: float | None = None,
        water_l: float | None = None,
    ) -> None:
        """Learn from a completed cycle; pass whichever measurements are available."""
        base = physics.estimate_cycle(settings, load_kg)
        measured = {
            "duration": (duration_min, base.duration_min),
            "energy": (energy_kwh, base.energy_kwh),
            "water": (water_l, base.water_l),
        }
        stats = self._stats.setdefault(settings.program.value, {})
        for measure, (actual, predicted) in measured.items():
            if actual is None or actual <= 0 or predicted <= 0:
                continue
            ratio = min(max(actual / predicted, RATIO_LIMITS[0]), RATIO_LIMITS[1])
            entry = stats.setdefault(measure, [0.0, 0.0])
            entry[0] += 1
            entry[1] += ratio

    def observations(self) -> int:
        return int(sum(m.get("duration", [0, 0])[0] for m in self._stats.values()))

    def to_dict(self) -> dict[str, Any]:
        return {p: {m: list(v) for m, v in measures.items()} for p, measures in self._stats.items()}

    def summary(self) -> dict[str, Any]:
        """Learned correction factors per programme, for display."""
        return {
            program: {
                "cycles": int(measures.get("duration", [0, 0])[0]),
                **{
                    f"{m}_factor": round(self.factor(Program(program), m), 3)
                    for m in MEASURES
                    if m in measures
                },
            }
            for program, measures in self._stats.items()
        }
