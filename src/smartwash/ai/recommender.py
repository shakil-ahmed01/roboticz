"""The cycle recommender: which programme, temperature, spin and options suit a load.

One random forest per setting learns from expert-labelled synthetic loads plus the household's own
feedback. Its raw output always passes through the fabric-care guardrails before it is used, so
personalisation can change preferences but never make a cycle unsafe for the fabric.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..knowledge import (
    detergent_dose_ml,
    enforce_care_constraints,
    expert_settings,
    explain,
    load_warnings,
    program_suits,
)
from ..schemas import CycleSettings, LoadProfile, Program, ProgramOption, Recommendation
from .estimator import CycleEstimator
from .features import encode_profiles

HEADS: tuple[str, ...] = ("program", "temperature_c", "spin_rpm", "prewash", "extra_rinse")
GUARDRAIL_CONFIDENCE_PENALTY = 0.85


def _targets(settings: CycleSettings) -> tuple[str, int, int, int, int]:
    return (
        settings.program.value,
        settings.temperature_c,
        settings.spin_rpm,
        int(settings.prewash),
        int(settings.extra_rinse),
    )


class CycleRecommender:
    def __init__(
        self, *, n_estimators: int = 60, min_samples_leaf: int = 3, random_state: int = 0
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.heads: dict[str, RandomForestClassifier] = {}
        self.version = "untrained"
        self.metrics: dict[str, Any] = {}

    def fit(
        self,
        profiles: Sequence[LoadProfile],
        settings: Sequence[CycleSettings],
        sample_weight: np.ndarray | None = None,
    ) -> CycleRecommender:
        X = encode_profiles(profiles)
        targets = list(zip(*(_targets(s) for s in settings), strict=True))
        for head, y in zip(HEADS, targets, strict=True):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=-1,
                random_state=self.random_state,
            )
            model.fit(X, np.asarray(y), sample_weight=sample_weight)
            model.n_jobs = 1  # recommendations are made one load at a time
            self.heads[head] = model
        return self

    def _probabilities(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not self.heads:
            raise RuntimeError("the recommender has not been trained")
        return {head: (m.classes_, m.predict_proba(X)) for head, m in self.heads.items()}

    @staticmethod
    def _decode(probs: dict[str, tuple[np.ndarray, np.ndarray]], row: int) -> CycleSettings:
        def best(head: str) -> Any:
            classes, p = probs[head]
            return classes[int(np.argmax(p[row]))]

        return CycleSettings(
            program=Program(best("program")),
            temperature_c=int(best("temperature_c")),
            spin_rpm=int(best("spin_rpm")),
            prewash=bool(best("prewash")),
            extra_rinse=bool(best("extra_rinse")),
        )

    def predict(self, profiles: Sequence[LoadProfile]) -> list[CycleSettings]:
        """The model's raw choices, before guardrails."""
        probs = self._probabilities(encode_profiles(profiles))
        return [self._decode(probs, i) for i in range(len(profiles))]

    def recommend(
        self, profile: LoadProfile, *, estimator: CycleEstimator, capacity_kg: float = 8.0
    ) -> Recommendation:
        probs = self._probabilities(encode_profiles([profile]))
        raw = self._decode(probs, 0)
        settings, adjustments = enforce_care_constraints(profile, raw)

        top = [float(probs[h][1][0].max()) for h in ("program", "temperature_c", "spin_rpm")]
        confidence = float(np.prod(top) ** (1.0 / len(top)))
        if adjustments:
            confidence *= GUARDRAIL_CONFIDENCE_PENALTY

        classes, program_probs = probs["program"]
        alternatives = sorted(
            (
                ProgramOption(program=Program(c), probability=round(float(p), 3))
                for c, p in zip(classes, program_probs[0], strict=True)
                if p >= 0.05
                and Program(c) is not settings.program
                and program_suits(profile, Program(c))
            ),
            key=lambda option: -option.probability,
        )[:3]

        estimate = estimator.estimate(settings, profile.weight_kg)
        warnings = load_warnings(profile, settings, capacity_kg)
        budget = profile.max_duration_min
        if budget is not None and estimate.duration_min > budget:
            warnings.append(
                f"This cycle takes about {estimate.duration_min:.0f} min, longer than your "
                f"{budget} min time budget; a shorter programme would not suit this load."
            )
        return Recommendation(
            settings=settings,
            detergent_ml=detergent_dose_ml(profile),
            estimate=estimate,
            confidence=round(confidence, 3),
            alternatives=alternatives,
            explanation=explain(profile, settings),
            adjustments=adjustments,
            warnings=warnings,
            model_version=self.version,
        )

    def evaluate(
        self, profiles: Sequence[LoadProfile], labels: Sequence[CycleSettings]
    ) -> dict[str, Any]:
        """Accuracy against held-out expert labels, plus how often the guardrails had to step in."""
        predictions = self.predict(profiles)
        consensus = [expert_settings(p) for p in profiles]
        metrics: dict[str, Any] = {"test_samples": len(profiles)}
        for i, head in enumerate(HEADS):
            metrics[f"{head}_accuracy"] = round(
                float(
                    np.mean(
                        [
                            _targets(p)[i] == _targets(t)[i]
                            for p, t in zip(predictions, labels, strict=True)
                        ]
                    )
                ),
                4,
            )
            metrics[f"{head}_agreement_with_expert"] = round(
                float(
                    np.mean(
                        [
                            _targets(p)[i] == _targets(c)[i]
                            for p, c in zip(predictions, consensus, strict=True)
                        ]
                    )
                ),
                4,
            )
        metrics["exact_match_with_expert"] = round(
            float(np.mean([p == c for p, c in zip(predictions, consensus, strict=True)])), 4
        )
        metrics["guardrail_intervention_rate"] = round(
            float(
                np.mean(
                    [
                        bool(enforce_care_constraints(profile, p)[1])
                        for profile, p in zip(profiles, predictions, strict=True)
                    ]
                )
            ),
            4,
        )
        return metrics
