"""User feedback: the household's own choices, which personalise the recommender."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..knowledge import enforce_care_constraints
from ..schemas import CycleSettings, LoadProfile

log = logging.getLogger(__name__)


class FeedbackRecord(BaseModel):
    profile: LoadProfile
    settings: CycleSettings
    """The settings the user actually chose (or rated)."""
    rating: int | None = Field(default=None, ge=1, le=5)
    """1-5 stars for the result; ``None`` for an implicit preference (the user overrode us)."""
    source: str = "explicit"
    timestamp: float = Field(default_factory=time.time)


class FeedbackStore:
    """Append-only JSON-lines file of feedback records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, record: FeedbackRecord) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")

    def load(self) -> list[FeedbackRecord]:
        if not self.path.exists():
            return []
        records = []
        with self._lock, self.path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    records.append(FeedbackRecord.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValidationError) as exc:
                    log.warning(
                        "Skipping malformed feedback line %d in %s: %s", number, self.path, exc
                    )
        return records

    def __len__(self) -> int:
        return len(self.load())


def training_examples(
    records: list[FeedbackRecord],
) -> list[tuple[LoadProfile, CycleSettings, float]]:
    """Feedback worth learning from, made safe by the care guardrails, with relative weights.

    Choices rated 4-5 stars (or implicit overrides) count fully, 3 stars half; poorly rated
    outcomes are not learned from.
    """
    examples = []
    for record in records:
        if record.rating is not None and record.rating <= 2:
            continue
        weight = 0.5 if record.rating == 3 else 1.0
        safe, _ = enforce_care_constraints(record.profile, record.settings)
        examples.append((record.profile, safe, weight))
    return examples
