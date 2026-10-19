"""Vendor-neutral helpers the cloud adapters share to map programmes, temperatures and spins."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from ..schemas import Program
from .base import pick_at_most

PROGRAM_FALLBACKS: dict[Program, tuple[Program, ...]] = {
    Program.COTTON: (Program.COTTON,),
    Program.ECO_40_60: (Program.ECO_40_60, Program.MIXED),
    Program.SYNTHETICS: (Program.SYNTHETICS,),
    Program.MIXED: (Program.MIXED, Program.SYNTHETICS),
    Program.WOOL: (Program.WOOL,),
    Program.DELICATES: (Program.DELICATES, Program.WOOL),
    Program.QUICK: (Program.QUICK,),
    Program.SPORTSWEAR: (Program.SPORTSWEAR, Program.SYNTHETICS),
    Program.BEDDING: (Program.BEDDING, Program.MIXED),
    Program.TOWELS: (Program.TOWELS, Program.COTTON),
    Program.DARK_WASH: (Program.DARK_WASH, Program.SYNTHETICS),
    Program.HYGIENE: (Program.HYGIENE, Program.COTTON),
}
"""Programmes to try, in order, when a machine lacks the requested one.

A substitute never has stronger drum action than the programme it replaces; temperature and spin
are capped separately with ``pick_at_most``.
"""

# Word labels mapped to a conservative upper bound, so "never exceed the requested value" holds
# even though vendors do not publish exact figures for them.
_TEMPERATURE_WORDS = {
    "ulextrahot": 90,
    "extrahot": 90,
    "semihot": 50,
    "ecowarm": 40,
    "tapcold": 15,
    "ulcold": 15,
    "ulwarm": 40,
    "ulhot": 60,
    "cold": 15,
    "cool": 30,
    "warm": 40,
    "hot": 60,
}
_SPIN_WORDS = {
    "extrahigh": 1600,
    "extralow": 400,
    "ulmedium": 1000,
    "delicate": 600,
    "nospin": 0,
    "ulhigh": 1400,
    "medium": 1000,
    "uloff": 0,
    "ullow": 600,
    "high": 1400,
    "low": 600,
    "off": 0,
}
_TRAILING_NUMBER = re.compile(r"(\d+)(?:c|rpm)?$")


def program_candidates(program: Program) -> tuple[Program, ...]:
    return PROGRAM_FALLBACKS.get(program, (program,))


def _normalise(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def label_number(label: str) -> int | None:
    """The number a label ends with (``"GC40"`` -> 40, ``"RPM1200"`` -> 1200), if any."""
    match = _TRAILING_NUMBER.search(_normalise(label))
    return int(match.group(1)) if match else None


def _parse(label: str, words: dict[str, int]) -> int | None:
    key = _normalise(label)
    number = label_number(key)
    if number is not None:
        return number
    for word in sorted(words, key=len, reverse=True):
        if key.endswith(word):
            return words[word]
    return None


def parse_temperature_label(label: str) -> int | None:
    """Wash temperature (°C, upper bound) a vendor label stands for, or ``None`` if unknown."""
    return _parse(label, _TEMPERATURE_WORDS)


def parse_spin_label(label: str) -> int | None:
    """Spin speed (rpm, upper bound) a vendor label stands for; ``None`` for rinse-hold/unknown."""
    if _normalise(label).endswith("rinsehold"):
        return None
    return _parse(label, _SPIN_WORDS)


def choose_at_most(
    requested: int, labels: Iterable[str], parse: Callable[[str], int | None]
) -> str | None:
    """The label with the highest value not above ``requested`` (``None`` if all are higher)."""
    values = {label: parse(label) for label in labels}
    best = pick_at_most(requested, [v for v in values.values() if v is not None])
    if best is None:
        return None
    return [label for label, value in values.items() if value == best][-1]
