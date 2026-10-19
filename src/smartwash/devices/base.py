"""The interface every washing-machine connection implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..schemas import FAULT_DETECTION_FIELDS, CycleSettings, MachineState, TelemetrySample


class DeviceError(RuntimeError):
    """Talking to the machine failed (network error, bad response, ...)."""


class DeviceAuthError(DeviceError):
    """The machine's cloud rejected our credentials."""


class DeviceNotReady(DeviceError):
    """The machine refused the command in its current state (door open, remote start off, busy)."""


class UnsupportedOperation(DeviceError):
    """The machine or its connection cannot do this."""


@dataclass(frozen=True)
class AdapterCapabilities:
    remote_start: bool = True
    select_program: bool = True
    """Whether programme, temperature and spin can be chosen remotely."""
    pause_resume: bool = True
    stop: bool = True
    telemetry_fields: frozenset[str] = field(default_factory=frozenset)
    """Which ``TelemetrySample`` fields this connection delivers."""

    @property
    def fault_detection(self) -> bool:
        return FAULT_DETECTION_FIELDS.issubset(self.telemetry_fields)


class WashingMachineAdapter(ABC):
    """A connection to one washing machine.

    Implementations must be safe to call from several threads (the controller's monitoring loop and
    the API share one adapter); the controller serialises calls, so a simple lock is enough.
    """

    name: ClassVar[str] = "base"

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    def connect(self) -> None:
        """Open connections and resolve the device. Called once before first use."""

    def close(self) -> None:
        """Release network resources."""

    @abstractmethod
    def get_state(self) -> MachineState: ...

    @abstractmethod
    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        """Select the programme and options, then start.

        ``load_kg`` is only a hint for the simulator; real machines weigh the load themselves.
        """

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def resume(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def read_telemetry(self) -> list[TelemetrySample]:
        """Return the telemetry received since the previous call, oldest first."""
        return []

    def describe(self) -> dict[str, Any]:
        """Identify the connected machine (model, id, ...)."""
        return {"adapter": self.name}

    def list_devices(self) -> list[dict[str, Any]]:
        """Washing machines reachable with the configured credentials."""
        return [self.describe()]

    def __enter__(self) -> WashingMachineAdapter:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def pick_at_most(requested: int, available: Iterable[int]) -> int | None:
    """Largest available value not above ``requested``, or ``None`` if every value is higher.

    Adapters use this to map temperature and spin onto what a machine supports without ever
    exceeding what the fabric-care rules allowed.
    """
    candidates = [value for value in available if value <= requested]
    return max(candidates) if candidates else None
