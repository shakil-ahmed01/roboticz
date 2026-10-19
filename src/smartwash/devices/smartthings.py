"""Samsung washers via the SmartThings REST API (https://api.smartthings.com/v1).

Authenticate with a Personal Access Token (https://account.smartthings.com/tokens, scopes
``r:devices:*`` and ``x:devices:*``) or an OAuth access token. PATs created on or after
30 December 2024 are valid for 24 hours only, so long-running setups need a fresh token daily or an
OAuth / service integration.

Start, pause and stop use the standard ``washerOperatingState`` capability (or Samsung's
``samsungce.washerOperatingState`` where a model disables the standard one, which is best effort).
Cycle, temperature, spin and rinse use Samsung's ``samsungce.washerCycle`` and ``custom.washer*``
capabilities, whose values differ per model: map our programmes to your washer's cycle ids with
``cycle_map``. ``list_devices()`` shows the supported ids and the currently selected one, so select
a cycle on the washer and read its id to build the map. Some models accept the cycle command but
ignore it; check the first run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from ..schemas import (
    ACTIVE_STATUSES,
    CyclePhase,
    CycleSettings,
    MachineState,
    MachineStatus,
    Program,
)
from .base import (
    AdapterCapabilities,
    DeviceAuthError,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
    WashingMachineAdapter,
)
from .mapping import (
    choose_at_most,
    label_number,
    parse_spin_label,
    parse_temperature_label,
    program_candidates,
)

log = logging.getLogger(__name__)

STANDARD_STATE = "washerOperatingState"
SAMSUNG_STATE = "samsungce.washerOperatingState"
CYCLE = "samsungce.washerCycle"
TEMPERATURE = "custom.washerWaterTemperature"
SPIN = "custom.washerSpinLevel"
RINSES = "custom.washerRinseCycles"
REMOTE = "remoteControlStatus"
_WASHER_CAPABILITIES = frozenset({STANDARD_STATE, SAMSUNG_STATE, CYCLE})
_RAW_CAPABILITIES = (
    STANDARD_STATE,
    SAMSUNG_STATE,
    CYCLE,
    TEMPERATURE,
    SPIN,
    RINSES,
    REMOTE,
    "switch",
)

_JOB_PHASES = {
    "preWash": CyclePhase.WASH,
    "wash": CyclePhase.WASH,
    "aIWash": CyclePhase.WASH,
    "airWash": CyclePhase.WASH,
    "rinse": CyclePhase.RINSE,
    "aIRinse": CyclePhase.RINSE,
    "spin": CyclePhase.SPIN,
    "aISpin": CyclePhase.SPIN,
    "finish": CyclePhase.DONE,
}

# Our command -> (washerOperatingState.setMachineState argument, samsungce command)
_START = ("run", "start")
_PAUSE = ("pause", "pause")
_RESUME = ("run", "resume")
_STOP = ("stop", "cancel")


@dataclass(frozen=True)
class _Level:
    """A selectable cycle setting (water temperature or spin level)."""

    capability: str
    option: str
    """Key in a cycle's ``supportedOptions``."""
    supported: str
    """Attribute listing every value the washer supports."""
    command: str
    parse: Callable[[str], int | None]
    unit: str


_TEMPERATURE_LEVEL = _Level(
    TEMPERATURE,
    "waterTemperature",
    "supportedWasherWaterTemperature",
    "setWasherWaterTemperature",
    parse_temperature_label,
    "°C",
)
_SPIN_LEVEL = _Level(
    SPIN, "spinLevel", "supportedWasherSpinLevel", "setWasherSpinLevel", parse_spin_label, "rpm"
)


class SmartThingsConfig(BaseModel):
    token: str
    device_id: str | None = None
    """The washer's deviceId; the first washer on the account is used when omitted."""
    base_url: str = "https://api.smartthings.com/v1"
    cycle_map: dict[str, str] = Field(default_factory=dict)
    """Our programme (e.g. ``cotton``) -> cycle id (``Table_00_Course_5B``, or just ``5B``)."""
    start_preselected_program: bool = False
    """Start the cycle selected on the washer when a programme has no ``cycle_map`` entry."""
    min_poll_interval_s: float = 10.0
    timeout_s: float = 20.0

    @field_validator("cycle_map")
    @classmethod
    def _check_programs(cls, value: dict[str, str]) -> dict[str, str]:
        valid = {program.value for program in Program}
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ValueError(f"unknown programme(s) {unknown}; use one of {sorted(valid)}")
        return value


class _Status:
    """The ``main`` component of a device status, ignoring capabilities the model disables."""

    def __init__(self, main: dict[str, Any]) -> None:
        self.main = main
        disabled = self._value(main, "custom.disabledCapabilities", "disabledCapabilities")
        self.disabled = set(disabled or [])

    @staticmethod
    def _value(main: dict[str, Any], capability: str, attribute: str) -> Any:
        return ((main.get(capability) or {}).get(attribute) or {}).get("value")

    def has(self, capability: str) -> bool:
        return capability in self.main and capability not in self.disabled

    def get(self, capability: str, attribute: str) -> Any:
        return self._value(self.main, capability, attribute) if self.has(capability) else None

    def cycles(self) -> dict[str, dict[str, Any]]:
        """Supported cycle codes and the options each allows."""
        cycles: dict[str, dict[str, Any]] = {}
        for cycle in self.get(CYCLE, "supportedCycles") or []:
            if isinstance(cycle, dict) and cycle.get("cycle"):
                cycles[str(cycle["cycle"])] = cycle.get("supportedOptions") or {}
            elif isinstance(cycle, str):
                cycles[cycle] = {}
        return cycles

    def raw(self) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        for capability in _RAW_CAPABILITIES:
            if capability in self.main:
                raw[capability] = {
                    attribute: state.get("value")
                    for attribute, state in self.main[capability].items()
                    if attribute != "supportedCycles"
                }
        raw["supported_cycles"] = sorted(self.cycles())
        return raw


class SmartThingsAdapter(WashingMachineAdapter):
    name = "smartthings"

    def __init__(self, config: SmartThingsConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._base = config.base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._device_id = config.device_id
        self._device: dict[str, Any] = {}
        self._cached: tuple[float, MachineState] | None = None
        self._last_status: _Status | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        mapped = bool(self.config.cycle_map)
        return AdapterCapabilities(
            remote_start=mapped or self.config.start_preselected_program, select_program=mapped
        )

    # ----------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        with self._lock:
            if self._device_id:
                self._device = self._request("GET", f"/devices/{self._device_id}")
                if not _is_washer(self._device):
                    raise DeviceError(f"SmartThings device {self._device_id} is not a washer")
                return
            devices = self._devices()
            washers = [d for d in devices if _is_washer(d)]
            if not washers:
                found = ", ".join(str(d.get("label") or d.get("name")) for d in devices) or "none"
                raise DeviceError(f"No washer on this SmartThings account; found: {found}")
            self._device = washers[0]
            self._device_id = washers[0]["deviceId"]
            log.info("Using SmartThings washer %s (%s)", self._device_id, self._device.get("label"))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, **_describe_device(self._device, self._last_status)}

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock:
            washers = [d for d in self._devices() if _is_washer(d)]
            return [
                _describe_device(d, self._status(d["deviceId"], remember=False)) for d in washers
            ]

    # --------------------------------------------------------------------- state

    def get_state(self) -> MachineState:
        with self._lock:
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self.config.min_poll_interval_s:
                return self._cached[1]
            health = self._request("GET", f"/devices/{self._id()}/health")
            if health.get("state") == "OFFLINE":
                state = MachineState(status=MachineStatus.OFFLINE, raw={"health": health})
            else:
                state = _parse_state(self._status(), self.config.cycle_map)
            self._cached = (now, state)
            return state

    # ------------------------------------------------------------------ commands

    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        with self._lock:
            status = self._status()
            if status.get(REMOTE, "remoteControlEnabled") != "true":
                raise DeviceNotReady(
                    "Remote control is off. Turn on Smart Control on the washer, then try again"
                )
            if _parse_state(status, self.config.cycle_map).status in ACTIVE_STATUSES:
                raise DeviceNotReady("A cycle is already running on the washer")

            cycles = status.cycles()
            commands: list[tuple[str, str, list[Any]]] = []
            mapped = self._mapped_cycle(settings.program)
            if mapped is None:
                if not self.config.start_preselected_program:
                    raise UnsupportedOperation(
                        f"No SmartThings cycle is mapped for '{settings.program.value}'. Add "
                        f"smartthings.cycle_map.{settings.program.value}: <cycle id>; this washer "
                        f"supports {', '.join(sorted(cycles)) or 'no listed cycles'}"
                    )
                cycle_id = status.get(CYCLE, "washerCycle")
                log.warning("Starting the cycle selected on the washer (%s) as-is", cycle_id)
            else:
                cycle_id = _full_cycle_id(mapped, status)
                if cycles and _cycle_code(cycle_id) not in cycles:
                    raise UnsupportedOperation(
                        f"Cycle {mapped} is not supported by this washer; supported: "
                        + ", ".join(sorted(cycles))
                    )
                commands.append((CYCLE, "setWasherCycle", [cycle_id]))
            options = cycles.get(_cycle_code(cycle_id)) if cycle_id else None

            settings_commands = (
                _level_command(status, options, _TEMPERATURE_LEVEL, settings.temperature_c),
                _level_command(status, options, _SPIN_LEVEL, settings.spin_rpm),
                _rinse_command(status, options) if settings.extra_rinse else None,
            )
            commands.extend(command for command in settings_commands if command is not None)
            if settings.prewash:
                log.warning("SmartThings has no prewash option; ignoring it")
            commands.append(_state_command(status, _START))

            for capability, command, arguments in commands:
                self._execute(capability, command, arguments)
            self._cached = None
            log.info("Started SmartThings cycle %s with %s", cycle_id, commands)

    def pause(self) -> None:
        self._operate(_PAUSE)

    def resume(self) -> None:
        self._operate(_RESUME)

    def stop(self) -> None:
        self._operate(_STOP)

    # ------------------------------------------------------------------- helpers

    def _operate(self, command: tuple[str, str]) -> None:
        with self._lock:
            self._execute(*_state_command(self._status(), command))
            self._cached = None

    def _mapped_cycle(self, program: Program) -> str | None:
        for candidate in program_candidates(program):
            if candidate.value in self.config.cycle_map:
                return self.config.cycle_map[candidate.value]
        return None

    def _execute(self, capability: str, command: str, arguments: list[Any]) -> None:
        payload: dict[str, Any] = {
            "component": "main",
            "capability": capability,
            "command": command,
        }
        if arguments:
            payload["arguments"] = arguments
        result = self._request(
            "POST", f"/devices/{self._id()}/commands", json={"commands": [payload]}
        )
        for outcome in result.get("results", []):
            if outcome.get("status") == "FAILED":
                raise DeviceError(f"The washer rejected {capability}.{command} {arguments}")

    def _status(self, device_id: str | None = None, *, remember: bool = True) -> _Status:
        payload = self._request("GET", f"/devices/{device_id or self._id()}/status")
        status = _Status((payload.get("components") or {}).get("main") or {})
        if remember:
            self._last_status = status
        return status

    def _devices(self) -> list[dict[str, Any]]:
        devices: list[dict[str, Any]] = []
        url: str | None = f"{self._base}/devices"
        while url:
            page = self._request("GET", url)
            devices.extend(page.get("items", []))
            url = ((page.get("_links") or {}).get("next") or {}).get("href")
        return devices

    def _id(self) -> str:
        if not self._device_id:
            self.connect()
        return str(self._device_id)

    def _request(self, method: str, path: str, *, json: Any = None) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self._base}{path}"
        try:
            response = self._client.request(
                method,
                url,
                headers={"Authorization": f"Bearer {self.config.token}"},
                json=json,
            )
        except httpx.HTTPError as exc:
            raise DeviceError(f"SmartThings request failed: {exc}") from exc
        if not response.is_success:
            raise _error(response)
        return _json(response)


def _is_washer(device: dict[str, Any]) -> bool:
    for component in device.get("components", []):
        if component.get("id") == "main":
            categories = {c.get("name") for c in component.get("categories", [])}
            capabilities = {c.get("id") for c in component.get("capabilities", [])}
            return "Washer" in categories or bool(capabilities & _WASHER_CAPABILITIES)
    return False


def _describe_device(device: dict[str, Any], status: _Status | None) -> dict[str, Any]:
    description = {
        "id": device.get("deviceId"),
        "name": device.get("label") or device.get("name"),
        "manufacturer": device.get("manufacturerName"),
        "model": (device.get("ocf") or {}).get("modelNumber") or device.get("presentationId"),
    }
    if status is not None:
        description["current_cycle"] = status.get(CYCLE, "washerCycle")
        description["supported_cycles"] = sorted(status.cycles())
    return description


def _parse_state(status: _Status, cycle_map: dict[str, str]) -> MachineState:
    machine = status.get(STANDARD_STATE, "machineState")
    operating = status.get(SAMSUNG_STATE, "operatingState")
    job = status.get(STANDARD_STATE, "washerJobState") or status.get(
        SAMSUNG_STATE, "washerJobState"
    )
    if machine == "run" or operating == "running":
        machine_status = MachineStatus.RUNNING
    elif machine == "pause" or operating == "paused":
        machine_status = MachineStatus.PAUSED
    else:
        machine_status = MachineStatus.IDLE
    if job == "delayWash":
        machine_status = MachineStatus.DELAYED_START
    elif job == "finish":
        machine_status = MachineStatus.FINISHED
    phase = _JOB_PHASES.get(job)
    if phase is None and machine_status is MachineStatus.IDLE:
        phase = CyclePhase.IDLE

    cycle = status.get(CYCLE, "washerCycle")
    program = None
    if cycle:
        program = next(
            (Program(p) for p, cid in cycle_map.items() if _cycle_code(cid) == _cycle_code(cycle)),
            None,
        )
    active = machine_status in ACTIVE_STATUSES
    temperature = status.get(TEMPERATURE, "washerWaterTemperature")
    spin = status.get(SPIN, "washerSpinLevel")
    remote = status.get(REMOTE, "remoteControlEnabled")
    return MachineState(
        status=machine_status,
        phase=phase,
        program=program,
        vendor_program=cycle,
        temperature_c=label_number(temperature) if temperature else None,
        spin_rpm=label_number(spin) if spin else None,
        progress_pct=status.get(SAMSUNG_STATE, "progress") if active else None,
        remaining_min=_remaining_min(status) if active else None,
        remote_start_allowed=None if remote is None else remote == "true",
        raw=status.raw(),
    )


def _remaining_min(status: _Status) -> float | None:
    remaining = status.get(SAMSUNG_STATE, "remainingTime")
    if isinstance(remaining, (int, float)):
        return float(remaining)
    completion = status.get(STANDARD_STATE, "completionTime")
    if not completion:
        return None
    try:
        finish = datetime.fromisoformat(str(completion).replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(max((finish - datetime.now(timezone.utc)).total_seconds(), 0.0) / 60.0, 1)


def _full_cycle_id(value: str, status: _Status) -> str:
    if value.startswith("Table_"):
        return value
    table = (status.get(CYCLE, "referenceTable") or {}).get("id")
    if not table:
        raise UnsupportedOperation(
            f"Cannot expand cycle id {value!r}: the washer reports no reference table; "
            "use the full id such as Table_00_Course_5B"
        )
    return f"{table}_Course_{value}"


def _cycle_code(cycle_id: str) -> str:
    return cycle_id.rsplit("_Course_", 1)[-1]


def _level_command(
    status: _Status, cycle_options: dict[str, Any] | None, level: _Level, requested: int
) -> tuple[str, str, list[Any]] | None:
    """Command setting ``level`` no higher than ``requested``; ``None`` if it cannot be set."""
    if not status.has(level.capability):
        log.warning("The washer cannot set %s remotely; its default applies", level.option)
        return None
    spec = (cycle_options or {}).get(level.option)
    if spec is not None:
        labels = list(spec.get("options") or [])
    else:
        labels = list(status.get(level.capability, level.supported) or [])
    labels = [label for label in labels if label != "none"]
    if not labels:
        default = (spec or {}).get("default")
        fixed = level.parse(default) if default else None
        if fixed is not None and fixed > requested:
            raise UnsupportedOperation(
                f"The selected cycle always uses {default} {level.unit}, "
                f"above the requested {requested}"
            )
        log.warning("%s is fixed for this cycle; the washer's default applies", level.option)
        return None
    label = choose_at_most(requested, labels, level.parse)
    if label is None:
        raise UnsupportedOperation(
            f"The selected cycle cannot run at or below {requested} {level.unit}; allowed: "
            + ", ".join(labels)
        )
    return (level.capability, level.command, [label])


def _rinse_command(
    status: _Status, cycle_options: dict[str, Any] | None
) -> tuple[str, str, list[Any]] | None:
    """One rinse more than the cycle's default."""
    if not status.has(RINSES):
        log.warning("The washer cannot add a rinse remotely; ignoring extra_rinse")
        return None
    spec = (cycle_options or {}).get("rinseCycle") or {}
    options = spec.get("options") or status.get(RINSES, "supportedWasherRinseCycles") or []
    default = spec.get("default") or status.get(RINSES, "washerRinseCycles")
    try:
        target = str(int(default) + 1)
    except (TypeError, ValueError):
        target = None
    if target not in options:
        log.warning("Cannot add a rinse to the selected cycle (options %s); ignoring it", options)
        return None
    return (RINSES, "setWasherRinseCycles", [target])


def _state_command(status: _Status, command: tuple[str, str]) -> tuple[str, str, list[Any]]:
    standard, samsung = command
    if status.has(STANDARD_STATE):
        return (STANDARD_STATE, "setMachineState", [standard])
    if status.has(SAMSUNG_STATE):
        return (SAMSUNG_STATE, samsung, [])
    raise UnsupportedOperation("This washer exposes no remote operating-state control")


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _error(response: httpx.Response) -> DeviceError:
    error = _json(response).get("error") or {}
    message = error.get("message") or response.text or response.reason_phrase
    details = "; ".join(d["message"] for d in error.get("details") or [] if d.get("message"))
    text = f"SmartThings returned {response.status_code}: {message}"
    if error.get("code"):
        text += f" ({error['code']})"
    if details:
        text += f": {details}"
    if response.status_code == 401:
        return DeviceAuthError(
            text + ". Personal access tokens created since 30 Dec 2024 expire after 24 hours"
        )
    if response.status_code == 403:
        return DeviceAuthError(text + ". The token needs the r:devices:* and x:devices:* scopes")
    if response.status_code in (409, 422):
        return DeviceNotReady(text)
    return DeviceError(text)
