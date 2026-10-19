"""Any washer integrated into Home Assistant, through the HA REST API.

Works with the LG ThinQ, Home Connect, SmartThings, Miele, Electrolux, Whirlpool, GE and other
integrations: point ``status_entity`` (plus optional entities) at the washer's entities and describe
the service calls that start, pause, resume and stop it. Authenticate with a long-lived access token
(HA profile -> Security). docs/cloud-adapters.md has complete LG ThinQ and Home Connect examples.

Service-call ``data`` may contain placeholders that are filled in when a cycle starts:
``{program}``, ``{temperature}`` and ``{spin}`` (the integration's own option values, from
``program_map`` and the select entities' options or ``temperature_map``/``spin_map``) and
``{temperature_c}`` / ``{spin_rpm}`` (plain numbers).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    TelemetrySample,
)
from .base import (
    AdapterCapabilities,
    DeviceAuthError,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
    WashingMachineAdapter,
    pick_at_most,
)
from .mapping import (
    choose_at_most,
    label_number,
    parse_spin_label,
    parse_temperature_label,
    program_candidates,
)

log = logging.getLogger(__name__)

_S = MachineStatus
DEFAULT_STATUS_MAP: dict[str, MachineStatus] = {
    # Home Connect: sensor.<washer>_operation_state
    "inactive": _S.IDLE,
    "ready": _S.IDLE,
    "delayedstart": _S.DELAYED_START,
    "run": _S.RUNNING,
    "pause": _S.PAUSED,
    "actionrequired": _S.PAUSED,
    "finished": _S.FINISHED,
    "error": _S.ERROR,
    "aborting": _S.RUNNING,
    # LG ThinQ: sensor.<washer>_current_status
    "power_off": _S.IDLE,
    "initial": _S.IDLE,
    "sleep": _S.IDLE,
    "frozen_prevent_initial": _S.IDLE,
    "reserved": _S.DELAYED_START,
    "detecting": _S.RUNNING,
    "running": _S.RUNNING,
    "prewash": _S.RUNNING,
    "soaking": _S.RUNNING,
    "rinsing": _S.RUNNING,
    "spinning": _S.RUNNING,
    "drying": _S.RUNNING,
    "steam_softening": _S.RUNNING,
    "refreshing": _S.RUNNING,
    "dispensing": _S.RUNNING,
    "detergent_amount": _S.RUNNING,
    "add_drain": _S.RUNNING,
    "cooling": _S.RUNNING,
    "wrinkle_care": _S.RUNNING,
    "change_condition": _S.RUNNING,
    "display_loadsize": _S.RUNNING,
    "frozen_prevent_running": _S.RUNNING,
    "frozen_prevent_pause": _S.PAUSED,
    "rinse_hold": _S.PAUSED,
    "end": _S.FINISHED,
    "complete": _S.FINISHED,
    # SmartThings: sensor.<washer>_machine_state
    "stop": _S.IDLE,
    # Miele: sensor.<washer>_status
    "off": _S.IDLE,
    "on": _S.IDLE,
    "idle": _S.IDLE,
    "programmed": _S.IDLE,
    "waiting_to_start": _S.DELAYED_START,
    "in_use": _S.RUNNING,
    "program_ended": _S.FINISHED,
    "failure": _S.ERROR,
    "program_interrupted": _S.ERROR,
    "service": _S.ERROR,
    "not_connected": _S.OFFLINE,
    # Home Assistant itself
    "unavailable": _S.OFFLINE,
    "unknown": _S.OFFLINE,
}
"""State of ``status_entity`` (lower-cased) -> status; ``status_map`` extends or overrides it."""

_P = CyclePhase
DEFAULT_PHASE_MAP: dict[str, CyclePhase] = {
    "prewash": _P.WASH,
    "pre_wash": _P.WASH,
    "soaking": _P.WASH,
    "soak": _P.WASH,
    "wash": _P.WASH,
    "main_wash": _P.WASH,
    "ai_wash": _P.WASH,
    "air_wash": _P.WASH,
    "rinsing": _P.RINSE,
    "rinse": _P.RINSE,
    "rinses": _P.RINSE,
    "interim_rinse": _P.RINSE,
    "final_rinse": _P.RINSE,
    "ai_rinse": _P.RINSE,
    "spinning": _P.SPIN,
    "spin": _P.SPIN,
    "ai_spin": _P.SPIN,
    "drain": _P.DRAIN,
    "add_drain": _P.DRAIN,
    "end": _P.DONE,
    "finish": _P.DONE,
    "finished": _P.DONE,
    "program_ended": _P.DONE,
}
"""State of ``phase_entity`` (or of ``status_entity``) -> cycle phase; ``phase_map`` extends it."""

_TRUE = {"on", "true", "open", "opened", "yes", "enabled", "allowed"}
_FALSE = {"off", "false", "closed", "locked", "no", "disabled", "not_allowed"}
_MINUTES_PER_UNIT = {"min": 1.0, "s": 1 / 60, "h": 60.0, "d": 1440.0}


class ServiceCall(BaseModel):
    service: str
    """``domain.service``, e.g. ``button.press``."""
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("service")
    @classmethod
    def _check_service(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", value):
            raise ValueError("service must look like 'domain.service'")
        return value


class HomeAssistantConfig(BaseModel):
    url: str = "http://homeassistant.local:8123"
    token: str
    status_entity: str
    phase_entity: str | None = None
    remaining_entity: str | None = None
    """Remaining time in min/s/h, or a finish-time timestamp sensor."""
    progress_entity: str | None = None
    door_entity: str | None = None
    remote_start_entity: str | None = None
    power_entity: str | None = None
    """Live power sensor (e.g. a smart plug); enables power telemetry while running."""
    program_select_entity: str | None = None
    temperature_select_entity: str | None = None
    spin_select_entity: str | None = None
    program_map: dict[str, str] = Field(default_factory=dict)
    """Our programme (e.g. ``cotton``) -> the integration's option value."""
    temperature_map: dict[int, str] = Field(default_factory=dict)
    """°C -> option value; when empty, the temperature select's options are parsed."""
    spin_map: dict[int, str] = Field(default_factory=dict)
    """rpm -> option value; when empty, the spin select's options are parsed."""
    status_map: dict[str, MachineStatus] = Field(default_factory=dict)
    phase_map: dict[str, CyclePhase] = Field(default_factory=dict)
    start: ServiceCall | None = None
    pause: ServiceCall | None = None
    resume: ServiceCall | None = None
    stop: ServiceCall | None = None
    prewash: ServiceCall | None = None
    extra_rinse: ServiceCall | None = None
    start_preselected_program: bool = False
    """When no programme selection is configured, start whatever programme is set on the washer."""
    allow_uncontrolled_levels: bool = False
    """Start even when the temperature or spin cannot be set through Home Assistant, so the
    washer's own setting applies. Only enable this if you always set them on the machine."""
    min_poll_interval_s: float = 0.0
    timeout_s: float = 15.0
    verify_ssl: bool = True

    @field_validator("program_map")
    @classmethod
    def _check_programs(cls, value: dict[str, str]) -> dict[str, str]:
        valid = {program.value for program in Program}
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ValueError(f"unknown programme(s) {unknown}; use one of {sorted(valid)}")
        return value


class HomeAssistantAdapter(WashingMachineAdapter):
    name = "home_assistant"

    def __init__(self, config: HomeAssistantConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._base = config.url.rstrip("/")
        self._client = client or httpx.Client(timeout=config.timeout_s, verify=config.verify_ssl)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._status_map = {
            **DEFAULT_STATUS_MAP,
            **{k.lower(): v for k, v in config.status_map.items()},
        }
        self._phase_map = {
            **DEFAULT_PHASE_MAP,
            **{k.lower(): v for k, v in config.phase_map.items()},
        }
        self._cached: tuple[float, MachineState] | None = None
        self._last_state: MachineState | None = None
        self._friendly_name: str | None = None
        self._warned: set[str] = set()

    @property
    def capabilities(self) -> AdapterCapabilities:
        config = self.config
        chooses = self._chooses_program()
        levels = self._controls("temperature") and self._controls("spin")
        return AdapterCapabilities(
            remote_start=config.start is not None
            and (chooses or config.start_preselected_program)
            and (levels or config.allow_uncontrolled_levels),
            select_program=chooses and levels,
            pause_resume=config.pause is not None and config.resume is not None,
            stop=config.stop is not None,
            telemetry_fields=frozenset({"power_w"}) if config.power_entity else frozenset(),
        )

    # ----------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        with self._lock, _unexpected_payloads():
            self._request("GET", "/api/")
            status = self._entity(self.config.status_entity, required=True)
            self._friendly_name = (status.get("attributes") or {}).get("friendly_name")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "url": self._base,
            "status_entity": self.config.status_entity,
            "name": self._friendly_name,
        }

    def list_devices(self) -> list[dict[str, Any]]:
        """Entities that look washer-related, to help fill in the configuration."""
        with self._lock, _unexpected_payloads():
            entities = self._request("GET", "/api/states") or []
        found = []
        for entity in entities:
            attributes = entity.get("attributes") or {}
            name = str(attributes.get("friendly_name") or "")
            if "wash" in entity.get("entity_id", "").lower() or "wash" in name.lower():
                found.append(
                    {
                        "id": entity.get("entity_id"),
                        "name": name or None,
                        "state": entity.get("state"),
                        **({"options": attributes["options"]} if "options" in attributes else {}),
                    }
                )
        return found

    # --------------------------------------------------------------------- state

    def get_state(self) -> MachineState:
        with self._lock, _unexpected_payloads():
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self.config.min_poll_interval_s:
                return self._cached[1]
            state = self._read_state()
            self._cached = (now, state)
            self._last_state = state
            return state

    def _read_state(self) -> MachineState:
        config = self.config
        status_entity = self._entity(config.status_entity, required=True)
        value = str(status_entity.get("state", "")).lower()
        raw: dict[str, Any] = {config.status_entity: status_entity.get("state")}
        status = self._status_map.get(value)
        if status is None:
            self._warn_once(
                value, "Unknown washer state %r; add it to home_assistant.status_map", value
            )
            status = MachineStatus.IDLE

        phase_value = value
        if config.phase_entity:
            phase_value = str(self._state_of(config.phase_entity, raw) or "").lower()
        phase = self._phase_map.get(phase_value)
        if status is MachineStatus.FINISHED:
            phase = CyclePhase.DONE
        elif phase is None and status is MachineStatus.IDLE:
            phase = CyclePhase.IDLE

        program_option = self._state_of(config.program_select_entity, raw)
        temperature = self._state_of(config.temperature_select_entity, raw)
        spin = self._state_of(config.spin_select_entity, raw)
        active = status in ACTIVE_STATUSES
        return MachineState(
            status=status,
            phase=phase,
            program=self._program_for(program_option),
            vendor_program=program_option,
            temperature_c=label_number(temperature) if temperature else None,
            spin_rpm=label_number(spin) if spin else None,
            progress_pct=_to_float(self._state_of(config.progress_entity, raw)) if active else None,
            remaining_min=self._remaining_min(raw) if active else None,
            door_open=_to_bool(self._state_of(config.door_entity, raw)),
            remote_start_allowed=_to_bool(self._state_of(config.remote_start_entity, raw)),
            error_code=value if status is MachineStatus.ERROR else None,
            raw=raw,
        )

    def read_telemetry(self) -> list[TelemetrySample]:
        """One power reading per call while running (phase falls back to ``wash`` if unknown)."""
        if not self.config.power_entity:
            return []
        with self._lock, _unexpected_payloads():
            state = self._last_state
            if state is None or state.status is not MachineStatus.RUNNING:
                return []
            entity = self._entity(self.config.power_entity)
            power = _to_float(entity.get("state")) if entity else None
            if power is None:
                return []
            unit = (entity.get("attributes") or {}).get("unit_of_measurement")
            if unit == "kW":
                power *= 1000.0
            return [
                TelemetrySample(
                    timestamp=time.time(), phase=state.phase or CyclePhase.WASH, power_w=power
                )
            ]

    # ------------------------------------------------------------------ commands

    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        with self._lock, _unexpected_payloads():
            config = self.config
            if config.start is None:
                raise UnsupportedOperation("Configure home_assistant.start to start the washer")
            state = self._read_state()
            if state.status in ACTIVE_STATUSES:
                raise DeviceNotReady("A cycle is already running on the washer")
            if state.status in (MachineStatus.OFFLINE, MachineStatus.ERROR):
                raise DeviceNotReady(f"Home Assistant reports the washer as {state.status.value}")
            if state.remote_start_allowed is False:
                raise DeviceNotReady("Remote start is disabled; enable it on the washer first")
            if state.door_open:
                raise DeviceNotReady("Close the washer door first")

            values: dict[str, Any] = {
                "temperature_c": settings.temperature_c,
                "spin_rpm": settings.spin_rpm,
            }
            program_option = None
            if self._chooses_program():
                program_option = self._program_option(settings.program)
                if program_option is None:
                    raise UnsupportedOperation(
                        f"No program_map entry for '{settings.program.value}' or a gentler "
                        f"substitute; mapped: {', '.join(config.program_map) or 'none'}"
                    )
                values["program"] = program_option
            elif not config.start_preselected_program:
                raise UnsupportedOperation(
                    "No way to choose the programme is configured: set program_map plus "
                    "program_select_entity or a {program} placeholder, or set "
                    "start_preselected_program: true to start what is selected on the washer"
                )

            # Work out every option first, so nothing on the washer changes if one is refused.
            selects: list[tuple[str, str]] = []
            for kind, entity, mapping, requested, parse, unit in (
                (
                    "temperature",
                    config.temperature_select_entity,
                    config.temperature_map,
                    settings.temperature_c,
                    parse_temperature_label,
                    "°C",
                ),
                (
                    "spin",
                    config.spin_select_entity,
                    config.spin_map,
                    settings.spin_rpm,
                    parse_spin_label,
                    "rpm",
                ),
            ):
                if not self._controls(kind):
                    if not config.allow_uncontrolled_levels:
                        raise UnsupportedOperation(
                            f"SmartWash cannot set the {kind} through Home Assistant, so it "
                            f"cannot keep it at or below {requested} {unit}. Set "
                            f"{kind}_select_entity or use {{{kind}}} or {_NUMBER[kind]} in the "
                            "start service data; or set allow_uncontrolled_levels: true if you "
                            "always set it on the washer yourself"
                        )
                    self._warn_once(kind, "The washer's own %s setting applies", kind)
                    continue
                if entity is None and not self._uses("{" + kind + "}"):
                    continue  # only the plain-number placeholder is used
                option = self._level_option(entity, mapping, requested, parse, unit, kind)
                values[kind] = option
                if entity:
                    selects.append((entity, option))

            if program_option is not None and config.program_select_entity:
                self._select(config.program_select_entity, program_option)
            for entity, option in selects:
                self._select(entity, option)

            for wanted, call, label in (
                (settings.prewash, config.prewash, "prewash"),
                (settings.extra_rinse, config.extra_rinse, "extra_rinse"),
            ):
                if wanted and call is not None:
                    self._call(call, values)
                elif wanted:
                    log.warning("No home_assistant.%s service is configured; ignoring it", label)
            self._call(config.start, values)
            self._cached = None
            log.info("Started the washer through Home Assistant with %s", values)

    def pause(self) -> None:
        self._simple_call(self.config.pause, "pause")

    def resume(self) -> None:
        self._simple_call(self.config.resume, "resume")

    def stop(self) -> None:
        self._simple_call(self.config.stop, "stop")

    # ------------------------------------------------------------------- helpers

    def _simple_call(self, call: ServiceCall | None, label: str) -> None:
        if call is None:
            raise UnsupportedOperation(f"Configure home_assistant.{label} to {label} the washer")
        with self._lock, _unexpected_payloads():
            self._call(call, {})
            self._cached = None

    def _controls(self, kind: str) -> bool:
        """Whether the temperature or spin can be set through Home Assistant."""
        entity = (
            self.config.temperature_select_entity
            if kind == "temperature"
            else self.config.spin_select_entity
        )
        return entity is not None or self._uses("{" + kind + "}") or self._uses(_NUMBER[kind])

    def _chooses_program(self) -> bool:
        has_target = self.config.program_select_entity is not None or self._uses("{program}")
        return bool(self.config.program_map) and has_target

    def _uses(self, placeholder: str) -> bool:
        calls = (self.config.start, self.config.prewash, self.config.extra_rinse)
        return any(call is not None and placeholder in json.dumps(call.data) for call in calls)

    def _program_option(self, program: Program) -> str | None:
        for candidate in program_candidates(program):
            if candidate.value in self.config.program_map:
                return self.config.program_map[candidate.value]
        return None

    def _program_for(self, option: str | None) -> Program | None:
        for program, mapped in self.config.program_map.items():
            if option is not None and mapped == option:
                return Program(program)
        return None

    def _level_option(
        self,
        entity: str | None,
        mapping: dict[int, str],
        requested: int,
        parse: Callable[[str], int | None],
        unit: str,
        kind: str,
    ) -> str:
        if mapping:
            value = pick_at_most(requested, mapping)
            if value is None:
                raise UnsupportedOperation(
                    f"{kind}_map has nothing at or below {requested} {unit}: {sorted(mapping)}"
                )
            return mapping[value]
        if entity is None:
            raise UnsupportedOperation(f"Set {kind}_map or {kind}_select_entity for {{{kind}}}")
        options = (self._entity(entity, required=True).get("attributes") or {}).get("options")
        option = choose_at_most(requested, options or [], parse)
        if option is None:
            raise UnsupportedOperation(
                f"{entity} offers nothing at or below {requested} {unit}: {options}"
            )
        return option

    def _remaining_min(self, raw: dict[str, Any]) -> float | None:
        if not self.config.remaining_entity:
            return None
        entity = self._entity(self.config.remaining_entity)
        if not entity:
            return None
        value = entity.get("state")
        raw[self.config.remaining_entity] = value
        unit = (entity.get("attributes") or {}).get("unit_of_measurement")
        return _parse_minutes(value, unit)

    def _state_of(self, entity_id: str | None, raw: dict[str, Any]) -> Any:
        if not entity_id:
            return None
        entity = self._entity(entity_id)
        value = entity.get("state") if entity else None
        raw[entity_id] = value
        return None if value in (None, "unavailable", "unknown") else value

    def _entity(self, entity_id: str, *, required: bool = False) -> dict[str, Any] | None:
        entity = self._request("GET", f"/api/states/{entity_id}", allow_404=not required)
        if entity is None:
            self._warn_once(entity_id, "Home Assistant has no entity %s", entity_id)
        return entity

    def _select(self, entity_id: str, option: str) -> None:
        self._call(
            ServiceCall(service="select.select_option"), {}, entity_id=entity_id, option=option
        )

    def _call(self, call: ServiceCall, values: dict[str, Any], **extra: Any) -> None:
        domain, service = call.service.split(".", 1)
        data = {**_fill(call.data, values), **extra}
        self._request("POST", f"/api/services/{domain}/{service}", json=data)

    def _request(self, method: str, path: str, *, json: Any = None, allow_404: bool = False) -> Any:
        try:
            response = self._client.request(
                method,
                f"{self._base}{path}",
                headers={"Authorization": f"Bearer {self.config.token}"},
                json=json,
            )
        except (httpx.HTTPError, RuntimeError) as exc:  # RuntimeError: client closed
            raise DeviceError(f"Home Assistant request failed: {exc}") from exc
        if response.status_code == 404 and allow_404:
            return None
        if response.is_success:
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise DeviceError(
                    f"Home Assistant sent a response that is not JSON for {method} {path}"
                ) from exc
        try:
            message = response.json().get("message")
        except (ValueError, AttributeError):
            message = None
        message = message or response.text or response.reason_phrase
        text = f"Home Assistant returned {response.status_code} for {method} {path}: {message}"
        if response.status_code in (401, 403):
            raise DeviceAuthError(text)
        if response.status_code == 400:
            raise DeviceNotReady(text)
        raise DeviceError(text)

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message, *args)


_NUMBER = {"temperature": "{temperature_c}", "spin": "{spin_rpm}"}


@contextmanager
def _unexpected_payloads() -> Iterator[None]:
    """Report malformed API responses as a DeviceError instead of crashing the caller."""
    try:
        yield
    except DeviceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise DeviceError(f"Unexpected response from Home Assistant: {exc!r}") from exc


def _fill(value: Any, values: dict[str, Any]) -> Any:
    """Replace ``{name}`` placeholders; a string that is exactly one keeps the value's type."""
    if isinstance(value, str):
        for name, replacement in values.items():
            token = "{" + name + "}"
            if value == token:
                return replacement
            value = value.replace(token, str(replacement))
        return value
    if isinstance(value, dict):
        return {key: _fill(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, values) for item in value]
    return value


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def _parse_minutes(value: Any, unit: str | None) -> float | None:
    """Minutes from a duration (in ``unit``), an ``H:MM[:SS]`` string or a finish timestamp."""
    if value in (None, "unavailable", "unknown"):
        return None
    number = _to_float(value)
    if number is not None:
        return round(number * _MINUTES_PER_UNIT.get(unit or "min", 1.0), 1)
    text = str(value)
    if re.fullmatch(r"\d+:\d{2}(:\d{2})?", text):
        parts = [int(p) for p in text.split(":")] + [0]
        return round(parts[0] * 60 + parts[1] + parts[2] / 60, 1)
    try:
        finish = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if finish.tzinfo is None:
        finish = finish.replace(tzinfo=timezone.utc)
    return round(max((finish - datetime.now(timezone.utc)).total_seconds(), 0.0) / 60.0, 1)
