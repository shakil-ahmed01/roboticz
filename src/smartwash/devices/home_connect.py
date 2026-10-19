"""Bosch, Siemens, Neff, Gaggenau and Thermador washers via the official Home Connect REST API.

API reference: https://api-docs.home-connect.com. Production: ``https://api.home-connect.com``;
the developer simulator (virtual appliances, same API, needs a developer-portal token):
``https://simulator.home-connect.com``.

Rate limits per client and user account: 1000 requests a day, 50 a minute, 5 programme starts and
5 stops a minute, 10 token refreshes a minute (100 a day), and 10 consecutive failed requests block
the client for 10 minutes. ``get_state`` is therefore cached for ``min_poll_interval_s`` (two
requests per refresh), which keeps a day of 1-minute polling during washes well inside the quota.

The API exposes no sensor telemetry, prewash or extra-rinse option for washers, so those are
unavailable here.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, model_validator

from ..schemas import CyclePhase, CycleSettings, MachineState, MachineStatus, Program
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

MEDIA_TYPE = "application/vnd.bsh.sdk.v1+json"
WASHER_TYPES = frozenset({"Washer", "WasherDryer"})

OPERATION_STATE = "BSH.Common.Status.OperationState"
DOOR_STATE = "BSH.Common.Status.DoorState"
REMOTE_CONTROL_ACTIVE = "BSH.Common.Status.RemoteControlActive"
REMOTE_START_ALLOWED = "BSH.Common.Status.RemoteControlStartAllowed"
LOCAL_CONTROL_ACTIVE = "BSH.Common.Status.LocalControlActive"
TEMPERATURE_OPTION = "LaundryCare.Washer.Option.Temperature"
SPIN_OPTION = "LaundryCare.Washer.Option.SpinSpeed"
REMAINING_TIME_OPTION = "BSH.Common.Option.RemainingProgramTime"
PROGRESS_OPTION = "BSH.Common.Option.ProgramProgress"
PAUSE_COMMAND = "BSH.Common.Command.PauseProgram"
RESUME_COMMAND = "BSH.Common.Command.ResumeProgram"

_OP = "BSH.Common.EnumType.OperationState."
_OPERATION_STATUS = {
    _OP + "Inactive": MachineStatus.IDLE,
    _OP + "Ready": MachineStatus.IDLE,
    _OP + "DelayedStart": MachineStatus.DELAYED_START,
    _OP + "Run": MachineStatus.RUNNING,
    _OP + "Pause": MachineStatus.PAUSED,
    _OP + "ActionRequired": MachineStatus.PAUSED,
    _OP + "Finished": MachineStatus.FINISHED,
    _OP + "Error": MachineStatus.ERROR,
    _OP + "Aborting": MachineStatus.RUNNING,
}
_BUSY_STATES = {_OP + s for s in ("DelayedStart", "Run", "Pause", "ActionRequired", "Aborting")}
_DOOR_OPEN = "BSH.Common.EnumType.DoorState.Open"

_P = "LaundryCare.Washer.Program."
PROGRAM_KEYS: dict[Program, tuple[str, ...]] = {
    Program.COTTON: (_P + "Cotton",),
    Program.ECO_40_60: (_P + "Cotton.Eco4060", _P + "Cotton.CottonEco"),
    Program.SYNTHETICS: (_P + "EasyCare",),
    Program.MIXED: (_P + "Mix",),
    Program.WOOL: (_P + "Wool",),
    Program.DELICATES: (_P + "DelicatesSilk",),
    Program.QUICK: (
        _P + "Super153045.Super1530",
        _P + "PowerSpeed59",
        _P + "Super153045.Super15",
    ),
    Program.SPORTSWEAR: (_P + "SportFitness",),
    Program.BEDDING: (_P + "Cotton",),
    Program.TOWELS: (_P + "Towels",),
    Program.DARK_WASH: (_P + "DarkWash",),
    Program.HYGIENE: (_P + "Sensitive",),
}
"""Our programmes in Home Connect terms, best match first (keys from the official API docs)."""

_PROGRAM_FOR_KEY: dict[str, Program] = {}
for _program, _keys in PROGRAM_KEYS.items():
    for _key in _keys:
        _PROGRAM_FOR_KEY.setdefault(_key, _program)

_T = "LaundryCare.Washer.EnumType.Temperature."
ALL_TEMPERATURES = tuple(
    _T + v
    for v in ("Cold", "GC20", "GC30", "GC40", "GC50", "GC60", "GC70", "GC80", "GC90")
    + ("UlCold", "UlWarm", "UlHot", "UlExtraHot")
)
_S = "LaundryCare.Washer.EnumType.SpinSpeed."
ALL_SPIN_SPEEDS = tuple(
    _S + v
    for v in ("Off", "RPM400", "RPM600", "RPM800", "RPM1000", "RPM1200", "RPM1400", "RPM1600")
    + ("UlOff", "UlLow", "UlMedium", "UlHigh")
)
"""Every value the API documents, used when a programme's own constraints are unavailable."""


class HomeConnectConfig(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None
    client_secret: str | None = None
    """Needed with ``refresh_token`` to renew the access token, which expires after 24 h."""
    ha_id: str | None = None
    """Appliance id (``haId``); the first washer on the account is used when omitted."""
    base_url: str = "https://api.home-connect.com"
    token_cache_path: str | None = None
    """Where refreshed tokens are stored so they survive restarts."""
    min_poll_interval_s: float = 60.0
    timeout_s: float = 20.0
    language: str = "en-GB"

    @model_validator(mode="after")
    def _check_credentials(self) -> HomeConnectConfig:
        if not self.access_token and not (self.refresh_token and self.client_secret):
            raise ValueError("set access_token, or refresh_token together with client_secret")
        return self


class HomeConnectAdapter(WashingMachineAdapter):
    name = "home_connect"

    def __init__(self, config: HomeConnectConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._base = config.base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._access_token = config.access_token
        self._refresh_token = config.refresh_token
        self._load_cached_tokens()
        self._ha_id = config.ha_id
        self._appliance: dict[str, Any] = {}
        self._cached: tuple[float, MachineState] | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities()

    # ----------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        with self._lock:
            if self._access_token is None:
                self._refresh()
            if self._ha_id:
                data = self._request("GET", f"/homeappliances/{self._ha_id}")["data"]
                if data.get("type") not in WASHER_TYPES:
                    raise DeviceError(
                        f"Appliance {self._ha_id} is a {data.get('type')}, not a washer"
                    )
                self._appliance = data
            else:
                washers = self._washers()
                if not washers:
                    found = self._request("GET", "/homeappliances")["data"]["homeappliances"]
                    kinds = ", ".join(f"{a.get('name')} ({a.get('type')})" for a in found) or "none"
                    raise DeviceError(f"No Home Connect washer on this account; found: {kinds}")
                self._appliance = washers[0]
                self._ha_id = washers[0]["haId"]
                log.info(
                    "Using Home Connect washer %s (%s)", self._ha_id, self._appliance.get("name")
                )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, **_describe_appliance(self._appliance)}

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_describe_appliance(a) for a in self._washers()]

    # --------------------------------------------------------------------- state

    def get_state(self) -> MachineState:
        with self._lock:
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self.config.min_poll_interval_s:
                return self._cached[1]
            state = self._read_state()
            self._cached = (now, state)
            return state

    def _read_state(self) -> MachineState:
        try:
            status = self._status()
        except DeviceNotReady as exc:  # 409: the appliance is not connected to the cloud
            return MachineState(status=MachineStatus.OFFLINE, error_code=str(exc))
        operation = status.get(OPERATION_STATE)
        machine_status = _OPERATION_STATUS.get(operation, MachineStatus.IDLE)
        phase = None
        if machine_status is MachineStatus.FINISHED:
            phase = CyclePhase.DONE
        elif machine_status is MachineStatus.IDLE:
            phase = CyclePhase.IDLE
        raw: dict[str, Any] = {"status": status}
        state = MachineState(
            status=machine_status,
            phase=phase,
            door_open=status.get(DOOR_STATE) == _DOOR_OPEN if DOOR_STATE in status else None,
            remote_start_allowed=status.get(REMOTE_START_ALLOWED),
            error_code=operation.rsplit(".", 1)[-1]
            if machine_status is MachineStatus.ERROR
            else None,
            raw=raw,
        )
        if operation not in _BUSY_STATES:
            return state
        program = self._request("GET", self._path("/programs/active"), allow_404=True)
        if not program:
            return state
        data = program["data"]
        options = {o["key"]: o.get("value") for o in data.get("options", [])}
        raw["active_program"] = data
        remaining = options.get(REMAINING_TIME_OPTION)
        temperature = options.get(TEMPERATURE_OPTION)
        spin = options.get(SPIN_OPTION)
        return state.model_copy(
            update={
                "program": _PROGRAM_FOR_KEY.get(data.get("key")),
                "vendor_program": data.get("key"),
                "temperature_c": label_number(temperature) if temperature else None,
                "spin_rpm": parse_spin_label(spin) if spin else None,
                "progress_pct": options.get(PROGRESS_OPTION),
                "remaining_min": round(remaining / 60.0, 1) if remaining is not None else None,
            }
        )

    # ------------------------------------------------------------------ commands

    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        with self._lock:
            status = self._status()
            operation = status.get(OPERATION_STATE)
            if operation in _BUSY_STATES:
                raise DeviceNotReady("A programme is already running on the washer")
            if status.get(REMOTE_CONTROL_ACTIVE) is False:
                raise DeviceNotReady(
                    "Remote control is disabled in the washer's Home Connect settings"
                )
            if status.get(REMOTE_START_ALLOWED) is False:
                raise DeviceNotReady(
                    "Remote start is off. Press the washer's Remote Start button (it switches "
                    "itself off again after about 24 hours), then try again"
                )
            if status.get(LOCAL_CONTROL_ACTIVE) is True:
                raise DeviceNotReady("Someone is operating the washer locally; try again shortly")
            if status.get(DOOR_STATE) == _DOOR_OPEN:
                raise DeviceNotReady("Close the washer door first")
            if operation is not None and operation != _OP + "Ready":
                state = operation.rsplit(".", 1)[-1]
                raise DeviceNotReady(f"The washer must be switched on and Ready (it is {state})")

            key = self._choose_program(settings.program)
            definition = self._program_options(key)
            chosen = (
                _option(key, definition, TEMPERATURE_OPTION, settings.temperature_c, "°C"),
                _option(key, definition, SPIN_OPTION, settings.spin_rpm, "rpm"),
            )
            options = [option for option in chosen if option is not None]
            if settings.prewash or settings.extra_rinse:
                log.warning("Home Connect has no prewash or extra-rinse option; ignoring them")
            self._request(
                "PUT",
                self._path("/programs/active"),
                body={"data": {"key": key, "options": options}},
            )
            self._cached = None
            log.info("Started %s with %s", key, options)

    def pause(self) -> None:
        self._command(PAUSE_COMMAND)

    def resume(self) -> None:
        self._command(RESUME_COMMAND)

    def stop(self) -> None:
        with self._lock:
            self._request("DELETE", self._path("/programs/active"))
            self._cached = None

    # ------------------------------------------------------------------- helpers

    def _command(self, key: str) -> None:
        with self._lock:
            self._request(
                "PUT", self._path(f"/commands/{key}"), body={"data": {"key": key, "value": True}}
            )
            self._cached = None

    def _choose_program(self, program: Program) -> str:
        programs = self._request("GET", self._path("/programs/available"))["data"]["programs"]
        startable = {
            p["key"]
            for p in programs
            if (p.get("constraints") or {}).get("execution") not in ("selectonly", "none")
        }
        for candidate in program_candidates(program):
            for key in PROGRAM_KEYS[candidate]:
                if key in startable:
                    return key
        names = ", ".join(sorted(k.removeprefix(_P) for k in startable)) or "none"
        raise UnsupportedOperation(
            f"This washer has no programme for {program.value}; remotely startable: {names}"
        )

    def _program_options(self, key: str) -> dict[str, list[str]] | None:
        """Allowed values of each option of a programme, or ``None`` if the API won't say."""
        try:
            data = self._request("GET", self._path(f"/programs/available/{key}"))["data"]
        except DeviceAuthError:
            raise
        except DeviceError as exc:
            log.warning("Could not read the options of %s (%s); using the generic list", key, exc)
            return None
        return {
            o["key"]: list((o.get("constraints") or {}).get("allowedvalues") or [])
            for o in data.get("options", [])
        }

    def _status(self) -> dict[str, Any]:
        items = self._request("GET", self._path("/status"))["data"]["status"]
        return {item["key"]: item.get("value") for item in items}

    def _washers(self) -> list[dict[str, Any]]:
        appliances = self._request("GET", "/homeappliances")["data"]["homeappliances"]
        return [a for a in appliances if a.get("type") in WASHER_TYPES]

    def _path(self, suffix: str) -> str:
        if not self._ha_id:
            self.connect()
        return f"/homeappliances/{self._ha_id}{suffix}"

    def _request(
        self, method: str, path: str, *, body: Any = None, allow_404: bool = False
    ) -> dict[str, Any] | None:
        if self._access_token is None:
            self._refresh()
        for attempt in range(2):
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": MEDIA_TYPE,
                "Accept-Language": self.config.language,
            }
            content = None
            if body is not None:
                headers["Content-Type"] = MEDIA_TYPE
                content = json.dumps(body)
            try:
                response = self._client.request(
                    method, f"{self._base}/api{path}", headers=headers, content=content
                )
            except httpx.HTTPError as exc:
                raise DeviceError(f"Home Connect request failed: {exc}") from exc
            if response.status_code == 401 and attempt == 0 and self._can_refresh():
                self._refresh()
                continue
            break
        if response.status_code == 404 and allow_404:
            return None
        if response.is_success:
            return response.json() if response.content else None
        raise _error(response)

    def _can_refresh(self) -> bool:
        return bool(self._refresh_token and self.config.client_secret)

    def _refresh(self) -> None:
        if not self._can_refresh():
            raise DeviceAuthError(
                "Home Connect access token missing or expired and no refresh token set"
            )
        try:
            response = self._client.post(
                f"{self._base}/security/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_secret": self.config.client_secret,
                },
            )
        except httpx.HTTPError as exc:
            raise DeviceError(f"Home Connect token refresh failed: {exc}") from exc
        payload = _json(response)
        if not response.is_success or "access_token" not in payload:
            reason = payload.get("error_description") or payload.get("error") or response.text
            raise DeviceAuthError(f"Home Connect token refresh failed: {reason}")
        self._access_token = payload["access_token"]
        self._refresh_token = payload.get("refresh_token", self._refresh_token)
        self._save_cached_tokens()
        log.info("Refreshed the Home Connect access token")

    def _load_cached_tokens(self) -> None:
        path = self.config.token_cache_path
        if not path or not Path(path).exists():
            return
        try:
            cached = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable token cache %s: %s", path, exc)
            return
        self._access_token = cached.get("access_token") or self._access_token
        self._refresh_token = cached.get("refresh_token") or self._refresh_token

    def _save_cached_tokens(self) -> None:
        path = self.config.token_cache_path
        if not path:
            return
        payload = json.dumps(
            {"access_token": self._access_token, "refresh_token": self._refresh_token}
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)


def _describe_appliance(appliance: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": appliance.get("haId"),
        "name": appliance.get("name"),
        "brand": appliance.get("brand"),
        "model": appliance.get("vib"),
        "enumber": appliance.get("enumber"),
        "type": appliance.get("type"),
        "connected": appliance.get("connected"),
    }


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


_OPTION_VALUES = {
    TEMPERATURE_OPTION: (ALL_TEMPERATURES, parse_temperature_label),
    SPIN_OPTION: (ALL_SPIN_SPEEDS, parse_spin_label),
}


def _option(
    program_key: str,
    definition: dict[str, list[str]] | None,
    option_key: str,
    requested: int,
    unit: str,
) -> dict[str, Any] | None:
    """The option value closest to ``requested`` without exceeding it, or ``None`` to omit it."""
    if definition is not None and option_key not in definition:
        log.warning("%s has no %s option; the machine's default applies", program_key, option_key)
        return None
    everything, parse = _OPTION_VALUES[option_key]
    allowed = (definition or {}).get(option_key) or list(everything)
    value = choose_at_most(requested, allowed, parse)
    if value is None:
        names = ", ".join(v.rsplit(".", 1)[-1] for v in allowed)
        raise UnsupportedOperation(
            f"{program_key} cannot run at or below {requested} {unit}; allowed: {names}"
        )
    return {"key": option_key, "value": value}


def _error(response: httpx.Response) -> DeviceError:
    error = _json(response).get("error") or {}
    key, description = error.get("key"), error.get("description")
    if key and description:
        detail = f"{description} ({key})"
    else:
        detail = description or key or response.text or response.reason_phrase
    message = f"Home Connect returned {response.status_code}: {detail}"
    if response.status_code in (401, 403):
        return DeviceAuthError(message)
    if response.status_code == 409:
        return DeviceNotReady(message)
    if response.status_code == 429:
        retry = response.headers.get("Retry-After")
        return DeviceError(message + (f"; retry after {retry} s" if retry else ""))
    return DeviceError(message)
