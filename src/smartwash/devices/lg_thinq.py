"""LG washers via the official LG ThinQ Connect API, using a Personal Access Token.

Create a token at https://connect-pat.lgthinq.com with your LG ThinQ account. Requests go to
``https://api-{region}.lgthinq.com`` (region from your country) with the headers LG's official
``thinqconnect`` SDK sends.

The ThinQ Connect API cannot choose a washer's course, temperature or spin, and has no pause: it
reports progress and can start or stop the course selected on the machine while Remote Start is on.
Starting is therefore opt-in (``start_preselected_program``), since the recommended temperature and
spin cannot be enforced; the Home Assistant LG integration has the same limits.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

from ..schemas import ACTIVE_STATUSES, CyclePhase, CycleSettings, MachineState, MachineStatus
from .base import (
    AdapterCapabilities,
    DeviceAuthError,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
    WashingMachineAdapter,
)

log = logging.getLogger(__name__)

API_KEY = "v6GFvkweNo7DK7yD3ylIZ9w52aKBU0eJ7wLXkSR3"
"""Public API key every ThinQ Connect client sends (from LG's Apache-2.0 ``thinqconnect`` SDK)."""
WASHER_TYPES = frozenset({"DEVICE_WASHER", "DEVICE_WASHTOWER_WASHER"})

_REGION_COUNTRIES = {
    "kic": "AU BD CN HK ID IN JP KH KR LA LK MM MY NP NZ PH SG TH TW VN",
    "aic": "AG AR AW BB BO BR BS BZ CA CL CO CR CU DM DO EC GD GT GY HN HT JM KN LC MX NI PA PE PR "
    "PY SR SV TT US UY VC VE",
    "eic": "AE AF AL AM AO AT AZ BA BE BF BG BH BJ BY CD CF CG CH CI CM CV CY CZ DE DJ DK DZ EE EG "
    "ES ET FI FR GA GB GE GH GM GN GQ GR HR HU IE IL IQ IR IS IT JO KE KG KW KZ LB LR LT LU LV LY "
    "MA MD ME MK ML MR MT MU MW NE NG NL NO OM PK PL PS PT QA RO RS RU RW SA SD SE SI SK SL SN SO "
    "ST SY TD TG TN TR TZ UA UG UZ XK YE ZA ZM",
}
REGIONS = {
    country: region
    for region, countries in _REGION_COUNTRIES.items()
    for country in countries.split()
}

_IDLE = {"POWER_OFF", "INITIAL", "SLEEP", "FROZEN_PREVENT_INITIAL"}
_STATUS = {
    **dict.fromkeys(_IDLE, MachineStatus.IDLE),
    "RESERVED": MachineStatus.DELAYED_START,
    "PAUSE": MachineStatus.PAUSED,
    "FROZEN_PREVENT_PAUSE": MachineStatus.PAUSED,
    "RINSE_HOLD": MachineStatus.PAUSED,
    "END": MachineStatus.FINISHED,
    "ERROR": MachineStatus.ERROR,
}
"""``runState.currentState`` -> status; every other state (RUNNING, RINSING, ...) is running."""
_PHASES = {
    "PREWASH": CyclePhase.WASH,
    "SOAKING": CyclePhase.WASH,
    "RUNNING": CyclePhase.WASH,
    "RINSING": CyclePhase.RINSE,
    "SPINNING": CyclePhase.SPIN,
    "ADD_DRAIN": CyclePhase.DRAIN,
    "END": CyclePhase.DONE,
}

_AUTH_CODES = {"1103", "1201", "1202", "1218", "1302", "1303", "1308"}
_NOT_READY_CODES = {"1222", "1223", "2205", "2209", "2212", "2301", "2302", "2303", "2304", "2305"}
_UNSUPPORTED_CODES = {"1219", "1220", "1221", "2101", "2201", "2202"}
_OFFLINE_CODE = "1222"
_REMOTE_OFF_CODE = "2301"


class LgThinqConfig(BaseModel):
    access_token: str
    """Personal Access Token from https://connect-pat.lgthinq.com."""
    country: str
    """ISO 3166 country code of your ThinQ account, e.g. ``GB`` or ``US``."""
    client_id: str | None = None
    """Stable id for this installation; a random one is generated when omitted."""
    device_id: str | None = None
    """The washer's deviceId; the first washer on the account is used when omitted."""
    location: str = "MAIN"
    start_preselected_program: bool = False
    """Allow ``start_cycle`` to start the course selected on the washer (settings not applied)."""
    base_url: str | None = None
    min_poll_interval_s: float = 30.0
    timeout_s: float = 15.0

    @field_validator("country")
    @classmethod
    def _check_country(cls, value: str) -> str:
        value = value.upper()
        if value not in REGIONS:
            raise ValueError(f"LG ThinQ Connect does not support country {value!r}")
        return value


class LgThinqAdapter(WashingMachineAdapter):
    name = "lg_thinq"

    def __init__(self, config: LgThinqConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        region = REGIONS[config.country]
        self._base = (config.base_url or f"https://api-{region}.lgthinq.com").rstrip("/")
        self._client = client or httpx.Client(timeout=config.timeout_s)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._client_id = config.client_id or str(uuid.uuid4())
        if config.client_id is None:
            log.info(
                "Generated LG ThinQ client id %s; set lg_thinq.client_id to keep it",
                self._client_id,
            )
        self._device_id = config.device_id
        self._device: dict[str, Any] = {}
        self._modes: list[str] = []
        self._cached: tuple[float, MachineState] | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            remote_start=self.config.start_preselected_program,
            select_program=False,
            pause_resume=False,
            stop=not self._modes or "STOP" in self._modes,
        )

    # ----------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        with self._lock, _unexpected_payloads():
            devices = self._request("GET", "devices") or []
            washers = [d for d in devices if _device_type(d) in WASHER_TYPES]
            if self._device_id:
                matches = [d for d in devices if d.get("deviceId") == self._device_id]
                if not matches:
                    raise DeviceError(f"LG ThinQ has no device {self._device_id}")
                if _device_type(matches[0]) not in WASHER_TYPES:
                    raise DeviceError(
                        f"LG ThinQ device {self._device_id} is a {_device_type(matches[0])}"
                    )
                self._device = matches[0]
            elif washers:
                self._device = washers[0]
                self._device_id = washers[0]["deviceId"]
                log.info(
                    "Using LG ThinQ washer %s (%s)", self._device_id, _info(self._device, "alias")
                )
            else:
                found = ", ".join(f"{_info(d, 'alias')} ({_device_type(d)})" for d in devices)
                raise DeviceError(f"No LG washer on this ThinQ account; found: {found or 'none'}")
            profile = self._request("GET", f"devices/{self._device_id}/profile") or {}
            operation = _for_location(profile.get("property"), self.config.location).get(
                "operation"
            )
            mode = (operation or {}).get("washerOperationMode") or {}
            self._modes = list((mode.get("value") or {}).get("w") or [])

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            **_describe(self._device),
            "location": self.config.location,
            "operation_modes": self._modes,
        }

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock, _unexpected_payloads():
            devices = self._request("GET", "devices") or []
        return [_describe(d) for d in devices if _device_type(d) in WASHER_TYPES]

    # --------------------------------------------------------------------- state

    def get_state(self) -> MachineState:
        with self._lock, _unexpected_payloads():
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self.config.min_poll_interval_s:
                return self._cached[1]
            state = self._read_state()
            self._cached = (now, state)
            return state

    def _read_state(self) -> MachineState:
        try:
            payload = self._request("GET", f"devices/{self._id()}/state")
        except DeviceNotReady as exc:
            if getattr(exc, "lg_code", None) == _OFFLINE_CODE:
                return MachineState(status=MachineStatus.OFFLINE, error_code=_OFFLINE_CODE)
            raise
        entry = _for_location(payload, self.config.location)
        current = (entry.get("runState") or {}).get("currentState")
        status = _STATUS.get(current, MachineStatus.RUNNING) if current else MachineStatus.IDLE
        phase = _PHASES.get(current) or (CyclePhase.IDLE if status is MachineStatus.IDLE else None)
        timer = entry.get("timer") or {}
        remaining = _minutes(timer, "remainHour", "remainMinute")
        total = _minutes(timer, "totalHour", "totalMinute")
        active = status in ACTIVE_STATUSES
        progress = None
        if active and remaining is not None and total:
            progress = round(100.0 * max(total - remaining, 0.0) / total, 1)
        remote = (entry.get("remoteControlEnable") or {}).get("remoteControlEnabled")
        return MachineState(
            status=status,
            phase=phase,
            progress_pct=progress,
            remaining_min=remaining if active else None,
            remote_start_allowed=remote if isinstance(remote, bool) else None,
            error_code=current if status is MachineStatus.ERROR else None,
            raw=entry,
        )

    # ------------------------------------------------------------------ commands

    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        if not self.config.start_preselected_program:
            raise UnsupportedOperation(
                "LG's ThinQ Connect API cannot choose the course, temperature or spin. Select "
                "the course on the washer, enable Remote Start and set "
                "lg_thinq.start_preselected_program: true to let SmartWash start it"
            )
        with self._lock, _unexpected_payloads():
            state = self._read_state()
            if state.status in ACTIVE_STATUSES:
                raise DeviceNotReady("A cycle is already running on the washer")
            if (state.raw.get("runState") or {}).get("currentState") == "POWER_OFF":
                raise DeviceNotReady(
                    "Switch the washer on, choose the course and enable Remote Start"
                )
            if state.remote_start_allowed is False:
                raise DeviceNotReady("Remote Start is off; enable it on the washer first")
            self._operate("START")
            log.warning(
                "Started the course selected on the washer; SmartWash could not apply %s at "
                "%s °C / %s rpm through the LG API",
                settings.program.value,
                settings.temperature_c,
                settings.spin_rpm,
            )

    def pause(self) -> None:
        raise UnsupportedOperation("The LG ThinQ Connect API cannot pause a washer")

    def resume(self) -> None:
        raise UnsupportedOperation("The LG ThinQ Connect API cannot resume a washer")

    def stop(self) -> None:
        with self._lock, _unexpected_payloads():
            self._operate("STOP")

    # ------------------------------------------------------------------- helpers

    def _operate(self, mode: str) -> None:
        if self._modes and mode not in self._modes:
            raise UnsupportedOperation(
                f"The washer does not accept {mode}; it accepts {self._modes}"
            )
        body = {
            "location": {"locationName": self.config.location},
            "operation": {"washerOperationMode": mode},
        }
        self._request("POST", f"devices/{self._id()}/control", json=body, control=True)
        self._cached = None

    def _id(self) -> str:
        if not self._device_id:
            self.connect()
        return str(self._device_id)

    def _request(
        self, method: str, endpoint: str, *, json: Any = None, control: bool = False
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "x-country": self.config.country,
            "x-message-id": base64.urlsafe_b64encode(uuid.uuid4().bytes)[:-2].decode(),
            "x-client-id": self._client_id,
            "x-api-key": API_KEY,
            "x-service-phase": "OP",
        }
        if control:
            headers["x-conditional-control"] = "true"
        try:
            response = self._client.request(
                method, f"{self._base}/{endpoint}", headers=headers, json=json
            )
        except (httpx.HTTPError, RuntimeError) as exc:  # RuntimeError: client closed
            raise DeviceError(f"LG ThinQ request failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError:
            if response.is_success:
                raise DeviceError("LG ThinQ sent a response that is not JSON") from None
            payload = {}
        if response.is_success:
            return payload.get("response") if isinstance(payload, dict) else None
        error = (payload.get("error") if isinstance(payload, dict) else None) or {}
        raise _error(response.status_code, str(error.get("code", "")), error.get("message"))


@contextmanager
def _unexpected_payloads() -> Iterator[None]:
    """Report malformed API responses as a DeviceError instead of crashing the caller."""
    try:
        yield
    except DeviceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise DeviceError(f"Unexpected response from LG ThinQ: {exc!r}") from exc


def _error(status_code: int, code: str, message: str | None) -> DeviceError:
    text = f"LG ThinQ returned {status_code}: {message or 'no details'} (code {code or '-'})"
    if code == _REMOTE_OFF_CODE:
        text += "; enable Remote Start on the washer"
    if code in _AUTH_CODES or status_code in (401, 403):
        error: DeviceError = DeviceAuthError(text)
    elif code in _NOT_READY_CODES:
        error = DeviceNotReady(text)
    elif code in _UNSUPPORTED_CODES:
        error = UnsupportedOperation(text)
    else:
        error = DeviceError(text)
    error.lg_code = code  # type: ignore[attr-defined]
    return error


def _for_location(payload: Any, location: str) -> dict[str, Any]:
    """The entry for ``location`` from a state/profile payload (a list for multi-drum devices)."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for entry in payload:
            if (entry.get("location") or {}).get("locationName") == location:
                return entry
        if len(payload) == 1:
            return payload[0]
    return {}


def _minutes(timer: dict[str, Any], hours_key: str, minutes_key: str) -> float | None:
    hours, minutes = timer.get(hours_key), timer.get(minutes_key)
    if hours is None and minutes is None:
        return None
    return float((hours or 0) * 60 + (minutes or 0))


def _info(device: dict[str, Any], key: str) -> Any:
    return (device.get("deviceInfo") or {}).get(key)


def _device_type(device: dict[str, Any]) -> str | None:
    return _info(device, "deviceType")


def _describe(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": device.get("deviceId"),
        "name": _info(device, "alias"),
        "model": _info(device, "modelName"),
        "type": _device_type(device),
    }
