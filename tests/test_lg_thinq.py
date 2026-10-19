import json

import httpx
import pytest
from pydantic import ValidationError

from smartwash.devices.base import (
    DeviceAuthError,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
)
from smartwash.devices.lg_thinq import API_KEY, LgThinqAdapter, LgThinqConfig
from smartwash.schemas import CyclePhase, CycleSettings, MachineStatus, Program

DEVICE = "MW2-0B530EFD-1ADF-4F54-A2C3-46C37F94C689"

PROFILE = {
    "notification": {"push": ["WASHING_IS_COMPLETE", "ERROR_DURING_WASHING"]},
    "property": [
        {
            "location": {"locationName": "MAIN"},
            "operation": {
                "washerOperationMode": {
                    "mode": ["w"],
                    "type": "enum",
                    "value": {"w": ["START", "STOP", "POWER_OFF", "POWER_ON"]},
                }
            },
            "runState": {
                "currentState": {"mode": ["r"], "type": "enum", "value": {"r": ["RUNNING", "END"]}}
            },
        }
    ],
}


def state(current="INITIAL", remote=True, remain=(0, 45), total=(0, 50)):
    return {
        "location": {"locationName": "MAIN"},
        "runState": {"currentState": current},
        "remoteControlEnable": {"remoteControlEnabled": remote},
        "timer": {
            "remainHour": remain[0],
            "remainMinute": remain[1],
            "totalHour": total[0],
            "totalMinute": total[1],
        },
        "cycle": {"cycleCount": 10},
    }


class FakeThinq:
    def __init__(self) -> None:
        self.state: object = state()
        self.devices = [
            {"deviceId": "fridge-1", "deviceInfo": {"deviceType": "DEVICE_REFRIGERATOR"}},
            {
                "deviceId": DEVICE,
                "deviceInfo": {
                    "deviceType": "DEVICE_WASHER",
                    "modelName": "FAFXU22027",
                    "alias": "Test washer",
                    "reportable": True,
                },
            },
        ]
        self.requests: list[httpx.Request] = []
        self.control_error: tuple[int, str] | None = None
        self.state_error: tuple[int, str] | None = None

    @staticmethod
    def ok(payload) -> httpx.Response:
        return httpx.Response(200, json={"messageId": "m", "timestamp": "t", "response": payload})

    @staticmethod
    def fail(status: int, code: str) -> httpx.Response:
        return httpx.Response(status, json={"error": {"code": code, "message": f"error {code}"}})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/devices":
            return self.ok(self.devices)
        if path == f"/devices/{DEVICE}/profile":
            return self.ok(PROFILE)
        if path == f"/devices/{DEVICE}/state":
            return self.fail(*self.state_error) if self.state_error else self.ok(self.state)
        if path == f"/devices/{DEVICE}/control":
            return self.fail(*self.control_error) if self.control_error else self.ok({})
        return self.fail(404, "1205")

    def controls(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/control")]


def make_adapter(thinq: FakeThinq, **config) -> LgThinqAdapter:
    settings = {"access_token": "pat", "country": "GB", "client_id": "client-1"}
    settings.update(config)
    adapter = LgThinqAdapter(
        LgThinqConfig(min_poll_interval_s=0, **settings),
        client=httpx.Client(transport=httpx.MockTransport(thinq)),
    )
    adapter.connect()
    return adapter


COTTON = CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200)


def test_connect_finds_the_washer_and_its_operation_modes():
    thinq = FakeThinq()
    adapter = make_adapter(thinq)
    described = adapter.describe()
    assert (described["id"], described["model"]) == (DEVICE, "FAFXU22027")
    assert described["operation_modes"] == ["START", "STOP", "POWER_OFF", "POWER_ON"]
    request = thinq.requests[0]
    assert str(request.url) == "https://api-eic.lgthinq.com/devices"
    assert request.headers["x-api-key"] == API_KEY
    assert request.headers["x-country"] == "GB"
    assert request.headers["x-client-id"] == "client-1"
    assert request.headers["x-service-phase"] == "OP"
    assert len(request.headers["x-message-id"]) == 22


def test_region_follows_the_country():
    thinq = FakeThinq()
    make_adapter(thinq, country="us")
    assert thinq.requests[0].url.host == "api-aic.lgthinq.com"


def test_unsupported_country_is_rejected():
    with pytest.raises(ValidationError, match="does not support"):
        LgThinqConfig(access_token="pat", country="ZZ")


def test_no_washer_on_account():
    thinq = FakeThinq()
    thinq.devices = thinq.devices[:1]
    with pytest.raises(DeviceError, match="DEVICE_REFRIGERATOR"):
        make_adapter(thinq)


def test_running_state_with_timer_and_progress():
    thinq = FakeThinq()
    thinq.state = [state("RINSING", remain=(0, 20), total=(1, 20))]
    reported = make_adapter(thinq).get_state()
    assert reported.status is MachineStatus.RUNNING
    assert reported.phase is CyclePhase.RINSE
    assert reported.remaining_min == 20.0
    assert reported.progress_pct == 75.0
    assert reported.remote_start_allowed is True


@pytest.mark.parametrize(
    ("current", "status"),
    [
        ("POWER_OFF", MachineStatus.IDLE),
        ("PAUSE", MachineStatus.PAUSED),
        ("RESERVED", MachineStatus.DELAYED_START),
        ("END", MachineStatus.FINISHED),
        ("ERROR", MachineStatus.ERROR),
        ("DETECTING", MachineStatus.RUNNING),
    ],
)
def test_status_mapping(current, status):
    thinq = FakeThinq()
    thinq.state = state(current)
    assert make_adapter(thinq).get_state().status is status


def test_disconnected_washer_is_offline():
    thinq = FakeThinq()
    thinq.state_error = (400, "1222")
    assert make_adapter(thinq).get_state().status is MachineStatus.OFFLINE


def test_start_requires_opt_in_because_settings_cannot_be_applied():
    thinq = FakeThinq()
    adapter = make_adapter(thinq)
    assert not adapter.capabilities.remote_start
    with pytest.raises(UnsupportedOperation, match="start_preselected_program"):
        adapter.start_cycle(COTTON)
    assert thinq.controls() == []


def test_start_and_stop_send_operation_modes():
    thinq = FakeThinq()
    adapter = make_adapter(thinq, start_preselected_program=True)
    assert adapter.capabilities.remote_start and not adapter.capabilities.select_program
    adapter.start_cycle(COTTON)
    adapter.stop()
    assert thinq.controls() == [
        {"location": {"locationName": "MAIN"}, "operation": {"washerOperationMode": "START"}},
        {"location": {"locationName": "MAIN"}, "operation": {"washerOperationMode": "STOP"}},
    ]
    control = [r for r in thinq.requests if r.url.path.endswith("/control")][0]
    assert control.headers["x-conditional-control"] == "true"


@pytest.mark.parametrize(
    ("washer", "message"),
    [
        (state("RUNNING"), "already running"),
        (state("POWER_OFF"), "Switch the washer on"),
        (state(remote=False), "Remote Start"),
    ],
)
def test_start_refuses_when_not_ready(washer, message):
    thinq = FakeThinq()
    thinq.state = washer
    with pytest.raises(DeviceNotReady, match=message):
        make_adapter(thinq, start_preselected_program=True).start_cycle(COTTON)


def test_remote_off_error_from_the_api_explains_the_fix():
    thinq = FakeThinq()
    thinq.control_error = (400, "2301")
    with pytest.raises(DeviceNotReady, match="enable Remote Start"):
        make_adapter(thinq, start_preselected_program=True).start_cycle(COTTON)


def test_invalid_token_is_an_auth_error():
    thinq = FakeThinq()
    thinq.state_error = (401, "1103")
    with pytest.raises(DeviceAuthError, match="1103"):
        make_adapter(thinq).get_state()


def test_pause_is_unsupported():
    adapter = make_adapter(FakeThinq())
    assert not adapter.capabilities.pause_resume
    with pytest.raises(UnsupportedOperation):
        adapter.pause()
    with pytest.raises(UnsupportedOperation):
        adapter.resume()


def test_closed_client_and_non_json_responses_are_device_errors():
    adapter = LgThinqAdapter(
        LgThinqConfig(access_token="pat", country="GB", device_id="washer-1"),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))
        ),
    )
    with pytest.raises(DeviceError, match="not JSON"):
        adapter.get_state()
    adapter._client.close()
    with pytest.raises(DeviceError, match="request failed"):
        adapter.get_state()
