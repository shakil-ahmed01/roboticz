import json
import os
import stat

import httpx
import pytest
from pydantic import ValidationError

from smartwash.devices.base import (
    DeviceAuthError,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
)
from smartwash.devices.home_connect import MEDIA_TYPE, HomeConnectAdapter, HomeConnectConfig
from smartwash.schemas import CyclePhase, CycleSettings, MachineStatus, Program

HA_ID = "BOSCH-WAW28440-68A40E000001"
BASE = f"/api/homeappliances/{HA_ID}"
OP = "BSH.Common.EnumType.OperationState."
T = "LaundryCare.Washer.EnumType.Temperature."
S = "LaundryCare.Washer.EnumType.SpinSpeed."
P = "LaundryCare.Washer.Program."


class FakeApi:
    """Canned responses per (method, path); records every request."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list] = {}
        self.requests: list[httpx.Request] = []

    def on(self, method: str, path: str, *responses) -> None:
        self.routes[(method, path)] = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.routes.get((request.method, request.url.path))
        if not queue:
            return httpx.Response(404, json={"error": {"key": "404", "description": "no route"}})
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        result = item(request) if callable(item) else item
        if isinstance(result, httpx.Response):
            return result
        status, body = result
        return httpx.Response(status) if body is None else httpx.Response(status, json=body)

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]


def status_payload(operation="Ready", remote_start=True, door="Closed", local=False):
    return {
        "data": {
            "status": [
                {"key": "BSH.Common.Status.OperationState", "value": OP + operation},
                {
                    "key": "BSH.Common.Status.DoorState",
                    "value": f"BSH.Common.EnumType.DoorState.{door}",
                },
                {"key": "BSH.Common.Status.RemoteControlActive", "value": True},
                {"key": "BSH.Common.Status.RemoteControlStartAllowed", "value": remote_start},
                {"key": "BSH.Common.Status.LocalControlActive", "value": local},
            ]
        }
    }


def appliances(*types):
    return {
        "data": {
            "homeappliances": [
                {
                    "haId": HA_ID if kind == "Washer" else f"BOSCH-{kind}-1",
                    "name": f"My {kind}",
                    "type": kind,
                    "brand": "BOSCH",
                    "vib": "WAW28440",
                    "enumber": "WAW28440/07",
                    "connected": True,
                }
                for kind in types
            ]
        }
    }


def available(*keys):
    return {
        "data": {
            "programs": [
                {"key": P + k, "constraints": {"execution": "selectandstart"}} for k in keys
            ]
        }
    }


def definition(key, temperatures=None, spins=None):
    options = []
    if temperatures is not None:
        options.append(
            {
                "key": "LaundryCare.Washer.Option.Temperature",
                "type": "LaundryCare.Washer.EnumType.Temperature",
                "constraints": {"allowedvalues": [T + t for t in temperatures]},
            }
        )
    if spins is not None:
        options.append(
            {
                "key": "LaundryCare.Washer.Option.SpinSpeed",
                "type": "LaundryCare.Washer.EnumType.SpinSpeed",
                "constraints": {"allowedvalues": [S + s for s in spins]},
            }
        )
    return {"data": {"key": P + key, "options": options}}


@pytest.fixture
def api():
    fake = FakeApi()
    fake.on("GET", "/api/homeappliances", (200, appliances("Oven", "Washer")))
    fake.on("GET", f"{BASE}/status", (200, status_payload()))
    return fake


def make_adapter(api, **overrides):
    clock = overrides.pop("clock", None)
    config = HomeConnectConfig(
        **{
            "access_token": "token-1",
            "min_poll_interval_s": 0,
            "idle_poll_interval_s": 0,
            **overrides,
        }
    )
    extra = {"clock": clock} if clock else {}
    client = httpx.Client(transport=httpx.MockTransport(api))
    adapter = HomeConnectAdapter(config, client=client, **extra)
    adapter.connect()
    return adapter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_connect_picks_the_first_washer(api):
    adapter = make_adapter(api)
    assert adapter.describe()["id"] == HA_ID
    assert adapter.describe()["model"] == "WAW28440"
    assert [d["id"] for d in adapter.list_devices()] == [HA_ID]
    request = api.requests[0]
    assert request.headers["Authorization"] == "Bearer token-1"
    assert request.headers["Accept"] == MEDIA_TYPE


def test_connect_without_washer_lists_what_was_found():
    api = FakeApi()
    api.on("GET", "/api/homeappliances", (200, appliances("Oven", "Dishwasher")))
    with pytest.raises(DeviceError, match="My Oven \\(Oven\\)"):
        make_adapter(api)


def test_configured_appliance_must_be_a_washer():
    api = FakeApi()
    api.on("GET", f"{BASE}", (200, {"data": {"haId": HA_ID, "type": "Oven"}}))
    with pytest.raises(DeviceError, match="not a washer"):
        make_adapter(api, ha_id=HA_ID)


def test_running_state_includes_programme_progress_and_options(api):
    api.on("GET", f"{BASE}/status", (200, status_payload("Run", door="Locked")))
    api.on(
        "GET",
        f"{BASE}/programs/active",
        (
            200,
            {
                "data": {
                    "key": P + "Cotton",
                    "options": [
                        {"key": "LaundryCare.Washer.Option.Temperature", "value": T + "GC40"},
                        {"key": "LaundryCare.Washer.Option.SpinSpeed", "value": S + "RPM1200"},
                        {"key": "BSH.Common.Option.RemainingProgramTime", "value": 3600},
                        {"key": "BSH.Common.Option.ProgramProgress", "value": 42},
                    ],
                }
            },
        ),
    )
    state = make_adapter(api).get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.program is Program.COTTON
    assert state.vendor_program == P + "Cotton"
    assert (state.temperature_c, state.spin_rpm) == (40, 1200)
    assert (state.remaining_min, state.progress_pct) == (60.0, 42)
    assert state.door_open is False
    assert state.remote_start_allowed is True


@pytest.mark.parametrize(
    ("operation", "status", "phase"),
    [
        ("Ready", MachineStatus.IDLE, CyclePhase.IDLE),
        ("Inactive", MachineStatus.IDLE, CyclePhase.IDLE),
        ("Finished", MachineStatus.FINISHED, CyclePhase.DONE),
        ("Error", MachineStatus.ERROR, None),
    ],
)
def test_idle_states_do_not_fetch_the_active_programme(api, operation, status, phase):
    api.on("GET", f"{BASE}/status", (200, status_payload(operation)))
    state = make_adapter(api).get_state()
    assert (state.status, state.phase) == (status, phase)
    assert not api.sent("GET", f"{BASE}/programs/active")
    if status is MachineStatus.ERROR:
        assert state.error_code == "Error"


def test_disconnected_appliance_reports_offline(api):
    error = {"error": {"key": "SDK.Error.HomeAppliance.Connection.Initialization.Timeout"}}
    api.on("GET", f"{BASE}/status", (409, error))
    assert make_adapter(api).get_state().status is MachineStatus.OFFLINE


def test_polling_is_slow_when_idle_and_fast_after_a_command(api):
    clock = Clock()
    adapter = make_adapter(api, min_poll_interval_s=60, idle_poll_interval_s=300, clock=clock)
    reads = lambda: len(api.sent("GET", f"{BASE}/status"))  # noqa: E731
    adapter.get_state()
    clock.now += 120
    adapter.get_state()
    assert reads() == 1  # idle: cached for 5 minutes
    clock.now += 181
    adapter.get_state()
    assert reads() == 2
    api.on("DELETE", f"{BASE}/programs/active", (204, None))
    adapter.stop()
    adapter.get_state()
    clock.now += 61
    adapter.get_state()
    assert reads() == 4  # just commanded: re-read at once, then every minute


def test_failures_back_off_instead_of_hammering_the_api(api):
    clock = Clock()
    adapter = make_adapter(api, clock=clock)
    reads = lambda: len(api.sent("GET", f"{BASE}/status"))  # noqa: E731
    api.on("GET", f"{BASE}/status", (503, {"error": {"key": "unavailable"}}))
    for _ in range(5):
        with pytest.raises(DeviceError):
            adapter.get_state()
    assert reads() == 1
    clock.now += 16
    api.on("GET", f"{BASE}/status", (200, status_payload()))
    assert adapter.get_state().status is MachineStatus.IDLE


def test_rate_limit_honours_retry_after(api):
    clock = Clock()
    adapter = make_adapter(api, clock=clock)
    too_many = {"error": {"key": "429", "description": "too many requests"}}
    api.on(
        "GET",
        f"{BASE}/status",
        lambda request: httpx.Response(429, headers={"Retry-After": "120"}, json=too_many),
    )
    with pytest.raises(DeviceError, match="too many"):
        adapter.get_state()
    clock.now += 100
    with pytest.raises(DeviceError, match="retrying in 20 s"):
        adapter.get_state()
    assert len(api.sent("GET", f"{BASE}/status")) == 1


def test_rejected_refresh_token_is_not_retried_for_an_hour(api):
    clock = Clock()
    api.on("GET", f"{BASE}/status", (401, {"error": {"key": "invalid_token"}}))
    api.on("POST", "/security/oauth/token", (400, {"error": "invalid_grant"}))
    adapter = make_adapter(api, refresh_token="old", client_secret="secret", clock=clock)
    for _ in range(3):
        with pytest.raises(DeviceAuthError):
            adapter.get_state()
        clock.now += 301  # past the auth back-off, so the API is asked again
    assert len(api.sent("POST", "/security/oauth/token")) == 1


def test_token_cache_creates_its_directory_and_write_errors_are_not_fatal(api, tmp_path):
    api.on("POST", "/security/oauth/token", (200, {"access_token": "t2", "refresh_token": "r2"}))
    cache = tmp_path / "new" / "dir" / "tokens.json"
    adapter = make_adapter(
        api,
        access_token=None,
        refresh_token="r1",
        client_secret="secret",
        token_cache_path=str(cache),
    )
    assert json.loads(cache.read_text())["refresh_token"] == "r2"
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    adapter.config.token_cache_path = str(blocked / "tokens.json")
    adapter._refresh()  # logs a warning instead of raising
    assert adapter._access_token == "t2"


@pytest.mark.parametrize(
    "response",
    [
        lambda request: httpx.Response(200, text="<html>proxy login</html>"),
        lambda request: httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_malformed_responses_are_device_errors(api, response):
    adapter = make_adapter(api)
    api.on("GET", f"{BASE}/status", response)
    with pytest.raises(DeviceError):
        adapter.get_state()


def test_closed_client_is_a_device_error(api):
    adapter = make_adapter(api)
    adapter._client.close()
    with pytest.raises(DeviceError, match="request failed"):
        adapter.get_state()


def test_start_sends_the_programme_with_options_capped_at_the_request(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton", "Cotton.Eco4060")))
    api.on(
        "GET",
        f"{BASE}/programs/available/{P}Cotton.Eco4060",
        (200, definition("Cotton.Eco4060", ["Cold", "GC20", "GC30", "GC40", "GC60"],
                         ["RPM400", "RPM800", "RPM1000", "RPM1200"])),
    )  # fmt: skip
    api.on("PUT", f"{BASE}/programs/active", (204, None))
    adapter = make_adapter(api)

    adapter.start_cycle(
        CycleSettings(program=Program.ECO_40_60, temperature_c=40, spin_rpm=1400, extra_rinse=True)
    )

    [request] = api.sent("PUT", f"{BASE}/programs/active")
    assert request.headers["Content-Type"] == MEDIA_TYPE
    assert json.loads(request.content) == {
        "data": {
            "key": P + "Cotton.Eco4060",
            "options": [
                {"key": "LaundryCare.Washer.Option.Temperature", "value": T + "GC40"},
                {"key": "LaundryCare.Washer.Option.SpinSpeed", "value": S + "RPM1200"},
            ],
        }
    }


def test_start_never_exceeds_the_requested_temperature(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Wool")))
    api.on("GET", f"{BASE}/programs/available/{P}Wool", (200, definition("Wool", ["GC40"], [])))
    with pytest.raises(UnsupportedOperation, match="at or below 30 °C"):
        make_adapter(api).start_cycle(
            CycleSettings(program=Program.WOOL, temperature_c=30, spin_rpm=600)
        )
    assert not api.sent("PUT", f"{BASE}/programs/active")


def test_start_rounds_spin_down_and_falls_back_to_a_gentler_programme(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton", "EasyCare")))
    api.on(
        "GET",
        f"{BASE}/programs/available/{P}EasyCare",
        (200, definition("EasyCare", ["GC20", "GC30", "GC40"], ["RPM800", "RPM1200"])),
    )
    api.on("PUT", f"{BASE}/programs/active", (204, None))
    make_adapter(api).start_cycle(
        CycleSettings(program=Program.MIXED, temperature_c=40, spin_rpm=1000)
    )
    body = json.loads(api.sent("PUT", f"{BASE}/programs/active")[0].content)
    assert body["data"]["key"] == P + "EasyCare"
    assert body["data"]["options"][1]["value"] == S + "RPM800"


def test_programme_without_a_spin_setting_is_refused(api):
    # The machine would spin at a speed SmartWash cannot see or cap.
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton")))
    api.on("GET", f"{BASE}/programs/available/{P}Cotton", (200, definition("Cotton", ["GC60"])))
    with pytest.raises(UnsupportedOperation, match="no spin speed setting"):
        make_adapter(api).start_cycle(
            CycleSettings(program=Program.COTTON, temperature_c=60, spin_rpm=1400)
        )
    assert not api.sent("PUT", f"{BASE}/programs/active")


def test_programme_without_a_temperature_setting_is_refused(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Mix")))
    api.on("GET", f"{BASE}/programs/available/{P}Mix", (200, definition("Mix", None, ["RPM800"])))
    with pytest.raises(UnsupportedOperation, match="no temperature setting"):
        make_adapter(api).start_cycle(
            CycleSettings(program=Program.MIXED, temperature_c=30, spin_rpm=800)
        )


def test_eco_programme_keeps_its_own_temperature(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton.Eco4060")))
    api.on(
        "GET",
        f"{BASE}/programs/available/{P}Cotton.Eco4060",
        (200, definition("Cotton.Eco4060", None, ["RPM1000", "RPM1400"])),
    )
    api.on("PUT", f"{BASE}/programs/active", (204, None))
    make_adapter(api).start_cycle(
        CycleSettings(program=Program.ECO_40_60, temperature_c=40, spin_rpm=1400)
    )
    body = json.loads(api.sent("PUT", f"{BASE}/programs/active")[0].content)
    assert body["data"]["options"] == [
        {"key": "LaundryCare.Washer.Option.SpinSpeed", "value": S + "RPM1400"}
    ]


def test_empty_programme_definition_sends_explicit_capped_values(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton")))
    api.on("GET", f"{BASE}/programs/available/{P}Cotton", (200, definition("Cotton")))
    api.on("PUT", f"{BASE}/programs/active", (204, None))
    make_adapter(api).start_cycle(
        CycleSettings(program=Program.COTTON, temperature_c=30, spin_rpm=800)
    )
    body = json.loads(api.sent("PUT", f"{BASE}/programs/active")[0].content)
    assert [o["value"] for o in body["data"]["options"]] == [T + "GC30", S + "RPM800"]


def test_bedding_never_escalates_to_the_harsher_cotton_programme(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton", "Mix")))
    api.on(
        "GET",
        f"{BASE}/programs/available/{P}Mix",
        (200, definition("Mix", ["GC40"], ["RPM1000"])),
    )
    api.on("PUT", f"{BASE}/programs/active", (204, None))
    make_adapter(api).start_cycle(
        CycleSettings(program=Program.BEDDING, temperature_c=40, spin_rpm=1000)
    )
    body = json.loads(api.sent("PUT", f"{BASE}/programs/active")[0].content)
    assert body["data"]["key"] == P + "Mix"


def test_missing_programme_without_substitute_is_unsupported(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton", "EasyCare")))
    with pytest.raises(UnsupportedOperation, match="no programme for wool"):
        make_adapter(api).start_cycle(
            CycleSettings(program=Program.WOOL, temperature_c=30, spin_rpm=600)
        )


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (status_payload(remote_start=False), "Remote Start"),
        (status_payload("Run"), "already running"),
        (status_payload(door="Open"), "door"),
        (status_payload(local=True), "locally"),
        (status_payload("Inactive"), "Ready"),
    ],
)
def test_start_refuses_when_the_washer_is_not_ready(api, status, message):
    api.on("GET", f"{BASE}/status", (200, status))
    with pytest.raises(DeviceNotReady, match=message):
        make_adapter(api).start_cycle(
            CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200)
        )


def test_conflict_from_the_api_is_not_ready_with_vendor_description(api):
    api.on("GET", f"{BASE}/programs/available", (200, available("Cotton")))
    api.on(
        "GET",
        f"{BASE}/programs/available/{P}Cotton",
        (200, definition("Cotton", ["GC40"], ["RPM1200"])),
    )
    conflict = {"error": {"key": "SDK.Error.WrongOperationState", "description": "Wrong state"}}
    api.on("PUT", f"{BASE}/programs/active", (409, conflict))
    with pytest.raises(DeviceNotReady, match="Wrong state \\(SDK.Error.WrongOperationState\\)"):
        make_adapter(api).start_cycle(
            CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200)
        )


def test_pause_resume_and_stop(api):
    for command in ("PauseProgram", "ResumeProgram"):
        api.on("PUT", f"{BASE}/commands/BSH.Common.Command.{command}", (204, None))
    api.on("DELETE", f"{BASE}/programs/active", (204, None))
    adapter = make_adapter(api)
    adapter.pause()
    adapter.resume()
    adapter.stop()
    pause = api.sent("PUT", f"{BASE}/commands/BSH.Common.Command.PauseProgram")[0]
    assert json.loads(pause.content) == {
        "data": {"key": "BSH.Common.Command.PauseProgram", "value": True}
    }
    assert api.sent("PUT", f"{BASE}/commands/BSH.Common.Command.ResumeProgram")
    assert api.sent("DELETE", f"{BASE}/programs/active")


def test_expired_token_without_refresh_details_is_an_auth_error(api):
    api.on("GET", f"{BASE}/status", (401, {"error": {"key": "invalid_token"}}))
    with pytest.raises(DeviceAuthError, match="invalid_token"):
        make_adapter(api).get_state()


def test_expired_token_is_refreshed_and_cached(api, tmp_path):
    cache = tmp_path / "tokens.json"

    def status(request):
        if request.headers["Authorization"] == "Bearer token-1":
            return 401, {"error": {"key": "invalid_token", "description": "expired"}}
        return 200, status_payload()

    def token(request):
        assert dict(httpx.QueryParams(request.content.decode())) == {
            "grant_type": "refresh_token",
            "refresh_token": "refresh-1",
            "client_secret": "secret",
        }
        return 200, {"access_token": "token-2", "refresh_token": "refresh-2", "expires_in": 86400}

    api.on("GET", f"{BASE}/status", status)
    api.on("POST", "/security/oauth/token", token)
    adapter = make_adapter(
        api, refresh_token="refresh-1", client_secret="secret", token_cache_path=str(cache)
    )
    assert adapter.get_state().status is MachineStatus.IDLE
    assert json.loads(cache.read_text()) == {
        "access_token": "token-2",
        "refresh_token": "refresh-2",
    }
    assert stat.S_IMODE(os.stat(cache).st_mode) == 0o600

    again = HomeConnectAdapter(
        HomeConnectConfig(
            access_token="token-1",
            refresh_token="refresh-1",
            client_secret="secret",
            token_cache_path=str(cache),
        ),
        client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    again.connect()
    assert api.requests[-1].headers["Authorization"] == "Bearer token-2"


def test_failed_refresh_is_an_auth_error(api):
    api.on("GET", f"{BASE}/status", (401, {"error": {"key": "invalid_token"}}))
    api.on(
        "POST",
        "/security/oauth/token",
        (400, {"error": "invalid_grant", "error_description": "invalid refresh_token"}),
    )
    adapter = make_adapter(api, refresh_token="bad", client_secret="secret")
    with pytest.raises(DeviceAuthError, match="invalid refresh_token"):
        adapter.get_state()


def test_config_needs_a_token_or_refresh_credentials():
    with pytest.raises(ValidationError):
        HomeConnectConfig()
    HomeConnectConfig(refresh_token="r", client_secret="s")


def test_capabilities_offer_control_but_no_sensor_telemetry(api):
    capabilities = make_adapter(api).capabilities
    assert capabilities.select_program and capabilities.pause_resume
    assert not capabilities.fault_detection
