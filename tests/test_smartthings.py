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
from smartwash.devices.smartthings import SmartThingsAdapter, SmartThingsConfig
from smartwash.schemas import CyclePhase, CycleSettings, MachineStatus, Program

DEVICE = "b854ca5f-dc54-140d-6349-758b4d973c41"
STATUS = f"/v1/devices/{DEVICE}/status"
COMMANDS = f"/v1/devices/{DEVICE}/commands"
HEALTH = f"/v1/devices/{DEVICE}/health"

EU_SPINS = ["rinseHold", "noSpin", "400", "800", "1000", "1200", "1400"]


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
            return httpx.Response(404, json={"error": {"code": "NotFound", "message": "no route"}})
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        status, body = item(request) if callable(item) else item
        return httpx.Response(status, json=body)

    def commands(self) -> list[dict]:
        sent = [r for r in self.requests if r.method == "POST" and r.url.path == COMMANDS]
        return [json.loads(r.content)["commands"][0] for r in sent]


def device(device_id=DEVICE, category="Washer", capabilities=("washerOperatingState",)):
    return {
        "deviceId": device_id,
        "label": "Machine à Laver" if category == "Washer" else "TV",
        "manufacturerName": "Samsung Electronics",
        "presentationId": "DA-WM-WM-01011",
        "ocf": {"modelNumber": "DA_WM_TP1_21_COMMON|20374641"},
        "components": [
            {
                "id": "main",
                "categories": [{"name": category, "categoryType": "manufacturer"}],
                "capabilities": [{"id": c, "version": 1} for c in capabilities],
            }
        ],
    }


def cycle(code, temperatures, spins=EU_SPINS, temperature_default="40", rinse_default="2"):
    return {
        "cycle": code,
        "cycleType": "washingOnly",
        "supportedOptions": {
            "spinLevel": {"raw": "A67F", "default": "1400", "options": spins},
            "rinseCycle": {
                "raw": "923F",
                "default": rinse_default,
                "options": ["0", "1", "2", "3", "4", "5"],
            },
            "waterTemperature": {
                "raw": "847E",
                "default": temperature_default,
                "options": temperatures,
            },
        },
    }


def status(
    machine="stop",
    job="none",
    remote="true",
    disabled=(),
    cycles=None,
    current="Table_02_Course_1C",
    temperature="40",
    spin="1000",
):
    def attr(value, unit=None):
        return {"value": value, **({"unit": unit} if unit else {})}

    cycles = cycles or [
        cycle("1C", [], temperature_default="none"),
        cycle("1B", ["cold", "20", "30", "40", "60", "90"]),
        cycle("2B", ["40"]),
    ]
    return {
        "components": {
            "main": {
                "custom.disabledCapabilities": {"disabledCapabilities": attr(list(disabled))},
                "washerOperatingState": {
                    "machineState": attr(machine),
                    "washerJobState": attr(job),
                    "supportedMachineStates": attr(["stop", "run", "pause"]),
                    "completionTime": attr("2025-04-25T10:34:12Z"),
                },
                "samsungce.washerOperatingState": {
                    "operatingState": attr(
                        {"run": "running", "pause": "paused"}.get(machine, "ready")
                    ),
                    "washerJobState": attr(job),
                    "progress": attr(40, "%"),
                    "remainingTime": attr(100, "min"),
                },
                "samsungce.washerCycle": {
                    "washerCycle": attr(current),
                    "referenceTable": attr({"id": "Table_02"}),
                    "supportedCycles": attr(cycles),
                },
                "custom.washerWaterTemperature": {
                    "washerWaterTemperature": attr(temperature),
                    "supportedWasherWaterTemperature": attr(
                        ["none", "cold", "20", "30", "40", "60", "90"]
                    ),
                },
                "custom.washerSpinLevel": {
                    "washerSpinLevel": attr(spin),
                    "supportedWasherSpinLevel": attr(EU_SPINS),
                },
                "custom.washerRinseCycles": {
                    "washerRinseCycles": attr("2"),
                    "supportedWasherRinseCycles": attr(["0", "1", "2", "3", "4", "5"]),
                },
                "remoteControlStatus": {"remoteControlEnabled": attr(remote)},
                "switch": {"switch": attr("on")},
            }
        }
    }


ACCEPTED = (200, {"results": [{"id": "a1", "status": "ACCEPTED"}]})


@pytest.fixture
def api():
    fake = FakeApi()
    fake.on("GET", "/v1/devices", (200, {"items": [device("tv-1", "Television", ()), device()]}))
    fake.on("GET", STATUS, (200, status()))
    fake.on("GET", HEALTH, (200, {"deviceId": DEVICE, "state": "ONLINE"}))
    fake.on("POST", COMMANDS, ACCEPTED)
    return fake


def make_adapter(api, **overrides):
    config = SmartThingsConfig(**{"token": "pat-1", "min_poll_interval_s": 0, **overrides})
    adapter = SmartThingsAdapter(config, client=httpx.Client(transport=httpx.MockTransport(api)))
    adapter.connect()
    return adapter


COTTON_40 = CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200)


def test_connect_finds_the_washer_across_pages():
    api = FakeApi()
    next_page = "https://api.smartthings.com/v1/devices/page2"
    api.on(
        "GET",
        "/v1/devices",
        (
            200,
            {"items": [device("tv-1", "Television", ())], "_links": {"next": {"href": next_page}}},
        ),
    )
    api.on("GET", "/v1/devices/page2", (200, {"items": [device()]}))
    adapter = make_adapter(api)
    assert adapter.describe()["id"] == DEVICE
    assert api.requests[0].headers["Authorization"] == "Bearer pat-1"


def test_connect_without_washer_lists_devices():
    api = FakeApi()
    api.on("GET", "/v1/devices", (200, {"items": [device("tv-1", "Television", ())]}))
    with pytest.raises(DeviceError, match="found: TV"):
        make_adapter(api)


def test_list_devices_shows_cycle_ids_for_building_the_map(api):
    [washer] = make_adapter(api).list_devices()
    assert washer["current_cycle"] == "Table_02_Course_1C"
    assert washer["supported_cycles"] == ["1B", "1C", "2B"]


def test_running_state(api):
    api.on("GET", STATUS, (200, status(machine="run", job="rinse", current="Table_02_Course_1B")))
    state = make_adapter(api, cycle_map={"cotton": "1B"}).get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.phase is CyclePhase.RINSE
    assert state.program is Program.COTTON
    assert state.vendor_program == "Table_02_Course_1B"
    assert (state.temperature_c, state.spin_rpm) == (40, 1000)
    assert (state.progress_pct, state.remaining_min) == (40, 100.0)
    assert state.remote_start_allowed is True


@pytest.mark.parametrize(
    ("machine", "job", "expected"),
    [
        ("stop", "none", MachineStatus.IDLE),
        ("pause", "wash", MachineStatus.PAUSED),
        ("run", "delayWash", MachineStatus.DELAYED_START),
        ("stop", "finish", MachineStatus.FINISHED),
    ],
)
def test_status_mapping(api, machine, job, expected):
    api.on("GET", STATUS, (200, status(machine=machine, job=job)))
    assert make_adapter(api).get_state().status is expected


def test_offline_device(api):
    api.on("GET", HEALTH, (200, {"deviceId": DEVICE, "state": "OFFLINE"}))
    assert make_adapter(api).get_state().status is MachineStatus.OFFLINE


def test_start_selects_cycle_options_then_runs(api):
    adapter = make_adapter(api, cycle_map={"cotton": "Table_02_Course_1B"})
    adapter.start_cycle(
        CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200, extra_rinse=True)
    )
    assert api.commands() == [
        {
            "component": "main",
            "capability": "samsungce.washerCycle",
            "command": "setWasherCycle",
            "arguments": ["Table_02_Course_1B"],
        },
        {
            "component": "main",
            "capability": "custom.washerWaterTemperature",
            "command": "setWasherWaterTemperature",
            "arguments": ["40"],
        },
        {
            "component": "main",
            "capability": "custom.washerSpinLevel",
            "command": "setWasherSpinLevel",
            "arguments": ["1200"],
        },
        {
            "component": "main",
            "capability": "custom.washerRinseCycles",
            "command": "setWasherRinseCycles",
            "arguments": ["3"],
        },
        {
            "component": "main",
            "capability": "washerOperatingState",
            "command": "setMachineState",
            "arguments": ["run"],
        },
    ]


def test_short_cycle_ids_are_expanded_and_values_rounded_down(api):
    make_adapter(api, cycle_map={"cotton": "1B"}).start_cycle(
        CycleSettings(program=Program.COTTON, temperature_c=30, spin_rpm=1000)
    )
    sent = {c["command"]: c.get("arguments") for c in api.commands()}
    assert sent["setWasherCycle"] == ["Table_02_Course_1B"]
    assert sent["setWasherWaterTemperature"] == ["30"]
    assert sent["setWasherSpinLevel"] == ["1000"]


def test_cycle_that_only_runs_hotter_is_refused_before_any_command(api):
    with pytest.raises(UnsupportedOperation, match="at or below 30 °C"):
        make_adapter(api, cycle_map={"synthetics": "2B"}).start_cycle(
            CycleSettings(program=Program.SYNTHETICS, temperature_c=30, spin_rpm=800)
        )
    assert api.commands() == []


def test_fixed_temperature_cycle_keeps_the_machine_default(api):
    make_adapter(api, cycle_map={"wool": "1C"}).start_cycle(
        CycleSettings(program=Program.WOOL, temperature_c=30, spin_rpm=600)
    )
    commands = [c["command"] for c in api.commands()]
    assert commands == ["setWasherCycle", "setWasherSpinLevel", "setMachineState"]


def test_word_levels_are_mapped_conservatively(api):
    us_cycle = cycle(
        "01",
        ["tapCold", "cold", "warm", "hot", "extraHot"],
        spins=["rinseHold", "noSpin", "low", "medium", "high", "extraHigh"],
    )
    api.on("GET", STATUS, (200, status(cycles=[us_cycle], current="Table_02_Course_01")))
    make_adapter(api, cycle_map={"cotton": "01"}).start_cycle(COTTON_40)
    sent = {c["command"]: c.get("arguments") for c in api.commands()}
    assert sent["setWasherWaterTemperature"] == ["warm"]
    assert sent["setWasherSpinLevel"] == ["medium"]


def test_unmapped_programme_lists_supported_cycles(api):
    with pytest.raises(UnsupportedOperation, match="cycle_map.cotton.*1B, 1C, 2B"):
        make_adapter(api).start_cycle(COTTON_40)


def test_fallback_programme_mapping(api):
    make_adapter(api, cycle_map={"synthetics": "1B"}).start_cycle(
        CycleSettings(program=Program.SPORTSWEAR, temperature_c=30, spin_rpm=800)
    )
    assert api.commands()[0]["arguments"] == ["Table_02_Course_1B"]


def test_preselected_cycle_start_still_caps_temperature_and_spin(api):
    api.on("GET", STATUS, (200, status(current="Table_02_Course_1B")))
    make_adapter(api, start_preselected_program=True).start_cycle(
        CycleSettings(program=Program.COTTON, temperature_c=30, spin_rpm=800)
    )
    sent = [(c["command"], c.get("arguments")) for c in api.commands()]
    assert sent == [
        ("setWasherWaterTemperature", ["30"]),
        ("setWasherSpinLevel", ["800"]),
        ("setMachineState", ["run"]),
    ]


def test_remote_control_off_is_not_ready(api):
    api.on("GET", STATUS, (200, status(remote="false")))
    with pytest.raises(DeviceNotReady, match="Smart Control"):
        make_adapter(api, cycle_map={"cotton": "1B"}).start_cycle(COTTON_40)


def test_already_running_is_not_ready(api):
    api.on("GET", STATUS, (200, status(machine="run", job="wash")))
    with pytest.raises(DeviceNotReady, match="already running"):
        make_adapter(api, cycle_map={"cotton": "1B"}).start_cycle(COTTON_40)


def test_samsung_operating_state_is_used_when_the_standard_one_is_disabled(api):
    api.on("GET", STATUS, (200, status(disabled=["washerOperatingState"])))
    adapter = make_adapter(api, cycle_map={"cotton": "1B"})
    adapter.start_cycle(COTTON_40)
    adapter.pause()
    adapter.stop()
    tail = [(c["capability"], c["command"], c.get("arguments")) for c in api.commands()[-3:]]
    assert tail == [
        ("samsungce.washerOperatingState", "start", None),
        ("samsungce.washerOperatingState", "pause", None),
        ("samsungce.washerOperatingState", "cancel", None),
    ]


def test_pause_resume_stop_use_set_machine_state(api):
    adapter = make_adapter(api)
    adapter.pause()
    adapter.resume()
    adapter.stop()
    assert [c["arguments"] for c in api.commands()] == [["pause"], ["run"], ["stop"]]


def test_failed_command_result_raises(api):
    api.on("POST", COMMANDS, (200, {"results": [{"id": "a1", "status": "FAILED"}]}))
    with pytest.raises(DeviceError, match="rejected"):
        make_adapter(api).pause()


def test_expired_token_mentions_pat_lifetime(api):
    api.on("GET", HEALTH, (401, {"error": {"code": "Unauthorized", "message": "Bad token"}}))
    with pytest.raises(DeviceAuthError, match="24 hours"):
        make_adapter(api).get_state()


def test_unprocessable_command_is_not_ready(api):
    error = {"error": {"code": "ConstraintViolationError", "message": "Invalid", "details": []}}
    api.on("POST", COMMANDS, (422, error))
    with pytest.raises(DeviceNotReady, match="ConstraintViolationError"):
        make_adapter(api).pause()


def test_cycle_map_keys_must_be_programmes():
    with pytest.raises(ValidationError, match="unknown programme"):
        SmartThingsConfig(token="t", cycle_map={"jeans": "1B"})


def test_capabilities_reflect_the_cycle_map(api):
    assert not make_adapter(api).capabilities.remote_start
    mapped = make_adapter(api, cycle_map={"cotton": "1B"}).capabilities
    assert mapped.remote_start and mapped.select_program and not mapped.fault_detection
