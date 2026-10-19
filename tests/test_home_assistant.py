import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from smartwash.devices.base import (
    DeviceAuthError,
    DeviceError,
    DeviceNotReady,
    UnsupportedOperation,
)
from smartwash.devices.home_assistant import (
    HomeAssistantAdapter,
    HomeAssistantConfig,
    ServiceCall,
    _parse_minutes,
)
from smartwash.schemas import CyclePhase, CycleSettings, MachineStatus, Program

TEMP = "laundry_care_washer_enum_type_temperature_"
SPIN = "laundry_care_washer_enum_type_spin_speed_"


class FakeHass:
    """Entity states plus a log of service calls, served like the HA REST API."""

    def __init__(self, states: dict[str, object]) -> None:
        self.states: dict[str, dict] = {}
        for entity_id, state in states.items():
            self.set(entity_id, state)
        self.calls: list[tuple[str, dict]] = []
        self.requests: list[httpx.Request] = []
        self.service_error: tuple[int, dict] | None = None

    def set(self, entity_id: str, state, **attributes) -> None:
        if isinstance(state, tuple):
            state, attributes = state
        self.states[entity_id] = {"entity_id": entity_id, "state": state, "attributes": attributes}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.headers.get("Authorization") != "Bearer ha-token":
            return httpx.Response(401, text="401: Unauthorized")
        if path == "/api/":
            return httpx.Response(200, json={"message": "API running."})
        if path == "/api/states":
            return httpx.Response(200, json=list(self.states.values()))
        if path.startswith("/api/states/"):
            entity = self.states.get(path.removeprefix("/api/states/"))
            if entity is None:
                return httpx.Response(404, json={"message": "Entity not found."})
            return httpx.Response(200, json=entity)
        if path.startswith("/api/services/") and request.method == "POST":
            if self.service_error:
                return httpx.Response(self.service_error[0], json=self.service_error[1])
            service = path.removeprefix("/api/services/").replace("/", ".")
            self.calls.append((service, json.loads(request.content)))
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"message": "Not found"})


LG_STATES = {
    "sensor.washer_current_status": "initial",
    "binary_sensor.washer_remote_start": "on",
    "sensor.washer_remaining_time": ("45", {"unit_of_measurement": "min"}),
}
LG_CONFIG = {
    "status_entity": "sensor.washer_current_status",
    "remote_start_entity": "binary_sensor.washer_remote_start",
    "remaining_entity": "sensor.washer_remaining_time",
    "start": {
        "service": "select.select_option",
        "data": {"entity_id": "select.washer_operation", "option": "start"},
    },
    "stop": {
        "service": "select.select_option",
        "data": {"entity_id": "select.washer_operation", "option": "stop"},
    },
}

HOME_CONNECT_CONFIG = {
    "status_entity": "sensor.washer_operation_state",
    "door_entity": "sensor.washer_door",
    "remote_start_entity": "binary_sensor.washer_remote_start",
    "progress_entity": "sensor.washer_program_progress",
    "program_map": {
        "cotton": "laundry_care_washer_program_cotton",
        "synthetics": "laundry_care_washer_program_easy_care",
    },
    "temperature_map": {20: TEMP + "g_c_20", 30: TEMP + "g_c_30", 40: TEMP + "g_c_40"},
    "spin_map": {800: SPIN + "r_p_m_800", 1200: SPIN + "r_p_m_1200"},
    "start": {
        "service": "home_connect.set_program_and_options",
        "data": {
            "device_id": "dev-123",
            "affects_to": "active_program",
            "program": "{program}",
            "laundry_care_washer_option_temperature": "{temperature}",
            "laundry_care_washer_option_spin_speed": "{spin}",
        },
    },
    "pause": {"service": "button.press", "data": {"entity_id": "button.washer_pause_program"}},
    "resume": {"service": "button.press", "data": {"entity_id": "button.washer_resume_program"}},
    "stop": {"service": "button.press", "data": {"entity_id": "button.washer_stop_program"}},
}
HOME_CONNECT_STATES = {
    "sensor.washer_operation_state": "ready",
    "sensor.washer_door": "closed",
    "binary_sensor.washer_remote_start": "on",
    "sensor.washer_program_progress": ("12", {"unit_of_measurement": "%"}),
}


def make_adapter(hass: FakeHass, **config) -> HomeAssistantAdapter:
    adapter = HomeAssistantAdapter(
        HomeAssistantConfig(url="http://ha.local:8123", token="ha-token", **config),
        client=httpx.Client(transport=httpx.MockTransport(hass)),
    )
    adapter.connect()
    return adapter


COTTON = CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200)


def test_lg_style_running_state():
    hass = FakeHass({**LG_STATES, "sensor.washer_current_status": "rinsing"})
    state = make_adapter(hass, **LG_CONFIG).get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.phase is CyclePhase.RINSE
    assert state.remaining_min == 45.0
    assert state.remote_start_allowed is True
    assert state.raw["sensor.washer_current_status"] == "rinsing"


@pytest.mark.parametrize(
    ("value", "status"),
    [
        ("power_off", MachineStatus.IDLE),
        ("end", MachineStatus.FINISHED),
        ("pause", MachineStatus.PAUSED),
        ("reserved", MachineStatus.DELAYED_START),
        ("error", MachineStatus.ERROR),
        ("unavailable", MachineStatus.OFFLINE),
        ("in_use", MachineStatus.RUNNING),
        ("run", MachineStatus.RUNNING),
        ("stop", MachineStatus.IDLE),
    ],
)
def test_default_status_map_covers_common_integrations(value, status):
    hass = FakeHass({**LG_STATES, "sensor.washer_current_status": value})
    assert make_adapter(hass, **LG_CONFIG).get_state().status is status


def test_status_map_override_and_unknown_states(caplog):
    hass = FakeHass({**LG_STATES, "sensor.washer_current_status": "Washing_Main"})
    adapter = make_adapter(hass, **LG_CONFIG)
    assert adapter.get_state().status is MachineStatus.IDLE
    assert "status_map" in caplog.text
    custom = make_adapter(hass, **{**LG_CONFIG, "status_map": {"Washing_Main": "running"}})
    assert custom.get_state().status is MachineStatus.RUNNING


def test_lg_style_start_needs_explicit_opt_ins():
    hass = FakeHass(LG_STATES)
    with pytest.raises(UnsupportedOperation, match="start_preselected_program"):
        make_adapter(hass, **LG_CONFIG).start_cycle(COTTON)
    preselected = make_adapter(hass, **LG_CONFIG, start_preselected_program=True)
    assert not preselected.capabilities.remote_start
    with pytest.raises(UnsupportedOperation, match="allow_uncontrolled_levels"):
        preselected.start_cycle(COTTON)
    assert hass.calls == []
    adapter = make_adapter(
        hass, **LG_CONFIG, start_preselected_program=True, allow_uncontrolled_levels=True
    )
    assert adapter.capabilities.remote_start and not adapter.capabilities.select_program
    adapter.start_cycle(COTTON)
    assert hass.calls == [
        ("select.select_option", {"entity_id": "select.washer_operation", "option": "start"})
    ]


def test_programme_choice_without_temperature_control_changes_nothing():
    hass = FakeHass(HOME_CONNECT_STATES)
    hass.set("select.washer_program", "laundry_care_washer_program_cotton")
    config = {
        **HOME_CONNECT_CONFIG,
        "program_select_entity": "select.washer_program",
        "start": {"service": "home_connect.start_selected_program", "data": {"device_id": "d"}},
    }
    adapter = make_adapter(hass, **config)
    assert not adapter.capabilities.select_program
    with pytest.raises(UnsupportedOperation, match="cannot set the temperature"):
        adapter.start_cycle(COTTON)
    assert hass.calls == []  # not even the programme select was touched


def test_unmapped_programme_never_falls_back_to_the_preselected_one():
    hass = FakeHass(HOME_CONNECT_STATES)
    adapter = make_adapter(hass, **HOME_CONNECT_CONFIG, start_preselected_program=True)
    with pytest.raises(UnsupportedOperation, match="'wool'"):
        adapter.start_cycle(CycleSettings(program=Program.WOOL, temperature_c=30, spin_rpm=600))
    assert hass.calls == []


def test_non_json_success_response_is_a_device_error():
    hass = FakeHass(HOME_CONNECT_STATES)
    adapter = make_adapter(hass, **HOME_CONNECT_CONFIG)
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))
    )
    with pytest.raises(DeviceError, match="not JSON"):
        adapter.get_state()


def test_home_connect_style_one_shot_start_fills_placeholders():
    hass = FakeHass(HOME_CONNECT_STATES)
    adapter = make_adapter(hass, **HOME_CONNECT_CONFIG)
    adapter.start_cycle(CycleSettings(program=Program.COTTON, temperature_c=60, spin_rpm=1000))
    assert hass.calls == [
        (
            "home_connect.set_program_and_options",
            {
                "device_id": "dev-123",
                "affects_to": "active_program",
                "program": "laundry_care_washer_program_cotton",
                "laundry_care_washer_option_temperature": TEMP + "g_c_40",
                "laundry_care_washer_option_spin_speed": SPIN + "r_p_m_800",
            },
        )
    ]


def test_programme_falls_back_to_a_gentler_mapping():
    hass = FakeHass(HOME_CONNECT_STATES)
    make_adapter(hass, **HOME_CONNECT_CONFIG).start_cycle(
        CycleSettings(program=Program.SPORTSWEAR, temperature_c=30, spin_rpm=800)
    )
    assert hass.calls[0][1]["program"] == "laundry_care_washer_program_easy_care"


def test_unmapped_programme_is_unsupported():
    hass = FakeHass(HOME_CONNECT_STATES)
    with pytest.raises(UnsupportedOperation, match="'wool'"):
        make_adapter(hass, **HOME_CONNECT_CONFIG).start_cycle(
            CycleSettings(program=Program.WOOL, temperature_c=30, spin_rpm=600)
        )
    assert hass.calls == []


def test_temperature_map_never_exceeds_the_request():
    hass = FakeHass(HOME_CONNECT_STATES)
    config = {**HOME_CONNECT_CONFIG, "temperature_map": {40: TEMP + "g_c_40"}}
    with pytest.raises(UnsupportedOperation, match="at or below 30"):
        make_adapter(hass, **config).start_cycle(
            CycleSettings(program=Program.COTTON, temperature_c=30, spin_rpm=800)
        )


def test_select_entities_are_set_in_order_from_their_options():
    hass = FakeHass(HOME_CONNECT_STATES)
    hass.set("select.washer_program", "laundry_care_washer_program_cotton")
    hass.set(
        "select.washer_temperature",
        TEMP + "g_c_30",
        options=[TEMP + "cold", TEMP + "g_c_20", TEMP + "g_c_30", TEMP + "g_c_40", TEMP + "g_c_60"],
    )
    hass.set("select.washer_spin_speed", SPIN + "r_p_m_800", options=[SPIN + "r_p_m_800"])
    config = {
        **HOME_CONNECT_CONFIG,
        "program_select_entity": "select.washer_program",
        "temperature_select_entity": "select.washer_temperature",
        "spin_select_entity": "select.washer_spin_speed",
        "temperature_map": {},
        "spin_map": {},
        "start": {"service": "home_connect.start_selected_program", "data": {"device_id": "d"}},
    }
    make_adapter(hass, **config).start_cycle(COTTON)
    assert hass.calls == [
        (
            "select.select_option",
            {"entity_id": "select.washer_program", "option": "laundry_care_washer_program_cotton"},
        ),
        (
            "select.select_option",
            {"entity_id": "select.washer_temperature", "option": TEMP + "g_c_40"},
        ),
        (
            "select.select_option",
            {"entity_id": "select.washer_spin_speed", "option": SPIN + "r_p_m_800"},
        ),
        ("home_connect.start_selected_program", {"device_id": "d"}),
    ]


def test_numeric_placeholders_keep_their_type_and_extras_are_called():
    hass = FakeHass(HOME_CONNECT_STATES)
    config = {
        **HOME_CONNECT_CONFIG,
        "start": {
            "service": "script.start_washer",
            "data": {"program": "{program}", "temp": "{temperature_c}", "label": "{spin_rpm} rpm"},
        },
        "extra_rinse": {"service": "switch.turn_on", "data": {"entity_id": "switch.rinse_plus"}},
    }
    make_adapter(hass, **config).start_cycle(
        CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200, extra_rinse=True)
    )
    assert hass.calls == [
        ("switch.turn_on", {"entity_id": "switch.rinse_plus"}),
        (
            "script.start_washer",
            {"program": "laundry_care_washer_program_cotton", "temp": 40, "label": "1200 rpm"},
        ),
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"binary_sensor.washer_remote_start": "off"}, "Remote start"),
        ({"sensor.washer_door": "open"}, "door"),
        ({"sensor.washer_operation_state": "run"}, "already running"),
        ({"sensor.washer_operation_state": "unavailable"}, "offline"),
    ],
)
def test_start_refuses_when_not_ready(changes, message):
    hass = FakeHass({**HOME_CONNECT_STATES, **changes})
    with pytest.raises(DeviceNotReady, match=message):
        make_adapter(hass, **HOME_CONNECT_CONFIG).start_cycle(COTTON)


def test_service_validation_error_is_not_ready():
    hass = FakeHass(HOME_CONNECT_STATES)
    hass.service_error = (400, {"message": "Remote control disabled"})
    with pytest.raises(DeviceNotReady, match="Remote control disabled"):
        make_adapter(hass, **HOME_CONNECT_CONFIG).pause()


def test_pause_resume_stop_and_unconfigured_commands():
    hass = FakeHass(HOME_CONNECT_STATES)
    adapter = make_adapter(hass, **HOME_CONNECT_CONFIG)
    adapter.pause()
    adapter.resume()
    adapter.stop()
    assert [c[1]["entity_id"] for c in hass.calls] == [
        "button.washer_pause_program",
        "button.washer_resume_program",
        "button.washer_stop_program",
    ]
    lg = make_adapter(FakeHass(LG_STATES), **LG_CONFIG)
    assert not lg.capabilities.pause_resume
    with pytest.raises(UnsupportedOperation, match="home_assistant.pause"):
        lg.pause()


def test_home_connect_state_with_progress_and_door():
    hass = FakeHass({**HOME_CONNECT_STATES, "sensor.washer_operation_state": "run"})
    hass.set("sensor.washer_door", "locked")
    state = make_adapter(hass, **HOME_CONNECT_CONFIG).get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.progress_pct == 12.0
    assert state.door_open is False


def test_power_telemetry_while_running():
    hass = FakeHass({**LG_STATES, "sensor.washer_current_status": "spinning"})
    hass.set("sensor.washer_power", "0.35", unit_of_measurement="kW")
    adapter = make_adapter(hass, **LG_CONFIG, power_entity="sensor.washer_power")
    assert adapter.capabilities.telemetry_fields == {"power_w"}
    assert adapter.read_telemetry() == []
    adapter.get_state()
    [sample] = adapter.read_telemetry()
    assert sample.power_w == pytest.approx(350.0)
    assert sample.phase is CyclePhase.SPIN


def test_bad_token_is_an_auth_error():
    adapter = HomeAssistantAdapter(
        HomeAssistantConfig(token="wrong", status_entity="sensor.washer_current_status"),
        client=httpx.Client(transport=httpx.MockTransport(FakeHass(LG_STATES))),
    )
    with pytest.raises(DeviceAuthError):
        adapter.connect()


def test_missing_status_entity_is_a_device_error():
    with pytest.raises(DeviceError, match="404"):
        make_adapter(FakeHass({}), **LG_CONFIG)


def test_list_devices_finds_washer_entities():
    hass = FakeHass({**LG_STATES, "light.kitchen": "on"})
    hass.set("select.washer_operation", "start", options=["start", "stop"])
    found = {d["id"]: d for d in make_adapter(hass, **LG_CONFIG).list_devices()}
    assert "light.kitchen" not in found
    assert found["select.washer_operation"]["options"] == ["start", "stop"]


def test_parse_minutes_formats():
    soon = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    assert 29.0 <= _parse_minutes(soon, None) <= 30.0
    assert _parse_minutes("1:30", None) == 90.0
    assert _parse_minutes("5400", "s") == 90.0
    assert _parse_minutes("1.5", "h") == 90.0
    assert _parse_minutes("unknown", None) is None


def test_service_names_are_validated():
    with pytest.raises(ValueError, match="domain.service"):
        ServiceCall(service="press button")
