from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode
from pydantic import ValidationError

from smartwash.devices.base import DeviceAuthError, DeviceError, DeviceNotReady
from smartwash.devices.mqtt import SENSOR_FIELDS, MqttConfig, MqttWasherAdapter, create_client
from smartwash.devices.mqtt_device import VirtualMqttWasher
from smartwash.schemas import (
    FAULT_DETECTION_FIELDS,
    CyclePhase,
    CycleSettings,
    FaultType,
    MachineStatus,
    Program,
)

BASE = "home/washer"
CONNACK_OK = ReasonCode(PacketTypes.CONNACK, "Success")
CONNACK_NOT_AUTHORISED = ReasonCode(PacketTypes.CONNACK, "Not authorized")
COTTON_40 = CycleSettings(program=Program.COTTON, temperature_c=40, spin_rpm=1200)
QUICK_30 = CycleSettings(program=Program.QUICK, temperature_c=30, spin_rpm=1000)
RUNNING_STATE = {
    "status": "running",
    "phase": "wash",
    "program": "cotton",
    "vendor_program": "Cotton",
    "temperature_c": 40,
    "spin_rpm": 1200,
    "progress_pct": 12.5,
    "remaining_min": 95,
    "door_open": False,
    "remote_start_allowed": True,
    "error_code": None,
}


def telemetry(timestamp: float, **overrides: Any) -> dict[str, Any]:
    sample = {
        "timestamp": timestamp,
        "phase": "wash",
        "drum_rpm": 49.2,
        "target_rpm": 50.0,
        "motor_current_a": 0.81,
        "vibration_g": 0.036,
        "water_temp_c": 38.9,
        "target_temp_c": 40.0,
        "water_level_l": 10.4,
        "power_w": 2130.0,
    }
    return {**sample, **overrides}


class FakeClient:
    """Stands in for paho's client: records what the adapter does and lets tests inject messages."""

    def __init__(
        self,
        *,
        connack: ReasonCode = CONNACK_OK,
        auto_connack: bool = True,
        connect_error: Exception | None = None,
    ) -> None:
        self.on_connect: Callable[..., None] | None = None
        self.on_disconnect: Callable[..., None] | None = None
        self.on_message: Callable[..., None] | None = None
        self.connack = connack
        self.auto_connack = auto_connack
        self.connect_error = connect_error
        self.responder: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
        self.publish_rc = 0
        self.address: tuple[str, int, int] | None = None
        self.subscriptions: list[tuple[str, int]] = []
        self.published: list[SimpleNamespace] = []
        self.loop_running = False
        self.disconnected = False

    def connect(self, host: str, port: int = 1883, keepalive: int = 60) -> int:
        if self.connect_error is not None:
            raise self.connect_error
        self.address = (host, port, keepalive)
        return 0

    def loop_start(self) -> None:
        self.loop_running = True
        if self.auto_connack:
            self.on_connect(self, None, None, self.connack, None)

    def loop_stop(self) -> None:
        self.loop_running = False

    def disconnect(self) -> int:
        self.disconnected = True
        return 0

    def subscribe(self, topics: Any, qos: int = 0) -> tuple[int, int]:
        self.subscriptions.extend(topics if isinstance(topics, list) else [(topics, qos)])
        return 0, 1

    def publish(
        self, topic: str, payload: Any = None, qos: int = 0, retain: bool = False
    ) -> SimpleNamespace:
        self.published.append(SimpleNamespace(topic=topic, payload=payload, qos=qos, retain=retain))
        if self.publish_rc == 0 and self.responder is not None and topic.endswith("/command"):
            ack = self.responder(json.loads(payload))
            if ack is not None:
                self.deliver(topic.rsplit("/", 1)[0] + "/ack", ack)
        return SimpleNamespace(rc=self.publish_rc, mid=len(self.published))

    # test helpers
    def deliver(self, topic: str, payload: Any) -> None:
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        if isinstance(payload, str):
            payload = payload.encode()
        self.on_message(self, None, SimpleNamespace(topic=topic, payload=payload, retain=False))

    def drop_connection(self) -> None:
        self.on_disconnect(self, None, None, "connection lost", None)

    def commands(self) -> list[dict[str, Any]]:
        return [json.loads(m.payload) for m in self.published if m.topic == f"{BASE}/command"]


def ok(command: dict[str, Any]) -> dict[str, Any]:
    return {"id": command["id"], "ok": True, "error": None}


@pytest.fixture
def fake() -> FakeClient:
    return FakeClient()


def make_adapter(fake: FakeClient, **config: Any) -> MqttWasherAdapter:
    cfg = MqttConfig(base_topic=BASE, **config)
    adapter = MqttWasherAdapter(cfg, client_factory=lambda _: fake, initial_state_wait_s=0)
    adapter.connect()
    return adapter


# ------------------------------------------------------------------ unit tests


def test_closed_adapter_does_not_silently_reconnect(fake: FakeClient) -> None:
    created = []

    def factory(_: Any) -> FakeClient:
        created.append(fake)
        return fake

    adapter = MqttWasherAdapter(
        MqttConfig(base_topic=BASE), client_factory=factory, initial_state_wait_s=0
    )
    adapter.connect()
    adapter.close()
    with pytest.raises(DeviceError, match="closed"):
        adapter.get_state()
    with pytest.raises(DeviceError, match="closed"):
        adapter.read_telemetry()
    assert len(created) == 1
    adapter.connect()  # an explicit connect opens it again
    assert len(created) == 2


def test_connect_subscribes_to_device_topics(fake: FakeClient) -> None:
    make_adapter(fake)
    assert fake.address == ("localhost", 1883, 30)
    assert fake.loop_running
    assert sorted(fake.subscriptions) == sorted(
        (f"{BASE}/{name}", 1) for name in ("availability", "state", "telemetry", "ack")
    )


def test_state_message_is_parsed(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.deliver(f"{BASE}/state", RUNNING_STATE)
    state = adapter.get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.phase is CyclePhase.WASH
    assert state.program is Program.COTTON
    assert (state.temperature_c, state.spin_rpm, state.remaining_min) == (40, 1200, 95.0)
    assert state.remote_start_allowed is True


def test_state_tolerates_vendor_specific_values(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.deliver(
        f"{BASE}/state",
        {"status": "running", "program": "Baumwolle 40", "phase": "soak", "drum_temp": 41},
    )
    state = adapter.get_state()
    assert state.status is MachineStatus.RUNNING
    assert state.program is None
    assert state.vendor_program == "Baumwolle 40"
    assert state.phase is None
    assert state.raw == {"drum_temp": 41}


def test_invalid_state_messages_are_ignored(
    fake: FakeClient, caplog: pytest.LogCaptureFixture
) -> None:
    adapter = make_adapter(fake)
    fake.deliver(f"{BASE}/state", RUNNING_STATE)
    fake.deliver(f"{BASE}/state", "{not json")
    fake.deliver(f"{BASE}/state", {"status": "exploded"})
    fake.deliver(f"{BASE}/state", ["running"])
    assert adapter.get_state().status is MachineStatus.RUNNING
    assert caplog.text.count("ignoring invalid state message") == 3


def test_state_is_offline_until_the_device_reports(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    state = adapter.get_state()
    assert state.status is MachineStatus.OFFLINE
    assert "no state" in state.raw["reason"]

    fake.deliver(f"{BASE}/state", RUNNING_STATE)
    fake.deliver(f"{BASE}/availability", "offline")
    assert adapter.get_state().status is MachineStatus.OFFLINE
    fake.deliver(f"{BASE}/availability", "online")
    assert adapter.get_state().status is MachineStatus.RUNNING

    fake.deliver(f"{BASE}/state", b"")  # retained state cleared
    assert adapter.get_state().status is MachineStatus.OFFLINE


def test_state_is_offline_while_the_broker_link_is_down(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.deliver(f"{BASE}/state", RUNNING_STATE)
    fake.drop_connection()
    state = adapter.get_state()
    assert state.status is MachineStatus.OFFLINE
    assert "broker" in state.raw["reason"]
    fake.on_connect(fake, None, None, CONNACK_OK, None)  # paho reconnected
    assert adapter.get_state().status is MachineStatus.RUNNING


def test_start_cycle_sends_command_and_waits_for_ack(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.responder = ok
    before = time.time()
    adapter.start_cycle(COTTON_40, load_kg=3.5)

    (message,) = [m for m in fake.published if m.topic == f"{BASE}/command"]
    assert (message.qos, message.retain) == (0, False)  # never redelivered late
    command = json.loads(message.payload)
    assert command["command"] == "start"
    assert command["settings"] == COTTON_40.model_dump(mode="json")
    assert command["load_kg"] == 3.5
    assert len(command["id"]) == 32
    assert before - 1 <= command["sent_at"] <= time.time() + 1


@pytest.mark.parametrize("method", ["pause", "resume", "stop"])
def test_simple_commands(fake: FakeClient, method: str) -> None:
    adapter = make_adapter(fake)
    fake.responder = ok
    getattr(adapter, method)()
    (command,) = fake.commands()
    assert command["command"] == method
    assert "settings" not in command


def test_refused_command_raises_device_not_ready(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.responder = lambda command: {"id": command["id"], "ok": False, "error": "door is open"}
    with pytest.raises(DeviceNotReady, match="door is open"):
        adapter.start_cycle(COTTON_40)


def test_missing_ack_raises_device_error(fake: FakeClient) -> None:
    adapter = make_adapter(fake, ack_timeout_s=0.1)
    with pytest.raises(DeviceError, match="no acknowledgement"):
        adapter.pause()
    assert adapter._pending == {}


def test_ack_for_another_command_is_ignored(fake: FakeClient) -> None:
    adapter = make_adapter(fake, ack_timeout_s=0.1)
    fake.responder = lambda command: {"id": "someone-else", "ok": True}
    with pytest.raises(DeviceError, match="no acknowledgement"):
        adapter.stop()


def test_fire_and_forget_does_not_wait(fake: FakeClient) -> None:
    adapter = make_adapter(fake, ack_timeout_s=0)
    started = time.monotonic()
    adapter.start_cycle(QUICK_30)
    assert time.monotonic() - started < 0.5
    assert [c["command"] for c in fake.commands()] == ["start"]


def test_commands_are_refused_while_disconnected(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.responder = ok
    fake.drop_connection()
    with pytest.raises(DeviceError, match="not connected"):
        adapter.start_cycle(COTTON_40)
    assert fake.commands() == []  # never queued for later delivery


def test_publish_failure_raises_device_error(fake: FakeClient) -> None:
    adapter = make_adapter(fake)
    fake.publish_rc = 4  # MQTT_ERR_NO_CONN
    with pytest.raises(DeviceError, match="could not send 'stop'"):
        adapter.stop()


def test_telemetry_is_buffered_until_read(
    fake: FakeClient, caplog: pytest.LogCaptureFixture
) -> None:
    adapter = make_adapter(fake)
    fake.deliver(f"{BASE}/telemetry", telemetry(1.0))
    fake.deliver(f"{BASE}/telemetry", [telemetry(2.0), telemetry(3.0, phase="spin")])
    fake.deliver(f"{BASE}/telemetry", "not json")
    fake.deliver(f"{BASE}/telemetry", [telemetry(4.0), {"phase": "wash"}, telemetry(5.0)])

    samples = adapter.read_telemetry()
    assert [s.timestamp for s in samples] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert samples[2].phase is CyclePhase.SPIN
    assert samples[0].power_w == 2130.0
    assert adapter.read_telemetry() == []
    assert "not valid JSON" in caplog.text
    assert "dropped 1 invalid telemetry sample(s) of 3" in caplog.text


def test_telemetry_buffer_is_bounded(fake: FakeClient) -> None:
    cfg = MqttConfig(base_topic=BASE)
    adapter = MqttWasherAdapter(
        cfg, client_factory=lambda _: fake, initial_state_wait_s=0, max_buffered_samples=3
    )
    adapter.connect()
    fake.deliver(f"{BASE}/telemetry", [telemetry(float(t)) for t in range(5)])
    assert [s.timestamp for s in adapter.read_telemetry()] == [2.0, 3.0, 4.0]


def test_unreachable_broker_raises_device_error() -> None:
    adapter = MqttWasherAdapter(MqttConfig(port=_free_port()), initial_state_wait_s=0)
    with pytest.raises(DeviceError, match=r"cannot reach the MQTT broker at localhost:\d+"):
        adapter.connect()


def test_rejected_credentials_raise_device_auth_error() -> None:
    fake = FakeClient(connack=CONNACK_NOT_AUTHORISED)
    with pytest.raises(DeviceAuthError, match="refused the connection: Not authorized"):
        make_adapter(fake, username="washer", password="wrong")
    assert fake.disconnected and not fake.loop_running


def test_silent_broker_times_out() -> None:
    fake = FakeClient(auto_connack=False)
    with pytest.raises(DeviceError, match="did not accept the connection within 0.1 s"):
        make_adapter(fake, connect_timeout_s=0.1)
    assert not fake.loop_running


def test_connect_is_lazy_and_close_is_idempotent(fake: FakeClient) -> None:
    adapter = MqttWasherAdapter(
        MqttConfig(base_topic=BASE), client_factory=lambda _: fake, initial_state_wait_s=0
    )
    assert adapter.get_state().status is MachineStatus.OFFLINE  # connected on demand
    assert fake.loop_running
    adapter.close()
    adapter.close()
    assert fake.disconnected and not fake.loop_running


def test_capabilities_follow_config() -> None:
    full = MqttWasherAdapter(MqttConfig()).capabilities
    assert full.telemetry_fields == SENSOR_FIELDS
    assert full.fault_detection
    partial = MqttWasherAdapter(MqttConfig(telemetry_fields=["power_w"])).capabilities
    assert partial.telemetry_fields == {"power_w"}
    assert not partial.fault_detection


def test_config_validation() -> None:
    assert MqttConfig(base_topic="home/washer/").base_topic == "home/washer"
    with pytest.raises(ValidationError, match="wildcards"):
        MqttConfig(base_topic="home/+/washer")
    with pytest.raises(ValidationError, match="unknown telemetry fields"):
        MqttConfig(telemetry_fields=["drum_rpm", "spin_speed"])


def test_describe(fake: FakeClient) -> None:
    adapter = make_adapter(fake, host="broker.lan", port=1884)
    fake.deliver(f"{BASE}/availability", "online")
    assert adapter.describe() == {
        "adapter": "mqtt",
        "host": "broker.lan",
        "port": 1884,
        "base_topic": BASE,
        "broker_connected": True,
        "device_online": True,
    }


# ----------------------------------------------------- tests with a real broker

MOSQUITTO = shutil.which("mosquitto") or next(
    (p for p in ("/usr/sbin/mosquitto", "/usr/local/sbin/mosquitto") if os.path.exists(p)), None
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(predicate: Callable[[], Any], timeout: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(0.02)


@pytest.fixture(scope="module")
def broker_port() -> Iterator[int]:
    if MOSQUITTO is None:
        pytest.skip("mosquitto is not installed")
    port = _free_port()
    process = subprocess.Popen(
        [MOSQUITTO, "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.skip("could not start a mosquitto broker")
                time.sleep(0.05)
        yield port
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.fixture
def broker_config(broker_port: int) -> MqttConfig:
    return MqttConfig(
        host="127.0.0.1",
        port=broker_port,
        base_topic=f"test/{uuid.uuid4().hex[:8]}",
        ack_timeout_s=3.0,
        connect_timeout_s=5.0,
    )


def status_is(adapter: MqttWasherAdapter, status: MachineStatus) -> Callable[[], bool]:
    return lambda: adapter.get_state().status is status


def test_virtual_device_end_to_end(broker_config: MqttConfig) -> None:
    device = VirtualMqttWasher(broker_config, time_scale=600, seed=1)
    device.start()
    adapter = MqttWasherAdapter(broker_config)
    try:
        adapter.connect()
        assert adapter.get_state().status is MachineStatus.IDLE  # retained state on connect
        assert adapter.describe()["device_online"] is True

        adapter.start_cycle(COTTON_40, load_kg=4.0)
        wait_for(status_is(adapter, MachineStatus.RUNNING))
        state = adapter.get_state()
        assert state.program is Program.COTTON
        assert state.temperature_c == 40

        with pytest.raises(DeviceNotReady, match="already running"):
            adapter.start_cycle(QUICK_30)
        with pytest.raises(DeviceNotReady, match="not paused"):
            adapter.resume()

        samples: list[Any] = []
        wait_for(lambda: samples.extend(adapter.read_telemetry()) or len(samples) >= 20)
        assert adapter.capabilities.fault_detection
        for sample in samples:
            assert all(getattr(sample, name) is not None for name in FAULT_DETECTION_FIELDS)
        timestamps = [s.timestamp for s in samples]
        assert timestamps == sorted(timestamps)

        adapter.pause()
        wait_for(status_is(adapter, MachineStatus.PAUSED))
        with pytest.raises(DeviceNotReady, match="no running cycle"):
            adapter.pause()
        adapter.resume()
        wait_for(status_is(adapter, MachineStatus.RUNNING))
        adapter.stop()
        wait_for(status_is(adapter, MachineStatus.IDLE))
    finally:
        device.stop()
    try:
        wait_for(status_is(adapter, MachineStatus.OFFLINE))
        assert adapter.get_state().raw["reason"] == "device reported offline"
    finally:
        adapter.close()


def test_cycle_runs_to_completion_over_mqtt(broker_config: MqttConfig) -> None:
    device = VirtualMqttWasher(broker_config, time_scale=3000, seed=2)
    device.start()
    adapter = MqttWasherAdapter(broker_config)
    try:
        adapter.start_cycle(QUICK_30, load_kg=2.5)
        samples: list[Any] = []

        def finished() -> bool:
            samples.extend(adapter.read_telemetry())
            return adapter.get_state().status is MachineStatus.FINISHED

        wait_for(finished, timeout=10.0)
        assert adapter.get_state().progress_pct == 100.0
        phases = {s.phase for s in samples}
        assert {CyclePhase.FILL, CyclePhase.WASH, CyclePhase.RINSE, CyclePhase.DRAIN} <= phases
        assert CyclePhase.SPIN in phases
    finally:
        device.stop()
        adapter.close()


def test_device_errors_surface_through_mqtt(broker_config: MqttConfig) -> None:
    device = VirtualMqttWasher(
        broker_config, time_scale=3000, fault=FaultType.DRAIN_BLOCKAGE, fault_severity=1.4, seed=3
    )
    device.start()
    adapter = MqttWasherAdapter(broker_config)
    try:
        adapter.start_cycle(QUICK_30)
        wait_for(status_is(adapter, MachineStatus.ERROR), timeout=10.0)
        assert adapter.get_state().error_code == "E_DRAIN"
        assert "fault" not in adapter.get_state().raw  # the device does not leak its ground truth
        with pytest.raises(DeviceNotReady, match="error E_DRAIN"):
            adapter.start_cycle(QUICK_30)
        adapter.stop()
        wait_for(status_is(adapter, MachineStatus.IDLE))
    finally:
        device.stop()
        adapter.close()


def test_last_will_marks_a_vanished_device_offline(broker_config: MqttConfig) -> None:
    availability, state = broker_config.topic("availability"), broker_config.topic("state")
    device = create_client(broker_config, "test-dying-device", will_topic=availability)
    device.connect(broker_config.host, broker_config.port, keepalive=10)
    device.loop_start()
    device.publish(availability, "online", qos=1, retain=True).wait_for_publish(2.0)
    device.publish(state, json.dumps({"status": "idle"}), qos=1, retain=True).wait_for_publish(2.0)
    adapter = MqttWasherAdapter(broker_config)
    try:
        adapter.connect()
        wait_for(status_is(adapter, MachineStatus.IDLE))
        device.loop_stop()
        device.socket().close()  # vanish without a clean DISCONNECT: the broker sends the will
        wait_for(status_is(adapter, MachineStatus.OFFLINE))
    finally:
        adapter.close()
