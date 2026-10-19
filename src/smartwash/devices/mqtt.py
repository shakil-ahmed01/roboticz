"""MQTT connection for DIY / ESP32 washing machines and anything bridged to MQTT.

The wire protocol is specified in ``docs/mqtt-protocol.md``; ``mqtt_device.VirtualMqttWasher`` is a
reference implementation of the device side.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..schemas import (
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
    WashingMachineAdapter,
)

logger = logging.getLogger(__name__)

AVAILABILITY_TOPIC = "availability"
STATE_TOPIC = "state"
TELEMETRY_TOPIC = "telemetry"
COMMAND_TOPIC = "command"
ACK_TOPIC = "ack"
ONLINE = "online"
OFFLINE = "offline"

SENSOR_FIELDS: frozenset[str] = frozenset(TelemetrySample.model_fields) - {"timestamp", "phase"}
"""Every sensor field a device can publish (the fault detector needs all but ``target_temp_c``)."""

_PROGRAMS = {program.value for program in Program}
_PHASES = {phase.value for phase in CyclePhase}
_AUTH_FAILURES = {134, 135}  # MQTT reason codes: bad user name or password, not authorised


class MqttConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(default=1883, ge=1, le=65535)
    username: str | None = None
    password: str | None = None
    base_topic: str = "smartwash/washer"
    client_id: str | None = None
    tls: bool = False
    ca_certs: str | None = Field(
        default=None, description="CA bundle for TLS (system CAs if unset)"
    )
    keepalive_s: int = Field(default=30, gt=0)
    ack_timeout_s: float = Field(default=5.0, ge=0, description="0 = fire and forget")
    connect_timeout_s: float = Field(default=10.0, gt=0)
    telemetry_fields: list[str] | None = Field(
        default=None, description="Sensor fields the device publishes (all of them if unset)"
    )

    @field_validator("username", "password", "client_id", "ca_certs")
    @classmethod
    def _blank_is_unset(cls, value: str | None) -> str | None:
        # ``${MQTT_USERNAME:-}`` in the config expands to "", which means "not set".
        return value or None

    @field_validator("base_topic")
    @classmethod
    def _check_base_topic(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value or "+" in value or "#" in value:
            raise ValueError("base_topic must be a non-empty topic without wildcards")
        return value

    @field_validator("telemetry_fields")
    @classmethod
    def _check_telemetry_fields(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (unknown := set(value) - SENSOR_FIELDS):
            raise ValueError(
                f"unknown telemetry fields {sorted(unknown)}; known: {sorted(SENSOR_FIELDS)}"
            )
        return value

    def topic(self, name: str) -> str:
        return f"{self.base_topic}/{name}"


def create_client(
    config: MqttConfig, client_id: str, *, will_topic: str | None = None
) -> mqtt.Client:
    """A paho client with the credentials and TLS settings from ``config``.

    ``will_topic`` makes the broker publish a retained ``offline`` there if the connection dies.
    """
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv311
    )
    if config.username is not None:
        client.username_pw_set(config.username, config.password)
    if config.tls or config.ca_certs:
        client.tls_set(ca_certs=config.ca_certs)
    if will_topic is not None:
        client.will_set(will_topic, OFFLINE, qos=1, retain=True)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    return client


def parse_state(data: Any) -> MachineState:
    """Build a ``MachineState`` from a device's state message, tolerating vendor-specific values.

    A programme that is not one of ours moves to ``vendor_program``, an unknown phase becomes
    ``None`` and keys we do not know end up in ``raw``.
    """
    if not isinstance(data, dict):
        raise ValueError("a state message must be a JSON object")
    fields = dict(data)
    program = fields.get("program")
    if program is not None and not (isinstance(program, str) and program in _PROGRAMS):
        fields["program"] = None
        if fields.get("vendor_program") is None:
            fields["vendor_program"] = str(program)
    phase = fields.get("phase")
    if phase is not None and not (isinstance(phase, str) and phase in _PHASES):
        fields["phase"] = None
    if "raw" not in fields:
        fields["raw"] = {k: v for k, v in data.items() if k not in MachineState.model_fields}
    return MachineState.model_validate(fields)


def _is_failure(reason_code: Any) -> bool:
    return bool(getattr(reason_code, "is_failure", reason_code != 0))


class _PendingAck:
    __slots__ = ("error", "event", "ok")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.ok = False
        self.error: str | None = None


class MqttWasherAdapter(WashingMachineAdapter):
    """A washing machine speaking the SmartWash MQTT protocol (DIY/ESP32 devices, bridges)."""

    name = "mqtt"

    def __init__(
        self,
        config: MqttConfig,
        *,
        client_factory: Callable[[MqttConfig], Any] | None = None,
        max_buffered_samples: int = 20_000,
        initial_state_wait_s: float = 1.0,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        self._initial_state_wait_s = initial_state_wait_s
        self._topics = {
            name: config.topic(name)
            for name in (AVAILABILITY_TOPIC, STATE_TOPIC, TELEMETRY_TOPIC, COMMAND_TOPIC, ACK_TOPIC)
        }
        fields = config.telemetry_fields
        self._capabilities = AdapterCapabilities(
            telemetry_fields=frozenset(fields) if fields is not None else SENSOR_FIELDS
        )
        self._connect_lock = threading.Lock()  # serialises connect() and close()
        self._command_lock = threading.Lock()  # one command in flight at a time
        self._lock = threading.Lock()  # guards everything below; never held while waiting
        self._client: Any = None
        self._closed = False
        self._link_up = False
        self._available: bool | None = None
        self._state: MachineState | None = None
        self._telemetry: deque[TelemetrySample] = deque(maxlen=max_buffered_samples)
        self._pending: dict[str, _PendingAck] = {}
        self._connack = threading.Event()
        self._connect_error: Any = None
        self._got_state = threading.Event()

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    # --------------------------------------------------------------- connection

    def connect(self) -> None:
        cfg = self._config
        where = f"{cfg.host}:{cfg.port}"
        with self._connect_lock:
            if self._client is not None:
                return
            self._closed = False
            client = self._client_factory(cfg)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            self._connack.clear()
            self._connect_error = None
            try:
                client.connect(cfg.host, cfg.port, keepalive=cfg.keepalive_s)
            except (OSError, ValueError) as exc:
                raise DeviceError(f"cannot reach the MQTT broker at {where}: {exc}") from exc
            client.loop_start()
            if not self._connack.wait(cfg.connect_timeout_s):
                _shutdown(client)
                raise DeviceError(
                    f"the MQTT broker at {where} did not accept the connection within "
                    f"{cfg.connect_timeout_s:g} s"
                )
            if self._connect_error is not None:
                reason = self._connect_error
                _shutdown(client)
                error = (
                    DeviceAuthError
                    if getattr(reason, "value", None) in _AUTH_FAILURES
                    else DeviceError
                )
                raise error(f"the MQTT broker at {where} refused the connection: {reason}")
            with self._lock:
                self._client = client
        # Retained availability and state arrive right after subscribing; give them a moment so the
        # first get_state() reflects the device rather than "no state yet".
        self._got_state.wait(self._initial_state_wait_s)

    def close(self) -> None:
        with self._connect_lock:
            self._closed = True  # until connect() is called again explicitly
            with self._lock:
                client, self._client = self._client, None
                self._link_up = False
                self._available = None
                self._state = None  # a reconnect receives fresh retained messages
            self._got_state.clear()
            if client is not None:
                _shutdown(client)

    # ------------------------------------------------------------------ queries

    def get_state(self) -> MachineState:
        self._require_client()
        with self._lock:
            if not self._link_up:
                reason = "MQTT broker connection lost"
            elif self._available is False:
                reason = "device reported offline"
            elif self._state is None:
                reason = "no state received from the device yet"
            else:
                return self._state.model_copy(deep=True)
        return MachineState(status=MachineStatus.OFFLINE, raw={"reason": reason})

    def read_telemetry(self) -> list[TelemetrySample]:
        self._require_client()
        with self._lock:
            samples = list(self._telemetry)
            self._telemetry.clear()
        return samples

    def describe(self) -> dict[str, Any]:
        cfg = self._config
        with self._lock:
            return {
                "adapter": self.name,
                "host": cfg.host,
                "port": cfg.port,
                "base_topic": cfg.base_topic,
                "broker_connected": self._link_up,
                "device_online": self._available,
            }

    # ----------------------------------------------------------------- commands

    def start_cycle(self, settings: CycleSettings, *, load_kg: float | None = None) -> None:
        self._send("start", settings=settings.model_dump(mode="json"), load_kg=load_kg)

    def pause(self) -> None:
        self._send("pause")

    def resume(self) -> None:
        self._send("resume")

    def stop(self) -> None:
        self._send("stop")

    def _send(self, command: str, **fields: Any) -> None:
        cfg = self._config
        client = self._require_client()
        with self._lock:
            link_up = self._link_up
        if not link_up:
            # Refuse rather than let paho queue the command and run it whenever the link returns.
            raise DeviceError(f"not connected to the MQTT broker at {cfg.host}:{cfg.port}")
        with self._command_lock:
            command_id = uuid.uuid4().hex
            waiter = _PendingAck() if cfg.ack_timeout_s > 0 else None
            if waiter is not None:
                with self._lock:
                    self._pending[command_id] = waiter
            try:
                payload = {"id": command_id, "command": command, "sent_at": round(time.time(), 3)}
                # QoS 0 on purpose: the acknowledgement already confirms delivery end to end, while
                # QoS 1 would let paho re-send a "start" after a reconnect, long after we gave up.
                info = client.publish(
                    self._topics[COMMAND_TOPIC], json.dumps({**payload, **fields}), qos=0
                )
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise DeviceError(
                        f"could not send '{command}' to the washer: {mqtt.error_string(info.rc)}"
                    )
                if waiter is None:
                    return
                if not waiter.event.wait(cfg.ack_timeout_s):
                    raise DeviceError(
                        f"no acknowledgement from the washer for '{command}' within "
                        f"{cfg.ack_timeout_s:g} s; is it online and subscribed to "
                        f"{self._topics[COMMAND_TOPIC]}?"
                    )
                if not waiter.ok:
                    raise DeviceNotReady(waiter.error or f"the washer refused '{command}'")
            finally:
                if waiter is not None:
                    with self._lock:
                        self._pending.pop(command_id, None)

    def _require_client(self) -> Any:
        if self._closed:
            raise DeviceError("the MQTT connection is closed")
        if self._client is None:
            self.connect()
        with self._lock:
            client = self._client
        if client is None:
            raise DeviceError("the MQTT connection is closed")
        return client

    # ---------------------------------------------------------- paho callbacks
    # These run on paho's network thread. An exception escaping them would kill that thread, so
    # message handlers log and drop bad input instead of raising.

    def _on_connect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None
    ) -> None:
        if _is_failure(reason_code):
            self._connect_error = reason_code
            logger.warning("MQTT broker refused the connection: %s", reason_code)
            self._connack.set()
            return
        self._connect_error = None
        client.subscribe(
            [
                (self._topics[name], 1)
                for name in (AVAILABILITY_TOPIC, STATE_TOPIC, TELEMETRY_TOPIC, ACK_TOPIC)
            ]
        )
        with self._lock:
            self._link_up = True
        self._connack.set()

    def _on_disconnect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None
    ) -> None:
        with self._lock:
            was_up, self._link_up = self._link_up, False
        if was_up:
            logger.warning("lost connection to the MQTT broker (%s); reconnecting", reason_code)

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        handlers = {
            self._topics[TELEMETRY_TOPIC]: self._handle_telemetry,
            self._topics[STATE_TOPIC]: self._handle_state,
            self._topics[ACK_TOPIC]: self._handle_ack,
            self._topics[AVAILABILITY_TOPIC]: self._handle_availability,
        }
        handler = handlers.get(message.topic)
        if handler is None:
            return
        try:
            handler(bytes(message.payload))
        except Exception:
            logger.exception("failed to handle MQTT message on %s", message.topic)

    def _handle_state(self, payload: bytes) -> None:
        if not payload:  # the retained state was cleared
            with self._lock:
                self._state = None
            return
        try:
            state = parse_state(json.loads(payload))
        except ValueError as exc:  # includes JSON and pydantic validation errors
            logger.warning("ignoring invalid state message: %s", exc)
            return
        with self._lock:
            self._state = state
        self._got_state.set()

    def _handle_telemetry(self, payload: bytes) -> None:
        try:
            data = json.loads(payload)
        except ValueError as exc:
            logger.warning("ignoring telemetry that is not valid JSON: %s", exc)
            return
        items = data if isinstance(data, list) else [data]
        samples: list[TelemetrySample] = []
        invalid = 0
        for item in items:
            try:
                samples.append(TelemetrySample.model_validate(item))
            except ValidationError:
                invalid += 1
        if invalid:
            logger.warning("dropped %d invalid telemetry sample(s) of %d", invalid, len(items))
        if samples:
            with self._lock:
                self._telemetry.extend(samples)

    def _handle_ack(self, payload: bytes) -> None:
        try:
            data = json.loads(payload)
            command_id = str(data["id"])
        except (ValueError, KeyError, TypeError):
            logger.warning("ignoring malformed acknowledgement: %r", payload[:200])
            return
        with self._lock:
            waiter = self._pending.get(command_id)
        if waiter is None:
            return  # an acknowledgement for a command we are no longer waiting for
        waiter.ok = data.get("ok") is True
        error = data.get("error")
        waiter.error = str(error) if error is not None else None
        waiter.event.set()

    def _handle_availability(self, payload: bytes) -> None:
        text = payload.decode("utf-8", "replace").strip().strip('"').lower()
        with self._lock:
            self._available = (text == ONLINE) if text else None
        if text == OFFLINE:
            self._got_state.set()  # get_state() is settled: the device is offline


def _default_client_factory(config: MqttConfig) -> mqtt.Client:
    return create_client(config, config.client_id or f"smartwash-{uuid.uuid4().hex[:8]}")


def _shutdown(client: Any) -> None:
    try:
        client.disconnect()
    except Exception:  # best effort while tearing down
        logger.debug("error while disconnecting from the MQTT broker", exc_info=True)
    client.loop_stop()
