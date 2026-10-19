"""A virtual washing machine on MQTT: the simulator speaking the device side of the protocol.

Run it with ``smartwash virtual-device`` to try the MQTT connection without hardware. It is also the
reference implementation for firmware authors; a real device does the same four things:

1. connect with a last will that publishes a retained ``offline`` to ``{base}/availability``, then
   publish a retained ``online`` there;
2. publish its state, retained, to ``{base}/state`` whenever it changes and every couple of seconds;
3. publish sensor readings to ``{base}/telemetry`` while a cycle runs;
4. execute commands from ``{base}/command`` and answer each one on ``{base}/ack``.

See ``docs/mqtt-protocol.md`` for the payloads.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
import uuid
from typing import Any

from pydantic import ValidationError

from ..schemas import (
    ACTIVE_STATUSES,
    CycleSettings,
    FaultType,
    MachineState,
    MachineStatus,
    TelemetrySample,
)
from .base import DeviceAuthError, DeviceError
from .mqtt import (
    ACK_TOPIC,
    AVAILABILITY_TOPIC,
    COMMAND_TOPIC,
    OFFLINE,
    ONLINE,
    STATE_TOPIC,
    TELEMETRY_TOPIC,
    MqttConfig,
    create_client,
)
from .simulator import WasherSimulator

logger = logging.getLogger(__name__)

DEFAULT_LOAD_KG = 4.0
"""Assumed load when a start command does not say (a real machine weighs the laundry)."""


class VirtualMqttWasher:
    """Runs a ``WasherSimulator`` ``time_scale`` times faster than real time, served on MQTT."""

    TICK_S = 0.1
    STATE_INTERVAL_S = 2.0
    TELEMETRY_INTERVAL_S = 1.0
    MAX_COMMAND_AGE_S = 30.0
    """Commands whose ``sent_at`` is older than this are refused (they were stuck somewhere)."""

    def __init__(
        self,
        config: MqttConfig,
        *,
        time_scale: float = 60.0,
        sample_period_s: float = 2.0,
        fault: FaultType | str = FaultType.NONE,
        fault_severity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.config = config
        self.time_scale = time_scale
        self._topics = {
            name: config.topic(name)
            for name in (AVAILABILITY_TOPIC, STATE_TOPIC, TELEMETRY_TOPIC, COMMAND_TOPIC, ACK_TOPIC)
        }
        self._sim = WasherSimulator(sample_period_s=sample_period_s, seed=seed, epoch=time.time())
        self._fault = FaultType(fault)
        self._severity = fault_severity
        self._lock = threading.Lock()  # guards the simulator and the bookkeeping below
        self._outbox: list[TelemetrySample] = []
        self._last_tick = time.monotonic()
        self._carry_s = 0.0
        self._state_key: tuple[Any, ...] | None = None
        self._state_sent_at = 0.0
        self._telemetry_sent_at = 0.0
        self._client: Any = None
        self._connected = threading.Event()
        self._connect_error: Any = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Connect to the broker, announce the machine and start simulating."""
        if self._thread is not None:
            raise RuntimeError("the virtual washer is already running")
        cfg = self.config
        where = f"{cfg.host}:{cfg.port}"
        client_id = (
            f"{cfg.client_id}-device"
            if cfg.client_id
            else f"smartwash-device-{uuid.uuid4().hex[:8]}"
        )
        client = create_client(cfg, client_id, will_topic=self._topics[AVAILABILITY_TOPIC])
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        self._connected.clear()
        try:
            client.connect(cfg.host, cfg.port, keepalive=cfg.keepalive_s)
        except (OSError, ValueError) as exc:
            raise DeviceError(f"cannot reach the MQTT broker at {where}: {exc}") from exc
        client.loop_start()
        if not self._connected.wait(cfg.connect_timeout_s) or self._connect_error is not None:
            reason = self._connect_error
            client.disconnect()
            client.loop_stop()
            if reason is None:
                raise DeviceError(
                    f"the MQTT broker at {where} did not accept the connection within "
                    f"{cfg.connect_timeout_s:g} s"
                )
            error = DeviceAuthError if getattr(reason, "value", None) in (134, 135) else DeviceError
            raise error(f"the MQTT broker at {where} refused the connection: {reason}")
        self._client = client
        with self._lock:
            self._last_tick = time.monotonic()
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="virtual-washer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop simulating, announce ``offline`` and disconnect."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        client, self._client = self._client, None
        if client is None:
            return
        info = client.publish(self._topics[AVAILABILITY_TOPIC], OFFLINE, qos=1, retain=True)
        # If we are no longer connected, the broker publishes our last will ("offline") instead.
        with contextlib.suppress(RuntimeError, ValueError):
            info.wait_for_publish(timeout=2.0)
        client.disconnect()
        client.loop_stop()

    def inject_fault(self, fault: FaultType | str, severity: float = 1.0) -> None:
        """Give the machine a fault (``none`` repairs it); applies to the running cycle too."""
        with self._lock:
            self._advance(time.monotonic())
            self._fault = FaultType(fault)
            self._severity = severity
            self._sim.set_fault(self._fault, severity)

    @property
    def state(self) -> MachineState:
        with self._lock:
            self._advance(time.monotonic())
            return self._sim.state()

    # --------------------------------------------------------------- simulation

    def _run(self) -> None:
        while not self._stopping.wait(self.TICK_S):
            try:
                self._tick()
            except Exception:
                logger.exception("virtual washer tick failed")

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._advance(now)
            state = self._state_payload(now)
            telemetry = self._take_telemetry(now, flush=state is not None)
        self._publish(self._client, telemetry, state)

    def _publish(self, client: Any, telemetry: str | None, state: str | None) -> None:
        # Readings go out before the state they led to, so a consumer reacting to a state change
        # (a finished cycle, say) already has every sample of it. Both use QoS 1 so the broker
        # keeps them in order.
        if client is None:
            return
        if telemetry is not None:
            client.publish(self._topics[TELEMETRY_TOPIC], telemetry, qos=1)
        if state is not None:
            client.publish(self._topics[STATE_TOPIC], state, qos=1, retain=True)

    def _advance(self, now: float) -> None:
        """Run the simulator up to wall-clock time ``now``. Caller holds the lock."""
        elapsed = max(0.0, now - self._last_tick) * self.time_scale + self._carry_s
        self._last_tick = now
        dt = self._sim.dt
        remaining = int(elapsed // dt)
        self._carry_s = elapsed - remaining * dt
        while remaining > 0:
            if self._sim.status is not MachineStatus.RUNNING:
                self._sim.advance_idle(dt * remaining)
                break
            record = self._sim.step()
            if record is not None:
                self._outbox.append(record.sample)
            remaining -= 1

    def _take_telemetry(self, now: float, *, flush: bool = False) -> str | None:
        """Pending samples as one message, if due (or if ``flush``). Caller holds the lock."""
        due = flush or now - self._telemetry_sent_at >= self.TELEMETRY_INTERVAL_S
        if not self._outbox or not due:
            return None
        batch = json.dumps([_sample_payload(sample) for sample in self._outbox])
        self._outbox.clear()
        self._telemetry_sent_at = now
        return batch

    def _state_payload(self, now: float, *, force: bool = False) -> str | None:
        """The retained state message, if it is due. Caller holds the lock."""
        state = self._sim.state()
        key = (
            state.status,
            state.phase,
            state.error_code,
            state.program,
            state.temperature_c,
            state.spin_rpm,
        )
        due = force or key != self._state_key or now - self._state_sent_at >= self.STATE_INTERVAL_S
        if not due:
            return None
        self._state_key = key
        self._state_sent_at = now
        # A real machine cannot know which fault it has, so the simulator's internals stay private.
        return state.model_dump_json(exclude={"raw"})

    # ----------------------------------------------------------------- commands

    def _on_connect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None
    ) -> None:
        if reason_code.is_failure:
            self._connect_error = reason_code
            self._connected.set()
            return
        self._connect_error = None
        client.subscribe(self._topics[COMMAND_TOPIC], qos=1)
        client.publish(self._topics[AVAILABILITY_TOPIC], ONLINE, qos=1, retain=True)
        with self._lock:
            state = self._state_payload(time.monotonic(), force=True)
        client.publish(self._topics[STATE_TOPIC], state, qos=1, retain=True)
        self._connected.set()

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        if message.topic != self._topics[COMMAND_TOPIC]:
            return
        try:
            self._handle_command(client, bytes(message.payload))
        except Exception:  # never let an exception kill paho's network thread
            logger.exception("failed to handle command %r", message.payload[:200])

    def _handle_command(self, client: Any, payload: bytes) -> None:
        try:
            command = json.loads(payload)
        except ValueError:
            logger.warning("ignoring command that is not valid JSON: %r", payload[:200])
            return
        if not isinstance(command, dict):
            logger.warning("ignoring command that is not a JSON object: %r", payload[:200])
            return
        with self._lock:
            now = time.monotonic()
            self._advance(now)
            error = self._execute(command)
            telemetry = self._take_telemetry(now, flush=True)
            state = self._state_payload(now, force=True)
        logger.info("command %s -> %s", command.get("command"), error or "ok")
        # The new state goes out before the acknowledgement, so a controller reading the state right
        # after the ack already sees the effect of its command.
        self._publish(client, telemetry, state)
        command_id = command.get("id")
        if command_id is None:
            logger.warning("command without an id; cannot acknowledge it")
            return
        ack = {"id": command_id, "ok": error is None, "error": error}
        client.publish(self._topics[ACK_TOPIC], json.dumps(ack), qos=1)

    def _execute(self, command: dict[str, Any]) -> str | None:
        """Carry out a command; return why it was refused, or ``None``. Caller holds the lock."""
        name = command.get("command")
        sent_at = command.get("sent_at")
        if isinstance(sent_at, (int, float)) and time.time() - sent_at > self.MAX_COMMAND_AGE_S:
            return f"refusing a stale command sent {time.time() - sent_at:.0f} s ago"
        sim = self._sim
        if name == "start":
            if sim.status in ACTIVE_STATUSES:
                return "a cycle is already running"
            if sim.status is MachineStatus.ERROR:
                return f"machine is in error {sim.error_code}; send stop to reset it"
            try:
                settings = CycleSettings.model_validate(command.get("settings"))
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or 'settings'}: {err['msg']}"
                    for err in exc.errors()
                )
                return f"invalid settings: {problems}"
            load = command.get("load_kg")
            valid_load = isinstance(load, (int, float)) and not isinstance(load, bool) and load > 0
            sim.start(
                settings,
                float(load) if valid_load else DEFAULT_LOAD_KG,
                fault=self._fault,
                severity=self._severity,
            )
            return None
        if name == "pause":
            if sim.status is not MachineStatus.RUNNING:
                return "no running cycle to pause"
            sim.pause()
            return None
        if name == "resume":
            if sim.status is not MachineStatus.PAUSED:
                return "the cycle is not paused"
            sim.resume()
            return None
        if name == "stop":
            sim.stop()
            return None
        return f"unknown command {name!r}"


def _sample_payload(sample: TelemetrySample) -> dict[str, Any]:
    payload: dict[str, Any] = {"timestamp": round(sample.timestamp, 3), "phase": sample.phase.value}
    for name in TelemetrySample.model_fields:
        if name in payload:
            continue
        value = getattr(sample, name)
        payload[name] = round(value, 4) if isinstance(value, float) else value
    return payload


def run_virtual_device(
    config: MqttConfig,
    *,
    time_scale: float = 60.0,
    fault: FaultType | str = FaultType.NONE,
    fault_severity: float = 1.0,
    seed: int | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Serve a virtual washer on MQTT until ``stop_event`` is set or Ctrl+C is pressed."""
    device = VirtualMqttWasher(
        config, time_scale=time_scale, fault=fault, fault_severity=fault_severity, seed=seed
    )
    device.start()
    logger.info(
        "virtual washer online at %s:%s under %s/ (%g x real time, fault: %s)",
        config.host,
        config.port,
        config.base_topic,
        time_scale,
        FaultType(fault).value,
    )
    stop_event = stop_event or threading.Event()
    try:
        while not stop_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
