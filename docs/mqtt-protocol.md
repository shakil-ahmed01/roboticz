# SmartWash MQTT protocol

Use MQTT to connect SmartWash to a DIY or ESP32-based washing machine, a sensor retrofit on an
existing machine, or any bridge (Node-RED, ESPHome, Tasmota, a Home Assistant automation) that can
publish JSON. SmartWash is the **controller**; the washing machine or its firmware is the
**device**.

`smartwash.devices.mqtt_device.VirtualMqttWasher` is a complete reference implementation of the
device side. Its source is the best companion to this document.

---

## Safety first

> ⚠️ Read this before connecting anything to a real machine.

A washing machine combines mains electricity, water, heat and a drum spinning at up to 1400 rpm.

- **Prefer non-invasive sensors**: a clamp-on current transformer, an accelerometer stuck to the
  cabinet, a temperature probe on the outside of the drain hose, an energy-metering smart plug. They
  need no changes inside the machine.
- **Never bypass, bridge or defeat the door lock, the water-level pressure switch, thermal fuses or
  any other interlock.** The machine's own safety electronics must always stay in charge. Remote
  start must be impossible while the door is open: the firmware should refuse `start` unless the
  machine reports the door closed, and must never actuate the lock itself.
- **Anything inside the machine is a job for a qualified electrician**: tapping the motor supply,
  the heater relay or the valves, or adding a relay. Unplug the machine before opening it, keep it
  on an RCD/GFCI-protected circuit, and keep low-voltage electronics isolated from mains wiring.
- Smart plugs used to measure power must be rated for the machine's full load. The heater alone
  draws about 2 kW, so use a plug rated for at least 16 A, not a 10 A one.
- Modifying an appliance may void its warranty and can affect insurance.

SmartWash never starts a cycle on its own. Automatic actions are limited to pausing the machine when
the AI detects a critical fault, and that behaviour can be switched off.

---

## Topics

`{base}` is the configured `base_topic` (default `smartwash/washer`; it must not contain the MQTT
wildcards `+` or `#`).

| Topic                  | Direction           | Retained | QoS | Payload                                |
|------------------------|---------------------|----------|-----|----------------------------------------|
| `{base}/availability`  | device → SmartWash  | yes      | 1   | plain text `online` / `offline`        |
| `{base}/state`         | device → SmartWash  | yes      | 1   | JSON object ([state](#state))          |
| `{base}/telemetry`     | device → SmartWash  | no       | 1   | JSON object or array ([telemetry](#telemetry)) |
| `{base}/command`       | SmartWash → device  | no       | 0   | JSON object ([commands](#commands))    |
| `{base}/ack`           | device → SmartWash  | no       | 1   | JSON object ([acknowledgements](#acknowledgements)) |

All JSON is UTF-8. Clients that can only publish at QoS 0 (for example the Arduino `PubSubClient`
library) still work; QoS 1 just makes delivery and ordering more robust.

### Availability and the last will

When it connects, the device sets its MQTT **last will** to a retained `offline` on
`{base}/availability`, then publishes a retained `online` there. If the device crashes or loses
power, the broker publishes the will and SmartWash shows the machine as `offline`. Before a clean
shutdown the device publishes `offline` itself, because a clean `DISCONNECT` does not trigger the
will.

SmartWash reports `offline` whenever the device says `offline`, when the broker connection is down,
or when no state has arrived yet.

---

## State

Retained JSON on `{base}/state`. Only `status` is required; send `null` or omit anything the
machine does not know.

| Field                  | Type            | Meaning                                                         |
|------------------------|-----------------|-----------------------------------------------------------------|
| `status`               | string, required| `idle`, `delayed_start`, `running`, `paused`, `finished`, `error` (`offline` is SmartWash's own) |
| `phase`                | string or null  | `idle`, `fill`, `wash`, `rinse`, `drain`, `spin`, `done`        |
| `program`              | string or null  | One of SmartWash's programmes (see below)                       |
| `vendor_program`       | string or null  | The machine's own name for the programme                        |
| `temperature_c`        | integer or null | Selected wash temperature, °C                                   |
| `spin_rpm`             | integer or null | Selected final spin speed, rpm                                  |
| `progress_pct`         | number or null  | 0–100                                                           |
| `remaining_min`        | number or null  | Estimated minutes left                                          |
| `door_open`            | bool or null    |                                                                 |
| `remote_start_allowed` | bool or null    | Whether the machine currently accepts a remote `start`          |
| `error_code`           | string or null  | The machine's error code while `status` is `error`              |

SmartWash programmes: `cotton`, `eco_40_60`, `synthetics`, `mixed`, `wool`, `delicates`, `quick`,
`sportswear`, `bedding`, `towels`, `dark_wash`, `hygiene`.

SmartWash is lenient with what it receives. A `program` it does not know moves to `vendor_program`,
an unknown `phase` is ignored, and extra keys are kept in the state's `raw` field for debugging. A
message without a valid `status` is dropped with a warning. An empty retained payload clears the
state.

Publish the state:

- immediately whenever `status`, `phase`, `error_code` or the selected programme changes;
- at least every ~2 seconds while a cycle runs, so progress and remaining time stay fresh;
- right after executing a command, **before** the acknowledgement (see below).

```json
{
  "status": "running",
  "phase": "wash",
  "program": "cotton",
  "vendor_program": "Cotton",
  "temperature_c": 40,
  "spin_rpm": 1200,
  "progress_pct": 31.5,
  "remaining_min": 83.0,
  "door_open": false,
  "remote_start_allowed": true,
  "error_code": null
}
```

---

## Telemetry

Sensor readings on `{base}/telemetry`, published only while a cycle runs: not while idle, paused or
finished. Each message is one sample object or, preferably, an array of samples in time order.

| Field             | Unit            | Meaning                                                              |
|-------------------|-----------------|----------------------------------------------------------------------|
| `timestamp`       | seconds         | Required. Unix time, or any clock that increases steadily            |
| `phase`           | –               | Required. `fill`, `wash`, `rinse`, `drain` or `spin`                 |
| `drum_rpm`        | rpm             | Measured drum speed                                                  |
| `target_rpm`      | rpm             | Speed the programme is asking for at this moment: follows spin ramps, `0` while the drum rests between tumbles |
| `motor_current_a` | A (RMS)         | Drum-motor current                                                   |
| `vibration_g`     | g (RMS)         | Vibration, RMS over the sample period, gravity removed               |
| `water_temp_c`    | °C              | Water temperature                                                    |
| `target_temp_c`   | °C or `null`    | Temperature the heater is working towards; `null` when no heating is demanded |
| `water_level_l`   | litres          | Free water in the drum                                               |
| `power_w`         | W               | Whole-machine power draw                                             |

```json
[
  {"timestamp": 1760870400.0, "phase": "wash", "drum_rpm": 49.6, "target_rpm": 50.0,
   "motor_current_a": 0.83, "vibration_g": 0.0371, "water_temp_c": 31.42, "target_temp_c": 40.0,
   "water_level_l": 10.9, "power_w": 2131.0},
  {"timestamp": 1760870402.0, "phase": "wash", "drum_rpm": 50.3, "target_rpm": 50.0,
   "motor_current_a": 0.84, "vibration_g": 0.0365, "water_temp_c": 31.49, "target_temp_c": 40.0,
   "water_level_l": 10.8, "power_w": 2129.0}
]
```

Missing sensors can be omitted or sent as `null`. A sample without a valid `timestamp` and `phase`
is dropped.

### What the AI needs

The **fault detector** (unbalanced load, heater failure, drain blockage, bearing wear, water-supply
problems, motor overload and unknown anomalies) needs **all** of `drum_rpm`, `target_rpm`,
`motor_current_a`, `vibration_g`, `water_temp_c`, `water_level_l` and `power_w`. `target_temp_c` is
optional but improves heater diagnosis. If your device provides fewer fields, list the ones it does
provide in `telemetry_fields` (see [configuration](#configuring-smartwash)). SmartWash then turns
fault detection off, but recommendations, remote control, progress tracking and notifications keep
working.

The detector analyses 60-second windows of machine time and was trained on one sample every
2 seconds.

### Timing

- Take one sample every **2 seconds** (1–5 s works). Aggregate faster sensor readings over the
  sample period: the mean for speed, temperature, level and power, and the RMS for vibration and
  current.
- Publish the collected samples as one array about **once a second**.
- **Flush pending samples before publishing a state change**, so SmartWash has every reading of a
  cycle by the time it sees `finished`, `paused` or `error`.

---

## Commands

SmartWash publishes commands to `{base}/command` at QoS 0. That is deliberate: the
[acknowledgement](#acknowledgements) already confirms delivery end to end, while QoS 1 would let a
client library or broker redeliver a `start` after a reconnect, long after the user gave up on it.

```json
{
  "id": "6f1c2e0a9b8d4c3e8f7a6b5c4d3e2f1a",
  "command": "start",
  "sent_at": 1760870400.125,
  "settings": {
    "program": "cotton",
    "temperature_c": 40,
    "spin_rpm": 1200,
    "prewash": false,
    "extra_rinse": false
  },
  "load_kg": 4.5
}
```

| Field      | Meaning                                                                                  |
|------------|------------------------------------------------------------------------------------------|
| `id`       | Unique id; echo it in the acknowledgement                                                |
| `command`  | `start`, `pause`, `resume` or `stop`                                                     |
| `sent_at`  | SmartWash's clock (Unix seconds) when it sent the command                                |
| `settings` | `start` only: `program`, `temperature_c` (20, 30, 40, 60 or 90), `spin_rpm` (400–1400 in steps of 200), `prewash`, `extra_rinse` |
| `load_kg`  | `start` only: the user's estimate of the load, or `null`. Real machines weigh the laundry themselves |

How a device should handle each command:

- **start**: refuse it if a cycle is running or paused, if the machine is in an error state (until it
  receives `stop`), if the door is open, or if remote start is disabled on the machine. Map the
  settings onto the machine's own programmes and **never exceed the requested temperature or spin
  speed**: if the exact value is not available, use the next lower one. The temperature and spin
  have already been checked against the fabric's care limits.
- **pause**: only while running.
- **resume**: only while paused.
- **stop**: always accepted; abort the cycle, drain, and clear an error state.
- Refuse unknown commands.
- If the device clock is synchronised (NTP), refuse commands whose `sent_at` is more than 30 seconds
  old. They were stuck somewhere, and a washing machine starting long after the user pressed the
  button is a surprise nobody wants.

After executing (or refusing) a command, publish the resulting [state](#state) **first**, then the
acknowledgement. SmartWash reads the state as soon as the acknowledgement arrives.

## Acknowledgements

```json
{"id": "6f1c2e0a9b8d4c3e8f7a6b5c4d3e2f1a", "ok": true, "error": null}
{"id": "6f1c2e0a9b8d4c3e8f7a6b5c4d3e2f1a", "ok": false, "error": "door is open"}
```

SmartWash waits up to `ack_timeout_s` (default 5 s) for the acknowledgement with the matching
`id`:

- `"ok": true`: the command succeeded.
- `"ok": false`: the command was refused. The `error` text is shown to the user (the REST API
  answers with HTTP 409), so make it a short, human-readable reason.
- No acknowledgement in time: SmartWash reports that the washer did not answer. The command may
  still have reached the device, so the next state message is the source of truth.

With `ack_timeout_s: 0` SmartWash does not wait (fire and forget). Only do this for devices that
cannot send acknowledgements, because refusals are then invisible.

SmartWash refuses to send commands while it is disconnected from the broker, and because commands
use QoS 0 they are never queued and delivered later.

---

## Configuring SmartWash

The MQTT connection takes these settings (`smartwash.devices.mqtt.MqttConfig`):

| Setting             | Default             | Meaning                                                    |
|---------------------|---------------------|------------------------------------------------------------|
| `host`              | `localhost`         | Broker host name or IP                                     |
| `port`              | `1883`              | Broker port (usually `8883` with TLS)                      |
| `username`          | –                   | Broker user name                                           |
| `password`          | –                   | Broker password (keep it in an environment variable)       |
| `base_topic`        | `smartwash/washer`  | Prefix of the five topics                                  |
| `client_id`         | random              | MQTT client id (the virtual device appends `-device`)      |
| `tls`               | `false`             | Use TLS                                                    |
| `ca_certs`          | system CAs          | CA bundle for TLS (setting it also enables TLS)            |
| `keepalive_s`       | `30`                | MQTT keep-alive interval                                   |
| `ack_timeout_s`     | `5.0`               | How long to wait for acknowledgements; `0` = fire and forget |
| `connect_timeout_s` | `10.0`              | How long to wait for the broker to accept the connection   |
| `telemetry_fields`  | all sensor fields   | The sensor fields your device publishes                    |

For example, in the SmartWash configuration file:

```yaml
machine:
  adapter: mqtt
  mqtt:
    host: 192.168.1.10
    username: smartwash
    password: ${MQTT_PASSWORD}
    base_topic: smartwash/washer
```

---

## Try it without hardware

Start a broker, a virtual washer and a monitor in three terminals:

```bash
mosquitto -p 1883                                   # any MQTT broker works
smartwash virtual-device                            # simulated washer, 60x real time
mosquitto_sub -h localhost -t 'smartwash/washer/#' -v
```

`smartwash virtual-device --help` lists its options, including the broker settings, time scale and
fault injection. Point SmartWash at the same broker with the configuration above, or drive the
washer by hand:

```bash
mosquitto_pub -h localhost -t smartwash/washer/command -m \
  '{"id": "1", "command": "start", "settings": {"program": "quick", "temperature_c": 30, "spin_rpm": 1000}}'
```

From Python:

```python
from smartwash.devices.mqtt import MqttConfig
from smartwash.devices.mqtt_device import run_virtual_device
from smartwash.schemas import FaultType

# A virtual washer with a partly blocked drain, for watching the fault detector at work.
run_virtual_device(MqttConfig(host="localhost"), time_scale=60, fault=FaultType.DRAIN_BLOCKAGE)
```

---

## DIY retrofit notes

These notes are a starting point for adding sensors to a machine you own. Read the
[safety section](#safety-first) first.

**Microcontroller.** An ESP32 with Wi-Fi is the usual choice. Use `PubSubClient` and `ArduinoJson`
with the Arduino framework, ESPHome, or MicroPython with `umqtt`. `PubSubClient` limits packets to
256 bytes by default, so call `setBufferSize(4096)` or the state and telemetry messages will be
silently dropped. `PubSubClient` publishes at QoS 0 only, which works but is less robust than
QoS 1.

**Sensors and the fields they provide:**

| Field             | Suggested sensor                                                     | Notes |
|-------------------|----------------------------------------------------------------------|-------|
| `vibration_g`     | MPU-6050 or ADXL345 accelerometer (I²C), firmly fixed to the cabinet top | Sample at ≥ 200 Hz and report the RMS of the acceleration magnitude minus 1 g. Cabinet readings are smaller than drum readings, so retrain the detector on your own data for best results |
| `drum_rpm`        | Hall-effect sensor (e.g. A3144) and a magnet on the drum pulley, or an optical tacho | Belt-driven machines only. Direct-drive motors are much harder to instrument |
| `motor_current_a` | SCT-013 split-core current transformer around the motor's supply wire | Needs the cabinet opened, so it is an electrician's job. Use the burden resistor and ADC biasing the sensor's datasheet calls for |
| `power_w`         | Energy-metering smart plug (Shelly, Tasmota with HLW8012/BL0937) rated ≥ 16 A | Non-invasive; reports the whole machine, heater included |
| `water_temp_c`    | DS18B20 waterproof probe                                             | Non-invasive option: strap it to the drain hose near the machine with thermal paste and insulation. It lags and reads only during draining, so treat it as approximate |
| `water_level_l`   | Low-range differential pressure sensor (0–10 kPa) on the level-sensing air hose, calibrated in litres | Must not disturb the machine's own pressure switch or leak. If in doubt, leave it out |
| `phase`, `target_rpm`, `target_temp_c` | Your firmware                                   | Only known when your firmware runs the programme. An observe-only retrofit cannot report setpoints, so list just the fields it has in `telemetry_fields` |

**Firmware outline** (pseudocode):

```text
setup:
    connect Wi-Fi
    MQTT connect with last will  {base}/availability = "offline" (retained)
    publish {base}/availability = "online" (retained)
    subscribe {base}/command
    publish current state (retained)

every 2 s while a cycle runs:
    read sensors -> append one sample to the outbox
every 1 s:
    publish the outbox as a JSON array on {base}/telemetry, then clear it
when status / phase / error changes (and every 2 s while running):
    flush the outbox first, then publish the state (retained)

on {base}/command:
    parse JSON; refuse anything invalid, stale or unsafe (door open, remote start off, busy)
    act only through the machine's own controls, which keep every interlock in charge
    flush the outbox, publish the new state, then publish {"id": ..., "ok": ..., "error": ...}
```
