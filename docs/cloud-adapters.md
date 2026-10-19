# Connecting a commercial smart washer

SmartWash talks to factory-made smart washers through their makers' cloud APIs, or through Home
Assistant for any brand it integrates. Pick the adapter for your machine:

| Your washer | Adapter | Chooses programme, °C, spin | Pause / resume | Sensor telemetry |
|---|---|---|---|---|
| Bosch, Siemens, Neff, Gaggenau, Thermador (Home Connect app) | `home_connect` | yes | yes | no |
| Samsung (SmartThings app) | `smartthings` | yes, after you map cycle ids | yes | no |
| LG (ThinQ app) | `lg_thinq` | no: starts the course set on the machine | no | no |
| Anything in Home Assistant (LG, Miele, Electrolux, Whirlpool, GE, …) | `home_assistant` | depends on the integration | if configured | optional power |

**What works without sensor telemetry.** Cycle recommendations, starting, stopping, status,
progress and finish notifications all work with these adapters. The AI fault detector needs
drum-speed, vibration, current, temperature, water-level and power readings that no maker's cloud
API publishes. For fault detection, add sensors and use the MQTT adapter
([mqtt-protocol.md](mqtt-protocol.md)).

**Fabric-care safety.** The adapters never run hotter or spin faster than the recommendation.
Temperature and spin are rounded down to the nearest value the machine offers for the chosen
programme. If the programme cannot go that low, the start is refused (`UnsupportedOperation`)
before any command is sent. When a machine lacks a programme, the adapter falls back only to a
programme with the same or gentler drum action; for example, mixed falls back to synthetics, but
wool never falls back to anything.

**Remote start must be enabled on the machine.** Every maker requires someone at the washer to
allow remote start after loading it and closing the door. See each section below.

Secrets belong in environment variables. The config loader expands `${VAR}`, and `config.yaml` is
git-ignored.

---

## Home Connect (Bosch, Siemens, Neff, Gaggenau, Thermador)

Uses the official [Home Connect API](https://api-docs.home-connect.com).

1. Create a free account at <https://developer.home-connect.com> and register an application. Pick
   the **Device Flow** OAuth flow, and note the client id and client secret.
2. Authorise SmartWash with your Home Connect app login, using the
   [device flow](https://api-docs.home-connect.com/authorization#device-flow):

   ```bash
   curl -X POST https://api.home-connect.com/security/oauth/device_authorization \
        -d client_id=$CLIENT_ID -d "scope=IdentifyAppliance Washer"
   # open verification_uri_complete from the reply, log in and grant access, then:
   curl -X POST https://api.home-connect.com/security/oauth/token \
        -d grant_type=device_code -d device_code=$DEVICE_CODE -d client_id=$CLIENT_ID
   ```

   The reply contains an `access_token` (valid 24 h) and a `refresh_token`. Use the
   `WasherDryer` scope instead of `Washer` for a washer-dryer.
3. Configure:

   ```yaml
   machine:
     adapter: home_connect
     home_connect:
       access_token: ${HOME_CONNECT_TOKEN}
       refresh_token: ${HOME_CONNECT_REFRESH_TOKEN}
       client_secret: ${HOME_CONNECT_CLIENT_SECRET}   # lets SmartWash renew the 24 h token
       token_cache_path: data/home_connect_tokens.json  # refreshed tokens survive restarts
       # ha_id: BOSCH-WAW28440-68A40E000001  # optional; the first washer is used by default
   ```

   `smartwash devices` lists the washers (`haId`, model, connection state) on the account.
4. On the washer itself, switch on **remote control** once in its Home Connect settings. Then
   press **Remote Start** each time you load it. The machine switches remote start off again after
   about 24 hours.

**Try it without hardware.** The developer portal has virtual appliances. Set
`base_url: https://simulator.home-connect.com` and use the simulator token shown in the portal.

**Limits.** The API allows 1000 requests per day, 50 per minute, and 5 programme starts and 5
stops per minute. Ten consecutive failed requests block the client for 10 minutes. Each status
refresh costs two requests, so the adapter caches the state for `min_poll_interval_s` (default
60 s). The API has no prewash or extra-rinse option for washers; those settings are ignored with a
warning. Programme keys are checked against `programs/available`, and allowed temperatures and
spins come from each programme's constraints.

## Samsung SmartThings

Uses the [SmartThings REST API](https://developer.smartthings.com/docs/api/public).

1. Create a token at <https://account.smartthings.com/tokens> with the **Devices** permissions to
   list, see and control all devices (`r:devices:*`, `x:devices:*`).
   [Tokens created since 30 December 2024 are valid for 24 hours only](https://developer.smartthings.com/docs/getting-started/authorization-and-permissions).
   For unattended use, create a SmartThings OAuth / service integration and pass its access token.
2. Run `smartwash devices`. Each washer is listed with its `id`, its `supported_cycles` and its
   `current_cycle`.
3. **Map our programmes to your model's cycle ids.** Samsung cycle ids, such as
   `Table_02_Course_1B`, differ between models. Select a programme on the washer (say Cotton),
   run `smartwash devices` again, and note `current_cycle`. Repeat for each programme you want to
   use. Short ids such as `1B` are expanded automatically.

   ```yaml
   machine:
     adapter: smartthings
     smartthings:
       token: ${SMARTTHINGS_TOKEN}
       device_id: b854ca5f-dc54-140d-6349-758b4d973c41   # optional; first washer by default
       cycle_map:            # example ids - read yours from `smartwash devices`
         cotton: Table_02_Course_1B
         synthetics: Table_02_Course_1C
         wool: Table_02_Course_25
         quick: Table_02_Course_1E
       # start_preselected_program: true  # start the cycle chosen on the washer if unmapped
   ```

4. Before starting remotely, turn on **Smart Control** on the washer.

**How a start works.** The adapter sends the following, in order: `setWasherCycle`, the water
temperature and spin level, one rinse more than the cycle default when an extra rinse is
recommended, and then `setMachineState: run`. Temperatures and spins come from the chosen cycle's
own option lists. Word levels (`cold`, `warm`, `high`, …) are treated as conservative upper bounds;
for example, `high` counts as 1400 rpm.

**Limits.** There is no prewash option. Some models accept the cycle command but silently ignore
it, so watch the first run. Models that disable the standard `washerOperatingState` capability are
driven through Samsung's `samsungce.washerOperatingState` commands on a best-effort basis. State is
cached for `min_poll_interval_s` (default 10 s).

## LG ThinQ (native)

Uses the official ThinQ Connect API, with the same endpoints and headers as LG's
[`thinqconnect`](https://github.com/thinq-connect/pythinqconnect) SDK.

1. Create a Personal Access Token at <https://connect-pat.lgthinq.com> with your LG ThinQ account.
   Give it the device list, status and control permissions.
2. Configure:

   ```yaml
   machine:
     adapter: lg_thinq
     lg_thinq:
       access_token: ${LG_THINQ_TOKEN}
       country: GB                      # the country of your ThinQ account
       client_id: 6f1a8a52-5d0e-4a8e-9c3b-2b1f0f5a7d11   # any UUID; keep it the same
       start_preselected_program: true  # see the limitation below
   ```

3. Choose the course on the washer, close the door, and press and hold **Remote Start**.

**Limits.** The ThinQ Connect API cannot choose a washer's course, temperature or spin, and cannot
pause. SmartWash shows the recommendation for you to set on the machine. It can then start that
course (only with `start_preselected_program: true`), stop it, and track its progress and
remaining time. Washers and WashTower washers are supported; for WashCombo models, use Home
Assistant.

## Home Assistant (any brand)

Uses the [Home Assistant REST API](https://contributors.home-assistant.io/core/internals/rest-api/). Create a
long-lived access token on your HA profile page, under **Security**. Entity ids depend on the name
of your washer device in HA; find them under *Settings → Devices & services → Entities*. Run
`smartwash devices` to list the HA entities whose name contains "wash".

How the adapter reads the washer:

- `status_entity`'s state is mapped to a status. The defaults cover the Home Connect, LG ThinQ,
  SmartThings and Miele integrations; add unusual states with `status_map`.
- `phase_entity`, which defaults to the status entity, gives the wash phase.
- `remaining_entity` may be a duration (min/s/h), an `H:MM` string or a finish-time timestamp.
- `progress_entity`, `door_entity` and `remote_start_entity` are optional.
- `power_entity` adds power telemetry while running, for example from a smart plug. That lets
  SmartWash measure each cycle's energy.

How it controls the washer: `start`, `pause`, `resume`, `stop`, `prewash` and `extra_rinse` are
HA service calls. If `program_select_entity`, `temperature_select_entity` or `spin_select_entity`
are set, the adapter calls `select.select_option` on each of them first. Service data can use
placeholders: `{program}`, `{temperature}` and `{spin}`, filled with the integration's option
values from `program_map`, `temperature_map` and `spin_map` or parsed from the selects' options,
and `{temperature_c}` and `{spin_rpm}`, filled with plain numbers.

### Example: LG ThinQ integration

```yaml
machine:
  adapter: home_assistant
  home_assistant:
    url: http://homeassistant.local:8123
    token: ${HA_TOKEN}
    status_entity: sensor.washer_current_status
    remaining_entity: sensor.washer_remaining_time
    remote_start_entity: binary_sensor.washer_remote_start
    start_preselected_program: true      # LG cannot choose courses remotely
    start:
      service: select.select_option
      data: {entity_id: select.washer_operation, option: start}
    stop:
      service: select.select_option
      data: {entity_id: select.washer_operation, option: stop}
```

### Example: Home Connect integration

`device_id` is the washer's HA device id: open the device page, and it is the last part of the URL.

```yaml
machine:
  adapter: home_assistant
  home_assistant:
    url: http://homeassistant.local:8123
    token: ${HA_TOKEN}
    status_entity: sensor.washer_operation_state
    door_entity: sensor.washer_door
    remote_start_entity: binary_sensor.washer_remote_start
    progress_entity: sensor.washer_program_progress
    remaining_entity: sensor.washer_program_finish_time
    program_map:
      cotton: laundry_care_washer_program_cotton
      eco_40_60: laundry_care_washer_program_cotton_eco_4060
      synthetics: laundry_care_washer_program_easy_care
      mixed: laundry_care_washer_program_mix
      wool: laundry_care_washer_program_wool
      delicates: laundry_care_washer_program_delicates_silk
      quick: laundry_care_washer_program_super_153045_super_1530
      sportswear: laundry_care_washer_program_sport_fitness
      bedding: laundry_care_washer_program_cotton
      towels: laundry_care_washer_program_towels
      dark_wash: laundry_care_washer_program_dark_wash
      hygiene: laundry_care_washer_program_sensitive
    temperature_map:
      20: laundry_care_washer_enum_type_temperature_g_c_20
      30: laundry_care_washer_enum_type_temperature_g_c_30
      40: laundry_care_washer_enum_type_temperature_g_c_40
      60: laundry_care_washer_enum_type_temperature_g_c_60
      90: laundry_care_washer_enum_type_temperature_g_c_90
    spin_map:
      400: laundry_care_washer_enum_type_spin_speed_r_p_m_400
      600: laundry_care_washer_enum_type_spin_speed_r_p_m_600
      800: laundry_care_washer_enum_type_spin_speed_r_p_m_800
      1000: laundry_care_washer_enum_type_spin_speed_r_p_m_1000
      1200: laundry_care_washer_enum_type_spin_speed_r_p_m_1200
      1400: laundry_care_washer_enum_type_spin_speed_r_p_m_1400
    start:
      service: home_connect.set_program_and_options
      data:
        device_id: 0123456789abcdef0123456789abcdef
        affects_to: active_program
        program: "{program}"
        laundry_care_washer_option_temperature: "{temperature}"
        laundry_care_washer_option_spin_speed: "{spin}"
    pause:  {service: button.press, data: {entity_id: button.washer_pause_program}}
    resume: {service: button.press, data: {entity_id: button.washer_resume_program}}
    stop:   {service: button.press, data: {entity_id: button.washer_stop_program}}
```

### Other integrations

Point `status_entity` at the washer's status sensor. For Miele that is `sensor.<washer>_status`;
for SmartThings, `sensor.<washer>_machine_state` with `phase_entity: sensor.<washer>_job_state`.
Then add whichever service calls the integration offers. Without a way to choose the programme,
set `start_preselected_program: true` to start whatever is selected on the machine.
