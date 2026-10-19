# The SmartWash AI models

SmartWash combines three learned components with hard-coded domain knowledge:

| Component | Learns from | Model | Code |
|---|---|---|---|
| Cycle recommender | synthetic expert-labelled loads + your feedback | 5 random forests | `ai/recommender.py` |
| Cycle estimator | physics + your machine's completed cycles | hierarchical calibration factors | `ai/estimator.py` |
| Fault detector | simulated healthy/faulty telemetry | random forest + isolation forest + per-phase envelope | `ai/anomaly.py` |

The fabric-care guardrails (`knowledge.py`) are deliberately *not* learned. They are the safety
layer every model output passes through.

Train everything with `smartwash train`, which takes about 30 s on a laptop. The models are saved
to `models/` together with their evaluation metrics (`metadata.json`). They retrain
automatically if your scikit-learn version changes, because pickled models are not portable
across versions.

## Cycle recommender

**Input:** a load profile, encoded as 25 numbers:
- fabric and colour (one-hot);
- soil level and dry weight;
- stain types (multi-hot);
- sensitive skin;
- eco priority;
- time budget;
- water hardness.

**Output:** one random forest (60 trees, `min_samples_leaf=3`) per setting: programme (12 classes),
temperature (20/30/40/60/90 °C), spin (400-1400 rpm), pre-wash and extra rinse. The detergent
dose comes from a formula, not the model: 9 ml/kg, scaled by soil level, water hardness and fabric.

**Training data.** The training data is 20,000 synthetic loads drawn from a realistic household
mix: cotton 22%, mixed 14%, synthetics 11%, and so on, with fabric-specific weight ranges,
stains, eco preferences and time budgets. Each load is labelled by `knowledge.expert_settings`, a
rule-based laundry expert built from:
- care-label limits;
- colour fastness;
- soil and stain handling (pre-wash for mud and protein stains, at least 40 °C for grease and
  tannin);
- eco programmes;
- quick programmes for small, lightly soiled loads.

The expert samples the spread of choices real experts make:
- 25% wash coloured cotton at 30 °C rather than 40 °C;
- a light soil level lowers the temperature 80% of the time;
- eco mode follows a soft threshold around an eco priority of 0.7;
- the spin speed varies by one step.

That spread is what makes the model's **confidence** and **alternatives** meaningful. Confidence
is the geometric mean of the top probabilities for programme, temperature and spin. It is reduced
by 15% when the guardrails had to correct the raw output.

**Evaluation.** The evaluation uses 4,000 held-out loads:

| Setting | Agreement with the expert's consensus choice |
|---|---|
| Programme | 96.2% |
| Temperature | 97.4% |
| Spin | 99.3% |
| Pre-wash / extra rinse | 98.4% / 100% |
| All five exactly | 92.4% |
| Raw outputs the guardrails had to correct | 0.5% |

**Personalisation.** Feedback comes from `POST /feedback`, or is recorded automatically when you
start a cycle with your own settings for a described load. Each piece of feedback joins the
training set with 30 times the weight of a synthetic example:
- Ratings of 1-2 stars are not learned from.
- A 3-star rating counts half.
- Every piece of feedback is passed through the guardrails first, so a mistake (wool at 60 °C)
  can never be learned.

In the tests, five examples of "we wash everyday coloured cotton at 30 °C / 1000 rpm" shift
recommendations for similar loads, while wool and towels are unaffected. Retrain with
`POST /model/retrain` or `smartwash train`.

## Fabric-care guardrails

`enforce_care_constraints(profile, settings)` returns safe settings plus a note for each change:

1. **Programme:** if its drum action is harsher than the fabric tolerates, or it needs a hotter
   wash than the load allows (Hygiene ≥ 60 °C, Eco 40-60 ≥ 40 °C), it is replaced with the fabric's
   default programme. If that doesn't fit either, it falls back to Cotton and then Delicates.
2. **Temperature:** capped at the lowest of the fabric limit, the colour limit and the programme
   limit:
   - fabric limits: wool and delicates 30 °C, denim and sportswear 40 °C, synthetics 60 °C;
   - colour limits: colours and darks 40 °C, lights 60 °C;
   - programme limits: Quick 40 °C, for example.
3. **Spin:** capped by fabric (delicates 600 rpm, wool 800, sportswear 1000, ...) and by programme.
4. **Pre-wash:** removed for wool, delicates and programmes without one.
5. **Extra rinse:** always on for sensitive skin and baby clothes.

The adapters add a second layer:
- They round temperature and spin *down* to what the machine supports.
- They refuse to start if the machine can't go low enough.
- They only substitute an equal or gentler programme.

## Cycle estimator

`physics.py` models a typical 8 kg front-loader:
- the fill rate;
- heating, with a 2 kW element, the thermal mass of water plus laundry, and heat loss;
- the thermostat hold;
- intermittent tumbling with a programme-specific duty cycle;
- the drain rate;
- spin ramps.

It also includes eco programmes, which heat less and wash longer. The simulated and analytic
estimates agree within about 1% on duration and energy for most programmes.

Every cycle that finishes teaches it your machine:
- Duration is always learned.
- Energy is learned when telemetry includes power.

For each programme *p* and measure (duration, energy, water):

```text
machine factor = (Σ all ratios + k) / (n_all + k)
programme factor = (Σ ratios_p + k · machine factor) / (n_p + k)       k = 2
estimate = physics estimate × programme factor
```

A programme you have never used still benefits from what other programmes taught it. After three
cycles at 1.3× the physics estimate, the factor is about 1.25. Ratios are clipped to 0.2-5 so one
broken reading can't do damage.

## Fault detector

**Telemetry it needs.** Each sample needs these fields; `target_temp_c` is optional, and without it
heater faults can't be detected. See [mqtt-protocol.md](mqtt-protocol.md).

| Field | Unit |
|---|---|
| `drum_rpm` | rpm |
| `target_rpm` | rpm |
| `motor_current_a` | A |
| `vibration_g` | g RMS |
| `water_temp_c` | °C |
| `water_level_l` | L |
| `power_w` | W |
| `target_temp_c` (optional) | °C |

Each sample also carries `phase`: fill, wash, rinse, drain or spin. The detector is designed for
roughly one sample every 2 s.

**Windows and features.** The stream is cut into windows of one phase, up to 60 s long. A window
closes early on a phase change or a gap of more than 15 s, such as a pause, and windows under 30 s
are dropped. Each window becomes 29 features:

- phase (one-hot);
- **drum:** target and actual speed, speed deficit (absolute and relative), speed variation and
  slope (spin ramp versus a stall);
- **vibration:** mean, 95th percentile, variation, and vibration per unit of (speed/1000)²,
  which separates imbalance and bearing wear from normal high-speed vibration;
- **motor current:** mean, 95th percentile, mean while turning;
- **heating:** temperature and its slope, how often heat is demanded, temperature deficit,
  heater power (total power minus motor power), and *heating efficiency*. Heating efficiency is
  the measured temperature rise divided by what a working 2 kW heater achieves at the measured
  water level. It is about 1 when healthy and near 0 when the heater has failed, whatever the
  target temperature;
- **water:** level and its slope (fill and drain rates);
- **power:** mean, 95th percentile, and non-motor power.

**Models.**
- A **random forest** (150 trees) classifies each window as healthy or one of the six known faults.
  A fault needs at least 50% probability.
- An **isolation forest** (200 trees), trained on healthy windows, scores unusual *combinations* of
  signals.
- A **per-phase envelope** records the 0.1-99.9% range of every feature in healthy cycles, phase by
  phase. It flags single signals far outside it, which an isolation forest dilutes across many
  features. The diagnosis names the most unusual signal.

If the classifier sees no known fault but either novelty check fires, the window is reported as
`unknown_anomaly`.

**Debouncing.** A fault is confirmed after 2 consecutive windows agree with at least 60%
confidence, and once per cycle. Both numbers are configurable. Confirmed faults trigger a
notification, and critical ones pause the cycle:

| Fault | What the simulator does | Severity | Auto-pause |
|---|---|---|---|
| `drain_blockage` | drain rate falls to 4-21% | critical | yes |
| `motor_overload` | current ×1.3-2.3 and the drum can't hold its speed | critical | yes |
| `heater_failure` | heater power falls to 0-73% | warning | no |
| `bearing_wear` | extra vibration (more at speed) and friction current | warning | no |
| `unbalanced_load` | imbalance limiter caps the spin at about 330-880 rpm, and vibration rises | warning | no |
| `water_supply` | fill rate falls to 10-38% | warning | no |
| `unknown_anomaly` | anything else outside healthy behaviour | warning | no |

**Training data.** There are 300 simulated cycles:
- **Mix:** 30% are healthy, and the rest are spread evenly over the six faults.
- **Settings:** 70% use expert settings for random loads. 30% use arbitrary hand-picked
  programmes and loads, so normal includes real usage and not just ideal choices.
- **Machine variation:** the machine varies from cycle to cycle (fill, drain and heater rates,
  imbalance, inlet and room temperature).
- **Severity:** faults range from early-stage (0.3) to severe (1.4).
- **Exercising the fault:** each faulty cycle uses settings that exercise its fault. A broken
  heater is invisible in a cold wash, and imbalance only shows at high spin.
- **Labels:** a window gets a fault label only while the fault can be seen, for example a drain
  fault during drains.
- **Speed:** timed holds are shortened to 35% to speed up generation. Fills, drains and heating
  run at real speed.

**Calibration and evaluation.** Training holds out whole cycles: 70% train, 10% calibrate and 20%
test, stratified by healthy versus faulty.

Windows within one cycle are strongly correlated, so a per-window false-alarm rate understates
how often a debounced alarm fires. The novelty thresholds are therefore raised until none of
40 extra *complete, real-length* healthy cycles would raise a confirmed alarm. The rate users
would see is then measured on 30 more:

| Metric | Result |
|---|---|
| Window accuracy (test cycles) | 95.7% |
| Faulty test cycles caught: drain, heater, bearing, imbalance, water supply | 100% |
| Faulty test cycles caught: motor overload | 89% (mild overloads resemble heavy loads) |
| Misdiagnosed faulty cycles | 0% |
| False alarms: healthy test cycles / full-length healthy hold-out | 0% / 0% |
| Median delay from a fault first showing to its confirmation | 1 extra window (~1 min) |

## Limits, and how to go further with real data

- **Synthetic data.** Every number above comes from the same expert rules and simulator the
  models were trained on. It shows the models learned what they were taught, not how a
  particular real machine behaves. Real sensors are noisier and real faults are messier.
- **Recommendations.** The expert encodes general guidance. Care labels win, and so does your
  experience: feed it back, and the recommender adapts.
- **Moving the fault detector to real data:**
  1. Log your machine's telemetry. It arrives in the format the detector uses.
  2. Label the cycles where a technician confirmed a fault.
  3. Retrain the classifier on real windows mixed with simulated ones, through the same
     `window_features` and `FaultDetector.fit` API.
  4. Recalibrate the novelty thresholds on your own healthy cycles with
     `calibrate_on_healthy_cycles`.

  Even before that, the novelty checks adapt best: refit them on a few weeks of your machine's
  healthy cycles and they learn its personal "normal".
