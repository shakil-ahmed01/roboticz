"""Command-line interface. Run ``smartwash --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from . import __version__
from .config import ConfigError, Settings, expand_env, load_settings
from .devices.base import DeviceError, WashingMachineAdapter
from .schemas import (
    SPIN_SPEEDS_RPM,
    TEMPERATURES_C,
    ColorGroup,
    CycleSettings,
    Fabric,
    FaultType,
    LoadProfile,
    MachineStatus,
    Program,
    Recommendation,
    Severity,
    SoilLevel,
    StainType,
    WaterHardness,
)

if TYPE_CHECKING:
    from .ai.registry import ModelBundle

log = logging.getLogger("smartwash")
SIMULATED_FAULTS = [f.value for f in FaultType if f is not FaultType.UNKNOWN]


def _values(enum: type) -> list[str]:
    return [member.value for member in enum]


def _add_profile_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    group = parser.add_argument_group("load (the AI chooses the settings)")
    group.add_argument("--fabric", choices=_values(Fabric), required=required)
    group.add_argument("--color", choices=_values(ColorGroup), default="colors")
    group.add_argument("--soil", choices=_values(SoilLevel), default="normal")
    group.add_argument("--weight", type=float, default=4.0, help="dry weight in kg (default 4)")
    group.add_argument(
        "--stain", action="append", choices=_values(StainType), default=[], help="repeatable"
    )
    group.add_argument("--sensitive-skin", action="store_true")
    group.add_argument(
        "--eco", type=float, default=0.5, help="0 = best/fastest clean ... 1 = lowest energy"
    )
    group.add_argument("--max-duration", type=int, help="time budget in minutes")
    group.add_argument("--water-hardness", choices=_values(WaterHardness), default="medium")


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("explicit settings (instead of, or checked against, a load)")
    group.add_argument("--program", choices=_values(Program))
    group.add_argument("--temperature", type=int, choices=TEMPERATURES_C)
    group.add_argument("--spin", type=int, choices=SPIN_SPEEDS_RPM)
    group.add_argument("--prewash", action="store_true")
    group.add_argument("--extra-rinse", action="store_true")


def _profile(args: argparse.Namespace) -> LoadProfile | None:
    if args.fabric is None:
        return None
    return LoadProfile(
        fabric=args.fabric,
        color=args.color,
        soil=args.soil,
        weight_kg=args.weight,
        stains=args.stain,
        sensitive_skin=args.sensitive_skin,
        eco_priority=args.eco,
        max_duration_min=args.max_duration,
        water_hardness=args.water_hardness,
    )


def _settings(args: argparse.Namespace) -> CycleSettings | None:
    if args.program is None:
        if args.temperature is not None or args.spin is not None:
            raise SystemExit("--temperature and --spin need --program")
        return None
    if args.temperature is None or args.spin is None:
        raise SystemExit("--program needs --temperature and --spin")
    return CycleSettings(
        program=args.program,
        temperature_c=args.temperature,
        spin_rpm=args.spin,
        prewash=args.prewash,
        extra_rinse=args.extra_rinse,
    )


def _load_models(settings: Settings) -> ModelBundle:
    from .ai.feedback import FeedbackStore
    from .ai.registry import ModelBundle

    return ModelBundle.load_or_train(
        settings.models.dir,
        auto_train=settings.models.auto_train,
        feedback=FeedbackStore(settings.data_dir / "feedback.jsonl").load(),
        recommender_samples=settings.models.recommender_samples,
        fault_cycles=settings.models.fault_cycles,
    )


def _connect(settings: Settings) -> WashingMachineAdapter:
    from .devices.factory import create_adapter

    adapter = create_adapter(settings.machine)
    adapter.connect()
    return adapter


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def format_recommendation(rec: Recommendation) -> str:
    s = rec.settings
    yes_no = {True: "yes", False: "no"}
    lines = [
        f"Recommended cycle (confidence {rec.confidence:.0%})",
        f"  Programme    {s.program.value}",
        f"  Temperature  {s.temperature_c} °C",
        f"  Spin         {s.spin_rpm} rpm",
        f"  Pre-wash     {yes_no[s.prewash]}",
        f"  Extra rinse  {yes_no[s.extra_rinse]}",
        f"  Detergent    about {rec.detergent_ml} ml",
        f"  Estimate     {rec.estimate.duration_min:.0f} min, {rec.estimate.energy_kwh:.2f} kWh, "
        f"{rec.estimate.water_l:.0f} L of water",
        "Why:",
        *(f"  - {reason}" for reason in rec.explanation),
    ]
    if rec.adjustments:
        lines += ["Adjusted by the fabric-care rules:", *(f"  - {a}" for a in rec.adjustments)]
    if rec.warnings:
        lines += ["Warnings:", *(f"  ! {w}" for w in rec.warnings)]
    if rec.alternatives:
        options = ", ".join(f"{a.program.value} ({a.probability:.0%})" for a in rec.alternatives)
        lines.append(f"Alternatives: {options}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ commands


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    from .ai.feedback import FeedbackStore
    from .ai.registry import ModelBundle, load_calibration

    feedback = (
        [] if args.no_feedback else FeedbackStore(settings.data_dir / "feedback.jsonl").load()
    )
    bundle = ModelBundle.train(
        recommender_samples=args.samples or settings.models.recommender_samples,
        fault_cycles=args.fault_cycles or settings.models.fault_cycles,
        feedback=feedback,
        seed=args.seed,
        estimator=load_calibration(settings.models.dir),
    )
    bundle.save(settings.models.dir)
    rec, det = bundle.metadata["recommender"], bundle.metadata["fault_detector"]
    print(f"Saved models to {settings.models.dir}")
    print(
        f"Cycle recommender ({rec['train_samples']} training loads, "
        f"{rec['feedback_examples']} from your feedback):"
    )
    print(
        f"  agreement with laundry expert: programme {rec['program_agreement_with_expert']:.1%}, "
        f"temperature {rec['temperature_c_agreement_with_expert']:.1%}, "
        f"spin {rec['spin_rpm_agreement_with_expert']:.1%}"
    )
    print(f"  guardrails had to correct {rec['guardrail_intervention_rate']:.1%} of raw outputs")
    print(f"Fault detector ({det['training_cycles']} simulated training cycles):")
    print(
        f"  window accuracy {det['window_accuracy']:.1%}, "
        f"false alarms on healthy cycles {det['cycle_false_alarm_rate']:.1%}"
    )
    for fault, rate in det["cycle_detection_rate"].items():
        print(f"  {fault:<16} detected in {rate:.0%} of test cycles")
    return 0


def cmd_recommend(args: argparse.Namespace, settings: Settings) -> int:
    models = _load_models(settings)
    profile = _profile(args)
    assert profile is not None
    rec = models.recommender.recommend(
        profile, estimator=models.estimator, capacity_kg=settings.machine.capacity_kg
    )
    print(rec.model_dump_json(indent=2) if args.json else format_recommendation(rec))
    return 0


def cmd_simulate(args: argparse.Namespace, settings: Settings) -> int:
    """Run one virtual cycle, at full speed, through the fault detector."""
    from .ai.anomaly import FaultDebouncer, TelemetryWindower
    from .devices.simulator import WasherSimulator

    models = _load_models(settings)
    profile = _profile(args) or LoadProfile(fabric="cotton", weight_kg=args.weight)
    cycle = (
        _settings(args)
        or models.recommender.recommend(profile, estimator=models.estimator).settings
    )
    estimate = models.estimator.estimate(cycle, profile.weight_kg)
    sim = WasherSimulator(seed=args.seed)
    sim.start(cycle, profile.weight_kg, fault=FaultType(args.fault), severity=args.severity)
    print(
        f"Simulating {cycle.program.value} {cycle.temperature_c} °C {cycle.spin_rpm} rpm with "
        f"{profile.weight_kg:g} kg (injected fault: {args.fault}); "
        f"estimate {estimate.duration_min:.0f} min"
    )
    windower = TelemetryWindower()
    debouncer = FaultDebouncer(
        settings.monitoring.consecutive_windows, settings.monitoring.min_confidence
    )
    phase = None
    energy_j = 0.0
    paused = False
    while sim.status is MachineStatus.RUNNING and not paused:
        record = sim.step()
        if record is None:
            continue
        sample = record.sample
        energy_j += (sample.power_w or 0.0) * sim.dt
        minute = sample.timestamp / 60.0
        if sample.phase is not phase:
            phase = sample.phase
            print(f"{minute:7.1f} min  {phase.value}")
        for window in windower.push(sample):
            diagnosis = models.detector.assess(window)
            if diagnosis is None or debouncer.update(diagnosis) is None:
                continue
            print(
                f"{minute:7.1f} min  !! {diagnosis.fault.value} detected "
                f"({diagnosis.confidence:.0%} confident, {diagnosis.severity.value})"
            )
            print(f"             {diagnosis.message}")
            print(f"             Advice: {diagnosis.recommended_action}")
            if (
                diagnosis.severity is Severity.CRITICAL
                and settings.monitoring.auto_pause_on_critical
            ):
                print("             SmartWash pauses the cycle here.")
                paused = True
                break
    duration = sim.t / 60.0
    outcome = "paused by SmartWash" if paused else sim.status.value
    if sim.error_code:
        outcome += f" ({sim.error_code})"
    print(
        f"Cycle {outcome} after {duration:.0f} min, {energy_j / 3.6e6:.2f} kWh "
        f"(estimated {estimate.duration_min:.0f} min, {estimate.energy_kwh:.2f} kWh)."
    )
    if not debouncer.confirmed:
        print("No faults detected.")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .api import create_app

    host = args.host or settings.api.host
    port = args.port or settings.api.port
    if host not in ("127.0.0.1", "localhost", "::1") and not settings.api.api_key:
        log.warning(
            "Serving on %s without an API key: anyone who can reach this port can control your "
            "washing machine. Set api.api_key in the config.",
            host,
        )
    uvicorn.run(create_app(settings), host=host, port=port)
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    with _connect(settings) as adapter:
        _print_json(adapter.get_state().model_dump(mode="json"))
    return 0


def cmd_devices(args: argparse.Namespace, settings: Settings) -> int:
    with _connect(settings) as adapter:
        _print_json(adapter.list_devices())
    return 0


def cmd_start(args: argparse.Namespace, settings: Settings) -> int:
    from .controller import StartCycleRequest, build_controller

    if settings.machine.adapter == "simulator":
        raise SystemExit(
            "The simulator only lives inside a running server: use `smartwash serve` and the API "
            "(or `smartwash simulate` for an offline run)."
        )
    profile, cycle = _profile(args), _settings(args)
    if profile is None and cycle is None:
        raise SystemExit(
            "describe the load (--fabric ...) and/or give --program/--temperature/--spin"
        )
    controller = build_controller(settings)
    try:
        result = controller.start_cycle(StartCycleRequest(profile=profile, settings=cycle))
    finally:
        controller.close()
    if result.recommendation:
        print(format_recommendation(result.recommendation))
    for note in result.adjustments:
        print(f"Adjusted: {note}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print(f"Started. The machine reports: {result.state.status.value}.")
    return 0


def _command(action: str) -> Callable[[argparse.Namespace, Settings], int]:
    def run(args: argparse.Namespace, settings: Settings) -> int:
        with _connect(settings) as adapter:
            getattr(adapter, action)()
            _print_json(adapter.get_state().model_dump(mode="json"))
        return 0

    return run


def cmd_virtual_device(args: argparse.Namespace, settings: Settings) -> int:
    from .devices.mqtt import MqttConfig
    from .devices.mqtt_device import run_virtual_device

    raw = expand_env(settings.machine.mqtt, "machine.mqtt")
    overrides = {
        "host": args.host,
        "port": args.port,
        "base_topic": args.base_topic,
        "username": args.username,
        "password": args.password,
    }
    config = MqttConfig.model_validate({**raw, **{k: v for k, v in overrides.items() if v}})
    print(
        f"Virtual washer on mqtt://{config.host}:{config.port}/{config.base_topic} at "
        f"{args.time_scale:g}x real time (fault: {args.fault}). Ctrl+C to stop."
    )
    run_virtual_device(
        config,
        time_scale=args.time_scale,
        fault=FaultType(args.fault),
        fault_severity=args.severity,
        seed=args.seed,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartwash", description="AI brain for smart washing machines."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-c", "--config", help="YAML config (default: $SMARTWASH_CONFIG or ./config.yaml)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    train = commands.add_parser("train", help="train the AI models (and learn from feedback)")
    train.add_argument("--samples", type=int, help="synthetic loads for the recommender")
    train.add_argument("--fault-cycles", type=int, help="simulated cycles for the fault detector")
    train.add_argument("--no-feedback", action="store_true", help="ignore stored feedback")
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    recommend = commands.add_parser("recommend", help="recommend a cycle for a load")
    _add_profile_args(recommend, required=True)
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=cmd_recommend)

    simulate = commands.add_parser(
        "simulate", help="run a virtual cycle through the fault detector (offline, instant)"
    )
    _add_profile_args(simulate, required=False)
    _add_settings_args(simulate)
    simulate.add_argument("--fault", choices=SIMULATED_FAULTS, default="none")
    simulate.add_argument(
        "--severity", type=float, default=1.0, help="0.3 = early stage, 1.4 = severe"
    )
    simulate.add_argument("--seed", type=int, default=0)
    simulate.set_defaults(func=cmd_simulate)

    serve = commands.add_parser("serve", help="run the REST API and live monitoring")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    commands.add_parser("status", help="show the machine's state").set_defaults(func=cmd_status)
    commands.add_parser("devices", help="list washers on the configured account").set_defaults(
        func=cmd_devices
    )

    start = commands.add_parser("start", help="start a cycle on the configured machine")
    _add_profile_args(start, required=False)
    _add_settings_args(start)
    start.set_defaults(func=cmd_start)
    for action in ("pause", "resume", "stop"):
        commands.add_parser(action, help=f"{action} the running cycle").set_defaults(
            func=_command(action)
        )

    virtual = commands.add_parser(
        "virtual-device", help="run a simulated washer that speaks the MQTT protocol"
    )
    virtual.add_argument("--host", help="MQTT broker (default: machine.mqtt.host or localhost)")
    virtual.add_argument("--port", type=int)
    virtual.add_argument("--base-topic")
    virtual.add_argument("--username")
    virtual.add_argument("--password")
    virtual.add_argument("--time-scale", type=float, default=60.0, help="default 60x real time")
    virtual.add_argument("--fault", choices=SIMULATED_FAULTS, default="none")
    virtual.add_argument("--severity", type=float, default=1.0)
    virtual.add_argument("--seed", type=int)
    virtual.set_defaults(func=cmd_virtual_device)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args, load_settings(args.config))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except DeviceError as exc:
        print(f"Machine error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
