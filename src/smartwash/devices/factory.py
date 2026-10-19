"""Builds the configured washing-machine connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import WashingMachineAdapter
from .home_assistant import HomeAssistantAdapter
from .home_connect import HomeConnectAdapter
from .lg_thinq import LgThinqAdapter
from .mqtt import MqttWasherAdapter
from .simulator import SimulatedWasherAdapter
from .smartthings import SmartThingsAdapter

if TYPE_CHECKING:
    from ..config import MachineConfig


def create_adapter(machine: MachineConfig) -> WashingMachineAdapter:
    """A new, not yet connected adapter for ``machine.adapter``."""
    config = machine.adapter_config()
    if machine.adapter == "simulator":
        sim = machine.simulator
        return SimulatedWasherAdapter(
            time_scale=sim.time_scale,
            sample_period_s=sim.sample_period_s,
            fault=sim.fault,
            fault_severity=sim.fault_severity,
            default_load_kg=sim.default_load_kg,
            seed=sim.seed,
        )
    adapters = {
        "mqtt": MqttWasherAdapter,
        "home_connect": HomeConnectAdapter,
        "smartthings": SmartThingsAdapter,
        "home_assistant": HomeAssistantAdapter,
        "lg_thinq": LgThinqAdapter,
    }
    return adapters[machine.adapter](config)
