"""Configuration: one YAML file, with ``${ENV_VAR}`` references for secrets."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .devices.home_assistant import HomeAssistantConfig
from .devices.home_connect import HomeConnectConfig
from .devices.lg_thinq import LgThinqConfig
from .devices.mqtt import MqttConfig
from .devices.smartthings import SmartThingsConfig
from .schemas import FaultType

AdapterName = Literal[
    "simulator", "mqtt", "home_connect", "smartthings", "home_assistant", "lg_thinq"
]

ADAPTER_CONFIGS: dict[str, type[BaseModel]] = {
    "mqtt": MqttConfig,
    "home_connect": HomeConnectConfig,
    "smartthings": SmartThingsConfig,
    "home_assistant": HomeAssistantConfig,
    "lg_thinq": LgThinqConfig,
}

DEFAULT_CONFIG_FILE = "config.yaml"
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """The configuration file is missing, malformed or incomplete."""


def expand_env(value: Any, where: str = "config") -> Any:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in every string of a parsed YAML document."""
    if isinstance(value, dict):
        return {k: expand_env(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"{where} uses ${{{name}}}, but the environment variable is not set")

    return _ENV_REFERENCE.sub(substitute, value)


class SimulatorConfig(BaseModel):
    time_scale: float = Field(default=60.0, gt=0, description="Machine seconds per real second")
    sample_period_s: float = Field(default=2.0, gt=0)
    fault: FaultType = FaultType.NONE
    fault_severity: float = Field(default=1.0, ge=0)
    default_load_kg: float = Field(default=4.0, gt=0)
    seed: int | None = None


class MachineConfig(BaseModel):
    adapter: AdapterName = "simulator"
    capacity_kg: float = Field(default=8.0, gt=0)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    # Connection settings for the other adapters stay raw until that adapter is selected, so
    # unused example blocks (and their unset ${ENV_VARS}) do no harm.
    mqtt: dict[str, Any] = Field(default_factory=dict)
    home_connect: dict[str, Any] = Field(default_factory=dict)
    smartthings: dict[str, Any] = Field(default_factory=dict)
    home_assistant: dict[str, Any] = Field(default_factory=dict)
    lg_thinq: dict[str, Any] = Field(default_factory=dict)

    def adapter_config(self) -> BaseModel:
        """The validated connection settings of the selected adapter."""
        if self.adapter == "simulator":
            return self.simulator
        raw = expand_env(getattr(self, self.adapter) or {}, f"machine.{self.adapter}")
        try:
            return ADAPTER_CONFIGS[self.adapter].model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(f"machine.{self.adapter} is incomplete:\n{exc}") from exc


class ModelsConfig(BaseModel):
    dir: Path = Path("models")
    auto_train: bool = True
    recommender_samples: int = Field(default=20_000, ge=500)
    fault_cycles: int = Field(default=300, ge=20)


class MonitoringConfig(BaseModel):
    enabled: bool = True
    interval_s: float = Field(default=2.0, gt=0)
    consecutive_windows: int = Field(default=2, ge=1)
    """How many one-minute telemetry windows in a row must agree before a fault is reported."""
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    auto_pause_on_critical: bool = True
    """Pause the cycle when a critical fault (drain blockage, motor overload) is confirmed."""


class NotificationsConfig(BaseModel):
    webhook_url: str | None = None
    """Receives a JSON POST for each event (works with Home Assistant, Slack, n8n, ...)."""
    timeout_s: float = Field(default=5.0, gt=0)
    events: list[str] = Field(
        default_factory=lambda: [
            "cycle_started",
            "cycle_finished",
            "cycle_stopped",
            "machine_error",
            "fault_detected",
            "auto_paused",
        ]
    )

    @field_validator("webhook_url")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: str | None = None
    """When set, every request except /health needs an ``X-API-Key`` header with this value."""

    @field_validator("api_key")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None


class Settings(BaseModel):
    machine: MachineConfig = Field(default_factory=MachineConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    data_dir: Path = Path("data")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from ``path``, ``$SMARTWASH_CONFIG`` or ./config.yaml (defaults otherwise).

    Relative paths in the file are resolved against the file's directory.
    """
    if path is None:
        path = os.environ.get("SMARTWASH_CONFIG")
        if path is None and Path(DEFAULT_CONFIG_FILE).exists():
            path = DEFAULT_CONFIG_FILE
    if path is None:
        return Settings()
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file {path} does not exist")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    machine = data.pop("machine", None) or {}
    data = expand_env(data)
    adapter_blocks = {k: v for k, v in machine.items() if k in ADAPTER_CONFIGS}
    machine = {
        **expand_env({k: v for k, v in machine.items() if k not in ADAPTER_CONFIGS}, "machine"),
        **adapter_blocks,
    }
    try:
        settings = Settings.model_validate({**data, "machine": machine})
    except ValidationError as exc:
        raise ConfigError(f"{path} has invalid settings:\n{exc}") from exc
    base = path.resolve().parent
    if not settings.models.dir.is_absolute():
        settings.models.dir = base / settings.models.dir
    if not settings.data_dir.is_absolute():
        settings.data_dir = base / settings.data_dir
    return settings
