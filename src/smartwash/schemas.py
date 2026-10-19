"""Domain model shared by the AI models, the device adapters, the controller and the API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

TEMPERATURES_C: tuple[int, ...] = (20, 30, 40, 60, 90)
"""Wash temperatures the system reasons about. 20 °C means a cold wash (no heating)."""

SPIN_SPEEDS_RPM: tuple[int, ...] = (400, 600, 800, 1000, 1200, 1400)


class Fabric(str, Enum):
    """Dominant fabric (or garment type) in the load."""

    COTTON = "cotton"
    SYNTHETIC = "synthetic"
    WOOL = "wool"
    DELICATE = "delicate"  # silk, lace, lingerie, viscose
    DENIM = "denim"
    BEDDING = "bedding"  # sheets, pillowcases, duvet covers
    TOWELS = "towels"
    SPORTSWEAR = "sportswear"  # technical / elastane sports kit
    MIXED = "mixed"  # cotton/synthetic mix
    BABY = "baby"  # baby clothes


class ColorGroup(str, Enum):
    WHITES = "whites"
    LIGHTS = "lights"
    COLORS = "colors"
    DARKS = "darks"


class SoilLevel(str, Enum):
    LIGHT = "light"
    NORMAL = "normal"
    HEAVY = "heavy"


class StainType(str, Enum):
    PROTEIN = "protein"  # blood, sweat, egg, dairy
    GREASE = "grease"  # oil, make-up, food fat
    TANNIN = "tannin"  # wine, coffee, tea, fruit
    MUD = "mud"  # mud, grass, soil


class WaterHardness(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


class Program(str, Enum):
    """Vendor-neutral wash programmes. Adapters map them onto the machine's own programmes."""

    COTTON = "cotton"
    ECO_40_60 = "eco_40_60"
    SYNTHETICS = "synthetics"
    MIXED = "mixed"
    WOOL = "wool"
    DELICATES = "delicates"
    QUICK = "quick"
    SPORTSWEAR = "sportswear"
    BEDDING = "bedding"
    TOWELS = "towels"
    DARK_WASH = "dark_wash"
    HYGIENE = "hygiene"


class LoadProfile(BaseModel):
    """What is going into the drum: the input to the cycle recommender."""

    fabric: Fabric
    color: ColorGroup = ColorGroup.COLORS
    soil: SoilLevel = SoilLevel.NORMAL
    weight_kg: float = Field(default=4.0, gt=0, le=15, description="Dry weight of the load")
    stains: list[StainType] = Field(default_factory=list)
    sensitive_skin: bool = False
    eco_priority: float = Field(
        default=0.5, ge=0, le=1, description="0 = best/fastest clean, 1 = lowest energy use"
    )
    max_duration_min: int | None = Field(
        default=None, gt=0, description="Optional time budget for the cycle"
    )
    water_hardness: WaterHardness = WaterHardness.MEDIUM


class CycleSettings(BaseModel):
    """A concrete, machine-independent wash cycle."""

    program: Program
    temperature_c: int
    spin_rpm: int
    prewash: bool = False
    extra_rinse: bool = False

    @field_validator("temperature_c")
    @classmethod
    def _check_temperature(cls, value: int) -> int:
        if value not in TEMPERATURES_C:
            raise ValueError(f"temperature_c must be one of {TEMPERATURES_C}")
        return value

    @field_validator("spin_rpm")
    @classmethod
    def _check_spin(cls, value: int) -> int:
        if value not in SPIN_SPEEDS_RPM:
            raise ValueError(f"spin_rpm must be one of {SPIN_SPEEDS_RPM}")
        return value


class CycleEstimate(BaseModel):
    duration_min: float
    energy_kwh: float
    water_l: float


class ProgramOption(BaseModel):
    program: Program
    probability: float


class Recommendation(BaseModel):
    settings: CycleSettings
    detergent_ml: int
    estimate: CycleEstimate
    confidence: float = Field(ge=0, le=1)
    alternatives: list[ProgramOption] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(
        default_factory=list,
        description="Corrections the fabric-care guardrails made to the model's raw output",
    )
    warnings: list[str] = Field(default_factory=list)
    model_version: str


class MachineStatus(str, Enum):
    OFFLINE = "offline"
    IDLE = "idle"
    DELAYED_START = "delayed_start"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


ACTIVE_STATUSES = frozenset(
    {MachineStatus.RUNNING, MachineStatus.PAUSED, MachineStatus.DELAYED_START}
)
"""Statuses in which a cycle is in progress."""


class CyclePhase(str, Enum):
    IDLE = "idle"
    FILL = "fill"
    WASH = "wash"
    RINSE = "rinse"
    DRAIN = "drain"
    SPIN = "spin"
    DONE = "done"


class MachineState(BaseModel):
    """Snapshot of the machine as reported by an adapter. Unknown fields stay ``None``."""

    status: MachineStatus
    phase: CyclePhase | None = None
    program: Program | None = None
    vendor_program: str | None = Field(
        default=None, description="Programme name/key exactly as the machine reports it"
    )
    temperature_c: int | None = None
    spin_rpm: int | None = None
    progress_pct: float | None = None
    remaining_min: float | None = None
    door_open: bool | None = None
    remote_start_allowed: bool | None = None
    error_code: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict, description="Vendor data, for debugging")


class TelemetrySample(BaseModel):
    """One sensor reading taken while a cycle runs.

    ``timestamp`` is in seconds of machine time. ``target_rpm`` is the drum speed the programme is
    asking for at that moment and ``target_temp_c`` the water temperature it is heating to (``None``
    when no heating is demanded).
    """

    timestamp: float
    phase: CyclePhase
    drum_rpm: float | None = None
    target_rpm: float | None = None
    motor_current_a: float | None = None
    vibration_g: float | None = None
    water_temp_c: float | None = None
    target_temp_c: float | None = None
    water_level_l: float | None = None
    power_w: float | None = None


FAULT_DETECTION_FIELDS: frozenset[str] = frozenset(
    {
        "drum_rpm",
        "target_rpm",
        "motor_current_a",
        "vibration_g",
        "water_temp_c",
        "water_level_l",
        "power_w",
    }
)
"""Telemetry fields the fault detector needs (``target_temp_c`` is optional)."""


class FaultType(str, Enum):
    NONE = "none"
    UNBALANCED_LOAD = "unbalanced_load"
    HEATER_FAILURE = "heater_failure"
    DRAIN_BLOCKAGE = "drain_blockage"
    BEARING_WEAR = "bearing_wear"
    WATER_SUPPLY = "water_supply"
    MOTOR_OVERLOAD = "motor_overload"
    UNKNOWN = "unknown_anomaly"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Diagnosis(BaseModel):
    """The fault detector's verdict on one window of telemetry."""

    timestamp: float
    phase: CyclePhase
    fault: FaultType
    confidence: float = Field(ge=0, le=1)
    anomaly_score: float = Field(description="Isolation-forest score; above ~0.6 is unusual")
    severity: Severity
    message: str
    recommended_action: str
