import pytest

from smartwash.devices.mapping import (
    PROGRAM_FALLBACKS,
    choose_at_most,
    label_number,
    parse_spin_label,
    parse_temperature_label,
)
from smartwash.schemas import Program


@pytest.mark.parametrize(
    ("label", "celsius"),
    [
        ("LaundryCare.Washer.EnumType.Temperature.GC40", 40),
        ("LaundryCare.Washer.EnumType.Temperature.Cold", 15),
        ("LaundryCare.Washer.EnumType.Temperature.UlWarm", 40),
        ("LaundryCare.Washer.EnumType.Temperature.UlExtraHot", 90),
        ("laundry_care_washer_enum_type_temperature_g_c_60", 60),
        ("tapCold", 15),
        ("extraHot", 90),
        ("20", 20),
        ("40°C", 40),
        ("none", None),
        ("medium", None),
    ],
)
def test_temperature_labels(label, celsius):
    assert parse_temperature_label(label) == celsius


@pytest.mark.parametrize(
    ("label", "rpm"),
    [
        ("LaundryCare.Washer.EnumType.SpinSpeed.RPM1200", 1200),
        ("LaundryCare.Washer.EnumType.SpinSpeed.Off", 0),
        ("LaundryCare.Washer.EnumType.SpinSpeed.UlMedium", 1000),
        ("laundry_care_washer_enum_type_spin_speed_r_p_m_800", 800),
        ("noSpin", 0),
        ("extraHigh", 1600),
        ("rinseHold", None),
        ("none", None),
    ],
)
def test_spin_labels(label, rpm):
    assert parse_spin_label(label) == rpm


def test_choose_at_most_never_exceeds_the_request():
    labels = ["none", "cold", "20", "30", "40", "60", "90"]
    for requested in (20, 30, 40, 60, 90):
        chosen = choose_at_most(requested, labels, parse_temperature_label)
        assert parse_temperature_label(chosen) <= requested
    assert choose_at_most(50, labels, parse_temperature_label) == "40"
    assert choose_at_most(10, ["20", "30"], parse_temperature_label) is None


def test_label_number_reads_only_trailing_numbers():
    assert label_number("Table_00_Course_5B") is None
    assert label_number("GC30") == 30
    assert label_number("warm") is None


def test_every_programme_has_a_fallback_chain_starting_with_itself():
    assert set(PROGRAM_FALLBACKS) == set(Program)
    for program, chain in PROGRAM_FALLBACKS.items():
        assert chain[0] is program


def test_fallbacks_are_never_harsher_than_the_programme_they_replace():
    knowledge = pytest.importorskip("smartwash.knowledge")
    specs = getattr(knowledge, "PROGRAM_SPECS", None)
    if specs is None:
        pytest.skip("knowledge.PROGRAM_SPECS not available")
    for program, chain in PROGRAM_FALLBACKS.items():
        for substitute in chain[1:]:
            assert specs[substitute].intensity <= specs[program].intensity, (program, substitute)
