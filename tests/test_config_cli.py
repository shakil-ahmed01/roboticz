from __future__ import annotations

import json
import textwrap

import pytest

from smartwash.cli import main
from smartwash.config import ConfigError, Settings, load_settings
from smartwash.devices.factory import create_adapter
from smartwash.devices.home_assistant import HomeAssistantAdapter
from smartwash.devices.home_connect import HomeConnectAdapter, HomeConnectConfig
from smartwash.devices.lg_thinq import LgThinqAdapter
from smartwash.devices.mqtt import MqttWasherAdapter
from smartwash.devices.simulator import SimulatedWasherAdapter
from smartwash.devices.smartthings import SmartThingsAdapter


def write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_environment_variables_fill_in_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HC_TOKEN", "abc123")
    path = write(
        tmp_path,
        """
        machine:
          adapter: home_connect
          home_connect:
            access_token: ${HC_TOKEN}
            base_url: ${HC_URL:-https://simulator.home-connect.com}
          smartthings:
            token: ${NOT_SET_BUT_UNUSED}
        models:
          dir: trained
        """,
    )
    settings = load_settings(path)
    config = settings.machine.adapter_config()
    assert isinstance(config, HomeConnectConfig)
    assert config.access_token == "abc123"
    assert config.base_url == "https://simulator.home-connect.com"
    assert settings.models.dir == tmp_path / "trained"
    assert settings.data_dir == tmp_path / "data"


def test_missing_secret_for_the_selected_adapter_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ST_TOKEN", raising=False)
    path = write(
        tmp_path,
        """
        machine:
          adapter: smartthings
          smartthings:
            token: ${ST_TOKEN}
        """,
    )
    with pytest.raises(ConfigError, match="ST_TOKEN"):
        load_settings(path).machine.adapter_config()


def test_bad_config_files(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_settings(write(tmp_path, "machine: [unclosed"))
    with pytest.raises(ConfigError, match="invalid settings"):
        load_settings(write(tmp_path, "machine:\n  adapter: toaster\n"))
    incomplete = load_settings(write(tmp_path, "machine:\n  adapter: home_assistant\n"))
    with pytest.raises(ConfigError, match="incomplete"):
        incomplete.machine.adapter_config()


def test_defaults_without_a_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMARTWASH_CONFIG", raising=False)
    settings = load_settings()
    assert settings.machine.adapter == "simulator"
    assert settings.api.host == "127.0.0.1"
    assert settings.api.api_key is None


@pytest.mark.parametrize(
    ("adapter", "block", "expected"),
    [
        ("simulator", {}, SimulatedWasherAdapter),
        ("mqtt", {"host": "broker.local"}, MqttWasherAdapter),
        ("home_connect", {"access_token": "t"}, HomeConnectAdapter),
        ("smartthings", {"token": "t"}, SmartThingsAdapter),
        ("home_assistant", {"token": "t", "status_entity": "sensor.washer"}, HomeAssistantAdapter),
        ("lg_thinq", {"access_token": "t", "country": "GB"}, LgThinqAdapter),
    ],
)
def test_factory_builds_every_adapter(adapter, block, expected):
    settings = Settings.model_validate({"machine": {"adapter": adapter, adapter: block}})
    instance = create_adapter(settings.machine)
    assert isinstance(instance, expected)
    instance.close()


@pytest.fixture
def cli_config(tmp_path, model_dir):
    return str(write(tmp_path, f"models:\n  dir: {model_dir}\n  auto_train: false\n"))


def test_cli_recommend(cli_config, capsys):
    assert main(["-c", cli_config, "recommend", "--fabric", "wool", "--weight", "1"]) == 0
    out = capsys.readouterr().out
    assert "Programme    wool" in out
    assert "Why:" in out
    assert main(["-c", cli_config, "recommend", "--fabric", "towels", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["settings"]["program"] == "towels"


def test_cli_simulate_catches_an_injected_fault(cli_config, capsys):
    args = ["-c", cli_config, "simulate", "--program", "cotton", "--temperature", "40"]
    assert main([*args, "--spin", "1200", "--fault", "drain_blockage"]) == 0
    out = capsys.readouterr().out
    assert "drain_blockage detected" in out
    assert "SmartWash pauses the cycle here." in out
    assert main(["-c", cli_config, "simulate", "--fabric", "cotton"]) == 0
    assert "No faults detected." in capsys.readouterr().out


def test_cli_status_and_errors(cli_config, tmp_path, capsys):
    assert main(["-c", cli_config, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "idle"
    with pytest.raises(SystemExit, match="only lives inside a running server"):
        main(["-c", cli_config, "start", "--fabric", "cotton"])
    assert main(["-c", str(tmp_path / "nope.yaml"), "status"]) == 2
    assert "Configuration error" in capsys.readouterr().err
