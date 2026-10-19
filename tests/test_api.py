from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smartwash.api import create_app
from smartwash.config import ApiConfig, MonitoringConfig, Settings

from .test_controller import CloudLikeAdapter, make_controller

COTTON = {"fabric": "cotton", "color": "colors", "weight_kg": 4}


@pytest.fixture
def client(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    settings = Settings(monitoring=MonitoringConfig(enabled=False))
    with TestClient(create_app(settings, controller=controller)) as client:
        yield client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["adapter"] == "simulator"
    assert body["fault_detection"] is True


def test_recommend(client):
    response = client.post("/recommend", json={"fabric": "wool", "color": "darks", "weight_kg": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["settings"]["program"] == "wool"
    assert body["settings"]["temperature_c"] <= 30
    assert body["explanation"]


def test_invalid_load_is_rejected(client):
    assert client.post("/recommend", json={"fabric": "kevlar"}).status_code == 422
    assert client.post("/cycles", json={}).status_code == 422


def test_cycle_lifecycle(client):
    response = client.post("/cycles", json={"profile": COTTON})
    assert response.status_code == 201
    started = response.json()
    assert started["recommendation"]["settings"] == started["settings"]
    assert client.get("/machine").json()["state"]["status"] == "running"
    assert client.post("/cycles", json={"profile": COTTON}).status_code == 409
    assert client.post("/machine/pause").json()["status"] == "paused"
    assert client.post("/machine/pause").status_code == 409
    assert client.post("/machine/resume").json()["status"] == "running"
    assert client.post("/machine/stop").json()["status"] == "idle"
    history = client.get("/cycles").json()
    assert [c["outcome"] for c in history] == ["stopped"]


def test_own_settings_are_checked_against_the_load(client):
    response = client.post(
        "/cycles",
        json={
            "profile": {"fabric": "delicate", "weight_kg": 1},
            "settings": {"program": "cotton", "temperature_c": 60, "spin_rpm": 1400},
        },
    )
    body = response.json()
    assert body["settings"]["program"] == "delicates"
    assert body["settings"]["spin_rpm"] <= 600
    assert body["adjustments"]


def test_diagnostics_feedback_and_model(client):
    assert client.get("/diagnostics").json()["fault_detection"] is True
    feedback = {
        "profile": COTTON,
        "settings": {"program": "cotton", "temperature_c": 30, "spin_rpm": 1000},
        "rating": 5,
    }
    assert client.post("/feedback", json=feedback).json() == {
        "stored": True,
        "feedback_records": 1,
    }
    assert client.get("/model").json()["feedback_records"] == 1


def test_fault_injection(client):
    response = client.post("/simulator/fault", json={"fault": "heater_failure", "severity": 0.7})
    assert response.json()["fault"] == "heater_failure"


def test_api_key_protects_everything_but_health(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock)
    settings = Settings(api=ApiConfig(api_key="s3cret"), monitoring=MonitoringConfig(enabled=False))
    with TestClient(create_app(settings, controller=controller)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/machine").status_code == 401
        assert client.get("/machine", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/machine", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_unsupported_operations_return_501(models, tmp_path, clock):
    controller = make_controller(models, tmp_path, clock, adapter=CloudLikeAdapter())
    settings = Settings(monitoring=MonitoringConfig(enabled=False))
    with TestClient(create_app(settings, controller=controller)) as client:
        client.post("/cycles", json={"profile": COTTON})
        assert client.post("/machine/pause").status_code == 501
        assert client.post("/simulator/fault", json={"fault": "bearing_wear"}).status_code == 501
        assert client.get("/diagnostics").json()["unavailable_reason"]
