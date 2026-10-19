"""REST API for SmartWash. Interactive documentation is served at ``/docs``."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from . import __version__
from .config import Settings
from .controller import (
    CycleRecord,
    DiagnosticsReport,
    FaultInjection,
    FeedbackRequest,
    MachineSnapshot,
    RetrainInProgress,
    StartCycleRequest,
    StartCycleResult,
    WasherController,
    build_controller,
)
from .devices.base import DeviceAuthError, DeviceError, DeviceNotReady, UnsupportedOperation
from .schemas import LoadProfile, MachineState, Recommendation

DESCRIPTION = """
AI brain for your smart washing machine.

* **POST /recommend** - describe a load, get the programme, temperature, spin, options and
  detergent dose, with an explanation and time/energy estimate.
* **POST /cycles** - start a cycle from a load description (the AI chooses) or your own settings
  (checked against the load's care limits).
* **GET /machine**, **GET /diagnostics** - live status and AI fault detection.
* **POST /feedback**, **POST /model/retrain** - teach it your household's preferences.
"""


def create_app(
    settings: Settings | None = None, *, controller: WasherController | None = None
) -> FastAPI:
    """Build the API. Without ``controller``, one is built from ``settings`` at startup."""
    settings = settings or Settings()
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ctl = controller or build_controller(settings)
        app.state.controller = ctl
        if settings.monitoring.enabled:
            ctl.start_monitoring()
        try:
            yield
        finally:
            if controller is None:
                ctl.close()
            else:
                ctl.stop_monitoring()

    app = FastAPI(
        title="SmartWash", version=__version__, description=DESCRIPTION, lifespan=lifespan
    )

    def require_api_key(key: str | None = Security(api_key_header)) -> None:
        expected = settings.api.api_key
        if expected and not (key and secrets.compare_digest(key, expected)):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")

    def get_controller(request: Request) -> WasherController:
        return request.app.state.controller

    def error(status: int) -> Any:
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc)})

        return handler

    app.add_exception_handler(DeviceNotReady, error(409))
    app.add_exception_handler(RetrainInProgress, error(409))
    app.add_exception_handler(UnsupportedOperation, error(501))
    app.add_exception_handler(DeviceAuthError, error(502))
    app.add_exception_handler(DeviceError, error(502))

    @app.get("/health", tags=["system"])
    def health(request: Request) -> dict[str, Any]:
        ctl = get_controller(request)
        return {
            "status": "ok",
            "version": __version__,
            "adapter": ctl.adapter.name,
            "monitoring": ctl.is_monitoring,
            "fault_detection": ctl.fault_detection,
        }

    router = APIRouter(dependencies=[Depends(require_api_key)])
    ctl_dep = Depends(get_controller)

    @router.get("/machine", response_model=MachineSnapshot, tags=["machine"])
    def machine(ctl: WasherController = ctl_dep) -> MachineSnapshot:
        return ctl.snapshot()

    @router.post("/machine/pause", response_model=MachineState, tags=["machine"])
    def pause(ctl: WasherController = ctl_dep) -> MachineState:
        return ctl.pause()

    @router.post("/machine/resume", response_model=MachineState, tags=["machine"])
    def resume(ctl: WasherController = ctl_dep) -> MachineState:
        return ctl.resume()

    @router.post("/machine/stop", response_model=MachineState, tags=["machine"])
    def stop(ctl: WasherController = ctl_dep) -> MachineState:
        return ctl.stop()

    @router.post("/recommend", response_model=Recommendation, tags=["ai"])
    def recommend(profile: LoadProfile, ctl: WasherController = ctl_dep) -> Recommendation:
        return ctl.recommend(profile)

    @router.post("/cycles", response_model=StartCycleResult, status_code=201, tags=["machine"])
    def start_cycle(
        request: StartCycleRequest, ctl: WasherController = ctl_dep
    ) -> StartCycleResult:
        return ctl.start_cycle(request)

    @router.get("/cycles", response_model=list[CycleRecord], tags=["machine"])
    def cycle_history(
        limit: int = Query(default=20, ge=1, le=500), ctl: WasherController = ctl_dep
    ) -> list[CycleRecord]:
        return ctl.history.recent(limit)

    @router.get("/diagnostics", response_model=DiagnosticsReport, tags=["ai"])
    def diagnostics(ctl: WasherController = ctl_dep) -> DiagnosticsReport:
        return ctl.diagnostics()

    @router.post("/feedback", status_code=201, tags=["ai"])
    def feedback(request: FeedbackRequest, ctl: WasherController = ctl_dep) -> dict[str, Any]:
        return {"stored": True, "feedback_records": ctl.add_feedback(request)}

    @router.get("/model", tags=["ai"])
    def model(ctl: WasherController = ctl_dep) -> dict[str, Any]:
        return ctl.model_info()

    @router.post("/model/retrain", tags=["ai"])
    def retrain(ctl: WasherController = ctl_dep) -> dict[str, Any]:
        return ctl.retrain()

    @router.post("/simulator/fault", tags=["simulator"])
    def inject_fault(request: FaultInjection, ctl: WasherController = ctl_dep) -> dict[str, Any]:
        return ctl.inject_fault(request.fault, request.severity)

    app.include_router(router)
    return app
