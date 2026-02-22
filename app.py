from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tronk_engine import (
    BOARD_RADIUS,
    DEFAULT_BOT,
    GameConfig,
    TronkSimulation,
    compare_python_vs_c,
    iter_board_tiles,
)
from tronk_ml import MLSessionStore


class SimulateRequest(BaseModel):
    bots: List[str] = Field(..., min_length=6, max_length=6)
    seed: int = 1
    max_steps: int = Field(default=300, ge=1, le=5000)
    engine: str = Field(default="c")


class SimulateResponse(BaseModel):
    winner_id: Optional[int]
    total_ticks: int
    steps_played: int
    board_radius: int
    move_interval: int
    engine_used: str
    frames: list
    logs: dict
    errors: dict


class CompareResponse(BaseModel):
    match: bool
    python: dict
    c: dict
    first_diff: Optional[dict]


class MLNewRequest(BaseModel):
    seed: int = 1
    max_steps: int = Field(default=300, ge=1, le=10000)


class MLStepRequest(BaseModel):
    session_id: str
    actions: List[Any] = Field(..., min_length=6, max_length=6)


class MLCallRequest(BaseModel):
    session_id: str
    player_id: int = Field(..., ge=0, le=5)
    function: str
    args: List[int] = Field(default_factory=list)


class MLSaveRequest(BaseModel):
    session_id: str
    name: str = "ml_run"


class MLLoadRequest(BaseModel):
    filename: str


app = FastAPI(title="Tronk Engine Replica", version="0.1.0")
ML_STORE = MLSessionStore(Path(__file__).parent / "ml_runs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/default-bot")
def default_bot() -> dict:
    return {"code": DEFAULT_BOT}


@app.get("/api/board")
def board() -> dict:
    return {
        "radius": BOARD_RADIUS,
        "tiles": [[q, r] for q, r in iter_board_tiles(BOARD_RADIUS)],
    }


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    if len(req.bots) != 6:
        raise HTTPException(status_code=400, detail="Exactly 6 bots are required")

    engine = req.engine.lower().strip()
    use_c_core = engine != "python"

    config = GameConfig(max_steps=req.max_steps, seed=req.seed, use_c_core=use_c_core)
    sim = TronkSimulation(req.bots, config=config)
    result = sim.run()

    return SimulateResponse(
        winner_id=result.winner_id,
        total_ticks=result.total_ticks,
        steps_played=result.steps_played,
        board_radius=BOARD_RADIUS,
        move_interval=config.move_interval,
        engine_used=sim.core_mode,
        frames=result.frames,
        logs=result.logs,
        errors=result.errors,
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: SimulateRequest) -> CompareResponse:
    if len(req.bots) != 6:
        raise HTTPException(status_code=400, detail="Exactly 6 bots are required")

    config = GameConfig(max_steps=req.max_steps, seed=req.seed, use_c_core=True)
    report = compare_python_vs_c(req.bots, config=config)
    return CompareResponse(**report)


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ml")
def ml_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "ml.html")


@app.post("/api/ml/new")
def ml_new(req: MLNewRequest) -> dict:
    session = ML_STORE.new_session(seed=req.seed, max_steps=req.max_steps)
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "snapshot": session.env.snapshot(),
    }


@app.get("/api/ml/state/{session_id}")
def ml_state(session_id: str) -> dict:
    try:
        session = ML_STORE.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "snapshot": session.env.snapshot(),
    }


@app.post("/api/ml/step")
def ml_step(req: MLStepRequest) -> dict:
    try:
        session = ML_STORE.get_session(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        snapshot = session.env.step(req.actions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "session_id": session.session_id,
        "snapshot": snapshot,
    }


@app.post("/api/ml/call")
def ml_call(req: MLCallRequest) -> dict:
    try:
        session = ML_STORE.get_session(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        value = session.env.call(req.player_id, req.function, req.args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "session_id": session.session_id,
        "player_id": req.player_id,
        "function": req.function,
        "args": req.args,
        "value": value,
    }


@app.post("/api/ml/save")
def ml_save(req: MLSaveRequest) -> dict:
    try:
        path = ML_STORE.save_session(req.session_id, req.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "ok": True,
        "filename": path.name,
    }


@app.get("/api/ml/saved")
def ml_saved() -> dict:
    return {
        "files": ML_STORE.list_saved(),
    }


@app.post("/api/ml/load")
def ml_load(req: MLLoadRequest) -> dict:
    try:
        payload = ML_STORE.load_saved(req.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return payload
