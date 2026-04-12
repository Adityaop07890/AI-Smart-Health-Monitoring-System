"""
server/app.py — FastAPI application for AI Smart Health Monitoring System.
Implements the full OpenEnv spec AND serves the HTML frontend.
"""

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from env import HealthEnv  # noqa: E402

app = FastAPI(
    title="AI Smart Health Monitoring System",
    description="OpenEnv-compatible RL environment for health monitoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: HealthEnv = HealthEnv()
_initialized: bool = False
_UI_PATH = Path(__file__).parent.parent / "static" / "index.html"


class StepRequest(BaseModel):
    action: int


class GradeRequest(BaseModel):
    heart_rate: int
    temperature: float
    action: int


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the frontend dashboard (satisfies the OpenEnv 200 ping check)."""
    if _UI_PATH.exists():
        return HTMLResponse(_UI_PATH.read_text())
    return JSONResponse({"status": "ok", "name": "health-monitoring-env", "version": "1.0.0"})


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "ok", "name": "health-monitoring-env", "version": "1.0.0"}


@app.post("/reset")
def reset() -> Dict[str, Any]:
    global _env, _initialized
    _env = HealthEnv()
    state = _env.reset()
    _initialized = True
    return {"state": state}


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    if request.action not in (0, 1, 2):
        raise HTTPException(status_code=422, detail=f"Invalid action {request.action}.")
    next_state, reward, done, info = _env.step(request.action)
    return {"state": next_state, "reward": reward, "done": done, "info": info}


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return _env.state()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {"name": "easy_task",   "difficulty": "easy",   "grader_endpoint": "/grade/easy_task",
             "description": "Detect elevated heart rate (>100 bpm) and issue at least a warning.",
             "success_condition": "heart_rate > 100 AND action >= 1"},
            {"name": "medium_task", "difficulty": "medium", "grader_endpoint": "/grade/medium_task",
             "description": "Respond correctly using both HR and temperature signals.",
             "success_condition": "(heart_rate > 100 OR temperature > 38) AND action == 1"},
            {"name": "hard_task",   "difficulty": "hard",   "grader_endpoint": "/grade/hard_task",
             "description": "Emergency alert when HR>120 OR temp>39 (no partial credit).",
             "success_condition": "(heart_rate > 120 OR temperature > 39) AND action == 2"},
        ]
    }


def _compute_score(heart_rate: int, temperature: float, action: int) -> float:
    if heart_rate > 120 or temperature > 39.0:
        correct = 2
    elif heart_rate > 100 or temperature > 38.0:
        correct = 1
    else:
        correct = 0
    if action == correct:
        return 1.0
    elif abs(action - correct) == 1:
        return 0.5
    return 0.0


@app.post("/grade/easy_task")
def grade_easy(request: GradeRequest) -> Dict[str, Any]:
    if request.heart_rate > 100:
        score = 1.0 if request.action >= 1 else 0.0
    else:
        score = 1.0 if request.action == 0 else 0.5
    return {"task": "easy_task", "score": round(score, 4), "passed": score >= 0.5,
            "vitals": {"heart_rate": request.heart_rate, "temperature": request.temperature},
            "action": request.action}


@app.post("/grade/medium_task")
def grade_medium(request: GradeRequest) -> Dict[str, Any]:
    score = _compute_score(request.heart_rate, request.temperature, request.action)
    return {"task": "medium_task", "score": round(score, 4), "passed": score >= 0.5,
            "vitals": {"heart_rate": request.heart_rate, "temperature": request.temperature},
            "action": request.action}


@app.post("/grade/hard_task")
def grade_hard(request: GradeRequest) -> Dict[str, Any]:
    if request.heart_rate > 120 or request.temperature > 39.0:
        score = 1.0 if request.action == 2 else 0.0
    elif request.heart_rate > 100 or request.temperature > 38.0:
        score = 1.0 if request.action == 1 else (0.5 if request.action == 2 else 0.0)
    else:
        score = 1.0 if request.action == 0 else 0.5
    return {"task": "hard_task", "score": round(score, 4), "passed": score >= 0.5,
            "vitals": {"heart_rate": request.heart_rate, "temperature": request.temperature},
            "action": request.action}


def main():
    """Entry point for running the server directly (multi-mode deployment)."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
