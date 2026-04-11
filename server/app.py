"""
FastAPI server for the AI Smart Health Monitoring System.
- Serves the frontend dashboard at GET /
- Provides environment API: /reset, /step, /state, /tasks
- Provides 3 grader endpoints with scores strictly in (0, 1)

Import fix: uses importlib to load env.py by absolute filesystem path,
so it works regardless of Python working directory or sys.path state.
"""
import importlib.util
import sys
from pathlib import Path

# ── Robust import of HealthEnv from sibling env.py ──────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / "env.py"
_spec = importlib.util.spec_from_file_location("health_env_module", _ENV_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["health_env_module"] = _mod
_spec.loader.exec_module(_mod)
HealthEnv = _mod.HealthEnv
# ────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="AI Smart Health Monitoring System",
    description="OpenEnv-compatible health monitoring environment with graders",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = HealthEnv()


# ── Frontend ─────────────────────────────────────────────────────────────────

_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend.html"

@app.get("/", response_class=HTMLResponse)
def frontend():
    """Serve the health monitoring dashboard."""
    if _FRONTEND_PATH.exists():
        return HTMLResponse(content=_FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
    <html><body style="background:#050a0f;color:#00e5ff;font-family:monospace;padding:40px">
    <h2>AI Smart Health Monitoring System</h2>
    <p>frontend.html not found. API is running at <a href="/docs">/docs</a></p>
    </body></html>
    """)


# ── Pydantic models ──────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action: int  # 0, 1, or 2


class GradeStep(BaseModel):
    heart_rate: int
    temperature: float
    action: int


class GradeRequest(BaseModel):
    steps: List[GradeStep]


# ── Environment endpoints ─────────────────────────────────────────────────────

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state, "episode_id": env._episode_id}


@app.post("/step")
def step(req: StepRequest):
    if req.action not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="action must be 0, 1, or 2")
    next_state, reward, done, info = env.step(req.action)
    return {"state": next_state, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "easy_task",
                "description": "Detect high heart rate (>100) and send at least a warning",
                "success_condition": "heart_rate > 100 AND action >= 1",
                "grader_endpoint": "/grade/easy_task",
            },
            {
                "name": "medium_task",
                "description": "Detect an increasing trend in heart rate and respond appropriately",
                "success_condition": "heart_rate increasing over time AND action >= 1",
                "grader_endpoint": "/grade/medium_task",
            },
            {
                "name": "hard_task",
                "description": "Issue emergency alert if heart_rate > 120 OR temperature > 39",
                "success_condition": "(heart_rate > 120 OR temperature > 39) AND action == 2",
                "grader_endpoint": "/grade/hard_task",
            },
        ]
    }


# ── Grader helpers ────────────────────────────────────────────────────────────

def _to_open(raw: float) -> float:
    """
    Map a ratio in [0, 1] to the OPEN interval (0, 1).
    Formula:  score = 0.05 + raw * 0.90
      raw=0.0 → 0.05  (strictly > 0)
      raw=1.0 → 0.95  (strictly < 1)
    """
    clamped = max(0.0, min(1.0, raw))
    return round(0.05 + clamped * 0.90, 6)


# ── Grader endpoints ──────────────────────────────────────────────────────────

@app.post("/grade/easy_task")
def grade_easy_task(req: GradeRequest):
    """Easy Task: heart_rate > 100 must trigger action >= 1."""
    if not req.steps:
        return {"task": "easy_task", "score": _to_open(0.0),
                "detail": "No steps provided — minimum score assigned"}

    high_hr = [s for s in req.steps if s.heart_rate > 100]
    if not high_hr:
        return {"task": "easy_task", "score": _to_open(0.5),
                "detail": "No high-HR steps in trajectory — neutral score"}

    correct = sum(1 for s in high_hr if s.action >= 1)
    raw = correct / len(high_hr)
    return {
        "task": "easy_task",
        "score": _to_open(raw),
        "detail": f"{correct}/{len(high_hr)} high-HR steps triggered a warning or alert",
    }


@app.post("/grade/medium_task")
def grade_medium_task(req: GradeRequest):
    """Medium Task: detect rising HR trend and respond with action >= 1."""
    if len(req.steps) < 2:
        return {"task": "medium_task", "score": _to_open(0.0),
                "detail": "Need at least 2 steps to detect a trend"}

    trend_total = 0
    trend_correct = 0
    for i in range(1, len(req.steps)):
        if req.steps[i].heart_rate > req.steps[i - 1].heart_rate:
            trend_total += 1
            if req.steps[i].action >= 1:
                trend_correct += 1

    if trend_total == 0:
        return {"task": "medium_task", "score": _to_open(0.5),
                "detail": "No increasing-trend steps found — neutral score"}

    raw = trend_correct / trend_total
    return {
        "task": "medium_task",
        "score": _to_open(raw),
        "detail": f"{trend_correct}/{trend_total} rising-trend steps responded correctly",
    }


@app.post("/grade/hard_task")
def grade_hard_task(req: GradeRequest):
    """Hard Task: heart_rate > 120 OR temperature > 39 must trigger action == 2."""
    if not req.steps:
        return {"task": "hard_task", "score": _to_open(0.0),
                "detail": "No steps provided — minimum score assigned"}

    critical = [s for s in req.steps if s.heart_rate > 120 or s.temperature > 39.0]
    if not critical:
        return {"task": "hard_task", "score": _to_open(0.5),
                "detail": "No critical steps in trajectory — neutral score"}

    correct = sum(1 for s in critical if s.action == 2)
    raw = correct / len(critical)
    return {
        "task": "hard_task",
        "score": _to_open(raw),
        "detail": f"{correct}/{len(critical)} critical steps issued emergency alert",
    }
