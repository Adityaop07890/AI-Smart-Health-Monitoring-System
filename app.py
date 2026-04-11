import sys
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from env import HealthEnv

app = FastAPI(title="AI Health Monitoring", version="1.0.0")
env = HealthEnv()

class StepRequest(BaseModel):
    action: int

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs, "episode_id": env.state()["episode_id"]}

@app.post("/step")
def step(body: StepRequest):
    if body.action not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="action must be 0, 1, or 2")
    obs, reward, done, info = env.step(body.action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"name": "easy_task", "grader_endpoint": "/grade/easy_task"},
        {"name": "medium_task", "grader_endpoint": "/grade/medium_task"},
        {"name": "hard_task", "grader_endpoint": "/grade/hard_task"},
    ]}

@app.get("/grade/easy_task")
def grade_easy():
    hr = env._state.get("heart_rate", 0)
    return {"task": "easy_task", "passed": hr > 100, "heart_rate": hr}

@app.get("/grade/medium_task")
def grade_medium():
    hrs = env._heart_rates
    passed = len(hrs) >= 2 and hrs[-1] > hrs[0]
    return {"task": "medium_task", "passed": passed, "heart_rates": hrs}

@app.get("/grade/hard_task")
def grade_hard():
    hr = env._state.get("heart_rate", 0)
    temp = env._state.get("temperature", 0.0)
    return {"task": "hard_task", "passed": hr > 120 or temp > 39.0, "heart_rate": hr, "temperature": temp}

@app.get("/")
def root():
    return {"status": "ok", "service": "AI Health Monitoring"}