import random
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import HealthEnv

app = FastAPI(title="AI Health Monitoring")

_env = HealthEnv()


def _clamp(score: float) -> float:
    return round(max(0.01, min(0.99, score)), 4)


def _optimal_action(hr: int, tmp: float) -> int:
    if hr > 120 or tmp > 39.0:
        return 2
    elif hr > 100 or tmp > 38.0:
        return 1
    return 0


@app.post("/reset")
def reset():
    state = _env.reset()
    return {"observation": state, "episode_id": _env._episode_id}


@app.post("/step")
def step(action: int):
    state, reward, done, info = _env.step(action)
    return {"observation": state, "reward": reward, "done": done,
            "terminated": done, "truncated": False, "info": info}


@app.get("/state")
def state():
    return _env.state()


@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"id": "easy_task",   "name": "easy_task",   "grader_endpoint": "/grade/easy_task",   "has_grader": True},
        {"id": "medium_task", "name": "medium_task", "grader_endpoint": "/grade/medium_task", "has_grader": True},
        {"id": "hard_task",   "name": "hard_task",   "grader_endpoint": "/grade/hard_task",   "has_grader": True},
    ]}


@app.get("/grade/easy_task")
def grade_easy_task():
    local = HealthEnv()
    hits, total = 0, 50
    for _ in range(total):
        local.reset()
        hr = random.randint(101, 160)
        tmp = round(random.uniform(36.0, 38.9), 1)
        local._state = {"heart_rate": hr, "temperature": tmp}
        _, reward, _, _ = local.step(_optimal_action(hr, tmp))
        if reward >= 0.5:
            hits += 1
    return {"task": "easy_task", "score": _clamp(hits / total)}


@app.get("/grade/medium_task")
def grade_medium_task():
    local = HealthEnv()
    hits, total = 0, 50
    for _ in range(total):
        local.reset()
        base = random.randint(75, 95)
        ok = 0
        for i in range(5):
            hr = min(base + (i + 1) * random.randint(4, 10), 160)
            tmp = round(random.uniform(36.5, 38.5), 1)
            local._state = {"heart_rate": hr, "temperature": tmp}
            action = _optimal_action(hr, tmp)
            if i >= 2 and action == 0:
                action = 1
            _, reward, _, _ = local.step(action)
            if reward >= 0.5:
                ok += 1
        if ok >= 3:
            hits += 1
    return {"task": "medium_task", "score": _clamp(hits / total)}


@app.get("/grade/hard_task")
def grade_hard_task():
    local = HealthEnv()
    hits, total = 0, 50
    for _ in range(total):
        local.reset()
        if random.random() < 0.5:
            local._state = {"heart_rate": random.randint(121, 160),
                            "temperature": round(random.uniform(36.0, 38.9), 1)}
        else:
            local._state = {"heart_rate": random.randint(55, 119),
                            "temperature": round(random.uniform(39.1, 40.5), 1)}
        _, reward, _, _ = local.step(2)
        if reward >= 0.9:
            hits += 1
    return {"task": "hard_task", "score": _clamp(hits / total)}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return "<h1>AI Health Monitor</h1><p>Endpoints: /reset /step /state /tasks /grade/easy_task /grade/medium_task /grade/hard_task</p>"


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
