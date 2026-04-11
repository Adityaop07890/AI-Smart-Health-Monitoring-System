import random
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import HealthEnv

app = FastAPI(title="AI Health Monitoring")
_env = HealthEnv()


def _clamp(score: float) -> float:
    """Score must be strictly between 0 and 1 — never 0.0 or 1.0."""
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
    return {
        "observation": state,
        "reward": reward,
        "done": done,
        "terminated": done,
        "truncated": False,
        "info": info,
    }


@app.get("/state")
def state():
    return _env.state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy_task",   "name": "easy_task",   "grader_endpoint": "/grade/easy_task"},
            {"id": "medium_task", "name": "medium_task", "grader_endpoint": "/grade/medium_task"},
            {"id": "hard_task",   "name": "hard_task",   "grader_endpoint": "/grade/hard_task"},
        ]
    }


@app.get("/grade/easy_task")
def grade_easy_task():
    local = HealthEnv()
    hits, total = 0, 40
    for _ in range(total):
        local.reset()
        hr  = random.randint(101, 160)
        tmp = round(random.uniform(36.0, 38.9), 1)
        local._state = {"heart_rate": hr, "temperature": tmp}
        _, reward, _, _ = local.step(1 if hr <= 120 else 2)
        if reward >= 0.5:
            hits += 1
    return {"task": "easy_task", "score": _clamp(hits / total)}


@app.get("/grade/medium_task")
def grade_medium_task():
    local = HealthEnv()
    hits, total = 0, 40
    for _ in range(total):
        local.reset()
        base = random.randint(75, 95)
        ok = 0
        for i in range(5):
            hr  = min(base + (i + 1) * random.randint(4, 10), 160)
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
    hits, total = 0, 40
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
