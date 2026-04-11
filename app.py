import random
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

state = {
    "heart_rate": 75,
    "temperature": 37.0,
    "step_count": 0,
    "total_reward": 0.0,
    "done": False,
    "hr_history": []
}

def random_vitals():
    return {
        "heart_rate": random.randint(55, 160),
        "temperature": round(random.uniform(36.0, 40.5), 1)
    }

def compute_reward(hr, temp, action):
    if hr > 120 or temp > 39.0:
        correct = 2
    elif hr > 100 or temp > 38.0:
        correct = 1
    else:
        correct = 0
    if action == correct:
        return 0.9
    elif abs(action - correct) == 1:
        return 0.5
    else:
        return 0.1

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    v = random_vitals()
    state.update({
        "heart_rate": v["heart_rate"],
        "temperature": v["temperature"],
        "step_count": 0,
        "total_reward": 0.0,
        "done": False,
        "hr_history": [v["heart_rate"]]
    })
    return {"observation": v, "info": {}}

class StepRequest(BaseModel):
    action: int

@app.post("/step")
def step(body: StepRequest):
    r = compute_reward(state["heart_rate"], state["temperature"], body.action)
    state["total_reward"] += r
    state["step_count"] += 1
    v = random_vitals()
    state["heart_rate"] = v["heart_rate"]
    state["temperature"] = v["temperature"]
    state["hr_history"].append(v["heart_rate"])
    state["done"] = state["step_count"] >= 20
    return {"observation": v, "reward": r, "done": state["done"], "info": {}}

@app.get("/state")
def get_state():
    return {
        "heart_rate": state["heart_rate"],
        "temperature": state["temperature"],
        "step_count": state["step_count"],
        "total_reward": round(state["total_reward"], 4),
        "done": state["done"]
    }

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"name": "easy_task", "grader_endpoint": "/grade/easy_task"},
        {"name": "medium_task", "grader_endpoint": "/grade/medium_task"},
        {"name": "hard_task", "grader_endpoint": "/grade/hard_task"}
    ]}

def _score():
    steps = max(state["step_count"], 1)
    avg = state["total_reward"] / steps
    easy = round(min(0.89, max(0.11, avg * 0.9)), 4)
    hr_hist = state["hr_history"]
    if len(hr_hist) >= 2:
        inc = sum(1 for i in range(1, len(hr_hist)) if hr_hist[i] > hr_hist[i-1])
        medium = round(min(0.89, max(0.11, inc / (len(hr_hist)-1) * 0.85)), 4)
    else:
        medium = 0.5
    hard = round(min(0.89, max(0.11, avg * 0.85)), 4)
    return easy, medium, hard

@app.get("/grade/easy_task")
def grade_easy():
    e, _, _ = _score()
    return {"task": "easy_task", "score": e}

@app.get("/grade/medium_task")
def grade_medium():
    _, m, _ = _score()
    return {"task": "medium_task", "score": m}

@app.get("/grade/hard_task")
def grade_hard():
    _, _, h = _score()
    return {"task": "hard_task", "score": h}
