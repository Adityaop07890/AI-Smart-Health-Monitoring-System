import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

episode_state = {
    "heart_rate": 75,
    "temperature": 37.0,
    "step_count": 0,
    "total_reward": 0.0,
    "done": False,
    "heart_rate_history": []
}

def random_vitals():
    return {
        "heart_rate": random.randint(55, 160),
        "temperature": round(random.uniform(36.0, 40.5), 1)
    }

def compute_reward(heart_rate: int, temperature: float, action: int) -> float:
    if heart_rate > 120 or temperature > 39.0:
        correct = 2
    elif heart_rate > 100 or temperature > 38.0:
        correct = 1
    else:
        correct = 0

    if action == correct:
        return 0.9       
    elif abs(action - correct) == 1:
        return 0.5
    else:
        return 0.1       

def compute_task_score() -> dict:
    history = episode_state["heart_rate_history"]
    steps = episode_state["step_count"]
    if steps == 0:
        return {"easy": 0.5, "medium": 0.5, "hard": 0.5}

    avg_reward = episode_state["total_reward"] / max(steps, 1)
    easy_score = min(0.89, max(0.11, round(avg_reward * 0.9, 4)))

    # Medium: reward for detecting increasing trend
    if len(history) >= 3:
        increasing = sum(1 for i in range(1, len(history)) if history[i] > history[i-1])
        medium_score = min(0.89, max(0.11, round(increasing / max(len(history)-1, 1) * 0.85, 4)))
    else:
        medium_score = 0.5

    hard_score = min(0.89, max(0.11, round(avg_reward * 0.85, 4)))

    return {"easy": easy_score, "medium": medium_score, "hard": hard_score}


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Smart Health Monitoring System"}


@app.post("/reset")
def reset():
    vitals = random_vitals()
    episode_state.update({
        "heart_rate": vitals["heart_rate"],
        "temperature": vitals["temperature"],
        "step_count": 0,
        "total_reward": 0.0,
        "done": False,
        "heart_rate_history": [vitals["heart_rate"]]
    })
    return {"observation": vitals, "info": {}}


class StepRequest(BaseModel):
    action: int

@app.post("/step")
def step(body: StepRequest):
    action = body.action
    hr = episode_state["heart_rate"]
    temp = episode_state["temperature"]

    reward = compute_reward(hr, temp, action)
    episode_state["total_reward"] += reward
    episode_state["step_count"] += 1

    vitals = random_vitals()
    episode_state["heart_rate"] = vitals["heart_rate"]
    episode_state["temperature"] = vitals["temperature"]
    episode_state["heart_rate_history"].append(vitals["heart_rate"])

    done = episode_state["step_count"] >= 20
    episode_state["done"] = done

    return {
        "observation": vitals,
        "reward": reward,
        "done": done,
        "info": {"step": episode_state["step_count"]}
    }


@app.get("/state")
def state():
    return {
        "heart_rate": episode_state["heart_rate"],
        "temperature": episode_state["temperature"],
        "step_count": episode_state["step_count"],
        "total_reward": round(episode_state["total_reward"], 4),
        "done": episode_state["done"]
    }


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy_task", "grader_endpoint": "/grade/easy_task"},
            {"name": "medium_task", "grader_endpoint": "/grade/medium_task"},
            {"name": "hard_task", "grader_endpoint": "/grade/hard_task"}
        ]
    }


@app.get("/grade/easy_task")
def grade_easy():
    scores = compute_task_score()
    return {"task": "easy_task", "score": scores["easy"]}

@app.get("/grade/medium_task")
def grade_medium():
    scores = compute_task_score()
    return {"task": "medium_task", "score": scores["medium"]}

@app.get("/grade/hard_task")
def grade_hard():
    scores = compute_task_score()
    return {"task": "hard_task", "score": scores["hard"]}
