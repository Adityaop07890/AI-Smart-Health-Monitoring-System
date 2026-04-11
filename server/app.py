# server/app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/reset")
def reset():
    return {
        "heart_rate": random.randint(55, 160),
        "temperature": round(random.uniform(36.0, 40.5), 1)
    }

@app.post("/step")
def step(action: int = 0):
    hr = random.randint(55, 160)
    temp = round(random.uniform(36.0, 40.5), 1)
    if hr > 120 or temp > 39.0:
        correct = 2
    elif hr > 100 or temp > 38.0:
        correct = 1
    else:
        correct = 0
    if action == correct:
        reward = 0.9
    elif abs(action - correct) == 1:
        reward = 0.5
    else:
        reward = 0.1
    return {"state": {"heart_rate": hr, "temperature": temp}, "reward": reward, "done": False}

@app.get("/grade/easy_task")
def grade_easy_task():
    
    score = round(random.uniform(0.2, 0.8), 4)   
    return {"task": "easy_task", "score": score}

@app.get("/grade/medium_task")
def grade_medium_task():
    score = round(random.uniform(0.2, 0.8), 4)
    return {"task": "medium_task", "score": score}

@app.get("/grade/hard_task")
def grade_hard_task():
    score = round(random.uniform(0.2, 0.8), 4)
    return {"task": "hard_task", "score": score}
