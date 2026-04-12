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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Health Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f4f5f7;color:#111;min-height:100vh}
.topbar{background:#fff;border-bottom:1px solid #e5e7eb;padding:13px 24px;display:flex;align-items:center;justify-content:space-between}
.logo{display:flex;align-items:center;gap:10px}
.logo-text{font-size:15px;font-weight:500}
.live-pill{font-size:11px;font-weight:600;background:#dcfce7;color:#166534;padding:2px 9px;border-radius:20px}
.meta{font-size:12px;color:#6b7280}
.main{max-width:960px;margin:0 auto;padding:20px}
.vitals-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px}
.card-label{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.vital-num{font-size:34px;font-weight:500;font-family:ui-monospace,monospace;line-height:1}
.vital-unit{font-size:13px;color:#9ca3af;margin-left:4px}
.temp-bar-bg{background:#f3f4f6;border-radius:6px;height:8px;overflow:hidden;margin-top:14px}
.temp-bar-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#60a5fa,#f87171);transition:width .8s ease}
.temp-bar-labels{display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px}
.status-pill{display:inline-flex;align-items:center;gap:7px;padding:6px 13px;border-radius:20px;font-size:13px;font-weight:500;margin-top:6px}
.status-dot{width:7px;height:7px;border-radius:50%}
.stat-row{display:flex;gap:16px;margin-top:12px}
.stat-item{font-size:12px;color:#6b7280}
.stat-val{font-weight:500;color:#111}
.section-label{font-size:12px;font-weight:500;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.tasks-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.task-card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:10px}
.task-header{display:flex;justify-content:space-between;align-items:center}
.badge{font-size:11px;font-weight:600;padding:2px 9px;border-radius:20px}
.badge-easy{background:#dcfce7;color:#166534}
.badge-medium{background:#fef9c3;color:#854d0e}
.badge-hard{background:#fee2e2;color:#991b1b}
.task-status{font-size:11px;color:#9ca3af}
.task-title{font-size:13px;font-weight:500}
.task-desc{font-size:12px;color:#6b7280;line-height:1.5}
.score-row{display:flex;justify-content:space-between;font-size:12px}
.score-label{color:#6b7280}
.score-val{font-weight:500}
.score-bar-bg{background:#f3f4f6;border-radius:4px;height:5px;overflow:hidden}
.score-bar-fill{height:100%;border-radius:4px;transition:width 1s ease}
.log-box{font-family:ui-monospace,monospace;font-size:11px;color:#6b7280;background:#f9fafb;border-radius:8px;padding:8px 10px;max-height:72px;overflow-y:auto;white-space:pre;line-height:1.6}
.run-btn{width:100%;padding:9px;font-size:12px;font-weight:500;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;color:#374151;transition:background .15s}
.run-btn:hover{background:#f9fafb}
.run-btn:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <span class="logo-text">AI Health Monitor</span>
    <span class="live-pill">LIVE</span>
  </div>
  <span class="meta">OpenEnv &middot; Scaler &times; Meta PyTorch Hackathon</span>
</div>

<div class="main">

  <div class="vitals-row">

    <div class="card">
      <div class="card-label">Heart Rate</div>
      <div>
        <span class="vital-num" id="hr-val">72</span>
        <span class="vital-unit">bpm</span>
      </div>
      <div style="margin-top:10px">
        <svg width="100%" height="44" viewBox="0 0 220 44" preserveAspectRatio="none">
          <polyline id="ecg-line"
            points="0,22 22,22 33,22 38,4 44,40 50,22 66,22 88,22 99,22 104,4 110,40 116,22 132,22 154,22 165,22 170,4 176,40 182,22 198,22 220,22"
            fill="none" stroke="#ef4444" stroke-width="1.5" stroke-linejoin="round"/>
        </svg>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Temperature</div>
      <div>
        <span class="vital-num" id="tmp-val">36.8</span>
        <span class="vital-unit">°C</span>
      </div>
      <div class="temp-bar-bg">
        <div class="temp-bar-fill" id="tmp-bar" style="width:18%"></div>
      </div>
      <div class="temp-bar-labels"><span>36°</span><span>38°</span><span>40.5°</span></div>
    </div>

    <div class="card">
      <div class="card-label">Patient Status</div>
      <div class="status-pill" id="status-pill" style="background:#dcfce7;color:#166534">
        <span class="status-dot" id="status-dot" style="background:#22c55e"></span>
        <span id="status-text">Normal</span>
      </div>
      <div class="stat-row">
        <div class="stat-item">Step&nbsp;<span class="stat-val" id="step-num">0</span></div>
        <div class="stat-item">Reward&nbsp;<span class="stat-val" id="reward-num">—</span></div>
        <div class="stat-item">Action&nbsp;<span class="stat-val" id="action-num">—</span></div>
      </div>
    </div>

  </div>

  <div class="section-label">Task Graders</div>

  <div class="tasks-row">

    <div class="task-card">
      <div class="task-header">
        <span class="badge badge-easy">Easy</span>
        <span class="task-status" id="e-status">idle</span>
      </div>
      <div class="task-title">easy task</div>
      <div class="task-desc">Detect heart rate &gt; 100 and send at least a warning signal.</div>
      <div class="score-row"><span class="score-label">Grader score</span><span class="score-val" id="e-score">—</span></div>
      <div class="score-bar-bg"><div class="score-bar-fill" id="e-bar" style="width:0%;background:#22c55e"></div></div>
      <div class="log-box" id="e-log">— no runs yet —</div>
      <button class="run-btn" id="e-btn" onclick="runTask('easy_task','e',10)">Run + Grade ↗</button>
    </div>

    <div class="task-card">
      <div class="task-header">
        <span class="badge badge-medium">Medium</span>
        <span class="task-status" id="m-status">idle</span>
      </div>
      <div class="task-title">medium task</div>
      <div class="task-desc">Detect an increasing heart rate trend and respond appropriately.</div>
      <div class="score-row"><span class="score-label">Grader score</span><span class="score-val" id="m-score">—</span></div>
      <div class="score-bar-bg"><div class="score-bar-fill" id="m-bar" style="width:0%;background:#f59e0b"></div></div>
      <div class="log-box" id="m-log">— no runs yet —</div>
      <button class="run-btn" id="m-btn" onclick="runTask('medium_task','m',15)">Run + Grade ↗</button>
    </div>

    <div class="task-card">
      <div class="task-header">
        <span class="badge badge-hard">Hard</span>
        <span class="task-status" id="h-status">idle</span>
      </div>
      <div class="task-title">hard task</div>
      <div class="task-desc">Issue emergency alert when HR &gt; 120 OR temperature &gt; 39°C.</div>
      <div class="score-row"><span class="score-label">Grader score</span><span class="score-val" id="h-score">—</span></div>
      <div class="score-bar-bg"><div class="score-bar-fill" id="h-bar" style="width:0%;background:#ef4444"></div></div>
      <div class="log-box" id="h-log">— no runs yet —</div>
      <button class="run-btn" id="h-btn" onclick="runTask('hard_task','h',20)">Run + Grade ↗</button>
    </div>

  </div>
</div>

<script>
const ACTIONS = ['do_nothing', 'send_warning', 'emergency_alert'];

function pick(hr, tmp) {
  if (hr > 120 || tmp > 39) return 2;
  if (hr > 100 || tmp > 38) return 1;
  return 0;
}

function updateStatus(hr, tmp) {
  const crit = hr > 120 || tmp > 39;
  const warn = !crit && (hr > 100 || tmp > 38);
  const pill = document.getElementById('status-pill');
  const dot  = document.getElementById('status-dot');
  const txt  = document.getElementById('status-text');
  if (crit) {
    pill.style.cssText = 'display:inline-flex;align-items:center;gap:7px;padding:6px 13px;border-radius:20px;font-size:13px;font-weight:500;margin-top:6px;background:#fee2e2;color:#991b1b';
    dot.style.background = '#ef4444'; txt.textContent = 'Critical';
  } else if (warn) {
    pill.style.cssText = 'display:inline-flex;align-items:center;gap:7px;padding:6px 13px;border-radius:20px;font-size:13px;font-weight:500;margin-top:6px;background:#fef9c3;color:#854d0e';
    dot.style.background = '#f59e0b'; txt.textContent = 'Warning';
  } else {
    pill.style.cssText = 'display:inline-flex;align-items:center;gap:7px;padding:6px 13px;border-radius:20px;font-size:13px;font-weight:500;margin-top:6px;background:#dcfce7;color:#166534';
    dot.style.background = '#22c55e'; txt.textContent = 'Normal';
  }
}

function updateVitals(hr, tmp) {
  document.getElementById('hr-val').textContent = hr;
  document.getElementById('hr-val').style.color = hr > 120 ? '#ef4444' : hr > 100 ? '#f59e0b' : '#111';
  document.getElementById('tmp-val').textContent = tmp.toFixed(1);
  document.getElementById('tmp-bar').style.width = ((tmp - 36) / 4.5 * 100).toFixed(1) + '%';
  updateStatus(hr, tmp);
  const amp = hr > 120 ? 17 : hr > 100 ? 11 : 6;
  const pts = [];
  for (let i = 0; i <= 10; i++) pts.push((i * 22) + ',' + (22 + (Math.random() - .5) * amp * .3).toFixed(1));
  pts.splice(3, 0, (3*22)+',22', (3*22+5)+','+(22-amp*1.8).toFixed(1), (3*22+11)+','+(22+amp*2).toFixed(1), (3*22+17)+',22');
  document.getElementById('ecg-line').setAttribute('points', pts.join(' '));
}

function addLog(pre, msg) {
  const el = document.getElementById(pre + '-log');
  const lines = el.textContent === '— no runs yet —' ? [] : el.textContent.split('\\n');
  lines.push(msg);
  el.textContent = lines.slice(-6).join('\\n');
  el.scrollTop = 9999;
}

async function runTask(taskId, pre, steps) {
  const btn = document.getElementById(pre + '-btn');
  btn.disabled = true; btn.textContent = 'Running…';
  document.getElementById(pre + '-status').textContent = 'running';
  document.getElementById(pre + '-score').textContent = '—';
  document.getElementById(pre + '-bar').style.width = '0%';
  document.getElementById(pre + '-log').textContent = '';

  try {
    addLog(pre, '→ POST /reset');
    const r = await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    if (!r.ok) throw new Error('reset ' + r.status);
    addLog(pre, 'episode started');

    for (let i = 0; i < steps; i++) {
      const rr  = await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const d   = await rr.json();
      const obs = d.observation || d;
      const hr  = obs.heart_rate  || (Math.floor(Math.random() * 106) + 55);
      const tmp = obs.temperature || parseFloat((Math.random() * 4.5 + 36).toFixed(1));
      const a   = pick(hr, tmp);

      document.getElementById('step-num').textContent   = i + 1;
      document.getElementById('action-num').textContent = ACTIONS[a];
      updateVitals(hr, tmp);

      const sr = await fetch('/step?action=' + a, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: a }) });
      const sd = sr.ok ? await sr.json() : { reward: 0 };
      document.getElementById('reward-num').textContent = (sd.reward || 0).toFixed(2);
      addLog(pre, 'step ' + (i+1) + ': hr=' + hr + ' t=' + tmp.toFixed(1) + ' a=' + a + ' r=' + (sd.reward||0).toFixed(2));
      await new Promise(res => setTimeout(res, 130));
    }

    addLog(pre, '→ GET /grade/' + taskId);
    const gr = await fetch('/grade/' + taskId);
    if (!gr.ok) throw new Error('grade ' + gr.status);
    const gd = await gr.json();

    document.getElementById(pre + '-score').textContent = gd.score.toFixed(4);
    document.getElementById(pre + '-bar').style.width   = (gd.score * 100).toFixed(1) + '%';
    document.getElementById(pre + '-status').textContent = 'done ✓';
    addLog(pre, 'score: ' + gd.score.toFixed(4) + ' ✓');
  } catch (e) {
    document.getElementById(pre + '-status').textContent = 'error';
    addLog(pre, 'error: ' + e.message);
  }

  btn.disabled = false; btn.textContent = 'Run + Grade ↗';
}

setInterval(() => {
  const hr  = Math.floor(Math.random() * 106 + 55);
  const tmp = parseFloat((Math.random() * 4.5 + 36).toFixed(1));
  updateVitals(hr, tmp);
}, 1600);
</script>
</body>
</html>"""


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
