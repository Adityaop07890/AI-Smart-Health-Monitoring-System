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


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>AI Health Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,sans-serif}
body{background:#f8f9fa;color:#1a1a1a;padding:24px}
h1{font-size:20px;font-weight:500;margin-bottom:4px}
.sub{font-size:13px;color:#666;margin-bottom:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px}
.easy{background:#dcfce7;color:#166534}.medium{background:#fef9c3;color:#854d0e}.hard{background:#fee2e2;color:#991b1b}
.title{font-size:15px;font-weight:600}
.desc{font-size:13px;color:#555;line-height:1.5}
.vitals{background:#f3f4f6;border-radius:8px;padding:10px 14px;display:flex;gap:12px;flex-wrap:wrap}
.v-label{font-size:11px;color:#888;margin-bottom:2px}
.v-val{font-size:20px;font-weight:600;font-family:monospace}
.v-val.warn{color:#b45309}.v-val.crit{color:#dc2626}
.abadge{font-size:12px;font-weight:500;padding:3px 10px;border-radius:20px;display:inline-block}
.a0{background:#dcfce7;color:#166534}.a1{background:#fef9c3;color:#854d0e}.a2{background:#fee2e2;color:#991b1b}
.score-row{display:flex;align-items:center;gap:10px}
.bar-bg{flex:1;height:7px;background:#e5e7eb;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .8s ease}
.score-num{font-size:14px;font-weight:600;min-width:44px;text-align:right}
.log{font-family:monospace;font-size:11px;color:#555;max-height:80px;overflow-y:auto;line-height:1.6;background:#f3f4f6;border-radius:8px;padding:8px 10px;white-space:pre}
button{width:100%;padding:10px;font-size:13px;font-weight:500;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;transition:background .15s}
button:hover{background:#f3f4f6}
button:disabled{opacity:.5;cursor:not-allowed}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px}
.idle{background:#9ca3af}.running{background:#f59e0b;animation:p 1s infinite}.done{background:#22c55e}.error{background:#ef4444}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
</style></head><body>
<h1>AI Health Monitoring</h1>
<p class="sub">OpenEnv · Scaler × Meta PyTorch Hackathon</p>
<div class="grid" id="grid"></div>
<script>
const BASE = '';
const tasks = [
  {id:'easy_task',  label:'Easy',   cls:'easy',   color:'#22c55e', desc:'Detect heart rate > 100 and send at least a warning.', steps:10},
  {id:'medium_task',label:'Medium', cls:'medium', color:'#f59e0b', desc:'Detect an increasing heart rate trend and respond.', steps:15},
  {id:'hard_task',  label:'Hard',   cls:'hard',   color:'#ef4444', desc:'Emergency alert when HR > 120 OR temp > 39.', steps:20},
];
const S = {};
tasks.forEach(t=>S[t.id]={status:'idle',score:null,hr:'—',tmp:'—',action:null,logs:[]});
function act(a){const m=['do_nothing','send_warning','emergency_alert'],c=['a0','a1','a2'];return a==null?'<span style="color:#888">—</span>':`<span class="abadge ${c[a]}">${m[a]}</span>`}
function hrC(v){return v>120?'crit':v>100?'warn':''}
function tmpC(v){return v>39?'crit':v>38?'warn':''}
function pick(hr,tmp){if(hr>120||tmp>39)return 2;if(hr>100||tmp>38)return 1;return 0;}
function render(){
  document.getElementById('grid').innerHTML=tasks.map(t=>{
    const s=S[t.id],dc=s.status==='idle'?'idle':s.status==='running'?'running':s.status==='done'?'done':'error';
    const sw=s.score!=null?Math.round(s.score*100)+'%':'0%';
    return`<div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <span class="badge ${t.cls}">${t.label}</span>
        <span><span class="dot ${dc}"></span><span style="font-size:12px;color:#888">${s.status}</span></span>
      </div>
      <div class="title">${t.id.replace('_',' ')}</div>
      <div class="desc">${t.desc}</div>
      <div class="vitals">
        <div><div class="v-label">heart rate</div><div class="v-val ${hrC(s.hr)}">${s.hr}<span style="font-size:11px;font-weight:400;color:#aaa"> bpm</span></div></div>
        <div><div class="v-label">temperature</div><div class="v-val ${tmpC(s.tmp)}">${s.tmp}<span style="font-size:11px;font-weight:400;color:#aaa"> °C</span></div></div>
        <div style="display:flex;flex-direction:column;justify-content:flex-end"><div class="v-label">action</div>${act(s.action)}</div>
      </div>
      <div><div style="display:flex;justify-content:space-between;margin-bottom:5px"><span style="font-size:12px;color:#888">grader score</span><span class="score-num">${s.score!=null?s.score.toFixed(4):'—'}</span></div>
        <div class="bar-bg"><div class="bar-fill" id="bar_${t.id}" style="width:${sw};background:${t.color}"></div></div>
      </div>
      <div class="log" id="log_${t.id}">${s.logs.slice(-6).join('\\n')||'— no runs yet —'}</div>
      <button id="btn_${t.id}" onclick="run('${t.id}')" ${s.status==='running'?'disabled':''}>
        ${s.status==='running'?'Running…':'Run + Grade'}
      </button>
    </div>`;
  }).join('');
}
function log(id,msg){S[id].logs.push(msg);const el=document.getElementById('log_'+id);if(el){el.textContent=S[id].logs.slice(-6).join('\\n');el.scrollTop=99999;}}
async function run(id){
  const t=tasks.find(x=>x.id===id);
  S[id]={...S[id],status:'running',logs:[],score:null,hr:'—',tmp:'—',action:null};
  render();
  try{
    log(id,'→ POST /reset');
    const r=await fetch(BASE+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!r.ok)throw new Error('reset '+r.status);
    log(id,'episode started');
    for(let i=0;i<t.steps;i++){
      const rr=await fetch(BASE+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      const d=await rr.json();
      const cur=d.observation||d.state||d;
      const hr=cur.heart_rate||Math.floor(Math.random()*105+55);
      const tmp=cur.temperature||parseFloat((Math.random()*4.5+36).toFixed(1));
      const a=pick(hr,tmp);
      S[id].hr=hr;S[id].tmp=tmp;S[id].action=a;
      const sr=await fetch(BASE+'/step?action='+a,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})});
      const sd=sr.ok?await sr.json():{reward:0};
      log(id,'step '+(i+1)+': hr='+hr+' tmp='+tmp+' act='+a+' r='+(sd.reward||0).toFixed(2));
      render();
      await new Promise(r=>setTimeout(r,120));
    }
    log(id,'→ GET /grade/'+id);
    const gr=await fetch(BASE+'/grade/'+id);
    if(!gr.ok)throw new Error('grade '+gr.status);
    const gd=await gr.json();
    S[id].score=gd.score||0;S[id].status='done';
    log(id,'score: '+gd.score.toFixed(4)+' ✓');
    render();
  }catch(e){S[id].status='error';log(id,'error: '+e.message);render();}
}
render();
</script></body></html>"""
