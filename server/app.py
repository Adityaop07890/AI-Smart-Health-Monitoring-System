import math
import random
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

# ── Helpers ────────────────────────────────────────────────────────────────────

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

def safe_score(x, lo=0.15, hi=0.85):
    """Return a score strictly inside (0, 1) — never 0.0 or 1.0."""
    try:
        x = float(x)
        if not math.isfinite(x):
            return 0.50
        return round(min(hi, max(lo, x)), 4)
    except Exception:
        return 0.50

# ── Frontend ───────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Health Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #060b10;
    --panel: #0c1520;
    --border: #1a3040;
    --accent: #00e5ff;
    --accent2: #ff3860;
    --warn: #ffb300;
    --green: #00e676;
    --text: #cde8f0;
    --muted: #4a7080;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Barlow', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,229,255,0.06) 0%, transparent 70%),
      repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,229,255,0.03) 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,229,255,0.03) 40px);
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px; border-bottom: 1px solid var(--border);
    background: rgba(6,11,16,0.9); backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-pulse {
    width: 34px; height: 34px; border-radius: 50%;
    border: 2px solid var(--accent); position: relative;
    display: flex; align-items: center; justify-content: center;
  }
  .logo-pulse::before {
    content: ''; position: absolute; width: 100%; height: 100%;
    border-radius: 50%; border: 2px solid var(--accent);
    animation: ping 1.8s ease-out infinite;
  }
  @keyframes ping { 0%{transform:scale(1);opacity:1} 100%{transform:scale(2.2);opacity:0} }
  .logo-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); }
  .logo h1 { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); }
  .logo p { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; }
  .status-bar { display: flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: blink 1.4s step-end infinite; }
  @keyframes blink { 50%{opacity:0} }
  main { padding: 28px 32px; max-width: 1300px; margin: 0 auto; }
  .vitals-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 22px; position: relative;
    overflow: hidden; transition: border-color 0.3s;
  }
  .card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0.5;
  }
  .card-label { font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; font-family: var(--mono); }
  .card-value { font-family: var(--mono); font-size: 2.4rem; color: var(--accent); line-height: 1; transition: color 0.3s; }
  .card-unit { font-size: 0.75rem; color: var(--muted); margin-top: 4px; font-family: var(--mono); }
  .card.danger .card-value { color: var(--accent2); }
  .card.danger { border-color: rgba(255,56,96,0.4); }
  .card.warn .card-value { color: var(--warn); }
  .card.warn { border-color: rgba(255,179,0,0.4); }
  .card.normal .card-value { color: var(--green); }
  #ecg-canvas { width: 100%; height: 60px; display: block; margin-top: 10px; }
  .main-grid { display: grid; grid-template-columns: 1fr 380px; gap: 20px; margin-bottom: 24px; }
  .chart-card { padding: 20px 22px; }
  .chart-card h2 { font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; font-family: var(--mono); }
  #hr-canvas { width: 100%; height: 200px; display: block; }
  .control-panel { display: flex; flex-direction: column; gap: 14px; }
  .section-title { font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); font-family: var(--mono); margin-bottom: 4px; }
  .action-btns { display: flex; flex-direction: column; gap: 10px; }
  .action-btn {
    display: flex; align-items: center; gap: 14px; background: transparent;
    border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px;
    cursor: pointer; color: var(--text); font-family: var(--sans);
    font-size: 0.88rem; font-weight: 600; transition: all 0.2s; text-align: left;
  }
  .action-btn:hover { border-color: var(--accent); background: rgba(0,229,255,0.05); }
  .action-btn.disabled-state { opacity: 0.35; cursor: not-allowed; pointer-events: none; }
  .btn-icon { width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }
  .btn-0 .btn-icon { background: rgba(0,230,118,0.12); }
  .btn-1 .btn-icon { background: rgba(255,179,0,0.12); }
  .btn-2 .btn-icon { background: rgba(255,56,96,0.12); }
  .btn-0:hover { border-color: var(--green); }
  .btn-1:hover { border-color: var(--warn); }
  .btn-2:hover { border-color: var(--accent2); }
  .btn-sub { font-size: 0.7rem; color: var(--muted); font-weight: 400; margin-top: 2px; }
  .ctrl-btns { display: flex; gap: 10px; }
  .ctrl-btn {
    flex: 1; padding: 11px; border-radius: 8px; border: 1px solid var(--border);
    background: transparent; color: var(--text); font-family: var(--sans);
    font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
  }
  .ctrl-btn.primary { border-color: var(--accent); color: var(--accent); }
  .ctrl-btn.primary:hover { background: rgba(0,229,255,0.1); }
  .ctrl-btn.secondary:hover { border-color: var(--muted); }
  .stats-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }
  .stat-mini { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .stat-mini .label { font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); font-family: var(--mono); }
  .stat-mini .val { font-family: var(--mono); font-size: 1.45rem; color: var(--accent); margin-top: 4px; }
  .score-panel { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 24px; }
  .score-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
  .score-card .task-name { font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); font-family: var(--mono); margin-bottom: 10px; }
  .score-bar-wrap { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; margin-bottom: 8px; }
  .score-bar { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--green)); transition: width 0.6s ease; width: 0%; }
  .score-val { font-family: var(--mono); font-size: 1.3rem; color: var(--accent); }
  .score-tag { font-size: 0.6rem; color: var(--muted); margin-top: 2px; font-family: var(--mono); }
  .log-card { padding: 16px 22px; }
  .log-card h2 { font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); font-family: var(--mono); margin-bottom: 12px; }
  #log { height: 130px; overflow-y: auto; font-family: var(--mono); font-size: 0.72rem; color: var(--muted); display: flex; flex-direction: column-reverse; gap: 4px; }
  .log-entry { padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
  .log-entry .ts { color: var(--muted); margin-right: 8px; }
  .log-entry .msg-ok { color: var(--green); }
  .log-entry .msg-warn { color: var(--warn); }
  .log-entry .msg-err { color: var(--accent2); }
  .log-entry .msg-info { color: var(--accent); }
  .reward-flash {
    position: fixed; top: 80px; right: 32px; background: var(--panel);
    border: 1px solid var(--accent); border-radius: 8px; padding: 10px 18px;
    font-family: var(--mono); font-size: 0.8rem; color: var(--accent);
    opacity: 0; transform: translateY(-8px); transition: opacity 0.2s, transform 0.2s;
    pointer-events: none; z-index: 999;
  }
  .reward-flash.show { opacity: 1; transform: translateY(0); }
  .done-banner {
    display: none; position: fixed; inset: 0;
    background: rgba(6,11,16,0.85); backdrop-filter: blur(8px);
    z-index: 200; align-items: center; justify-content: center;
    flex-direction: column; gap: 20px; text-align: center;
  }
  .done-banner.show { display: flex; }
  .done-banner h2 { font-size: 2rem; font-weight: 700; color: var(--accent); letter-spacing: 0.1em; }
  .done-score { font-family: var(--mono); font-size: 1.1rem; color: var(--green); }
  @media (max-width: 900px) {
    .vitals-row { grid-template-columns: 1fr 1fr; }
    .main-grid { grid-template-columns: 1fr; }
    .score-panel { grid-template-columns: 1fr; }
    main { padding: 16px; }
    header { padding: 14px 16px; }
  }
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-pulse"><div class="logo-dot"></div></div>
    <div><h1>BioPulse AI</h1><p>Health Monitoring System</p></div>
  </div>
  <div class="status-bar">
    <div class="status-dot"></div>
    <span id="status-text">STANDBY — PRESS RESET TO BEGIN</span>
  </div>
</header>
<main>
  <div class="vitals-row">
    <div class="card" id="hr-card" style="grid-column:span 2;">
      <div class="card-label">❤ Heart Rate</div>
      <div class="card-value" id="hr-val">—</div>
      <div class="card-unit">BPM</div>
      <canvas id="ecg-canvas" height="60"></canvas>
    </div>
    <div class="card" id="temp-card">
      <div class="card-label">🌡 Temperature</div>
      <div class="card-value" id="temp-val">—</div>
      <div class="card-unit">°C</div>
    </div>
    <div class="card" id="alert-card">
      <div class="card-label">⚡ Alert Level</div>
      <div class="card-value" id="alert-val" style="font-size:1.2rem;margin-top:6px;">WAITING</div>
      <div class="card-unit" id="alert-sub">Start episode to monitor</div>
    </div>
  </div>
  <div class="main-grid">
    <div>
      <div class="card chart-card" style="margin-bottom:16px;">
        <h2>Heart Rate History</h2>
        <canvas id="hr-canvas" height="200"></canvas>
      </div>
      <div class="stats-row">
        <div class="stat-mini"><div class="label">Step</div><div class="val" id="step-val">0 / 20</div></div>
        <div class="stat-mini"><div class="label">Total Reward</div><div class="val" id="reward-val">0.0000</div></div>
        <div class="stat-mini"><div class="label">Avg Reward</div><div class="val" id="avg-reward-val">—</div></div>
      </div>
    </div>
    <div class="card control-panel">
      <div class="section-title">Episode Control</div>
      <div class="ctrl-btns">
        <button class="ctrl-btn primary" onclick="resetEpisode()">↺ RESET</button>
        <button class="ctrl-btn secondary" id="auto-btn" onclick="toggleAuto()">▶ AUTO</button>
      </div>
      <div class="section-title" style="margin-top:4px;">Take Action</div>
      <div class="action-btns" id="action-btns">
        <button class="action-btn btn-0 disabled-state" onclick="takeAction(0)">
          <div class="btn-icon">✓</div>
          <div><div>Do Nothing</div><div class="btn-sub">Vitals are normal</div></div>
        </button>
        <button class="action-btn btn-1 disabled-state" onclick="takeAction(1)">
          <div class="btn-icon">⚠</div>
          <div><div>Send Warning</div><div class="btn-sub">Mildly concerning vitals</div></div>
        </button>
        <button class="action-btn btn-2 disabled-state" onclick="takeAction(2)">
          <div class="btn-icon">🚨</div>
          <div><div>Emergency Alert</div><div class="btn-sub">Critical vitals detected</div></div>
        </button>
      </div>
      <div>
        <div class="section-title">Last Reward</div>
        <div style="font-family:var(--mono);font-size:1.5rem;color:var(--green);" id="last-reward-val">—</div>
      </div>
    </div>
  </div>
  <div class="score-panel">
    <div class="score-card">
      <div class="task-name">Easy Task — HR Detection</div>
      <div class="score-bar-wrap"><div class="score-bar" id="score-bar-0"></div></div>
      <div class="score-val" id="score-0">—</div>
      <div class="score-tag">Detect HR &gt; 100, send warning</div>
    </div>
    <div class="score-card">
      <div class="task-name">Medium Task — Trend Analysis</div>
      <div class="score-bar-wrap"><div class="score-bar" id="score-bar-1"></div></div>
      <div class="score-val" id="score-1">—</div>
      <div class="score-tag">Detect increasing HR trend</div>
    </div>
    <div class="score-card">
      <div class="task-name">Hard Task — Multi-Sensor</div>
      <div class="score-bar-wrap"><div class="score-bar" id="score-bar-2"></div></div>
      <div class="score-val" id="score-2">—</div>
      <div class="score-tag">Emergency alert for critical vitals</div>
    </div>
  </div>
  <div class="card log-card">
    <h2>Event Log</h2>
    <div id="log"></div>
  </div>
</main>
<div class="reward-flash" id="reward-flash"></div>
<div class="done-banner" id="done-banner">
  <div style="font-size:3rem;">🏁</div>
  <h2>EPISODE COMPLETE</h2>
  <div class="done-score" id="done-summary"></div>
  <p>Press RESET to start a new episode</p>
  <button class="ctrl-btn primary" style="max-width:200px;margin-top:8px;" onclick="closeDone()">CLOSE</button>
</div>
<script>
let hrHistory=[], totalReward=0, stepCount=0, isActive=false, autoInterval=null, ecgOffset=0;
const ecgCanvas=document.getElementById('ecg-canvas'), ecgCtx=ecgCanvas.getContext('2d');
const ECG_POINTS=200; let ecgData=new Array(ECG_POINTS).fill(0);
function ecgWave(t){const tp=((t%60)/60);if(tp<0.05)return Math.sin(tp/0.05*Math.PI)*0.15;if(tp<0.1)return -Math.sin((tp-0.05)/0.05*Math.PI)*0.05;if(tp<0.2)return 0;if(tp<0.22)return Math.sin((tp-0.2)/0.02*Math.PI)*0.3;if(tp<0.24)return -Math.sin((tp-0.22)/0.02*Math.PI);if(tp<0.26)return Math.sin((tp-0.24)/0.02*Math.PI)*1.4;if(tp<0.28)return -Math.sin((tp-0.26)/0.02*Math.PI)*0.3;if(tp<0.35)return Math.sin((tp-0.28)/0.07*Math.PI)*0.2;if(tp<0.42)return Math.sin((tp-0.35)/0.07*Math.PI)*0.2;return 0;}
function drawECG(){ecgCanvas.width=ecgCanvas.offsetWidth;ecgCanvas.height=60;const w=ecgCanvas.width,h=ecgCanvas.height;ecgCtx.clearRect(0,0,w,h);ecgOffset+=0.8;ecgData.shift();ecgData.push(ecgWave(ecgOffset));const grad=ecgCtx.createLinearGradient(0,0,w,0);grad.addColorStop(0,'rgba(0,229,255,0)');grad.addColorStop(0.7,'rgba(0,229,255,0.6)');grad.addColorStop(1,'#00e5ff');ecgCtx.beginPath();ecgCtx.strokeStyle=grad;ecgCtx.lineWidth=1.5;ecgCtx.shadowColor='#00e5ff';ecgCtx.shadowBlur=6;for(let i=0;i<ECG_POINTS;i++){const x=(i/ECG_POINTS)*w,y=h/2-ecgData[i]*(h*0.38);i===0?ecgCtx.moveTo(x,y):ecgCtx.lineTo(x,y);}ecgCtx.stroke();requestAnimationFrame(drawECG);}
drawECG();
const hrCanvas=document.getElementById('hr-canvas'), hrCtx=hrCanvas.getContext('2d');
function drawHRChart(){hrCanvas.width=hrCanvas.offsetWidth;hrCanvas.height=200;const w=hrCanvas.width,h=hrCanvas.height;hrCtx.clearRect(0,0,w,h);if(hrHistory.length<2){hrCtx.fillStyle='rgba(74,112,128,0.4)';hrCtx.font='0.72rem Share Tech Mono,monospace';hrCtx.textAlign='center';hrCtx.fillText('No data yet — reset to start',w/2,h/2);return;}const mn=Math.min(...hrHistory,55)-5,mx=Math.max(...hrHistory,160)+5;const pad={t:10,b:28,l:36,r:10},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b;const toX=i=>pad.l+(i/(hrHistory.length-1))*cw,toY=v=>pad.t+(1-(v-mn)/(mx-mn))*ch;hrCtx.strokeStyle='rgba(26,48,64,0.8)';hrCtx.lineWidth=1;[60,80,100,120,140].forEach(v=>{const y=toY(v);hrCtx.beginPath();hrCtx.moveTo(pad.l,y);hrCtx.lineTo(w-pad.r,y);hrCtx.stroke();hrCtx.fillStyle='rgba(74,112,128,0.7)';hrCtx.font='9px Share Tech Mono';hrCtx.textAlign='right';hrCtx.fillText(v,pad.l-4,y+3);});const dangerY=toY(120);hrCtx.fillStyle='rgba(255,56,96,0.05)';hrCtx.fillRect(pad.l,pad.t,cw,dangerY-pad.t);const grad=hrCtx.createLinearGradient(0,pad.t,0,h-pad.b);grad.addColorStop(0,'rgba(0,229,255,0.18)');grad.addColorStop(1,'rgba(0,229,255,0)');hrCtx.beginPath();hrHistory.forEach((v,i)=>i===0?hrCtx.moveTo(toX(i),toY(v)):hrCtx.lineTo(toX(i),toY(v)));hrCtx.lineTo(toX(hrHistory.length-1),h-pad.b);hrCtx.lineTo(toX(0),h-pad.b);hrCtx.closePath();hrCtx.fillStyle=grad;hrCtx.fill();hrCtx.beginPath();hrCtx.strokeStyle='#00e5ff';hrCtx.lineWidth=2;hrCtx.shadowColor='#00e5ff';hrCtx.shadowBlur=4;hrHistory.forEach((v,i)=>i===0?hrCtx.moveTo(toX(i),toY(v)):hrCtx.lineTo(toX(i),toY(v)));hrCtx.stroke();hrHistory.forEach((v,i)=>{hrCtx.beginPath();hrCtx.arc(toX(i),toY(v),3,0,Math.PI*2);hrCtx.fillStyle=v>120?'#ff3860':v>100?'#ffb300':'#00e5ff';hrCtx.shadowColor=hrCtx.fillStyle;hrCtx.shadowBlur=6;hrCtx.fill();});}
async function api(m,p,b){const o={method:m,headers:{'Content-Type':'application/json'}};if(b!==undefined)o.body=JSON.stringify(b);const r=await fetch(p,o);return r.json();}
function log(msg,type='info'){const el=document.getElementById('log');const now=new Date().toLocaleTimeString('en',{hour12:false});const div=document.createElement('div');div.className='log-entry';div.innerHTML=`<span class="ts">${now}</span><span class="msg-${type}">${msg}</span>`;el.prepend(div);while(el.children.length>60)el.removeChild(el.lastChild);}
function flashReward(r){const el=document.getElementById('reward-flash');el.textContent=`REWARD +${r}`;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),1200);}
function updateVitals(hr,temp){document.getElementById('hr-val').textContent=hr;document.getElementById('temp-val').textContent=temp.toFixed(1);const hrC=document.getElementById('hr-card'),tC=document.getElementById('temp-card'),aC=document.getElementById('alert-card'),aV=document.getElementById('alert-val'),aS=document.getElementById('alert-sub');hrC.className='card';tC.className='card';aC.className='card';if(hr>120||temp>39){hrC.classList.add('danger');tC.classList.add('danger');aC.classList.add('danger');aV.textContent='CRITICAL';aS.textContent='Immediate action required';}else if(hr>100||temp>38){hrC.classList.add('warn');tC.classList.add('warn');aC.classList.add('warn');aV.textContent='WARNING';aS.textContent='Monitor closely';}else{hrC.classList.add('normal');tC.classList.add('normal');aC.classList.add('normal');aV.textContent='NORMAL';aS.textContent='Vitals within range';}}
function updateStats(){document.getElementById('step-val').textContent=`${stepCount} / 20`;document.getElementById('reward-val').textContent=totalReward.toFixed(4);document.getElementById('avg-reward-val').textContent=stepCount>0?(totalReward/stepCount).toFixed(4):'—';}
async function fetchScores(){try{const[e,m,h]=await Promise.all([api('GET','/grade/easy_task'),api('GET','/grade/medium_task'),api('GET','/grade/hard_task')]);[e.score,m.score,h.score].forEach((s,i)=>{document.getElementById(`score-${i}`).textContent=s.toFixed(4);document.getElementById(`score-bar-${i}`).style.width=`${s*100}%`;});}catch(e){console.warn('Score fetch failed',e);}}
function setActionBtns(e){document.querySelectorAll('.action-btn').forEach(b=>b.classList.toggle('disabled-state',!e));}
async function resetEpisode(){if(autoInterval)toggleAuto();const d=await api('POST','/reset');const obs=d.observation;hrHistory=[obs.heart_rate];totalReward=0;stepCount=0;isActive=true;updateVitals(obs.heart_rate,obs.temperature);updateStats();drawHRChart();setActionBtns(true);document.getElementById('status-text').textContent='MONITORING — EPISODE ACTIVE';document.getElementById('last-reward-val').textContent='—';document.getElementById('done-banner').classList.remove('show');log(`Episode reset. HR=${obs.heart_rate} BPM, Temp=${obs.temperature}°C`,'info');fetchScores();}
async function takeAction(action){if(!isActive)return;const labels=['DO_NOTHING','SEND_WARNING','EMERGENCY_ALERT'],types=['ok','warn','err'];const d=await api('POST','/step',{action});totalReward+=d.reward;stepCount+=1;hrHistory.push(d.observation.heart_rate);document.getElementById('last-reward-val').textContent=`+${d.reward}`;updateVitals(d.observation.heart_rate,d.observation.temperature);updateStats();drawHRChart();flashReward(d.reward);log(`Step ${stepCount}: Action=${labels[action]}, Reward=${d.reward}, HR=${d.observation.heart_rate}`,types[action]);fetchScores();if(d.done){isActive=false;setActionBtns(false);if(autoInterval)toggleAuto();document.getElementById('status-text').textContent='EPISODE COMPLETE';const avg=(totalReward/stepCount).toFixed(4);document.getElementById('done-summary').textContent=`Total Reward: ${totalReward.toFixed(4)}  |  Avg: ${avg}  |  Steps: ${stepCount}`;document.getElementById('done-banner').classList.add('show');log(`Episode done. Total reward=${totalReward.toFixed(4)}`,'info');}}
function toggleAuto(){const btn=document.getElementById('auto-btn');if(autoInterval){clearInterval(autoInterval);autoInterval=null;btn.textContent='▶ AUTO';btn.style.borderColor='';btn.style.color='';return;}if(!isActive){resetEpisode().then(()=>startAuto());return;}startAuto();}
function startAuto(){const btn=document.getElementById('auto-btn');btn.textContent='■ STOP';btn.style.borderColor='var(--accent2)';btn.style.color='var(--accent2)';autoInterval=setInterval(async()=>{if(!isActive){toggleAuto();return;}const hr=parseInt(document.getElementById('hr-val').textContent)||75,temp=parseFloat(document.getElementById('temp-val').textContent)||37;let a=0;if(hr>120||temp>39)a=2;else if(hr>100||temp>38)a=1;await takeAction(a);},700);}
function closeDone(){document.getElementById('done-banner').classList.remove('show');}
fetchScores();
window.addEventListener('resize',()=>drawHRChart());
</script>
</body>
</html>"""

# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=HTML, status_code=200)

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
        {"name": "easy_task",   "grader_endpoint": "/grade/easy_task"},
        {"name": "medium_task", "grader_endpoint": "/grade/medium_task"},
        {"name": "hard_task",   "grader_endpoint": "/grade/hard_task"}
    ]}

# ── Graders ────────────────────────────────────────────────────────────────────

def _score():
    steps = max(state["step_count"], 1)
    avg = state["total_reward"] / steps

    easy = safe_score(avg * 0.90)

    hr_hist = state["hr_history"]
    if len(hr_hist) >= 2:
        inc = sum(1 for i in range(1, len(hr_hist)) if hr_hist[i] > hr_hist[i - 1])
        medium = safe_score(inc / (len(hr_hist) - 1) * 0.80)
    else:
        medium = safe_score(0.50)

    hard = safe_score(avg * 0.80)
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


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
