"""
Power Outage Risk Dashboard — Lightweight Flask Server
Reads pre-aggregated JSON files. RAM usage: ~30 MB (vs 500+ MB before).
"""
import os, json
from flask import Flask, jsonify, render_template_string

DATA_DIR = os.environ.get("DATA_DIR", "data")

def load(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)

# Load everything once at startup — tiny files, takes <1 second
print("Loading pre-aggregated data...")
FILTER_OPTIONS = load("filter_options.json")
KPI_ALL        = load("kpi_all.json")
INSIGHTS_ALL   = load("insights.json")
RISK_ALL       = load("risk.json")
STATES_ALL     = load("states.json")
MONTHS_ALL     = load("months.json")
EVENTS_ALL     = load("events.json")
STATE_MONTH_ALL= load("state_month.json")
POINTS_ALL     = load("points.json")
TOP_UTIL_ALL   = load("top_utilities.json")
print(f"Loaded. Points: {len(POINTS_ALL)}, States: {len(STATES_ALL)}")

# ─────────────────────────────────────────────
# FILTER HELPERS  (filter pre-aggregated data by state / risk)
# ─────────────────────────────────────────────
def filter_records(records, state="All", ownership="All", event="All", risk="All",
                   month_min=1, month_max=12):
    out = []
    for r in records:
        if state != "All" and r.get("state","") != state: continue
        if ownership != "All" and r.get("ownership","") != ownership: continue
        if event != "All" and r.get("event_type","") != event: continue
        if risk != "All" and r.get("risk_category","") != risk: continue
        m = r.get("month", 0)
        if m and not (month_min <= m <= month_max): continue
        out.append(r)
    return out

def build_payload(state, ownership, event, risk, month_min, month_max, top_n):
    # Filter point-level data
    pts = filter_records(POINTS_ALL, state, ownership, event, risk, month_min, month_max)
    top = filter_records(TOP_UTIL_ALL, state, ownership, event, risk, month_min, month_max)

    # For aggregated data, filter where possible
    def flt_state(records):
        if state == "All": return records
        return [r for r in records if r.get("state","") == state]

    def flt_event(records):
        if event == "All": return records
        return [r for r in records if r.get("event_type","") == event]

    def flt_month(records):
        if month_min == 1 and month_max == 12: return records
        return [r for r in records if month_min <= r.get("month",0) <= month_max]

    states_filtered      = flt_state(STATES_ALL)
    months_filtered      = flt_month(MONTHS_ALL)
    events_filtered      = flt_event(EVENTS_ALL)[:top_n]
    state_month_filtered = flt_month(flt_state(STATE_MONTH_ALL))

    # Recompute KPIs from pre-aggregated state data (accurate, not sampled)
    if states_filtered:
        def sagg(k, fn):
            vals = [r[k] for r in states_filtered if r.get(k) is not None]
            return fn(vals) if vals else 0

        kpi = {
            "total_rows":      KPI_ALL["total_rows"],   # always show full count
            "total_utilities": KPI_ALL["total_utilities"],
            "total_states":    len(states_filtered),    # exact count from aggregated data
            "avg_saidi":  sagg("avg_saidi",  lambda v: sum(v)/len(v)),
            "avg_saifi":  sagg("avg_saifi",  lambda v: sum(v)/len(v)),
            "avg_caidi":  sagg("avg_caidi",  lambda v: sum(v)/len(v)),
            "total_damage":   sagg("total_damage",   sum),
            "total_injuries": sagg("total_injuries", sum),
            "total_deaths":   sagg("total_deaths",   sum),
            "high_risk":   sagg("high_risk_count",  sum) if "high_risk_count" in (states_filtered[0] if states_filtered else {}) else KPI_ALL["high_risk"],
            "medium_risk": KPI_ALL["medium_risk"],
            "low_risk":    KPI_ALL["low_risk"],
        }
        # Use full KPI when no filters applied
        if state == "All" and ownership == "All" and event == "All" and risk == "All" and month_min == 1 and month_max == 12:
            kpi = KPI_ALL
        # Top state by SAIDI
        from collections import defaultdict
        state_saidi = defaultdict(list)
        for r in pts:
            if r.get("state") and r.get("saidi"):
                state_saidi[r["state"]].append(r["saidi"])
        top_state = max(state_saidi, key=lambda s: sum(state_saidi[s])/len(state_saidi[s]), default=None) if state_saidi else None
        top_state_val = (sum(state_saidi[top_state])/len(state_saidi[top_state])) if top_state else 0

        event_dmg = defaultdict(float)
        for r in pts:
            if r.get("event_type"): event_dmg[r["event_type"]] += r.get("damage",0) or 0
        top_event = max(event_dmg, key=event_dmg.get, default=None) if event_dmg else None

        insights = {
            "high_risk_records":    kpi["high_risk"],
            "top_state_saidi":      top_state,
            "top_state_saidi_value":round(top_state_val, 2),
            "top_event_damage":     top_event,
            "top_event_damage_value": event_dmg.get(top_event,0) if top_event else 0,
        }
        risk_vals = {"High Risk":0,"Medium Risk":0,"Low Risk":0}
        for r in pts: risk_vals[r.get("risk_category","Low Risk")] = risk_vals.get(r.get("risk_category","Low Risk"),0)+1
        risk_pl = {"labels":list(risk_vals.keys()),"values":list(risk_vals.values())}
    else:
        kpi      = KPI_ALL
        insights = INSIGHTS_ALL
        risk_pl  = RISK_ALL

    return {
        "kpi":          kpi,
        "insights":     insights,
        "risk":         risk_pl,
        "states":       states_filtered,
        "months":       months_filtered,
        "events":       events_filtered,
        "top_utilities":top[:20],
        "points":       pts[:2000],
        "state_month":  state_month_filtered,
    }

_cache = {}
def get_cached(state, ownership, event, risk, month_min, month_max, top_n):
    key = (state, ownership, event, risk, month_min, month_max, top_n)
    if key not in _cache:
        if len(_cache) > 100: _cache.clear()
        _cache[key] = build_payload(state, ownership, event, risk, month_min, month_max, top_n)
    return _cache[key]

# ─────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Power Outage Risk Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<style>
:root{--bg:#202124;--panel:#111827;--border:#374151;--text:#fff;--muted:#d1d5db;--blue:#2196f3}
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--text)}
.container{padding:18px}.title{font-size:34px;font-weight:700;margin-bottom:6px}
.subtitle{color:var(--muted);margin-bottom:18px;font-size:16px}
.filters{display:grid;grid-template-columns:1fr 1fr;gap:16px;background:linear-gradient(135deg,#07142f,#091b42);border:1px solid var(--border);border-radius:16px;padding:16px;margin-bottom:18px}
.filter-col{display:flex;flex-direction:column;gap:12px}
.filter-group label{display:block;font-size:13px;margin-bottom:4px;color:var(--muted)}
.filter-group select,.filter-group input{width:100%;padding:10px;border-radius:12px;border:1px solid #4b5563;background:#374151;color:white}
.btn{padding:11px 16px;background:var(--blue);color:white;border:none;border-radius:12px;cursor:pointer;font-weight:700}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
.kpi{border-radius:18px;padding:16px;color:#111827;box-shadow:0 8px 18px rgba(0,0,0,.10);transition:transform .2s}
.kpi:hover{transform:translateY(-4px)}.kpi .label{font-size:14px;color:#374151;margin-bottom:8px}.kpi .value{font-size:24px;font-weight:800}
.insight{background:linear-gradient(135deg,#07142f,#091b42);border:1px solid var(--border);border-left:6px solid #3b82f6;border-radius:14px;padding:16px;margin-bottom:18px}
.insight-title{font-size:18px;font-weight:700;margin-bottom:8px}
.section{margin-top:18px;margin-bottom:10px;font-size:28px;font-weight:700}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:10px}
.plot{width:100%;height:430px}.plot-tall{width:100%;height:620px}.plot-map{width:100%;height:700px}
.leaflet-map{width:100%;height:720px;border-radius:12px;overflow:hidden}
.map-toolbar{display:flex;gap:8px;margin-bottom:10px}
.map-tab{padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:#1f2937;color:white;cursor:pointer}
.map-tab.active{background:#2563eb;border-color:#2563eb}
.table-wrap{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:12px;overflow-x:auto}
table{width:100%;border-collapse:collapse;color:white}th,td{padding:10px;border-bottom:1px solid var(--border);text-align:left;font-size:13px}th{color:#93c5fd}
.legend-note{color:#cbd5e1;font-size:13px;margin-top:6px}.small-muted{font-size:12px;color:#6b7280}.sparkline{margin-top:8px}
.loading{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;align-items:center;justify-content:center;z-index:9999}
.loading-box{background:#111827;color:white;padding:18px 24px;border-radius:14px;border:1px solid #374151;font-size:16px}
@media(max-width:1200px){.kpis{grid-template-columns:repeat(2,1fr)}.grid-2,.filters{grid-template-columns:1fr}}
</style></head><body>
<div class="loading" id="loadingOverlay"><div class="loading-box">Updating dashboard...</div></div>
<div class="container">
<div class="title">⚡ Power Outage Risk Dashboard</div>
<div class="subtitle">Interactive dashboard for outage reliability, storm events, damage, utility risk, maps, and trends.</div>
<div class="filters">
  <div class="filter-col">
    <div class="filter-group"><label>State</label><select id="stateFilter"></select></div>
    <div class="filter-group"><label>Ownership</label><select id="ownershipFilter"></select></div>
    <div class="filter-group"><label>Event</label><select id="eventFilter"></select></div>
    <div class="filter-group"><label>Risk</label><select id="riskFilter"></select></div>
  </div>
  <div class="filter-col">
    <div class="filter-group"><label>Month</label>
      <input type="range" id="monthMin" min="1" max="12" value="1">
      <input type="range" id="monthMax" min="1" max="12" value="12">
      <div id="monthLabel">1 - 12</div>
    </div>
    <div class="filter-group"><label>Map Metric</label>
      <select id="mapMetric">
        <option value="avg_saidi">Average SAIDI</option><option value="avg_saifi">Average SAIFI</option>
        <option value="avg_caidi">Average CAIDI</option><option value="total_damage">Total Damage</option>
        <option value="total_injuries">Total Injuries</option><option value="total_deaths">Total Deaths</option>
        <option value="high_risk_count">High Risk Count</option><option value="utility_count">Utility Count</option>
      </select>
    </div>
    <div class="filter-group"><label>Map Color</label>
      <select id="mapColor">
        <option value="Reds">Reds</option><option value="Blues">Blues</option><option value="Greens">Greens</option>
        <option value="Purples">Purples</option><option value="Oranges">Oranges</option>
        <option value="Viridis">Viridis</option><option value="Cividis">Cividis</option><option value="Turbo">Turbo</option>
      </select>
    </div>
    <div class="filter-group"><button class="btn" onclick="updateDashboard()">Update Dashboard</button></div>
  </div>
</div>
<div id="kpiRow" class="kpis"></div>
<div class="insight"><div class="insight-title">Quick Insights</div><div id="insightText"></div></div>
<div class="section">📊 Risk Overview</div>
<div class="grid-2">
  <div class="card"><div id="riskPie" class="plot"></div></div>
  <div class="card"><div id="saidiHist" class="plot"></div></div>
</div>
<div class="section">⚡ Reliability Analysis</div>
<div class="grid-2">
  <div class="card"><div id="saifiHist" class="plot"></div></div>
  <div class="card"><div id="caidiHist" class="plot"></div></div>
</div>
<div class="card" style="margin-bottom:16px"><div id="bubbleScatter" class="plot-tall"></div></div>
<div class="section">🌩 Weather Impact Analysis</div>
<div class="grid-2">
  <div class="card"><div id="eventBar" class="plot"></div></div>
  <div class="card"><div id="damageHist" class="plot"></div></div>
</div>
<div class="section">🗺 Dynamic Map</div>
<div class="grid-2">
  <div class="card"><div id="stateMap" class="plot-map"></div></div>
  <div class="card"><div id="stateBar" class="plot-map"></div></div>
</div>
<div class="section">📍 Cluster / Bubble / Risk Point Map</div>
<div class="card" style="margin-bottom:16px">
  <div class="map-toolbar">
    <button class="map-tab active" id="clusterBtn" onclick="setLeafletMode('cluster')">Cluster</button>
    <button class="map-tab" id="bubbleBtn" onclick="setLeafletMode('bubble')">Bubble</button>
    <button class="map-tab" id="riskBtn" onclick="setLeafletMode('risk')">Risk</button>
  </div>
  <div id="leafletMap" class="leaflet-map"></div>
  <div class="legend-note">Map uses state centroids with jitter for spatial exploration.</div>
</div>
<div class="section">🎞 Animated Monthly Map</div>
<div class="card" style="margin-bottom:16px"><div id="animatedMap" class="plot-map"></div></div>
<div class="section">📈 Monthly Trends</div>
<div class="grid-2">
  <div class="card"><div id="trendLine" class="plot"></div></div>
  <div class="card"><div id="damageMonthBar" class="plot"></div></div>
</div>
<div class="section">🏆 Top Utilities by Risk Score</div>
<div class="table-wrap">
  <table>
    <thead><tr><th>Utility #</th><th>Utility Name</th><th>State</th><th>Ownership</th><th>Event</th><th>SAIDI</th><th>SAIFI</th><th>Damage</th><th>Risk</th></tr></thead>
    <tbody id="topUtilitiesBody"></tbody>
  </table>
</div>
</div>
<script>
let currentData=null,leafletMap=null,leafletLayer=null,leafletMode="cluster";
function showLoading(s=true){document.getElementById("loadingOverlay").style.display=s?"flex":"none";}
function fillDropdown(id,vals){const el=document.getElementById(id);el.innerHTML="";vals.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;el.appendChild(o);});}
function fmtN(x){return Number(x||0).toLocaleString();}
function fmtF(x,d=2){return Number(x||0).toFixed(d);}
function pct(c,p){return(!p||p===0)?0:((c-p)/p)*100;}
function arrowHTML(ch,low=false){const good=low?ch<0:ch>0;const arr=ch>0?"▲":ch<0?"▼":"●";const col=ch===0?"#6b7280":(good?"#10b981":"#ef4444");return`<span style="color:${col};font-size:13px;font-weight:700">${arr} ${Math.abs(ch).toFixed(1)}%</span>`;}
function animVal(el,s,e,dur=800,dec=0){const t0=performance.now();function u(now){const p=Math.min((now-t0)/dur,1);const v=s+(e-s)*p;dec>0?el.textContent=v.toFixed(dec):el.textContent=Math.round(v).toLocaleString();p<1&&requestAnimationFrame(u);}requestAnimationFrame(u);}
function sparkline(vals,color="#3b82f6"){if(!vals||!vals.length)return"";const w=120,h=28,mn=Math.min(...vals),mx=Math.max(...vals),r=(mx-mn)||1;const pts=vals.map((v,i)=>`${(i/((vals.length-1)||1))*w},${h-((v-mn)/r)*h}`).join(" ");return`<svg class="sparkline" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/></svg>`;}
async function initDashboard(){showLoading(true);const res=await fetch("/init");const d=await res.json();fillDropdown("stateFilter",d.states);fillDropdown("ownershipFilter",d.ownerships);fillDropdown("eventFilter",d.events);fillDropdown("riskFilter",d.risks);await updateDashboard();}
function getFilters(){let mn=parseInt(document.getElementById("monthMin").value),mx=parseInt(document.getElementById("monthMax").value);if(mn>mx)[mn,mx]=[mx,mn];document.getElementById("monthLabel").textContent=`${mn} - ${mx}`;return{state:document.getElementById("stateFilter").value,ownership:document.getElementById("ownershipFilter").value,event:document.getElementById("eventFilter").value,risk:document.getElementById("riskFilter").value,month_min:mn,month_max:mx,top_n:10};}
async function updateDashboard(){showLoading(true);try{const f=getFilters();const qs=new URLSearchParams(f).toString();const res=await fetch("/data?"+qs);currentData=await res.json();renderInsights();renderKPIs();renderRiskPie();renderHist("saidiHist",currentData.points,"saidi","SAIDI Distribution","#2563eb");renderHist("saifiHist",currentData.points,"saifi","SAIFI Distribution","#f59e0b");renderHist("caidiHist",currentData.points,"caidi","CAIDI Distribution","#14b8a6");renderBubble();renderTopEvents();renderHist("damageHist",currentData.points,"log_damage","Log Total Damage","#7c3aed");renderMap();renderLeafletMap();renderAnimatedMap();renderMonthly();renderTopUtilities();}finally{showLoading(false);}}
function renderInsights(){const i=currentData.insights||{};document.getElementById("insightText").innerHTML=`• High-risk records: <b>${fmtN(i.high_risk_records||0)}</b><br>• Highest avg SAIDI state: <b>${i.top_state_saidi||"N/A"}</b> (${fmtF(i.top_state_saidi_value||0)})<br>• Most damaging event type: <b>${i.top_event_damage||"N/A"}</b> (${fmtN(i.top_event_damage_value||0)})`;}
function renderKPIs(){const k=currentData.kpi||{};const months=currentData.months||[];const pm=months.length>1?months[months.length-2]:null;const lm=months.length>0?months[months.length-1]:null;const sc=pm?pct(lm.avg_saidi,pm.avg_saidi):0,fc=pm?pct(lm.avg_saifi,pm.avg_saifi):0,dc=pm?pct(lm.total_damage,pm.total_damage):0,rc=pm?pct(lm.high_risk_count,pm.high_risk_count):0;const ss=months.map(d=>d.avg_saidi||0),sf=months.map(d=>d.avg_saifi||0),sd=months.map(d=>d.total_damage||0),sr=months.map(d=>d.high_risk_count||0);const cards=[{title:"Rows",value:k.total_rows||0,display:fmtN(k.total_rows||0),color:"#dbeafe",delta:"",spark:""},{title:"Utilities",value:k.total_utilities||0,display:fmtN(k.total_utilities||0),color:"#ede9fe",delta:"",spark:""},{title:"States",value:k.total_states||0,display:fmtN(k.total_states||0),color:"#dcfce7",delta:"",spark:""},{title:"Avg SAIDI",value:k.avg_saidi||0,display:fmtF(k.avg_saidi||0),color:"#fee2e2",delta:arrowHTML(sc,true)+' <span class="small-muted">vs prev month</span>',spark:sparkline(ss,"#ef4444")},{title:"Avg SAIFI",value:k.avg_saifi||0,display:fmtF(k.avg_saifi||0),color:"#fef3c7",delta:arrowHTML(fc,true)+' <span class="small-muted">vs prev month</span>',spark:sparkline(sf,"#f59e0b")},{title:"Avg CAIDI",value:k.avg_caidi||0,display:fmtF(k.avg_caidi||0),color:"#e0f2fe",delta:"",spark:""},{title:"Total Damage",value:k.total_damage||0,display:fmtN(k.total_damage||0),color:"#fae8ff",delta:arrowHTML(dc,true)+' <span class="small-muted">vs prev month</span>',spark:sparkline(sd,"#8b5cf6")},{title:"Total Injuries",value:k.total_injuries||0,display:fmtN(k.total_injuries||0),color:"#ede9fe",delta:"",spark:""},{title:"Total Deaths",value:k.total_deaths||0,display:fmtN(k.total_deaths||0),color:"#fee2e2",delta:"",spark:""},{title:"High Risk",value:k.high_risk||0,display:fmtN(k.high_risk||0),color:"#fecaca",delta:arrowHTML(rc,true)+' <span class="small-muted">vs prev month</span>',spark:sparkline(sr,"#dc2626")},{title:"Medium Risk",value:k.medium_risk||0,display:fmtN(k.medium_risk||0),color:"#fde68a",delta:"",spark:""},{title:"Low Risk",value:k.low_risk||0,display:fmtN(k.low_risk||0),color:"#bbf7d0",delta:"",spark:""}];const row=document.getElementById("kpiRow");row.innerHTML="";cards.forEach(c=>{const div=document.createElement("div");div.className="kpi";div.style.background=`linear-gradient(135deg,${c.color} 0%,#ffffff 100%)`;div.innerHTML=`<div class="label">${c.title}</div><div class="value kpi-value" data-target="${c.value}" data-decimals="${String(c.value).includes('.')?2:0}">${c.display}</div><div style="margin-top:6px;min-height:18px">${c.delta||""}</div>${c.spark}`;row.appendChild(div);});document.querySelectorAll(".kpi-value").forEach(el=>{const t=parseFloat(el.dataset.target||"0");const d=parseInt(el.dataset.decimals||"0");animVal(el,0,t,800,d);});}
function renderRiskPie(){const r=currentData.risk||{labels:[],values:[]};Plotly.react("riskPie",[{type:"pie",labels:r.labels,values:r.values,hole:0.5,marker:{colors:["#ef4444","#f59e0b","#10b981"]}}],{title:"Risk Category Distribution",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});}
function renderHist(t,data,f,title,color){Plotly.react(t,[{type:"histogram",x:data.map(d=>d[f]||0),marker:{color:color}}],{title,paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});}
function renderBubble(){const d=currentData.points||[];Plotly.react("bubbleScatter",[{type:"scatter",mode:"markers",x:d.map(p=>p.saidi||0),y:d.map(p=>p.saifi||0),text:d.map(p=>p.utility_name||"Unknown"),marker:{size:d.map(p=>Math.max(6,Math.min(20,Math.sqrt((p.damage||0)+1)/70))),color:d.map(p=>p.risk_category==="High Risk"?"#ef4444":p.risk_category==="Medium Risk"?"#f59e0b":"#10b981"),opacity:0.7}}],{title:"SAIDI vs SAIFI Bubble",xaxis:{title:"SAIDI"},yaxis:{title:"SAIFI"},paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});}
function renderTopEvents(){const d=currentData.events||[];const pd=[...d].slice(0,10).reverse();Plotly.react("eventBar",[{type:"bar",x:pd.map(e=>e.count),y:pd.map(e=>e.event_type),orientation:"h",marker:{color:pd.map(e=>e.count),colorscale:"Blues"}}],{title:"Top Event Types",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});}
function renderMap(){const states=currentData.states||[];const mm=document.getElementById("mapMetric").value;const mc=document.getElementById("mapColor").value;Plotly.react("stateMap",[{type:"choropleth",locationmode:"USA-states",locations:states.map(d=>d.state),z:states.map(d=>d[mm]),colorscale:mc,marker:{line:{color:"white",width:0.5}},colorbar:{title:mm}}],{title:"US State Map",geo:{scope:"usa",bgcolor:"#111827"},paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});const sorted=[...states].sort((a,b)=>(b[mm]||0)-(a[mm]||0)).slice(0,15).reverse();Plotly.react("stateBar",[{type:"bar",x:sorted.map(d=>d[mm]),y:sorted.map(d=>d.state),orientation:"h",marker:{color:sorted.map(d=>d[mm]),colorscale:mc}}],{title:"Top States",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});}
function renderAnimatedMap(){const data=currentData.state_month||[];const mm=document.getElementById("mapMetric").value;const mc=document.getElementById("mapColor").value;Plotly.react("animatedMap",[{type:"choropleth",locationmode:"USA-states",locations:data.map(d=>d.state),z:data.map(d=>d[mm]||0),colorscale:mc,marker:{line:{color:"white",width:0.5}}}],{title:"Animated Monthly Map",geo:{scope:"usa",bgcolor:"#111827"},paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"},updatemenus:[],sliders:[]},{responsive:true});const months=[...new Set(data.map(d=>d.month))].sort((a,b)=>a-b);const frames=months.map(m=>{const sub=data.filter(d=>d.month===m);return{name:String(m),data:[{type:"choropleth",locationmode:"USA-states",locations:sub.map(d=>d.state),z:sub.map(d=>d[mm]||0),colorscale:mc,marker:{line:{color:"white",width:0.5}}}]};});Plotly.addFrames("animatedMap",frames);Plotly.relayout("animatedMap",{updatemenus:[{type:"buttons",showactive:false,x:0.05,y:1.15,buttons:[{label:"Play",method:"animate",args:[null,{fromcurrent:true,frame:{duration:900,redraw:true},transition:{duration:300}}]}]}],sliders:[{active:0,currentvalue:{prefix:"Month: "},steps:months.map(m=>({label:String(m),method:"animate",args:[[String(m)],{mode:"immediate",frame:{duration:500,redraw:true},transition:{duration:200}}]}))}]});}
function renderMonthly(){const d=currentData.months||[];Plotly.react("trendLine",[{type:"scatter",mode:"lines+markers",x:d.map(m=>m.month),y:d.map(m=>m.avg_saidi),name:"Avg SAIDI"},{type:"scatter",mode:"lines+markers",x:d.map(m=>m.month),y:d.map(m=>m.avg_saifi),name:"Avg SAIFI"}],{title:"Monthly SAIDI vs SAIFI",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});Plotly.react("damageMonthBar",[{type:"bar",x:d.map(m=>m.month),y:d.map(m=>m.total_damage),marker:{color:d.map(m=>m.total_damage),colorscale:"Purples"}}],{title:"Monthly Total Damage",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"white"}},{responsive:true});}
function renderTopUtilities(){const d=currentData.top_utilities||[];const body=document.getElementById("topUtilitiesBody");body.innerHTML="";d.forEach(r=>{const tr=document.createElement("tr");tr.innerHTML=`<td>${r.utility_number||""}</td><td>${r.utility_name||""}</td><td>${r.state||""}</td><td>${r.ownership||""}</td><td>${r.event_type||""}</td><td>${(r.saidi||0).toFixed(2)}</td><td>${(r.saifi||0).toFixed(2)}</td><td>${Math.round(r.damage||0).toLocaleString()}</td><td>${r.risk_category||""}</td>`;body.appendChild(tr);});}
function getRiskColor(r){return r==="High Risk"?"#ef4444":r==="Medium Risk"?"#f59e0b":"#10b981";}
function setLeafletMode(m){leafletMode=m;["clusterBtn","bubbleBtn","riskBtn"].forEach(id=>document.getElementById(id).classList.remove("active"));document.getElementById(m==="cluster"?"clusterBtn":m==="bubble"?"bubbleBtn":"riskBtn").classList.add("active");renderLeafletMap();}
function initLeaflet(){if(leafletMap)return;leafletMap=L.map("leafletMap").setView([39.5,-98.35],4);L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{attribution:"&copy; OpenStreetMap &copy; CARTO",subdomains:"abcd",maxZoom:19}).addTo(leafletMap);}
function renderLeafletMap(){initLeaflet();if(leafletLayer){leafletMap.removeLayer(leafletLayer);leafletLayer=null;}const data=currentData.points||[];if(leafletMode==="cluster"){const g=L.markerClusterGroup();data.forEach(d=>{if(d.lat==null||d.lon==null)return;const m=L.circleMarker([d.lat,d.lon],{radius:6,color:getRiskColor(d.risk_category),fillColor:getRiskColor(d.risk_category),fillOpacity:0.75,weight:1});m.bindPopup(`<b>${d.utility_name||"Unknown"}</b><br>State: ${d.state||""}<br>Risk: ${d.risk_category||""}<br>SAIDI: ${(d.saidi||0).toFixed(2)}<br>Damage: ${Math.round(d.damage||0).toLocaleString()}`);g.addLayer(m);});leafletLayer=g;}else if(leafletMode==="bubble"){const g=L.layerGroup();data.forEach(d=>{if(d.lat==null||d.lon==null)return;const r=Math.max(5,Math.min(18,Math.sqrt((d.damage||0)+1)/90));const m=L.circleMarker([d.lat,d.lon],{radius:r,color:"#2563eb",fillColor:"#60a5fa",fillOpacity:0.45,weight:1});m.bindPopup(`<b>${d.utility_name||"Unknown"}</b><br>Damage: ${Math.round(d.damage||0).toLocaleString()}`);g.addLayer(m);});leafletLayer=g;}else{const g=L.layerGroup();data.forEach(d=>{if(d.lat==null||d.lon==null)return;const m=L.circleMarker([d.lat,d.lon],{radius:7,color:getRiskColor(d.risk_category),fillColor:getRiskColor(d.risk_category),fillOpacity:0.7,weight:1});m.bindPopup(`<b>${d.utility_name||"Unknown"}</b><br>Risk: ${d.risk_category||""}<br>Score: ${(d.risk_score||0).toFixed(4)}`);g.addLayer(m);});leafletLayer=g;}leafletLayer.addTo(leafletMap);}
initDashboard();
</script></body></html>"""

# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/init")
def init():
    return jsonify(FILTER_OPTIONS)

@app.route("/data")
def data():
    from flask import request
    state     = request.args.get("state","All")
    ownership = request.args.get("ownership","All")
    event     = request.args.get("event","All")
    risk      = request.args.get("risk","All")
    month_min = int(request.args.get("month_min",1))
    month_max = int(request.args.get("month_max",12))
    top_n     = int(request.args.get("top_n",10))
    return jsonify(get_cached(state,ownership,event,risk,month_min,month_max,top_n))

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8000))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0",port=port,debug=False)
