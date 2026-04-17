"""
Power Outage Risk Dashboard - Flask Server
Converted from Google Colab for permanent deployment on Render / Railway / etc.

HOW TO USE:
  1. Place your dashboard_clean_dataset.parquet (or .csv) in the same folder as this file.
  2. pip install -r requirements.txt
  3. python app.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from functools import lru_cache
from flask import Flask, jsonify, request, render_template_string

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETTINGS — change DATA_PATH if your file has a different name/location
# ─────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "dashboard_clean_dataset.parquet")
TOP_N_EVENTS     = 10
TOP_N_UTILITIES  = 20
POINT_SAMPLE_SIZE = 3000

VALID_US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY'
}

STATE_GEO = {
    "AL":[32.31,-86.90],"AK":[64.20,-153.37],"AZ":[34.27,-111.43],"AR":[34.89,-92.37],
    "CA":[36.78,-119.42],"CO":[39.55,-105.78],"CT":[41.60,-72.64],"DE":[38.99,-75.50],
    "FL":[27.99,-81.76],"GA":[33.04,-83.64],"HI":[20.79,-156.33],"ID":[44.07,-114.74],
    "IL":[40.35,-88.99],"IN":[39.76,-86.13],"IA":[42.00,-93.21],"KS":[38.52,-96.72],
    "KY":[37.67,-84.67],"LA":[30.99,-91.96],"ME":[44.69,-69.38],"MD":[39.04,-76.64],
    "MA":[42.23,-71.53],"MI":[44.31,-85.60],"MN":[46.39,-94.64],"MS":[32.74,-89.67],
    "MO":[38.46,-92.29],"MT":[46.88,-110.36],"NE":[41.49,-99.90],"NV":[38.80,-116.42],
    "NH":[43.19,-71.57],"NJ":[40.06,-74.41],"NM":[34.84,-106.25],"NY":[42.96,-75.37],
    "NC":[35.63,-79.81],"ND":[47.52,-99.78],"OH":[40.36,-82.99],"OK":[35.56,-96.92],
    "OR":[44.57,-122.07],"PA":[41.20,-77.19],"RI":[41.70,-71.51],"SC":[33.84,-80.90],
    "SD":[44.37,-100.35],"TN":[35.86,-86.35],"TX":[31.97,-99.90],"UT":[39.32,-111.09],
    "VT":[44.05,-72.71],"VA":[37.77,-78.17],"WA":[47.40,-120.56],"WV":[38.64,-80.62],
    "WI":[44.26,-89.62],"WY":[42.75,-107.30]
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    print(f"Loading data from: {DATA_PATH}")
    if DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)

    # Standardise state column
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.upper().str.strip()
        df = df[df["State"].isin(VALID_US_STATES)].copy()

    # Convert category columns back to string for JSON serialisation
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].astype(str)

    # Ensure key numeric columns exist
    for c in ["IEEE_AllEvents_SAIDI_min_per_yr","IEEE_AllEvents_SAIFI_times_per_yr",
              "IEEE_AllEvents_CAIDI_min_per_interruption","total_damage_usd",
              "total_injuries","total_deaths","human_impact_score","risk_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # lat/lon: use real coords if available, else state centroids + jitter
    rng = np.random.default_rng(42)
    lat_cands = [c for c in df.columns if "lat" in c.lower()]
    lon_cands = [c for c in df.columns if ("lon" in c.lower() or "lng" in c.lower())]
    if lat_cands and lon_cands:
        df["lat"] = pd.to_numeric(df[lat_cands[0]], errors="coerce")
        df["lon"] = pd.to_numeric(df[lon_cands[0]], errors="coerce")
    else:
        df["lat"] = df["State"].map(lambda s: STATE_GEO.get(s,[39.5,-98.35])[0])
        df["lon"] = df["State"].map(lambda s: STATE_GEO.get(s,[39.5,-98.35])[1])
        df["lat"] += rng.uniform(-1.6, 1.6, len(df))
        df["lon"] += rng.uniform(-1.6, 1.6, len(df))

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


df_dash = load_data()

# Filter option lists
STATE_OPTIONS    = ["All"] + sorted(df_dash["State"].dropna().unique().tolist())
OWNERSHIP_OPTIONS = (["All"] + sorted(df_dash["Ownership"].dropna().unique().tolist())
                     if "Ownership" in df_dash.columns else ["All"])
EVENT_OPTIONS    = (["All"] + sorted(df_dash["EVENT_TYPE"].dropna().unique().tolist())
                    if "EVENT_TYPE" in df_dash.columns else ["All"])
RISK_OPTIONS     = ["All","High Risk","Medium Risk","Low Risk"]

# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
def clean_record_dict(df_in):
    return json.loads(df_in.to_json(orient="records"))

def get_filtered_df(state="All", ownership="All", event="All", risk="All",
                    month_min=1, month_max=12):
    mask = pd.Series(True, index=df_dash.index)
    if state != "All" and "State" in df_dash.columns:
        mask &= df_dash["State"].astype(str) == state
    if ownership != "All" and "Ownership" in df_dash.columns:
        mask &= df_dash["Ownership"].astype(str) == ownership
    if event != "All" and "EVENT_TYPE" in df_dash.columns:
        mask &= df_dash["EVENT_TYPE"].astype(str) == event
    if risk != "All" and "risk_category" in df_dash.columns:
        mask &= df_dash["risk_category"].astype(str) == risk
    if "BEGIN_MONTH" in df_dash.columns:
        mask &= df_dash["BEGIN_MONTH"].between(month_min, month_max)
    return df_dash.loc[mask].copy()

def make_payload(filtered_df, top_n=TOP_N_EVENTS, point_sample_size=POINT_SAMPLE_SIZE):
    if filtered_df.empty:
        return {"kpi":{},"insights":{},"risk":{"labels":[],"values":[]},
                "states":[],"months":[],"events":[],"top_utilities":[],
                "points":[],"state_month":[]}

    point_df = (filtered_df.sample(point_sample_size, random_state=42)
                if len(filtered_df) > point_sample_size else filtered_df.copy())

    kpi = {
        "total_rows":       int(len(filtered_df)),
        "total_utilities":  int(filtered_df["Utility Number"].nunique()) if "Utility Number" in filtered_df.columns else 0,
        "total_states":     int(filtered_df["State"].nunique()) if "State" in filtered_df.columns else 0,
        "avg_saidi":        float(filtered_df["IEEE_AllEvents_SAIDI_min_per_yr"].mean()) if "IEEE_AllEvents_SAIDI_min_per_yr" in filtered_df.columns else 0,
        "avg_saifi":        float(filtered_df["IEEE_AllEvents_SAIFI_times_per_yr"].mean()) if "IEEE_AllEvents_SAIFI_times_per_yr" in filtered_df.columns else 0,
        "avg_caidi":        float(filtered_df["IEEE_AllEvents_CAIDI_min_per_interruption"].mean()) if "IEEE_AllEvents_CAIDI_min_per_interruption" in filtered_df.columns else 0,
        "total_damage":     float(filtered_df["total_damage_usd"].sum()) if "total_damage_usd" in filtered_df.columns else 0,
        "total_injuries":   float(filtered_df["total_injuries"].sum()) if "total_injuries" in filtered_df.columns else 0,
        "total_deaths":     float(filtered_df["total_deaths"].sum()) if "total_deaths" in filtered_df.columns else 0,
        "high_risk":        int((filtered_df["risk_category"].astype(str)=="High Risk").sum()) if "risk_category" in filtered_df.columns else 0,
        "medium_risk":      int((filtered_df["risk_category"].astype(str)=="Medium Risk").sum()) if "risk_category" in filtered_df.columns else 0,
        "low_risk":         int((filtered_df["risk_category"].astype(str)=="Low Risk").sum()) if "risk_category" in filtered_df.columns else 0,
    }
    # Replace NaN in kpi
    kpi = {k: (0 if (isinstance(v, float) and np.isnan(v)) else v) for k,v in kpi.items()}

    insights = {"high_risk_records": kpi["high_risk"],
                "top_state_saidi": None, "top_state_saidi_value": None,
                "top_event_damage": None, "top_event_damage_value": None}
    if {"State","IEEE_AllEvents_SAIDI_min_per_yr"}.issubset(filtered_df.columns):
        tmp = filtered_df.groupby("State")["IEEE_AllEvents_SAIDI_min_per_yr"].mean().sort_values(ascending=False)
        if len(tmp):
            insights["top_state_saidi"] = str(tmp.index[0])
            insights["top_state_saidi_value"] = float(tmp.iloc[0])
    if {"EVENT_TYPE","total_damage_usd"}.issubset(filtered_df.columns):
        tmp = filtered_df.groupby("EVENT_TYPE")["total_damage_usd"].sum().sort_values(ascending=False)
        if len(tmp):
            insights["top_event_damage"] = str(tmp.index[0])
            insights["top_event_damage_value"] = float(tmp.iloc[0])

    risk_counts = filtered_df["risk_category"].astype(str).value_counts() if "risk_category" in filtered_df.columns else pd.Series(dtype=int)
    risk = {"labels": risk_counts.index.tolist(), "values": [int(v) for v in risk_counts.values]}

    state_df = filtered_df.groupby("State").agg(
        avg_saidi=("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
        avg_saifi=("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
        avg_caidi=("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
        total_damage=("total_damage_usd","sum"),
        total_injuries=("total_injuries","sum"),
        total_deaths=("total_deaths","sum"),
        utility_count=("Utility Number","nunique"),
        high_risk_count=("risk_category", lambda s: (s.astype(str)=="High Risk").sum())
    ).reset_index() if "State" in filtered_df.columns else pd.DataFrame()

    month_df = filtered_df.groupby("BEGIN_MONTH").agg(
        avg_saidi=("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
        avg_saifi=("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
        avg_caidi=("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
        total_damage=("total_damage_usd","sum"),
        total_injuries=("total_injuries","sum"),
        total_deaths=("total_deaths","sum"),
        high_risk_count=("risk_category", lambda s: (s.astype(str)=="High Risk").sum())
    ).reset_index().sort_values("BEGIN_MONTH") if "BEGIN_MONTH" in filtered_df.columns else pd.DataFrame()

    event_df = filtered_df.groupby("EVENT_TYPE").agg(
        count=("EVENT_TYPE","size"),
        avg_saidi=("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
        avg_saifi=("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
        total_damage=("total_damage_usd","sum"),
        total_injuries=("total_injuries","sum"),
        total_deaths=("total_deaths","sum")
    ).reset_index().sort_values("count", ascending=False).head(top_n) if "EVENT_TYPE" in filtered_df.columns else pd.DataFrame()

    top_util_df = (filtered_df.sort_values("risk_score", ascending=False).head(TOP_N_UTILITIES).copy()
                   if "risk_score" in filtered_df.columns else filtered_df.head(TOP_N_UTILITIES).copy())

    state_month_df = pd.DataFrame()
    if {"BEGIN_MONTH","State"}.issubset(filtered_df.columns):
        state_month_df = filtered_df.groupby(["BEGIN_MONTH","State"]).agg(
            avg_saidi=("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
            avg_saifi=("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
            avg_caidi=("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
            total_damage=("total_damage_usd","sum"),
            total_injuries=("total_injuries","sum"),
            total_deaths=("total_deaths","sum"),
            high_risk_count=("risk_category", lambda s: (s.astype(str)=="High Risk").sum()),
            utility_count=("Utility Number","nunique")
        ).reset_index()

    rename_map = {
        "Utility Name":"utility_name","Utility Number":"utility_number",
        "State":"state","Ownership":"ownership","EVENT_TYPE":"event_type",
        "risk_category":"risk_category","risk_score":"risk_score",
        "IEEE_AllEvents_SAIDI_min_per_yr":"saidi","IEEE_AllEvents_SAIFI_times_per_yr":"saifi",
        "IEEE_AllEvents_CAIDI_min_per_interruption":"caidi","total_damage_usd":"damage",
        "log_total_damage":"log_damage","total_injuries":"injuries","total_deaths":"deaths",
        "human_impact_score":"impact","BEGIN_MONTH":"month","MAGNITUDE":"magnitude",
        "lat":"lat","lon":"lon"
    }
    for df_ in [state_df, month_df, event_df, top_util_df, state_month_df, point_df]:
        df_.rename(columns=rename_map, inplace=True)

    # Fill NaN for JSON
    def safe_records(df_):
        return json.loads(df_.fillna(0).to_json(orient="records")) if not df_.empty else []

    return {
        "kpi": kpi, "insights": insights, "risk": risk,
        "states":      safe_records(state_df),
        "months":      safe_records(month_df),
        "events":      safe_records(event_df),
        "top_utilities": safe_records(top_util_df),
        "points":      safe_records(point_df),
        "state_month": safe_records(state_month_df),
    }

@lru_cache(maxsize=256)
def get_payload_cached(state, ownership, event, risk, month_min, month_max, top_n):
    filtered_df = get_filtered_df(state=state, ownership=ownership, event=event,
                                  risk=risk, month_min=month_min, month_max=month_max)
    return make_payload(filtered_df, top_n=top_n, point_sample_size=POINT_SAMPLE_SIZE)

# ─────────────────────────────────────────────
# HTML TEMPLATE  (identical to your Colab version)
# ─────────────────────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Power Outage Risk Dashboard</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>

<style>
  :root{
    --bg:#202124;
    --panel:#111827;
    --border:#374151;
    --text:#ffffff;
    --muted:#d1d5db;
    --blue:#2196f3;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--text)}
  .container{padding:18px}
  .title{font-size:34px;font-weight:700;margin-bottom:6px}
  .subtitle{color:var(--muted);margin-bottom:18px;font-size:16px}
  .filters{
    display:grid;grid-template-columns:1fr 1fr;gap:16px;
    background:linear-gradient(135deg,#07142f,#091b42);
    border:1px solid var(--border);
    border-radius:16px;padding:16px;margin-bottom:18px;
    box-shadow:0 6px 20px rgba(0,0,0,.18)
  }
  .filter-col{display:flex;flex-direction:column;gap:12px}
  .filter-group label{display:block;font-size:13px;margin-bottom:4px;color:var(--muted)}
  .filter-group select,.filter-group input{
    width:100%;padding:10px;border-radius:12px;border:1px solid #4b5563;background:#374151;color:white
  }
  .btn{
    padding:11px 16px;background:var(--blue);color:white;border:none;border-radius:12px;
    cursor:pointer;font-weight:700;box-shadow:0 4px 12px rgba(33,150,243,.25)
  }
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
  .kpi{
    border-radius:18px;padding:16px;color:#111827;
    box-shadow:0 8px 18px rgba(0,0,0,.10);
    transition:transform .2s ease, box-shadow .2s ease
  }
  .kpi:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(0,0,0,.16)}
  .kpi .label{font-size:14px;color:#374151;margin-bottom:8px}
  .kpi .value{font-size:24px;font-weight:800}
  .insight{
    background:linear-gradient(135deg,#07142f,#091b42);
    border:1px solid var(--border);border-left:6px solid #3b82f6;
    border-radius:14px;padding:16px;margin-bottom:18px
  }
  .insight-title{font-size:18px;font-weight:700;margin-bottom:8px}
  .section{margin-top:18px;margin-bottom:10px;font-size:28px;font-weight:700}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
  .card{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px;padding:10px;
    box-shadow:0 6px 20px rgba(0,0,0,.15)
  }
  .plot{width:100%;height:430px}
  .plot-tall{width:100%;height:620px}
  .plot-map{width:100%;height:700px}
  .leaflet-map{width:100%;height:720px;border-radius:12px;overflow:hidden}
  .map-toolbar{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}
  .map-tab{padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:#1f2937;color:white;cursor:pointer}
  .map-tab.active{background:#2563eb;border-color:#2563eb}
  .table-wrap{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px;padding:12px;overflow-x:auto
  }
  table{width:100%;border-collapse:collapse;color:white}
  th,td{padding:10px;border-bottom:1px solid var(--border);text-align:left;font-size:13px}
  th{color:#93c5fd}
  .legend-note{color:#cbd5e1;font-size:13px;margin-top:6px}
  .small-muted{font-size:12px;color:#6b7280}
  .sparkline{margin-top:8px}
  .loading{
    position:fixed; inset:0; background:rgba(0,0,0,0.45);
    display:none; align-items:center; justify-content:center; z-index:9999;
  }
  .loading-box{
    background:#111827; color:white; padding:18px 24px; border-radius:14px;
    border:1px solid #374151; font-size:16px; box-shadow:0 8px 24px rgba(0,0,0,.25)
  }
  @media (max-width:1200px){
    .kpis{grid-template-columns:repeat(2,1fr)}
    .grid-2,.filters{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<div class="loading" id="loadingOverlay">
  <div class="loading-box">Updating dashboard...</div>
</div>

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
      <div class="filter-group">
        <label>Month</label>
        <input type="range" id="monthMin" min="1" max="12" value="1">
        <input type="range" id="monthMax" min="1" max="12" value="12">
        <div id="monthLabel">1 - 12</div>
      </div>
      <div class="filter-group">
        <label>Map Metric</label>
        <select id="mapMetric">
          <option value="avg_saidi">Average SAIDI</option>
          <option value="avg_saifi">Average SAIFI</option>
          <option value="avg_caidi">Average CAIDI</option>
          <option value="total_damage">Total Damage</option>
          <option value="total_injuries">Total Injuries</option>
          <option value="total_deaths">Total Deaths</option>
          <option value="high_risk_count">High Risk Count</option>
          <option value="utility_count">Utility Count</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Map Color</label>
        <select id="mapColor">
          <option value="Reds">Reds</option>
          <option value="Blues">Blues</option>
          <option value="Greens">Greens</option>
          <option value="Purples">Purples</option>
          <option value="Oranges">Oranges</option>
          <option value="Viridis">Viridis</option>
          <option value="Cividis">Cividis</option>
          <option value="Turbo">Turbo</option>
        </select>
      </div>
      <div class="filter-group"><button class="btn" onclick="updateDashboard()">Update Dashboard</button></div>
    </div>
  </div>

  <div id="kpiRow" class="kpis"></div>

  <div class="insight">
    <div class="insight-title">Quick Insights</div>
    <div id="insightText"></div>
  </div>

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
  <div class="card" style="margin-bottom:16px;"><div id="bubbleScatter" class="plot-tall"></div></div>

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
  <div class="card" style="margin-bottom:16px;">
    <div class="map-toolbar">
      <button class="map-tab active" id="clusterBtn" onclick="setLeafletMode('cluster')">Cluster</button>
      <button class="map-tab" id="bubbleBtn" onclick="setLeafletMode('bubble')">Bubble</button>
      <button class="map-tab" id="riskBtn" onclick="setLeafletMode('risk')">Risk</button>
    </div>
    <div id="leafletMap" class="leaflet-map"></div>
    <div class="legend-note">If your source file does not contain true latitude/longitude, this map uses state centroids with jitter for visual exploration.</div>
  </div>

  <div class="section">🎞 Animated Monthly Map</div>
  <div class="card" style="margin-bottom:16px;"><div id="animatedMap" class="plot-map"></div></div>

  <div class="section">📈 Monthly Trends</div>
  <div class="grid-2">
    <div class="card"><div id="trendLine" class="plot"></div></div>
    <div class="card"><div id="damageMonthBar" class="plot"></div></div>
  </div>

  <div class="section">🏆 Top Utilities by Risk Score</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Utility Number</th><th>Utility Name</th><th>State</th><th>Ownership</th>
          <th>Event</th><th>SAIDI</th><th>SAIFI</th><th>Damage</th><th>Risk</th>
        </tr>
      </thead>
      <tbody id="topUtilitiesBody"></tbody>
    </table>
  </div>
</div>

<script>
let currentData = null;
let leafletMap = null;
let leafletLayer = null;
let leafletMode = "cluster";

function showLoading(show=true){
  document.getElementById("loadingOverlay").style.display = show ? "flex" : "none";
}

function fillDropdown(id, values){
  const el = document.getElementById(id);
  el.innerHTML = "";
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    el.appendChild(opt);
  });
}

function fmtNumber(x){ return Number(x || 0).toLocaleString(); }
function fmtFloat(x, d=2){ return Number(x || 0).toFixed(d); }

function pctChange(current, previous){
  if (!previous || previous === 0) return 0;
  return ((current - previous) / previous) * 100;
}

function arrowHTML(change, betterWhenLower=false){
  const positiveGood = betterWhenLower ? change < 0 : change > 0;
  const arrow = change > 0 ? "▲" : change < 0 ? "▼" : "●";
  const color = change === 0 ? "#6b7280" : (positiveGood ? "#10b981" : "#ef4444");
  return `<span style="color:${color}; font-size:13px; font-weight:700;">${arrow} ${Math.abs(change).toFixed(1)}%</span>`;
}

function animateValue(el, start, end, duration=800, decimals=0){
  const startTime = performance.now();
  function update(now){
    const progress = Math.min((now - startTime) / duration, 1);
    const value = start + (end - start) * progress;
    if (decimals > 0) el.textContent = value.toFixed(decimals);
    else el.textContent = Math.round(value).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function makeSparkline(values, color="#3b82f6"){
  if (!values || values.length === 0) return "";
  const w = 120, h = 28;
  const min = Math.min(...values), max = Math.max(...values);
  const range = (max - min) || 1;
  const points = values.map((v, i) => {
    const x = (i / ((values.length - 1) || 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(" ");
  return `<svg class="sparkline" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <polyline fill="none" stroke="${color}" stroke-width="2" points="${points}" />
  </svg>`;
}

async function initDashboard(){
  showLoading(true);
  const res = await fetch("/init");
  const data = await res.json();
  fillDropdown("stateFilter", data.states);
  fillDropdown("ownershipFilter", data.ownerships);
  fillDropdown("eventFilter", data.events);
  fillDropdown("riskFilter", data.risks);
  await updateDashboard();
}

function getFilters(){
  let monthMin = parseInt(document.getElementById("monthMin").value);
  let monthMax = parseInt(document.getElementById("monthMax").value);
  if (monthMin > monthMax) [monthMin, monthMax] = [monthMax, monthMin];
  document.getElementById("monthLabel").textContent = `${monthMin} - ${monthMax}`;
  return {
    state: document.getElementById("stateFilter").value,
    ownership: document.getElementById("ownershipFilter").value,
    event: document.getElementById("eventFilter").value,
    risk: document.getElementById("riskFilter").value,
    month_min: monthMin, month_max: monthMax, top_n: 10
  };
}

async function updateDashboard(){
  showLoading(true);
  try{
    const f = getFilters();
    const qs = new URLSearchParams(f).toString();
    const res = await fetch("/data?" + qs);
    currentData = await res.json();
    renderInsights(); renderKPIs(); renderRiskPie();
    renderHist("saidiHist", currentData.points, "saidi", "SAIDI Distribution", "#2563eb");
    renderHist("saifiHist", currentData.points, "saifi", "SAIFI Distribution", "#f59e0b");
    renderHist("caidiHist", currentData.points, "caidi", "CAIDI Distribution", "#14b8a6");
    renderBubble(); renderTopEvents();
    renderHist("damageHist", currentData.points, "log_damage", "Log Total Damage Distribution", "#7c3aed");
    renderMap(); renderLeafletMap(); renderAnimatedMap(); renderMonthly(); renderTopUtilities();
  } finally { showLoading(false); }
}

function renderInsights(){
  const i = currentData.insights || {};
  document.getElementById("insightText").innerHTML = `
    • High-risk records: <b>${fmtNumber(i.high_risk_records || 0)}</b><br>
    • Highest avg SAIDI state: <b>${i.top_state_saidi || "N/A"}</b> (${fmtFloat(i.top_state_saidi_value || 0)})<br>
    • Most damaging event type: <b>${i.top_event_damage || "N/A"}</b> (${fmtNumber(i.top_event_damage_value || 0)})
  `;
}

function renderKPIs(){
  const k = currentData.kpi || {};
  const months = currentData.months || [];
  const prevMonth = months.length > 1 ? months[months.length - 2] : null;
  const latestMonth = months.length > 0 ? months[months.length - 1] : null;
  const saidiChange  = prevMonth ? pctChange(latestMonth.avg_saidi, prevMonth.avg_saidi) : 0;
  const saifiChange  = prevMonth ? pctChange(latestMonth.avg_saifi, prevMonth.avg_saifi) : 0;
  const damageChange = prevMonth ? pctChange(latestMonth.total_damage, prevMonth.total_damage) : 0;
  const highRiskChange = prevMonth ? pctChange(latestMonth.high_risk_count, prevMonth.high_risk_count) : 0;
  const sparkSaidi  = months.map(d => d.avg_saidi || 0);
  const sparkSaifi  = months.map(d => d.avg_saifi || 0);
  const sparkDamage = months.map(d => d.total_damage || 0);
  const sparkRisk   = months.map(d => d.high_risk_count || 0);
  const cards = [
    {title:"Rows",         value:k.total_rows||0,     display:fmtNumber(k.total_rows||0),     color:"#dbeafe", delta:"", spark:""},
    {title:"Utilities",    value:k.total_utilities||0, display:fmtNumber(k.total_utilities||0), color:"#ede9fe", delta:"", spark:""},
    {title:"States",       value:k.total_states||0,   display:fmtNumber(k.total_states||0),   color:"#dcfce7", delta:"", spark:""},
    {title:"Avg SAIDI",    value:k.avg_saidi||0,      display:fmtFloat(k.avg_saidi||0),       color:"#fee2e2", delta:arrowHTML(saidiChange,true)+' <span class="small-muted">vs prev month</span>',  spark:makeSparkline(sparkSaidi,"#ef4444")},
    {title:"Avg SAIFI",    value:k.avg_saifi||0,      display:fmtFloat(k.avg_saifi||0),       color:"#fef3c7", delta:arrowHTML(saifiChange,true)+' <span class="small-muted">vs prev month</span>',  spark:makeSparkline(sparkSaifi,"#f59e0b")},
    {title:"Avg CAIDI",    value:k.avg_caidi||0,      display:fmtFloat(k.avg_caidi||0),       color:"#e0f2fe", delta:"", spark:""},
    {title:"Total Damage", value:k.total_damage||0,   display:fmtNumber(k.total_damage||0),   color:"#fae8ff", delta:arrowHTML(damageChange,true)+' <span class="small-muted">vs prev month</span>', spark:makeSparkline(sparkDamage,"#8b5cf6")},
    {title:"Total Injuries",value:k.total_injuries||0,display:fmtNumber(k.total_injuries||0), color:"#ede9fe", delta:"", spark:""},
    {title:"Total Deaths", value:k.total_deaths||0,   display:fmtNumber(k.total_deaths||0),   color:"#fee2e2", delta:"", spark:""},
    {title:"High Risk",    value:k.high_risk||0,      display:fmtNumber(k.high_risk||0),      color:"#fecaca", delta:arrowHTML(highRiskChange,true)+' <span class="small-muted">vs prev month</span>',spark:makeSparkline(sparkRisk,"#dc2626")},
    {title:"Medium Risk",  value:k.medium_risk||0,    display:fmtNumber(k.medium_risk||0),    color:"#fde68a", delta:"", spark:""},
    {title:"Low Risk",     value:k.low_risk||0,       display:fmtNumber(k.low_risk||0),       color:"#bbf7d0", delta:"", spark:""}
  ];
  const row = document.getElementById("kpiRow");
  row.innerHTML = "";
  cards.forEach(c => {
    const div = document.createElement("div");
    div.className = "kpi";
    div.style.background = `linear-gradient(135deg, ${c.color} 0%, #ffffff 100%)`;
    div.innerHTML = `
      <div class="label">${c.title}</div>
      <div class="value kpi-value" data-target="${c.value}" data-decimals="${String(c.value).includes('.') ? 2 : 0}">${c.display}</div>
      <div style="margin-top:6px; min-height:18px;">${c.delta || ""}</div>
      ${c.spark}
    `;
    row.appendChild(div);
  });
  document.querySelectorAll(".kpi-value").forEach(el => {
    const target = parseFloat(el.dataset.target || "0");
    const decimals = parseInt(el.dataset.decimals || "0");
    animateValue(el, 0, target, 800, decimals);
  });
}

function renderRiskPie(){
  const risk = currentData.risk || {labels:[], values:[]};
  Plotly.react("riskPie", [{type:"pie", labels:risk.labels, values:risk.values, hole:0.5,
    marker:{colors:["#ef4444","#f59e0b","#10b981"]}}],
    {title:"Risk Category Distribution", paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}},
    {responsive:true});
}

function renderHist(target, data, field, title, color){
  Plotly.react(target, [{type:"histogram", x:data.map(d => d[field] || 0), marker:{color:color}}],
    {title, paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}}, {responsive:true});
}

function renderBubble(){
  const data = currentData.points || [];
  Plotly.react("bubbleScatter", [{
    type:"scatter", mode:"markers",
    x:data.map(d => d.saidi || 0), y:data.map(d => d.saifi || 0),
    text:data.map(d => d.utility_name || "Unknown"),
    marker:{
      size:data.map(d => Math.max(6, Math.min(20, Math.sqrt((d.damage||0)+1)/70))),
      color:data.map(d => d.risk_category==="High Risk"?"#ef4444":d.risk_category==="Medium Risk"?"#f59e0b":"#10b981"),
      opacity:0.7
    }
  }], {title:"Dynamic Bubble Chart: SAIDI vs SAIFI", xaxis:{title:"SAIDI"}, yaxis:{title:"SAIFI"},
    paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}}, {responsive:true});
}

function renderTopEvents(){
  const data = currentData.events || [];
  const plotData = [...data].slice(0,10).reverse();
  Plotly.react("eventBar", [{type:"bar", x:plotData.map(d=>d.count), y:plotData.map(d=>d.event_type),
    orientation:"h", marker:{color:plotData.map(d=>d.count), colorscale:"Blues"}}],
    {title:"Top Event Types", paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}},{responsive:true});
}

function renderMap(){
  const states = currentData.states || [];
  const mapMetric = document.getElementById("mapMetric").value;
  const mapColor  = document.getElementById("mapColor").value;
  Plotly.react("stateMap", [{type:"choropleth", locationmode:"USA-states",
    locations:states.map(d=>d.state), z:states.map(d=>d[mapMetric]),
    colorscale:mapColor, marker:{line:{color:"white",width:0.5}}, colorbar:{title:mapMetric}}],
    {title:"US Dynamic State Map", geo:{scope:"usa",bgcolor:"#111827"},
      paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}},{responsive:true});
  const sorted = [...states].sort((a,b)=>(b[mapMetric]||0)-(a[mapMetric]||0)).slice(0,15).reverse();
  Plotly.react("stateBar", [{type:"bar", x:sorted.map(d=>d[mapMetric]), y:sorted.map(d=>d.state),
    orientation:"h", marker:{color:sorted.map(d=>d[mapMetric]), colorscale:mapColor}}],
    {title:"Top States by Selected Metric", paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}},{responsive:true});
}

function renderAnimatedMap(){
  const data = currentData.state_month || [];
  const mapMetric = document.getElementById("mapMetric").value;
  const mapColor  = document.getElementById("mapColor").value;
  Plotly.react("animatedMap", [{type:"choropleth", locationmode:"USA-states",
    locations:data.map(d=>d.state), z:data.map(d=>d[mapMetric]||0), text:data.map(d=>"Month: "+d.month),
    colorscale:mapColor, marker:{line:{color:"white",width:0.5}}}],
    {title:"Animated Monthly State Map", geo:{scope:"usa",bgcolor:"#111827"},
      paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}, updatemenus:[], sliders:[]},{responsive:true});
  const months = [...new Set(data.map(d=>d.month))].sort((a,b)=>a-b);
  const frames = months.map(m => {
    const subset = data.filter(d=>d.month===m);
    return {name:String(m), data:[{type:"choropleth", locationmode:"USA-states",
      locations:subset.map(d=>d.state), z:subset.map(d=>d[mapMetric]||0),
      text:subset.map(d=>"Month: "+d.month), colorscale:mapColor, marker:{line:{color:"white",width:0.5}}}]};
  });
  Plotly.addFrames("animatedMap", frames);
  Plotly.relayout("animatedMap", {
    updatemenus:[{type:"buttons", showactive:false, x:0.05, y:1.15,
      buttons:[{label:"Play", method:"animate",
        args:[null, {fromcurrent:true, frame:{duration:900,redraw:true}, transition:{duration:300}}]}]}],
    sliders:[{active:0, currentvalue:{prefix:"Month: "},
      steps:months.map(m=>({label:String(m), method:"animate",
        args:[[String(m)],{mode:"immediate",frame:{duration:500,redraw:true},transition:{duration:200}}]}))}]
  });
}

function renderMonthly(){
  const data = currentData.months || [];
  Plotly.react("trendLine", [
    {type:"scatter",mode:"lines+markers",x:data.map(d=>d.month),y:data.map(d=>d.avg_saidi),name:"Avg SAIDI"},
    {type:"scatter",mode:"lines+markers",x:data.map(d=>d.month),y:data.map(d=>d.avg_saifi),name:"Avg SAIFI"}
  ], {title:"Monthly SAIDI vs SAIFI Trend", paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}},{responsive:true});
  Plotly.react("damageMonthBar", [{type:"bar", x:data.map(d=>d.month), y:data.map(d=>d.total_damage),
    marker:{color:data.map(d=>d.total_damage), colorscale:"Purples"}}],
    {title:"Monthly Total Damage", paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"white"}},{responsive:true});
}

function renderTopUtilities(){
  const data = currentData.top_utilities || [];
  const body = document.getElementById("topUtilitiesBody");
  body.innerHTML = "";
  data.forEach(d => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${d.utility_number||""}</td><td>${d.utility_name||""}</td><td>${d.state||""}</td>
      <td>${d.ownership||""}</td><td>${d.event_type||""}</td>
      <td>${(d.saidi||0).toFixed(2)}</td><td>${(d.saifi||0).toFixed(2)}</td>
      <td>${Math.round(d.damage||0).toLocaleString()}</td><td>${d.risk_category||""}</td>
    `;
    body.appendChild(row);
  });
}

function getRiskColor(risk){
  if (risk==="High Risk") return "#ef4444";
  if (risk==="Medium Risk") return "#f59e0b";
  return "#10b981";
}

function setLeafletMode(mode){
  leafletMode = mode;
  ["clusterBtn","bubbleBtn","riskBtn"].forEach(id => document.getElementById(id).classList.remove("active"));
  if (mode==="cluster") document.getElementById("clusterBtn").classList.add("active");
  if (mode==="bubble")  document.getElementById("bubbleBtn").classList.add("active");
  if (mode==="risk")    document.getElementById("riskBtn").classList.add("active");
  renderLeafletMap();
}

function initLeaflet(){
  if (leafletMap) return;
  leafletMap = L.map("leafletMap").setView([39.5,-98.35],4);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    {attribution:"&copy; OpenStreetMap &copy; CARTO", subdomains:"abcd", maxZoom:19}).addTo(leafletMap);
}

function renderLeafletMap(){
  initLeaflet();
  if (leafletLayer){ leafletMap.removeLayer(leafletLayer); leafletLayer = null; }
  const data = currentData.points || [];

  if (leafletMode==="cluster"){
    const grp = L.markerClusterGroup();
    data.forEach(d => {
      if (d.lat==null||d.lon==null) return;
      const m = L.circleMarker([d.lat,d.lon],{radius:6, color:getRiskColor(d.risk_category),
        fillColor:getRiskColor(d.risk_category), fillOpacity:0.75, weight:1});
      m.bindPopup(`<b>${d.utility_name||"Unknown"}</b><br>State: ${d.state||""}<br>Event: ${d.event_type||""}<br>
        Risk: ${d.risk_category||""}<br>SAIDI: ${(d.saidi||0).toFixed(2)}<br>SAIFI: ${(d.saifi||0).toFixed(2)}<br>
        Damage: ${Math.round(d.damage||0).toLocaleString()}`);
      grp.addLayer(m);
    });
    leafletLayer = grp;
  }
  if (leafletMode==="bubble"){
    const grp = L.layerGroup();
    data.forEach(d => {
      if (d.lat==null||d.lon==null) return;
      const radius = Math.max(5,Math.min(18,Math.sqrt((d.damage||0)+1)/90));
      const m = L.circleMarker([d.lat,d.lon],{radius, color:"#2563eb", fillColor:"#60a5fa", fillOpacity:0.45, weight:1});
      m.bindPopup(`<b>${d.utility_name||"Unknown"}</b><br>Damage: ${Math.round(d.damage||0).toLocaleString()}`);
      grp.addLayer(m);
    });
    leafletLayer = grp;
  }
  if (leafletMode==="risk"){
    const grp = L.layerGroup();
    data.forEach(d => {
      if (d.lat==null||d.lon==null) return;
      const m = L.circleMarker([d.lat,d.lon],{radius:7, color:getRiskColor(d.risk_category),
        fillColor:getRiskColor(d.risk_category), fillOpacity:0.7, weight:1});
      m.bindPopup(`<b>${d.utility_name||"Unknown"}</b><br>Risk: ${d.risk_category||""}<br>Score: ${(d.risk_score||0).toFixed(4)}`);
      grp.addLayer(m);
    });
    leafletLayer = grp;
  }
  leafletLayer.addTo(leafletMap);
}

initDashboard();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/init")
def init():
    return jsonify({
        "states": STATE_OPTIONS,
        "ownerships": OWNERSHIP_OPTIONS,
        "events": EVENT_OPTIONS,
        "risks": RISK_OPTIONS,
    })

@app.route("/data")
def data():
    state     = request.args.get("state", "All")
    ownership = request.args.get("ownership", "All")
    event     = request.args.get("event", "All")
    risk      = request.args.get("risk", "All")
    month_min = int(request.args.get("month_min", 1))
    month_max = int(request.args.get("month_max", 12))
    top_n     = int(request.args.get("top_n", TOP_N_EVENTS))
    payload = get_payload_cached(state, ownership, event, risk, month_min, month_max, top_n)
    return jsonify(payload)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
