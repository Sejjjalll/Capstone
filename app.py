"""
Power Outage Risk Dashboard - Flask Server (Memory-Optimized for Render Free Tier)
"""

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
DATA_PATH         = os.environ.get("DATA_PATH", "dashboard_clean_dataset.parquet")
TOP_N_EVENTS      = 10
TOP_N_UTILITIES   = 20
POINT_SAMPLE_SIZE = 1500   # reduced from 3000 to save RAM

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

# Only load the columns the dashboard actually needs — saves ~60% RAM
KEEP_COLS = [
    "State", "Ownership", "EVENT_TYPE", "risk_category", "risk_score",
    "Utility Number", "Utility Name", "BEGIN_MONTH",
    "IEEE_AllEvents_SAIDI_min_per_yr",
    "IEEE_AllEvents_SAIFI_times_per_yr",
    "IEEE_AllEvents_CAIDI_min_per_interruption",
    "total_damage_usd", "log_total_damage",
    "total_injuries", "total_deaths", "human_impact_score",
    "MAGNITUDE",
]

# ─────────────────────────────────────────────
# DATA LOADING  (runs once at startup)
# ─────────────────────────────────────────────
def load_data():
    print(f"Loading data from: {DATA_PATH}")

    if DATA_PATH.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(DATA_PATH)
        available = pf.schema_arrow.names
        cols = [c for c in KEEP_COLS if c in available]
        df = pf.read(columns=cols).to_pandas()
    else:
        sample = pd.read_csv(DATA_PATH, nrows=0)
        cols = [c for c in KEEP_COLS if c in sample.columns]
        df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)

    # State filter
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.upper().str.strip()
        df = df[df["State"].isin(VALID_US_STATES)].copy()

    # Downcast numerics to float32 (halves RAM vs float64)
    float_cols = [
        "IEEE_AllEvents_SAIDI_min_per_yr","IEEE_AllEvents_SAIFI_times_per_yr",
        "IEEE_AllEvents_CAIDI_min_per_interruption","total_damage_usd",
        "log_total_damage","total_injuries","total_deaths",
        "human_impact_score","risk_score","MAGNITUDE",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    if "BEGIN_MONTH" in df.columns:
        df["BEGIN_MONTH"] = pd.to_numeric(df["BEGIN_MONTH"], errors="coerce").astype("Int8")
    if "Utility Number" in df.columns:
        df["Utility Number"] = pd.to_numeric(df["Utility Number"], errors="coerce").astype("Int32")

    # String → category (massive RAM saving)
    for c in ["State","Ownership","EVENT_TYPE","risk_category","Utility Name"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan","None","NULL",""]), c] = pd.NA
            df[c] = df[c].astype("category")

    # lat/lon using state centroids + fixed jitter
    rng = np.random.default_rng(42)
    df["lat"] = df["State"].astype(str).map(lambda s: STATE_GEO.get(s,[39.5,-98.35])[0]).astype("float32")
    df["lon"] = df["State"].astype(str).map(lambda s: STATE_GEO.get(s,[39.5,-98.35])[1]).astype("float32")
    df["lat"] += rng.uniform(-1.6, 1.6, len(df)).astype("float32")
    df["lon"] += rng.uniform(-1.6, 1.6, len(df)).astype("float32")

    gc.collect()
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"Loaded {len(df):,} rows | {df.shape[1]} cols | {mem_mb:.1f} MB RAM")
    return df


df_dash = load_data()

STATE_OPTIONS     = ["All"] + sorted(df_dash["State"].dropna().astype(str).unique().tolist())
OWNERSHIP_OPTIONS = (["All"] + sorted(df_dash["Ownership"].dropna().astype(str).unique().tolist())
                     if "Ownership" in df_dash.columns else ["All"])
EVENT_OPTIONS     = (["All"] + sorted(df_dash["EVENT_TYPE"].dropna().astype(str).unique().tolist())
                     if "EVENT_TYPE" in df_dash.columns else ["All"])
RISK_OPTIONS      = ["All","High Risk","Medium Risk","Low Risk"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _col(df, name): return name in df.columns

def safe_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
    return float(v)

def safe_records(df_):
    if df_.empty: return []
    for c in df_.select_dtypes(["category"]).columns:
        df_[c] = df_[c].astype(str)
    return json.loads(df_.fillna(0).to_json(orient="records"))

def get_filtered_df(state, ownership, event, risk, month_min, month_max):
    mask = pd.Series(True, index=df_dash.index)
    if state != "All" and _col(df_dash,"State"):
        mask &= df_dash["State"].astype(str) == state
    if ownership != "All" and _col(df_dash,"Ownership"):
        mask &= df_dash["Ownership"].astype(str) == ownership
    if event != "All" and _col(df_dash,"EVENT_TYPE"):
        mask &= df_dash["EVENT_TYPE"].astype(str) == event
    if risk != "All" and _col(df_dash,"risk_category"):
        mask &= df_dash["risk_category"].astype(str) == risk
    if _col(df_dash,"BEGIN_MONTH"):
        mask &= df_dash["BEGIN_MONTH"].between(month_min, month_max)
    return df_dash.loc[mask]

def make_payload(fdf, top_n=TOP_N_EVENTS):
    if fdf.empty:
        return {"kpi":{},"insights":{},"risk":{"labels":[],"values":[]},
                "states":[],"months":[],"events":[],"top_utilities":[],"points":[],"state_month":[]}

    point_df = (fdf.sample(POINT_SAMPLE_SIZE, random_state=42)
                if len(fdf) > POINT_SAMPLE_SIZE else fdf).copy()

    def cm(c): return safe_float(fdf[c].mean()) if _col(fdf,c) else 0.0
    def cs(c): return safe_float(fdf[c].sum())  if _col(fdf,c) else 0.0
    def rc(r): return int((fdf["risk_category"].astype(str)==r).sum()) if _col(fdf,"risk_category") else 0

    kpi = {
        "total_rows":int(len(fdf)),
        "total_utilities":int(fdf["Utility Number"].nunique()) if _col(fdf,"Utility Number") else 0,
        "total_states":int(fdf["State"].nunique()) if _col(fdf,"State") else 0,
        "avg_saidi":cm("IEEE_AllEvents_SAIDI_min_per_yr"),
        "avg_saifi":cm("IEEE_AllEvents_SAIFI_times_per_yr"),
        "avg_caidi":cm("IEEE_AllEvents_CAIDI_min_per_interruption"),
        "total_damage":cs("total_damage_usd"),
        "total_injuries":cs("total_injuries"),
        "total_deaths":cs("total_deaths"),
        "high_risk":rc("High Risk"),"medium_risk":rc("Medium Risk"),"low_risk":rc("Low Risk"),
    }

    insights = {"high_risk_records":kpi["high_risk"],
                "top_state_saidi":None,"top_state_saidi_value":None,
                "top_event_damage":None,"top_event_damage_value":None}
    if _col(fdf,"State") and _col(fdf,"IEEE_AllEvents_SAIDI_min_per_yr"):
        tmp = fdf.groupby("State",observed=True)["IEEE_AllEvents_SAIDI_min_per_yr"].mean().sort_values(ascending=False)
        if len(tmp): insights["top_state_saidi"]=str(tmp.index[0]); insights["top_state_saidi_value"]=safe_float(tmp.iloc[0])
    if _col(fdf,"EVENT_TYPE") and _col(fdf,"total_damage_usd"):
        tmp = fdf.groupby("EVENT_TYPE",observed=True)["total_damage_usd"].sum().sort_values(ascending=False)
        if len(tmp): insights["top_event_damage"]=str(tmp.index[0]); insights["top_event_damage_value"]=safe_float(tmp.iloc[0])

    rvc = fdf["risk_category"].astype(str).value_counts() if _col(fdf,"risk_category") else pd.Series(dtype=int)
    risk_pl = {"labels":rvc.index.tolist(),"values":[int(v) for v in rvc.values]}

    def grp_state():
        if not _col(fdf,"State"): return pd.DataFrame()
        agg = {"avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
               "avg_saifi":("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
               "avg_caidi":("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
               "total_damage":("total_damage_usd","sum"),
               "total_injuries":("total_injuries","sum"),
               "total_deaths":("total_deaths","sum")}
        if _col(fdf,"Utility Number"): agg["utility_count"]=("Utility Number","nunique")
        if _col(fdf,"risk_category"):  agg["high_risk_count"]=("risk_category",lambda s:(s.astype(str)=="High Risk").sum())
        return fdf.groupby("State",observed=True).agg(**agg).reset_index().rename(columns={"State":"state"})

    def grp_month():
        if not _col(fdf,"BEGIN_MONTH"): return pd.DataFrame()
        agg = {"avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
               "avg_saifi":("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
               "avg_caidi":("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
               "total_damage":("total_damage_usd","sum"),
               "total_injuries":("total_injuries","sum"),
               "total_deaths":("total_deaths","sum")}
        if _col(fdf,"risk_category"): agg["high_risk_count"]=("risk_category",lambda s:(s.astype(str)=="High Risk").sum())
        return fdf.groupby("BEGIN_MONTH",observed=True).agg(**agg).reset_index().rename(columns={"BEGIN_MONTH":"month"}).sort_values("month")

    def grp_event():
        if not _col(fdf,"EVENT_TYPE"): return pd.DataFrame()
        agg = {"count":("EVENT_TYPE","size"),
               "avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
               "total_damage":("total_damage_usd","sum"),
               "total_injuries":("total_injuries","sum"),
               "total_deaths":("total_deaths","sum")}
        return fdf.groupby("EVENT_TYPE",observed=True).agg(**agg).reset_index().sort_values("count",ascending=False).head(top_n).rename(columns={"EVENT_TYPE":"event_type"})

    def grp_state_month():
        if not (_col(fdf,"BEGIN_MONTH") and _col(fdf,"State")): return pd.DataFrame()
        agg = {"avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
               "avg_saifi":("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
               "total_damage":("total_damage_usd","sum"),
               "total_injuries":("total_injuries","sum"),
               "total_deaths":("total_deaths","sum")}
        if _col(fdf,"risk_category"): agg["high_risk_count"]=("risk_category",lambda s:(s.astype(str)=="High Risk").sum())
        if _col(fdf,"Utility Number"): agg["utility_count"]=("Utility Number","nunique")
        return fdf.groupby(["BEGIN_MONTH","State"],observed=True).agg(**agg).reset_index().rename(columns={"BEGIN_MONTH":"month","State":"state"})

    sort_col = "risk_score" if _col(fdf,"risk_score") else None
    top_util = (fdf.sort_values(sort_col,ascending=False).head(TOP_N_UTILITIES) if sort_col else fdf.head(TOP_N_UTILITIES)).copy()

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
    point_df = point_df.rename(columns=rename_map)
    top_util  = top_util.rename(columns=rename_map)

    gc.collect()
    return {
        "kpi":kpi,"insights":insights,"risk":risk_pl,
        "states":safe_records(grp_state()),
        "months":safe_records(grp_month()),
        "events":safe_records(grp_event()),
        "top_utilities":safe_records(top_util),
        "points":safe_records(point_df),
        "state_month":safe_records(grp_state_month()),
    }

_cache = {}
def get_payload_cached(state,ownership,event,risk,month_min,month_max,top_n):
    key=(state,ownership,event,risk,month_min,month_max,top_n)
    if key not in _cache:
        if len(_cache)>50: _cache.clear()
        _cache[key]=make_payload(get_filtered_df(state,ownership,event,risk,month_min,month_max),top_n=top_n)
    return _cache[key]

# ─────────────────────────────────────────────
# HTML TEMPLATE
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
  :root{--bg:#202124;--panel:#111827;--border:#374151;--text:#ffffff;--muted:#d1d5db;--blue:#2196f3;}
  *{box-sizing:border-box}
  body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--text)}
  .container{padding:18px}
  .title{font-size:34px;font-weight:700;margin-bottom:6px}
  .subtitle{color:var(--muted);margin-bottom:18px;font-size:16px}
  .filters{display:grid;grid-template-columns:1fr 1fr;gap:16px;background:linear-gradient(135deg,#07142f,#091b42);border:1px solid var(--border);border-radius:16px;padding:16px;margin-bottom:18px;box-shadow:0 6px 20px rgba(0,0,0,.18)}
  .filter-col{display:flex;flex-direction:column;gap:12px}
  .filter-group label{display:block;font-size:13px;margin-bottom:4px;color:var(--muted)}
  .filter-group select,.filter-group input{width:100%;padding:10px;border-radius:12px;border:1px solid #4b5563;background:#374151;color:white}
  .btn{padding:11px 16px;background:var(--blue);color:white;border:none;border-radius:12px;cursor:pointer;font-weight:700;box-shadow:0 4px 12px rgba(33,150,243,.25)}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
  .kpi{border-radius:18px;padding:16px;color:#111827;box-shadow:0 8px 18px rgba(0,0,0,.10);transition:transform .2s ease,box-shadow .2s ease}
  .kpi:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(0,0,0,.16)}
  .kpi .label{font-size:14px;color:#374151;margin-bottom:8px}
  .kpi .value{font-size:24px;font-weight:800}
  .insight{background:linear-gradient(135deg,#07142f,#091b42);border:1px solid var(--border);border-left:6px solid #3b82f6;border-radius:14px;padding:16px;margin-bottom:18px}
  .insight-title{font-size:18px;font-weight:700;margin-bottom:8px}
  .section{margin-top:18px;margin-bottom:10px;font-size:28px;font-weight:700}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:10px;box-shadow:0 6px 20px rgba(0,0,0,.15)}
  .plot{width:100%;height:430px}.plot-tall{width:100%;height:620px}.plot-map{width:100%;height:700px}
  .leaflet-map{width:100%;height:720px;border-radius:12px;overflow:hidden}
  .map-toolbar{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}
  .map-tab{padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:#1f2937;color:white;cursor:pointer}
  .map-tab.active{background:#2563eb;border-color:#2563eb}
  .table-wrap{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:12px;overflow-x:auto}
  table{width:100%;border-collapse:collapse;color:white}
  th,td{padding:10px;border-bottom:1px solid var(--border);text-align:left;font-size:13px}
  th{color:#93c5fd}
  .legend-note{color:#cbd5e1;font-size:13px;margin-top:6px}
  .small-muted{font-size:12px;color:#6b7280}
  .sparkline{margin-top:8px}
  .loading{position:fixed;inset:0;background:rgba(0,0,0,0.45);display:none;align-items:center;justify-content:center;z-index:9999}
  .loading-box{background:#111827;color:white;padding:18px 24px;border-radius:14px;border:1px solid #374151;font-size:16px;box-shadow:0 8px 24px rgba(0,0,0,.25)}
  @media(max-width:1200px){.kpis{grid-template-columns:repeat(2,1fr)}.grid-2,.filters{grid-template-columns:1fr}}
</style>
</head>
<body>
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
          <option value="Reds">Reds</option><option value="Blues">Blues</option>
          <option value="Greens">Greens</option><option value="Purples">Purples</option>
          <option value="Oranges">Oranges</option><option value="Viridis">Viridis</option>
          <option value="Cividis">Cividis</option><option value="Turbo">Turbo</option>
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
    <div class="legend-note">Map uses state centroids with jitter if no exact coordinates are in your dataset.</div>
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
      <thead><tr><th>Utility Number</th><th>Utility Name</th><th>State</th><th>Ownership</th><th>Event</th><th>SAIDI</th><th>SAIFI</th><th>Damage</th><th>Risk</th></tr></thead>
      <tbody id="topUtilitiesBody"></tbody>
    </table>
  </div>
</div>
<script>
let currentData=null,leafletMap=null,leafletLayer=null,leafletMode="cluster";
function showLoading(s=true){document.getElementById("loadingOverlay").style.display=s?"flex":"none";}
function fillDropdown(id,vals){const el=document.getElementById(id);el.innerHTML="";vals.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;el.appendChild(o);});}
function fmtNumber(x){return Number(x||0).toLocaleString();}
function fmtFloat(x,d=2){return Number(x||0).toFixed(d);}
function pctChange(c,p){if(!p||p===0)return 0;return((c-p)/p)*100;}
function arrowHTML(ch,low=false){const good=low?ch<0:ch>0;const arr=ch>0?"▲":ch<0?"▼":"●";const col=ch===0?"#6b7280":(good?"#10b981":"#ef4444");return`<span style="color:${col};font-size:13px;font-weight:700;">${arr} ${Math.abs(ch).toFixed(1)}%</span>`;}
function animateValue(el,s,e,dur=800,dec=0){const t0=performance.now();function u(now){const p=Math.min((now-t0)/dur,1);const v=s+(e-s)*p;dec>0?el.textContent=v.toFixed(dec):el.textContent=Math.round(v).toLocaleString();p<1&&requestAnimationFrame(u);}requestAnimationFrame(u);}
function makeSparkline(vals,color="#3b82f6"){if(!vals||!vals.length)return"";const w=120,h=28,mn=Math.min(...vals),mx=Math.max(...vals),r=(mx-mn)||1;const pts=vals.map((v,i)=>`${(i/((vals.length-1)||1))*w},${h-((v-mn)/r)*h}`).join(" ");return`<svg class="sparkline" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/></svg>`;}
async function initDashboard(){showLoading(true);const res=await fetch("/init");const data=await res.json();fillDropdown("stateFilter",data.states);fillDropdown("ownershipFilter",data.ownerships);fillDropdown("eventFilter",data.events);fillDropdown("riskFilter",data.risks);await updateDashboard();}
function getFilters(){let mn=parseInt(document.getElementById("monthMin").value),mx=parseInt(document.getElementById("monthMax").value);if(mn>mx)[mn,mx]=[mx,mn];document.getElementById("monthLabel").textContent=`${mn} - ${mx}`;return{state:document.getElementById("stateFilter").value,ownership:document.getElementById("ownershipFilter").value,event:document.getElementById("eventFilter").value,risk:document.getElementById("riskFilter").value,month_min:mn,month_max:mx,top_n:10};}
async function updateDashboard(){showLoading(true);try{const f=getFilters();const qs=new URLSearchParams(f).toString();const res=await fetch("/data?"+qs);currentData=await res.json();renderInsights();renderKPIs();renderRiskPie();renderHist("saidiHist",currentData.points,"saidi","SAIDI Distribution","#2563eb");renderHist("saifiHist",currentData.points,"saifi","SAIFI Distribution","#f59e0b");renderHist("caidiHist",currentData.points,"caidi","CAIDI Distribution","#14b8a6");renderBubble();renderTopEvents();renderHist("damageHist",currentData.points,"log_damage","Log Total Damage","#7c3aed");renderMap();renderLeafletMap();renderAnimatedMap();renderMonthly();renderTopUtilities();}finally{showLoading(false);}}
function renderInsights(){const i=currentData.insights||{};document.getElementById("insightText").innerHTML=`• High-risk records: <b>${fmtNumber(i.high_risk_records||0)}</b><br>• Highest avg SAIDI state: <b>${i.top_state_saidi||"N/A"}</b> (${fmtFloat(i.top_state_saidi_value||0)})<br>• Most damaging event type: <b>${i.top_event_damage||"N/A"}</b> (${fmtNumber(i.top_event_damage_value||0)})`;}
function renderKPIs(){const k=currentData.kpi||{};const months=currentData.months||[];const pm=months.length>1?months[months.length-2]:null;const lm=months.length>0?months[months.length-1]:null;const sc=pm?pctChange(lm.avg_saidi,pm.avg_saidi):0,fc=pm?pctChange(lm.avg_saifi,pm.avg_saifi):0,dc=pm?pctChange(lm.total_damage,pm.total_damage):0,rc2=pm?pctChange(lm.high_risk_count,pm.high_risk_count):0;const ss=months.map(d=>d.avg_saidi||0),sf=months.map(d=>d.avg_saifi||0),sd=months.map(d=>d.total_damage||0),sr=months.map(d=>d.high_risk_count||0);const cards=[{title:"Rows",value:k.total_rows||0,display:fmtNumber(k.total_rows||0),color:"#dbeafe",delta:"",spark:""},{title:"Utilities",value:k.total_utilities||0,display:fmtNumber(k.total_utilities||0),color:"#ede9fe",delta:"",spark:""},{title:"States",value:k.total_states||0,display:fmtNumber(k.total_states||0),color:"#dcfce7",delta:"",spark:""},{title:"Avg SAIDI",value:k.avg_saidi||0,display:fmtFloat(k.avg_saidi||0),color:"#fee2e2",delta:arrowHTML(sc,true)+' <span class="small-muted">vs prev month</span>',spark:makeSparkline(ss,"#ef4444")},{title:"Avg SAIFI",value:k.avg_saifi||0,display:fmtFloat(k.avg_saifi||0),color:"#fef3c7",delta:arrowHTML(fc,true)+' <span class="small-muted">vs prev month</span>',spark:makeSparkline(sf,"#f59e0b")},{title:"Avg CAIDI",value:k.avg_caidi||0,display:fmtFloat(k.avg_caidi||0),color:"#e0f2fe",delta:"",spark:""},{title:"Total Damage",value:k.total_damage||0,display:fmtNumber(k.total_damage||0),color:"#fae8ff",delta:arrowHTML(dc,true)+' <span class="small-muted">vs prev month</span>',spark:makeSparkline(sd,"#8b5cf6")},{title:"Total Injuries",value:k.total_injuries||0,display:fmtNumber(k.total_injuries||0),color:"#ede9fe",delta:"",spark:""},{title:"Total Deaths",value:k.total_deaths||0,display:fmtNumber(k.total_deaths||0),color:"#fee2e2",delta:"",spark:""},{title:"High Risk",value:k.high_risk||0,display:fmtNumber(k.high_risk||0),color:"#fecaca",delta:arrowHTML(rc2,true)+' <span class="small-muted">vs prev month</span>',spark:makeSparkline(sr,"#dc2626")},{title:"Medium Risk",value:k.medium_risk||0,display:fmtNumber(k.medium_risk||0),color:"#fde68a",delta:"",spark:""},{title:"Low Risk",value:k.low_risk||0,display:fmtNumber(k.low_risk||0),color:"#bbf7d0",delta:"",spark:""}];const row=document.getElementById("kpiRow");row.innerHTML="";cards.forEach(c=>{const div=document.createElement("div");div.className="kpi";div.style.background=`linear-gradient(135deg,${c.color} 0%,#ffffff 100%)`;div.innerHTML=`<div class="label">${c.title}</div><div class="value kpi-value" data-target="${c.value}" data-decimals="${String(c.value).includes('.')?2:0}">${c.display}</div><div style="margin-top:6px;min-height:18px;">${c.delta||""}</div>${c.spark}`;row.appendChild(div);});document.querySelectorAll(".kpi-value").forEach(el=>{const t=parseFloat(el.dataset.target||"0");const d=parseInt(el.dataset.decimals||"0");animateValue(el,0,t,800,d);});}
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
</script>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/init")
def init():
    return jsonify({"states":STATE_OPTIONS,"ownerships":OWNERSHIP_OPTIONS,
                    "events":EVENT_OPTIONS,"risks":RISK_OPTIONS})

@app.route("/data")
def data():
    state     = request.args.get("state","All")
    ownership = request.args.get("ownership","All")
    event     = request.args.get("event","All")
    risk      = request.args.get("risk","All")
    month_min = int(request.args.get("month_min",1))
    month_max = int(request.args.get("month_max",12))
    top_n     = int(request.args.get("top_n",TOP_N_EVENTS))
    return jsonify(get_payload_cached(state,ownership,event,risk,month_min,month_max,top_n))

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8000))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0",port=port,debug=False)
