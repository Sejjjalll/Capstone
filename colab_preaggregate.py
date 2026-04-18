"""
Run this cell in your Colab notebook BEFORE deploying.
It reads your full dataset and saves tiny pre-aggregated summary files.
Total output size: ~1-2 MB instead of hundreds of MB.
"""

import os
import gc
import numpy as np
import pandas as pd

DATA_PATH   = "dashboard_clean_output/dashboard_clean_dataset.parquet"
OUTPUT_DIR  = "dashboard_prebuilt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POINT_SAMPLE = 2000   # number of individual points for scatter / map

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded: {df.shape}")

# ── Standardise key columns ──────────────────
if "State" in df.columns:
    df["State"] = df["State"].astype(str).str.upper().str.strip()

for c in ["IEEE_AllEvents_SAIDI_min_per_yr","IEEE_AllEvents_SAIFI_times_per_yr",
          "IEEE_AllEvents_CAIDI_min_per_interruption","total_damage_usd",
          "log_total_damage","total_injuries","total_deaths","human_impact_score","risk_score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "BEGIN_MONTH" in df.columns:
    df["BEGIN_MONTH"] = pd.to_numeric(df["BEGIN_MONTH"], errors="coerce")

for c in ["State","Ownership","EVENT_TYPE","risk_category"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan","None","NULL",""]), c] = np.nan

# ── lat / lon ─────────────────────────────────
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
rng = np.random.default_rng(42)
df["lat"] = df["State"].map(lambda s: STATE_GEO.get(s,[39.5,-98.35])[0])
df["lon"] = df["State"].map(lambda s: STATE_GEO.get(s,[39.5,-98.35])[1])
df["lat"] += rng.uniform(-1.6, 1.6, len(df))
df["lon"] += rng.uniform(-1.6, 1.6, len(df))

print(f"Rows after prep: {len(df):,}")

# ── Helper ────────────────────────────────────
def safe_agg(df_, by, agg_dict):
    return df_.groupby(by, observed=True).agg(**agg_dict).reset_index()

# ─────────────────────────────────────────────
# 1. FILTER OPTIONS
# ─────────────────────────────────────────────
opts = {
    "states":     ["All"] + sorted(df["State"].dropna().unique().tolist()),
    "ownerships": ["All"] + sorted(df["Ownership"].dropna().unique().tolist()) if "Ownership" in df.columns else ["All"],
    "events":     ["All"] + sorted(df["EVENT_TYPE"].dropna().unique().tolist()) if "EVENT_TYPE" in df.columns else ["All"],
    "risks":      ["All","High Risk","Medium Risk","Low Risk"],
}
pd.Series(opts).to_json(f"{OUTPUT_DIR}/filter_options.json")
print("✓ filter_options.json")

# ─────────────────────────────────────────────
# 2. KPI SUMMARY  (full dataset)
# ─────────────────────────────────────────────
def make_kpi(d):
    def rc(r): return int((d["risk_category"].astype(str)==r).sum()) if "risk_category" in d.columns else 0
    return {
        "total_rows":      int(len(d)),
        "total_utilities": int(d["Utility Number"].nunique()) if "Utility Number" in d.columns else 0,
        "total_states":    int(d["State"].nunique()) if "State" in d.columns else 0,
        "avg_saidi":  float(d["IEEE_AllEvents_SAIDI_min_per_yr"].mean())  if "IEEE_AllEvents_SAIDI_min_per_yr" in d.columns else 0,
        "avg_saifi":  float(d["IEEE_AllEvents_SAIFI_times_per_yr"].mean()) if "IEEE_AllEvents_SAIFI_times_per_yr" in d.columns else 0,
        "avg_caidi":  float(d["IEEE_AllEvents_CAIDI_min_per_interruption"].mean()) if "IEEE_AllEvents_CAIDI_min_per_interruption" in d.columns else 0,
        "total_damage":   float(d["total_damage_usd"].sum())  if "total_damage_usd" in d.columns else 0,
        "total_injuries": float(d["total_injuries"].sum())    if "total_injuries" in d.columns else 0,
        "total_deaths":   float(d["total_deaths"].sum())      if "total_deaths" in d.columns else 0,
        "high_risk":   rc("High Risk"),
        "medium_risk": rc("Medium Risk"),
        "low_risk":    rc("Low Risk"),
    }

import json, math
def clean_val(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return 0
    return v

kpi_all = {k: clean_val(v) for k,v in make_kpi(df).items()}
with open(f"{OUTPUT_DIR}/kpi_all.json","w") as f:
    json.dump(kpi_all, f)
print("✓ kpi_all.json")

# ─────────────────────────────────────────────
# 3. STATE AGGREGATION
# ─────────────────────────────────────────────
agg = {
    "avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
    "avg_saifi":("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
    "avg_caidi":("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
    "total_damage":("total_damage_usd","sum"),
    "total_injuries":("total_injuries","sum"),
    "total_deaths":("total_deaths","sum"),
}
if "Utility Number" in df.columns: agg["utility_count"]=("Utility Number","nunique")
if "risk_category" in df.columns:  agg["high_risk_count"]=("risk_category",lambda s:(s.astype(str)=="High Risk").sum())

state_df = safe_agg(df,"State",agg).rename(columns={"State":"state"})
state_df.fillna(0).to_json(f"{OUTPUT_DIR}/states.json", orient="records")
print(f"✓ states.json  ({len(state_df)} rows)")

# ─────────────────────────────────────────────
# 4. MONTHLY AGGREGATION
# ─────────────────────────────────────────────
if "BEGIN_MONTH" in df.columns:
    agg_m = {
        "avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
        "avg_saifi":("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
        "avg_caidi":("IEEE_AllEvents_CAIDI_min_per_interruption","mean"),
        "total_damage":("total_damage_usd","sum"),
        "total_injuries":("total_injuries","sum"),
        "total_deaths":("total_deaths","sum"),
    }
    if "risk_category" in df.columns: agg_m["high_risk_count"]=("risk_category",lambda s:(s.astype(str)=="High Risk").sum())
    month_df = safe_agg(df,"BEGIN_MONTH",agg_m).rename(columns={"BEGIN_MONTH":"month"}).sort_values("month")
    month_df.fillna(0).to_json(f"{OUTPUT_DIR}/months.json", orient="records")
    print(f"✓ months.json  ({len(month_df)} rows)")

# ─────────────────────────────────────────────
# 5. EVENT TYPE AGGREGATION
# ─────────────────────────────────────────────
if "EVENT_TYPE" in df.columns:
    agg_e = {
        "count":("EVENT_TYPE","size"),
        "avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
        "total_damage":("total_damage_usd","sum"),
        "total_injuries":("total_injuries","sum"),
        "total_deaths":("total_deaths","sum"),
    }
    event_df = safe_agg(df,"EVENT_TYPE",agg_e).rename(columns={"EVENT_TYPE":"event_type"}).sort_values("count",ascending=False)
    event_df.fillna(0).to_json(f"{OUTPUT_DIR}/events.json", orient="records")
    print(f"✓ events.json  ({len(event_df)} rows)")

# ─────────────────────────────────────────────
# 6. STATE × MONTH
# ─────────────────────────────────────────────
if "BEGIN_MONTH" in df.columns and "State" in df.columns:
    agg_sm = {
        "avg_saidi":("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
        "avg_saifi":("IEEE_AllEvents_SAIFI_times_per_yr","mean"),
        "total_damage":("total_damage_usd","sum"),
        "total_injuries":("total_injuries","sum"),
        "total_deaths":("total_deaths","sum"),
    }
    if "risk_category" in df.columns: agg_sm["high_risk_count"]=("risk_category",lambda s:(s.astype(str)=="High Risk").sum())
    if "Utility Number" in df.columns: agg_sm["utility_count"]=("Utility Number","nunique")
    sm_df = safe_agg(df,["BEGIN_MONTH","State"],agg_sm).rename(columns={"BEGIN_MONTH":"month","State":"state"})
    sm_df.fillna(0).to_json(f"{OUTPUT_DIR}/state_month.json", orient="records")
    print(f"✓ state_month.json  ({len(sm_df)} rows)")

# ─────────────────────────────────────────────
# 7. RISK DISTRIBUTION
# ─────────────────────────────────────────────
if "risk_category" in df.columns:
    rc = df["risk_category"].astype(str).value_counts()
    risk_data = {"labels": rc.index.tolist(), "values": [int(v) for v in rc.values]}
    with open(f"{OUTPUT_DIR}/risk.json","w") as f:
        json.dump(risk_data, f)
    print("✓ risk.json")

# ─────────────────────────────────────────────
# 8. POINT SAMPLE  (for scatter + leaflet maps)
# ─────────────────────────────────────────────
point_cols = [c for c in [
    "State","Ownership","EVENT_TYPE","risk_category","risk_score",
    "Utility Number","Utility Name","BEGIN_MONTH",
    "IEEE_AllEvents_SAIDI_min_per_yr","IEEE_AllEvents_SAIFI_times_per_yr",
    "IEEE_AllEvents_CAIDI_min_per_interruption",
    "total_damage_usd","log_total_damage","total_injuries","total_deaths",
    "human_impact_score","MAGNITUDE","lat","lon"
] if c in df.columns]

points = df[point_cols].sample(min(POINT_SAMPLE, len(df)), random_state=42).copy()
points = points.rename(columns={
    "Utility Name":"utility_name","Utility Number":"utility_number",
    "State":"state","Ownership":"ownership","EVENT_TYPE":"event_type",
    "risk_category":"risk_category","risk_score":"risk_score",
    "IEEE_AllEvents_SAIDI_min_per_yr":"saidi","IEEE_AllEvents_SAIFI_times_per_yr":"saifi",
    "IEEE_AllEvents_CAIDI_min_per_interruption":"caidi","total_damage_usd":"damage",
    "log_total_damage":"log_damage","total_injuries":"injuries","total_deaths":"deaths",
    "human_impact_score":"impact","BEGIN_MONTH":"month","MAGNITUDE":"magnitude",
})
points.fillna(0).to_json(f"{OUTPUT_DIR}/points.json", orient="records")
print(f"✓ points.json  ({len(points)} rows)")

# ─────────────────────────────────────────────
# 9. TOP UTILITIES
# ─────────────────────────────────────────────
if "risk_score" in df.columns:
    top_util = df.sort_values("risk_score", ascending=False).head(50).copy()
else:
    top_util = df.head(50).copy()

top_util = top_util[point_cols].rename(columns={
    "Utility Name":"utility_name","Utility Number":"utility_number",
    "State":"state","Ownership":"ownership","EVENT_TYPE":"event_type",
    "risk_category":"risk_category","risk_score":"risk_score",
    "IEEE_AllEvents_SAIDI_min_per_yr":"saidi","IEEE_AllEvents_SAIFI_times_per_yr":"saifi",
    "IEEE_AllEvents_CAIDI_min_per_interruption":"caidi","total_damage_usd":"damage",
    "log_total_damage":"log_damage","total_injuries":"injuries","total_deaths":"deaths",
    "human_impact_score":"impact","BEGIN_MONTH":"month","MAGNITUDE":"magnitude",
})
top_util.fillna(0).to_json(f"{OUTPUT_DIR}/top_utilities.json", orient="records")
print(f"✓ top_utilities.json  ({len(top_util)} rows)")

# ─────────────────────────────────────────────
# 10. INSIGHTS
# ─────────────────────────────────────────────
insights = {"high_risk_records": kpi_all["high_risk"],
            "top_state_saidi": None, "top_state_saidi_value": None,
            "top_event_damage": None, "top_event_damage_value": None}

if "State" in df.columns and "IEEE_AllEvents_SAIDI_min_per_yr" in df.columns:
    tmp = df.groupby("State",observed=True)["IEEE_AllEvents_SAIDI_min_per_yr"].mean().sort_values(ascending=False)
    insights["top_state_saidi"] = str(tmp.index[0])
    insights["top_state_saidi_value"] = float(tmp.iloc[0])

if "EVENT_TYPE" in df.columns and "total_damage_usd" in df.columns:
    tmp = df.groupby("EVENT_TYPE",observed=True)["total_damage_usd"].sum().sort_values(ascending=False)
    insights["top_event_damage"] = str(tmp.index[0])
    insights["top_event_damage_value"] = float(tmp.iloc[0])

with open(f"{OUTPUT_DIR}/insights.json","w") as f:
    json.dump(insights, f)
print("✓ insights.json")

# ─────────────────────────────────────────────
# CHECK SIZES
# ─────────────────────────────────────────────
print("\n── File sizes ──")
total = 0
for fn in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f"{OUTPUT_DIR}/{fn}")
    total += size
    print(f"  {fn:<30} {size/1024:.1f} KB")
print(f"\n  TOTAL: {total/1024:.1f} KB  ({total/1e6:.2f} MB)")

# ─────────────────────────────────────────────
# DOWNLOAD ALL AS ZIP
# ─────────────────────────────────────────────
import shutil
shutil.make_archive("dashboard_prebuilt", "zip", OUTPUT_DIR)
print("\n✅ Done! Downloading dashboard_prebuilt.zip ...")

from google.colab import files
files.download("dashboard_prebuilt.zip")
