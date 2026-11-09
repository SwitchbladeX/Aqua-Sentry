from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Water Management AI", page_icon="🌊", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
VARIABLES = {
    "Dissolved oxygen": {"key": "do", "folder": "dissolved_oxygen", "prefix": "DO", "column": "DO_mgL", "unit": "mg/L", "code": "00300"},
    "Temperature": {"key": "temp", "folder": "temperature", "prefix": "Temp", "column": "temp_C", "unit": "°C", "code": "00010"},
    "pH": {"key": "ph", "folder": "ph", "prefix": "pH", "column": "pH", "unit": "pH", "code": "00400"},
    "Salinity": {"key": "salinity", "folder": "salinity", "prefix": "Salinity", "column": "salinity_PSU", "unit": "PSU", "code": "00480"},
}
VALUE_COLUMNS = [item["column"] for item in VARIABLES.values()]
LAGS = [1, 3, 6, 12]
FEATURE_COLUMNS = VALUE_COLUMNS + [f"{col}_lag{lag}h" for col in VALUE_COLUMNS for lag in LAGS] + ["hour", "month"]
HIERARCHIES = {"do": "DO_VALIDATION_HIERARCHY.csv", "temp": "TEMP_VALIDATION_HIERARCHY.csv", "ph": "PH_VALIDATION_HIERARCHY.csv", "salinity": "SALINITY_VALIDATION_HIERARCHY.csv"}
STATIONS = {
    "West Harbor at Oyster Bay, NY — 01303152": "01303152", "Peconic River at Riverhead, NY — 01304562": "01304562",
    "South side of Shelter Island, NY — 01304200": "01304200", "Orient Point Harbor, NY — 01304190": "01304190",
    "Forge River at Mastic, NY — 01304057": "01304057", "Hempstead Bay at Freeport, NY — 01304998": "01304998",
    "Enter a different USGS station": "custom",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { color-scheme: light; }
.stApp {background:linear-gradient(145deg,#f3f8ff 0%,#f6f9fd 52%,#edf5ff 100%);color:#18324b;}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1rem;padding-bottom:2rem;max-width:1500px;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#123b64,#164b78);}
[data-testid="stSidebar"] * {color:#f3f8ff !important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] [data-baseweb="select"] > div {background:white !important;color:#18324b !important;}
[data-testid="stSidebar"] [data-baseweb="select"] span {color:#18324b !important;}
[data-testid="stMetric"] {background:white;border:1px solid #d8e6f4;padding:16px 18px;border-radius:16px;box-shadow:0 6px 20px rgba(31,82,128,.07);}
[data-testid="stMetricLabel"] {color:#58738d !important;font-weight:600;}
[data-testid="stMetricValue"] {color:#183b5a;}
[data-testid="stExpander"] {background:white;border:1px solid #d8e6f4;border-radius:14px;}
.stButton > button,.stDownloadButton > button {background:#1769aa !important;color:white !important;border:0;border-radius:10px;font-weight:700;}
.stButton > button:hover {background:#105487 !important;}
[data-testid="stCaptionContainer"] {color:#60778d !important;}
.hero {padding:17px 26px;border-radius:18px;background:linear-gradient(112deg,#123b64,#1769aa 58%,#328cc4);color:white;margin:0 0 18px;box-shadow:0 10px 28px rgba(23,75,120,.18);}
.hero h1 {font-size:2.2rem;margin:0;color:white;letter-spacing:-.04em;line-height:1.2;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs(site: str, days: int = 90) -> pd.DataFrame:
    response = requests.get("https://waterservices.usgs.gov/nwis/iv/", params={"format":"json", "sites":site, "parameterCd":"00010,00300,00400,00480,90860", "period":f"P{days}D"}, timeout=45, headers={"User-Agent":"WaterManagementAI/1.0"})
    response.raise_for_status()
    series = response.json().get("value", {}).get("timeSeries", [])
    mapping = {meta["code"]:meta["column"] for meta in VARIABLES.values()}
    mapping["90860"] = "salinity_PSU"
    parts = []
    for item in series:
        codes, groups = item.get("variable", {}).get("variableCode", []), item.get("values", [])
        if not codes or not groups: continue
        col = mapping.get(codes[0].get("value"))
        if col is None: continue
        data = pd.DataFrame(groups[0].get("value", []))
        if not {"dateTime", "value"}.issubset(data.columns): continue
        data["time"] = pd.to_datetime(data["dateTime"], utc=True, errors="coerce")
        data[col] = pd.to_numeric(data["value"], errors="coerce")
        parts.append(data.dropna(subset=["time"]).set_index("time")[[col]].groupby(level=0).median())
    if not parts: raise ValueError("USGS returned no supported water-quality series for this station.")
    raw = pd.concat(parts, axis=1).groupby(level=0).median(numeric_only=True).sort_index()
    return raw.loc[:, ~raw.columns.duplicated()]


def fetch_noaa(station: str, product: str, days: int = 90) -> pd.Series:
    end = datetime.now(timezone.utc)
    response = requests.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter", params={"product":product,"application":"WaterManagementAI","begin_date":(end-timedelta(days=days)).strftime("%Y%m%d"),"end_date":end.strftime("%Y%m%d"),"station":station,"time_zone":"gmt","units":"metric","format":"json","interval":"h"}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"): raise ValueError(payload["error"].get("message", "NOAA API error"))
    data = pd.DataFrame(payload.get("data", []))
    if not {"t", "v"}.issubset(data.columns): raise ValueError(f"No NOAA {product} observations returned.")
    data["time"] = pd.to_datetime(data["t"], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["v"], errors="coerce")
    return data.dropna(subset=["time", "value"]).set_index("time")["value"].groupby(level=0).median().sort_index()


def hourly_history(raw: pd.DataFrame, noaa_station: str) -> tuple[pd.DataFrame, dict[str,str]]:
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index, utc=True)
    result = raw.sort_index().resample("1h").median(numeric_only=True)
    for col in VALUE_COLUMNS:
        if col not in result: result[col] = np.nan
    result = result[VALUE_COLUMNS]
    notes = {}
    for col, product in [("temp_C", "water_temperature"), ("salinity_PSU", "salinity")]:
        if noaa_station.strip() and result[col].isna().any():
            try:
                fallback = fetch_noaa(noaa_station.strip(), product).resample("1h").median().reindex(result.index)
                mask = result[col].isna() & fallback.notna()
                result.loc[mask, col] = fallback.loc[mask]
                notes[col] = f"NOAA CO-OPS filled {int(mask.sum())} hourly gaps"
            except Exception as exc: notes[col] = f"NOAA CO-OPS unavailable: {exc}"
    return result, notes


def normalize_station(value) -> str:
    value = str(value).strip().replace("USGS-", "").removesuffix(".0")
    return value.zfill(8) if value.isdigit() and len(value) <= 8 else value

@st.cache_resource(show_spinner=False)
def load_assets():
    models, hierarchies, metrics = {}, {}, {}
    for meta in VARIABLES.values():
        key, folder = meta["key"], MODEL_ROOT / meta["folder"]
        models[key] = {}
        for h in range(1,25):
            path = folder / f'{meta["prefix"]}_{h}h.json'
            if not path.exists(): raise FileNotFoundError(f"Missing model: {path.relative_to(ROOT)}")
            model = xgb.XGBRegressor(); model.load_model(str(path)); models[key][h] = model
        hierarchy_path = folder / HIERARCHIES[key]
        if not hierarchy_path.exists(): raise FileNotFoundError(f"Missing hierarchy: {hierarchy_path.relative_to(ROOT)}")
        hierarchies[key] = pd.read_csv(hierarchy_path)
        metric_path = folder / HIERARCHIES[key].replace("HIERARCHY", "METHOD_METRICS")
        metrics[key] = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    return models, hierarchies, metrics


def make_features(history: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    row = {col:float(history.at[timestamp,col]) for col in VALUE_COLUMNS}
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            t = timestamp-pd.Timedelta(hours=lag)
            row[f"{col}_lag{lag}h"] = float(history.at[t,col]) if t in history.index and pd.notna(history.at[t,col]) else np.nan
    row.update(hour=int(timestamp.hour), month=int(timestamp.month))
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def choose_method(hierarchy: pd.DataFrame, metrics: pd.DataFrame, station: str, horizon: int) -> str:
    hcol = "horizon" if "horizon" in hierarchy else "forecast_hour"
    scol = next((c for c in ("site_code","station","station_id") if c in hierarchy), None)
    mcol = next((c for c in ("selected_method","method") if c in hierarchy), None)
    if scol and mcol:
        rows = hierarchy[(hierarchy[scol].map(normalize_station)==normalize_station(station)) & (pd.to_numeric(hierarchy[hcol],errors="coerce")==horizon)]
        if not rows.empty:
            method = str(rows.iloc[0][mcol]).lower()
            if "persist" in method: return "persistence"
            if "xg" in method: return "xgboost"
    if not metrics.empty:
        h = next((c for c in ("horizon","forecast_hour") if c in metrics), None)
        x = next((c for c in ("XGBoost_MAE","xgb_mae") if c in metrics), None)
        p = next((c for c in ("Persistence_MAE","persistence_mae") if c in metrics), None)
        if h and x and p:
            rows=metrics[pd.to_numeric(metrics[h],errors="coerce")==horizon]
            xv,pv=pd.to_numeric(rows[x],errors="coerce").mean(),pd.to_numeric(rows[p],errors="coerce").mean()
            if pd.notna(xv) and pd.notna(pv): return "xgboost" if xv<pv else "persistence"
    return "xgboost"


def make_forecast(history: pd.DataFrame, station: str, assets):
    models, hierarchies, metrics = assets
    complete=[]
    for ts in history.index:
        features=make_features(history,ts)
        if features.notna().all(axis=None): complete.append(ts)
    shared=history.dropna(subset=VALUE_COLUMNS,how="any")
    if shared.empty: raise ValueError("Recent observations for all four parameters are needed to create a forecast.")
    origin=complete[-1] if complete else shared.index[-1]
    forecasts=[]
    for label,meta in VARIABLES.items():
        key,col=meta["key"],meta["column"]
        for horizon in range(1,25):
            method=choose_method(hierarchies[key],metrics[key],station,horizon)
            value=float(history.at[origin,col])
            source="Persistence"
            if method=="xgboost" and complete:
                model=models[key][horizon]
                names=model.get_booster().feature_names
                if names and names!=FEATURE_COLUMNS: raise ValueError(f"{meta['prefix']}_{horizon}h model feature order differs from app schema.")
                candidate=float(model.predict(make_features(history,origin))[0])
                if key=="salinity":
                    baseline=float(history[col].dropna().tail(24).median())
                    if baseline>5 and candidate < baseline*0.5:
                        source="Persistence (salinity model scale guard)"
                    else:
                        value=candidate; source="XGBoost"
                else:
                    value=candidate; source="XGBoost"
            forecasts.append({"variable":key,"label":label,"horizon":horizon,"timestamp":origin+pd.Timedelta(hours=horizon),"prediction":value,"method_source":source,"origin":origin})
    return pd.DataFrame(forecasts)

st.markdown('<div class="hero"><h1>🌊 Water Management AI</h1></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("Monitoring station")
    choice=st.selectbox("USGS station",list(STATIONS))
    station=st.text_input("USGS station ID","") if STATIONS[choice]=="custom" else STATIONS[choice]
    noaa=st.text_input("Optional NOAA CO-OPS station ID",help="Used to fill missing temperature or salinity observations.")
    refresh=st.button("Refresh data",use_container_width=True)
if refresh: st.cache_data.clear()
if not re.fullmatch(r"\d{8}(\d{7})?",station.strip()): st.error("Enter a valid USGS station ID (8 or 15 digits)."); st.stop()
try:
    assets=load_assets()
    with st.spinner("Loading station observations…"):
        history,source_notes=hourly_history(fetch_usgs(station.strip()),noaa)
except Exception as exc:
    st.error(f"Could not load station data or model assets: {exc}"); st.stop()

st.subheader(f"Station {station.strip()}")
cols=st.columns(4)
for i,(label,meta) in enumerate(VARIABLES.items()):
    series=history[meta["column"]].dropna()
    with cols[i]:
        if series.empty:
            st.metric(label,"Unavailable"); st.caption("No recent observation returned")
        else:
            latest=float(series.iloc[-1]); prior=series.loc[series.index<=series.index[-1]-pd.Timedelta(hours=24)]
            delta=latest-float(prior.iloc[-1]) if not prior.empty else None
            st.metric(label,f"{latest:.2f} {meta['unit']}",delta=f"{delta:+.2f} {meta['unit']} vs 24h" if delta is not None else None,delta_color="normal")
            st.caption(source_notes.get(meta["column"],"USGS NWIS")+f" · {series.index[-1].strftime('%b %d, %H:%M UTC')}")
try:
    forecast=make_forecast(history,station.strip(),assets)
except Exception as exc:
    st.warning(f"Forecast unavailable: {exc}"); st.stop()

origin=forecast["origin"].iloc[0]
start,end=origin-pd.Timedelta(hours=24),origin+pd.Timedelta(hours=24)
fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.11)
colors={"observed":"#1769aa","forecast":"#53a4dc"}
for row,(label,meta) in enumerate(VARIABLES.items(),start=1):
    col=meta["column"]
    observed=history.loc[(history.index>=start)&(history.index<=origin),col].dropna()
    subset=forecast[forecast.variable==meta["key"]].sort_values("horizon")
    if not observed.empty:
        fig.add_trace(go.Scatter(x=observed.index,y=observed.values,mode="lines",name="Observed",legendgroup="observed",showlegend=row==1,line={"color":colors["observed"],"width":2.8},hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}<extra>Observed</extra>"),row=row,col=1)
    if not subset.empty:
        fig.add_trace(go.Scatter(x=subset.timestamp,y=subset.prediction,mode="lines+markers",name="24h outlook",legendgroup="forecast",showlegend=row==1,line={"color":colors["forecast"],"width":2.5,"dash":"dash"},marker={"size":5},hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}<extra>Forecast</extra>"),row=row,col=1)
    # Use axis titles rather than manually anchored annotations; this works across Plotly versions
    # and keeps subplot labels separated from the plotted data.
    fig.update_yaxes(title_text=f"{label}<br>({meta['unit']})",row=row,col=1,gridcolor="#e3edf6",zerolinecolor="#d4e2ef",title_font={"size":12})
fig.add_vline(x=origin,line_width=1.5,line_dash="dot",line_color="#e4a43b")
fig.add_annotation(x=origin,y=1.015,xref="x",yref="paper",text="NOW",showarrow=False,font={"size":10,"color":"#805916"},bgcolor="#fff7e6",borderpad=4)
fig.update_xaxes(range=[start,end],title_text="Time (UTC)",row=4,col=1,gridcolor="#edf2f8")
fig.update_layout(height=1080,hovermode="x unified",plot_bgcolor="white",paper_bgcolor="rgba(255,255,255,0)",font={"family":"DM Sans, sans-serif","color":"#29495b"},legend={"orientation":"h","y":1.045,"x":.7},margin={"t":80,"b":45,"l":125,"r":22})
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
if forecast["method_source"].str.contains("scale guard").any():
    st.markdown(
        '<p style="color:#d32f2f; font-weight:700;">Salinity model not working</p>',
        unsafe_allow_html=True,
    )
with st.expander("Data sources and forecast details"):
    source_rows=[]
    for label,meta in VARIABLES.items():
        valid=history[meta["column"]].dropna()
        source_rows.append({"Parameter":label,"Latest source":source_notes.get(meta["column"],"USGS NWIS"),"Latest observation (UTC)":valid.index[-1].strftime("%Y-%m-%d %H:%M") if not valid.empty else "Unavailable","Hourly observations":int(valid.size)})
    st.dataframe(pd.DataFrame(source_rows),use_container_width=True,hide_index=True)
    for note in source_notes.values(): st.caption(note)
    st.caption("Blue solid lines show observed readings; lighter blue dashed lines show forecasts. Forecasts are not a substitute for field measurements.")
st.caption("Forecasts use XGBoost or persistence based on the station validation hierarchy. All timestamps are UTC.")
