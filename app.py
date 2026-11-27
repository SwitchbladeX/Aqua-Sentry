from __future__ import annotations

import base64
import html
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import available_timezones

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="AquaSentry", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
MODEL_VARIABLES = {
    "Dissolved oxygen": {"key": "do", "folder": "dissolved_oxygen", "prefix": "DO", "column": "DO_mgL", "unit": "mg/L", "code": "00300"},
    "Temperature": {"key": "temp", "folder": "temperature", "prefix": "Temp", "column": "temp_C", "unit": "°C", "code": "00010"},
    "pH": {"key": "ph", "folder": "ph", "prefix": "pH", "column": "pH", "unit": "pH", "code": "00400"},
    "Salinity": {"key": "salinity", "folder": "salinity", "prefix": "Salinity", "column": "salinity_PSU", "unit": "PSU", "code": "90860"},
}
VARIABLES = {label: meta for label, meta in MODEL_VARIABLES.items() if meta["key"] != "salinity"}
VALUE_COLUMNS = [meta["column"] for meta in MODEL_VARIABLES.values()]
LAGS = [1, 3, 6, 12]
FEATURE_COLUMNS = VALUE_COLUMNS + [f"{col}_lag{lag}h" for col in VALUE_COLUMNS for lag in LAGS] + ["hour", "month"]
HIERARCHIES = {
    "do": "DO_VALIDATION_HIERARCHY.csv",
    "temp": "TEMP_VALIDATION_HIERARCHY.csv",
    "ph": "PH_VALIDATION_HIERARCHY.csv",
}
MIN_ADAPTATION_EXAMPLES = 12
MIN_VALIDATION_EXAMPLES = 6
FULL_ADAPTATION_EXAMPLES = 48
INTERPOLATION_MAX_GAP_HOURS = 3
PIPELINE_VERSION = "aquasentry-20261124-brand-preserved-clean-ui"

LEGACY_SHUTDOWN = datetime(2027, 2, 22, tzinfo=timezone.utc)
LEGACY_USGS = datetime.now(timezone.utc) < LEGACY_SHUTDOWN
try:
    USGS_API_KEY = st.secrets.get("USGS_API_KEY", "")
except Exception:
    USGS_API_KEY = os.environ.get("USGS_API_KEY", "")
SIMULATE_DOWN = os.environ.get("AQUASENTRY_SIMULATE_DOWN", "") == "1"

STATIONS = {
    "Crabtree Swamp at Conway, SC — 02110701": {"provider": "usgs", "id": "02110701"},
    "Spring Creek near Spring, TX — 08068500": {"provider": "usgs", "id": "08068500"},
    "West Harbor at Oyster Bay, NY — 01303152": {"provider": "usgs", "id": "01303152"},
    "Flax Pond at Old Field, NY — 01304057": {"provider": "usgs", "id": "01304057"},
    "Peconic River at County Hwy 105, NY — 01304562": {"provider": "usgs", "id": "01304562"},
    "Enter a different USGS station": {"provider": "custom", "id": "custom"},
}
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {color-scheme:light;}
html,body,.stApp,.stApp input,.stApp button,.stApp select {font-family:'DM Sans',sans-serif;}
.stApp,[data-testid="stAppViewContainer"] {background:#0b1d35;color:#e8f0fa;}
[data-testid="stHeader"] {background:transparent !important;box-shadow:none !important;border:none !important;backdrop-filter:none !important;}
.block-container {max-width:1460px;padding-top:3.5rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {background:#19344f;border-right:1px solid #2b4865;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top:0 !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.85rem;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#e8f0fa !important;}
[data-testid="stSidebar"] h3 {padding-top:0;margin-top:0;}
[data-testid="stSidebar"] input {background:#f4f7fb !important;color:#19344f !important;-webkit-text-fill-color:#19344f !important;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#f4f7fb !important;color:#19344f !important;border-color:#d7e2ee !important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dfe7f1;border-radius:16px;padding:16px 18px;box-shadow:0 6px 20px rgba(0,0,0,.08);}
[data-testid="stMetricLabel"] {color:#60798e !important;}
[data-testid="stMetricValue"] {color:#183f60 !important;font-size:1.8rem;}
[data-testid="stMain"] h2,[data-testid="stMain"] h3 {color:#edf3fc;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {color:#b3c5da;}
[data-testid="stExpander"] {background:#fff;color:#234b6c;border:1px solid #dce7f2;border-radius:12px;}
[data-testid="stExpander"] summary,[data-testid="stExpander"] label,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {color:#234b6c !important;}
[data-testid="stExpander"] [data-testid="stCaptionContainer"] {color:#60798e !important;}
[data-testid="stSidebar"] [data-testid="stExpander"] label,
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {color:#234b6c !important;}
.stButton > button,.stDownloadButton > button,[data-testid="stBaseButton-primary"] {background:#246fba !important;color:#fff !important;border:1px solid #246fba !important;border-radius:10px;min-height:44px;font-weight:600;transition:transform .16s ease,box-shadow .16s ease,background .16s ease;}
.stButton > button:hover,.stDownloadButton > button:hover {transform:scale(1.025);background:#195b9d !important;box-shadow:0 5px 14px rgba(36,111,186,.25);}
.stButton > button:active,.stDownloadButton > button:active {transform:scale(.99);}
.stButton > button:focus-visible,.stDownloadButton > button:focus-visible {outline:3px solid #a9cff6;outline-offset:3px;}
.stButton > button p,.stDownloadButton > button p,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {color:#fff !important;}
.hero {display:flex;align-items:center;gap:20px;background:#fff;border:1px solid #dfe7f1;border-radius:20px;padding:26px 30px;margin-bottom:18px;box-shadow:0 10px 30px rgba(0,0,0,.12);}
.hero h1 {color:#153856;font-size:2.65rem;letter-spacing:-.045em;padding:0 0 6px;margin:0;line-height:1.15;}
.hero p {color:#59738b;margin:0;font-size:1rem;}
.brand-logo {width:82px;height:82px;flex:0 0 82px;border-radius:50%;object-fit:contain;background:#fff;border:1px solid #e1e9f2;padding:4px;}
.station-id {font-size:.76rem;color:#a6bdd5;margin-top:4px;}
.station-title {font-size:1.35rem;font-weight:700;color:#f0f5fc;margin:3px 0 6px;}
[data-testid="stPlotlyChart"] {background:#fff;border:1px solid #dfe7f1;border-radius:18px;padding:10px;overflow:hidden;}
.footer {border-top:1px solid #2b425d;margin-top:18px;padding-top:10px;color:#a6bdd5;font-size:.78rem;}
@media(prefers-reduced-motion:reduce){.stButton > button,.stDownloadButton > button{transition:none;}}
@media(max-width:700px){.hero{gap:14px;padding:20px}.hero h1{font-size:2rem}.hero p{font-size:.9rem}.brand-logo{width:60px;height:60px;flex-basis:60px}[data-testid="stMetricValue"]{font-size:1.45rem}.block-container{padding-top:3.8rem}}
</style>
"""

DISPLAY_TZ = "UTC"
STATION_TIMEZONES = {
    "02110701": "America/New_York",
    "08068500": "America/Chicago",
    "01303152": "America/New_York",
    "01304057": "America/New_York",
    "01304562": "America/New_York",
}
TIMEZONE_OPTIONS = [
    "Station local (automatic)", "UTC", "America/New_York", "America/Chicago",
    "America/Denver", "America/Los_Angeles", "Europe/London", "Europe/Paris",
    "Asia/Kolkata", "Asia/Tokyo", "Australia/Sydney",
]


def format_time(value, tz=None):
    if value is None or pd.isna(value):
        return "Unavailable"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(tz or DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


def display_frame(frame, tz=None):
    frame = frame.copy()
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.DatetimeTZDtype):
            frame[col] = frame[col].map(lambda v: format_time(v, tz=tz))
    return frame


def forecast_export(forecast, tz=None):
    tz = tz or DISPLAY_TZ
    result = forecast.copy()
    for col in ["origin", "timestamp"]:
        result[col + "_utc"] = result[col].map(lambda t: pd.Timestamp(t).isoformat())
        result[col + "_local"] = result[col].map(lambda t: format_time(t, tz=tz))
        result = result.drop(columns=col)
    generated = forecast.attrs.get("generated_at_utc")
    result["generated_at_utc"] = generated
    result["generated_at_local"] = format_time(generated, tz=tz)
    result["display_timezone"] = tz
    return result


def selected_station_health(raw):
    now = pd.Timestamp.now(tz="UTC")
    expected = pd.date_range(now.floor("h") - pd.Timedelta(days=7), now.floor("h"), freq="h", inclusive="left")
    rows = []
    for label, meta in VARIABLES.items():
        values = raw[meta["column"]].dropna().sort_index()
        values = values.loc[values.index <= now]
        timestamp = values.index[-1] if len(values) else None
        age = (now - timestamp).total_seconds() / 3600 if timestamp is not None else np.nan
        hours = int(values.resample("1h").count().reindex(expected, fill_value=0).gt(0).sum()) if len(values) else 0
        rows.append({"Parameter": label, "Latest raw observation": format_time(timestamp), "Age (hours)": round(age, 2) if np.isfinite(age) else np.nan, "Completed hours with samples / 168": f"{hours}/168", "Status": "Unavailable" if timestamp is None else ("Fresh" if age <= 3 else "Reading delayed")})
    return pd.DataFrame(rows)


def check_default_station_feeds(source_choice, progress_holder=None):
    rows = []
    usgs_stations = [(label, config) for label, config in STATIONS.items() if config["provider"] == "usgs"]
    for index, (label, config) in enumerate(usgs_stations, 1):
        if progress_holder is not None:
            try:
                progress_holder.update(label=f"Checking {label} ({index}/{len(usgs_stations)})")
            except Exception:
                pass
        try:
            records, _ = fetch_usgs(config["id"], days=7, source_choice=source_choice)
            now = pd.Timestamp.now(tz="UTC")
            for parameter, meta in VARIABLES.items():
                group = records.loc[records["column"].eq(meta["column"]) & records["time"].le(now)]
                timestamp = group["time"].max() if len(group) else pd.NaT
                age = (now - timestamp).total_seconds() / 3600 if pd.notna(timestamp) else np.nan
                rows.append({"Station": label, "Parameter": parameter, "Latest observation": timestamp, "Age (hours)": round(age, 2) if np.isfinite(age) else np.nan, "Separate series": group["series"].nunique(), "Status": "Unavailable" if pd.isna(timestamp) else ("Fresh series available" if age <= 3 else "Reading delayed")})
        except Exception as exc:
            rows.append({"Station": label, "Parameter": "All", "Latest observation": pd.NaT, "Age (hours)": np.nan, "Separate series": np.nan, "Status": f"Could not load station data: {exc}"})
    return pd.DataFrame(rows)


def normalize_station(value):
    value = str(value).strip().replace("USGS-", "").removesuffix(".0")
    return value.zfill(8) if value.isdigit() and len(value) <= 8 else value


def clean_values(values, col):
    values = pd.to_numeric(values, errors="coerce")
    values = values.where(np.isfinite(values))
    if col in ("DO_mgL", "salinity_PSU"):
        values = values.mask(values < 0)
    if col == "pH":
        values = values.mask((values < 0) | (values > 14))
    return values


def _request(url, **kwargs):
    if SIMULATE_DOWN:
        raise requests.ConnectionError("AQUASENTRY_SIMULATE_DOWN is set")
    last = None
    for attempt in range(3):
        try:
            response = requests.get(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last


def _api_headers():
    headers = {"User-Agent": "AquaSentry/1.3"}
    if USGS_API_KEY:
        headers["X-Api-Key"] = USGS_API_KEY
    return headers


@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs(site, days=90, source_choice="Legacy + current (auto)"):
    use_legacy = LEGACY_USGS and source_choice != "Current only"
    mapping = {meta["code"]: meta["column"] for meta in MODEL_VARIABLES.values()}
    parts, names, warnings = [], [], []
    if use_legacy:
        try:
            response = _request(
                "https://waterservices.usgs.gov/nwis/iv/",
                params={"format": "json", "sites": site, "parameterCd": ",".join(mapping), "period": f"P{days}D"},
                timeout=(10, 35), headers=_api_headers(),
            )
            for item in response.json().get("value", {}).get("timeSeries", []):
                codes = item.get("variable", {}).get("variableCode", [])
                col = mapping.get(codes[0].get("value")) if codes else None
                if col is None:
                    continue
                name = item.get("sourceInfo", {}).get("siteName")
                if name:
                    names.append(name)
                sentinel = pd.to_numeric(item.get("variable", {}).get("noDataValue"), errors="coerce")
                for number, group in enumerate(item.get("values", [])):
                    data = pd.DataFrame(group.get("value", []))
                    if not {"dateTime", "value"}.issubset(data.columns):
                        continue
                    values = pd.to_numeric(data["value"], errors="coerce")
                    if pd.notna(sentinel):
                        values = values.mask(values == sentinel)
                    part = pd.DataFrame({
                        "time": pd.to_datetime(data["dateTime"], utc=True, errors="coerce"),
                        "column": col, "value": clean_values(values, col),
                        "series": f"{item.get('name', col)} / group {number}",
                    }).dropna(subset=["time", "value"])
                    if not part.empty:
                        parts.append(part)
        except Exception as exc:
            warnings.append(f"USGS legacy request unavailable: {exc}")
    present = set(pd.concat(parts)["column"]) if parts else set()
    for code, col in mapping.items():
        if col in present:
            continue
        url = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
        params = {"f": "json", "monitoring_location_id": f"USGS-{site}", "parameter_code": code, "time": f"P{days}D", "limit": 10000}
        pages, visited = [], set()
        try:
            while url:
                response = _request(url, params=params, timeout=(10, 45), headers=_api_headers())
                if response.url in visited:
                    raise ValueError("Repeated USGS pagination URL")
                visited.add(response.url)
                payload = response.json()
                rows = []
                for feature in payload.get("features", []):
                    prop = feature.get("properties", {})
                    if prop.get("monitoring_location_id") != f"USGS-{site}" or str(prop.get("parameter_code")) != code:
                        continue
                    rows.append({"time": prop.get("time"), "value": prop.get("value"), "column": col, "series": f"USGS time series {prop.get('time_series_id', 'unknown')}"})
                if rows:
                    pages.append(pd.DataFrame(rows))
                link = next((link.get("href") for link in payload.get("links", []) if link.get("rel") == "next"), None)
                url = urljoin(response.url, link) if link else None
                params = None
                if len(visited) > 100:
                    raise ValueError("USGS pagination limit exceeded")
            if pages:
                part = pd.concat(pages, ignore_index=True)
                part["time"] = pd.to_datetime(part["time"], utc=True, errors="coerce")
                values = pd.to_numeric(part["value"], errors="coerce")
                part["value"] = clean_values(values.mask(values <= -99990), col)
                part = part.dropna(subset=["time", "value"])
                if not part.empty:
                    parts.append(part)
        except Exception as exc:
            if col != "salinity_PSU":
                warnings.append(f"USGS {code} request unavailable: {exc}")
    if not parts:
        raise ValueError("USGS returned no supported water-quality observations. " + " | ".join(warnings))
    records = pd.concat(parts, ignore_index=True).drop_duplicates(["time", "column", "series"], keep="last")
    records.attrs["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    records.attrs["provider_warnings"] = warnings
    fallback_name = next((label.split(" — ")[0] for label, config in STATIONS.items() if config["id"] == site), f"USGS {site}")
        return records, names[0] if names else fallback_name


def pick_newest_series(records):
    parts = []
    for meta in MODEL_VARIABLES.values():
        group = records.loc[records["column"].eq(meta["column"])]
        if group.empty:
            continue
        latest = group.groupby("series")["time"].max()
        chosen = latest.idxmax()
        part = group.loc[group["series"].eq(chosen)].groupby("time")["value"].median()
        parts.append(part.rename(meta["column"]))
    if not parts:
        raise ValueError("No usable sensor series.")
    result = pd.concat(parts, axis=1).sort_index().reindex(columns=VALUE_COLUMNS)
    result.attrs.update(records.attrs)
    return result


def interpolate_history(history, sources, max_gap=INTERPOLATION_MAX_GAP_HOURS):
    history = history.copy().replace([np.inf, -np.inf], np.nan)
    sources = sources.copy()
    for col in history.columns:
        series = history[col]
        missing = series.isna()
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        sizes = missing.groupby(groups).transform("sum")
        estimate = series.interpolate(method="time", limit_area="inside")
        fill = missing & sizes.le(max_gap) & estimate.notna()
        history.loc[fill, col] = estimate.loc[fill]
        sources.loc[fill, col] = f"Interpolated (internal gap <= {max_gap}h)"
    return history, sources


def hourly_history(raw, provider, noaa_station=""):
    raw = raw.copy().sort_index()
    raw.index = pd.to_datetime(raw.index, utc=True)
    hourly = raw.resample("1h").median().reindex(columns=VALUE_COLUMNS)
    hourly = hourly.loc[hourly.index < pd.Timestamp.now(tz="UTC").floor("h")]
    sources = pd.DataFrame("Missing", index=hourly.index, columns=VALUE_COLUMNS).mask(hourly.notna(), provider)
    notes = {}
    hourly, sources = interpolate_history(hourly, sources)
    for col in VALUE_COLUMNS:
        count = int(sources[col].astype(str).str.startswith("Interpolated").sum())
        if count:
            notes[col] = f"{count} interpolated hourly estimates"
    return hourly, sources, notes


def asset_signature():
    paths = []
    for meta in VARIABLES.values():
        folder = MODEL_ROOT / meta["folder"]
        paths.extend(folder.glob("*.json"))
        paths.extend(folder.glob("*.csv"))
    return tuple((str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(paths))

@st.cache_resource(show_spinner=False)
def load_assets(signature):
    models, hierarchies, metrics, issues = {}, {}, {}, []
    for meta in VARIABLES.values():
        key, folder = meta["key"], MODEL_ROOT / meta["folder"]
        models[key] = {}
        for horizon in range(1, 25):
            path = folder / f'{meta["prefix"]}_{horizon}h.json'
            try:
                model = xgb.XGBRegressor()
                model.load_model(str(path))
                booster = model.get_booster()
                if booster.num_features() != len(FEATURE_COLUMNS):
                    raise ValueError("Expected 22 features.")
                if booster.feature_names and booster.feature_names != FEATURE_COLUMNS:
                    raise ValueError("Model feature names/order differ from app schema.")
                models[key][horizon] = model
            except Exception as exc:
                issues.append(f"{path.relative_to(ROOT)}: {exc}")
        for target, path in [(hierarchies, folder / HIERARCHIES[key]), (metrics, folder / HIERARCHIES[key].replace("HIERARCHY", "METHOD_METRICS"))]:
            try:
                target[key] = pd.read_csv(path, dtype=str)
            except Exception as exc:
                target[key] = pd.DataFrame()
                issues.append(f"{path.relative_to(ROOT)}: {exc}")
    return models, hierarchies, metrics, issues


def make_features(history, timestamp):
    row = {col: history.at[timestamp, col] for col in VALUE_COLUMNS}
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            row[f"{col}_lag{lag}h"] = history[col].get(timestamp - pd.Timedelta(hours=lag), np.nan)
    row.update(hour=timestamp.hour, month=timestamp.month)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=float).replace([np.inf, -np.inf], np.nan)


def build_feature_history(history):
    result = history.reindex(columns=VALUE_COLUMNS).copy()
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            result[f"{col}_lag{lag}h"] = history[col].reindex(history.index - pd.Timedelta(hours=lag)).to_numpy()
    result["hour"] = result.index.hour
    result["month"] = result.index.month
    result = result.reindex(columns=FEATURE_COLUMNS).astype(float)
    return result.loc[np.isfinite(result.to_numpy()).all(axis=1)]


def choose_method(hierarchy, metrics, station, horizon):
    columns = {str(c).strip().lower(): c for c in hierarchy.columns}
    hcol = columns.get("horizon") or columns.get("forecast_hour")
    scol = columns.get("site_code") or columns.get("station") or columns.get("station_id")
    mcol = columns.get("selected_method") or columns.get("method")
    if all(col is not None for col in (hcol, scol, mcol)):
        rows = hierarchy[(hierarchy[scol].map(normalize_station) == normalize_station(station)) & (pd.to_numeric(hierarchy[hcol], errors="coerce") == horizon)]
        if not rows.empty:
            method = str(rows.iloc[0][mcol]).lower()
            if "persist" in method:
                return "persistence"
            if "xg" in method:
                return "xgboost"
    columns = {str(c).strip().lower(): c for c in metrics.columns}
    hcol = columns.get("horizon") or columns.get("forecast_hour")
    xcol = columns.get("xgboost_mae") or columns.get("xgb_mae") or columns.get("model_mae")
    pcol = columns.get("persistence_mae") or columns.get("persist_mae")
    if all(col is not None for col in (hcol, xcol, pcol)):
        rows = metrics[pd.to_numeric(metrics[hcol], errors="coerce") == horizon]
        xvalue = pd.to_numeric(rows[xcol], errors="coerce").mean()
        pvalue = pd.to_numeric(rows[pcol], errors="coerce").mean()
        if np.isfinite(xvalue) and np.isfinite(pvalue):
            return "xgboost" if xvalue < pvalue else "persistence"
    return "xgboost"


def effective_method(key, hierarchy, metrics, station, horizon):
    return choose_method(hierarchy, metrics, station, horizon)


def base_predictions(key, horizon, features, baseline, assets, method):
    baseline = np.asarray(baseline, dtype=float)
    prediction = baseline.copy()
    sources = np.full(len(baseline), "Persistence", dtype=object)
    if method == "xgboost":
        model = assets[0].get(key, {}).get(horizon)
        if model is None:
            sources[:] = "Persistence (model unavailable)"
            return prediction, sources
        candidate = np.asarray(model.predict(features), dtype=float)
        finite = np.isfinite(candidate)
        prediction[finite] = candidate[finite]
        sources[finite] = "XGBoost"
        sources[~finite] = "Persistence (non-finite model output)"
    return prediction, sources


@st.cache_data(ttl=900, show_spinner=False)
def build_station_adaptation(history, hourly_provenance, station, signature, policy_version, _assets):
    interpolated = hourly_provenance.reindex_like(history).apply(lambda col: col.astype(str).str.startswith("Interpolated"))
    history = history.mask(interpolated)
    features = build_feature_history(history)
    adaptation = {}
    for meta in VARIABLES.values():
        key, col = meta["key"], meta["column"]
        adaptation[key] = {}
        for horizon in range(1, 25):
            method = effective_method(key, _assets[1][key], _assets[2][key], station, horizon)
            item = {"correction": 0.0, "examples": 0, "median_error": 0.0, "method": method, "validated": False, "validation_mae_before": None, "validation_mae_after": None, "validation_examples": 0, "training_examples": 0, "source_counts": {}, "status": "Insufficient examples"}
            adaptation[key][horizon] = item
            if features.empty:
                continue
            if method == "xgboost" and horizon not in _assets[0].get(key, {}):
                item["status"] = "Model unavailable"
                continue
            targets = features.index + pd.Timedelta(hours=horizon)
            actual = history[col].reindex(targets).to_numpy(dtype=float)
            source_labels = hourly_provenance[col].reindex(targets).to_numpy()
            valid = np.isfinite(actual)
            origins = features.index[valid]
            if not len(origins):
                continue
            try:
                predicted, labels = base_predictions(key, horizon, features.loc[origins], history[col].reindex(origins).to_numpy(), _assets, method)
            except Exception:
                item["status"] = "Prediction unavailable"
                continue
            errors = actual[valid] - predicted
            expected = "XGBoost" if method == "xgboost" else "Persistence"
            good = np.isfinite(errors) & (labels == expected)
            errors, origins, source_labels = errors[good], origins[good], source_labels[valid][good]
            count = len(errors)
            item.update(examples=count, median_error=float(np.median(errors)) if count else 0.0, source_counts={str(s): int(np.sum(source_labels == s)) for s in pd.unique(source_labels)})
            if count < MIN_ADAPTATION_EXAMPLES + MIN_VALIDATION_EXAMPLES:
                continue
            split = min(int(count * .7), count - MIN_VALIDATION_EXAMPLES)
            cutoff = origins[split]
            horizon_delta = pd.Timedelta(hours=horizon)
            train_mask = (origins[:split] + horizon_delta) < cutoff
            train = errors[:split][train_mask]
            validation = errors[split:]
            item.update(training_examples=len(train), validation_examples=len(validation))
            if len(train) < MIN_ADAPTATION_EXAMPLES:
                item["status"] = "Insufficient examples after temporal purge"
                continue
            trial = float(np.median(train)) * min(1.0, len(train) / FULL_ADAPTATION_EXAMPLES)
            before, after = float(np.mean(np.abs(validation))), float(np.mean(np.abs(validation - trial)))
            validated = after < before
            item.update(correction=trial if validated else 0.0, validated=validated, validation_mae_before=before, validation_mae_after=after, status="Validated correction" if validated else "No validated improvement")
    return adaptation
def forecast_input_details(features, sources, origin):
    missing = features.columns[features.iloc[0].isna()].tolist()
    estimated = []
    for col in VALUE_COLUMNS:
        for lag in [0] + LAGS:
            time = origin - pd.Timedelta(hours=lag)
            if time in sources.index and str(sources.at[time, col]).startswith("Interpolated"):
                estimated.append(col if lag == 0 else f"{col}_lag{lag}h")
    return missing, estimated


def make_forecast(history, sources, station, assets, adaptation):
    available = history.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if available.empty:
        raise ValueError("No supported measurements are available. Interpolation cannot create an absent series.")
    origin = available.index[-1]
    if pd.Timestamp.now(tz="UTC") - (origin + pd.Timedelta(hours=1)) > pd.Timedelta(hours=3):
        raise ValueError("Latest completed observation hour is delayed by more than three hours. History remains visible; a current forecast is withheld.")
    features = make_features(history, origin)
    water_columns = [col for col in FEATURE_COLUMNS if col not in ("hour", "month")]
    if not np.isfinite(features[water_columns].to_numpy()).any():
        raise ValueError("No water measurements support this forecast origin.")
    missing, estimated = forecast_input_details(features, sources, origin)
    rows = []
    for label, meta in VARIABLES.items():
        key, col = meta["key"], meta["column"]
        baseline = float(history.at[origin, col])
        if not np.isfinite(baseline):
            continue
        for horizon in range(1, 25):
            method = effective_method(key, assets[1][key], assets[2][key], station, horizon)
            try:
                values, method_sources = base_predictions(key, horizon, features, [baseline], assets, method)
                original, source = float(values[0]), str(method_sources[0])
            except Exception:
                original, source = baseline, "Persistence (model prediction unavailable)"
            if not np.isfinite(original):
                continue
            correction, examples = 0.0, 0
            if not missing and not estimated and source in ("XGBoost", "Persistence"):
                item = adaptation.get(key, {}).get(horizon, {})
                candidate = float(item.get("correction", 0.0))
                if item.get("validated", False) and np.isfinite(candidate):
                    correction, examples = candidate, int(item.get("examples", 0))
            if source == "XGBoost":
                if missing and estimated:
                    source = "XGBoost (partial + interpolated inputs)"
                elif missing:
                    source = "XGBoost (partial inputs)"
                elif estimated:
                    source = "XGBoost (interpolated inputs)"
            rows.append({"variable": key, "label": label, "horizon": horizon, "timestamp": origin + pd.Timedelta(hours=horizon), "original_prediction": original, "prediction": original + correction, "station_correction": correction, "adaptation_examples": examples, "method_source": source, "origin": origin, "missing_input_count": len(missing), "missing_inputs": "; ".join(missing), "interpolated_input_count": len(estimated), "interpolated_inputs": "; ".join(estimated)})
    if not rows:
        raise ValueError("No finite model forecast or usable persistence reading is available.")
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def get_forecast(history, sources, station, _assets, signature, adaptation_enabled, policy_version):
    adaptation, issue = {}, None
    if adaptation_enabled:
        try:
            adaptation = build_station_adaptation(history, sources, station, signature, policy_version, _assets)
        except Exception as exc:
            issue = f"Station adjustment unavailable; base forecasts remain active: {exc}"
    forecast = make_forecast(history, sources, station, _assets, adaptation)
    forecast.attrs["generated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    return forecast, adaptation, issue


def draw_forecast(history, forecast, hours, sources, tz=None):
    tz = tz or DISPLAY_TZ
    origin = forecast.origin.iloc[0]
    start, end = origin - pd.Timedelta(hours=hours), origin + pd.Timedelta(hours=24)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.075, subplot_titles=[f'{label} · {meta["unit"]}' for label, meta in VARIABLES.items()])
    has_corrections = forecast.station_correction.abs().gt(1e-9).any()
    legend_shown = set()
    for row, (label, meta) in enumerate(VARIABLES.items(), 1):
        observed = history.loc[(history.index >= start) & (history.index <= end), meta["column"]]
        subset = forecast[forecast.variable == meta["key"]].sort_values("horizon")
        unit = "" if meta["key"] == "ph" else " " + meta["unit"]
        provenance = sources[meta["column"]].reindex(observed.index).fillna("Missing")
        fig.add_trace(go.Scatter(x=observed.index, y=observed, mode="lines", connectgaps=False, name="Hourly history", legendgroup="observed", showlegend=row == 1, line=dict(color="#246fba", width=2.8), customdata=np.array([[str(p), format_time(t, tz=tz)] for t, p in zip(observed.index, provenance)], dtype=object).reshape(-1, 2), hovertemplate="%{customdata[1]}<br>%{y:.2f}" + unit + "<br>%{customdata[0]}<extra>History</extra>"), row=row, col=1)
        interpolated = provenance.astype(str).str.startswith("Interpolated")
        if interpolated.any():
            fig.add_trace(go.Scatter(x=observed.index[interpolated], y=observed.loc[interpolated], mode="markers", name="Interpolated estimate", legendgroup="interpolated", showlegend="interpolated" not in legend_shown, marker=dict(size=7, symbol="diamond-open", color="#d78c1e"), customdata=[format_time(t, tz=tz) for t in observed.index[interpolated]], hovertemplate="%{customdata}<br>%{y:.2f}" + unit + "<extra>Interpolated, not measured</extra>"), row=row, col=1)
            legend_shown.add("interpolated")
        if subset.empty:
            continue
        if has_corrections:
            fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.original_prediction, mode="lines", name="Before station adjustment", legendgroup="base", showlegend="base" not in legend_shown, line=dict(color="#9aaec2", width=1.6, dash="dot"), customdata=[format_time(t, tz=tz) for t in subset.timestamp], hovertemplate="%{customdata}<br>%{y:.2f}" + unit + "<extra>Before adjustment</extra>"), row=row, col=1)
            legend_shown.add("base")
        fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.prediction, mode="lines+markers", name="Forecast", legendgroup="forecast", showlegend="forecast" not in legend_shown, customdata=np.column_stack((subset[["method_source", "station_correction", "adaptation_examples"]].to_numpy(), [format_time(t, tz=tz) for t in subset.timestamp])), line=dict(color="#16a394", width=2.8, dash="dash"), marker=dict(size=4, color="#16a394"), hovertemplate="%{customdata[3]}<br>%{y:.2f}" + unit + "<br>%{customdata[0]}<br>Adjustment: %{customdata[1]:+.3f}<extra>Forecast</extra>"), row=row, col=1)
        legend_shown.add("forecast")
    fig.add_vrect(x0=origin, x1=end, fillcolor="#246fba", opacity=.035, line_width=0)
    fig.add_vline(x=origin, line_dash="dot", line_color="#95a9bb", line_width=1.4)
    fig.add_annotation(x=origin, y=1.065, xref="x", yref="paper", text="FORECAST START", showarrow=False, font=dict(size=10, color="#70869a"))
    ticks = pd.date_range(start.ceil("6h"), end, freq="6h")
    ticktext = [t.tz_convert(tz).strftime("%b %d<br>%H:%M %Z") for t in ticks]
    fig.update_xaxes(tickmode="array", tickvals=list(ticks), ticktext=ticktext, range=[start, end], showline=False, zeroline=False, gridcolor="#edf2f7", tickfont=dict(size=11, color="#70869a"))
    fig.update_xaxes(title_text=f"Time ({tz})", row=3, col=1)
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="#edf2f7", tickfont=dict(size=11, color="#70869a"), nticks=5)
    fig.update_layout(height=790, hovermode="x unified", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(family="DM Sans, sans-serif", color="#34556f", size=12), hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dce7f2", font=dict(family="DM Sans, sans-serif", color="#214a6b", size=13)), legend=dict(orientation="h", y=1.12, x=0, title_text=""), margin=dict(l=55, r=20, t=110, b=45))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def show_adjustment_details(adaptation, forecast, enabled, tz=None):
    with st.expander("Station adjustment details"):
        if not enabled:
            st.write("Station adjustment is off. Forecasts use the selected base methods without corrections.")
        summary, details = [], []
        for label, meta in VARIABLES.items():
            items = adaptation.get(meta["key"], {})
            applied = forecast.loc[forecast.variable.eq(meta["key"]) & forecast.station_correction.abs().gt(1e-9)]
            status = "Off" if not enabled else ("Active" if len(applied) else "No correction applied")
            summary.append({"Parameter": label, "Adjusted horizons": f"{len(applied)}/24", "Historical examples (max per horizon)": max([item.get("examples", 0) for item in items.values()], default=0), "Median correction": float(applied.station_correction.median()) if len(applied) else 0.0, "Status": status})
            for horizon, item in items.items():
                flat = {key: value for key, value in item.items() if key != "source_counts"}
                details.append({"Parameter": label, "Horizon": horizon, **flat, **item.get("source_counts", {})})
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        if details:
            st.caption("Corrections require at least 12 training and 6 later validation examples, with overlapping future training labels excluded. Historical interpolation is excluded from this check. Corrections are only applied to complete, non-interpolated forecast inputs. This is an internal check, not independent validation.")
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
        st.dataframe(display_frame(forecast.drop(columns="variable"), tz=tz), use_container_width=True, hide_index=True)


def show_data_sources(history, sources, notes, issues, expanded=False, tz=None):
    with st.expander("Data sources and technical details", expanded=expanded):
        rows = []
        for label, meta in VARIABLES.items():
            series = history[meta["column"]].dropna()
            time = series.index[-1] if not series.empty else None
            rows.append({"Parameter": label, "Latest hourly reading": format_time(time, tz=tz), "Source": sources.at[time, meta["column"]] if time is not None else "Missing", "Hourly observations / estimates": len(series), "Notes": notes.get(meta["column"], "Direct provider measurements")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Completed hourly medians are labeled by hour start. Internal gaps of at most three hourly bins may use time interpolation, explicitly labeled as estimates. Entirely absent series and leading/trailing gaps are not filled. No forward filling or extrapolation is used.")
        if not history.empty:
            row = make_features(history, history.index[-1]).iloc[0]
            missing = row.index[~np.isfinite(row.to_numpy(dtype=float))].tolist()
            st.write("Missing inputs for the latest hour:", ", ".join(missing) if missing else "None")
        st.caption("The original 22-feature schema is retained. Unresolved inputs remain NaN for selected XGBoost models. Models are not retrained, and accuracy under missing-input patterns or geographic transfer has not been independently established. No calibrated uncertainty or hypoxia-event classifier is implemented.")
        st.caption("Salinity is no longer displayed or forecast. Existing models still use the original 22-feature schema: an unambiguous measured salinity series may remain a hidden input; unavailable input slots stay NaN. Removing those inputs requires retraining.")
        st.caption("Provisional observations may be revised. Separate sensor series are not automatically combined. Worldwide locations marked setup required are not connected feeds.")
        if issues:
            st.write("Model asset issues:")
            st.code("\n".join(issues))
    audit = []
    for label, meta in VARIABLES.items():
        col = meta["column"]
        mask = sources[col].astype(str).str.startswith("Interpolated")
        for time in sources.index[mask]:
            audit.append({"Parameter": label, "Hourly bin": format_time(time, tz=tz), "Hourly bin UTC": time.isoformat(), "Estimated value": history.at[time, col], "Source": sources.at[time, col]})
    with st.expander("Interpolation audit"):
        if audit:
            frame = pd.DataFrame(audit)
            st.warning(f"{len(frame)} hourly history values are interpolated estimates, not measured observations.")
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button("Download interpolation audit CSV", frame.to_csv(index=False).encode("utf-8"), file_name="aquasentry_interpolation_audit.csv", mime="text/csv")
        else:
            st.write("No hourly history values were interpolated.")


def show_brand_banner():
    logo_path = ROOT / "Logo.png"
    logo_html = ""
    if logo_path.is_file():
        try:
            encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            logo_html = '<img class="brand-logo" src="data:image/png;base64,' + encoded_logo + '" alt="AquaSentry logo">'
        except OSError:
            logo_html = ""
    st.markdown('<div class="hero">' + logo_html + '<div><h1>AquaSentry</h1><p>Predict Hypoxia. Prevent Hypoxia. Save Marine Ecosystems.</p></div></div>', unsafe_allow_html=True)


def load_selected_station(config, source_choice):
    station = config["id"]
    if not re.fullmatch(r"\d{8}(\d{7})?", station):
        st.info("Select a station or enter an 8- or 15-digit USGS station ID.")
        return None
    records, name = fetch_usgs(station, source_choice=source_choice)
    return records, name, "USGS Direct"


def main():
    global DISPLAY_TZ
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Monitoring station")
        choice = st.selectbox("Select station", list(STATIONS), label_visibility="collapsed")
        config = dict(STATIONS[choice])
        if config["provider"] == "custom":
            config = {"provider": "usgs", "id": st.text_input("USGS station ID").strip()}
        hours = st.selectbox("Chart history", [24, 48, 72], format_func=lambda value: f"Past {value} hours")
        if st.button("Refresh data", use_container_width=True):
            fetch_usgs.clear()
            build_station_adaptation.clear()
            get_forecast.clear()
        with st.expander("Settings"):
            adaptation_enabled = st.checkbox("Station adjustment", value=False, key="station_adjustment_enabled", help="Apply station-specific corrections only when they improve historical validation results.")
            zones = sorted(available_timezones())
            options = TIMEZONE_OPTIONS + [zone for zone in zones if zone not in TIMEZONE_OPTIONS]
            selected_timezone = st.selectbox("Display timezone", options, key="aquasentry_display_timezone")
            DISPLAY_TZ = STATION_TIMEZONES.get(config["id"], "UTC") if selected_timezone == "Station local (automatic)" else selected_timezone
            source_choice = st.selectbox("USGS data source", ["Legacy + current (auto)", "Current only"], key="aquasentry_usgs_source", help="Legacy API shuts down 2027-02-22.")
        check_feeds = st.button("Check all five station feeds", use_container_width=True)
    tz = DISPLAY_TZ
    show_brand_banner()
    station = config["id"].strip()
    try:
        with st.spinner("Loading station data…"):
            loaded = load_selected_station(config, source_choice)
            if loaded is None:
                return
            records, name, provider = loaded
            raw = pick_newest_series(records)
            history, sources, notes = hourly_history(raw, provider)
        if history.dropna(how="all").empty:
            raise ValueError("No supported completed hourly observations are available.")
    except Exception as exc:
        st.error(f"Could not load station data: {exc}")
        return
    st.markdown('<div class="station-id">' + html.escape(provider) + ' ' + html.escape(station) + '</div><div class="station-title">' + html.escape(name) + '</div>', unsafe_allow_html=True)
    st.caption(f"Latest hourly readings · {tz}")
    now = pd.Timestamp.now(tz="UTC")
    for box, (label, meta) in zip(st.columns(3), VARIABLES.items()):
        series = history[meta["column"]].dropna()
        with box:
            if series.empty:
                st.metric(label, "Unavailable")
                st.caption("No recent reading")
                continue
            time, value = series.index[-1], float(series.iloc[-1])
            prior = series.get(time - pd.Timedelta(hours=24), np.nan)
            unit = "" if meta["key"] == "ph" else " " + meta["unit"]
            change = value - float(prior) if pd.notna(prior) else None
            if change is not None and abs(change) < .005:
                change = 0.0
            delta = f"{change:+.2f}{unit} vs yesterday" if change is not None and change != 0 else None
            st.metric(label, f"{value:.2f}{unit}", delta=delta, delta_color="normal")
            if change == 0.0:
                st.caption("No change vs yesterday")
            st.caption(f'{sources.at[time, meta["column"]]} · {format_time(time, tz=tz)}')
            if now - (time + pd.Timedelta(hours=1)) > pd.Timedelta(hours=3):
                st.caption("Reading delayed")
    forecast, adaptation = None, {}
    assets = None
    asset_issues = []
    asset_error = None
    try:
        signature = asset_signature()
        with st.spinner("Loading models…"):
            assets = load_assets(signature)
            asset_issues = assets[3]
    except Exception as exc:
        asset_error = exc
        st.error(f"Model assets could not be loaded: {exc}")
    if assets is not None:
        try:
            message = "Updating forecast and station adjustment…" if adaptation_enabled else "Updating forecast…"
            with st.spinner(message):
                forecast, adaptation, issue = get_forecast(history, sources, station, assets, signature, adaptation_enabled, PIPELINE_VERSION)
            if issue:
                st.caption(issue)
        except Exception as exc:
            st.warning(f"Forecast unavailable: {exc}")
    if forecast is not None:
        origin = forecast.origin.iloc[0]
        age = max(0.0, (now - (origin + pd.Timedelta(hours=1))).total_seconds() / 3600)
        st.markdown("### Water-quality forecast")
        with st.expander("Forecast timing and input details", expanded=False):
            st.write("Last forecast generated: " + format_time(forecast.attrs.get("generated_at_utc"), tz=tz))
            st.write("Forecast reference hour starts: " + format_time(origin, tz=tz))
            st.caption("The reference is the start label of the latest completed hourly-median bin, not the current wall-clock time. That bin ends at " + format_time(origin + pd.Timedelta(hours=1), tz=tz) + ".")
            st.write("Prediction target times: " + format_time(forecast.timestamp.min(), tz=tz) + " through " + format_time(forecast.timestamp.max(), tz=tz))
            upcoming = forecast.loc[forecast.timestamp.gt(now)]
            if not upcoming.empty:
                target = upcoming.timestamp.min()
                st.write("Next future prediction target: " + format_time(target, tz=tz))
                current_table = upcoming.loc[upcoming.timestamp.eq(target), ["label", "prediction", "method_source"]].rename(columns={"label": "Parameter", "prediction": "Forecast value", "method_source": "Method"})
                current_table["Unit"] = current_table["Parameter"].map({label: meta["unit"] for label, meta in VARIABLES.items()})
                st.dataframe(current_table, use_container_width=True, hide_index=True)
            else:
                st.warning("No prediction targets remain in the future. Refresh data.")
            st.caption("Station adjustment: " + ("On" if adaptation_enabled else "Off"))
            if age > 3:
                st.warning("Data is delayed. This forecast starts from the displayed older origin, not from now.")
            missing = forecast.missing_inputs.iloc[0]
            estimated = forecast.interpolated_inputs.iloc[0]
            if missing:
                st.info("Selected models are running with some NaN legacy inputs; local skill for this pattern is unverified. Missing: " + missing)
            if estimated:
                st.warning("Measurements were unavailable for some inputs. Interpolation was used for: " + estimated + ". These values are estimates, not recorded readings.")
            elif sources.apply(lambda col: col.astype(str).str.startswith("Interpolated")).to_numpy().any():
                st.caption("Some historical values are interpolated estimates. See the marked chart points and interpolation audit.")
            omitted = [label for label, meta in VARIABLES.items() if meta["key"] not in set(forecast.variable)]
            if omitted:
                st.caption("No usable forecast for: " + ", ".join(omitted) + ". Other parameters continue.")
        draw_forecast(history, forecast, hours, sources, tz=tz)
        st.caption("Solid line: hourly history · Open diamonds: interpolated estimates · Dashed line: forecast")
        st.download_button("Download forecast CSV", forecast_export(forecast, tz=tz).to_csv(index=False).encode("utf-8"), file_name=f"aquasentry_{station}_forecast.csv", mime="text/csv", use_container_width=True)
        show_adjustment_details(adaptation, forecast, adaptation_enabled, tz=tz)
    with st.expander("Observation freshness and coverage", expanded=False):
        st.dataframe(selected_station_health(raw), use_container_width=True, hide_index=True)
        st.caption("Provider response retrieved: " + format_time(records.attrs.get("retrieved_at_utc"), tz=tz))
        st.caption("Coverage counts hours with samples; freshness is not a sensor-accuracy guarantee.")
        for provider_warning in records.attrs.get("provider_warnings", []):
            st.caption(provider_warning)
    if check_feeds:
        with st.status("Checking station feeds…", expanded=True) as status_holder:
            st.session_state["aquasentry_station_health"] = check_default_station_feeds(source_choice, progress_holder=status_holder)
            st.session_state["aquasentry_station_health_time"] = pd.Timestamp.now(tz="UTC").isoformat()
            status_holder.update(label="Station feed check complete", state="complete")
    if "aquasentry_station_health" in st.session_state:
        with st.expander("Five-station feed check", expanded=False):
            st.caption("Checked: " + format_time(st.session_state["aquasentry_station_health_time"], tz=tz))
            st.dataframe(display_frame(st.session_state["aquasentry_station_health"], tz=tz), use_container_width=True, hide_index=True)
    show_data_sources(history, sources, notes, asset_issues, expanded=forecast is None, tz=tz)
    st.markdown('<div class="footer">AquaSentry · Research forecasting dashboard</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
