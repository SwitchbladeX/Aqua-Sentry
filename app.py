from __future__ import annotations

import base64
import html
import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urljoin
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
INPUT_VARIABLES = {
    "Dissolved oxygen": {"key": "do", "folder": "dissolved_oxygen", "prefix": "DO", "column": "DO_mgL", "unit": "mg/L", "code": "00300"},
    "Temperature": {"key": "temp", "folder": "temperature", "prefix": "Temp", "column": "temp_C", "unit": "°C", "code": "00010"},
    "pH": {"key": "ph", "folder": "ph", "prefix": "pH", "column": "pH", "unit": "pH", "code": "00400"},
    "Salinity": {"key": "salinity", "folder": "salinity", "prefix": "Salinity", "column": "salinity_PSU", "unit": "PSU", "code": "90860"},
}
VARIABLES = {label: meta for label, meta in INPUT_VARIABLES.items() if meta["key"] != "salinity"}
VALUE_COLUMNS = [meta["column"] for meta in INPUT_VARIABLES.values()]
LAGS = [1, 3, 6, 12]
FEATURE_COLUMNS = VALUE_COLUMNS + [f"{col}_lag{lag}h" for col in VALUE_COLUMNS for lag in LAGS] + ["hour", "month"]
HIERARCHIES = {
    "do": "DO_VALIDATION_HIERARCHY.csv",
    "temp": "TEMP_VALIDATION_HIERARCHY.csv",
    "ph": "PH_VALIDATION_HIERARCHY.csv",
    "salinity": "SALINITY_VALIDATION_HIERARCHY.csv",
}
MIN_ADAPTATION_EXAMPLES = 12
MIN_VALIDATION_EXAMPLES = 6
FULL_ADAPTATION_EXAMPLES = 48
SALINITY_PERSISTENCE_ONLY = True
INTERPOLATION_MAX_GAP_HOURS = 3
PIPELINE_VERSION = "aquasentry-20261124-three-parameters-timezones-health"
STATIONS = {
    "Crabtree Swamp at Conway, SC — 02110701": {"provider": "usgs", "id": "02110701", "timezone": "America/New_York"},
    "Spring Creek near Spring, TX — 08068500": {"provider": "usgs", "id": "08068500", "timezone": "America/Chicago"},
    "West Harbor at Oyster Bay, NY — 01303152": {"provider": "usgs", "id": "01303152", "timezone": "America/New_York"},
    "Flax Pond at Old Field, NY — 01304057": {"provider": "usgs", "id": "01304057", "timezone": "America/New_York"},
    "Peconic River at County Hwy 105, NY — 01304562": {"provider": "usgs", "id": "01304562", "timezone": "America/New_York"},
    "Enter a different USGS station": {"provider": "custom", "id": "custom", "timezone": "UTC"},
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
FRESHNESS_HOURS = 3
TIMEZONE_CHOICES = ["Station local (automatic)", "UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "Europe/London", "Europe/Paris", "Asia/Kolkata", "Asia/Tokyo", "Australia/Sydney"]


def format_time(value, tz=None):
    if value is None or pd.isna(value):
        return "Unavailable"
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
    return timestamp.tz_convert(tz or DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


def display_frame(frame):
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.DatetimeTZDtype):
            result[column] = result[column].map(format_time)
    return result


def export_forecast(forecast):
    result = forecast.copy()
    for column in ["origin", "timestamp"]:
        if column in result:
            result[column + "_utc"] = result[column].map(lambda t: pd.Timestamp(t).isoformat())
            result[column + "_local"] = result[column].map(format_time)
            result = result.drop(columns=column)
    generated = forecast.attrs.get("generated_at_utc")
    result["generated_at_utc"] = generated
    result["generated_at_local"] = format_time(generated)
    result["display_timezone"] = DISPLAY_TZ
    return result


def observation_health(raw):
    now = pd.Timestamp.now(tz="UTC")
    recent_hours = pd.date_range(now.floor("h") - pd.Timedelta(days=7), now.floor("h"), freq="h", inclusive="left")
    rows = []
    for label, meta in VARIABLES.items():
        values = raw[meta["column"]].dropna().sort_index()
        values = values.loc[values.index <= now]
        latest = values.index[-1] if len(values) else None
        age = (now - latest).total_seconds() / 3600 if latest is not None else np.nan
        covered = values.resample("1h").count().reindex(recent_hours, fill_value=0).gt(0).sum() if len(values) else 0
        rows.append({"Parameter": label, "Latest raw observation": format_time(latest), "Age (hours)": round(age, 2) if np.isfinite(age) else np.nan, "Completed hours with samples / 168": f"{covered}/168", "Feed status": "Unavailable" if latest is None else ("Fresh" if age <= FRESHNESS_HOURS else "Delayed")})
    return pd.DataFrame(rows)


def check_default_stations():
    rows = []
    for label, config in STATIONS.items():
        if config["provider"] != "usgs":
            continue
        try:
            records, _ = fetch_usgs(config["id"], days=7)
            for parameter, meta in VARIABLES.items():
                group = records.loc[records["column"].eq(meta["column"])]
                now = pd.Timestamp.now(tz="UTC")
                group = group.loc[group["time"] <= now]
                latest = group["time"].max() if len(group) else pd.NaT
                age = (now - latest).total_seconds() / 3600 if pd.notna(latest) else np.nan
                rows.append({"Station": label, "Parameter": parameter, "Latest observation": format_time(latest), "Age (hours)": round(age, 2) if np.isfinite(age) else np.nan, "Separate series": group["series"].nunique(), "Status": "Unavailable" if pd.isna(latest) else ("Fresh series available" if age <= FRESHNESS_HOURS else "Delayed")})
        except Exception as exc:
            rows.append({"Station": label, "Parameter": "All", "Status": f"Provider request failed: {exc}"})
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs(site, days=90):
    site = normalize_station(site)
    if not re.fullmatch(r"\d{8,15}", site):
        raise ValueError("Use an 8- to 15-digit numeric USGS station ID.")
    mapping = {meta["code"]: meta["column"] for meta in INPUT_VARIABLES.values()}
    parts, names, failures = [], [], []
    try:
        response = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"format": "json", "sites": site, "parameterCd": ",".join(mapping), "period": f"P{days}D"},
            timeout=(10, 35), headers={"User-Agent": "AquaSentry/1.3"},
        )
        response.raise_for_status()
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
        failures.append(f"Legacy API: {type(exc).__name__}: {exc}")
    present = set(pd.concat(parts)["column"]) if parts else set()
    for code, col in mapping.items():
        if col in present:
            continue
        url = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
        params = {"f": "json", "monitoring_location_id": f"USGS-{site}", "parameter_code": code, "time": f"P{days}D", "limit": 10000}
        seen, pages = set(), []
        try:
            while url:
                response = requests.get(url, params=params, timeout=(10, 45), headers={"User-Agent": "AquaSentry/1.3"})
                response.raise_for_status()
                if response.url in seen:
                    raise ValueError("Repeated pagination URL; partial series discarded.")
                seen.add(response.url)
                payload = response.json()
                rows = []
                for feature in payload.get("features", []):
                    props = feature.get("properties", {})
                    if props.get("monitoring_location_id") != f"USGS-{site}" or str(props.get("parameter_code")) != code:
                        continue
                    rows.append({"time": props.get("time"), "value": props.get("value"), "column": col, "series": f"USGS time series {props.get('time_series_id', 'unknown')}"})
                if rows:
                    pages.append(pd.DataFrame(rows))
                link = next((entry.get("href") for entry in payload.get("links", []) if entry.get("rel") == "next"), None)
                url = urljoin(response.url, link) if link else None
                params = None
                if len(seen) > 100:
                    raise ValueError("Pagination safety limit reached; partial series discarded.")
            if pages:
                part = pd.concat(pages, ignore_index=True)
                part["time"] = pd.to_datetime(part["time"], utc=True, errors="coerce")
                part["value"] = clean_values(pd.to_numeric(part["value"], errors="coerce").mask(lambda s: s.le(-99990)), col)
                part = part.dropna(subset=["time", "value"])
                if not part.empty:
                    parts.append(part)
        except Exception as exc:
            if col != "salinity_PSU":
                failures.append(f"Modern API {code}: {type(exc).__name__}: {exc}")
    if not parts:
        raise ValueError("No supported observations returned. " + " | ".join(failures))
    result = pd.concat(parts, ignore_index=True).drop_duplicates(["time", "column", "series"], keep="last")
    result.attrs["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    result.attrs["provider_warnings"] = failures
    fallback_name = next((label.split(" — ")[0] for label, config in STATIONS.items() if config["id"] == site), f"USGS {site}")
    return result, names[0] if names else fallback_name


@st.cache_data(ttl=900, show_spinner=False)
def fetch_noaa(station, product, days=90):
    end = datetime.now(timezone.utc)
    cursor = end - timedelta(days=days)
    parts = []
    while cursor <= end:
        stop = min(cursor + timedelta(days=29), end)
        response = requests.get(
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            params={"product": product, "application": "AquaSentry", "begin_date": cursor.strftime("%Y%m%d"), "end_date": stop.strftime("%Y%m%d"), "station": station, "time_zone": "gmt", "units": "metric", "format": "json", "interval": "h"}, timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(payload["error"].get("message", "NOAA API error"))
        data = pd.DataFrame(payload.get("data", []))
        if {"t", "v"}.issubset(data.columns):
            parts.append(pd.DataFrame({"time": pd.to_datetime(data["t"], utc=True, errors="coerce"), "value": pd.to_numeric(data["v"], errors="coerce")}))
        cursor = datetime.combine(stop.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if not parts:
        raise ValueError(f"No NOAA {product} observations returned.")
    series = pd.concat(parts).dropna().groupby("time")["value"].median().sort_index()
    return clean_values(series, "salinity_PSU" if product == "salinity" else "temp_C").dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def erddap_metadata(server, dataset):
    response = requests.get(f"{server}/info/{dataset}/index.json", timeout=60)
    response.raise_for_status()
    table = response.json()["table"]
    frame = pd.DataFrame(table["rows"], columns=table["columnNames"])
    attrs = {}
    for _, row in frame.loc[frame["Row Type"].eq("attribute")].iterrows():
        attrs.setdefault(row["Variable Name"], {})[row["Attribute Name"]] = row["Value"]
    return attrs


def erddap_factor(col, attrs):
    unit = str(attrs.get("units", "")).lower().replace(" ", "").replace("µ", "u").replace("μ", "u")
    if col == "temp_C" and unit in ("degc", "degree_celsius", "degrees_celsius", "celsius", "°c"):
        return 1.0
    if col == "salinity_PSU" and unit in ("1", "psu", "pss-78", "1e-3", "0.001"):
        return 1.0
    if col == "DO_mgL":
        if unit in ("mg/l", "mgl-1", "mg.l-1", "g/m3", "gm-3"):
            return 1.0
        if unit in ("umol/l", "umoll-1", "mmol/m3", "mmolm-3", "mmolm^-3"):
            return 0.031998
        if unit in ("mol/m3", "molm-3"):
            return 31.998
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_erddap(server, dataset, selections, depth, days=90):
    attrs = erddap_metadata(server, dataset)
    fields = ["time", "depth", "midgid"] + sorted(set(selections.values()))
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = ",".join(fields) + f'&time>={start}&depth={float(depth)}&orderBy("time")'
    response = requests.get(f"{server}/tabledap/{dataset}.csv?" + quote(query, safe=',&=><()'), timeout=120)
    response.raise_for_status()
    data = pd.read_csv(io.StringIO(response.text), skiprows=[1])
    if data.empty:
        raise ValueError("No recent data at this exact sensor depth.")
    if data.groupby("time")["midgid"].nunique().gt(1).any():
        raise ValueError("Overlapping deployments need explicit selection before these data can be combined.")
    parts = []
    for col, field in selections.items():
        factor = erddap_factor(col, attrs[field])
        if factor is None:
            continue
        values = pd.to_numeric(data[field], errors="coerce")
        for key in ("_FillValue", "missing_value"):
            sentinel = pd.to_numeric(attrs[field].get(key), errors="coerce")
            if pd.notna(sentinel):
                values = values.mask(values == sentinel)
        parts.append(pd.DataFrame({"time": pd.to_datetime(data["time"], utc=True, errors="coerce"), "column": col, "value": clean_values(values * factor, col), "series": f"{field}; depth {depth} m"}))
    if not parts:
        raise ValueError("No supported sensors selected.")
    return pd.concat(parts, ignore_index=True).dropna(subset=["time", "value"]), "Mace Head Met-Ocean Buoy, Ireland"


def select_sensor_series(records, station):
    parts = []
    with st.sidebar.expander("Sensor / series selection"):
        st.caption("Separate sensor series are not automatically mixed.")
        for label, meta in VARIABLES.items():
            group = records.loc[records["column"].eq(meta["column"])]
            choices = sorted(group["series"].unique())
            if not choices:
                continue
            selected = st.selectbox(label, choices, key=f"sensor_{station}_{meta['key']}") if len(choices) > 1 else choices[0]
            part = group.loc[group["series"].eq(selected)].groupby("time")["value"].median()
            parts.append(part.rename(meta["column"]))
    auxiliary = records.loc[records["column"].eq("salinity_PSU")]
    if auxiliary["series"].nunique() == 1:
        parts.append(auxiliary.groupby("time")["value"].median().rename("salinity_PSU"))
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


def hourly_history(raw, provider, noaa_station):
    raw = raw.copy().sort_index()
    raw.index = pd.to_datetime(raw.index, utc=True)
    hourly = raw.resample("1h").median().reindex(columns=VALUE_COLUMNS)
    hourly = hourly.loc[hourly.index < pd.Timestamp.now(tz="UTC").floor("h")]
    sources = pd.DataFrame("Missing", index=hourly.index, columns=VALUE_COLUMNS).mask(hourly.notna(), provider)
    notes = {}
    for col, product in [("temp_C", "water_temperature")]:
        filled = 0
        if noaa_station.strip() and hourly[col].isna().any():
            try:
                fallback = fetch_noaa(noaa_station.strip(), product).resample("1h").median().reindex(hourly.index)
                mask = hourly[col].isna() & fallback.notna()
                hourly.loc[mask, col] = fallback.loc[mask]
                sources.loc[mask, col] = "NOAA CO-OPS"
                filled = int(mask.sum())
            except Exception as exc:
                notes[col] = f"NOAA fallback unavailable: {exc}"
        notes.setdefault(col, f"NOAA filled {filled} hourly gaps" if noaa_station.strip() else "NOAA fallback not configured")
    hourly, sources = interpolate_history(hourly, sources)
    for col in VALUE_COLUMNS:
        count = int(sources[col].astype(str).str.startswith("Interpolated").sum())
        if count:
            notes[col] = (notes.get(col, "") + "; " if notes.get(col) else "") + f"{count} interpolated hourly estimates"
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
    if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
        return "persistence"
    return choose_method(hierarchy, metrics, station, horizon)


def base_predictions(key, horizon, features, baseline, assets, method):
    baseline = np.asarray(baseline, dtype=float)
    prediction = baseline.copy()
    sources = np.full(len(baseline), "Persistence", dtype=object)
    if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
        sources[:] = "Persistence (salinity ML disabled)"
        return prediction, sources
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
            if key == "salinity":
                item["status"] = "Disabled: salinity persistence-only policy"
                continue
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
            train_mask = (origins[:split] + pd.Timedelta(hours=horizon + 1)) <= origins[split]
            train, validation = errors[:split][train_mask], errors[split:]
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
    if pd.Timestamp.now(tz="UTC") - (origin + pd.Timedelta(hours=1)) > pd.Timedelta(hours=FRESHNESS_HOURS):
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
            if not missing and not estimated and key != "salinity" and source in ("XGBoost", "Persistence"):
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
def get_forecast(history, sources, station, _assets, signature, adaptation_enabled):
    adaptation, issue = {}, None
    if adaptation_enabled:
        try:
            adaptation = build_station_adaptation(history, sources, station, signature, PIPELINE_VERSION, _assets)
        except Exception as exc:
            issue = f"Station adjustment unavailable; base forecasts remain active: {exc}"
    forecast = make_forecast(history, sources, station, _assets, adaptation)
    forecast.attrs["generated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    return forecast, adaptation, issue


def draw_forecast(history, forecast, hours, sources):
    origin = forecast.origin.iloc[0]
    start, end = origin - pd.Timedelta(hours=hours), origin + pd.Timedelta(hours=24)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.10, subplot_titles=[f'{label} · {meta["unit"]}' for label, meta in VARIABLES.items()])
    shown = set()
    for row, (label, meta) in enumerate(VARIABLES.items(), 1):
        observed = history.loc[(history.index >= start) & (history.index <= origin), meta["column"]]
        subset = forecast.loc[forecast.variable.eq(meta["key"])].sort_values("horizon")
        provenance = sources[meta["column"]].reindex(observed.index).fillna("Missing")
        details = np.array([[format_time(t), str(p)] for t, p in zip(observed.index, provenance)], dtype=object).reshape(-1, 2)
        unit = "" if meta["key"] == "ph" else " " + meta["unit"]
        fig.add_trace(go.Scatter(x=observed.index, y=observed, mode="lines", connectgaps=False, name="Completed hourly history", legendgroup="history", showlegend=row == 1, line=dict(color="#246fba", width=2.8), customdata=details, hovertemplate="%{customdata[0]}<br>%{y:.2f}" + unit + "<br>%{customdata[1]}<extra>History</extra>"), row=row, col=1)
        mask = provenance.astype(str).str.startswith("Interpolated")
        if mask.any():
            fig.add_trace(go.Scatter(x=observed.index[mask], y=observed.loc[mask], mode="markers", name="Interpolated estimate", legendgroup="interpolation", showlegend="interpolation" not in shown, marker=dict(size=7, symbol="diamond-open", color="#d78c1e"), customdata=[[format_time(t)] for t in observed.index[mask]], hovertemplate="%{customdata[0]}<br>%{y:.2f}" + unit + "<extra>Interpolated, not measured</extra>"), row=row, col=1)
            shown.add("interpolation")
        if subset.empty:
            continue
        if subset.station_correction.abs().gt(1e-9).any():
            fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.original_prediction, mode="lines", name="Before station adjustment", legendgroup="base", showlegend="base" not in shown, line=dict(color="#9aaec2", width=1.6, dash="dot"), customdata=[[format_time(t)] for t in subset.timestamp], hovertemplate="%{customdata[0]}<br>%{y:.2f}" + unit + "<extra>Before adjustment</extra>"), row=row, col=1)
            shown.add("base")
        custom = [[format_time(t), method, correction] for t, method, correction in zip(subset.timestamp, subset.method_source, subset.station_correction)]
        fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.prediction, mode="lines+markers", name="Forecast", legendgroup="forecast", showlegend="forecast" not in shown, customdata=custom, line=dict(color="#16a394", width=2.8, dash="dash"), marker=dict(size=4), hovertemplate="%{customdata[0]}<br>%{y:.2f}" + unit + "<br>%{customdata[1]}<br>Adjustment: %{customdata[2]:+.3f}<extra>Forecast</extra>"), row=row, col=1)
        shown.add("forecast")
    fig.add_vrect(x0=origin, x1=end, fillcolor="#246fba", opacity=.035, line_width=0)
    fig.add_vline(x=origin, line_dash="dot", line_color="#95a9bb", line_width=1.4)
    fig.add_annotation(x=origin, y=1.07, xref="x", yref="paper", text="FORECAST REFERENCE HOUR", showarrow=False, font=dict(size=10, color="#70869a"))
    ticks = pd.date_range(start.ceil("6h"), end, freq="6h")
    ticktext = [t.tz_convert(DISPLAY_TZ).strftime("%b %d<br>%H:%M %Z") for t in ticks]
    fig.update_xaxes(range=[start, end], tickmode="array", tickvals=list(ticks), ticktext=ticktext, showline=False, zeroline=False, gridcolor="#edf2f7", tickfont=dict(size=11, color="#70869a"))
    fig.update_xaxes(title_text=f"Time · {DISPLAY_TZ}", row=3, col=1)
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="#edf2f7", tickfont=dict(size=11, color="#70869a"), nticks=5)
    fig.update_layout(height=790, hovermode="x unified", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(family="DM Sans, sans-serif", color="#34556f", size=12), legend=dict(orientation="h", y=1.15, x=0), margin=dict(l=55, r=20, t=120, b=45))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def show_adjustment_details(adaptation, forecast, enabled):
    with st.expander("Station adjustment details"):
        if not enabled:
            st.write("Station adjustment is off. Forecasts use the selected base methods without corrections.")
        summary, details = [], []
        for label, meta in VARIABLES.items():
            items = adaptation.get(meta["key"], {})
            applied = forecast.loc[forecast.variable.eq(meta["key"]) & forecast.station_correction.abs().gt(1e-9)]
            status = "Persistence only" if meta["key"] == "salinity" else ("Off" if not enabled else ("Active" if len(applied) else "No correction applied"))
            summary.append({"Parameter": label, "Adjusted horizons": f"{len(applied)}/24", "Historical examples (max per horizon)": max([item.get("examples", 0) for item in items.values()], default=0), "Median correction": float(applied.station_correction.median()) if len(applied) else 0.0, "Status": status})
            for horizon, item in items.items():
                flat = {key: value for key, value in item.items() if key != "source_counts"}
                details.append({"Parameter": label, "Horizon": horizon, **flat, **item.get("source_counts", {})})
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        if details:
            st.caption("Corrections require at least 12 training and 6 later validation examples, with overlapping future training labels excluded. Historical interpolation is excluded from this check. Corrections are only applied to complete, non-interpolated forecast inputs. This is an internal check, not independent validation.")
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
        st.dataframe(display_frame(forecast.drop(columns="variable")), use_container_width=True, hide_index=True)


def show_data_sources(history, sources, notes, issues, expanded=False):
    with st.expander("Data sources and technical details", expanded=expanded):
        rows = []
        for label, meta in VARIABLES.items():
            series = history[meta["column"]].dropna()
            timestamp = series.index[-1] if len(series) else None
            rows.append({"Parameter": label, "Latest hourly bin start": format_time(timestamp), "Source": sources.at[timestamp, meta["column"]] if timestamp is not None else "Missing", "Hourly observations / estimates": len(series), "Notes": notes.get(meta["column"], "Direct provider measurements")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Hourly medians are labeled by hour start, not by the exact instant of the last sample. Only completed hours enter forecasting. Internal gaps of at most three hours may be interpolated and are marked as estimates; absent series and trailing gaps are never fabricated.")
        st.caption("Existing regression models retain their trained 22-feature input schema. Salinity is not displayed or forecast and no salinity model is loaded. If unambiguous measured salinity exists, it is retained only as a legacy predictor for the remaining models; otherwise its input slots remain NaN. This is not a retrained salinity-free model.")
        st.caption("Station adjustment is a regression-bias correction checked on later historical examples. It is not the experimental hypoxia-threshold adaptation mechanism and does not establish calibrated hypoxia probabilities. No hypoxia percentage is fabricated from these regression forecasts.")
        st.caption("Provisional data may be revised. A fresh feed is not proof of sensor accuracy or forecast skill. Display timezone changes do not change model inputs, UTC feature hours, or forecast values.")
        if issues:
            st.write("Model asset issues:")
            st.code("\n".join(issues))
    audit = []
    for label, meta in VARIABLES.items():
        col = meta["column"]
        mask = sources[col].astype(str).str.startswith("Interpolated")
        for timestamp in sources.index[mask]:
            audit.append({"Parameter": label, "Hourly bin start": format_time(timestamp), "Hourly bin start UTC": timestamp.isoformat(), "Estimated value": history.at[timestamp, col], "Source": sources.at[timestamp, col]})
    with st.expander("Interpolation audit"):
        if audit:
            frame = pd.DataFrame(audit)
            st.warning(f"{len(frame)} displayed hourly values are interpolated estimates, not recorded measurements.")
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button("Download interpolation audit CSV", frame.to_csv(index=False).encode("utf-8"), file_name="aquasentry_interpolation_audit.csv", mime="text/csv")
        else:
            st.write("No displayed hourly history values were interpolated.")


def show_brand_banner():
    logo_path = ROOT / "Logo.png"
    logo_html = ""
    if logo_path.is_file():
        try:
            encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            logo_html = '<img class="brand-logo" src="data:image/png;base64,' + encoded_logo + '" alt="AquaSentry logo">'
        except OSError:
            logo_html = ""
    st.markdown('<div class="hero">' + logo_html + '<div><h1>AquaSentry</h1><p>Monitor water quality. Explore forecasts. Support aquatic research.</p></div></div>', unsafe_allow_html=True)


def load_selected_station(config):
    station = config["id"]
    if not re.fullmatch(r"\d{8,15}", station):
        st.info("Select a station or enter an 8- to 15-digit USGS station ID.")
        return None
    records, name = fetch_usgs(station)
    return records, name, "USGS Direct"


def main():
    global DISPLAY_TZ
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Monitoring station")
        choice = st.selectbox("Select station", list(STATIONS), label_visibility="collapsed")
        config = dict(STATIONS[choice])
        if config["provider"] == "custom":
            config = {"provider": "usgs", "id": st.text_input("USGS station ID").strip(), "timezone": "UTC"}
        hours = st.selectbox("Chart history", [24, 48, 72], format_func=lambda value: f"Past {value} hours")
        with st.expander("Settings"):
            available = sorted(available_timezones())
            choices = TIMEZONE_CHOICES + [zone for zone in available if zone not in TIMEZONE_CHOICES]
            selected_tz = st.selectbox("Display timezone", choices, key="display_timezone_choice", help="Changes timestamps only. Computation and model feature hours stay in UTC; daylight saving is handled by the named timezone.")
            DISPLAY_TZ = config.get("timezone", "UTC") if selected_tz == "Station local (automatic)" else selected_tz
            st.caption(f"Active timezone: {DISPLAY_TZ}")
            adaptation_enabled = st.checkbox("Station adjustment", value=True, key="station_adjustment_enabled", help="Apply regression-bias corrections only when they improve the existing historical validation check. This is not a hypoxia-probability classifier.")
            noaa = st.text_input("Optional NOAA temperature fallback station", key="noaa_fallback_station", help="Use only a station demonstrably appropriate for this water body. This does not supply pH or dissolved oxygen.")
        if st.button("Refresh data and forecast", use_container_width=True):
            fetch_usgs.clear()
            fetch_noaa.clear()
            build_station_adaptation.clear()
            get_forecast.clear()
        check_all = st.button("Check all five station feeds", use_container_width=True)
    show_brand_banner()
    st.caption("Research forecasts of dissolved oxygen, water temperature, and pH. Station-specific hypoxia probabilities are not yet deployed.")
    if check_all:
        with st.spinner("Checking current parameter availability at all five stations…"):
            st.session_state["station_feed_health"] = check_default_stations()
            st.session_state["station_feed_health_checked_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    if "station_feed_health" in st.session_state:
        with st.expander("Five-station feed check", expanded=check_all):
            st.caption("Checked " + format_time(st.session_state["station_feed_health_checked_at"]) + ". Freshness is not a sensor-accuracy guarantee. Select a station to inspect individual series.")
            st.dataframe(st.session_state["station_feed_health"], use_container_width=True, hide_index=True)
    station = config["id"].strip()
    try:
        with st.spinner("Loading station data…"):
            loaded = load_selected_station(config)
            if loaded is None:
                return
            records, name, provider = loaded
            raw = select_sensor_series(records, station)
            history, sources, notes = hourly_history(raw, provider, noaa)
        if history[[meta["column"] for meta in VARIABLES.values()]].dropna(how="all").empty:
            raise ValueError("No supported completed hourly observations are available.")
    except Exception as exc:
        st.error(f"Could not load station data: {exc}")
        st.info("An unavailable feed is not replaced with fabricated readings. Try Refresh or another station.")
        return
    st.markdown('<div class="station-id">' + html.escape(provider) + ' ' + html.escape(station) + '</div><div class="station-title">' + html.escape(name) + '</div>', unsafe_allow_html=True)
    st.caption(f"Latest completed hourly readings · {DISPLAY_TZ}")
    for box, (label, meta) in zip(st.columns(3), VARIABLES.items()):
        series = history[meta["column"]].dropna()
        with box:
            if series.empty:
                st.metric(label, "Unavailable")
                st.caption("No supported reading")
                continue
            timestamp, value = series.index[-1], float(series.iloc[-1])
            prior = series.get(timestamp - pd.Timedelta(hours=24), np.nan)
            unit = "" if meta["key"] == "ph" else " " + meta["unit"]
            change = value - float(prior) if pd.notna(prior) else None
            delta = f"{change:+.2f}{unit} vs 24 h earlier" if change is not None and abs(change) >= .005 else None
            st.metric(label, f"{value:.2f}{unit}", delta=delta)
            st.caption(f'{sources.at[timestamp, meta["column"]]} · hour starting {format_time(timestamp)}')
    with st.expander("Observation freshness and coverage", expanded=False):
        st.caption("Raw observations, before interpolation. Coverage counts hours with at least one selected sample; it is not a laboratory quality assessment.")
        st.dataframe(observation_health(raw), use_container_width=True, hide_index=True)
        st.caption("Provider response retrieved: " + format_time(records.attrs.get("retrieved_at_utc")))
        for warning in records.attrs.get("provider_warnings", []):
            st.caption(warning)
    forecast, adaptation, assets = None, {}, None
    try:
        signature = asset_signature()
        with st.spinner("Preparing forecast and station adjustment…"):
            assets = load_assets(signature)
            forecast, adaptation, issue = get_forecast(history, sources, station, assets, signature, adaptation_enabled)
        if issue:
            st.caption(issue)
    except Exception as exc:
        st.warning(f"Forecast unavailable: {exc}")
    if forecast is not None:
        origin = forecast.origin.iloc[0]
        generated = forecast.attrs.get("generated_at_utc")
        now = pd.Timestamp.now(tz="UTC")
        st.markdown("### Water-quality forecast")
        st.write("Last forecast generated: " + format_time(generated))
        st.write("Forecast reference hour starts: " + format_time(origin))
        st.caption("The reference is the start label of the latest completed hourly-median bin, not the current wall-clock time. That bin ends at " + format_time(origin + pd.Timedelta(hours=1)) + ".")
        st.write("Prediction target times: " + format_time(forecast.timestamp.min()) + " through " + format_time(forecast.timestamp.max()))
        upcoming = forecast.loc[forecast.timestamp.gt(now)]
        if upcoming.empty:
            st.warning("No targets in this forecast are still in the future. Refresh the provider data.")
        else:
            target = upcoming.timestamp.min()
            current_table = upcoming.loc[upcoming.timestamp.eq(target), ["label", "prediction", "method_source"]].rename(columns={"label": "Parameter", "prediction": "Forecast value", "method_source": "Method"})
            st.write("Next future prediction target: " + format_time(target))
            st.dataframe(current_table, use_container_width=True, hide_index=True)
        missing = forecast.missing_inputs.iloc[0]
        estimated = forecast.interpolated_inputs.iloc[0]
        if missing:
            st.info("Selected models are running with some NaN legacy inputs; local skill for this pattern is unverified. Missing: " + missing)
        if estimated:
            st.warning("Interpolated estimates are used for some inputs: " + estimated)
        omitted = [label for label, meta in VARIABLES.items() if meta["key"] not in set(forecast.variable)]
        if omitted:
            st.caption("No current supported forecast for: " + ", ".join(omitted))
        draw_forecast(history, forecast, hours, sources)
        st.caption("Solid: completed hourly history · Open diamonds: interpolated estimates · Dashed: forecast · Full timestamps and UTC offsets appear on hover.")
        st.download_button("Download forecast CSV", export_forecast(forecast).to_csv(index=False).encode("utf-8"), file_name=f"aquasentry_{station}_forecast.csv", mime="text/csv", use_container_width=True)
        show_adjustment_details(adaptation, forecast, adaptation_enabled)
    show_data_sources(history, sources, notes, assets[3] if assets is not None else [], expanded=forecast is None)
    st.markdown('<div class="footer">AquaSentry · Research forecasting dashboard</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
