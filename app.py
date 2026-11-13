from __future__ import annotations

import base64
import html
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

st.set_page_config(page_title="AquaSentry", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
VARIABLES = {
    "Dissolved oxygen": {"key": "do", "folder": "dissolved_oxygen", "prefix": "DO", "column": "DO_mgL", "unit": "mg/L", "code": "00300"},
    "Temperature": {"key": "temp", "folder": "temperature", "prefix": "Temp", "column": "temp_C", "unit": "°C", "code": "00010"},
    "pH": {"key": "ph", "folder": "ph", "prefix": "pH", "column": "pH", "unit": "pH", "code": "00400"},
    "Salinity": {"key": "salinity", "folder": "salinity", "prefix": "Salinity", "column": "salinity_PSU", "unit": "PSU", "code": "90860"},
}
VALUE_COLUMNS = [meta["column"] for meta in VARIABLES.values()]
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
PIPELINE_VERSION = "aquasentry-2026-ui-2"
STATIONS = {
    "West Harbor at Oyster Bay, NY — 01303152": "01303152",
    "Peconic River at Riverhead, NY — 01304562": "01304562",
    "South side of Shelter Island, NY — 01304200": "01304200",
    "Orient Point Harbor, NY — 01304190": "01304190",
    "Forge River at Mastic, NY — 01304057": "01304057",
    "Hempstead Bay at Freeport, NY — 01304998": "01304998",
    "Enter a different USGS station": "custom",
}
# Retained, but coordinates are not used to infer station identity or co-location.
STATION_METADATA = {
    "01303152": {"name": "West Harbor at Oyster Bay, NY", "lat": 40.8784, "lon": -73.5352},
    "01304562": {"name": "Peconic River at Riverhead, NY", "lat": 40.9234, "lon": -72.6687},
    "01304200": {"name": "South side of Shelter Island, NY", "lat": 41.0543, "lon": -72.3382},
    "01304190": {"name": "Orient Point Harbor, NY", "lat": 41.1440, "lon": -72.2351},
    "01304057": {"name": "Forge River at Mastic, NY", "lat": 40.7812, "lon": -72.8590},
    "01304998": {"name": "Hempstead Bay at Freeport, NY", "lat": 40.6354, "lon": -73.5801},
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


def normalize_station(value) -> str:
    value = str(value).strip().replace("USGS-", "").removesuffix(".0")
    return value.zfill(8) if value.isdigit() and len(value) <= 8 else value


@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs(site: str, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    response = requests.get(
        "https://waterservices.usgs.gov/nwis/iv/",
        params={"format": "json", "sites": site, "parameterCd": "00010,00300,00400,90860", "period": f"P{days}D"},
        timeout=45, headers={"User-Agent": "AquaSentry/1.1"},
    )
    response.raise_for_status()
    series = response.json().get("value", {}).get("timeSeries", [])
    mapping = {meta["code"]: meta["column"] for meta in VARIABLES.values()}
    parts, names = [], []
    for item in series:
        codes = item.get("variable", {}).get("variableCode", [])
        col = mapping.get(codes[0].get("value")) if codes else None
        if col is None:
            continue
        name = item.get("sourceInfo", {}).get("siteName")
        if name:
            names.append(name)
        sentinel = pd.to_numeric(item.get("variable", {}).get("noDataValue"), errors="coerce")
        for group in item.get("values", []):
            data = pd.DataFrame(group.get("value", []))
            if not {"dateTime", "value"}.issubset(data.columns):
                continue
            times = pd.to_datetime(data["dateTime"], utc=True, errors="coerce")
            values = pd.to_numeric(data["value"], errors="coerce")
            values = values.where(np.isfinite(values))
            if pd.notna(sentinel):
                values = values.mask(values == sentinel)
            if col in ("DO_mgL", "salinity_PSU"):
                values = values.mask(values < 0)
            if col == "pH":
                values = values.mask((values < 0) | (values > 14))
            frame = pd.DataFrame({"time": times, "column": col, "value": values}).dropna()
            if not frame.empty:
                parts.append(frame)
    if not parts:
        raise ValueError("USGS returned no supported water-quality observations.")
    raw = pd.concat(parts, ignore_index=True).pivot_table(
        index="time", columns="column", values="value", aggfunc="median"
    ).sort_index()
    raw.columns.name = None
    raw.attrs["station_name"] = names[0] if names else f"USGS {site}"
    provenance = pd.DataFrame("Missing", index=raw.index, columns=raw.columns)
    provenance = provenance.mask(raw.notna(), "USGS Direct")
    return raw, provenance


@st.cache_data(ttl=900, show_spinner=False)
def fetch_noaa(station: str, product: str, days: int = 90) -> pd.Series:
    end = datetime.now(timezone.utc)
    cursor = end - timedelta(days=days)
    parts = []
    while cursor <= end:
        stop = min(cursor + timedelta(days=29), end)
        response = requests.get(
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            params={
                "product": product, "application": "AquaSentry",
                "begin_date": cursor.strftime("%Y%m%d"), "end_date": stop.strftime("%Y%m%d"),
                "station": station, "time_zone": "gmt", "units": "metric",
                "format": "json", "interval": "h",
            }, timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(payload["error"].get("message", "NOAA API error"))
        data = pd.DataFrame(payload.get("data", []))
        if {"t", "v"}.issubset(data.columns):
            parts.append(pd.DataFrame({
                "time": pd.to_datetime(data["t"], utc=True, errors="coerce"),
                "value": pd.to_numeric(data["v"], errors="coerce"),
            }).dropna())
        cursor = datetime.combine(stop.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if not parts:
        raise ValueError(f"No NOAA {product} observations returned.")
    result = pd.concat(parts).set_index("time")["value"].groupby(level=0).median().sort_index()
    result = result.where(np.isfinite(result))
    if product == "salinity":
        result = result.where(result >= 0)
    return result.dropna()


def hourly_history(raw, provenance, noaa_station):
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index()
    hourly = raw.resample("1h").median().reindex(columns=VALUE_COLUMNS)
    # Preserve the trained bin-start convention; exclude the unfinished current hour.
    hourly = hourly.loc[hourly.index < pd.Timestamp.now(tz="UTC").floor("h")]
    sources = pd.DataFrame("Missing", index=hourly.index, columns=VALUE_COLUMNS)
    sources = sources.mask(hourly.notna(), "USGS Direct")
    notes = {}
    for col, product in [("temp_C", "water_temperature"), ("salinity_PSU", "salinity")]:
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
    return hourly, sources, notes


def asset_signature():
    paths = sorted(MODEL_ROOT.rglob("*.json")) + sorted(MODEL_ROOT.rglob("*.csv"))
    return tuple((str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


@st.cache_resource(show_spinner=False)
def load_assets(signature):
    models, hierarchies, metrics = {}, {}, {}
    for meta in VARIABLES.values():
        key, folder = meta["key"], MODEL_ROOT / meta["folder"]
        models[key] = {}
        for horizon in range(1, 25):
            path = folder / f'{meta["prefix"]}_{horizon}h.json'
            if not path.exists():
                raise FileNotFoundError(f"Missing model: {path.relative_to(ROOT)}")
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            booster = model.get_booster()
            if booster.num_features() != len(FEATURE_COLUMNS):
                raise ValueError(f"{path.name}: expected 22 features, found {booster.num_features()}.")
            if booster.feature_names and booster.feature_names != FEATURE_COLUMNS:
                raise ValueError(f"{path.name}: model feature names/order differ from app schema.")
            models[key][horizon] = model
        path = folder / HIERARCHIES[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing hierarchy: {path.relative_to(ROOT)}")
        hierarchies[key] = pd.read_csv(path, dtype=str)
        path = folder / HIERARCHIES[key].replace("HIERARCHY", "METHOD_METRICS")
        metrics[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return models, hierarchies, metrics


def make_features(history, timestamp):
    row = {col: history.at[timestamp, col] for col in VALUE_COLUMNS}
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            time = timestamp - pd.Timedelta(hours=lag)
            row[f"{col}_lag{lag}h"] = history.at[time, col] if time in history.index else np.nan
    row.update(hour=timestamp.hour, month=timestamp.month)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=float)


def build_feature_history(history):
    result = history.reindex(columns=VALUE_COLUMNS).copy()
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            wanted = history.index - pd.Timedelta(hours=lag)
            result[f"{col}_lag{lag}h"] = history[col].reindex(wanted).to_numpy()
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
        rows = hierarchy[
            (hierarchy[scol].map(normalize_station) == normalize_station(station))
            & (pd.to_numeric(hierarchy[hcol], errors="coerce") == horizon)
        ]
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
    models, _, _ = assets
    baseline = np.asarray(baseline, dtype=float)
    prediction = baseline.copy()
    sources = np.full(len(baseline), "Persistence", dtype=object)
    if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
        sources[:] = "Persistence (salinity ML disabled)"
        return prediction, sources
    if method == "xgboost":
        candidate = np.asarray(models[key][horizon].predict(features), dtype=float)
        finite = np.isfinite(candidate)
        prediction[finite] = candidate[finite]
        sources[finite] = "XGBoost"
        sources[~finite] = "Persistence (non-finite model output)"
        if key == "salinity":
            guard = (baseline > 5) & (candidate < baseline * 0.5)
            prediction[guard] = baseline[guard]
            sources[guard] = "Persistence (salinity model scale guard)"
    return prediction, sources


@st.cache_data(ttl=900, show_spinner=False)
def build_station_adaptation(history, hourly_provenance, station, signature, policy_version, _assets):
    _, hierarchies, metrics = _assets
    features = build_feature_history(history)
    adaptation = {}
    for meta in VARIABLES.values():
        key, col = meta["key"], meta["column"]
        adaptation[key] = {}
        for horizon in range(1, 25):
            method = effective_method(key, hierarchies[key], metrics[key], station, horizon)
            item = {
                "correction": 0.0, "examples": 0, "median_error": 0.0, "method": method,
                "validated": False, "validation_mae_before": None, "validation_mae_after": None,
                "validation_examples": 0, "training_examples": 0,
                "source_counts": {"USGS Direct": 0, "NOAA CO-OPS": 0}, "status": "Insufficient examples",
            }
            adaptation[key][horizon] = item
            if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
                item["status"] = "Disabled: salinity persistence-only policy"
                continue
            if features.empty:
                continue
            targets = features.index + pd.Timedelta(hours=horizon)
            actual = history[col].reindex(targets).to_numpy(dtype=float)
            sources = hourly_provenance[col].reindex(targets).to_numpy()
            valid = np.isfinite(actual) & np.isin(sources, ["USGS Direct", "NOAA CO-OPS"])
            origins = features.index[valid]
            if not len(origins):
                continue
            predicted, _ = base_predictions(
                key, horizon, features.loc[origins], history[col].reindex(origins).to_numpy(), _assets, method,
            )
            errors = actual[valid] - predicted
            finite = np.isfinite(errors)
            errors, origins, sources = errors[finite], origins[finite], sources[valid][finite]
            count = len(errors)
            item.update(
                examples=count, median_error=float(np.median(errors)) if count else 0.0,
                source_counts={source: int(np.sum(sources == source)) for source in ("USGS Direct", "NOAA CO-OPS")},
            )
            if count < MIN_ADAPTATION_EXAMPLES + MIN_VALIDATION_EXAMPLES:
                continue
            split = min(int(count * .7), count - MIN_VALIDATION_EXAMPLES)
            validation_start = origins[split]
            # A target-bin median is only available after that hour has ended.
            train_mask = (origins[:split] + pd.Timedelta(hours=horizon + 1)) <= validation_start
            training_errors = errors[:split][train_mask]
            validation_errors = errors[split:]
            item.update(training_examples=len(training_errors), validation_examples=len(validation_errors))
            if len(training_errors) < MIN_ADAPTATION_EXAMPLES:
                item["status"] = "Insufficient examples after temporal purge"
                continue
            trial = float(np.median(training_errors)) * min(1.0, len(training_errors) / FULL_ADAPTATION_EXAMPLES)
            before = float(np.mean(np.abs(validation_errors)))
            after = float(np.mean(np.abs(validation_errors - trial)))
            validated = after < before
            item.update(
                correction=trial if validated else 0.0, validated=validated,
                validation_mae_before=before, validation_mae_after=after,
                status="Validated correction" if validated else "No validated improvement",
            )
    return adaptation


def make_forecast(history, station, assets, adaptation):
    _, hierarchies, metrics = assets
    features = build_feature_history(history)
    if features.empty:
        missing = [col for col in VALUE_COLUMNS if history[col].notna().sum() == 0]
        detail = f" Missing measurements: {', '.join(missing)}." if missing else " Some current readings or required historical readings are missing."
        raise ValueError("Not enough complete hourly observations to create a forecast." + detail)
    origin = features.index[-1]
    rows = []
    for label, meta in VARIABLES.items():
        key, col = meta["key"], meta["column"]
        for horizon in range(1, 25):
            method = effective_method(key, hierarchies[key], metrics[key], station, horizon)
            values, sources = base_predictions(
                key, horizon, features.loc[[origin]], [history.at[origin, col]], assets, method,
            )
            item = adaptation.get(key, {}).get(horizon, {})
            correction = float(item.get("correction", 0.0)) if item.get("validated", False) else 0.0
            if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
                correction = 0.0
            rows.append({
                "variable": key, "label": label, "horizon": horizon,
                "timestamp": origin + pd.Timedelta(hours=horizon),
                "original_prediction": float(values[0]), "prediction": float(values[0] + correction),
                "station_correction": correction, "adaptation_examples": int(item.get("examples", 0)),
                "method_source": sources[0], "origin": origin,
            })
    return pd.DataFrame(rows)


def get_forecast(history, sources, station, assets, signature, adaptation_enabled):
    adaptation = (
        build_station_adaptation(history, sources, station, signature, PIPELINE_VERSION, assets)
        if adaptation_enabled else {}
    )
    return make_forecast(history, station, assets, adaptation), adaptation


def draw_forecast(history, forecast, hours):
    origin = forecast.origin.iloc[0]
    start, end = origin - pd.Timedelta(hours=hours), origin + pd.Timedelta(hours=24)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=.075,
        subplot_titles=[f'{label} · {meta["unit"]}' for label, meta in VARIABLES.items()],
    )
    has_corrections = forecast.station_correction.abs().gt(1e-9).any()
    for row, (label, meta) in enumerate(VARIABLES.items(), 1):
        observed = history.loc[(history.index >= start) & (history.index <= end), meta["column"]]
        subset = forecast[forecast.variable == meta["key"]].sort_values("horizon")
        unit = "" if meta["key"] == "ph" else " " + meta["unit"]
        fig.add_trace(go.Scatter(
            x=observed.index, y=observed, mode="lines", connectgaps=False,
            name="Recorded readings", legendgroup="observed", showlegend=row == 1,
            line=dict(color="#246fba", width=2.8),
            hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}" + unit + "<extra>Recorded</extra>",
        ), row=row, col=1)
        if has_corrections:
            fig.add_trace(go.Scatter(
                x=subset.timestamp, y=subset.original_prediction, mode="lines",
                name="Before station adjustment", legendgroup="base", showlegend=row == 1,
                line=dict(color="#9aaec2", width=1.6, dash="dot"),
                hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}" + unit + "<extra>Before adjustment</extra>",
            ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=subset.timestamp, y=subset.prediction, mode="lines+markers",
            name="Forecast", legendgroup="forecast", showlegend=row == 1,
            customdata=subset[["method_source", "station_correction", "adaptation_examples"]].to_numpy(),
            line=dict(color="#16a394", width=2.8, dash="dash"), marker=dict(size=4, color="#16a394"),
            hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}" + unit + "<br>%{customdata[0]}<br>Adjustment: %{customdata[1]:+.3f}<extra>Forecast</extra>",
        ), row=row, col=1)
    fig.add_vrect(x0=origin, x1=end, fillcolor="#246fba", opacity=.035, line_width=0)
    fig.add_vline(x=origin, line_dash="dot", line_color="#95a9bb", line_width=1.4)
    fig.add_annotation(x=origin, y=1.065, xref="x", yref="paper", text="FORECAST START",
                       showarrow=False, font=dict(size=10, color="#70869a"))
    fig.update_xaxes(range=[start, end], showline=False, zeroline=False,
                     gridcolor="#edf2f7", tickfont=dict(size=11, color="#70869a"))
    fig.update_xaxes(title_text="Time (UTC)", row=4, col=1)
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="#edf2f7",
                     tickfont=dict(size=11, color="#70869a"), nticks=5)
    fig.update_layout(
        height=940, hovermode="x unified", template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif", color="#34556f", size=12),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dce7f2",
                        font=dict(family="DM Sans, sans-serif", color="#214a6b", size=13)),
        legend=dict(orientation="h", y=1.12, x=0, title_text=""),
        margin=dict(l=55, r=20, t=110, b=45),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def show_adjustment_details(adaptation, forecast, enabled):
    with st.expander("Station adjustment details"):
        if not enabled:
            st.write("Station adjustment is off. Forecasts use the selected base methods without corrections.")
        summary, details = [], []
        for label, meta in VARIABLES.items():
            items = adaptation.get(meta["key"], {})
            active = [item for item in items.values() if item.get("validated")]
            if meta["key"] == "salinity":
                status = "Persistence only"
            elif not enabled:
                status = "Off"
            else:
                status = "Active" if active else "No validated correction"
            summary.append({
                "Parameter": label, "Adjusted horizons": f"{len(active)}/24",
                "Historical examples (max per horizon)": max([item.get("examples", 0) for item in items.values()], default=0),
                "Median correction": float(np.median([item["correction"] for item in active])) if active else 0.0,
                "Status": status,
            })
            for horizon, item in items.items():
                flat = {key: value for key, value in item.items() if key != "source_counts"}
                details.append({"Parameter": label, "Horizon": horizon, **flat, **item["source_counts"]})
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        if details:
            st.caption("Corrections require at least 12 training and 6 later validation examples, with overlapping future training labels excluded. They are applied only when validation error improves. This is an internal validation check, not an independent performance guarantee.")
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
        st.dataframe(forecast.drop(columns="variable"), use_container_width=True, hide_index=True)


def show_data_sources(history, sources, notes, expanded=False):
    with st.expander("Data sources and technical details", expanded=expanded):
        rows = []
        for label, meta in VARIABLES.items():
            series = history[meta["column"]].dropna()
            time = series.index[-1] if not series.empty else None
            rows.append({
                "Parameter": label,
                "Latest hourly reading (UTC)": str(time) if time is not None else "Unavailable",
                "Source": sources.at[time, meta["column"]] if time is not None else "Missing",
                "Hourly observations": len(series), "Notes": notes.get(meta["column"], "USGS Direct only"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Readings are medians of completed hourly bins, labeled by the start of the hour. No interpolation or forward filling is used. Provisional USGS observations may be revised. NOAA fallback uses the station supplied in Settings; its location must be appropriate for the USGS site.")
        if not history.empty:
            row = make_features(history, history.index[-1]).iloc[0]
            missing = row.index[~np.isfinite(row.to_numpy(dtype=float))].tolist()
            st.write("Missing inputs for the latest hour:", ", ".join(missing) if missing else "None")
        st.caption("The forecasting pipeline retains the original 22-feature schema and all 96 model-file references. Salinity models are checked but not used for inference. Missing salinity still prevents complete model features for all four targets.")
        st.caption("Salinity uses the forecast-origin reading as an unadjusted constant. Its ML predictions and station corrections remain disabled pending retraining. No calibrated uncertainty or hypoxia-event classifier is implemented here.")


def show_brand_banner():
    logo_path = ROOT / "Logo.png"
    logo_html = ""
    if logo_path.is_file():
        try:
            encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            logo_html = '<img class="brand-logo" src="data:image/png;base64,' + encoded_logo + '" alt="AquaSentry logo">'
        except OSError:
            logo_html = ""
    st.markdown(
        '<div class="hero">' + logo_html + '<div><h1>AquaSentry</h1>'
        '<p>Predicting, Protecting, Preventing.</p></div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Monitoring station")
        choice = st.selectbox("Select station", list(STATIONS), label_visibility="collapsed")
        station = st.text_input("USGS station ID") if STATIONS[choice] == "custom" else STATIONS[choice]
        hours = st.selectbox("Chart history", [24, 48, 72], format_func=lambda value: f"Past {value} hours")
        if st.button("Refresh data", use_container_width=True):
            fetch_usgs.clear()
            fetch_noaa.clear()
            build_station_adaptation.clear()
        with st.expander("Settings"):
            adaptation_enabled = st.checkbox(
                "Station adjustment", value=True, key="station_adjustment_enabled",
                help="Apply station-specific corrections only when they improve historical validation results.",
            )
            noaa = st.text_input(
                "NOAA fallback station", key="noaa_fallback_station",
                help="Optional. Fills missing temperature and salinity using the NOAA station you provide. Use an appropriate nearby station.",
            )
    show_brand_banner()
    station = station.strip()
    if not re.fullmatch(r"\d{8}(\d{7})?", station):
        st.info("Select a station or enter an 8- or 15-digit USGS station ID.")
        return
    try:
        with st.spinner("Loading station data…"):
            raw, provenance = fetch_usgs(station)
            history, sources, notes = hourly_history(raw, provenance, noaa)
    except Exception as exc:
        st.error(f"Could not load station data: {exc}")
        return
    name = raw.attrs.get("station_name", f"USGS {station}")
    st.markdown(
        '<div class="station-id">USGS ' + html.escape(station) + '</div>'
        '<div class="station-title">' + html.escape(name) + '</div>', unsafe_allow_html=True,
    )
    st.caption("Latest hourly readings · All times UTC")
    now = pd.Timestamp.now(tz="UTC")
    boxes = st.columns(4)
    for box, (label, meta) in zip(boxes, VARIABLES.items()):
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
            st.caption(f'{sources.at[time, meta["column"]]} · {time:%b %d, %H:%M} UTC')
            if (now - (time + pd.Timedelta(hours=1))).total_seconds() > 10800:
                st.caption("Reading delayed")
    st.caption("Salinity forecast: held constant while its prediction models are improved.")
    forecast, adaptation = None, {}
    try:
        signature = asset_signature()
        message = "Updating forecast and station adjustment…" if adaptation_enabled else "Updating forecast…"
        with st.spinner(message):
            assets = load_assets(signature)
            forecast, adaptation = get_forecast(history, sources, station, assets, signature, adaptation_enabled)
    except Exception as exc:
        st.warning(f"Forecast unavailable: {exc}")
    if forecast is not None:
        origin = forecast.origin.iloc[0]
        latest = history.dropna(how="all").index[-1]
        age = max(0.0, (now - (origin + pd.Timedelta(hours=1))).total_seconds() / 3600)
        gap = max(0.0, (latest - origin).total_seconds() / 3600)
        st.markdown("### Water-quality forecast")
        a, b, c = st.columns(3)
        a.metric("Forecast starts", origin.strftime("%b %d · %H:%M"))
        b.metric("Time since forecast data", f"{age:.1f} h")
        c.metric("Station adjustment", "On" if adaptation_enabled else "Off")
        if age > 3 or gap > 3:
            st.warning("Some data is delayed. This forecast uses older complete readings.")
        draw_forecast(history, forecast, hours)
        st.caption("Solid line: recorded readings · Dashed line: forecast")
        st.download_button(
            "Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"aquasentry_{station}_forecast.csv", mime="text/csv", use_container_width=True,
        )
        show_adjustment_details(adaptation, forecast, adaptation_enabled)
    show_data_sources(history, sources, notes, expanded=forecast is None)
    st.markdown('<div class="footer">AquaSentry · Research forecasting dashboard</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
