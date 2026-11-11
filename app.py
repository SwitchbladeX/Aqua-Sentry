from __future__ import annotations

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

st.set_page_config(page_title="AquaSentry | Water intelligence", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
VARIABLES = {
    "Dissolved oxygen": {"key": "do", "folder": "dissolved_oxygen", "prefix": "DO", "column": "DO_mgL", "unit": "mg/L", "code": "00300"},
    "Temperature": {"key": "temp", "folder": "temperature", "prefix": "Temp", "column": "temp_C", "unit": "°C", "code": "00010"},
    "pH": {"key": "ph", "folder": "ph", "prefix": "pH", "column": "pH", "unit": "pH", "code": "00400"},
    "Salinity": {"key": "salinity", "folder": "salinity", "prefix": "Salinity", "column": "salinity_PSU", "unit": "PSU", "code": "90860"},
}
VALUE_COLUMNS = [m["column"] for m in VARIABLES.values()]
LAGS = [1, 3, 6, 12]
FEATURE_COLUMNS = VALUE_COLUMNS + [f"{c}_lag{h}h" for c in VALUE_COLUMNS for h in LAGS] + ["hour", "month"]
HIERARCHIES = {"do": "DO_VALIDATION_HIERARCHY.csv", "temp": "TEMP_VALIDATION_HIERARCHY.csv", "ph": "PH_VALIDATION_HIERARCHY.csv", "salinity": "SALINITY_VALIDATION_HIERARCHY.csv"}
MIN_ADAPTATION_EXAMPLES = 12
MIN_VALIDATION_EXAMPLES = 6
FULL_ADAPTATION_EXAMPLES = 48
# Explicit operational policy, not a retrained or repaired salinity model.
SALINITY_PERSISTENCE_ONLY = True
PIPELINE_VERSION = "aquasentry-ui-backend-1"
STATIONS = {
    "West Harbor at Oyster Bay, NY — 01303152": "01303152",
    "Peconic River at Riverhead, NY — 01304562": "01304562",
    "South side of Shelter Island, NY — 01304200": "01304200",
    "Orient Point Harbor, NY — 01304190": "01304190",
    "Forge River at Mastic, NY — 01304057": "01304057",
    "Hempstead Bay at Freeport, NY — 01304998": "01304998",
    "Enter a different USGS station": "custom",
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {color-scheme:light;}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:#f3f7fa;color:#173347;}
[data-testid="stHeader"] {background:rgba(243,247,250,.92);}
.block-container {max-width:1460px;padding-top:1.6rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#102f40;border-right:1px solid #234858;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#e7f2f5;}
[data-testid="stMetric"] {background:white;border:1px solid #dde7ec;border-radius:16px;padding:18px;box-shadow:0 4px 18px #193b4e05;}
[data-testid="stMetricLabel"] {color:#58717e;}
[data-testid="stMetricValue"] {color:#123e50;font-size:1.8rem;}
[data-testid="stExpander"] {background:white;border:1px solid #dde7ec;border-radius:12px;}
.stButton > button {background:#087e8b;color:white;border:0;border-radius:9px;}
.hero {background:linear-gradient(115deg,#102f40,#155568 65%,#087e8b);border-radius:22px;padding:32px;margin-bottom:22px;color:#fff;}
.hero .eyebrow {font-size:.72rem;letter-spacing:.18em;color:#9ddbdc;font-weight:700;}
.hero h1 {color:white;font-size:2.8rem;letter-spacing:-.05em;padding:8px 0;line-height:1.1;}
.hero p {color:#d4e8ee;margin:0;max-width:700px;}
.section-kicker {text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;color:#5a7c88;margin-top:20px;}
.station-title {font-size:1.45rem;font-weight:700;color:#163d50;margin:4px 0 14px;}
.footer {border-top:1px solid #dbe5eb;margin-top:24px;padding-top:16px;color:#617d89;font-size:.8rem;}
@media(max-width:700px){.hero{padding:24px}.hero h1{font-size:2.1rem}[data-testid="stMetricValue"]{font-size:1.45rem}}
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
    mapping = {m["code"]: m["column"] for m in VARIABLES.values()}
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
            # Negative DO/salinity and pH outside its defined scale are rejected.
            if col in ("DO_mgL", "salinity_PSU"):
                values = values.mask(values < 0)
            if col == "pH":
                values = values.mask((values < 0) | (values > 14))
            frame = pd.DataFrame({"time": times, "column": col, "value": values}).dropna()
            if not frame.empty:
                parts.append(frame)
    if not parts:
        raise ValueError("USGS returned no supported water-quality observations.")
    # Combine duplicate series by timestamp AND variable; do not discard a sensor column.
    long = pd.concat(parts, ignore_index=True)
    raw = long.pivot_table(index="time", columns="column", values="value", aggfunc="median").sort_index()
    raw.columns.name = None
    raw.attrs["station_name"] = names[0] if names else f"USGS {site}"
    raw.attrs["series_names"] = sorted(set(names))
    provenance = pd.DataFrame("Missing", index=raw.index, columns=raw.columns)
    provenance = provenance.mask(raw.notna(), "USGS Direct")
    return raw, provenance


@st.cache_data(ttl=900, show_spinner=False)
def fetch_noaa(station: str, product: str, days: int = 90) -> pd.Series:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    parts = []
    # Small date windows avoid requesting the entire period in one product-limited call.
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=29), end)
        response = requests.get(
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            params={"product": product, "application": "AquaSentry", "begin_date": cursor.strftime("%Y%m%d"), "end_date": stop.strftime("%Y%m%d"), "station": station, "time_zone": "gmt", "units": "metric", "format": "json", "interval": "h"},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(payload["error"].get("message", "NOAA API error"))
        data = pd.DataFrame(payload.get("data", []))
        if {"t", "v"}.issubset(data.columns):
            frame = pd.DataFrame({"time": pd.to_datetime(data["t"], utc=True, errors="coerce"), "value": pd.to_numeric(data["v"], errors="coerce")}).dropna()
            parts.append(frame)
        cursor = stop.normalize() + timedelta(days=1) if isinstance(stop, pd.Timestamp) else datetime.combine(stop.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if not parts:
        raise ValueError(f"No NOAA {product} observations returned.")
    result = pd.concat(parts).set_index("time")["value"].groupby(level=0).median().sort_index()
    return result.where(np.isfinite(result) & (result >= 0) if product == "salinity" else np.isfinite(result)).dropna()


def hourly_history(raw, provenance, noaa_station):
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index()
    hourly = raw.resample("1h").median().reindex(columns=VALUE_COLUMNS)
    # Retain trained bin-start convention, but never use an unfinished hourly bin.
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
        for h in range(1, 25):
            path = folder / f'{meta["prefix"]}_{h}h.json'
            if not path.exists():
                raise FileNotFoundError(f"Missing model: {path.relative_to(ROOT)}")
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            booster = model.get_booster()
            if booster.num_features() != len(FEATURE_COLUMNS):
                raise ValueError(f"{path.name}: expected 22 features, found {booster.num_features()}.")
            if booster.feature_names and booster.feature_names != FEATURE_COLUMNS:
                raise ValueError(f"{path.name}: feature names/order differ from app schema.")
            models[key][h] = model
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
            t = timestamp - pd.Timedelta(hours=lag)
            row[f"{col}_lag{lag}h"] = history.at[t, col] if t in history.index else np.nan
    row.update(hour=timestamp.hour, month=timestamp.month)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=float)


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
    cols = {str(c).strip().lower(): c for c in hierarchy.columns}
    hcol = cols.get("horizon") or cols.get("forecast_hour")
    scol = cols.get("site_code") or cols.get("station") or cols.get("station_id")
    mcol = cols.get("selected_method") or cols.get("method")
    if all(c is not None for c in (hcol, scol, mcol)):
        rows = hierarchy[(hierarchy[scol].map(normalize_station) == normalize_station(station)) & (pd.to_numeric(hierarchy[hcol], errors="coerce") == horizon)]
        if not rows.empty:
            method = str(rows.iloc[0][mcol]).lower()
            if "persist" in method:
                return "persistence"
            if "xg" in method:
                return "xgboost"
    cols = {str(c).strip().lower(): c for c in metrics.columns}
    hcol = cols.get("horizon") or cols.get("forecast_hour")
    xcol = cols.get("xgboost_mae") or cols.get("xgb_mae") or cols.get("model_mae")
    pcol = cols.get("persistence_mae") or cols.get("persist_mae")
    if all(c is not None for c in (hcol, xcol, pcol)):
        rows = metrics[pd.to_numeric(metrics[hcol], errors="coerce") == horizon]
        x = pd.to_numeric(rows[xcol], errors="coerce").mean()
        p = pd.to_numeric(rows[pcol], errors="coerce").mean()
        if np.isfinite(x) and np.isfinite(p):
            return "xgboost" if x < p else "persistence"
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
        for h in range(1, 25):
            method = effective_method(key, hierarchies[key], metrics[key], station, h)
            item = {"correction": 0.0, "examples": 0, "median_error": 0.0, "method": method, "validated": False, "validation_mae_before": None, "validation_mae_after": None, "validation_examples": 0, "training_examples": 0, "source_counts": {"USGS Direct": 0, "NOAA CO-OPS": 0}, "status": "Insufficient examples"}
            adaptation[key][h] = item
            # Raw persistence only while the salinity model is suspended.
            if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
                item["status"] = "Disabled: salinity persistence-only policy"
                continue
            if features.empty:
                continue
            target_times = features.index + pd.Timedelta(hours=h)
            actual = history[col].reindex(target_times).to_numpy(dtype=float)
            sources = hourly_provenance[col].reindex(target_times).to_numpy()
            valid = np.isfinite(actual) & np.isin(sources, ["USGS Direct", "NOAA CO-OPS"])
            origins = features.index[valid]
            if not len(origins):
                continue
            predicted, _ = base_predictions(key, h, features.loc[origins], history[col].reindex(origins).to_numpy(), _assets, method)
            errors = actual[valid] - predicted
            finite = np.isfinite(errors)
            errors, origins, sources = errors[finite], origins[finite], sources[valid][finite]
            count = len(errors)
            item.update(examples=count, median_error=float(np.median(errors)) if count else 0.0, source_counts={s: int(np.sum(sources == s)) for s in ("USGS Direct", "NOAA CO-OPS")})
            if count < MIN_ADAPTATION_EXAMPLES + MIN_VALIDATION_EXAMPLES:
                continue
            split = min(int(count * .7), count - MIN_VALIDATION_EXAMPLES)
            validation_start = origins[split]
            # Labels are hourly-bin medians; they are not available before bin end.
            train_mask = (origins[:split] + pd.Timedelta(hours=h + 1)) <= validation_start
            training_errors, validation_errors = errors[:split][train_mask], errors[split:]
            item.update(training_examples=len(training_errors), validation_examples=len(validation_errors))
            if len(training_errors) < MIN_ADAPTATION_EXAMPLES:
                item["status"] = "Insufficient examples after temporal purge"
                continue
            correction = float(np.median(training_errors)) * min(1.0, len(training_errors) / FULL_ADAPTATION_EXAMPLES)
            before = float(np.mean(np.abs(validation_errors)))
            after = float(np.mean(np.abs(validation_errors - correction)))
            validated = after < before
            item.update(correction=correction if validated else 0.0, validated=validated, validation_mae_before=before, validation_mae_after=after, status="Validated correction" if validated else "No validated improvement")
    return adaptation


def make_forecast(history, station, assets, adaptation):
    _, hierarchies, metrics = assets
    features = build_feature_history(history)
    if features.empty:
        missing = [c for c in VALUE_COLUMNS if history[c].notna().sum() == 0]
        detail = f" Entirely missing: {', '.join(missing)}." if missing else " Check missing current values and exact 1/3/6/12-hour lags."
        raise ValueError("Not enough complete historical observations for the existing 22-feature pipeline." + detail)
    origin = features.index[-1]
    rows = []
    for label, meta in VARIABLES.items():
        key, col = meta["key"], meta["column"]
        for h in range(1, 25):
            method = effective_method(key, hierarchies[key], metrics[key], station, h)
            pred, sources = base_predictions(key, h, features.loc[[origin]], [history.at[origin, col]], assets, method)
            item = adaptation.get(key, {}).get(h, {})
            correction = float(item.get("correction", 0.0)) if item.get("validated", False) else 0.0
            if key == "salinity" and SALINITY_PERSISTENCE_ONLY:
                correction = 0.0
            rows.append({"variable": key, "label": label, "horizon": h, "timestamp": origin + pd.Timedelta(hours=h), "original_prediction": float(pred[0]), "prediction": float(pred[0] + correction), "station_correction": correction, "adaptation_examples": int(item.get("examples", 0)), "method_source": sources[0], "origin": origin})
    return pd.DataFrame(rows)


def draw_forecast(history, forecast, hours):
    origin = forecast.origin.iloc[0]
    start, end = origin - pd.Timedelta(hours=hours), origin + pd.Timedelta(hours=24)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.075, subplot_titles=[f'{label} · {meta["unit"]}' for label, meta in VARIABLES.items()])
    for row, (label, meta) in enumerate(VARIABLES.items(), 1):
        observed = history.loc[(history.index >= start), meta["column"]]
        subset = forecast[forecast.variable == meta["key"]].sort_values("horizon")
        fig.add_trace(go.Scatter(x=observed.index, y=observed, mode="lines", connectgaps=False, name="Observed hourly median", legendgroup="obs", showlegend=row == 1, line=dict(color="#155568", width=2.5), hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}<extra>Observed</extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.original_prediction, mode="lines", name="Base forecast", legendgroup="base", showlegend=row == 1, visible=True if subset.station_correction.abs().gt(1e-9).any() else "legendonly", line=dict(color="#8ca1ae", width=1.5, dash="dot")), row=row, col=1)
        custom = subset[["method_source", "station_correction", "adaptation_examples"]].to_numpy()
        fig.add_trace(go.Scatter(x=subset.timestamp, y=subset.prediction, customdata=custom, mode="lines+markers", name="Operational forecast", legendgroup="forecast", showlegend=row == 1, line=dict(color="#08a6a0", width=2.5, dash="dash"), marker=dict(size=4), hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}<br>%{customdata[0]}<br>Correction: %{customdata[1]:+.3f}<br>Historical examples: %{customdata[2]}<extra></extra>"), row=row, col=1)
        fig.update_yaxes(gridcolor="#e9eff3", zerolinecolor="#e9eff3", row=row, col=1)
    fig.add_vrect(x0=origin, x1=end, fillcolor="#08a6a0", opacity=.04, line_width=0)
    fig.add_vline(x=origin, line_dash="dot", line_color="#c38b36", line_width=1.5)
    fig.add_annotation(x=origin, y=1.075, xref="x", yref="paper", text="FORECAST ORIGIN", showarrow=False, font=dict(size=10, color="#926824"))
    fig.update_xaxes(range=[start, end], gridcolor="#edf2f5")
    fig.update_xaxes(title_text="Hourly bin start · UTC", row=4, col=1)
    fig.update_layout(height=960, hovermode="x unified", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", font=dict(family="DM Sans, sans-serif", color="#234756"), legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=55, r=20, t=115, b=50))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## AquaSentry")
        st.caption("ESTUARY MONITORING WORKSPACE")
        st.divider()
        choice = st.selectbox("Monitoring station", list(STATIONS))
        station = st.text_input("USGS station ID") if STATIONS[choice] == "custom" else STATIONS[choice]
        noaa = st.text_input("Optional NOAA station ID", help="Only use a scientifically appropriate station. NOAA gap filling does not establish that two locations are equivalent.")
        hours = st.selectbox("Observed history on chart", [24, 48, 72], format_func=lambda n: f"Past {n} hours")
        if st.button("Refresh observations", use_container_width=True):
            fetch_usgs.clear()
            fetch_noaa.clear()
            build_station_adaptation.clear()
        st.divider()
        st.caption("24 forecast horizons · four water-quality variables")
        st.caption("Salinity ML suspended. Salinity uses unadjusted persistence.")
        st.caption("Station menu labels are retained from the original app; the main heading uses the returned USGS site name.")
    st.markdown('<div class="hero"><div class="eyebrow">WATER INTELLIGENCE / LONG ISLAND</div><h1>AquaSentry</h1><p>Predicting water conditions. Protecting aquatic ecosystems.</p></div>', unsafe_allow_html=True)
    station = station.strip()
    if not re.fullmatch(r"\d{8}(\d{7})?", station):
        st.info("Select a station or enter an 8- or 15-digit USGS station ID.")
        return
    try:
        with st.spinner("Retrieving station observations…"):
            raw, provenance = fetch_usgs(station)
            history, sources, notes = hourly_history(raw, provenance, noaa)
    except Exception as exc:
        st.error(f"Could not load station observations: {exc}")
        return
    name = raw.attrs.get("station_name", f"USGS {station}")
    st.markdown('<div class="section-kicker">Monitoring location · USGS ' + html.escape(station) + '</div><div class="station-title">' + html.escape(name) + '</div>', unsafe_allow_html=True)
    st.caption("Measurements below are hourly medians from completed bins, not instantaneous readings. All timestamps are UTC.")
    cols = st.columns(4)
    now = pd.Timestamp.now(tz="UTC")
    for box, (label, meta) in zip(cols, VARIABLES.items()):
        series = history[meta["column"]].dropna()
        with box:
            if series.empty:
                st.metric(label, "Unavailable")
                st.caption("No completed hourly observation returned")
                continue
            t, value = series.index[-1], float(series.iloc[-1])
            prior = series.get(t - pd.Timedelta(hours=24), np.nan)
            unit = "" if meta["key"] == "ph" else " " + meta["unit"]
            delta = f"{value - prior:+.2f}{unit} vs exact 24h" if pd.notna(prior) else None
            st.metric(label, f"{value:.2f}{unit}", delta=delta, delta_color="off")
            st.caption(f'{sources.at[t, meta["column"]]} · {t:%b %d, %H:%M} UTC')
            if (now - (t + pd.Timedelta(hours=1))).total_seconds() > 10800:
                st.caption("Delayed: this variable's latest completed bin ended over 3 hours ago.")
    st.info("Salinity model safeguard: the existing salinity XGBoost predictions are disabled. The salinity forecast holds the forecast-origin observation constant; it is not a repaired marine salinity model. Salinity station corrections are also disabled.")
    forecast, adaptation, assets_error = None, {}, None
    try:
        signature = asset_signature()
        with st.spinner("Checking model assets and validating station adjustments…"):
            assets = load_assets(signature)
            adaptation = build_station_adaptation(history, sources, station, signature, PIPELINE_VERSION, assets)
            forecast = make_forecast(history, station, assets, adaptation)
    except Exception as exc:
        assets_error = str(exc)
    if assets_error:
        st.warning(f"Forecast unavailable: {assets_error}")
        st.caption("Observed measurements and data-source diagnostics remain available below. Missing salinity is not imputed into other models.")
    if forecast is not None:
        origin = forecast.origin.iloc[0]
        latest = history.dropna(how="all").index[-1]
        origin_age = max(0., (now - (origin + pd.Timedelta(hours=1))).total_seconds() / 3600)
        gap = max(0., (latest - origin).total_seconds() / 3600)
        st.markdown("### Forecast workspace")
        a, b, c = st.columns(3)
        a.metric("Forecast-origin bin (UTC)", origin.strftime("%b %d · %H:%M"))
        b.metric("Time since origin bin ended", f"{origin_age:.1f} h")
        c.metric("Complete-feature lag", f"{gap:.1f} h")
        st.caption("The origin is the latest bin with all 22 finite features. Complete-feature lag compares it with the latest bin containing any observation. Forecast timestamps retain the original bin-start convention.")
        if origin_age > 3 or gap > 3:
            st.warning("Delayed forecast: inspect individual observation timestamps and missing features before relying on these projections.")
        else:
            st.caption("Freshness checks are within the dashboard's 3-hour display threshold; this is not a scientific accuracy guarantee.")
        draw_forecast(history, forecast, hours)
        st.caption("Solid: observations, including any measurements newer than the complete-feature origin. Dashed: operational forecast. No uncertainty intervals or hypoxia-risk probabilities are estimated.")
        summary, details = [], []
        for label, meta in VARIABLES.items():
            items = adaptation.get(meta["key"], {})
            active = [v for v in items.values() if v.get("validated")]
            summary.append({"Parameter": label, "Adapted horizons": f"{len(active)}/24", "Historical examples (maximum per horizon)": max([v.get("examples", 0) for v in items.values()], default=0), "Median applied correction": float(np.median([v["correction"] for v in active])) if active else 0., "Status": "Persistence only; ML/correction disabled" if meta["key"] == "salinity" else "Active" if active else "No validated correction"})
            for h, item in items.items():
                flat = {k: v for k, v in item.items() if k != "source_counts"}
                details.append({"Parameter": label, "Horizon": h, **flat, **item["source_counts"]})
        with st.expander("Station adjustment · validation and method details"):
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
            st.caption("Corrections require at least 12 training and 6 later validation examples after purging training labels that would not be available at the first validation origin. Corrections are confidence-scaled and applied only when validation MAE improves. This internal check is not an independent prospective performance estimate.")
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
            st.dataframe(forecast.drop(columns="variable"), use_container_width=True, hide_index=True)
        st.download_button("Export 24-hour forecast · CSV", forecast.to_csv(index=False).encode("utf-8"), file_name=f"aquasentry_{station}_forecast.csv", mime="text/csv")
    with st.expander("Data sources · availability and missing features", expanded=forecast is None):
        rows = []
        for label, meta in VARIABLES.items():
            series = history[meta["column"]].dropna()
            t = series.index[-1] if not series.empty else None
            rows.append({"Parameter": label, "Latest hourly bin (UTC)": str(t) if t is not None else "Unavailable", "Source": sources.at[t, meta["column"]] if t is not None else "Missing", "Hourly observations": len(series), "Notes": notes.get(meta["column"], "USGS Direct only")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("No interpolation, forward filling, or fabricated salinity values are used. Duplicate sensor observations are combined by median. USGS provisional observations may be revised. NOAA gap filling requires the station ID you supplied; co-location has not been verified by this app.")
        if not history.empty:
            diagnostic = make_features(history, history.index[-1]).iloc[0]
            missing = diagnostic.index[~np.isfinite(diagnostic.to_numpy(dtype=float))].tolist()
            st.write("Missing/non-finite features at latest completed bin:", ", ".join(missing) if missing else "None")
        st.caption("Runtime asset checks require all 96 existing model files and hierarchy files. Salinity files are checked for compatibility but not used for inference while suspended.")
    st.markdown('<div class="footer">AquaSentry · Research forecasting dashboard<br>Regression forecasts and persistence baselines—not an implemented hypoxia warning classifier. Marine salinity retraining and independent station-level validation remain necessary.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
