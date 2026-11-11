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

st.set_page_config(page_title="AquaSentry", page_icon="🌊", layout="wide")
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
MIN_ADAPTATION_EXAMPLES = 12
MIN_VALIDATION_EXAMPLES = 6
FULL_ADAPTATION_EXAMPLES = 48
STATIONS = {
    "West Harbor at Oyster Bay, NY — 01303152": "01303152", "Peconic River at Riverhead, NY — 01304562": "01304562",
    "South side of Shelter Island, NY — 01304200": "01304200", "Orient Point Harbor, NY — 01304190": "01304190",
    "Forge River at Mastic, NY — 01304057": "01304057", "Hempstead Bay at Freeport, NY — 01304998": "01304998",
    "Enter a different USGS station": "custom",
}
STATION_METADATA = {
    "01303152": {"name": "West Harbor at Oyster Bay, NY", "lat": 40.8784, "lon": -73.5352},
    "01304562": {"name": "Peconic River at Riverhead, NY", "lat": 40.9234, "lon": -72.6687},
    "01304200": {"name": "South side of Shelter Island, NY", "lat": 41.0543, "lon": -72.3382},
    "01304190": {"name": "Orient Point Harbor, NY", "lat": 41.1440, "lon": -72.2351},
    "01304057": {"name": "Forge River at Mastic, NY", "lat": 40.7812, "lon": -72.8590},
    "01304998": {"name": "Hempstead Bay at Freeport, NY", "lat": 40.6354, "lon": -73.5801},
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
.hero {padding:22px 28px;border-radius:18px;background:linear-gradient(112deg,#123b64,#1769aa 58%,#328cc4);color:white;margin:0 0 20px;box-shadow:0 10px 28px rgba(23,75,120,.18);}
.hero h1 {font-size:2.2rem;margin:0 0 6px 0;color:white;letter-spacing:-.04em;line-height:1.2;}
.hero p {font-size:1.05rem;margin:0;color:#e1effe;font-weight:400;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs(site: str, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        response = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={
                "format": "json",
                "sites": site,
                "parameterCd": "00010,00300,00400,90860",
                "period": f"P{days}D",
            },
            timeout=45,
            headers={"User-Agent": "AquaSentry/1.0"},
        )
        response.raise_for_status()
        series = response.json().get("value", {}).get("timeSeries", [])
        st.write(
            "USGS raw parameter metadata:",
            [
                {
                    "codes": [
                        code.get("value")
                        for code in item.get("variable", {}).get("variableCode", [])
                    ],
                    "description": item.get("variable", {}).get("variableDescription"),
                    "value_groups": len(item.get("values", [])),
                }
                for item in series
            ],
        )
    except Exception as exc:
        raise ValueError(f"USGS API request failed: {exc}")

    mapping = {meta["code"]: meta["column"] for meta in VARIABLES.values()}
    parts = []

    for item in series:
        codes = item.get("variable", {}).get("variableCode", [])
        groups = item.get("values", [])

        if not codes or not groups:
            continue

        col = mapping.get(codes[0].get("value"))
        if col is None:
            continue

        data = pd.DataFrame(groups[0].get("value", []))
        if not {"dateTime", "value"}.issubset(data.columns):
            continue

        data["time"] = pd.to_datetime(data["dateTime"], utc=True, errors="coerce")
        data[col] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["time", col])

        if data.empty:
            continue

        measurements = (
            data.set_index("time")[[col]]
            .groupby(level=0)
            .median()
            .sort_index()
        )
        parts.append(measurements)

    if not parts:
        raise ValueError("USGS returned no supported water-quality observations.")

    raw = (
        pd.concat(parts, axis=1)
        .groupby(level=0)
        .median(numeric_only=True)
        .sort_index()
    )

    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.index.name = "time"

    provenance = pd.DataFrame(
        "Missing",
        index=raw.index,
        columns=raw.columns,
    )
    for col in raw.columns:
        provenance.loc[raw[col].notna(), col] = "USGS Direct"

    return raw, provenance


def fetch_noaa(station: str, product: str, days: int = 90) -> pd.Series:
    end = datetime.now(timezone.utc)
    try:
        response = requests.get(
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            params={
                "product": product,
                "application": "AquaSentry",
                "begin_date": (end - timedelta(days=days)).strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "station": station,
                "time_zone": "gmt",
                "units": "metric",
                "format": "json",
                "interval": "h",
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"NOAA API request failed: {exc}")

    if payload.get("error"):
        raise ValueError(payload["error"].get("message", "NOAA API error"))
    data = pd.DataFrame(payload.get("data", []))
    if not {"t", "v"}.issubset(data.columns):
        raise ValueError(f"No NOAA {product} observations returned.")
    data["time"] = pd.to_datetime(data["t"], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["v"], errors="coerce")
    return data.dropna(subset=["time", "value"]).set_index("time")["value"].groupby(level=0).median().sort_index()


def hourly_history(
    raw: pd.DataFrame,
    provenance: pd.DataFrame,
    noaa_station: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:

    raw = raw.copy()
    provenance = provenance.copy()

    raw.index = pd.to_datetime(raw.index, utc=True)
    provenance.index = pd.to_datetime(provenance.index, utc=True)

    raw = raw.sort_index()
    provenance = provenance.reindex(raw.index)

    hourly = raw.resample("1h").median()

    for col in VALUE_COLUMNS:
        if col not in hourly.columns:
            hourly[col] = np.nan

    hourly = hourly[VALUE_COLUMNS]

    hourly_provenance = pd.DataFrame(
        "Missing",
        index=hourly.index,
        columns=VALUE_COLUMNS,
    )

    for col in VALUE_COLUMNS:
        if col in raw.columns:
            direct_count = raw[col].resample("1h").count()
            hourly_provenance.loc[
                direct_count > 0, col
            ] = "USGS Direct"

    notes = {}

    for col, product in [
        ("temp_C", "water_temperature"),
        ("salinity_PSU", "salinity"),
    ]:
        noaa_filled = 0
        noaa_error = None

        if noaa_station.strip() and hourly[col].isna().any():
            try:
                fallback = fetch_noaa(
                    noaa_station.strip(),
                    product,
                )

                fallback.index = pd.to_datetime(
                    fallback.index,
                    utc=True,
                )

                fallback = (
                    fallback.resample("1h")
                    .median()
                    .reindex(hourly.index)
                )

                mask = hourly[col].isna() & fallback.notna()

                hourly.loc[mask, col] = fallback.loc[mask]
                hourly_provenance.loc[mask, col] = "NOAA CO-OPS"

                noaa_filled = int(mask.sum())

            except Exception as exc:
                noaa_error = f"NOAA CO-OPS unavailable: {exc}"

        available_sources = hourly_provenance.loc[
            hourly_provenance[col] != "Missing",
            col,
        ]

        latest_source = (
            available_sources.iloc[-1]
            if not available_sources.empty
            else "Unavailable"
        )

        if noaa_error:
            notes[col] = noaa_error
        elif not noaa_station.strip():
            notes[col] = f"Latest: {latest_source}; NOAA fallback not configured"
        else:
            notes[col] = (
                f"Latest: {latest_source}; "
                f"NOAA filled {noaa_filled} hourly gaps"
            )

    for col in ["DO_mgL", "pH"]:
        available_sources = hourly_provenance.loc[
            hourly_provenance[col] != "Missing",
            col,
        ]

        notes[col] = (
            f"Latest: {available_sources.iloc[-1]}"
            if not available_sources.empty
            else "Unavailable"
        )

    return hourly, hourly_provenance, notes


def normalize_station(value) -> str:
    value = str(value).strip().replace("USGS-", "").removesuffix(".0")
    return value.zfill(8) if value.isdigit() and len(value) <= 8 else value


@st.cache_resource(show_spinner=False)
def load_assets():
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
            models[key][h] = model
        hierarchy_path = folder / HIERARCHIES[key]
        if not hierarchy_path.exists():
            raise FileNotFoundError(f"Missing hierarchy: {hierarchy_path.relative_to(ROOT)}")
        hierarchies[key] = pd.read_csv(hierarchy_path)
        metric_path = folder / HIERARCHIES[key].replace("HIERARCHY", "METHOD_METRICS")
        metrics[key] = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    return models, hierarchies, metrics


def make_features(history: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    row = {col: float(history.at[timestamp, col]) for col in VALUE_COLUMNS}
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            t = timestamp - pd.Timedelta(hours=lag)
            row[f"{col}_lag{lag}h"] = (
                float(history.at[t, col])
                if t in history.index and pd.notna(history.at[t, col])
                else np.nan
            )
    row.update(hour=int(timestamp.hour), month=int(timestamp.month))
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def choose_method(hierarchy: pd.DataFrame, metrics: pd.DataFrame, station: str, horizon: int) -> str:
    hierarchy_columns = {str(column).lower(): column for column in hierarchy.columns}
    hcol = hierarchy_columns.get("horizon") or hierarchy_columns.get("forecast_hour")
    scol = (
        hierarchy_columns.get("site_code")
        or hierarchy_columns.get("station")
        or hierarchy_columns.get("station_id")
    )
    mcol = hierarchy_columns.get("selected_method") or hierarchy_columns.get("method")

    if hcol is not None and scol is not None and mcol is not None:
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

    if not metrics.empty:
        metric_columns = {str(column).lower(): column for column in metrics.columns}
        metric_horizon = (
            metric_columns.get("horizon")
            or metric_columns.get("forecast_hour")
        )
        x_column = (
            metric_columns.get("xgboost_mae")
            or metric_columns.get("xgb_mae")
        )
        p_column = (
            metric_columns.get("persistence_mae")
            or metric_columns.get("persist_mae")
        )

        if metric_horizon is not None and x_column is not None and p_column is not None:
            rows = metrics[
                pd.to_numeric(metrics[metric_horizon], errors="coerce") == horizon
            ]
            x_value = pd.to_numeric(rows[x_column], errors="coerce").mean()
            p_value = pd.to_numeric(rows[p_column], errors="coerce").mean()
            if pd.notna(x_value) and pd.notna(p_value):
                return "xgboost" if x_value < p_value else "persistence"

    return "xgboost"


def build_feature_history(history: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []
    timestamps = []
    for timestamp in history.index:
        features = make_features(history, timestamp)
        values = features.to_numpy(dtype=float)
        if np.isfinite(values).all():
            feature_rows.append(features.iloc[0])
            timestamps.append(timestamp)
    if not feature_rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=float)
    result = pd.DataFrame(feature_rows, index=pd.DatetimeIndex(timestamps))
    return result[FEATURE_COLUMNS].astype(float)


@st.cache_data(ttl=900, show_spinner=False)
def build_station_adaptation(history: pd.DataFrame, hourly_provenance: pd.DataFrame, station: str, _assets) -> dict:
    models, hierarchies, metrics = _assets
    feature_history = build_feature_history(history)
    if feature_history.empty:
        return {}

    origins = feature_history.index
    adaptation = {}

    for label, meta in VARIABLES.items():
        key, col = meta["key"], meta["column"]
        adaptation[key] = {}

        for horizon in range(1, 25):
            method = choose_method(hierarchies[key], metrics[key], station, horizon)
            valid_origins = []
            actual_list = []
            baseline_list = []
            source_list = []

            for origin in origins:
                target_time = origin + pd.Timedelta(hours=horizon)
                if target_time not in history.index or target_time not in hourly_provenance.index:
                    continue
                if pd.isna(history.at[target_time, col]):
                    continue
                source = hourly_provenance.at[target_time, col]
                if source not in ("USGS Direct", "NOAA CO-OPS"):
                    continue
                source_list.append(source)
                actual = history.at[target_time, col]
                valid_origins.append(origin)
                actual_list.append(actual)
                baseline_list.append(history.at[origin, col])

            if not valid_origins:
                adaptation[key][horizon] = {
                    "correction": 0.0,
                    "examples": 0,
                    "median_error": 0.0,
                    "method": method,
                    "validated": False,
                    "validation_mae_before": None,
                    "validation_mae_after": None,
                    "validation_examples": 0,
                    "source_counts": {"USGS Direct": 0, "NOAA CO-OPS": 0},
                }
                continue

            valid_origins = pd.DatetimeIndex(valid_origins)
            actual = np.array(actual_list, dtype=float)
            baseline = np.array(baseline_list, dtype=float)

            if method == "xgboost":
                model = models[key][horizon]
                names = model.get_booster().feature_names
                if names and names != FEATURE_COLUMNS:
                    raise ValueError(
                        f'{meta["prefix"]}_{horizon}h model feature order differs from app schema.'
                    )
                predicted = np.asarray(
                    model.predict(feature_history.loc[valid_origins]),
                    dtype=float,
                )
                if key == "salinity":
                    scale_guard = (baseline > 5) & (predicted < baseline * 0.5)
                    predicted[scale_guard] = baseline[scale_guard]
            else:
                predicted = baseline.copy()

            errors = actual - predicted
            finite = np.isfinite(errors)
            errors = errors[finite]
            error_times = valid_origins[finite]
            valid_sources = np.asarray(source_list, dtype=object)[finite]

            order = np.argsort(error_times)
            errors = errors[order]
            valid_sources = valid_sources[order]
            source_counts = {
                "USGS Direct": int(np.sum(valid_sources == "USGS Direct")),
                "NOAA CO-OPS": int(np.sum(valid_sources == "NOAA CO-OPS")),
            }
            count = int(errors.size)

            correction = 0.0
            median_error = float(np.median(errors)) if count else 0.0
            validation_mae_before = None
            validation_mae_after = None
            validation_count = 0
            validated = False

            if count >= MIN_ADAPTATION_EXAMPLES + MIN_VALIDATION_EXAMPLES:
                split = int(count * 0.7)
                training_errors = errors[:split]
                validation_errors = errors[split:]
                validation_count = int(validation_errors.size)

                if (
                    training_errors.size >= MIN_ADAPTATION_EXAMPLES
                    and validation_count >= MIN_VALIDATION_EXAMPLES
                ):
                    training_median = float(np.median(training_errors))
                    training_confidence = min(
                        1.0,
                        training_errors.size / FULL_ADAPTATION_EXAMPLES,
                    )
                    trial_correction = training_median * training_confidence
                    validation_mae_before = float(np.mean(np.abs(validation_errors)))
                    validation_mae_after = float(
                        np.mean(np.abs(validation_errors - trial_correction))
                    )
                    validated = validation_mae_after < validation_mae_before

                    if validated:
                        correction = trial_correction

            adaptation[key][horizon] = {
                "correction": float(correction),
                "examples": count,
                "median_error": median_error,
                "method": method,
                "validated": validated,
                "validation_mae_before": validation_mae_before,
                "validation_mae_after": validation_mae_after,
                "validation_examples": validation_count,
                "source_counts": source_counts,
            }

    return adaptation


def make_forecast(history: pd.DataFrame, station: str, assets, adaptation: dict) -> pd.DataFrame:
    models, hierarchies, metrics = assets
    feature_history = build_feature_history(history)
    if feature_history.empty:
        raise ValueError(
            "Not enough complete historical observations to create a forecast."
        )

    origin = feature_history.index[-1]
    forecasts = []
    for label, meta in VARIABLES.items():
        key, col = meta["key"], meta["column"]
        for horizon in range(1, 25):
            method = choose_method(hierarchies[key], metrics[key], station, horizon)
            value = float(history.at[origin, col])
            source = "Persistence"

            if method == "xgboost" and origin in feature_history.index:
                model = models[key][horizon]
                names = model.get_booster().feature_names
                if names and names != FEATURE_COLUMNS:
                    raise ValueError(
                        f'{meta["prefix"]}_{horizon}h model feature order differs from app schema.'
                    )
                candidate = float(model.predict(feature_history.loc[[origin]])[0])
                if np.isfinite(candidate):
                    if key == "salinity":
                        baseline = float(history[col].dropna().tail(24).median())
                        if baseline > 5 and candidate < baseline * 0.5:
                            source = "Persistence (salinity model scale guard)"
                        else:
                            value = candidate
                            source = "XGBoost"
                    else:
                        value = candidate
                        source = "XGBoost"

            details = adaptation.get(key, {}).get(
                horizon,
                {"correction": 0.0, "examples": 0, "median_error": 0.0},
            )
            correction = float(details["correction"])
            adapted_value = value + correction

            forecasts.append({
                "variable": key,
                "label": label,
                "horizon": horizon,
                "timestamp": origin + pd.Timedelta(hours=horizon),
                "original_prediction": value,
                "prediction": adapted_value,
                "station_correction": correction,
                "adaptation_examples": int(details["examples"]),
                "method_source": source,
                "origin": origin,
            })

    return pd.DataFrame(forecasts)


st.markdown('<div class="hero"><h1>🌊 AquaSentry</h1><p>Predicting water conditions. Protecting aquatic ecosystems.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Station")
    choice = st.selectbox("USGS station", list(STATIONS))
    station = st.text_input("USGS station ID", "") if STATIONS[choice] == "custom" else STATIONS[choice]
    noaa = st.text_input(
        "Optional NOAA CO-OPS station ID",
        help="Used to fill missing temperature or salinity observations.",
    )
    refresh = st.button("Refresh", use_container_width=True)

if refresh:
    st.cache_data.clear()

if not re.fullmatch(r"\d{8}(\d{7})?", station.strip()):
    st.error("Enter a valid USGS station ID (8 or 15 digits).")
    st.stop()

try:
    assets = load_assets()
    with st.spinner("Loading station observations..."):
        raw_history, raw_provenance = fetch_usgs(station.strip())

    st.write("USGS columns returned:", raw_history.columns.tolist())

    if "salinity_PSU" in raw_history.columns:
        st.write(
            "Raw salinity observations:",
            raw_history["salinity_PSU"].notna().sum()
        )
        st.write(
            "Latest raw salinity readings:",
            raw_history["salinity_PSU"].dropna().tail(10)
        )
    else:
        st.error("USGS did not return a salinity_PSU column.")

    history, hourly_provenance, source_notes = hourly_history(
    raw_history,
    raw_provenance,
    noaa,
)

except Exception as exc:
    st.error(f"Could not load station data: {exc}")
    st.stop()

st.subheader(f"Station: {station.strip()}")

cols = st.columns(4)
for i, (label, meta) in enumerate(VARIABLES.items()):
    series = history[meta["column"]].dropna()
    with cols[i]:
        if series.empty:
            st.metric(label, "Unavailable")
            st.caption("No recent observation returned")
        else:
            latest = float(series.iloc[-1])
            prior = series.loc[series.index <= series.index[-1] - pd.Timedelta(hours=24)]
            delta = latest - float(prior.iloc[-1]) if not prior.empty else None
            st.metric(
                label,
                f'{latest:.2f} {meta["unit"]}',
                delta=f'{delta:+.2f} {meta["unit"]} vs 24h' if delta is not None else None,
                delta_color="normal",
            )
            st.caption(
                source_notes.get(meta["column"], "USGS NWIS")
                + f' · {series.index[-1].strftime("%b %d, %H:%M UTC")}'
            )

try:
    with st.spinner("Updating station adjustment from recent measurements..."):
        adaptation = build_station_adaptation(history, hourly_provenance, station.strip(), assets)
        forecast = make_forecast(history, station.strip(), assets, adaptation)
except Exception as exc:
    st.warning(f"Forecast unavailable: {exc}")
    st.stop()

origin = forecast["origin"].iloc[0]
start, end = origin - pd.Timedelta(hours=24), origin + pd.Timedelta(hours=24)

latest_obs_time = history.dropna(how="all").index[-1] if not history.empty else origin
origin_age_hours = (latest_obs_time - origin).total_seconds() / 3600.0

current_time = pd.Timestamp.now(tz="UTC")
latest_data_age_hours = (
    current_time - latest_obs_time
).total_seconds() / 3600.0

st.subheader("Forecast origin & freshness")
fresh_col1, fresh_col2, fresh_col3 = st.columns(3)
with fresh_col1:
    st.metric("Forecast Origin (UTC)", origin.strftime("%Y-%m-%d %H:%M"))
with fresh_col2:
    st.metric("Latest Observation (UTC)", latest_obs_time.strftime("%Y-%m-%d %H:%M"))
with fresh_col3:
    if origin_age_hours > 3.0:
        st.metric("Origin Age", f"{origin_age_hours:.1f} hours", delta="Delayed (>3h)", delta_color="inverse")
    else:
        st.metric("Origin Age", f"{origin_age_hours:.1f} hours", delta="Fresh (≤3h)", delta_color="normal")

if origin_age_hours > 3.0:
    st.warning(f"⚠️ Forecast origin is {origin_age_hours:.1f} hours behind the latest available observation. Predictions are based on the complete feature history up to {origin.strftime('%Y-%m-%d %H:%M UTC')}.")
else:
    st.success(f"✅ Forecast origin is up to date (age: {origin_age_hours:.1f} hours).")

if latest_data_age_hours > 3.0:
    st.warning(
        f"Latest observations are {latest_data_age_hours:.1f} hours old. "
        "The station data may be delayed."
    )

st.subheader("Station adjustment")
adaptation_rows = []
for label, meta in VARIABLES.items():
    key = meta["key"]
    details = [adaptation.get(key, {}).get(h, {}) for h in range(1, 25)]
    active = [item for item in details if item.get("validated", False)]
    counts = [item.get("examples", 0) for item in details]
    corrections = [item.get("correction", 0.0) for item in active]
    adaptation_rows.append({
        "Parameter": label,
        "Adapted horizons": f"{len(active)}/24",
        "Historical examples": max(counts, default=0),
        "Median applied correction": (
            f'{float(np.median(corrections)):+.3f} {meta["unit"]}' if corrections else "Not active"
        ),
        "Status": "Active" if active else "No validated improvement",
    })

adapt_cols = st.columns(4)
for index, row in enumerate(adaptation_rows):
    with adapt_cols[index]:
        st.metric(row["Parameter"], row["Status"])
        st.caption(
            f'{row["Adapted horizons"]} horizons adapted · '
            f'{row["Historical examples"]} max examples'
        )
st.caption(
    "Corrections are learned from historical forecasts simulated using only "
    "measurements available at each past forecast origin. A correction is "
    "considered only when at least 18 valid examples are available: "
    "12 for training and 6 for chronological validation. It is applied "
    "only when it improves validation MAE."
)

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.11)
colors = {
    "observed": "#1769aa",
    "forecast": "#328cc4",
    "original": "#58738d",
}
for row, (label, meta) in enumerate(VARIABLES.items(), start=1):
    col = meta["column"]
    observed = history.loc[(history.index >= start) & (history.index <= origin), col].dropna()
    subset = forecast[forecast.variable == meta["key"]].sort_values("horizon")

    if not observed.empty:
        fig.add_trace(
            go.Scatter(
                x=observed.index,
                y=observed.values,
                mode="lines",
                name="Observed",
                legendgroup="observed",
                showlegend=row == 1,
                line={"color": colors["observed"], "width": 2.5},
                hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}<extra>Observed</extra>",
            ),
            row=row,
            col=1,
        )

    if not subset.empty:
        if subset["station_correction"].abs().gt(1e-9).any():
            fig.add_trace(
                go.Scatter(
                    x=subset.timestamp,
                    y=subset.original_prediction,
                    mode="lines",
                    name="Original model",
                    legendgroup="original",
                    showlegend=row == 1,
                    line={"color": colors["original"], "width": 1.5, "dash": "dot"},
                    hovertemplate="%{x|%b %d %H:%M} UTC<br>%{y:.2f}<extra>Original model</extra>",
                ),
                row=row,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=subset.timestamp,
                y=subset.prediction,
                mode="lines+markers",
                name="Station-adjusted forecast",
                legendgroup="forecast",
                showlegend=row == 1,
                line={"color": colors["forecast"], "width": 2.2},
                marker={"size": 4},
                customdata=subset[["station_correction", "adaptation_examples"]].to_numpy(),
                hovertemplate=(
                    "%{x|%b %d %H:%M} UTC<br>%{y:.2f}"
                    "<br>Correction: %{customdata[0]:+.3f}"
                    "<br>Historical examples: %{customdata[1]}"
                    "<extra>Station-adjusted forecast</extra>"
                ),
            ),
            row=row,
            col=1,
        )

    fig.update_yaxes(
        title_text=f'{label}<br>({meta["unit"]})',
        row=row,
        col=1,
        gridcolor="#e3edf6",
        zerolinecolor="#d4e2ef",
        title_font={"size": 11},
    )

fig.add_vline(x=origin, line_width=1.5, line_dash="dot", line_color="#e4a43b")
fig.add_annotation(
    x=origin,
    y=1.015,
    xref="x",
    yref="paper",
    text="FORECAST ORIGIN",
    showarrow=False,
    font={"size": 10, "color": "#805916"},
    bgcolor="#fff7e6",
    borderpad=4,
)
fig.update_xaxes(range=[start, end], title_text="Time (UTC)", row=4, col=1, gridcolor="#edf2f8")
fig.update_layout(
    height=1000,
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="rgba(255,255,255,0)",
    font={"family": "DM Sans, sans-serif", "color": "#18324b"},
    legend={"orientation": "h", "y": 1.04, "x": 0.5, "xanchor": "center"},
    margin={"t": 70, "b": 40, "l": 110, "r": 20},
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Adjustment details"):
    detail_rows = []
    for label, meta in VARIABLES.items():
        key = meta["key"]
        for horizon in range(1, 25):
            item = adaptation.get(key, {}).get(horizon, {})
            sc = item.get("source_counts", {"USGS Direct": 0, "NOAA CO-OPS": 0})
            detail_rows.append({
                "Parameter": label,
                "Horizon": f"+{horizon}h",
                "Selected base method": item.get("method", "unavailable"),
                "Historical examples": item.get("examples", 0),
                "USGS Direct examples": sc.get("USGS Direct", 0),
                "NOAA CO-OPS examples": sc.get("NOAA CO-OPS", 0),
                "Median historical error": item.get("median_error", 0.0),
                "Applied correction": item.get("correction", 0.0),
                "Validated": item.get("validated", False),
                "Validation examples": item.get("validation_examples", 0),
                "Validation MAE before": item.get("validation_mae_before"),
                "Validation MAE after": item.get("validation_mae_after"),
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    comparison = forecast[[
        "label",
        "horizon",
        "original_prediction",
        "station_correction",
        "prediction",
        "adaptation_examples",
        "method_source",
    ]].copy()
    comparison.columns = [
        "Parameter",
        "Horizon",
        "Original prediction",
        "Applied correction",
        "Adapted prediction",
        "Historical examples",
        "Base method",
    ]
    st.dataframe(comparison, use_container_width=True, hide_index=True)

with st.expander("Data sources"):
    source_rows = []
    for label, meta in VARIABLES.items():
        valid = history[meta["column"]].dropna()
        source_rows.append({
            "Parameter": label,
            "Latest source": source_notes.get(meta["column"], "USGS NWIS"),
            "Latest observation (UTC)": (
                valid.index[-1].strftime("%Y-%m-%d %H:%M") if not valid.empty else "Unavailable"
            ),
            "Hourly observations": int(valid.size),
        })
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
    for note in source_notes.values():
        st.caption(note)

st.caption(
    "Forecasts use XGBoost or persistence based on the station validation hierarchy, then apply "
    "station-specific historical error corrections when enough examples are available. "
    "All timestamps are UTC."
)
