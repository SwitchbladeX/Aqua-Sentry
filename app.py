from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="Long Island Estuary Water Quality Portal",
    page_icon="🌊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
VARIABLES = {
    "Dissolved oxygen": {
        "key": "do", "folder": "dissolved_oxygen", "prefix": "DO",
        "column": "DO_mgL", "unit": "mg/L", "usgs_code": "00300",
        "noaa_product": None,
    },
    "Temperature": {
        "key": "temp", "folder": "temperature", "prefix": "Temp",
        "column": "temp_C", "unit": "°C", "usgs_code": "00010",
        "noaa_product": "water_temperature",
    },
    "pH": {
        "key": "ph", "folder": "ph", "prefix": "pH",
        "column": "pH", "unit": "pH", "usgs_code": "00400",
        "noaa_product": None,
    },
    "Salinity": {
        "key": "salinity", "folder": "salinity", "prefix": "Salinity",
        "column": "salinity_PSU", "unit": "PSU", "usgs_code": "00480",
        "noaa_product": "salinity",
    },
}
VALUE_COLUMNS = ["DO_mgL", "temp_C", "pH", "salinity_PSU"]
LAGS = [1, 3, 6, 12]
FEATURE_COLUMNS = (
    VALUE_COLUMNS
    + [f"{col}_lag{lag}h" for col in VALUE_COLUMNS for lag in LAGS]
    + ["hour", "month"]
)
HIERARCHY_FILES = {
    "do": "DO_VALIDATION_HIERARCHY.csv",
    "temp": "TEMP_VALIDATION_HIERARCHY.csv",
    "ph": "PH_VALIDATION_HIERARCHY.csv",
    "salinity": "SALINITY_VALIDATION_HIERARCHY.csv",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { background: #f7f9fc; }
    [data-testid="stSidebar"] { background: #102238; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stSidebar"] input { color: #172033 !important; }
    .small-note { color: #64748b; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)



@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs_history(site_id: str, days: int = 90) -> pd.DataFrame:
    """Retrieve USGS instantaneous values; no synthetic values are created."""
    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site_id,
        "parameterCd": "00010,00300,00400,00480",
        "period": f"P{days}D",
    }
    response = requests.get(
        url, params=params, timeout=45,
        headers={"User-Agent": "LongIslandEstuaryWaterQualityPortal/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    series = payload.get("value", {}).get("timeSeries", [])
    if not series:
        raise ValueError("USGS returned no instantaneous-value series for this station.")

    pieces = []
    code_to_column = {v["usgs_code"]: v["column"] for v in VARIABLES.values()}
    for item in series:
        codes = item.get("variable", {}).get("variableCode", [])
        if not codes:
            continue
        code = codes[0].get("value")
        column = code_to_column.get(code)
        if not column:
            continue
        value_groups = item.get("values", [])
        if not value_groups:
            continue
        observations = value_groups[0].get("value", [])
        if not observations:
            continue
        part = pd.DataFrame(observations)
        if not {"dateTime", "value"}.issubset(part.columns):
            continue
        part["timestamp"] = pd.to_datetime(part["dateTime"], utc=True, errors="coerce")
        part[column] = pd.to_numeric(part["value"], errors="coerce")
        part = part[["timestamp", column]].dropna(subset=["timestamp"])
        part = part.set_index("timestamp")
        pieces.append(part.groupby(level=0).median(numeric_only=True))

    if not pieces:
        raise ValueError("USGS returned data, but no supported parameter series could be parsed.")
    raw = pd.concat(pieces, axis=1)
    raw = raw.groupby(level=0).median(numeric_only=True).sort_index()
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.attrs["source"] = "USGS NWIS instantaneous values"
    return raw


def fetch_noaa_product(station_id: str, product: str, days: int = 90) -> pd.Series:
    """Retrieve compatible NOAA CO-OPS hourly temperature or salinity."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "product": product,
        "application": "LongIslandEstuaryWaterQualityPortal",
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": station_id,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "interval": "h",
    }
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload["error"].get("message", "NOAA returned an API error."))
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"NOAA returned no {product} observations for station {station_id}.")
    frame = pd.DataFrame(rows)
    value_key = "v"
    if value_key not in frame:
        raise ValueError(f"NOAA response did not contain a value field for {product}.")
    frame["timestamp"] = pd.to_datetime(frame["t"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame[value_key], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"]).set_index("timestamp")
    return frame["value"].groupby(level=0).median().sort_index()


def merge_optional_noaa(
    history: pd.DataFrame, noaa_station: str, days: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Use NOAA only for temperature/salinity, only when user supplies a station ID."""
    sources = {col: "USGS NWIS" for col in history.columns}
    if not noaa_station.strip():
        return history, sources

    result = history.copy()
    for col, product in [("temp_C", "water_temperature"), ("salinity_PSU", "salinity")]:
        if col in result.columns and result[col].notna().sum() > 0:
            continue
        try:
            series = fetch_noaa_product(noaa_station.strip(), product, days)
            result[col] = series.reindex(result.index)
            sources[col] = f"NOAA CO-OPS ({product})"
        except Exception as exc:
            sources[col] = f"Unavailable from USGS; NOAA fallback failed: {exc}"
    return result, sources


def load_uploaded_sensor_csv(uploaded_file) -> pd.DataFrame:
    """Accept a timestamp column and any subset of the four model variables."""
    frame = pd.read_csv(uploaded_file)
    aliases = {
        "timestamp": "timestamp", "datetime": "timestamp", "date_time": "timestamp",
        "date": "timestamp", "time": "timestamp",
        "do_mgl": "DO_mgL", "do": "DO_mgL", "dissolved_oxygen": "DO_mgL",
        "temp_c": "temp_C", "temperature": "temp_C",
        "ph": "pH", "salinity_psu": "salinity_PSU", "salinity": "salinity_PSU",
    }
    rename = {}
    for col in frame.columns:
        normalized = re.sub(r"[^a-z0-9_]", "", str(col).strip().lower())
        if normalized in aliases:
            rename[col] = aliases[normalized]
    frame = frame.rename(columns=rename)
    if "timestamp" not in frame:
        raise ValueError("CSV needs a timestamp column (timestamp, datetime, date_time, date, or time).")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
    for col in VALUE_COLUMNS:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    usable = [c for c in VALUE_COLUMNS if c in frame]
    if not usable:
        raise ValueError("CSV must contain at least one recognized water-quality column.")
    return frame[usable].sort_index()


def build_hourly_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly median; missing hours remain missing, never blanket-filled."""
    if raw.empty:
        raise ValueError("No observations were returned.")
    frame = raw.copy()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.resample("1h").median(numeric_only=True)
    for col in VALUE_COLUMNS:
        if col not in frame:
            frame[col] = np.nan
    return frame[VALUE_COLUMNS]

def find_file_case_insensitive(folder: Path, expected_name: str):
    if not folder.exists():
        return None
    target = expected_name.lower()
    for path in folder.iterdir():
        if path.is_file() and path.name.lower() == target:
            return path
    return None


def normalize_hierarchy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "site_code": "station", "station": "station", "station_id": "station",
        "horizon": "horizon", "forecast_hour": "horizon", "forecast_horizon": "horizon",
        "selected_method": "method", "method": "method",
        "xgboost_mae": "xgb_mae", "xgb_mae": "xgb_mae",
        "persistence_mae": "persistence_mae",
        "validation_rows": "validation_rows",
    }
    rename = {}
    for col in frame.columns:
        key = re.sub(r"[^a-z0-9_]", "", str(col).strip().lower())
        if key in aliases:
            rename[col] = aliases[key]
    frame = frame.rename(columns=rename)
    required = {"station", "horizon", "method"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Hierarchy CSV is missing required columns: {sorted(required - set(frame.columns))}")
    frame["station"] = frame["station"].astype(str).str.strip()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["method"] = frame["method"].astype(str).str.strip().str.lower()
    frame.loc[frame["method"].str.contains("xg", na=False), "method"] = "xgboost"
    frame.loc[frame["method"].str.contains("persist", na=False), "method"] = "persistence"
    return frame.dropna(subset=["horizon"])


@st.cache_resource(show_spinner=False)
def load_models_and_hierarchies():
    models = {}
    hierarchies = {}
    problems = []
    for label, meta in VARIABLES.items():
    key = meta["key"]
    folder = MODEL_ROOT / meta["folder"]
        models[key] = {}
        for horizon in range(1, 25):
            filename = f'{meta["prefix"]}_{horizon}h.json'
            path = find_file_case_insensitive(folder, filename)
            if path is None:
                problems.append(f"Missing model: {folder.relative_to(ROOT)}/{filename}")
                continue
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            models[key][horizon] = model

        hierarchy_path = find_file_case_insensitive(folder, HIERARCHY_FILES[key])
        if hierarchy_path is None:
            problems.append(f"Missing hierarchy CSV: {folder.relative_to(ROOT)}/{HIERARCHY_FILES[key]}")
            hierarchies[key] = pd.DataFrame()
        else:
            hierarchies[key] = normalize_hierarchy_columns(pd.read_csv(hierarchy_path))
    return models, hierarchies, problems


def make_feature_row(history: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Build the exact 20-column feature order used in the training notebook."""
    row = {}
    current = history.loc[timestamp]
    for col in VALUE_COLUMNS:
        row[col] = float(current[col])
    for col in VALUE_COLUMNS:
        for lag in LAGS:
            lag_time = timestamp - pd.Timedelta(hours=lag)
            value = history.at[lag_time, col] if lag_time in history.index else np.nan
            row[f"{col}_lag{lag}h"] = float(value) if pd.notna(value) else np.nan
    row["hour"] = int(timestamp.hour)
    row["month"] = int(timestamp.month)
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def get_station_method(hierarchy: pd.DataFrame, station: str, horizon: int):
    if hierarchy.empty:
        return None
    match = hierarchy[
        (hierarchy["station"].astype(str) == str(station))
        & (hierarchy["horizon"] == horizon)
    ]
    if match.empty:
        return None
    method = str(match.iloc[0]["method"]).lower()
    return method if method in {"xgboost", "persistence"} else None


def forecast_all(
    history: pd.DataFrame,
    station: str,
    models: dict,
    hierarchies: dict,
):
    complete_times = []
    for timestamp in history.index:
        try:
            features = make_feature_row(history, timestamp)
            if features[FEATURE_COLUMNS].notna().all(axis=None):
                complete_times.append(timestamp)
        except (KeyError, TypeError, ValueError):
            continue
    if not complete_times:
        raise ValueError(
            "There is no timestamp with all four current measurements and the required "
            "1-, 3-, 6-, and 12-hour lag values. The app will not invent missing inputs."
        )
    origin = complete_times[-1]
    feature_row = make_feature_row(history, origin)
    forecasts = []
    for key, meta in VARIABLES.items():
        for horizon in range(1, 25):
            method = get_station_method(hierarchies.get(key, pd.DataFrame()), station, horizon)
            if method is None:
                method = "persistence"
                method_source = "Fallback: no station+horizon hierarchy row"
            else:
                method_source = "Station-specific validation hierarchy"
            if method == "xgboost":
                model = models.get(key, {}).get(horizon)
                if model is None:
                    raise ValueError(f"Missing {meta['prefix']}_{horizon}h model.")
                booster_names = model.get_booster().feature_names
                if booster_names and list(booster_names) != FEATURE_COLUMNS:
                    raise ValueError(
                        f"{meta['prefix']}_{horizon}h feature schema does not match the "
                        "training feature order. Forecast stopped rather than guessing."
                    )
                value = float(model.predict(feature_row)[0])
            else:
                value = float(feature_row.iloc[0][meta["column"]])
            forecasts.append({
                "variable": meta["key"], "label": meta["column"],
                "horizon": horizon, "timestamp": origin + pd.Timedelta(hours=horizon),
                "prediction": value, "method": method, "method_source": method_source,
            })
    return pd.DataFrame(forecasts), origin


st.title("Long Island Estuary Water Quality Portal")
st.caption("Station observations, transparent data provenance, and 24-hour parameter forecasts")

with st.sidebar:
    st.header("Station setup")
    station_id = st.text_input("USGS station ID", value="01491000", help="Enter an 8- or 15-character USGS monitoring-location ID.")
    days = st.selectbox("Historical context", [30, 60, 90], index=2)
    st.markdown("---")
    st.subheader("Optional compatible NOAA station")
    noaa_station = st.text_input(
        "NOAA CO-OPS station ID",
        value="",
        help="Optional. Used only as a fallback for water temperature/salinity when USGS does not provide that variable. Confirm it represents the same water body/location.",
    )
    st.markdown("---")
    st.subheader("Optional sensor data")
    sensor_upload = st.file_uploader("Upload sensor CSV", type=["csv"])
    st.caption("CSV needs timestamps and recognized columns such as DO_mgL, temp_C, pH, salinity_PSU.")
    refresh = st.button("Refresh observations", use_container_width=True)

if refresh:
    st.cache_data.clear()

if not re.fullmatch(r"\d{8}(\d{7})?", station_id.strip()):
    st.error("Enter a valid-looking USGS station ID (8 or 15 digits).")
    st.stop()

models, hierarchies, model_problems = load_models_and_hierarchies()
if model_problems:
    st.error("The model suite is incomplete. Fix these repository paths before forecasting:")
    for problem in model_problems:
        st.write(f"- {problem}")
    st.stop()

with st.spinner(f"Retrieving up to {days} days of observations..."):
    try:
        raw_history = fetch_usgs_history(station_id.strip(), days)
        source_notes = {col: "USGS NWIS" for col in raw_history.columns}
        raw_history, source_notes = merge_optional_noaa(raw_history, noaa_station, days)
        if sensor_upload is not None:
            sensor_data = load_uploaded_sensor_csv(sensor_upload)
            raw_history = raw_history.combine_first(sensor_data)
            for col in sensor_data.columns:
                if sensor_data[col].notna().any():
                    source_notes[col] = "User-uploaded sensor CSV fills missing timestamps"
        hourly = build_hourly_history(raw_history)
    except Exception as exc:
        st.error(f"Could not retrieve/prepare station observations: {exc}")
        st.info("Check the station ID, network connection, and whether the station reports the required parameters.")
        st.stop()

latest_by_variable = {}
for col in VALUE_COLUMNS:
    valid = hourly[col].dropna()
    latest_by_variable[col] = valid.index.max() if not valid.empty else pd.NaT

st.subheader(f"Station {station_id.strip()} — observation status")
metric_cols = st.columns(4)
for i, (label, meta) in enumerate(VARIABLES.items()):
    col = meta["column"]
    valid = hourly[col].dropna()
    with metric_cols[i]:
        if valid.empty:
            st.metric(label, "Unavailable")
            st.caption("No compatible source returned this variable")
        else:
            latest_time = valid.index.max()
            age_hours = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() / 3600
            st.metric(label, f"{valid.iloc[-1]:.2f} {meta['unit']}")
            st.caption(f"{source_notes.get(col, 'USGS NWIS')} · {age_hours:.1f}h old")

coverage = pd.DataFrame({
    "Parameter": [v["column"] for v in VARIABLES.values()],
    "Available hourly records": [int(hourly[v["column"]].notna().sum()) for v in VARIABLES.values()],
    "Coverage": [f"{hourly[v['column']].notna().mean() * 100:.1f}%" for v in VARIABLES.values()],
    "Latest observation (UTC)": [
        latest_by_variable[v["column"]].strftime("%Y-%m-%d %H:%M") if pd.notna(latest_by_variable[v["column"]]) else "Unavailable"
        for v in VARIABLES.values()
    ],
    "Source": [source_notes.get(v["column"], "Unavailable") for v in VARIABLES.values()],
})
with st.expander("Data coverage and provenance", expanded=False):
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    st.write(
        "USGS measurements are the primary source. NOAA CO-OPS is queried only when you provide a station ID "
        "and USGS has no observations for temperature or salinity. NOAA is not used for dissolved oxygen or pH. "
        "No conductivity-to-salinity conversion, zero-filling, or long-gap interpolation is performed."
    )

st.subheader("Historical station record")
history_chart = go.Figure()
for col, label in [
    ("DO_mgL", "Dissolved oxygen (mg/L)"),
    ("temp_C", "Temperature (°C)"),
    ("pH", "pH"),
    ("salinity_PSU", "Salinity (PSU)"),
]:
    history_chart.add_trace(go.Scatter(
        x=hourly.index, y=hourly[col], mode="lines", name=label,
        connectgaps=False,
    ))
history_chart.update_layout(
    height=420, hovermode="x unified",
    xaxis_title="Time (UTC)", yaxis_title="Observed value",
    legend_title="Parameter",
)
st.plotly_chart(history_chart, use_container_width=True)
st.caption("Separate parameter units share the plot; use the data table for exact values. Gaps are intentionally left visible.")

with st.expander("Hourly observations table"):
    st.dataframe(hourly.tail(24 * 14), use_container_width=True)

st.subheader("Recent trends")
trend_cols = st.columns(4)
for i, (label, meta) in enumerate(VARIABLES.items()):
    series = hourly[meta["column"]].dropna()
    with trend_cols[i]:
        if len(series) >= 2:
            first, last = series.iloc[0], series.iloc[-1]
            st.metric(label, f"{last:.2f} {meta['unit']}", delta=f"{last - first:+.2f} over available record")
        else:
            st.metric(label, "Insufficient data")
st.caption("These are descriptive changes across the available historical record, not a causal analysis or proof of station-specific model adaptation.")

st.subheader("24-hour forecast")
try:
    forecast, forecast_origin = forecast_all(hourly, station_id.strip(), models, hierarchies)
except Exception as exc:
    st.warning(f"Forecast is not available yet: {exc}")
    st.info(
        "The model requires all four present measurements and exact 1-, 3-, 6-, and 12-hour lag values. "
        "Missing inputs are not replaced with zeros or synthetic estimates."
    )
    st.stop()

st.caption(f"Forecast issued from the latest complete feature timestamp: {forecast_origin.strftime('%Y-%m-%d %H:%M UTC')}")
forecast_tabs = st.tabs(["Dissolved oxygen", "Temperature", "pH", "Salinity", "Forecast methods"])
for tab, (label, meta) in zip(forecast_tabs[:4], VARIABLES.items()):
    with tab:
        subset = forecast[forecast["variable"] == meta["key"]].sort_values("horizon")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset["timestamp"], y=subset["prediction"],
            mode="lines+markers", name=f"{label} forecast",
            connectgaps=False,
        ))
        if meta["key"] == "do":
            fig.add_hline(y=3.0, line_dash="dash", line_color="#dc2626",
                          annotation_text="3 mg/L reference threshold")
            fig.add_hline(y=5.0, line_dash="dot", line_color="#d97706",
                          annotation_text="5 mg/L reference")
        fig.update_layout(
            height=390, xaxis_title="Forecast time (UTC)",
            yaxis_title=meta["unit"], hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        selected = st.select_slider(
            f"Inspect {label} forecast hour",
            options=list(range(1, 25)), value=1, key=f"hour_{meta['key']}",
        )
        row = subset.loc[subset["horizon"] == selected].iloc[0]
        st.metric(f"{label} at +{selected}h", f"{row['prediction']:.3f} {meta['unit']}")
        st.caption(f"Method: {row['method']} · {row['method_source']}")

with forecast_tabs[4]:
    method_table = forecast.pivot(index="horizon", columns="variable", values="method").reset_index()
    method_table = method_table.rename(columns={
        "horizon": "Forecast hour", "do": "DO", "temp": "Temperature",
        "ph": "pH", "salinity": "Salinity",
    })
    st.dataframe(method_table, use_container_width=True, hide_index=True)
    st.write(
        "For each variable and horizon, the app uses the station-specific validation hierarchy when a matching "
        "station+horizon row exists. If there is no row, it explicitly falls back to persistence (the latest "
        "observed value); that fallback is not presented as a hierarchy decision."
    )

st.subheader("Model intelligence")
int_cols = st.columns(3)
with int_cols[0]:
    st.metric("Forecast horizons", "24")
    st.caption("One independent model per variable and horizon")
with int_cols[1]:
    st.metric("Model suite", "96 regressors")
    st.caption("4 water-quality variables × 24 horizons")
with int_cols[2]:
    st.metric("Station context", f"{days} days")
    st.caption("Used for lags, coverage, and trend analysis")

with st.expander("How the data and prediction hierarchy works", expanded=False):
    st.markdown("""
    **Observation-source order**
    1. **USGS NWIS** is queried first for water temperature, dissolved oxygen, pH, and direct salinity.
    2. **NOAA CO-OPS** can supplement water temperature and salinity only when you explicitly supply a NOAA station ID and USGS has no observations for that variable. Confirm the NOAA station is physically representative of the USGS site before using it.
    3. **User sensor CSV** can fill timestamps not present in the API record. Its values are labeled as user-uploaded data.
    4. If a parameter is still missing, it stays missing. The app does not silently borrow a distant station or invent values.

    **Prediction-method order**
    - For each target variable and forecast hour, look up the station and horizon in that variable's validation hierarchy CSV.
    - If the hierarchy selected XGBoost, use that exact horizon-specific JSON model.
    - If it selected persistence, carry forward the latest observed value.
    - If no station+horizon row exists, use a clearly labeled persistence fallback.

    **Three-month station context**
    - The selected 30/60/90-day record supplies current values, lag features, coverage, and descriptive trends.
    - This version does **not** automatically retrain or fine-tune the 96 base models. Three months of data alone is not enough to justify adaptation: a safe adaptation workflow needs correctly aligned future labels for each horizon, chronological holdout validation, and proof that the adapted model beats both the base model and persistence.
    """)

with st.expander("Important scientific limitations", expanded=True):
    st.warning(
        "This is a research prototype, not a validated operational warning service. "
        "The former hypoxia classifier and arbitrary WQI score have been removed. "
        "A dissolved-oxygen forecast below 3 mg/L is a threshold-based flag for review, "
        "not a calibrated probability of a hypoxia event."
    )
    st.markdown(
        "- USGS and NOAA stations may differ in location, depth, sensor, sampling interval, and QA status.\n"
        "- NOAA data are not automatically matched to a USGS station; the operator must confirm the station pairing.\n"
        "- The app does not yet perform spatial interpolation or automatic NERRS retrieval. NERRS CDMO access requires authorization.\n"
        "- Forecast quality depends on the model's training domain and the completeness/freshness of the station record."
    )

st.divider()
st.caption(
    "Data services: USGS Water Data / NWIS and, when explicitly configured, NOAA CO-OPS. "
    "All timestamps shown in this app are UTC unless otherwise labeled."
)
