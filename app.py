from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="Long Island Estuary Water Quality",
    page_icon="🌊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
VARIABLES = {
    "Dissolved oxygen": {
        "key": "do", "folder": "dissolved_oxygen", "prefix": "DO",
        "column": "DO_mgL", "unit": "mg/L", "usgs_code": "00300",
    },
    "Temperature": {
        "key": "temp", "folder": "temperature", "prefix": "Temp",
        "column": "temp_C", "unit": "°C", "usgs_code": "00010",
    },
    "pH": {
        "key": "ph", "folder": "ph", "prefix": "pH",
        "column": "pH", "unit": "pH", "usgs_code": "00400",
    },
    "Salinity": {
        "key": "salinity", "folder": "salinity", "prefix": "Salinity",
        "column": "salinity_PSU", "unit": "PSU", "usgs_code": "00480",
    },
}
VALUE_COLUMNS = ["DO_mgL", "temp_C", "pH", "salinity_PSU"]
STATION_PRESETS = {
    "West Harbor at Oyster Bay, NY — 01303152": "01303152",
    "Peconic River at Riverhead, NY — 01304562": "01304562",
    "South side of Shelter Island, NY — 01304200": "01304200",
    "Orient Point Harbor, NY — 01304190": "01304190",
    "Forge River at Mastic, NY — 01304057": "01304057",
    "Hempstead Bay at Freeport, NY — 01304998": "01304998",
    "Enter a different USGS station": "custom",
}
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
SALINITY_PLAUSIBLE_RANGE = (0.0, 45.0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { color-scheme: light; }
.stApp { background: #f3f8fa; color: #102b46; }
[data-testid="stHeader"] { background: rgba(243,248,250,.96); }
[data-testid="stSidebar"] { background: #102b46; }
[data-testid="stSidebar"] * { color: #f4fbfc !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    background: #ffffff !important; color: #102b46 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #102b46 !important; }
.stButton > button, .stDownloadButton > button,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
    background: #087f8c !important; color: #ffffff !important;
    border: 1px solid #087f8c !important; border-radius: 9px !important;
    font-weight: 700 !important; box-shadow: none !important;
}
.stButton > button *, .stDownloadButton > button *,
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-secondary"] * {
    color: #ffffff !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
    background: #075b68 !important; border-color: #075b68 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #c9f1e8 !important; color: #102b46 !important; border-color: #c9f1e8 !important;
}
[data-testid="stSidebar"] .stButton > button * { color: #102b46 !important; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d5e5ea; padding: 14px; border-radius: 12px; }
[data-testid="stExpander"] { background: #ffffff; border: 1px solid #d5e5ea; border-radius: 12px; }
[data-testid="stCaptionContainer"] { color: #496273 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_usgs_history(site_id: str, days: int = 90) -> pd.DataFrame:
    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site_id,
        "parameterCd": "00010,00300,00400,00480,90860",
        "period": f"P{days}D",
    }
    response = requests.get(
        url, params=params, timeout=45,
        headers={"User-Agent": "LongIslandEstuaryWaterQualityPortal/1.0"},
    )
    response.raise_for_status()
    series = response.json().get("value", {}).get("timeSeries", [])
    if not series:
        raise ValueError("USGS returned no instantaneous-value series for this station.")

    code_to_column = {value["usgs_code"]: value["column"] for value in VARIABLES.values()}
    code_to_column.update({"90860": "salinity_PSU"})
    pieces = []
    for item in series:
        codes = item.get("variable", {}).get("variableCode", [])
        groups = item.get("values", [])
        if not codes or not groups:
            continue
        column = code_to_column.get(codes[0].get("value"))
        observations = groups[0].get("value", [])
        if column is None or not observations:
            continue
        part = pd.DataFrame(observations)
        if not {"dateTime", "value"}.issubset(part.columns):
            continue
        part["timestamp"] = pd.to_datetime(part["dateTime"], utc=True, errors="coerce")
        part[column] = pd.to_numeric(part["value"], errors="coerce")
        part = part[["timestamp", column]].dropna(subset=["timestamp"]).set_index("timestamp")
        pieces.append(part.groupby(level=0).median(numeric_only=True))

    if not pieces:
        raise ValueError("USGS returned data, but no supported parameter series could be parsed.")
    raw = pd.concat(pieces, axis=1)
    raw = raw.groupby(level=0).median(numeric_only=True).sort_index()
    raw = raw.loc[:, ~raw.columns.duplicated()]
    return raw


def fetch_noaa_product(station_id: str, product: str, days: int = 90) -> pd.Series:
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
    response = requests.get(
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        params=params, timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload["error"].get("message", "NOAA returned an API error."))
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"NOAA returned no {product} observations for station {station_id}.")
    frame = pd.DataFrame(rows)
    if "v" not in frame or "t" not in frame:
        raise ValueError(f"NOAA response did not contain timestamp and value fields for {product}.")
    frame["timestamp"] = pd.to_datetime(frame["t"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["v"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"]).set_index("timestamp")
    return frame["value"].groupby(level=0).median().sort_index()


def build_hourly_history(raw: pd.DataFrame) -> pd.DataFrame:
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


def apply_observation_hierarchy(
    hourly: pd.DataFrame, noaa_station: str, days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    result = hourly.copy()
    provenance = pd.DataFrame(index=result.index, columns=VALUE_COLUMNS, dtype=object)
    for col in VALUE_COLUMNS:
        provenance.loc[result[col].notna(), col] = "USGS NWIS"

    fallback_notes = {}
    if noaa_station.strip():
        for col, product in [("temp_C", "water_temperature"), ("salinity_PSU", "salinity")]:
            missing = result[col].isna()
            if not missing.any():
                continue
            try:
                series = fetch_noaa_product(noaa_station.strip(), product, days)
                values = series.resample("1h").median().reindex(result.index)
                mask = missing & values.notna()
                result.loc[mask, col] = values.loc[mask]
                provenance.loc[mask, col] = f"NOAA CO-OPS ({product})"
                fallback_notes[col] = f"NOAA CO-OPS supplied {int(mask.sum())} hourly observations"
            except Exception as exc:
                fallback_notes[col] = f"NOAA CO-OPS unavailable: {exc}"

    source_notes = {}
    for col in VALUE_COLUMNS:
        valid_times = result.index[result[col].notna()]
        if len(valid_times) == 0:
            source_notes[col] = "Unavailable"
        else:
            source_notes[col] = str(provenance.at[valid_times[-1], col])
    return result, provenance, source_notes, fallback_notes


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
        "persistence_mae": "persistence_mae", "validation_rows": "validation_rows",
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
    frame["station"] = frame["station"].astype(str).str.strip().str.replace(r"^USGS-", "", regex=True)
    frame["station"] = frame["station"].str.replace(r"\.0$", "", regex=True)
    frame["station"] = frame["station"].apply(
        lambda value: value.zfill(8) if value.isdigit() and len(value) <= 8 else value
    )
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["method"] = frame["method"].astype(str).str.strip().str.lower()
    frame.loc[frame["method"].str.contains("xg", na=False), "method"] = "xgboost"
    frame.loc[frame["method"].str.contains("persist", na=False), "method"] = "persistence"
    return frame.dropna(subset=["horizon"])


def normalize_metrics_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "site_code": "station", "station": "station", "station_id": "station",
        "horizon": "horizon", "forecast_hour": "horizon",
        "xgboost_mae": "xgb_mae", "xgb_mae": "xgb_mae",
        "persistence_mae": "persistence_mae", "validation_rows": "validation_rows",
        "rows": "validation_rows",
    }
    rename = {}
    for col in frame.columns:
        key = re.sub(r"[^a-z0-9_]", "", str(col).strip().lower())
        if key in aliases:
            rename[col] = aliases[key]
    frame = frame.rename(columns=rename)
    if "station" in frame:
        frame["station"] = frame["station"].astype(str).str.strip().str.replace(r"^USGS-", "", regex=True)
        frame["station"] = frame["station"].str.replace(r"\.0$", "", regex=True)
        frame["station"] = frame["station"].apply(
            lambda value: value.zfill(8) if value.isdigit() and len(value) <= 8 else value
        )
    for col in ["horizon", "xgb_mae", "persistence_mae", "validation_rows"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


@st.cache_resource(show_spinner=False)
def load_models_and_hierarchies():
    models = {}
    hierarchies = {}
    metrics = {}
    problems = []
    for meta in VARIABLES.values():
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
            hierarchies[key] = normalize_hierarchy_columns(pd.read_csv(hierarchy_path, dtype=str))
        metrics_name = HIERARCHY_FILES[key].replace("HIERARCHY", "METHOD_METRICS")
        metrics_path = find_file_case_insensitive(folder, metrics_name)
        metrics[key] = (
            normalize_metrics_columns(pd.read_csv(metrics_path, dtype=str))
            if metrics_path is not None else pd.DataFrame()
        )
    return models, hierarchies, metrics, problems


def make_feature_row(history: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
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
    return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def get_station_method(
    hierarchy: pd.DataFrame, station: str, horizon: int,
    metrics: pd.DataFrame | None = None,
):
    station = str(station).strip().replace("USGS-", "")
    station = station.zfill(8) if station.isdigit() and len(station) <= 8 else station
    if not hierarchy.empty:
        match = hierarchy[(hierarchy["station"].astype(str) == station) & (hierarchy["horizon"] == horizon)]
        if not match.empty:
            method = str(match.iloc[0]["method"]).lower()
            if method in {"xgboost", "persistence"}:
                return method, "Station validation result"
    required = {"horizon", "xgb_mae", "persistence_mae"}
    if metrics is not None and not metrics.empty and required.issubset(metrics.columns):
        rows = metrics[metrics["horizon"] == horizon].dropna(
            subset=["xgb_mae", "persistence_mae"]
        )
        if not rows.empty:
            if "validation_rows" in rows and rows["validation_rows"].notna().any():
                weights = rows["validation_rows"].fillna(1).clip(lower=1).to_numpy(dtype=float)
            else:
                weights = np.ones(len(rows), dtype=float)
            xgb_mae = float(np.average(rows["xgb_mae"].to_numpy(dtype=float), weights=weights))
            persistence_mae = float(np.average(rows["persistence_mae"].to_numpy(dtype=float), weights=weights))
            if np.isfinite(xgb_mae) and np.isfinite(persistence_mae):
                method = "xgboost" if xgb_mae < persistence_mae else "persistence"
                return method, "Pooled validation MAE"
    return "xgboost", "No station metrics; use shared XGBoost model"


def forecast_all(history: pd.DataFrame, station: str, models: dict, hierarchies: dict, metrics: dict):
    complete_features = []
    for timestamp in history.index:
        try:
            features = make_feature_row(history, timestamp)
            if features[FEATURE_COLUMNS].notna().all(axis=None):
                complete_features.append(timestamp)
        except (KeyError, TypeError, ValueError):
            continue

    shared_observations = history[VALUE_COLUMNS].dropna(how="any")
    if shared_observations.empty:
        raise ValueError("A shared forecast origin requires recent observations for all four parameters.")
    shared_origin = shared_observations.index[-1]
    full_origin = complete_features[-1] if complete_features else None
    forecast_origin = full_origin if full_origin is not None else shared_origin
    forecasts = []

    for meta in VARIABLES.values():
        key = meta["key"]
        target_col = meta["column"]
        for horizon in range(1, 25):
            method, method_source = get_station_method(
                hierarchies.get(key, pd.DataFrame()), station, horizon,
                metrics.get(key, pd.DataFrame()),
            )
            value = None
            if method == "xgboost" and full_origin is not None:
                feature_row = make_feature_row(history, full_origin)
                model = models.get(key, {}).get(horizon)
                if model is not None:
                    booster_names = model.get_booster().feature_names
                    if booster_names and list(booster_names) != FEATURE_COLUMNS:
                        raise ValueError(f"{meta['prefix']}_{horizon}h feature schema does not match training order.")
                    value = float(model.predict(feature_row)[0])
                else:
                    method = "persistence"
                    method_source = "Horizon model unavailable"
            elif method == "xgboost":
                method = "unavailable"
                method_source = "Required input history incomplete"

            if value is None and method == "persistence":
                value = float(history.at[forecast_origin, target_col])
            if value is None:
                value = np.nan
            forecasts.append({
                "variable": key,
                "label": target_col,
                "horizon": horizon,
                "timestamp": forecast_origin + pd.Timedelta(hours=horizon),
                "prediction": value,
                "method": method,
                "method_source": method_source,
                "origin": forecast_origin,
            })

    if not forecasts:
        raise ValueError("No forecast can be produced because there are no shared observations.")
    result = pd.DataFrame(forecasts)
    result["salinity_flagged"] = False
    salinity_mask = result["variable"].eq("salinity") & result["prediction"].notna()
    result.loc[salinity_mask, "salinity_flagged"] = ~result.loc[salinity_mask, "prediction"].between(
        SALINITY_PLAUSIBLE_RANGE[0], SALINITY_PLAUSIBLE_RANGE[1], inclusive="both"
    )
    return result


st.title("Long Island Estuary Water Quality")

with st.sidebar:
    st.header("Station")
    station_choice = st.selectbox("USGS monitoring station", list(STATION_PRESETS.keys()), index=0)
    if STATION_PRESETS[station_choice] == "custom":
        station_id = st.text_input("USGS station ID", value="")
    else:
        station_id = STATION_PRESETS[station_choice]
    noaa_station = st.text_input(
        "Optional NOAA CO-OPS station ID",
        value="",
        help="Used only for missing temperature or salinity observations. Use a location known to represent this station.",
    )
    refresh = st.button("Refresh", use_container_width=True)

if refresh:
    st.cache_data.clear()

if not station_id.strip() or not re.fullmatch(r"\d{8}(\d{7})?", station_id.strip()):
    st.error("Enter a valid USGS station ID (8 or 15 digits).")
    st.stop()

models, hierarchies, validation_metrics, model_problems = load_models_and_hierarchies()
if model_problems:
    st.error("The model suite is incomplete. Fix these repository paths before forecasting:")
    for problem in model_problems:
        st.write(f"- {problem}")
    st.stop()

with st.spinner("Loading station observations..."):
    try:
        raw_history = fetch_usgs_history(station_id.strip(), 90)
        hourly = build_hourly_history(raw_history)
        hourly, provenance, source_notes, fallback_notes = apply_observation_hierarchy(
            hourly, noaa_station, 90
        )
    except Exception as exc:
        st.error(f"Could not retrieve or prepare station observations: {exc}")
        st.info("This station may not publish continuous water-quality measurements. Try another listed station or check its USGS station page.")
        st.stop()

st.subheader(f"Station {station_id.strip()}")
metric_cols = st.columns(4)
for i, (label, meta) in enumerate(VARIABLES.items()):
    col = meta["column"]
    valid = hourly[col].dropna()
    with metric_cols[i]:
        if valid.empty:
            st.metric(label, "Unavailable")
            st.caption("No observation returned")
        else:
            latest_time = valid.index[-1]
            age_hours = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() / 3600
            st.metric(label, f"{valid.iloc[-1]:.2f} {meta['unit']}")
            st.caption(f"{source_notes.get(col, 'Unknown source')} · {age_hours:.1f}h old")

try:
    forecast = forecast_all(hourly, station_id.strip(), models, hierarchies, validation_metrics)
except Exception as exc:
    st.warning(f"Forecast unavailable: {exc}")
    st.stop()

forecast_origin = forecast["origin"].iloc[0]
window_start = forecast_origin - pd.Timedelta(hours=24)
window_end = forecast_origin + pd.Timedelta(hours=24)
figure = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    subplot_titles=[f"{label} ({meta['unit']})" for label, meta in VARIABLES.items()],
)
colors = {"observed": "#087f8c", "forecast": "#102b46", "flagged": "#c62828"}

for row, (label, meta) in enumerate(VARIABLES.items(), start=1):
    col = meta["column"]
    observed = hourly.loc[(hourly.index >= window_start) & (hourly.index <= forecast_origin), col].dropna()
    subset = forecast[forecast["variable"] == meta["key"]].sort_values("horizon")
    if not observed.empty:
        figure.add_trace(go.Scatter(
            x=observed.index, y=observed.values, mode="lines+markers",
            name="Observed" if row == 1 else label + " observed",
            legendgroup="observed", showlegend=row == 1,
            line={"color": colors["observed"], "width": 2},
            marker={"size": 5}, connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>%{y:.3f}<extra>Observed</extra>",
        ), row=row, col=1)

    ordinary = subset[subset["prediction"].notna() & ~subset["salinity_flagged"]]
    if not ordinary.empty:
        figure.add_trace(go.Scatter(
            x=ordinary["timestamp"], y=ordinary["prediction"], mode="lines+markers",
            name="Forecast" if row == 1 else label + " forecast",
            legendgroup="forecast", showlegend=row == 1,
            line={"color": colors["forecast"], "width": 2, "dash": "dash"},
            marker={"size": 6}, connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>%{y:.3f}<extra>Forecast</extra>",
        ), row=row, col=1)

    flagged = subset[subset["salinity_flagged"]]
    if not flagged.empty:
        figure.add_trace(go.Scatter(
            x=flagged["timestamp"], y=flagged["prediction"], mode="markers",
            name="Salinity outside 0–45 PSU screen", legendgroup="flagged",
            showlegend=row == 4,
            marker={"size": 10, "symbol": "x", "color": colors["flagged"], "line": {"width": 2}},
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>%{y:.3f} PSU<br>Flagged: outside 0–45 PSU screen<extra></extra>",
        ), row=row, col=1)
    figure.update_yaxes(title_text=meta["unit"], row=row, col=1)

figure.add_vline(
    x=forecast_origin, line_width=2, line_dash="dot", line_color="#d18b24",
    annotation_text="NOW", annotation_position="top",
)
figure.update_xaxes(range=[window_start, window_end], title_text="Time (UTC)", row=4, col=1)
figure.update_layout(
    height=900, hovermode="x unified", plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff", font={"color": "#102b46"},
    legend={"orientation": "h", "y": 1.02, "x": 0},
    margin={"t": 90, "b": 35, "l": 65, "r": 20},
)
st.plotly_chart(figure, use_container_width=True)

flagged_forecasts = forecast[forecast["salinity_flagged"]]
if not flagged_forecasts.empty:
    st.warning(
        "Some salinity predictions fall outside the 0–45 PSU plausibility screen. "
        "They are marked with red crosses and should not be treated as reliable forecasts."
    )
    st.dataframe(
        flagged_forecasts[["horizon", "timestamp", "prediction", "method", "method_source"]]
        .rename(columns={"horizon": "Forecast hour", "timestamp": "Time (UTC)", "prediction": "Salinity (PSU)"}),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Data sources"):
    source_rows = []
    for label, meta in VARIABLES.items():
        col = meta["column"]
        valid = hourly[col].dropna()
        source_rows.append({
            "Parameter": label,
            "Latest source": source_notes.get(col, "Unavailable"),
            "Latest observation (UTC)": valid.index[-1].strftime("%Y-%m-%d %H:%M") if not valid.empty else "Unavailable",
            "Hourly observations": int(valid.size),
        })
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
    for note in fallback_notes.values():
        st.caption(note)
    st.caption("USGS NWIS observations are used directly. NOAA CO-OPS observations are used only when an optional station is supplied and fill missing temperature or salinity cells. Missing measurements remain missing; no spatial or temporal estimates are generated.")

st.caption("Forecasts use XGBoost or persistence according to the station validation hierarchy and pooled validation metrics. All times are UTC.")
