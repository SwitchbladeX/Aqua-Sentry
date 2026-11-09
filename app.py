from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import StringIO

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
STATION_PRESETS = {
    "West Harbor at Oyster Bay, NY — 01303152 (Long Island)": "01303152",
    "Choptank River near Greensboro, MD — 01491000 (comparison site)": "01491000",
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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { background: #f7f9fc; }
    [data-testid="stSidebar"] { background: #102238; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div { color: #172033 !important; background: #ffffff !important; }
    div.stButton > button, div.stDownloadButton > button,
    button[kind="primary"], button[kind="secondary"] {
        background: #17465e !important; color: #ffffff !important;
        border: 1px solid #17465e !important; border-radius: 9px !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover,
    button[kind="primary"]:hover, button[kind="secondary"]:hover {
        background: #0f3448 !important; color: #ffffff !important; border-color: #0f3448 !important;
    }
    [data-testid="stSidebar"] div.stButton > button { background: #e6f1f5 !important; color: #102238 !important; border-color: #e6f1f5 !important; }
    [data-testid="stSidebar"] div.stButton > button:hover { background: #ffffff !important; color: #102238 !important; }
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
                                                                             
                                                                            
        "parameterCd": "00010,00300,00400,00480,90860",
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
    code_to_column.update({"90860": "salinity_PSU", "00480": "salinity_PSU"})
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


                                                                       
                                                                          
                                                                       
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usgs_site_metadata(site_id: str) -> dict:
    """Get coordinates and HUC for a USGS site from the NWIS site service."""
    response = requests.get(
        "https://waterservices.usgs.gov/nwis/site/",
        params={"format": "rdb", "sites": site_id, "siteOutput": "expanded"},
        timeout=30,
        headers={"User-Agent": "LongIslandEstuaryWaterQualityPortal/1.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), sep="\t", comment="#", dtype=str)
    if "site_no" not in frame.columns:
        raise ValueError("USGS site metadata did not include site_no.")
    frame = frame[frame["site_no"].astype(str).str.fullmatch(r"\d{8}(\d{7})?")]
    if frame.empty:
        raise ValueError(f"USGS could not find station {site_id}.")
    row = frame.iloc[0]
    return {
        "lat": pd.to_numeric(row.get("dec_lat_va"), errors="coerce"),
        "lon": pd.to_numeric(row.get("dec_long_va"), errors="coerce"),
        "huc": str(row.get("huc_cd", ""))[:8],
        "name": str(row.get("station_nm", site_id)),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def discover_nearby_usgs_sites(site_id: str, radius_km: float = 20.0) -> list:
    """Discover nearby USGS continuous water-quality sites; do not guess coordinates."""
    meta = fetch_usgs_site_metadata(site_id)
    lat, lon = float(meta["lat"]), float(meta["lon"])
    lat_span = radius_km / 111.0
    lon_span = radius_km / max(20.0, 111.0 * np.cos(np.deg2rad(lat)))
    bbox = f"{lon-lon_span},{lat-lat_span},{lon+lon_span},{lat+lat_span}"
    response = requests.get(
        "https://waterservices.usgs.gov/nwis/site/",
        params={
            "format": "rdb", "bBox": bbox, "siteStatus": "all",
            "hasDataTypeCd": "iv", "parameterCd": "00010,00300,00400,00480,90860",
            "siteOutput": "expanded",
        },
        timeout=35,
        headers={"User-Agent": "LongIslandEstuaryWaterQualityPortal/1.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), sep="\t", comment="#", dtype=str)
    needed = {"site_no", "dec_lat_va", "dec_long_va"}
    if not needed.issubset(frame.columns):
        return []
    frame = frame[frame["site_no"].astype(str).str.fullmatch(r"\d{8}(\d{7})?")]
    frame["lat"] = pd.to_numeric(frame["dec_lat_va"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["dec_long_va"], errors="coerce")
    frame = frame.dropna(subset=["lat", "lon"])
    if meta["huc"]:
        frame = frame[frame.get("huc_cd", pd.Series(index=frame.index, dtype=str)).astype(str).str[:8] == meta["huc"]]
    frame = frame[frame["site_no"].astype(str) != site_id]
    if frame.empty:
        return []
    def distance_km(row):
        p1, p2 = np.deg2rad(lat), np.deg2rad(float(row["lat"]))
        dp = np.deg2rad(float(row["lat"]) - lat)
        dl = np.deg2rad(float(row["lon"]) - lon)
        a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))
    frame["distance_km"] = frame.apply(distance_km, axis=1)
    frame = frame[frame["distance_km"] <= radius_km].sort_values("distance_km").head(8)
    return [
        {"site": str(r["site_no"]), "distance_km": float(r["distance_km"]),
         "name": str(r.get("station_nm", r["site_no"]))}
        for _, r in frame.iterrows()
    ]


def apply_observation_hierarchy(
    hourly: pd.DataFrame, station: str, days: int, noaa_station: str,
    sensor_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Fill only missing cells in a disclosed order; never extrapolate or invent values."""
    result = hourly.copy()
    sensor_hourly = None
    if sensor_data is not None and not sensor_data.empty:
        sensor_hourly = sensor_data.resample("1h").median(numeric_only=True)
        result = result.reindex(result.index.union(sensor_hourly.index).sort_values())
    provenance = pd.DataFrame(index=result.index, columns=VALUE_COLUMNS, dtype=object)
    for col in VALUE_COLUMNS:
        if col in result:
            provenance.loc[result[col].notna(), col] = "USGS NWIS"

                                                                                       
    if sensor_hourly is not None:
        sensor_hourly = sensor_hourly.reindex(result.index)
        for col in VALUE_COLUMNS:
            if col in sensor_hourly:
                mask = result[col].isna() & sensor_hourly[col].notna()
                result.loc[mask, col] = sensor_hourly.loc[mask, col]
                provenance.loc[mask, col] = "User sensor CSV"

                                                                                                
    noaa_notes = {}
    if noaa_station.strip():
        for col, product in [("temp_C", "water_temperature"), ("salinity_PSU", "salinity")]:
            if result[col].isna().any():
                try:
                    series = fetch_noaa_product(noaa_station.strip(), product, days)
                    values = series.resample("1h").median().reindex(result.index)
                    mask = result[col].isna() & values.notna()
                    result.loc[mask, col] = values.loc[mask]
                    provenance.loc[mask, col] = f"NOAA CO-OPS ({product})"
                    noaa_notes[col] = f"NOAA CO-OPS filled {int(mask.sum())} hourly gaps"
                except Exception as exc:
                    noaa_notes[col] = f"NOAA fallback unavailable: {exc}"

                                                                                 
                                                                         
    spatial_notes = {}
    if result.isna().any(axis=None):
        try:
            neighbors = discover_nearby_usgs_sites(station, radius_km=20.0)
            neighbor_frames = []
            for neighbor in neighbors[:8]:
                try:
                    nh = build_hourly_history(fetch_usgs_history(neighbor["site"], days))
                    nh.attrs["site"] = neighbor["site"]
                    nh.attrs["distance_km"] = neighbor["distance_km"]
                    neighbor_frames.append((neighbor, nh.reindex(result.index)))
                except Exception:
                    continue
            for col in VALUE_COLUMNS:
                mask = result[col].isna()
                filled = 0
                for ts in result.index[mask]:
                    candidates = []
                    for info, frame in neighbor_frames:
                        value = frame.at[ts, col] if ts in frame.index else np.nan
                        if pd.notna(value) and info["distance_km"] > 0:
                            candidates.append((float(value), float(info["distance_km"])))
                    if len(candidates) >= 2:
                        weights = np.array([1.0 / (d*d) for _, d in candidates])
                        vals = np.array([v for v, _ in candidates])
                        result.at[ts, col] = float(np.average(vals, weights=weights))
                        provenance.at[ts, col] = "Spatial IDW (2+ nearby USGS sites)"
                        filled += 1
                if filled:
                    spatial_notes[col] = f"Spatial interpolation filled {filled} hours using 2+ nearby USGS sites"
        except Exception as exc:
            spatial_notes["discovery"] = f"Spatial fallback unavailable: {exc}"

                                                                                       
    temporal_notes = {}
    for col in VALUE_COLUMNS:
        series = result[col].copy()
        missing = series.isna()
        run_ids = (missing != missing.shift(fill_value=False)).cumsum()
        filled_mask = pd.Series(False, index=series.index)
        interpolated = series.interpolate(method="time")
        for _, run_index in series[missing].groupby(run_ids[missing]).groups.items():
            run_index = list(run_index)
            if len(run_index) > 2:
                continue
            first_pos = series.index.get_loc(run_index[0])
            last_pos = series.index.get_loc(run_index[-1])
            if first_pos == 0 or last_pos >= len(series.index) - 1:
                continue
            if pd.isna(series.iloc[first_pos - 1]) or pd.isna(series.iloc[last_pos + 1]):
                continue
            result.loc[run_index, col] = interpolated.loc[run_index]
            filled_mask.loc[run_index] = True
        if filled_mask.any():
            provenance.loc[filled_mask, col] = "Temporal interpolation (complete interior gap <=2h)"
            temporal_notes[col] = f"Temporal interpolation filled {int(filled_mask.sum())} hours"

    source_notes = {}
    confidence = {}
    for col in VALUE_COLUMNS:
        valid = result[col].dropna()
        if valid.empty:
            source_notes[col] = "Unavailable"
            confidence[col] = "Unavailable"
            continue
        latest_ts = valid.index.max()
        latest_source = provenance.at[latest_ts, col] if latest_ts in provenance.index else "Unknown"
        source_notes[col] = str(latest_source or "Unknown")
        age_h = (pd.Timestamp.now(tz="UTC") - latest_ts).total_seconds() / 3600
        if "Temporal" in source_notes[col]:
            level = "Low — temporally interpolated"
        elif "Spatial" in source_notes[col]:
            level = "Moderate — spatial estimate"
        elif "NOAA" in source_notes[col]:
            level = "Moderate — alternate agency source"
        elif age_h <= 6:
            level = "High — recent direct observation"
        elif age_h <= 24:
            level = "Moderate — direct observation"
        else:
            level = "Low — stale observation"
        confidence[col] = level
    return result, provenance, source_notes, {"confidence": confidence, "noaa": noaa_notes,
                                               "spatial": spatial_notes, "temporal": temporal_notes}


                                                                       
                             
                                                                       
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
    frame["station"] = frame["station"].astype(str).str.strip().str.replace(r"^USGS-", "", regex=True)
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
            hierarchies[key] = normalize_hierarchy_columns(pd.read_csv(hierarchy_path, dtype=str))
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


def forecast_all(history: pd.DataFrame, station: str, models: dict, hierarchies: dict):
    """Forecast each available target independently; use XGBoost only with full features."""
    forecasts = []
    for label, meta in VARIABLES.items():
        key, target_col = meta["key"], meta["column"]
        target_values = history[target_col].dropna()
        if target_values.empty:
            continue
        complete_times = []
        for timestamp in history.index:
            try:
                features = make_feature_row(history, timestamp)
                if features[FEATURE_COLUMNS].notna().all(axis=None):
                    complete_times.append(timestamp)
            except (KeyError, TypeError, ValueError):
                continue
        full_origin = complete_times[-1] if complete_times else None
        target_origin = target_values.index.max()
        method_rows = []
        for horizon in range(1, 25):
            selected_method = get_station_method(hierarchies.get(key, pd.DataFrame()), station, horizon)
            method_source = "Station-specific validation hierarchy" if selected_method else "No station+horizon hierarchy row"
            method = selected_method or "persistence"
            value = None
            origin = target_origin
            if method == "xgboost" and full_origin is not None:
                feature_row = make_feature_row(history, full_origin)
                model = models.get(key, {}).get(horizon)
                if model is not None:
                    booster_names = model.get_booster().feature_names
                    if booster_names and list(booster_names) != FEATURE_COLUMNS:
                        raise ValueError(f"{meta['prefix']}_{horizon}h feature schema does not match training order.")
                    value = float(model.predict(feature_row)[0])
                    origin = full_origin
                else:
                    method = "persistence"
                    method_source = "Missing horizon model; safe persistence fallback"
            elif method == "xgboost":
                method = "persistence"
                method_source = "XGBoost inputs incomplete; safe per-variable persistence fallback"
            if value is None:
                value = float(target_values.iloc[-1])
                if selected_method == "persistence":
                    method_source = "Station-specific validation hierarchy"
                elif not selected_method:
                    method_source = "Fallback: no station+horizon hierarchy row"
            forecasts.append({
                "variable": key, "label": target_col, "horizon": horizon,
                "timestamp": origin + pd.Timedelta(hours=horizon),
                "prediction": value, "method": method, "method_source": method_source,
                "origin": origin,
            })
    if not forecasts:
        raise ValueError("No forecast can be produced because none of the four variables has a usable observation.")
    return pd.DataFrame(forecasts)


                                                                       
                
                                                                       
st.title("Long Island Estuary Water Quality Portal")
st.caption("Station observations, transparent data provenance, and 24-hour parameter forecasts")

with st.sidebar:
    st.header("Station setup")
    station_choice = st.selectbox("Default stations", list(STATION_PRESETS.keys()), index=0)
    if STATION_PRESETS[station_choice] == "custom":
        station_id = st.text_input("Custom USGS station ID", value="", help="Enter an 8- or 15-digit USGS monitoring-location ID.")
    else:
        station_id = STATION_PRESETS[station_choice]
        st.caption(f"Selected USGS station: {station_id}")
    days = st.selectbox("Historical context", [30, 60, 90], index=2)
    st.markdown("---")
    st.subheader("Optional compatible NOAA station")
    noaa_station = st.text_input(
        "NOAA CO-OPS station ID",
        value="",
        help="Optional. Fills missing hourly temperature/salinity values only. Confirm it represents the same water body/location.",
    )
    st.markdown("---")
    st.subheader("Optional sensor data")
    sensor_upload = st.file_uploader("Upload sensor CSV", type=["csv"])
    st.caption("CSV needs timestamps and recognized columns such as DO_mgL, temp_C, pH, salinity_PSU.")
    refresh = st.button("Refresh observations", use_container_width=True)

if refresh:
    st.cache_data.clear()

if not station_id.strip() or not re.fullmatch(r"\d{8}(\d{7})?", station_id.strip()):
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
        hourly = build_hourly_history(raw_history)
        sensor_data = load_uploaded_sensor_csv(sensor_upload) if sensor_upload is not None else None
        hourly, provenance, source_notes, hierarchy_notes = apply_observation_hierarchy(
            hourly, station_id.strip(), days, noaa_station, sensor_data
        )
    except Exception as exc:
        st.error(f"Could not retrieve/prepare station observations: {exc}")
        st.info("The station may be valid but may not publish continuous water-quality measurements. Try West Harbor at Oyster Bay (01303152), or check the USGS station page.")
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
            st.caption(hierarchy_notes["confidence"].get(col, "Confidence unavailable"))

def station_familiarity(key):
    table = hierarchies.get(key, pd.DataFrame())
    if table.empty or "station" not in table:
        return "Not represented in validation hierarchy"
    return "Represented in validation hierarchy" if (table["station"].astype(str) == station_id.strip()).any() else "Not represented in validation hierarchy"

coverage = pd.DataFrame({
    "Parameter": [v["column"] for v in VARIABLES.values()],
    "Available hourly records": [int(hourly[v["column"]].notna().sum()) for v in VARIABLES.values()],
    "Coverage": [f"{hourly[v['column']].notna().mean() * 100:.1f}%" for v in VARIABLES.values()],
    "Latest observation (UTC)": [
        latest_by_variable[v["column"]].strftime("%Y-%m-%d %H:%M") if pd.notna(latest_by_variable[v["column"]]) else "Unavailable"
        for v in VARIABLES.values()
    ],
    "Latest source": [source_notes.get(v["column"], "Unavailable") for v in VARIABLES.values()],
    "Confidence": [hierarchy_notes["confidence"].get(v["column"], "Unavailable") for v in VARIABLES.values()],
    "Station familiarity": [station_familiarity(v["key"]) for v in VARIABLES.values()],
})
with st.expander("Data coverage and provenance", expanded=False):
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    for group_name in ["noaa", "spatial", "temporal"]:
        for note in hierarchy_notes.get(group_name, {}).values():
            st.caption(note)
    st.write(
        "Source order: direct USGS NWIS observations; local uploaded sensor readings and NOAA CO-OPS as alternate sources; "
        "spatial inverse-distance interpolation from at least two nearby USGS sites in the same HUC8; then temporal interpolation "
        "only across interior gaps of at most two hours. No extrapolation or conductivity-to-salinity conversion is performed. "
        "Confidence describes the latest measurement source and freshness. Station familiarity is separate: it indicates whether "
        "the station appears in that variable's validation hierarchy, not whether the measurement is accurate."
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
st.caption("Separate parameter units share the plot. Long gaps remain visible; interpolated hours are identified by source in the observations table.")

with st.expander("Hourly observations and per-hour provenance"):
    table_display = hourly.tail(24 * 14).copy()
    source_display = provenance.tail(24 * 14).copy()
    for col in VALUE_COLUMNS:
        table_display[f"{col} source"] = source_display[col]
    st.dataframe(table_display, use_container_width=True)

                                                  
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
    forecast = forecast_all(hourly, station_id.strip(), models, hierarchies)
except Exception as exc:
    st.warning(f"Forecast is not available yet: {exc}")
    st.info("A variable can be forecast with persistence when its own latest observation is available. XGBoost is used only when all 20 required features are present.")
    st.stop()

latest_origin = forecast["origin"].max()
st.caption(f"Forecasts use each variable's latest usable origin; latest origin in this run: {latest_origin.strftime('%Y-%m-%d %H:%M UTC')}")
forecast_tabs = st.tabs(["Dissolved oxygen", "Temperature", "pH", "Salinity", "Forecast methods"])
for tab, (label, meta) in zip(forecast_tabs[:4], VARIABLES.items()):
    with tab:
        subset = forecast[forecast["variable"] == meta["key"]].sort_values("horizon")
        if subset.empty:
            st.info(f"No {label} forecast: no usable observation is available for this variable.")
            continue
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
        "station+horizon row exists. If no row exists, or an XGBoost row exists but the 20 input features are incomplete, "
        "the app uses clearly labeled per-variable persistence instead of blocking every forecast."
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
    1. **USGS NWIS** direct station measurements, including salinity parameter 90860 where available (and 00480 where used).
    2. **Local sensor CSV** and explicitly configured **NOAA CO-OPS** data fill only missing hours; NOAA supports temperature/salinity here.
    3. **Spatial interpolation** uses inverse-distance weighting from at least two nearby USGS continuous-water-quality stations within 20 km and the same HUC8, at the same hourly timestamp.
    4. **Temporal interpolation** fills only interior gaps of at most two consecutive hours. It never extrapolates beyond observed data.
    5. Remaining gaps stay missing. Each displayed parameter reports its latest source and a separate confidence label.

    **Prediction-method order**
    - For each target variable and forecast hour, look up the station and horizon in that variable's validation hierarchy CSV.
    - If the hierarchy selected XGBoost, use that exact horizon-specific JSON model.
    - If it selected persistence, carry forward the latest observed value.
    - If no station+horizon row exists, use a clearly labeled persistence fallback.

    **Three-month station context**
    - The selected 30/60/90-day record supplies current values, lag features, coverage, and descriptive trends. Spatial and temporal estimates are explicitly marked and lower confidence.
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
        "- Spatial interpolation is conservative and may be unavailable when fewer than two compatible nearby stations report the same hour. NERRS CDMO access requires authorization.\n"
        "- Forecast quality depends on the model's training domain and the completeness/freshness of the station record."
    )

st.divider()
st.caption(
    "Data services: USGS Water Data / NWIS and, when explicitly configured, NOAA CO-OPS. "
    "All timestamps shown in this app are UTC unless otherwise labeled."
)
