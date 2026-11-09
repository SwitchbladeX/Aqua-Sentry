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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { color-scheme: light; }
.stApp { background: #edf7f7; color: #123047; }
[data-testid="stHeader"] { background: rgba(237,247,247,.96); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b2940 0%,#10566a 100%); }
[data-testid="stSidebar"] * { color: #f2fbfb !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input { background: #f7ffff !important; color: #123047 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #123047 !important; }
.stButton > button,
.stDownloadButton > button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"],
button[kind="primary"], button[kind="secondary"] {
    background: #087e8b !important;
    color: #ffffff !important;
    border: 1px solid #087e8b !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
.stButton > button *, .stDownloadButton > button *,
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-secondary"] * { color: #ffffff !important; }
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
    background: #075b68 !important; border-color: #075b68 !important; color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton > button { background: #c9f1e8 !important; color: #10384b !important; border-color: #c9f1e8 !important; }
[data-testid="stSidebar"] .stButton > button * { color: #10384b !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: #ffffff !important; color: #10384b !important; }
[data-testid="stBaseButton-secondary"] p, [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
[data-testid="stBaseButton-secondary"] span, [data-testid="stBaseButton-primary"] span { color: #ffffff !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span { color: #10384b !important; }
[data-testid="stBaseButton-secondary"]:not(:disabled), [data-testid="stBaseButton-primary"]:not(:disabled) { opacity: 1 !important; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #cce5e5; padding: 14px; border-radius: 12px; }
[data-testid="stExpander"] { background: #ffffff; border: 1px solid #cce5e5; border-radius: 12px; }
[data-testid="stTabs"] button { color: #17465e !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #087e8b !important; border-bottom-color: #087e8b !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #476575 !important; }
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
                    elif len(candidates) == 1:
                        result.at[ts, col] = candidates[0][0]
                        provenance.at[ts, col] = "Spatial estimate (1 nearby USGS site)"
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
    frame["station"] = frame["station"].str.replace(r"\.0$", "", regex=True)
    frame["station"] = frame["station"].apply(lambda value: value.zfill(8) if value.isdigit() and len(value) <= 8 else value)
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
        "persistence_mae": "persistence_mae",
        "validation_rows": "validation_rows", "rows": "validation_rows",
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
        frame["station"] = frame["station"].apply(lambda value: value.zfill(8) if value.isdigit() and len(value) <= 8 else value)
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
        metrics_name = HIERARCHY_FILES[key].replace("HIERARCHY", "METHOD_METRICS")
        metrics_path = find_file_case_insensitive(folder, metrics_name)
        if metrics_path is None:
            metrics[key] = pd.DataFrame()
        else:
            metrics[key] = normalize_metrics_columns(pd.read_csv(metrics_path, dtype=str))
    return models, hierarchies, metrics, problems


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


def get_station_method(hierarchy: pd.DataFrame, station: str, horizon: int, metrics: pd.DataFrame | None = None):
    station = str(station).strip().replace("USGS-", "")
    station = station.zfill(8) if station.isdigit() and len(station) <= 8 else station
    if not hierarchy.empty:
        match = hierarchy[
            (hierarchy["station"].astype(str) == station)
            & (hierarchy["horizon"] == horizon)
        ]
        if not match.empty:
            method = str(match.iloc[0]["method"]).lower()
            if method in {"xgboost", "persistence"}:
                return method, "Station validation result"
    if metrics is not None and not metrics.empty and {"horizon", "xgb_mae", "persistence_mae"}.issubset(metrics.columns):
        rows = metrics[metrics["horizon"] == horizon].copy()
        rows = rows.dropna(subset=["xgb_mae", "persistence_mae"])
        if not rows.empty:
            if "validation_rows" in rows and rows["validation_rows"].notna().any():
                weights = rows["validation_rows"].fillna(1).clip(lower=1).to_numpy(dtype=float)
            else:
                weights = np.ones(len(rows), dtype=float)
            xgb_mae = float(np.average(rows["xgb_mae"].to_numpy(dtype=float), weights=weights))
            persistence_mae = float(np.average(rows["persistence_mae"].to_numpy(dtype=float), weights=weights))
            if np.isfinite(xgb_mae) and np.isfinite(persistence_mae):
                return ("xgboost", "Pooled validation MAE") if xgb_mae < persistence_mae else ("persistence", "Pooled validation MAE")
    return "xgboost", "No station metrics; use shared XGBoost model"


def forecast_all(history: pd.DataFrame, station: str, models: dict, hierarchies: dict, metrics: dict):
    complete_times = []
    for timestamp in history.index:
        try:
            features = make_feature_row(history, timestamp)
            if features[FEATURE_COLUMNS].notna().all(axis=None):
                complete_times.append(timestamp)
        except (KeyError, TypeError, ValueError):
            continue
    full_origin = complete_times[-1] if complete_times else None
    forecasts = []
    for label, meta in VARIABLES.items():
        key, target_col = meta["key"], meta["column"]
        target_values = history[target_col].dropna()
        if target_values.empty:
            continue
        for horizon in range(1, 25):
            method, method_source = get_station_method(
                hierarchies.get(key, pd.DataFrame()), station, horizon,
                metrics.get(key, pd.DataFrame()),
            )
            value = None
            origin = full_origin
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
                value = float(target_values.iloc[-1])
                origin = target_values.index.max()
            if value is None:
                value = np.nan
                origin = target_values.index.max()
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

models, hierarchies, validation_metrics, model_problems = load_models_and_hierarchies()
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

coverage = pd.DataFrame({
    "Parameter": [v["column"] for v in VARIABLES.values()],
    "Available hourly records": [int(hourly[v["column"]].notna().sum()) for v in VARIABLES.values()],
    "Coverage": [f"{hourly[v['column']].notna().mean() * 100:.1f}%" for v in VARIABLES.values()],
    "Latest observation (UTC)": [
        latest_by_variable[v["column"]].strftime("%Y-%m-%d %H:%M") if pd.notna(latest_by_variable[v["column"]]) else "Unavailable"
        for v in VARIABLES.values()
    ],
    "Latest source": [source_notes.get(v["column"], "Unavailable") for v in VARIABLES.values()],
})
with st.expander("Data sources", expanded=False):
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    for group_name in ["noaa", "spatial", "temporal"]:
        for note in hierarchy_notes.get(group_name, {}).values():
            st.caption(note)

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


st.subheader("24-hour forecast")
try:
    forecast = forecast_all(hourly, station_id.strip(), models, hierarchies, validation_metrics)
except Exception as exc:
    st.warning(f"Forecast is not available yet: {exc}")
    st.info("A forecast needs recent measurements and sufficient history. Missing measurements are not fabricated.")
    st.stop()

latest_origin = forecast["origin"].max()
st.caption(f"Forecasts use each variable's latest usable origin; latest origin in this run: {latest_origin.strftime('%Y-%m-%d %H:%M UTC')}")
forecast_tabs = st.tabs(["Dissolved oxygen", "Temperature", "pH", "Salinity"])
for tab, (label, meta) in zip(forecast_tabs, VARIABLES.items()):
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
        available = subset[subset["prediction"].notna()]
        if available.empty:
            st.info(f"{label} forecast is unavailable because required measurements are missing.")
            continue
        selected = st.select_slider(
            f"Inspect {label} forecast hour",
            options=available["horizon"].astype(int).tolist(),
            value=int(available["horizon"].iloc[0]), key=f"hour_{meta['key']}",
        )
        row = available.loc[available["horizon"] == selected].iloc[0]
        st.metric(f"{label} at +{selected}h", f"{row['prediction']:.3f} {meta['unit']}")

st.divider()
st.caption(
    "Data services: USGS Water Data / NWIS and, when explicitly configured, NOAA CO-OPS. "
    "All timestamps shown in this app are UTC unless otherwise labeled."
)
