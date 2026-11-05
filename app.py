import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Long Island Estuary Water Quality Portal", layout="wide")

@st.cache_resource
def load_all_models():
    # Load all trained regression and classification XGBoost model artifacts
    models = {}
    model_files = {
        "do": "xgb_do_regression_weather_model(FINAL) (1).json",
        "temp": "xgb_temp_C_regression_weather_model(FINAL) (1).json",
        "pH": "xgb_pH_regression_weather_model(FINAL) (1).json",
        "salinity": "xgb_salinity_PSU_regression_weather_model(FINAL) (1).json",
        "hypoxia": "xgb_hypoxia_weather_model(FINAL).json"
    }
    
    for key, filename in model_files.items():
        if key == "hypoxia":
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
            
        try:
            model.load_model(filename)
            models[key] = model
        except Exception as e:
            st.error(f"Failed to load model file {filename}: {e}")
            raise e
            
    return models

def fetch_usgs_station_history(station_id, days=3):
    # Fetch multi-day historical telemetry from USGS Instantaneous Values API
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={station_id}&parameterCd=00010,00300,00400,00095&period=P{days}D"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"USGS API request failed with status code {response.status_code}")
    
    data = response.json()
    time_series = data.get("value", {}).get("timeSeries", [])
    if not time_series:
        raise ValueError(f"No time series data returned for station {station_id}")
        
    dfs = []
    param_map = {
        "00010": "temp_C",
        "00300": "DO_mgL",
        "00400": "pH",
        "00095": "salinity_PSU"
    }
    
    for ts in time_series:
        var_code = ts["variable"]["variableCode"][0]["value"]
        values = ts.get("values", [])[0].get("value", [])
        if values:
            temp_df = pd.DataFrame(values)[["dateTime", "value"]]
            temp_df["dateTime"] = pd.to_datetime(temp_df["dateTime"])
            temp_df["value"] = temp_df["value"].astype(float)
            
            col_name = param_map.get(var_code, var_code)
            temp_df = temp_df.rename(columns={"value": col_name})
            dfs.append(temp_df.set_index("dateTime"))
            
    if not dfs:
        raise ValueError(f"Failed to parse time series parameters for station {station_id}")
        
    combined_df = pd.concat(dfs, axis=1).sort_index().ffill().bfill().dropna()
    return combined_df

def align_and_predict(model, input_df):
    # Align DataFrame columns with model feature specifications for regression
    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names:
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_names]
    return model.predict(input_df)

def align_and_predict_proba(model, input_df):
    # Align DataFrame columns with model feature specifications for classification
    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names:
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_names]
    return model.predict_proba(input_df)

def generate_forecast_features_from_history(history_df, hours=24, site_cat_id=0):
    # Extract historical derivatives and rolling statistics from real USGS telemetry
    last_row = history_df.iloc[-1]
    base_temp = last_row["temp_C"]
    base_do = last_row["DO_mgL"]
    base_pH = last_row["pH"]
    base_sal = last_row["salinity_PSU"]
    
    # Calculate exact rate of change metrics from 15-minute historical sampling intervals
    dDO_1h = history_df["DO_mgL"].diff(4).iloc[-1] if len(history_df) >= 4 else 0.0
    dDO_3h = history_df["DO_mgL"].diff(12).iloc[-1] if len(history_df) >= 12 else 0.0
    dDO_6h = history_df["DO_mgL"].diff(24).iloc[-1] if len(history_df) >= 24 else 0.0
    dDO_12h = history_df["DO_mgL"].diff(48).iloc[-1] if len(history_df) >= 48 else 0.0
    dDO_24h = history_df["DO_mgL"].diff(96).iloc[-1] if len(history_df) >= 96 else 0.0
    
    dpH_3h = history_df["pH"].diff(12).iloc[-1] if len(history_df) >= 12 else 0.0
    dpH_6h = history_df["pH"].diff(24).iloc[-1] if len(history_df) >= 24 else 0.0
    
    dDO_accel = dDO_3h - (dDO_6h / 2.0)
    
    # Calculate actual rolling volatility from historical telemetry window
    do_std_12h = history_df["DO_mgL"].rolling(48, min_periods=1).std().iloc[-1]
    do_std_24h = history_df["DO_mgL"].rolling(96, min_periods=1).std().iloc[-1]
    ph_std_24h = history_df["pH"].rolling(96, min_periods=1).std().iloc[-1]
    
    future_rows = []
    current_time = datetime.datetime.now()
    
    for i in range(hours):
        step_time = current_time + datetime.timedelta(hours=i)
        hour_of_day = step_time.hour
        day_of_year = step_time.timetuple().tm_yday
        
        # Diurnal temperature cycle approximation
        temp_est = base_temp + 1.2 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
        
        # Exact thermodynamic equations matching model training script
        do_solubility = 14.652 - (0.41022 * temp_est) + (0.007991 * (temp_est**2)) - (0.000077774 * (temp_est**3))
        do_deficit = max(0.0, do_solubility - base_do)
        do_sat_ratio = base_do / (do_solubility + 1e-5)
        do_pH_ratio = base_do / (base_pH + 1e-5)
        do_temp_ratio = base_do / (temp_est + 1e-5)
        
        row = {
            'temp_C': temp_est,
            'DO_mgL': base_do,
            'pH': base_pH,
            'salinity_PSU': base_sal,
            'dDO_dt_1h': 0.0 if pd.isna(dDO_1h) else dDO_1h,
            'dDO_dt_3h': 0.0 if pd.isna(dDO_3h) else dDO_3h,
            'dDO_dt_6h': 0.0 if pd.isna(dDO_6h) else dDO_6h,
            'dDO_dt_12h': 0.0 if pd.isna(dDO_12h) else dDO_12h,
            'dDO_dt_24h': 0.0 if pd.isna(dDO_24h) else dDO_24h,
            'dpH_dt_3h': 0.0 if pd.isna(dpH_3h) else dpH_3h,
            'dpH_dt_6h': 0.0 if pd.isna(dpH_6h) else dpH_6h,
            'dDO_acceleration': 0.0 if pd.isna(dDO_accel) else dDO_accel,
            'DO_pH_ratio': do_pH_ratio,
            'DO_temp_ratio': do_temp_ratio,
            'DO_solubility_sat': do_solubility,
            'DO_saturation_ratio': do_sat_ratio,
            'DO_deficit': do_deficit,
            'DO_std_12h': 0.2 if pd.isna(do_std_12h) else do_std_12h,
            'DO_std_24h': 0.3 if pd.isna(do_std_24h) else do_std_24h,
            'pH_std_24h': 0.1 if pd.isna(ph_std_24h) else ph_std_24h,
            'sin_hour': np.sin(2 * np.pi * hour_of_day / 24.0),
            'cos_hour': np.cos(2 * np.pi * hour_of_day / 24.0),
            'sin_day': np.sin(2 * np.pi * day_of_year / 365.25),
            'cos_day': np.cos(2 * np.pi * day_of_year / 365.25),
            'site_code_cat': site_cat_id
        }
        
        future_rows.append(row)
        
    return pd.DataFrame(future_rows)

try:
    models = load_all_models()
except Exception as e:
    st.stop()

st.title("Long Island Estuary Water Quality Portal")

station_candidates = {
    "Station 01304500 (Western Long Island Sound - Pelham)": "01304500",
    "Station 01304200 (Western Long Island Sound - Execution Rocks)": "01304200",
    "Station 01193050 (Connecticut River Estuary)": "01193050",
    "Station 01311143 (Jamaica Bay Estuary)": "01311143",
    "Station 01302020 (Hudson River Estuary)": "01302020",
    "Station 01305500 (Housatonic River Basin)": "01305500",
    "Station 01304650 (East River Estuary Node)": "01304650",
    "Station 01302250 (Upper New York Harbor)": "01302250"
}

valid_stations = {}
for name, sid in station_candidates.items():
    try:
        fetch_usgs_station_history(sid, days=1)
        valid_stations[name] = sid
    except Exception:
        pass

if not valid_stations:
    st.error("No active stations found reporting all required parameters.")
    st.stop()

selected_station_name = st.sidebar.selectbox("Select Validated Station", list(valid_stations.keys()))
selected_station_id = valid_stations[selected_station_name]

try:
    station_history_df = fetch_usgs_station_history(selected_station_id, days=3)
except Exception as e:
    st.error(f"Failed to fetch USGS historical data: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Admin Testing Portal")
admin_pass = st.sidebar.text_input("Enter Admin Passcode", type="password")

show_raw_prob = False

if admin_pass == "admin123":
    st.sidebar.success("Admin Mode Active")
    sim_mode = st.sidebar.checkbox("Enable Simulation Mode", value=False)
    show_raw_prob = st.sidebar.checkbox("Show Raw Floating-Point Probability", value=True)
    if sim_mode:
        last_idx = station_history_df.index[-1]
        station_history_df.loc[last_idx, "DO_mgL"] = st.sidebar.slider("Simulated DO (mg/L)", 1.0, 12.0, float(station_history_df["DO_mgL"].iloc[-1]))
        station_history_df.loc[last_idx, "temp_C"] = st.sidebar.slider("Simulated Temp (°C)", 5.0, 35.0, float(station_history_df["temp_C"].iloc[-1]))
        station_history_df.loc[last_idx, "pH"] = st.sidebar.slider("Simulated pH", 5.0, 10.0, float(station_history_df["pH"].iloc[-1]))
        station_history_df.loc[last_idx, "salinity_PSU"] = st.sidebar.slider("Simulated Salinity (PSU)", 0.0, 40.0, float(station_history_df["salinity_PSU"].iloc[-1]))

forecast_df = generate_forecast_features_from_history(station_history_df, hours=24)

pred_temps = align_and_predict(models["temp"], forecast_df.copy())
pred_dos = align_and_predict(models["do"], forecast_df.copy())
pred_phs = align_and_predict(models["pH"], forecast_df.copy())
pred_salts = align_and_predict(models["salinity"], forecast_df.copy())

forecast_df_hypoxia = forecast_df.copy()
forecast_df_hypoxia["temp_C"] = pred_temps
forecast_df_hypoxia["DO_mgL"] = pred_dos
forecast_df_hypoxia["pH"] = pred_phs
forecast_df_hypoxia["salinity_PSU"] = pred_salts
forecast_df_hypoxia["DO_solubility_sat"] = 14.652 - (0.41022 * pred_temps) + (0.007991 * (pred_temps**2)) - (0.000077774 * (pred_temps**3))
forecast_df_hypoxia["DO_deficit"] = np.maximum(0.0, forecast_df_hypoxia["DO_solubility_sat"] - pred_dos)
forecast_df_hypoxia["DO_saturation_ratio"] = pred_dos / (forecast_df_hypoxia["DO_solubility_sat"] + 1e-5)
forecast_df_hypoxia["DO_pH_ratio"] = pred_dos / (pred_phs + 1e-5)
forecast_df_hypoxia["DO_temp_ratio"] = pred_dos / (pred_temps + 1e-5)

if hasattr(models["hypoxia"], "predict_proba"):
    raw_hypoxia_probs = align_and_predict_proba(models["hypoxia"], forecast_df_hypoxia)[:, 1]
else:
    raw_hypoxia_probs = align_and_predict(models["hypoxia"], forecast_df_hypoxia)

hypoxia_percentages = raw_hypoxia_probs * 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Time Control")
selected_hour = st.sidebar.slider("Advance Forecast Horizon (Hours)", min_value=0, max_value=23, value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Selected Hour Telemetry (+{}h)".format(selected_hour))
st.sidebar.metric(label="Forecasted DO", value=f"{pred_dos[selected_hour]:.2f} mg/L")
st.sidebar.metric(label="Forecasted Temperature", value=f"{pred_temps[selected_hour]:.2f} °C")
st.sidebar.metric(label="Forecasted pH", value=f"{pred_phs[selected_hour]:.2f}")
st.sidebar.metric(label="Forecasted Salinity", value=f"{pred_salts[selected_hour]:.2f} PSU")
st.sidebar.metric(label="Hypoxia Crash Probability", value=f"{hypoxia_percentages[selected_hour]:.1f}%")

if show_raw_prob:
    st.sidebar.metric(label="Raw Model Probability", value=f"{raw_hypoxia_probs[selected_hour]:.6f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dissolved Oxygen", "Temperature", "pH", "Salinity", "Hypoxia Probability"])

forecast_hours = [f"+{i}h" for i in range(24)]

with tab1:
    fig = px.line(x=forecast_hours, y=pred_dos, labels={"x": "Forecast Horizon", "y": "Dissolved Oxygen (mg/L)"}, title="Dissolved Oxygen 24-Hour Trajectory")
    fig.add_scatter(x=[forecast_hours[selected_hour]], y=[pred_dos[selected_hour]], mode='markers', marker=dict(size=14, color='red'), name='Selected Hour')
    fig.update_layout(yaxis=dict(autorange=True), xaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(x=forecast_hours, y=pred_temps, labels={"x": "Forecast Horizon", "y": "Temperature (°C)"}, title="Water Temperature 24-Hour Trajectory")
    fig.add_scatter(x=[forecast_hours[selected_hour]], y=[pred_temps[selected_hour]], mode='markers', marker=dict(size=14, color='red'), name='Selected Hour')
    fig.update_layout(yaxis=dict(autorange=True), xaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.line(x=forecast_hours, y=pred_phs, labels={"x": "Forecast Horizon", "y": "pH"}, title="pH 24-Hour Trajectory")
    fig.add_scatter(x=[forecast_hours[selected_hour]], y=[pred_phs[selected_hour]], mode='markers', marker=dict(size=14, color='red'), name='Selected Hour')
    fig.update_layout(yaxis=dict(autorange=True), xaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.line(x=forecast_hours, y=pred_salts, labels={"x": "Forecast Horizon", "y": "Salinity (PSU)"}, title="Salinity 24-Hour Trajectory")
    fig.add_scatter(x=[forecast_hours[selected_hour]], y=[pred_salts[selected_hour]], mode='markers', marker=dict(size=14, color='red'), name='Selected Hour')
    fig.update_layout(yaxis=dict(autorange=True), xaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_hours, y=hypoxia_percentages, mode='lines+markers', name='Crash Probability (%)'))
    fig.add_scatter(x=[forecast_hours[selected_hour]], y=[hypoxia_percentages[selected_hour]], mode='markers', marker=dict(size=14, color='red'), name='Selected Hour')
    fig.add_hline(y=30.0, line_dash="dash", line_color="red", annotation_text="Crash Threshold (30%)", annotation_position="top right")
    fig.update_layout(xaxis_title="Forecast Horizon", yaxis_title="Probability (%)", yaxis=dict(range=[0, 105], autorange=False), title="Hypoxia Crash Probability Percentage")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Estuary Parameter Guidelines & Oyster Health Reference")
st.markdown("""
* **Dissolved Oxygen (DO):** 
  * **Safe / Fine:** Above 5.0 mg/L. Oysters breathe comfortably and maintain metabolic activity.
  * **Hazardous / Bad:** Below 3.0 mg/L. Induces severe hypoxia, metabolic stress, and potential mortality if prolonged.
* **Water Temperature:** 
  * **Safe / Fine:** 15°C to 25°C. Optimal feeding and growth range for *Crassostrea virginica*.
  * **Hazardous / Bad:** Above 28°C. Drastically increases oxygen consumption rates while lowering dissolved oxygen solubility in water.
* **pH Levels:**
  * **Safe / Fine:** 7.5 to 8.5. Supports normal shell calcification and larval development.
  * **Hazardous / Bad:** Below 7.3. Acidic conditions weaken oyster shells and hinder growth.
* **Salinity:**
  * **Safe / Fine:** 15 to 30 PSU. Standard estuarine brackish comfort zone.
  * **Hazardous / Bad:** Sudden drops below 10 PSU due to heavy freshwater runoff can cause osmotic shock.
""")
