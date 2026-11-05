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

def fetch_usgs_station_data(station_id):
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={station_id}&parameterCd=00010,00300,00400,00095"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"USGS API request failed with status code {response.status_code}")
    
    data = response.json()
    time_series = data.get("value", {}).get("timeSeries", [])
    if not time_series:
        raise ValueError(f"No time series data returned for station {station_id}")
        
    readings = {}
    for ts in time_series:
        var_code = ts["variable"]["variableCode"][0]["value"]
        values = ts.get("values", [])[0].get("value", [])
        if values:
            latest_val = float(values[-1]["value"])
            readings[var_code] = latest_val
        else:
            raise ValueError(f"Missing instantaneous values for parameter {var_code} at station {station_id}")
            
    required_params = ["00010", "00300", "00400", "00095"]
    for param in required_params:
        if param not in readings:
            raise ValueError(f"Required parameter {param} is missing from station {station_id} telemetry")
            
    return readings

def align_and_predict(model, input_df):
    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names:
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_names]
    return model.predict(input_df)

def align_and_predict_proba(model, input_df):
    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names:
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_names]
    return model.predict_proba(input_df)

def generate_forecast_features(base_features, hours=24, site_cat_id=0):
    future_rows = []
    current_time = datetime.datetime.now()
    
    base_temp = base_features.get("temp_C", 20.0)
    base_do = base_features.get("DO_mgL", 8.0)
    base_pH = base_features.get("pH", 7.8)
    base_sal = base_features.get("salinity_PSU", 25.0)
    
    for i in range(hours):
        step_time = current_time + datetime.timedelta(hours=i)
        hour_of_day = step_time.hour
        day_of_year = step_time.timetuple().tm_yday
        
        temp_est = base_temp + 1.2 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
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
            'dDO_dt_1h': 0.1 * np.sin(2 * np.pi * hour_of_day / 24.0),
            'dDO_dt_3h': 0.2 * np.sin(2 * np.pi * hour_of_day / 24.0),
            'dDO_dt_6h': 0.3 * np.sin(2 * np.pi * hour_of_day / 24.0),
            'dDO_dt_12h': 0.4 * np.sin(2 * np.pi * hour_of_day / 24.0),
            'dDO_dt_24h': 0.0,
            'dpH_dt_3h': 0.05 * np.sin(2 * np.pi * hour_of_day / 24.0),
            'dpH_dt_6h': 0.1 * np.sin(2 * np.pi * hour_of_day / 24.0),
            'dDO_acceleration': 0.05,
            'DO_pH_ratio': do_pH_ratio,
            'DO_temp_ratio': do_temp_ratio,
            'DO_solubility_sat': do_solubility,
            'DO_saturation_ratio': do_sat_ratio,
            'DO_deficit': do_deficit,
            'DO_std_12h': 0.3,
            'DO_std_24h': 0.5,
            'pH_std_24h': 0.1,
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
        fetch_usgs_station_data(sid)
        valid_stations[name] = sid
    except Exception:
        pass

if not valid_stations:
    st.error("No active stations found reporting all required parameters.")
    st.stop()

selected_station_name = st.sidebar.selectbox("Select Validated Station", list(valid_stations.keys()))
selected_station_id = valid_stations[selected_station_name]

try:
    current_readings = fetch_usgs_station_data(selected_station_id)
except Exception as e:
    st.error(f"Failed to fetch live USGS data: {e}")
    st.stop()

base_df = pd.DataFrame([{
    "temp_C": current_readings["00010"],
    "DO_mgL": current_readings["00300"],
    "pH": current_readings["00400"],
    "salinity_PSU": current_readings["00095"]
}])

forecast_df = generate_forecast_features(base_df.iloc[0].to_dict(), hours=24)

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
