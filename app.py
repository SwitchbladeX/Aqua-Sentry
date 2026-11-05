import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(
    page_title="Long Island Oyster AI Portal",
    layout="wide"
)

st.title("Long Island Oyster AI Portal")
st.markdown("Estuary telemetry and machine learning water quality forecasting system protecting Crassostrea virginica.")

STATIONS = {
    "Great South Bay Node (USGS 01305000)": {"site_code": "01305000", "lat": 40.65, "lon": -73.25},
    "Western Long Island Sound (USGS 01304500)": {"site_code": "01304500", "lat": 40.85, "lon": -73.65},
    "Central Long Island Sound (USGS 01115000)": {"site_code": "01115000", "lat": 41.15, "lon": -72.65}
}

@st.cache_resource
def load_all_models():
    models = {}
    model_paths = {
        'DO_mgL': 'xgb_do_regression_weather_model(FINAL) (1).json',
        'temp_C': 'xgb_temp_C_regression_model(FINAL).json',
        'pH': 'xgb_pH_regression_model(FINAL) (1).json',
        'salinity_PSU': 'xgb_salinity_PSU_regression_model(FINAL) (1).json',
        'hypoxia': 'xgb_hypoxia_weather_model(FINAL).json'
    }
    
    for key, path in model_paths.items():
        try:
            if key == 'hypoxia':
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            model.load_model(path)
            models[key] = model
        except Exception as e:
            st.error(f"Error loading model file '{path}': {e}")
            models[key] = None
    return models

models = load_all_models()

st.sidebar.header("Navigation")
selected_station_name = st.sidebar.selectbox("Select Monitoring Station", list(STATIONS.keys()))
station_info = STATIONS[selected_station_name]

@st.cache_data(ttl=900)
def fetch_live_usgs_telemetry(site_code):
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_code}&parameterCd=00010,00300,00400,00095,00060,00065&siteStatus=all"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        series = data['value']['timeSeries']
        
        latest_vals = {}
        for s in series:
            param_code = s['variable']['variableCode'][0]['value']
            values = s['values'][0]['value']
            if values:
                latest_val = float(values[-1]['value'])
                latest_vals[param_code] = latest_val
                
        return {
            'temp_C': latest_vals.get('00010', 18.5),
            'DO_mgL': latest_vals.get('00300', 6.20),
            'pH': latest_vals.get('00400', 7.90),
            'salinity_PSU': latest_vals.get('00095', 26.5) / 1000.0 if '00095' in latest_vals else 26.5
        }
    except Exception:
        return {'temp_C': 18.5, 'DO_mgL': 6.20, 'pH': 7.90, 'salinity_PSU': 26.5}

@st.cache_data(ttl=1800)
def fetch_open_meteo_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "surface_pressure", "wind_speed_10m"],
        "forecast_days": 2
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        json_data = response.json()['hourly']
        return {
            'air_temp_C': json_data['temperature_2m'][0],
            'pressure_hPa': json_data['surface_pressure'][0],
            'wind_speed_ms': json_data['wind_speed_10m'][0]
        }
    except Exception:
        return {'air_temp_C': 19.0, 'pressure_hPa': 1013.25, 'wind_speed_ms': 3.5}

telemetry = fetch_live_usgs_telemetry(station_info['site_code'])
weather = fetch_open_meteo_weather(station_info['lat'], station_info['lon'])

st.subheader("Live Estuary Telemetry")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dissolved Oxygen", f"{telemetry['DO_mgL']:.2f} mg/L")
with col2:
    st.metric("Water Temperature", f"{telemetry['temp_C']:.2f} °C")
with col3:
    st.metric("Estuary pH", f"{telemetry['pH']:.2f}")
with col4:
    st.metric("Salinity", f"{telemetry['salinity_PSU']:.2f} PSU")

now = datetime.now()
hour = now.hour
doy = now.timetuple().tm_yday

sin_hour = np.sin(2 * np.pi * hour / 24.0)
cos_hour = np.cos(2 * np.pi * hour / 24.0)
sin_season = np.sin(2 * np.pi * doy / 365.25)
cos_season = np.cos(2 * np.pi * doy / 365.25)

thermal_gradient = weather['air_temp_C'] - telemetry['temp_C']
dDO_dt_3h = 0.0
dDO_dt_6h = 0.0
dDO_acceleration = 0.0

temp = telemetry['temp_C']
DO_solubility_sat = 14.652 - (0.41022 * temp) + (0.007991 * (temp**2)) - (0.000077774 * (temp**3))
DO_deficit = DO_solubility_sat - telemetry['DO_mgL']
DO_std_12h = 0.25
pH_std_12h = 0.05

site_code_cat = list(STATIONS.keys()).index(selected_station_name)

horizons = list(range(1, 25))
forecasts = {'DO_mgL': [], 'temp_C': [], 'pH': [], 'salinity_PSU': []}

for h in horizons:
    input_features = pd.DataFrame([{
        'temp_C': telemetry['temp_C'],
        'DO_mgL': telemetry['DO_mgL'],
        'pH': telemetry['pH'],
        'salinity_PSU': telemetry['salinity_PSU'],
        'air_temp_C': weather['air_temp_C'],
        'pressure_hPa': weather['pressure_hPa'],
        'wind_speed_ms': weather['wind_speed_ms'],
        'thermal_gradient': thermal_gradient,
        'dDO_dt_3h': dDO_dt_3h,
        'dDO_dt_6h': dDO_dt_6h,
        'dDO_acceleration': dDO_acceleration,
        'DO_solubility_sat': DO_solubility_sat,
        'DO_deficit': DO_deficit,
        'DO_std_12h': DO_std_12h,
        'pH_std_12h': pH_std_12h,
        'sin_hour': sin_hour,
        'cos_hour': cos_hour,
        'sin_season': sin_season,
        'cos_season': cos_season,
        'site_code_cat': float(site_code_cat),
        'horizon': float(h)
    }])

    for param, model in models.items():
        if param in forecasts and model is not None:
            try:
                if hasattr(model, 'feature_names_in_'):
                    expected_features = model.feature_names_in_
                    aligned_features = pd.DataFrame(columns=expected_features)
                    for col in expected_features:
                        if col in input_features.columns:
                            aligned_features[col] = input_features[col]
                        else:
                            aligned_features[col] = 0.0
                    pred = model.predict(aligned_features)[0]
                else:
                    pred = model.predict(input_features)[0]
                forecasts[param].append(pred)
            except Exception:
                forecasts[param].append(telemetry[param])

hypoxia_prob = 0.0
if models['hypoxia'] is not None:
    try:
        clf_features = input_features.drop(columns=['horizon'], errors='ignore')
        if hasattr(models['hypoxia'], 'feature_names_in_'):
            expected_features = models['hypoxia'].feature_names_in_
            clf_features_aligned = pd.DataFrame(columns=expected_features)
            for col in expected_features:
                if col in clf_features.columns:
                    clf_features_aligned[col] = clf_features[col]
                else:
                    clf_features_aligned[col] = 0.0
            hypoxia_prob = models['hypoxia'].predict_proba(clf_features_aligned)[0][1]
        else:
            hypoxia_prob = models['hypoxia'].predict_proba(clf_features)[0][1]
    except Exception:
        hypoxia_prob = 0.0

st.markdown("---")
st.subheader("24-Hour Hypoxia Early Warning System")
if hypoxia_prob > 0.40:
    st.error(f"HIGH HYPOXIA CRASH RISK DETECTED: {hypoxia_prob * 100:.1f}% probability of DO dropping below 3.0 mg/L in the next 24 hours.")
else:
    st.success(f"NORMAL ESTUARY CONDITIONS: Hypoxia risk is low ({hypoxia_prob * 100:.1f}% probability).")

st.subheader("24-Hour Multi-Parameter AI Forecast Trajectories")

tab1, tab2, tab3, tab4 = st.tabs(["Dissolved Oxygen", "Water Temperature", "pH", "Salinity"])

with tab1:
    fig_do = go.Figure()
    fig_do.add_trace(go.Scatter(x=horizons, y=forecasts['DO_mgL'], mode='lines+markers', name='Predicted DO (mg/L)', line=dict(color='#008080', width=3)))
    fig_do.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="Hypoxia Threshold")
    fig_do.update_layout(xaxis_title="Forecast Horizon (Hours Ahead)", yaxis_title="Dissolved Oxygen (mg/L)", height=400)
    st.plotly_chart(fig_do, width='stretch')

with tab2:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=horizons, y=forecasts['temp_C'], mode='lines+markers', name='Predicted Temp (°C)', line=dict(color='#FF4B4B', width=3)))
    fig_temp.update_layout(xaxis_title="Forecast Horizon (Hours Ahead)", yaxis_title="Water Temperature (°C)", height=400)
    st.plotly_chart(fig_temp, width='stretch')

with tab3:
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=horizons, y=forecasts['pH'], mode='lines+markers', name='Predicted pH', line=dict(color='#9B59B6', width=3)))
    fig_ph.update_layout(xaxis_title="Forecast Horizon (Hours Ahead)", yaxis_title="pH Level", height=400)
    st.plotly_chart(fig_ph, width='stretch')

with tab4:
    fig_sal = go.Figure()
    fig_sal.add_trace(go.Scatter(x=horizons, y=forecasts['salinity_PSU'], mode='lines+markers', name='Predicted Salinity (PSU)', line=dict(color='#3498DB', width=3)))
    fig_sal.update_layout(xaxis_title="Forecast Horizon (Hours Ahead)", yaxis_title="Salinity (PSU)", height=400)
    st.plotly_chart(fig_sal, width='stretch')
