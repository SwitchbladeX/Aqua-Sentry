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

CANDIDATE_STATIONS = {
    "West Harbor at Oyster Bay": {"site_code": "01303152", "lat": 40.875, "lon": -73.535},
    "Western Long Island Sound": {"site_code": "01304500", "lat": 40.850, "lon": -73.650},
    "Great South Bay Near Patchogue": {"site_code": "01305000", "lat": 40.650, "lon": -73.250},
    "Central Long Island Sound": {"site_code": "01115000", "lat": 41.150, "lon": -72.650},
    "Hudson River at Battery": {"site_code": "01164500", "lat": 40.700, "lon": -74.015},
    "East River at Wards Island": {"site_code": "01302020", "lat": 40.783, "lon": -73.916},
    "Jamaica Bay": {"site_code": "01311143", "lat": 40.633, "lon": -73.816},
    "Nissequogue River": {"site_code": "01303500", "lat": 40.891, "lon": -73.216}
}

@st.cache_data(ttl=3600)
def validate_stations(stations_dict):
    valid = {}
    for name, info in stations_dict.items():
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={info['site_code']}&parameterCd=00010,00300,00400,00095&siteStatus=all"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and response.text.strip():
                data = response.json()
                series = data.get('value', {}).get('timeSeries', [])
                found_params = set()
                for s in series:
                    p_code = s.get('variable', {}).get('variableCode', [{}])[0].get('value')
                    if p_code in ['00010', '00300', '00400', '00095']:
                        found_params.add(p_code)
                if len(found_params) == 4:
                    valid[name] = info
        except Exception:
            pass
    return valid

STATIONS = validate_stations(CANDIDATE_STATIONS)

if not STATIONS:
    st.error("No stations currently returning all required parameters.")
    st.stop()

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
            
    required_params = {
        '00010': 'Water Temperature (00010)',
        '00300': 'Dissolved Oxygen (00300)',
        '00400': 'pH (00400)',
        '00095': 'Salinity/Conductivity (00095)'
    }
    
    missing = [name for code, name in required_params.items() if code not in latest_vals]
    if missing:
        raise ValueError(f"USGS feed for station {site_code} is missing required parameters: {', '.join(missing)}")
        
    return {
        'temp_C': latest_vals['00010'],
        'DO_mgL': latest_vals['00300'],
        'pH': latest_vals['00400'],
        'salinity_PSU': latest_vals['00095'] / 1000.0
    }

@st.cache_data(ttl=1800)
def fetch_open_meteo_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "surface_pressure", "wind_speed_10m"],
        "forecast_days": 2
    }
    response = requests.get(url, params=params, timeout=10)
    json_data = response.json()['hourly']
    return {
        'air_temp_C': json_data['temperature_2m'][0],
        'pressure_hPa': json_data['surface_pressure'][0],
        'wind_speed_ms': json_data['wind_speed_10m'][0]
    }

telemetry_error = None
telemetry = None
try:
    telemetry = fetch_live_usgs_telemetry(station_info['site_code'])
except Exception as e:
    telemetry_error = str(e)

weather_error = None
weather = None
try:
    weather = fetch_open_meteo_weather(station_info['lat'], station_info['lon'])
except Exception as e:
    weather_error = str(e)

st.subheader("Live Estuary Telemetry")

if telemetry_error is not None:
    st.error(f"USGS Telemetry Fetch Failed: {telemetry_error}")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dissolved Oxygen", f"{telemetry['DO_mgL']:.2f} mg/L")
    with col2:
        st.metric("Water Temperature", f"{telemetry['temp_C']:.2f} °C")
    with col3:
        st.metric("Estuary pH", f"{telemetry['pH']:.2f}")
    with col4:
        st.metric("Salinity", f"{telemetry['salinity_PSU']:.2f} PSU")

if weather_error is not None:
    st.error(f"Open-Meteo Weather Fetch Failed: {weather_error}")

if telemetry_error is not None or weather_error is not None:
    st.warning("Forecast generation paused due to telemetry or weather API retrieval failure.")
else:
    now = datetime.now()
    current_hour = now.hour
    doy = now.timetuple().tm_yday
    site_code_cat = float(list(STATIONS.keys()).index(selected_station_name))

    horizons = list(range(1, 25))
    forecasts = {'DO_mgL': [], 'temp_C': [], 'pH': [], 'salinity_PSU': [], 'hypoxia_prob': []}

    for h in horizons:
        forecast_hour = (current_hour + h) % 24
        sin_hour = np.sin(2 * np.pi * forecast_hour / 24.0)
        cos_hour = np.cos(2 * np.pi * forecast_hour / 24.0)
        sin_season = np.sin(2 * np.pi * doy / 365.25)
        cos_season = np.cos(2 * np.pi * doy / 365.25)

        air_temp_projected = weather['air_temp_C'] + 2.0 * np.sin(2 * np.pi * (forecast_hour - 6) / 24.0)
        thermal_gradient = air_temp_projected - telemetry['temp_C']
        
        temp = telemetry['temp_C']
        DO_solubility_sat = 14.652 - (0.41022 * temp) + (0.007991 * (temp**2)) - (0.000077774 * (temp**3))
        DO_deficit = DO_solubility_sat - telemetry['DO_mgL']
        DO_std_12h = 0.25
        pH_std_12h = 0.05

        full_features = {
            'temp_C': telemetry['temp_C'],
            'DO_mgL': telemetry['DO_mgL'],
            'pH': telemetry['pH'],
            'salinity_PSU': telemetry['salinity_PSU'],
            'air_temp_C': air_temp_projected,
            'pressure_hPa': weather['pressure_hPa'],
            'wind_speed_ms': weather['wind_speed_ms'],
            'thermal_gradient': thermal_gradient,
            'dDO_dt_3h': 0.0,
            'dDO_dt_6h': 0.0,
            'dDO_acceleration': 0.0,
            'DO_solubility_sat': DO_solubility_sat,
            'DO_deficit': DO_deficit,
            'DO_std_12h': DO_std_12h,
            'pH_std_12h': pH_std_12h,
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'sin_season': sin_season,
            'cos_season': cos_season,
            'site_code_cat': site_code_cat,
            'horizon': float(h)
        }

        master_df = pd.DataFrame([full_features])

        for param, model in models.items():
            if param in forecasts and model is not None:
                booster = model.get_booster()
                expected_cols = booster.feature_names
                
                aligned_df = pd.DataFrame(columns=expected_cols)
                for col in expected_cols:
                    if col in master_df.columns:
                        aligned_df[col] = master_df[col]
                    else:
                        aligned_df[col] = 0.0
                        
                pred = model.predict(aligned_df)[0]
                forecasts[param].append(float(pred))

        if models.get('hypoxia') is not None:
            clf_model = models['hypoxia']
            booster = clf_model.get_booster()
            expected_cols = booster.feature_names
            
            aligned_clf = pd.DataFrame(columns=expected_cols)
            for col in expected_cols:
                if col in master_df.columns:
                    aligned_clf[col] = master_df[col]
                else:
                    aligned_clf[col] = 0.0
                    
            prob = float(clf_model.predict_proba(aligned_clf)[0][1])
            forecasts['hypoxia_prob'].append(prob)

    current_hypoxia_prob = forecasts['hypoxia_prob'][0] if forecasts['hypoxia_prob'] else 0.0

    st.markdown("---")
    st.subheader("24-Hour Hypoxia Early Warning System")
    if current_hypoxia_prob > 0.40:
        st.error(f"HIGH HYPOXIA CRASH RISK DETECTED: {current_hypoxia_prob * 100:.1f}% probability of DO dropping below 3.0 mg/L in the next 24 hours.")
    else:
        st.success(f"NORMAL ESTUARY CONDITIONS: Hypoxia risk is low ({current_hypoxia_prob * 100:.1f}% probability).")

    st.subheader("24-Hour Multi-Parameter AI Forecast Trajectories")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dissolved Oxygen", "Water Temperature", "pH", "Salinity", "Hypoxia Risk"])

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

    with tab5:
        fig_hyp = go.Figure()
        fig_hyp.add_trace(go.Scatter(x=horizons, y=forecasts['hypoxia_prob'], mode='lines+markers', name='Hypoxia Probability', line=dict(color='#E74C3C', width=3)))
        fig_hyp.add_hline(y=0.40, line_dash="dash", line_color="orange", annotation_text="Risk Warning Threshold")
        fig_hyp.update_layout(xaxis_title="Forecast Horizon (Hours Ahead)", yaxis_title="Hypoxia Probability", height=400)
        st.plotly_chart(fig_hyp, width='stretch')
