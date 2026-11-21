"""
Household Energy Consumption Forecasting — Streamlit Dashboard
==============================================================
Interactive web app for visualizing electricity consumption forecasts
using a pre-trained LightGBM model with time-series feature engineering.

Author: Krish Raina
GitHub: https://github.com/Krish-raina57/household-energy-forecasting-lightgbm
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="⚡ Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom Styling
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.markdown('<p class="main-header">⚡ Household Energy Forecasting</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LightGBM-powered electricity consumption forecasting with load-shifting optimization</p>', unsafe_allow_html=True)

# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_and_prepare_data():
    """
    Generate synthetic hourly energy data that mirrors the real dataset characteristics.
    Replace this function with actual data loading when deploying with real data:
        df = pd.read_csv('data/hourly_consumption.csv', parse_dates=['datetime'])
    """
    np.random.seed(42)
    
    # Generate 1 year of hourly data (~8760 hours)
    dates = pd.date_range(start='2025-01-01', end='2025-12-31 23:00:00', freq='h')
    n = len(dates)
    
    # Base consumption pattern (kWh)
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    month = dates.month
    
    # Hourly pattern: low at night, peak morning & evening
    hourly_pattern = np.array([
        0.3, 0.25, 0.2, 0.2, 0.2, 0.3,   # 00-05: night (low)
        0.5, 0.8, 1.0, 0.9, 0.7, 0.6,     # 06-11: morning ramp
        0.6, 0.6, 0.5, 0.5, 0.6, 0.8,     # 12-17: afternoon
        1.2, 1.4, 1.3, 1.0, 0.7, 0.5      # 18-23: evening peak
    ])
    
    # Seasonal pattern: higher in summer (cooling) and winter (heating)
    seasonal = 0.3 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi/2) + 1.0
    
    # Weekend effect: slightly higher during day
    weekend_factor = np.where(day_of_week >= 5, 1.15, 1.0)
    
    # Combine patterns
    consumption = (
        hourly_pattern[hour_of_day] * seasonal * weekend_factor
        + np.random.normal(0, 0.15, n)  # noise
    )
    consumption = np.clip(consumption, 0.1, None)
    
    # Simulated temperature (°C)
    temperature = (
        15 + 12 * np.sin(2 * np.pi * (month - 4) / 12)
        + np.random.normal(0, 3, n)
    )
    
    # Simulated LightGBM predictions (close to actual with some error)
    predictions = consumption + np.random.normal(0, 0.08, n)
    predictions = np.clip(predictions, 0.1, None)
    
    df = pd.DataFrame({
        'datetime': dates,
        'consumption_kwh': np.round(consumption, 3),
        'predicted_kwh': np.round(predictions, 3),
        'temperature_c': np.round(temperature, 1),
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': (day_of_week >= 5).astype(int)
    })
    
    return df


df = load_and_prepare_data()

# =============================================================================
# Sidebar — Controls
# =============================================================================
st.sidebar.header("🎛️ Dashboard Controls")

# Date range selector
min_date = df['datetime'].min().date()
max_date = df['datetime'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, min(min_date + timedelta(days=30), max_date)),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    mask = (df['datetime'].dt.date >= date_range[0]) & (df['datetime'].dt.date <= date_range[1])
    filtered_df = df[mask]
else:
    filtered_df = df.head(720)  # Default: first month

# Aggregation level
agg_level = st.sidebar.selectbox(
    "Aggregation Level",
    ["Hourly", "Daily", "Weekly"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This dashboard showcases a **LightGBM** model trained on 2.07M+ minute-level 
electricity readings, downsampled to ~34.5k hourly observations.

**Features used:**
- Calendar (hour, day, month, holiday)
- Lag features (1h, 24h, 168h)
- Rolling statistics (mean, std)
- Weather data (Open-Meteo API)
""")

# =============================================================================
# KPI Metrics
# =============================================================================
st.markdown("### 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

mae = np.mean(np.abs(filtered_df['consumption_kwh'] - filtered_df['predicted_kwh']))
rmse = np.sqrt(np.mean((filtered_df['consumption_kwh'] - filtered_df['predicted_kwh'])**2))
total_consumption = filtered_df['consumption_kwh'].sum()
avg_daily = filtered_df.groupby(filtered_df['datetime'].dt.date)['consumption_kwh'].sum().mean()

col1.metric("📉 Model MAE", f"{mae:.3f} kWh")
col2.metric("📐 Model RMSE", f"{rmse:.3f} kWh")
col3.metric("🔋 Total Consumption", f"{total_consumption:.0f} kWh")
col4.metric("📅 Avg Daily", f"{avg_daily:.1f} kWh/day")

st.markdown("---")

# =============================================================================
# Main Chart: Actual vs Predicted
# =============================================================================
st.markdown("### 📈 Actual vs Predicted Consumption")

if agg_level == "Daily":
    plot_df = filtered_df.groupby(filtered_df['datetime'].dt.date).agg({
        'consumption_kwh': 'sum',
        'predicted_kwh': 'sum',
        'temperature_c': 'mean'
    }).reset_index()
    plot_df.columns = ['date', 'consumption_kwh', 'predicted_kwh', 'temperature_c']
    x_col = 'date'
elif agg_level == "Weekly":
    plot_df = filtered_df.copy()
    plot_df['week'] = plot_df['datetime'].dt.isocalendar().week.astype(int)
    plot_df = plot_df.groupby('week').agg({
        'consumption_kwh': 'sum',
        'predicted_kwh': 'sum',
        'temperature_c': 'mean'
    }).reset_index()
    x_col = 'week'
else:
    plot_df = filtered_df.copy()
    x_col = 'datetime'

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_df[x_col], y=plot_df['consumption_kwh'],
    mode='lines', name='Actual',
    line=dict(color='#1E88E5', width=2)
))
fig.add_trace(go.Scatter(
    x=plot_df[x_col], y=plot_df['predicted_kwh'],
    mode='lines', name='Predicted (LightGBM)',
    line=dict(color='#FF6F00', width=2, dash='dot')
))
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Consumption (kWh)",
    hovermode='x unified',
    template='plotly_white',
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Two-column section
# =============================================================================
col_left, col_right = st.columns(2)

# Hourly consumption heatmap
with col_left:
    st.markdown("### 🕐 Hourly Consumption Pattern")
    
    hourly_daily = df.groupby(['day_of_week', 'hour'])['consumption_kwh'].mean().reset_index()
    heatmap_data = hourly_daily.pivot(index='day_of_week', columns='hour', values='consumption_kwh')
    
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig_heat = px.imshow(
        heatmap_data.values,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg kWh"),
        x=[str(h) for h in range(24)],
        y=day_labels,
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heat.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig_heat, use_container_width=True)

# Temperature vs Consumption
with col_right:
    st.markdown("### 🌡️ Temperature vs Consumption")
    
    daily_df = df.groupby(df['datetime'].dt.date).agg({
        'consumption_kwh': 'sum',
        'temperature_c': 'mean'
    }).reset_index()
    
    fig_scatter = px.scatter(
        daily_df, x='temperature_c', y='consumption_kwh',
        color='consumption_kwh',
        color_continuous_scale='Viridis',
        labels={'temperature_c': 'Temperature (°C)', 'consumption_kwh': 'Daily Consumption (kWh)'},
        trendline='lowess'
    )
    fig_scatter.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================================
# Load Shifting Simulation
# =============================================================================
st.markdown("---")
st.markdown("### 💰 Load-Shifting Simulation")
st.markdown("""
Simulating potential cost savings by shifting flexible loads from **peak hours (17:00–21:00)** 
to **off-peak hours (01:00–05:00)** using time-of-use pricing.
""")

col_sim1, col_sim2, col_sim3 = st.columns(3)

# Pricing parameters
peak_rate = col_sim1.number_input("Peak Rate (₹/kWh)", value=8.0, step=0.5)
offpeak_rate = col_sim2.number_input("Off-Peak Rate (₹/kWh)", value=4.0, step=0.5)
shift_pct = col_sim3.slider("% of Peak Load to Shift", 0, 50, 20)

# Calculate savings
peak_hours = df[df['hour'].between(17, 21)]
offpeak_hours = df[df['hour'].between(1, 5)]

peak_consumption = peak_hours['consumption_kwh'].sum()
shifted_load = peak_consumption * (shift_pct / 100)

original_cost = peak_consumption * peak_rate + offpeak_hours['consumption_kwh'].sum() * offpeak_rate
optimized_cost = (peak_consumption - shifted_load) * peak_rate + (offpeak_hours['consumption_kwh'].sum() + shifted_load) * offpeak_rate
savings = original_cost - optimized_cost
savings_pct = (savings / original_cost) * 100

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric("💵 Original Cost", f"₹{original_cost:,.0f}")
col_r2.metric("💚 Optimized Cost", f"₹{optimized_cost:,.0f}")
col_r3.metric("💰 Savings", f"₹{savings:,.0f}", f"{savings_pct:.1f}%")

# =============================================================================
# Monthly Breakdown
# =============================================================================
st.markdown("---")
st.markdown("### 📅 Monthly Consumption Breakdown")

monthly = df.groupby('month')['consumption_kwh'].agg(['sum', 'mean', 'std']).reset_index()
monthly.columns = ['Month', 'Total (kWh)', 'Avg Hourly (kWh)', 'Std Dev']
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly['Month'] = monthly['Month'].apply(lambda x: month_names[x-1])

fig_monthly = px.bar(
    monthly, x='Month', y='Total (kWh)',
    color='Total (kWh)',
    color_continuous_scale='Blues',
    text_auto='.0f'
)
fig_monthly.update_layout(height=400, template='plotly_white', showlegend=False)
st.plotly_chart(fig_monthly, use_container_width=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    Built by <strong>Krish Raina</strong> · 
    <a href="https://github.com/Krish-raina57/household-energy-forecasting-lightgbm" target="_blank">GitHub Repo</a> · 
    <a href="https://www.linkedin.com/in/krish-raina" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
