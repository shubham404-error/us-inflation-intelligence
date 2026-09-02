from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data import DATA_CUTOFF, START_DATE, fetch_model_data, latest_snapshot
from model import build_forecasts, model_diagnostics, driver_table

load_dotenv()

st.set_page_config(
    page_title="US Inflation Intelligence",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bloomberg-inspired, but not a copy.
st.markdown(
    """
    <style>
    .stApp {
        background: #0b0f14;
        color: #e7edf5;
    }
    [data-testid="stSidebar"] {
        background: #111720;
        border-right: 1px solid #28313d;
    }
    [data-testid="stSidebar"] * {
        color: #dce4ee !important;
    }
    .terminal-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: .02em;
        margin-bottom: 0;
    }
    .terminal-subtitle {
        color: #98a4b3;
        font-size: .9rem;
        margin-bottom: 1rem;
    }
    .section-label {
        color: #aeb9c7;
        font-size: .72rem;
        letter-spacing: .12em;
        font-weight: 700;
        text-transform: uppercase;
        margin: .8rem 0 .35rem;
    }
    .statusline {
        color: #8e9aaa;
        font-size: .78rem;
        border-bottom: 1px solid #28313d;
        padding-bottom: .6rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: #121923;
        border: 1px solid #28313d;
        padding: .65rem .75rem;
        border-radius: .35rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ba7b5 !important;
        font-size: .78rem;
    }
    div[data-testid="stMetricValue"] {
        color: #f3f6fa !important;
        font-size: 1.45rem;
    }
    .small-note {
        color: #7f8b9a;
        font-size: .72rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## US INFLATION INTELLIGENCE")
st.sidebar.caption("Macro research terminal")

page = st.sidebar.radio(
    "Navigate",
    ["Terminal", "Forecast", "Drivers", "Data"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"Data: {START_DATE} → {DATA_CUTOFF}")
st.sidebar.caption("Primary access: FRED API")

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED_API_KEY is missing. Create a local .env file with "
        "FRED_API_KEY=your_key_here."
    )
    st.stop()

@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    return fetch_model_data()

if st.sidebar.button("Refresh FRED data", use_container_width=True):
    load_data.clear()

try:
    data = load_data()
except Exception as exc:
    st.error(f"Data pipeline failed: {exc}")
    st.stop()

snapshot = latest_snapshot(data)

if page == "Data":
    st.markdown('<div class="terminal-title">DATA FOUNDATION</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal-subtitle">Point-in-time data through 31 August 2026. '
        'Latest observations depend on each series release schedule.</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        snapshot["series_status"],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Sources of record: BEA for PCE, BLS for CPI/shelter/unemployment, "
        "EIA for crude oil, University of Michigan for inflation expectations."
    )
    st.stop()

forecasts = build_forecasts(data)
diag = model_diagnostics(forecasts)
drivers = driver_table(forecasts)

if page == "Terminal":
    st.markdown('<div class="terminal-title">US INFLATION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="terminal-subtitle">Is U.S. inflation moving sustainably toward the Federal Reserve\'s 2% objective?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="statusline">DATA VINTAGE: {DATA_CUTOFF} &nbsp; | &nbsp; '
        f'LATEST PCE OBSERVATION: {snapshot["latest_pce_date"]} &nbsp; | &nbsp; '
        f'MODEL: {forecasts["selected_model"]}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PCE inflation", f'{snapshot["pce_inflation"]:.2f}%')
    c2.metric("Core PCE", f'{snapshot["core_pce"]:.2f}%')
    c3.metric("Fed target", "2.00%")
    c4.metric("3M forecast", f'{forecasts["point_forecast_3m"]:.2f}%')
    c5.metric("Distance to target", f'{forecasts["point_forecast_3m"] - 2:+.2f} pp')

    left, right = st.columns([2.25, 1])

    with left:
        st.markdown('<div class="section-label">PCE INFLATION · ACTUAL VS 3M FORECAST</div>', unsafe_allow_html=True)
        chart = forecasts["chart_df"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart.index,
                y=chart["pce_inflation"],
                mode="lines",
                name="PCE inflation",
                line=dict(color="#f4b942", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart["forecast_index"],
                y=chart["forecast"],
                mode="lines",
                name="Forecast",
                line=dict(color="#4fa3ff", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart["forecast_index"],
                y=chart["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart["forecast_index"],
                y=chart["lower"],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="80% prediction range",
                hoverinfo="skip",
            )
        )
        fig.add_hline(y=2.0, line_dash="dot", line_color="#8090a0", annotation_text="Fed 2%")
        fig.update_layout(
            height=430,
            margin=dict(l=8, r=8, t=10, b=8),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(color="#c9d2dd"),
            legend=dict(orientation="h", y=1.02, x=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="% YoY", gridcolor="#1d2631"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-label">INFLATION SNAPSHOT</div>', unsafe_allow_html=True)
        st.metric("Inflation pressure", f'{forecasts["pressure_score"]}/100')
        st.metric("Current regime", forecasts["regime"])
        st.metric("Model confidence", forecasts["confidence"])
        st.metric("6M forecast", f'{forecasts["point_forecast_6m"]:.2f}%')
        st.caption(
            "Confidence is a model-quality label based on recent walk-forward error, "
            "not a probability that the forecast will be correct."
        )

    st.markdown('<div class="section-label">AI MACRO BRIEF</div>', unsafe_allow_html=True)
    st.info(forecasts["macro_brief"])

elif page == "Forecast":
    st.markdown('<div class="terminal-title">FORECAST</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-subtitle">Baseline vs econometric vs machine-learning forecasts.</div>', unsafe_allow_html=True)

    fc = forecasts["forecast_table"].copy()
    st.dataframe(fc, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">WALK-FORWARD MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.dataframe(diag["performance"], use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">PREDICTION INTERVAL CALIBRATION</div>', unsafe_allow_html=True)
    st.dataframe(diag["coverage"], use_container_width=True, hide_index=True)
    st.caption(
        "XGBoost prediction ranges use temporal conformal calibration from walk-forward "
        "out-of-sample residuals. They are prediction intervals, not confidence intervals."
    )

elif page == "Drivers":
    st.markdown('<div class="terminal-title">DRIVERS</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-subtitle">What is pushing the model forecast higher or lower?</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">MODEL CONTRIBUTIONS</div>', unsafe_allow_html=True)
    st.dataframe(drivers, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">REGIME HISTORY</div>', unsafe_allow_html=True)
    regime_df = forecasts["regime_history"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=regime_df.index,
            y=regime_df["pce_inflation"],
            mode="lines",
            name="PCE inflation",
            line=dict(color="#f4b942", width=2),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(color="#c9d2dd"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="% YoY", gridcolor="#1d2631"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">PLAIN-ENGLISH INTERPRETATION</div>', unsafe_allow_html=True)
    st.write(
        f"Current inflation is {snapshot['pce_inflation']:.2f}%. "
        f"The model's 3-month forecast is {forecasts['point_forecast_3m']:.2f}%. "
        f"The current inflation environment is classified as {forecasts['regime'].lower()}."
    )

st.caption("Research tool only. This application is not investment advice or an official Federal Reserve forecast.")
