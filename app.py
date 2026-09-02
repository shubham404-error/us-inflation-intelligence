from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data import DATA_CUTOFF, START_DATE, fetch_model_data, latest_snapshot
from model import build_forecasts, driver_table, model_diagnostics


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="US Inflation Intelligence",
    page_icon="●",
    layout="wide",
)


# ============================================================
# DARK TERMINAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0f14;
        color: #e8edf3;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] {
        background-color: #111720;
        border-right: 1px solid #293340;
    }

    [data-testid="stSidebar"] * {
        color: #dce4ee !important;
    }

    .title {
        font-size: 2rem;
        font-weight: 700;
        color: #f3f6fa;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        font-size: 0.9rem;
        color: #8f9baa;
        margin-bottom: 1rem;
    }

    .status {
        font-size: 0.74rem;
        color: #7e8a99;
        border-top: 1px solid #28313d;
        border-bottom: 1px solid #28313d;
        padding: 0.55rem 0;
        margin-bottom: 1rem;
    }

    .section {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        color: #9aa6b5;
        margin: 1rem 0 0.45rem;
    }

    div[data-testid="stMetric"] {
        background: #121923;
        border: 1px solid #28313d;
        border-radius: 4px;
        padding: 0.7rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa6b5 !important;
        font-size: 0.75rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f6fa !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY CHECK
# ============================================================

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED_API_KEY is missing. "
        "Add it to Streamlit Cloud Secrets or your local .env file."
    )
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return fetch_model_data()


try:
    with st.spinner("Loading FRED data..."):
        data = load_data()

except Exception as exc:
    st.error(f"Data pipeline failed: {exc}")
    st.stop()


# ============================================================
# SNAPSHOT
# ============================================================

try:
    snapshot = latest_snapshot(data)

except Exception as exc:
    st.error(f"Snapshot failed: {exc}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">US INFLATION INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Forecasting U.S. PCE inflation and assessing whether inflation is moving toward 2%."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status">
        DATA WINDOW: {START_DATE} → {DATA_CUTOFF}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        LATEST PCE: {snapshot["latest_pce_date"]}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        OBSERVATIONS: {len(data):,}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RUN FORECAST
# ============================================================

try:

    with st.spinner(
        "Running SARIMAX and XGBoost walk-forward forecasting..."
    ):
        forecasts = build_forecasts(data)

except Exception as exc:

    st.error(
        "Forecasting pipeline failed."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# TOP KPIs
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Current PCE",
    f'{snapshot["pce_inflation"]:.2f}%',
)

c2.metric(
    "Core PCE",
    f'{snapshot["core_pce"]:.2f}%',
)

c3.metric(
    "Fed Target",
    "2.00%",
)

c4.metric(
    "3M Forecast",
    f'{forecasts["point_forecast_3m"]:.2f}%',
)

c5.metric(
    "6M Forecast",
    f'{forecasts["point_forecast_6m"]:.2f}%',
)


# ============================================================
# FORECAST CHART
# ============================================================

st.markdown(
    '<div class="section">PCE INFLATION · ACTUAL VS FORECAST</div>',
    unsafe_allow_html=True,
)

chart = forecasts["chart_df"]

history = chart[
    chart["pce_inflation"].notna()
]

future = chart[
    chart["forecast"].notna()
]

fig = go.Figure()


# Actual
fig.add_trace(
    go.Scatter(
        x=history.index,
        y=history["pce_inflation"],
        mode="lines",
        name="Actual",
        line=dict(
            color="#f4b942",
            width=2.2,
        ),
    )
)


# Forecast
fig.add_trace(
    go.Scatter(
        x=future.index,
        y=future["forecast"],
        mode="lines",
        name="Forecast",
        line=dict(
            color="#4fa3ff",
            width=2.2,
            dash="dash",
        ),
    )
)


# Prediction interval
fig.add_trace(
    go.Scatter(
        x=future.index,
        y=future["upper"],
        mode="lines",
        line=dict(
            width=0,
        ),
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter(
        x=future.index,
        y=future["lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(79,163,255,0.12)",
        line=dict(
            width=0,
        ),
        name="80% prediction range",
    )
)


# Fed target
fig.add_hline(
    y=2.0,
    line_dash="dot",
    line_color="#7b8795",
    annotation_text="Fed 2%",
    annotation_position="top left",
)


fig.update_layout(
    height=470,
    paper_bgcolor="#0b0f14",
    plot_bgcolor="#0b0f14",
    font=dict(
        color="#c9d2dd",
    ),
    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10,
    ),
    hovermode="x unified",
    xaxis=dict(
        showgrid=False,
    ),
    yaxis=dict(
        title="% YoY",
        gridcolor="#1c2530",
    ),
    legend=dict(
        orientation="h",
        y=1.05,
        x=0,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.caption(
    "Prediction range shown above is generated from historical "
    "walk-forward XGBoost errors using temporal conformal calibration."
)


# ============================================================
# FORECAST DETAILS
# ============================================================

left, right = st.columns([1.4, 1])


with left:

    st.markdown(
        '<div class="section">FORECAST OUTPUT</div>',
        unsafe_allow_html=True,
    )

    forecast_table = forecasts[
        "forecast_table"
    ].copy()

    forecast_table[
        "Forecast"
    ] = forecast_table[
        "Forecast"
    ].round(2)

    forecast_table[
        "Lower"
    ] = forecast_table[
        "Lower"
    ].round(2)

    forecast_table[
        "Upper"
    ] = forecast_table[
        "Upper"
    ].round(2)

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
    )


with right:

    st.markdown(
        '<div class="section">CURRENT STATE</div>',
        unsafe_allow_html=True,
    )

    st.metric(
        "Inflation Pressure",
        f'{forecasts["pressure_score"]}/100',
    )

    st.metric(
        "Current Regime",
        forecasts["regime"],
    )

    st.metric(
        "Model",
        forecasts["selected_model"],
    )

    st.metric(
        "Model Confidence",
        forecasts["confidence"],
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section">WALK-FORWARD PERFORMANCE</div>',
    unsafe_allow_html=True,
)

diagnostics = model_diagnostics(
    forecasts
)

st.dataframe(
    diagnostics["performance"],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CONFORMAL CALIBRATION
# ============================================================

st.markdown(
    '<div class="section">PREDICTION INTERVAL CALIBRATION</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    diagnostics["coverage"],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "The XGBoost interval is a prediction interval, not a confidence interval. "
    "Coverage is evaluated from historical walk-forward residuals."
)


# ============================================================
# DRIVERS
# ============================================================

st.markdown(
    '<div class="section">MODEL DRIVERS</div>',
    unsafe_allow_html=True,
)

drivers = driver_table(
    forecasts
)

if not drivers.empty:

    st.dataframe(
        drivers,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# AI-STYLE MACRO SUMMARY
# ============================================================

st.markdown(
    '<div class="section">MACRO SUMMARY</div>',
    unsafe_allow_html=True,
)

st.info(
    forecasts["macro_brief"]
)


# ============================================================
# DATA STATUS
# ============================================================

with st.expander(
    "View data status"
):

    st.dataframe(
        snapshot["series_status"],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "US Inflation Intelligence · Research / portfolio tool · "
    "Not an official Federal Reserve forecast or investment advice."
)
