from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data import DATA_CUTOFF, START_DATE, fetch_model_data, latest_snapshot
from model import build_forecasts, driver_table, model_diagnostics


# ============================================================
# APP CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="US Inflation Intelligence",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #0b0f14;
        color: #e8edf3;
    }

    [data-testid="stSidebar"] {
        background-color: #111720;
        border-right: 1px solid #293340;
    }

    [data-testid="stSidebar"] * {
        color: #dce4ee !important;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- HEADERS ---------- */

    .terminal-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.015em;
        color: #f2f5f8;
        margin-bottom: 0;
    }

    .terminal-subtitle {
        color: #919eae;
        font-size: 0.9rem;
        margin-top: 0.15rem;
        margin-bottom: 0.9rem;
    }

    .section-label {
        color: #9da9b8;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 1rem;
        margin-bottom: 0.45rem;
    }

    .statusline {
        color: #7f8b9b;
        font-size: 0.74rem;
        border-top: 1px solid #28313d;
        border-bottom: 1px solid #28313d;
        padding: 0.6rem 0;
        margin-bottom: 1rem;
    }

    .small-note {
        color: #7f8b9a;
        font-size: 0.72rem;
    }

    /* ---------- KPI CARDS ---------- */

    div[data-testid="stMetric"] {
        background: #121923;
        border: 1px solid #28313d;
        border-radius: 5px;
        padding: 0.7rem 0.8rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa6b5 !important;
        font-size: 0.75rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f2f5f8 !important;
        font-size: 1.5rem !important;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        width: 100%;
        background: #151d28;
        color: #dce4ee;
        border: 1px solid #35404d;
        border-radius: 4px;
    }

    .stButton > button:hover {
        border-color: #667789;
        color: #ffffff;
    }

    /* ---------- DATAFRAMES ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #28313d;
    }

    /* ---------- ALERTS ---------- */

    .stAlert {
        border-radius: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## US INFLATION INTELLIGENCE")
st.sidebar.caption("Macro research terminal")

page = st.sidebar.radio(
    "Navigation",
    [
        "Terminal",
        "Forecast",
        "Drivers",
        "Scenarios",
        "Data",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.caption(f"Historical window: {START_DATE} → {DATA_CUTOFF}")
st.sidebar.caption("Data access: FRED API")

if st.sidebar.button("Refresh FRED Data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# API KEY CHECK
# ============================================================

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED API key not found. Create a local `.env` file containing:\n\n"
        "`FRED_API_KEY=your_api_key_here`"
    )
    st.stop()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    return fetch_model_data()


with st.spinner("Loading FRED data..."):
    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Unable to load the FRED data pipeline: {exc}")
        st.stop()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = [
    "pcepi",
    "core_pce",
    "shelter",
    "unemployment",
    "real_consumption",
    "inflation_expectations",
    "oil",
    "pce_inflation",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        "The data pipeline is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# SNAPSHOT + MODELS
# ============================================================

with st.spinner("Running forecasting models..."):
    try:
        snapshot = latest_snapshot(data)
        forecasts = build_forecasts(data)
        diagnostics = model_diagnostics(forecasts)
        drivers = driver_table(forecasts)
    except Exception as exc:
        st.error(f"The forecasting pipeline failed: {exc}")
        st.stop()


# ============================================================
# HELPER
# ============================================================

def add_source_note(text: str) -> None:
    st.markdown(
        f'<div class="small-note">{text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TERMINAL PAGE
# ============================================================

if page == "Terminal":

    st.markdown(
        '<div class="terminal-title">US INFLATION INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="terminal-subtitle">'
        "Is U.S. inflation moving sustainably toward the Federal Reserve's 2% objective?"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="statusline">
            DATA VINTAGE: {DATA_CUTOFF}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            LATEST PCE: {snapshot["latest_pce_date"]}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            SELECTED MODEL: {forecasts["selected_model"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- TOP KPIs ----------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "PCE inflation",
        f'{snapshot["pce_inflation"]:.2f}%',
    )

    c2.metric(
        "Core PCE",
        f'{snapshot["core_pce"]:.2f}%',
    )

    c3.metric(
        "Fed target",
        "2.00%",
    )

    c4.metric(
        "3M forecast",
        f'{forecasts["point_forecast_3m"]:.2f}%',
    )

    c5.metric(
        "Distance to target",
        f'{forecasts["point_forecast_3m"] - 2:+.2f} pp',
    )

    # ---------- MAIN CONTENT ----------

    left, right = st.columns([2.35, 1])

    with left:

        st.markdown(
            '<div class="section-label">'
            "PCE INFLATION · ACTUAL VS FORECAST"
            "</div>",
            unsafe_allow_html=True,
        )

        chart_df = forecasts["chart_df"].copy()

        history = chart_df[
            chart_df["pce_inflation"].notna()
        ]

        forecast = chart_df[
            chart_df["forecast"].notna()
        ]

        fig = go.Figure()

        # Historical PCE
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["pce_inflation"],
                mode="lines",
                name="PCE inflation",
                line=dict(
                    color="#f4b942",
                    width=2.2,
                ),
            )
        )

        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["forecast"],
                mode="lines",
                name="Forecast",
                line=dict(
                    color="#4fa3ff",
                    width=2.2,
                    dash="dash",
                ),
            )
        )

        # Prediction interval upper
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["upper"],
                mode="lines",
                line=dict(
                    width=0,
                    color="rgba(79,163,255,0)",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Prediction interval lower
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(79,163,255,0.10)",
                line=dict(
                    width=0,
                    color="rgba(79,163,255,0)",
                ),
                name="Prediction range",
                hoverinfo="skip",
            )
        )

        # Fed target
        fig.add_hline(
            y=2.0,
            line_dash="dot",
            line_color="#7d8998",
            annotation_text="Fed 2%",
            annotation_position="top left",
        )

        fig.update_layout(
            height=450,
            margin=dict(
                l=10,
                r=10,
                t=15,
                b=10,
            ),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(
                color="#c9d2dd",
            ),
            legend=dict(
                orientation="h",
                y=1.04,
                x=0,
            ),
            hovermode="x unified",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="% YoY",
                gridcolor="#1c2530",
                zeroline=False,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        add_source_note(
            "Source: BEA PCE via FRED. Historical series through the configured data vintage."
        )

    with right:

        st.markdown(
            '<div class="section-label">INFLATION SNAPSHOT</div>',
            unsafe_allow_html=True,
        )

        st.metric(
            "Inflation pressure",
            f'{forecasts["pressure_score"]}/100',
        )

        st.metric(
            "Current regime",
            forecasts["regime"],
        )

        st.metric(
            "6M forecast",
            f'{forecasts["point_forecast_6m"]:.2f}%',
        )

        st.metric(
            "Model confidence",
            forecasts["confidence"],
        )

        st.markdown(
            '<div class="small-note">'
            "Confidence is a model-quality label based on recent out-of-sample "
            "forecasting performance. It is not the probability that the forecast is correct."
            "</div>",
            unsafe_allow_html=True,
        )

    # ---------- MACRO BRIEF ----------

    st.markdown(
        '<div class="section-label">AI MACRO BRIEF</div>',
        unsafe_allow_html=True,
    )

    st.info(
        forecasts["macro_brief"]
    )


# ============================================================
# FORECAST PAGE
# ============================================================

elif page == "Forecast":

    st.markdown(
        '<div class="terminal-title">FORECAST</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="terminal-subtitle">'
        "Model comparison, forecast ranges, and out-of-sample performance."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">FORECAST OUTPUT</div>',
        unsafe_allow_html=True,
    )

    forecast_table = forecasts["forecast_table"].copy()

    forecast_table["Forecast"] = forecast_table["Forecast"].round(2)

    if "Lower" in forecast_table.columns:
        forecast_table["Lower"] = forecast_table["Lower"].round(2)

    if "Upper" in forecast_table.columns:
        forecast_table["Upper"] = forecast_table["Upper"].round(2)

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-label">MODEL PERFORMANCE</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        diagnostics["performance"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-label">PREDICTION INTERVAL CALIBRATION</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        diagnostics["coverage"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "SARIMAX uses its model-based prediction interval. "
        "XGBoost uses temporal conformal calibration based on historical "
        "walk-forward errors. These are prediction intervals, not confidence intervals."
    )


# ============================================================
# DRIVERS PAGE
# ============================================================

elif page == "Drivers":

    st.markdown(
        '<div class="terminal-title">DRIVERS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="terminal-subtitle">'
        "The variables contributing most strongly to the model forecast."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">MODEL CONTRIBUTIONS</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        drivers,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Feature importance is not the same as causal impact. "
        "These values describe the model's contribution/usefulness."
    )

    # ---------- SHAP-LIKE VISUAL ----------

    if not drivers.empty:

        plot_drivers = drivers.copy()

        fig = go.Figure(
            go.Bar(
                x=plot_drivers["Relative strength"],
                y=plot_drivers["Driver"],
                orientation="h",
                marker=dict(
                    color="#4fa3ff",
                ),
            )
        )

        fig.update_layout(
            height=350,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(
                color="#c9d2dd",
            ),
            xaxis=dict(
                title="Relative model importance",
                gridcolor="#1c2530",
            ),
            yaxis=dict(
                showgrid=False,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ---------- REGIME ----------

    st.markdown(
        '<div class="section-label">INFLATION REGIME</div>',
        unsafe_allow_html=True,
    )

    regime_df = forecasts["regime_history"].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=regime_df.index,
            y=regime_df["pce_inflation"],
            mode="lines",
            name="PCE inflation",
            line=dict(
                color="#f4b942",
                width=2,
            ),
        )
    )

    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#7d8998",
        annotation_text="Fed 2%",
    )

    fig.update_layout(
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(
            color="#c9d2dd",
        ),
        xaxis=dict(
            showgrid=False,
        ),
        yaxis=dict(
            title="% YoY",
            gridcolor="#1c2530",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-label">PLAIN-ENGLISH VIEW</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"""
        Current PCE inflation is **{snapshot["pce_inflation"]:.2f}%**,
        while the model's 3-month forecast is
        **{forecasts["point_forecast_3m"]:.2f}%**.

        The current inflation environment is classified as
        **{forecasts["regime"].lower()}**.

        The inflation-pressure score is **{forecasts["pressure_score"]}/100**.
        """
    )


# ============================================================
# SCENARIOS PAGE
# ============================================================

elif page == "Scenarios":

    st.markdown(
        '<div class="terminal-title">SCENARIO LAB</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="terminal-subtitle">'
        "Simple what-if analysis for the major inflation risks."
        "</div>",
        unsafe_allow_html=True,
    )

    st.warning(
        "Scenario outputs are model estimates, not official Federal Reserve forecasts."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        oil_change = st.slider(
            "Oil price change",
            min_value=-30,
            max_value=30,
            value=0,
            step=5,
            format="%d%%",
        )

    with col2:
        expectations_change = st.slider(
            "Inflation expectations change",
            min_value=-100,
            max_value=100,
            value=0,
            step=25,
            format="%d bps",
        )

    with col3:
        demand_change = st.slider(
            "Demand / consumption shock",
            min_value=-10,
            max_value=10,
            value=0,
            step=1,
            format="%d%%",
        )

    # Simple transparent scenario approximation.
    # The final scenario engine should replace these coefficients
    # once backtested elasticities are established.
    oil_effect = oil_change * 0.006
    expectations_effect = expectations_change * 0.002
    demand_effect = demand_change * 0.015

    scenario_forecast = (
        forecasts["point_forecast_3m"]
        + oil_effect
        + expectations_effect
        + demand_effect
    )

    st.markdown(
        '<div class="section-label">SCENARIO RESULT</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Baseline 3M CPI",
        f'{forecasts["point_forecast_3m"]:.2f}%',
    )

    s2.metric(
        "Scenario CPI",
        f"{scenario_forecast:.2f}%",
    )

    s3.metric(
        "Change",
        f"{scenario_forecast - forecasts['point_forecast_3m']:+.2f} pp",
    )

    scenario_table = pd.DataFrame(
        {
            "Scenario": [
                "Baseline",
                "Oil shock",
                "Expectations shock",
                "Demand shock",
                "Combined scenario",
            ],
            "3M CPI": [
                forecasts["point_forecast_3m"],
                forecasts["point_forecast_3m"] + oil_effect,
                forecasts["point_forecast_3m"] + expectations_effect,
                forecasts["point_forecast_3m"] + demand_effect,
                scenario_forecast,
            ],
        }
    )

    scenario_table["3M CPI"] = scenario_table["3M CPI"].round(2)

    st.dataframe(
        scenario_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The current scenario panel is intentionally transparent and simple. "
        "Its coefficients must be replaced by empirically estimated scenario elasticities "
        "before being treated as production-grade."
    )


# ============================================================
# DATA PAGE
# ============================================================

elif page == "Data":

    st.markdown(
        '<div class="terminal-title">DATA FOUNDATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="terminal-subtitle">'
        "Series metadata, latest observations, and data provenance."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">SERIES STATUS</div>',
        unsafe_allow_html=True,
    )

    status = snapshot["series_status"].copy()

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-label">DATASET</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"""
        **Historical window:** {START_DATE} to {DATA_CUTOFF}

        **Rows:** {len(data):,}

        **Columns:** {len(data.columns):,}

        **Latest usable PCE inflation observation:** {snapshot["latest_pce_date"]}
        """
    )

    st.markdown(
        '<div class="section-label">SOURCES</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        - **BEA:** PCE, Core PCE, Real Personal Consumption Expenditures
        - **BLS:** CPI Shelter, Unemployment Rate
        - **EIA / FRED:** WTI crude oil
        - **University of Michigan:** Inflation expectations
        """
    )

    st.caption(
        "FRED is used as the common API access layer. "
        "Underlying source agencies are retained as provenance."
    )

    st.download_button(
        "Download processed dataset",
        data=data.to_csv().encode("utf-8"),
        file_name="us_inflation_processed_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="small-note">'
    "US Inflation Intelligence · Research tool only · "
    "Not an official Federal Reserve forecast and not investment advice."
    "</div>",
    unsafe_allow_html=True,
)
