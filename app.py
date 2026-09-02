from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

from data import (
    DATA_CUTOFF,
    START_DATE,
    fetch_model_data,
    latest_status,
)
from model import (
    build_forecasts,
    driver_table,
    model_diagnostics,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="US Inflation Intelligence",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       BASE
       ------------------------------------------------------- */

    .stApp {
        background: #0b0f14;
        color: #e8edf3;
    }

    .main .block-container {
        max-width: 1520px;
        padding-top: 4.3rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }

    [data-testid="stHeader"] {
        background: #0b0f14 !important;
        border-bottom: 1px solid #27303b !important;
    }

    [data-testid="stToolbar"] {
        background: #0b0f14 !important;
    }

    /* -------------------------------------------------------
       NAVIGATION
       ------------------------------------------------------- */

    .brand {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: .04em;
        color: #f3b340;
        margin-bottom: .05rem;
    }

    .brand-sub {
        font-size: .68rem;
        letter-spacing: .14em;
        color: #778494;
        text-transform: uppercase;
    }

    .nav-wrap {
        border-top: 1px solid #27303b;
        border-bottom: 1px solid #27303b;
        padding: .4rem 0;
        margin: .7rem 0 .8rem;
    }

    /* -------------------------------------------------------
       TYPOGRAPHY
       ------------------------------------------------------- */

    .eyebrow {
        color: #798697;
        font-size: .68rem;
        letter-spacing: .13em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .page-title {
        font-size: 1.9rem;
        line-height: 1.1;
        font-weight: 750;
        color: #f4f7fa;
        margin-top: .15rem;
    }

    .page-subtitle {
        color: #8b98a8;
        font-size: .83rem;
        margin-top: .25rem;
    }

    .terminal-line {
        font-size: .72rem;
        color: #697686;
        padding: .45rem 0;
        border-bottom: 1px solid #27303b;
        margin-bottom: .9rem;
    }

    .section-label {
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .13em;
        color: #9aa7b6;
        text-transform: uppercase;
        margin: .9rem 0 .45rem;
    }

    /* -------------------------------------------------------
       KPI
       ------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: #111720;
        border: 1px solid #29333f;
        border-radius: 3px;
        min-height: 104px;
        padding: .65rem .75rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #8d99a7 !important;
        font-size: .71rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f6f9 !important;
        font-size: 1.42rem !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: .69rem !important;
    }

    /* -------------------------------------------------------
       BUTTON NAV
       ------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        background: #10161e;
        border: 1px solid #2b3541;
        color: #aeb9c7;
        border-radius: 3px;
        min-height: 2rem;
        font-size: .7rem;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #7b8898;
        color: #ffffff;
        background: #151c25;
    }

    /* -------------------------------------------------------
       TABLES / ALERTS
       ------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #29333f;
        border-radius: 3px;
    }

    .stAlert {
        border-radius: 3px;
    }

    /* -------------------------------------------------------
       FOOTER
       ------------------------------------------------------- */

    .footer {
        border-top: 1px solid #27303b;
        margin-top: 1.2rem;
        padding-top: .65rem;
        font-size: .65rem;
        color: #657281;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def set_page(name: str) -> None:
    st.session_state["page"] = name


if "page" not in st.session_state:
    st.session_state["page"] = "Overview"


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [2.8, 1.2]
)

with header_left:
    st.markdown(
        '<div class="brand">US INFLATION INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-sub">Macro Research Terminal</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;color:#7d8997;font-size:.68rem;padding-top:.35rem;">'
        'DATA VINTAGE&nbsp;&nbsp;'
        f'<span style="color:#f3b340;">{DATA_CUTOFF}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TOP NAV
# ============================================================

st.markdown(
    '<div class="nav-wrap"></div>',
    unsafe_allow_html=True,
)

nav = st.columns(4)

pages = [
    ("Overview", "OVERVIEW"),
    ("Forecast", "FORECAST"),
    ("Drivers", "DRIVERS"),
    ("Data", "DATA"),
]

for col, (internal, label) in zip(
    nav,
    pages,
):
    with col:
        if st.button(
            label,
            key=f"nav_{internal}",
        ):
            set_page(internal)
            st.rerun()


# ============================================================
# API
# ============================================================

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED_API_KEY is missing. Add it under "
        "Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ============================================================
# DATA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def load_data():
    return fetch_model_data()


try:
    with st.spinner("Updating economic data..."):
        data = load_data()
except Exception as exc:
    st.error(
        f"Data pipeline failed: {exc}"
    )
    st.stop()


# ============================================================
# REFRESH
# ============================================================

refresh_col, status_col = st.columns(
    [1, 5]
)

with refresh_col:
    if st.button(
        "REFRESH DATA",
        key="refresh_data",
    ):
        load_data.clear()
        st.rerun()

with status_col:
    st.markdown(
        f'<div class="terminal-line">'
        f'WINDOW {START_DATE} → {DATA_CUTOFF}'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'{len(data):,} MONTHLY ROWS'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'FRED API ONLINE'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST ENGINE
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def run_models(frame):
    result = build_forecasts(
        frame,
        fast_mode=True,
    )
    return result


try:
    with st.spinner(
        "Running inflation models..."
    ):
        forecasts = run_models(
            data
        )
except Exception as exc:
    st.error(
        "Forecasting pipeline failed."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# COMMON CURRENT STATE
# ============================================================

target = data.dropna(
    subset=["pce_inflation"]
)

latest = target.iloc[-1]
latest_date = target.index[-1]

current_pce = float(
    latest["pce_inflation"]
)

current_core = float(
    latest["core_pce_inflation"]
)

# ============================================================
# OVERVIEW
# ============================================================

if st.session_state["page"] == "Overview":

    st.markdown(
        '<div class="eyebrow">US MACRO · PCE INFLATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Inflation at a glance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Where is inflation heading, what is driving it, and how far is it from 2%?"
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="terminal-line">'
        f'LATEST OBSERVATION {latest_date.strftime("%b %Y")}'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'PRIMARY FORECAST {forecasts["point_forecast_3m"]:.2f}% / 3M'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'CURRENT REGIME {forecasts["regime"].upper()}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "PCE",
        f"{current_pce:.2f}%",
    )

    c2.metric(
        "Core PCE",
        f"{current_core:.2f}%",
    )

    c3.metric(
        "Fed Target",
        "2.00%",
    )

    c4.metric(
        "3M Forecast",
        f'{forecasts["point_forecast_3m"]:.2f}%',
        f'{forecasts["point_forecast_3m"] - current_pce:+.2f} pp',
    )

    c5.metric(
        "6M Forecast",
        f'{forecasts["point_forecast_6m"]:.2f}%',
    )

    c6.metric(
        "Pressure",
        f'{forecasts["pressure_score"]}/100',
        forecasts["confidence"],
    )

    left, right = st.columns(
        [2.25, 1]
    )

    # Main chart
    with left:

        st.markdown(
            '<div class="section-label">'
            'PCE · ACTUAL / FORECAST'
            '</div>',
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

        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["pce_inflation"],
                mode="lines",
                name="Actual",
                line=dict(
                    color="#f3b340",
                    width=2.4,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future.index,
                y=future["forecast"],
                mode="lines",
                name="Forecast",
                line=dict(
                    color="#55a7ff",
                    width=2.2,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future.index,
                y=future["upper"],
                mode="lines",
                line=dict(
                    width=0,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future.index,
                y=future["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(85,167,255,0.10)",
                line=dict(
                    width=0,
                ),
                name="80% range",
            )
        )

        fig.add_hline(
            y=2.0,
            line_dash="dot",
            line_color="#747f8d",
            annotation_text="Fed 2%",
            annotation_position="top left",
        )

        fig.update_layout(
            height=440,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(
                color="#cbd4de"
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.03,
                x=0,
            ),
            xaxis=dict(
                showgrid=False,
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

    with right:

        st.markdown(
            '<div class="section-label">'
            'CURRENT STATE'
            '</div>',
            unsafe_allow_html=True,
        )

        st.metric(
            "Inflation Regime",
            forecasts["regime"],
        )

        st.metric(
            "Distance to 2%",
            f'{forecasts["point_forecast_3m"] - 2:+.2f} pp',
        )

        st.metric(
            "Model Confidence",
            forecasts["confidence"],
        )

        st.markdown(
            '<div class="section-label">MODEL MIX</div>',
            unsafe_allow_html=True,
        )

        # Model mix is calculated safely from the forecast object.
        weights = forecasts.get(
            "ensemble_weights",
            {"XGBoost": 0.50, "SARIMAX": 0.50},
        )

        xgb_weight = float(
            weights.get("XGBoost", 0.50)
        )

        sarimax_weight = float(
            weights.get("SARIMAX", 0.50)
        )

        st.write(
            f"XGBoost  **{xgb_weight:.0%}**"
        )

        st.write(
            f"SARIMAX  **{sarimax_weight:.0%}**"
        )

    st.markdown(
        '<div class="section-label">RESEARCH VIEW</div>',
        unsafe_allow_html=True,
    )

    st.info(
        forecasts["macro_brief"]
    )


# ============================================================
# FORECAST
# ============================================================

elif st.session_state["page"] == "Forecast":

    st.markdown(
        '<div class="eyebrow">FORECAST ENGINE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Forward inflation path</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Direct 1–6 month forecasts from a compact econometric + ML ensemble."
        '</div>',
        unsafe_allow_html=True,
    )

    table = forecasts[
        "forecast_table"
    ].copy()

    table["Forecast"] = table[
        "Forecast"
    ].round(2)

    table["Lower"] = table[
        "Lower"
    ].round(2)

    table["Upper"] = table[
        "Upper"
    ].round(2)

    st.markdown(
        '<div class="section-label">FORECAST CURVE</div>',
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3 = st.columns(3)

    fc1.metric(
        "1M",
        f'{table.iloc[0]["Forecast"]:.2f}%',
    )

    fc2.metric(
        "3M",
        f'{table.iloc[2]["Forecast"]:.2f}%',
    )

    fc3.metric(
        "6M",
        f'{table.iloc[5]["Forecast"]:.2f}%',
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    diagnostics = model_diagnostics(
        forecasts
    )

    st.markdown(
        '<div class="section-label">MODEL PERFORMANCE · 24M CHRONOLOGICAL HOLDOUT</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        diagnostics["performance"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-label">UNCERTAINTY CALIBRATION</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        diagnostics["coverage"],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "XGBoost prediction ranges use temporal conformal calibration. "
        "They are prediction intervals, not confidence intervals."
    )


# ============================================================
# DRIVERS
# ============================================================

elif st.session_state["page"] == "Drivers":

    st.markdown(
        '<div class="eyebrow">MODEL EXPLAINABILITY</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">What is driving the forecast?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Feature importance from the primary 3-month XGBoost model."
        '</div>',
        unsafe_allow_html=True,
    )

    drivers = driver_table(
        forecasts
    )

    left, right = st.columns(
        [1.3, 1]
    )

    with left:

        st.markdown(
            '<div class="section-label">MODEL IMPORTANCE</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            drivers,
            use_container_width=True,
            hide_index=True,
        )

        if not drivers.empty:

            plot = drivers.sort_values(
                "Importance"
            )

            fig = go.Figure(
                go.Bar(
                    x=plot["Importance"],
                    y=plot["Driver"],
                    orientation="h",
                    marker_color="#55a7ff",
                )
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5,
                ),
                paper_bgcolor="#0b0f14",
                plot_bgcolor="#0b0f14",
                font=dict(
                    color="#cbd4de"
                ),
                xaxis=dict(
                    title="Relative model importance",
                    gridcolor="#1c2530",
                ),
                yaxis=dict(
                    showgrid=False
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with right:

        st.markdown(
            '<div class="section-label">INFLATION REGIME</div>',
            unsafe_allow_html=True,
        )

        regime = forecasts[
            "regime_history"
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=regime.index,
                y=regime["pce_inflation"],
                mode="lines",
                name="PCE",
                line=dict(
                    color="#f3b340",
                    width=2.2,
                ),
            )
        )

        fig.add_hline(
            y=2.0,
            line_dash="dot",
            line_color="#747f8d",
            annotation_text="Fed 2%",
        )

        fig.update_layout(
            height=340,
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(
                color="#cbd4de"
            ),
            xaxis=dict(
                showgrid=False
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
            f"""
            **Current environment:** {forecasts["regime"]}

            **Pressure score:** {forecasts["pressure_score"]}/100

            **Model confidence:** {forecasts["confidence"]}
            """
        )

        st.caption(
            "Driver importance is model-based and does not establish economic causality."
        )


# ============================================================
# DATA
# ============================================================

elif st.session_state["page"] == "Data":

    st.markdown(
        '<div class="eyebrow">DATA & PROVENANCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Data foundation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Small, transparent data universe used by the forecast engine."
        '</div>',
        unsafe_allow_html=True,
    )

    status = latest_status(
        data
    )

    st.markdown(
        '<div class="section-label">SOURCE STATUS</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-label">DATASET</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{len(data):,}",
    )

    c2.metric(
        "Columns",
        f"{len(data.columns):,}",
    )

    c3.metric(
        "Latest PCE",
        latest_date.strftime("%b %Y"),
    )

    st.download_button(
        "DOWNLOAD PROCESSED DATA",
        data=data.to_csv().encode(
            "utf-8"
        ),
        file_name="us_inflation_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-label">RAW MODEL DATA</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        data.tail(24),
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'US Inflation Intelligence · Research / portfolio tool · '
    'Not an official Federal Reserve forecast or investment advice.'
    '</div>',
    unsafe_allow_html=True,
)
