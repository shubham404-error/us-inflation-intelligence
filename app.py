from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

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

load_dotenv()

st.set_page_config(
    page_title="US Inflation Intelligence",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:#0b0f14;
        color:#e8edf3;
    }
    .main .block-container {
        max-width:1520px;
        padding-top:4.5rem !important;
        padding-bottom:2rem !important;
        padding-left:1.5rem !important;
        padding-right:1.5rem !important;
    }
    [data-testid="stHeader"] {
        background:#0b0f14 !important;
        border-bottom:1px solid #27303b !important;
    }
    .brand {
        color:#f3b340;
        font-size:1.2rem;
        font-weight:800;
        letter-spacing:.04em;
    }
    .brand-sub {
        color:#748191;
        font-size:.65rem;
        letter-spacing:.14em;
        text-transform:uppercase;
    }
    .page-title {
        color:#f4f7fa;
        font-size:1.85rem;
        font-weight:750;
        margin-top:.15rem;
    }
    .eyebrow,.section-label {
        color:#96a3b2;
        font-size:.68rem;
        letter-spacing:.13em;
        text-transform:uppercase;
        font-weight:800;
    }
    .page-subtitle {
        color:#8b98a8;
        font-size:.82rem;
        margin:.2rem 0 .85rem;
    }
    .terminal-line {
        color:#697686;
        font-size:.7rem;
        padding:.5rem 0;
        border-top:1px solid #27303b;
        border-bottom:1px solid #27303b;
        margin-bottom:.9rem;
    }
    div[data-testid="stMetric"] {
        background:#111720;
        border:1px solid #29333f;
        border-radius:3px;
        min-height:100px;
        padding:.65rem .75rem;
    }
    div[data-testid="stMetricLabel"] {
        color:#8996a5 !important;
        font-size:.7rem !important;
    }
    div[data-testid="stMetricValue"] {
        color:#f3f6f9 !important;
        font-size:1.4rem !important;
    }
    .stButton > button {
        background:#10161e;
        border:1px solid #2b3541;
        color:#aeb9c7;
        border-radius:3px;
        font-size:.68rem;
        font-weight:700;
        letter-spacing:.08em;
    }
    .stButton > button:hover {
        background:#151c25;
        border-color:#6d7b8c;
        color:#fff;
    }
    [data-testid="stDataFrame"] {
        border:1px solid #29333f;
        border-radius:3px;
    }
    .footer {
        color:#657281;
        font-size:.64rem;
        border-top:1px solid #27303b;
        padding-top:.6rem;
        margin-top:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"


def go(page: str):
    st.session_state["page"] = page


# Header
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        '<div class="brand">US INFLATION INTELLIGENCE</div>'
        '<div class="brand-sub">Macro Research Terminal</div>',
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f'<div style="text-align:right;color:#7d8997;font-size:.67rem;">'
        f'DATA VINTAGE<br><span style="color:#f3b340;">{DATA_CUTOFF}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

nav = st.columns(4)
for col, name in zip(
    nav,
    ["Overview", "Forecast", "Drivers", "Data"],
):
    with col:
        if st.button(
            name.upper(),
            key=f"nav_{name}",
            use_container_width=True,
        ):
            go(name)
            st.rerun()

st.markdown(
    f'<div class="terminal-line">'
    f'WINDOW {START_DATE} → {DATA_CUTOFF}'
    f' &nbsp;|&nbsp; FRED API'
    f' &nbsp;|&nbsp; MODEL SCHEMA 0.5'
    f'</div>',
    unsafe_allow_html=True,
)

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED_API_KEY is missing. Add it under Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return fetch_model_data()


@st.cache_data(ttl=3600, show_spinner=False)
def run_models(frame):
    return build_forecasts(
        frame,
        fast_mode=True,
    )


if st.button("REFRESH DATA", key="refresh"):
    load_data.clear()
    run_models.clear()
    st.rerun()

try:
    data = load_data()
except Exception as exc:
    st.error(f"Data pipeline failed: {exc}")
    st.stop()

try:
    forecasts = run_models(data)
except Exception as exc:
    st.error("Forecasting pipeline failed.")
    st.exception(exc)
    st.stop()


# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------

if st.session_state["page"] == "Overview":

    st.markdown(
        '<div class="eyebrow">US MACRO · PCE INFLATION</div>'
        '<div class="page-title">Inflation at a glance</div>'
        '<div class="page-subtitle">'
        "Where is inflation heading, what is driving it, and how far is it from 2%?"
        '</div>',
        unsafe_allow_html=True,
    )

    latest = data.dropna(
        subset=["pce_inflation", "core_pce_inflation"]
    ).iloc[-1]

    current_pce = float(latest["pce_inflation"])
    current_core = float(latest["core_pce_inflation"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("PCE", f"{current_pce:.2f}%")
    c2.metric("Core PCE", f"{current_core:.2f}%")
    c3.metric("Fed Target", "2.00%")
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

    left, right = st.columns([2.3, 1])

    with left:
        st.markdown(
            '<div class="section-label">PCE · ACTUAL / FORECAST</div>',
            unsafe_allow_html=True,
        )

        chart = forecasts["chart_df"]
        history = chart[chart["pce_inflation"].notna()]
        future = chart[chart["forecast"].notna()]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["pce_inflation"],
                name="Actual",
                mode="lines",
                line=dict(
                    color="#f3b340",
                    width=2.3,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future.index,
                y=future["forecast"],
                name="Forecast",
                mode="lines",
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
                line=dict(width=0),
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
                fillcolor="rgba(85,167,255,.10)",
                line=dict(width=0),
                name="Prediction range",
            )
        )

        fig.add_hline(
            y=2,
            line_dash="dot",
            line_color="#727e8d",
            annotation_text="Fed 2%",
        )

        fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(color="#cbd4de"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.04,
                x=0,
            ),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                title="% YoY",
                gridcolor="#1c2530",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.markdown(
            '<div class="section-label">CURRENT STATE</div>',
            unsafe_allow_html=True,
        )

        st.metric(
            "Regime",
            forecasts["regime"],
        )
        st.metric(
            "Distance to 2%",
            f'{forecasts["point_forecast_3m"] - 2:+.2f} pp',
        )
        st.metric(
            "Confidence",
            forecasts["confidence"],
        )

        st.markdown(
            '<div class="section-label">MODEL MIX</div>',
            unsafe_allow_html=True,
        )

        weights = forecasts["ensemble_weights"]
        st.write(
            f'XGBoost&nbsp;&nbsp; **{weights["XGBoost"]:.0%}**',
            unsafe_allow_html=True,
        )
        st.write(
            f'SARIMAX&nbsp;&nbsp; **{weights["SARIMAX"]:.0%}**',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">RESEARCH VIEW</div>',
        unsafe_allow_html=True,
    )
    st.info(forecasts["macro_brief"])


# ------------------------------------------------------------
# FORECAST
# ------------------------------------------------------------

elif st.session_state["page"] == "Forecast":

    st.markdown(
        '<div class="eyebrow">FORECAST ENGINE</div>'
        '<div class="page-title">Forward inflation path</div>'
        '<div class="page-subtitle">'
        "Direct monthly forecasts with an econometric + machine-learning ensemble."
        '</div>',
        unsafe_allow_html=True,
    )

    table = forecasts["forecast_table"].copy()
    for col in ["Forecast", "Lower", "Upper"]:
        table[col] = table[col].round(2)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    diagnostics = model_diagnostics(
        forecasts
    )

    st.markdown(
        '<div class="section-label">MODEL PERFORMANCE · CHRONOLOGICAL HOLDOUT</div>',
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
        "XGBoost uncertainty uses temporal conformal calibration. "
        "Prediction intervals are not confidence intervals."
    )


# ------------------------------------------------------------
# DRIVERS
# ------------------------------------------------------------

elif st.session_state["page"] == "Drivers":

    st.markdown(
        '<div class="eyebrow">EXPLAINABILITY</div>'
        '<div class="page-title">What is driving the forecast?</div>'
        '<div class="page-subtitle">'
        "The strongest features used by the 3-month XGBoost model."
        '</div>',
        unsafe_allow_html=True,
    )

    drivers = driver_table(
        forecasts
    )

    left, right = st.columns(
        [1.35, 1]
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
                height=380,
                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5,
                ),
                paper_bgcolor="#0b0f14",
                plot_bgcolor="#0b0f14",
                font=dict(color="#cbd4de"),
                xaxis=dict(
                    title="Relative importance",
                    gridcolor="#1c2530",
                ),
                yaxis=dict(showgrid=False),
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
            y=2,
            line_dash="dot",
            line_color="#727e8d",
            annotation_text="Fed 2%",
        )

        fig.update_layout(
            height=350,
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(color="#cbd4de"),
            xaxis=dict(showgrid=False),
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

            **Inflation pressure:** {forecasts["pressure_score"]}/100

            **Confidence:** {forecasts["confidence"]}
            """
        )

        st.caption(
            "Model importance describes model behavior. It does not prove causality."
        )


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

elif st.session_state["page"] == "Data":

    st.markdown(
        '<div class="eyebrow">DATA & PROVENANCE</div>'
        '<div class="page-title">Data foundation</div>'
        '<div class="page-subtitle">'
        "The small set of public economic series used by the model."
        '</div>',
        unsafe_allow_html=True,
    )

    status = latest_status(data)

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True,
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
        "Latest",
        data.index[-1].strftime("%b %Y"),
    )

    st.download_button(
        "DOWNLOAD PROCESSED DATA",
        data=data.to_csv().encode("utf-8"),
        file_name="us_inflation_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander(
        "View recent model data"
    ):
        st.dataframe(
            data.tail(24),
            use_container_width=True,
        )


st.markdown(
    '<div class="footer">'
    "US Inflation Intelligence · Research / portfolio tool · "
    "Not an official Federal Reserve forecast or investment advice."
    "</div>",
    unsafe_allow_html=True,
)
