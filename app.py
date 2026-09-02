from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data import DATA_CUTOFF, START_DATE, fetch_model_data
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
# STYLE
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
        padding-bottom: 2rem;
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
    }

    .subtitle {
        font-size: 0.9rem;
        color: #8f9baa;
        margin-bottom: 1rem;
    }

    .status {
        font-size: 0.75rem;
        color: #7e8a99;
        border-top: 1px solid #28313d;
        border-bottom: 1px solid #28313d;
        padding: 0.6rem 0;
        margin-bottom: 1rem;
    }

    .section {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        color: #9aa6b5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
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
# API KEY
# ============================================================

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED_API_KEY is missing. Add it to Streamlit Cloud Secrets."
    )
    st.stop()


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
        "Data",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.caption(
    f"Window: {START_DATE} → {DATA_CUTOFF}"
)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# DATA
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
# DIRECT SNAPSHOT
# ============================================================
# IMPORTANT:
# We DO NOT call latest_snapshot().
# We directly use the verified internal dataframe names.

try:

    target = data.dropna(
        subset=["pce_inflation"]
    )

    if target.empty:
        raise RuntimeError(
            "No PCE inflation observations available."
        )

    latest = target.iloc[-1]
    latest_date = target.index[-1]

    current_pce = float(
        latest["pce_inflation"]
    )

    current_core = float(
        latest["core_pce_inflation"]
    )

except Exception as exc:

    st.error(
        f"Unable to create dashboard snapshot: {exc}"
    )

    st.write("Columns returned by data.py:")
    st.write(list(data.columns))

    st.stop()


# ============================================================
# FORECAST
# ============================================================

try:

    with st.spinner(
        "Running forecasting models..."
    ):
        forecasts = build_forecasts(
            data
        )

except Exception as exc:

    st.error(
        "Forecasting pipeline failed."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# TERMINAL
# ============================================================

if page == "Terminal":

    st.markdown(
        '<div class="title">US INFLATION INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Is U.S. inflation moving sustainably toward the Federal Reserve's 2% objective?"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status">
            DATA VINTAGE: {DATA_CUTOFF}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            LATEST PCE: {latest_date.strftime("%Y-%m")}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            OBSERVATIONS: {len(data):,}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            MODEL: {forecasts["selected_model"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # KPI ROW
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "PCE inflation",
        f"{current_pce:.2f}%",
    )

    c2.metric(
        "Core PCE",
        f"{current_core:.2f}%",
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
        "6M forecast",
        f'{forecasts["point_forecast_6m"]:.2f}%',
    )


    # ========================================================
    # MAIN CHART
    # ========================================================

    st.markdown(
        '<div class="section">'
        "PCE INFLATION · ACTUAL VS FORECAST"
        "</div>",
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
                color="#f4b942",
                width=2.2,
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
                color="#4fa3ff",
                width=2.2,
                dash="dash",
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
            fillcolor="rgba(79,163,255,0.12)",
            line=dict(width=0),
            name="80% prediction range",
        )
    )

    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#7d8794",
        annotation_text="Fed 2%",
        annotation_position="top left",
    )

    fig.update_layout(
        height=450,
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(
            color="#c9d2dd"
        ),
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.05,
            x=0,
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


    # ========================================================
    # CURRENT STATE
    # ========================================================

    left, right = st.columns(
        [1.5, 1]
    )

    with left:

        st.markdown(
            '<div class="section">'
            "AI MACRO BRIEF"
            "</div>",
            unsafe_allow_html=True,
        )

        st.info(
            forecasts["macro_brief"]
        )

    with right:

        st.markdown(
            '<div class="section">'
            "CURRENT STATE"
            "</div>",
            unsafe_allow_html=True,
        )

        st.metric(
            "Inflation pressure",
            f'{forecasts["pressure_score"]}/100',
        )

        st.metric(
            "Regime",
            forecasts["regime"],
        )

        st.metric(
            "Confidence",
            forecasts["confidence"],
        )


# ============================================================
# FORECAST PAGE
# ============================================================

elif page == "Forecast":

    st.markdown(
        '<div class="title">FORECAST</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Econometric and machine-learning forecast comparison."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section">'
        "FORECAST OUTPUT"
        "</div>",
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


    st.markdown(
        '<div class="section">'
        "MODEL PERFORMANCE"
        "</div>",
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


    st.markdown(
        '<div class="section">'
        "PREDICTION INTERVAL CALIBRATION"
        "</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        diagnostics["coverage"],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DRIVERS PAGE
# ============================================================

elif page == "Drivers":

    st.markdown(
        '<div class="title">DRIVERS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "What matters most to the forecasting model?"
        "</div>",
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

        fig = go.Figure(
            go.Bar(
                x=drivers[
                    "Relative strength"
                ],
                y=drivers[
                    "Driver"
                ],
                orientation="h",
                marker=dict(
                    color="#4fa3ff"
                ),
            )
        )

        fig.update_layout(
            height=400,
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(
                color="#c9d2dd"
            ),
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
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


    st.markdown(
        '<div class="section">'
        "INFLATION REGIME"
        "</div>",
        unsafe_allow_html=True,
    )

    regime_df = forecasts[
        "regime_history"
    ]

    regime_fig = go.Figure()

    regime_fig.add_trace(
        go.Scatter(
            x=regime_df.index,
            y=regime_df[
                "pce_inflation"
            ],
            mode="lines",
            name="PCE",
            line=dict(
                color="#f4b942",
                width=2,
            ),
        )
    )

    regime_fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="#7d8794",
    )

    regime_fig.update_layout(
        height=350,
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(
            color="#c9d2dd"
        ),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
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
        regime_fig,
        use_container_width=True,
    )


# ============================================================
# DATA PAGE
# ============================================================

elif page == "Data":

    st.markdown(
        '<div class="title">DATA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Data actually returned by the FRED pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    st.write(
        f"Rows: {len(data):,}"
    )

    st.write(
        f"Columns: {len(data.columns):,}"
    )

    st.write(
        f"Latest observation: "
        f"{data.index[-1].strftime('%Y-%m')}"
    )

    st.dataframe(
        data.tail(20),
        use_container_width=True,
    )

    st.download_button(
        "Download processed data",
        data=data.to_csv().encode(
            "utf-8"
        ),
        file_name=(
            "us_inflation_data.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "US Inflation Intelligence · "
    "Research / portfolio tool · "
    "Not an official Federal Reserve forecast."
)
