from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from data import (
    DATA_CUTOFF,
    START_DATE,
    fetch_model_data,
)
from model import (
    build_forecasts,
    driver_table,
    model_diagnostics,
)


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
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background-color: #0b0f14;
        color: #e8edf3;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 5.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stHeader"] {
        background-color: #0b0f14 !important;
        border-bottom: 1px solid #28313d !important;
    }

    [data-testid="stToolbar"] {
        background-color: #0b0f14 !important;
    }

    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background-color: #111720 !important;
        border-right: 1px solid #293340 !important;
    }

    [data-testid="stSidebar"] * {
        color: #dce4ee !important;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem !important;
    }

    /* =========================
       TITLES
       ========================= */

    .terminal-title {
        font-size: 2rem;
        line-height: 1.15;
        font-weight: 700;
        color: #f3f6fa;
        margin: 0;
    }

    .terminal-subtitle {
        font-size: 0.9rem;
        color: #8f9baa;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
    }

    .section {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        color: #9aa6b5;
        margin-top: 1rem;
        margin-bottom: 0.45rem;
    }

    .statusline {
        font-size: 0.74rem;
        color: #7e8a99;
        border-top: 1px solid #28313d;
        border-bottom: 1px solid #28313d;
        padding: 0.6rem 0;
        margin-bottom: 1rem;
    }

    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {
        background-color: #121923;
        border: 1px solid #28313d;
        border-radius: 4px;
        padding: 0.7rem;
        min-height: 100px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa6b5 !important;
        font-size: 0.74rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f6fa !important;
        font-size: 1.45rem !important;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        background-color: #151d28;
        color: #dce4ee;
        border: 1px solid #35404d;
        border-radius: 4px;
    }

    .stButton > button:hover {
        background-color: #1a2430;
        border-color: #657487;
        color: #ffffff;
    }

    /* =========================
       TABLES
       ========================= */

    [data-testid="stDataFrame"] {
        border: 1px solid #28313d;
        border-radius: 3px;
    }

    /* =========================
       INFO / WARNING
       ========================= */

    .stAlert {
        border-radius: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API CHECK
# ============================================================

if not os.getenv("FRED_API_KEY"):
    st.error(
        "FRED_API_KEY is missing. "
        "Add it to Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## US INFLATION INTELLIGENCE"
)

st.sidebar.caption(
    "Macro research terminal"
)

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

st.sidebar.caption(
    "Data access: FRED API"
)

if st.sidebar.button(
    "Refresh FRED data",
    use_container_width=True,
):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def load_data():
    return fetch_model_data()


try:

    with st.spinner(
        "Loading FRED data..."
    ):
        data = load_data()

except Exception as exc:

    st.error(
        f"Data pipeline failed: {exc}"
    )

    st.stop()


# ============================================================
# DIRECT SNAPSHOT
# ============================================================

try:

    if "pce_inflation" not in data.columns:

        st.error(
            "PCE inflation column is missing."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(data.columns)
        )

        st.stop()

    target_data = data.dropna(
        subset=["pce_inflation"]
    )

    if target_data.empty:

        st.error(
            "No usable PCE inflation observations."
        )

        st.stop()

    latest_row = (
        target_data.iloc[-1]
    )

    latest_date = (
        target_data.index[-1]
    )

    current_pce = float(
        latest_row[
            "pce_inflation"
        ]
    )

    current_core = float(
        latest_row[
            "core_pce_inflation"
        ]
    )

except Exception as exc:

    st.error(
        f"Unable to read latest PCE data: {exc}"
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(data.columns)
    )

    st.stop()


# ============================================================
# RUN FORECASTING
# ============================================================

try:

    with st.spinner(
        "Running forecast models..."
    ):

        forecasts = build_forecasts(
            data,
            fast_mode=True,
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
        """
        <div class="terminal-title">
            US INFLATION INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="terminal-subtitle">
            Is U.S. inflation moving sustainably toward
            the Federal Reserve's 2% objective?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="statusline">
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
    # CHART
    # ========================================================

    st.markdown(
        """
        <div class="section">
            PCE INFLATION · ACTUAL VS FORECAST
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart = forecasts[
        "chart_df"
    ]

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
            y=history[
                "pce_inflation"
            ],
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
            y=future[
                "forecast"
            ],
            mode="lines",
            name="Forecast",
            line=dict(
                color="#4fa3ff",
                width=2.2,
                dash="dash",
            ),
        )
    )

    # Upper interval
    fig.add_trace(
        go.Scatter(
            x=future.index,
            y=future[
                "upper"
            ],
            mode="lines",
            line=dict(
                width=0,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lower interval
    fig.add_trace(
        go.Scatter(
            x=future.index,
            y=future[
                "lower"
            ],
            mode="lines",
            fill="tonexty",
            fillcolor=(
                "rgba(79,163,255,0.12)"
            ),
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
        line_color="#7d8794",
        annotation_text="Fed 2%",
        annotation_position="top left",
    )

    fig.update_layout(
        height=460,
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
        legend=dict(
            orientation="h",
            y=1.05,
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

    st.caption(
        "Source: BEA PCE via FRED. "
        "Forecast interval: temporal conformal calibration."
    )


    # ========================================================
    # STATE + BRIEF
    # ========================================================

    left, right = st.columns(
        [1.5, 1]
    )

    with left:

        st.markdown(
            """
            <div class="section">
                MACRO BRIEF
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            forecasts[
                "macro_brief"
            ]
        )

    with right:

        st.markdown(
            """
            <div class="section">
                CURRENT STATE
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Inflation pressure",
            f'{forecasts["pressure_score"]}/100',
        )

        st.metric(
            "Inflation regime",
            forecasts["regime"],
        )

        st.metric(
            "Model confidence",
            forecasts["confidence"],
        )

        st.metric(
            "Distance to 2%",
            f'{forecasts["point_forecast_3m"] - 2:+.2f} pp',
        )


# ============================================================
# FORECAST PAGE
# ============================================================

elif page == "Forecast":

    st.markdown(
        """
        <div class="terminal-title">
            FORECAST
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="terminal-subtitle">
            Econometric benchmark vs machine-learning model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section">
            FORECAST OUTPUT
        </div>
        """,
        unsafe_allow_html=True,
    )

    forecast_table = (
        forecasts[
            "forecast_table"
        ].copy()
    )

    for column in [
        "Forecast",
        "Lower",
        "Upper",
    ]:

        if column in forecast_table.columns:

            forecast_table[
                column
            ] = forecast_table[
                column
            ].round(2)

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        """
        <div class="section">
            MODEL PERFORMANCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    diagnostics = model_diagnostics(
        forecasts
    )

    st.dataframe(
        diagnostics[
            "performance"
        ],
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        """
        <div class="section">
            PREDICTION INTERVAL CALIBRATION
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        diagnostics[
            "coverage"
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "XGBoost uncertainty is estimated using temporal conformal "
        "calibration. Prediction intervals are not confidence intervals."
    )


# ============================================================
# DRIVERS PAGE
# ============================================================

elif page == "Drivers":

    st.markdown(
        """
        <div class="terminal-title">
            DRIVERS
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="terminal-subtitle">
            Which variables matter most to the model?
        </div>
        """,
        unsafe_allow_html=True,
    )

    drivers = driver_table(
        forecasts
    )

    if not drivers.empty:

        st.markdown(
            """
            <div class="section">
                MODEL IMPORTANCE
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            height=380,
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
                title="Relative importance",
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

        st.caption(
            "Feature importance describes model behavior. "
            "It does not establish causation."
        )


    st.markdown(
        """
        <div class="section">
            INFLATION REGIME
        </div>
        """,
        unsafe_allow_html=True,
    )

    regime_df = forecasts[
        "regime_history"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=regime_df.index,
            y=regime_df[
                "pce_inflation"
            ],
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
        line_color="#7d8794",
        annotation_text="Fed 2%",
    )

    fig.update_layout(
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
        fig,
        use_container_width=True,
    )


# ============================================================
# DATA PAGE
# ============================================================

elif page == "Data":

    st.markdown(
        """
        <div class="terminal-title">
            DATA
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="terminal-subtitle">
            FRED data currently used by the forecasting pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section">
            DATASET
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Rows",
        f"{len(data):,}",
    )

    d2.metric(
        "Columns",
        f"{len(data.columns):,}",
    )

    d3.metric(
        "Latest date",
        data.index[-1].strftime(
            "%Y-%m"
        ),
    )

    st.dataframe(
        data.tail(25),
        use_container_width=True,
    )

    st.download_button(
        "Download processed dataset",
        data=data.to_csv().encode(
            "utf-8"
        ),
        file_name=(
            "us_inflation_data.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "US Inflation Intelligence · Research / portfolio tool · "
    "Not an official Federal Reserve forecast or investment advice."
)
