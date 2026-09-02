from __future__ import annotations

import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from data import DATA_CUTOFF, build_dataset, data_quality, latest_metrics
from model import FED_TARGET, build_forecast
from ai import ask_gemini, default_brief


st.set_page_config(
    page_title="US Inflation Intelligence",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
    --bg: #0a0d12;
    --panel: #10151d;
    --panel2: #0d1219;
    --border: #26303d;
    --text: #eef2f7;
    --muted: #8b98a8;
    --accent: #72e0b2;
    --warn: #ffc857;
    --danger: #ff7b7b;
}
.stApp {
    background: var(--bg);
    color: var(--text);
}
.block-container {
    max-width: 1280px;
    padding: 1rem 0.9rem 4rem 0.9rem;
}
[data-testid="stHeader"] {
    background: rgba(10,13,18,0.92);
}
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
}
[data-testid="stMetricLabel"] {
    color: var(--muted);
}
[data-testid="stMetricValue"] {
    color: var(--text);
}
.terminal-title {
    font-size: 1.22rem;
    font-weight: 750;
    letter-spacing: 0.03em;
    margin: 0;
}
.terminal-subtitle {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 0.15rem;
}
.eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.68rem;
    font-weight: 700;
}
.hero {
    background: linear-gradient(145deg, #111821, #0d1218);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.1rem;
    margin: 0.65rem 0 0.9rem 0;
}
.hero-value {
    font-size: clamp(2.5rem, 8vw, 4.2rem);
    line-height: 0.95;
    font-weight: 800;
    letter-spacing: -0.05em;
}
.hero-label {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 0.35rem;
}
.signal {
    display: inline-block;
    padding: 0.28rem 0.55rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--accent);
    font-size: 0.72rem;
    margin-top: 0.7rem;
}
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 0.95rem;
    margin-bottom: 0.8rem;
}
.card-title {
    font-weight: 720;
    font-size: 0.9rem;
}
.card-note {
    color: var(--muted);
    font-size: 0.77rem;
    line-height: 1.45;
    margin-top: 0.35rem;
}
.ai-box {
    background: #0f1720;
    border: 1px solid #344253;
    border-radius: 16px;
    padding: 0.95rem;
}
.footer {
    color: #667385;
    font-size: 0.7rem;
    text-align: center;
    padding-top: 1.3rem;
}
@media (max-width: 600px) {
    .block-container {
        padding: 0.65rem 0.65rem 3rem 0.65rem;
    }
    .hero {
        padding: 0.95rem;
    }
    .card {
        padding: 0.8rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    return build_dataset()


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(data: pd.DataFrame):
    return build_forecast(data)


def fmt_pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1f}%"


def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame):
    hist = history[["date", "pce_inflation"]].tail(72).copy()
    hist["type"] = "Actual"

    future = pd.DataFrame(
        {
            "date": pd.date_range(
                start=hist["date"].max() + pd.offsets.MonthEnd(1),
                periods=len(forecast),
                freq="ME",
            ),
            "pce_inflation": forecast["point"].values,
            "lower_80": forecast["lower_80"].values,
            "upper_80": forecast["upper_80"].values,
            "type": "Forecast",
        }
    )

    chart_base = alt.Chart(future).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
    )

    band = chart_base.mark_area(opacity=0.18).encode(
        y=alt.Y("lower_80:Q", title="PCE inflation (%)"),
        y2="upper_80:Q",
    )

    forecast_line = chart_base.mark_line(strokeWidth=3).encode(
        y="pce_inflation:Q",
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("pce_inflation:Q", title="Forecast", format=".2f"),
            alt.Tooltip("lower_80:Q", title="80% low", format=".2f"),
            alt.Tooltip("upper_80:Q", title="80% high", format=".2f"),
        ],
    )

    actual_line = alt.Chart(hist).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("pce_inflation:Q", title="PCE inflation (%)"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("pce_inflation:Q", title="Actual", format=".2f"),
        ],
    )

    target = alt.Chart(
        pd.DataFrame({"target": [FED_TARGET]})
    ).mark_rule(strokeDash=[6, 4]).encode(y="target:Q")

    return (actual_line + band + forecast_line + target).properties(height=320)


def render_header(state):
    st.markdown(
        f"""
        <div>
          <div class="terminal-title">US INFLATION INTELLIGENCE</div>
          <div class="terminal-subtitle">PCE-focused inflation forecasting and plain-English macro analysis · cutoff {DATA_CUTOFF}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.segmented_control(
        "Navigate",
        ["Overview", "Forecast", "Drivers", "AI", "Data"],
        default="Overview",
        label_visibility="collapsed",
        key="page",
    )
    return page or "Overview"


def render_overview(data, result):
    state = result.state
    latest = latest_metrics(data)

    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Current inflation state</div>
          <div class="hero-value">{latest["pce"]:.1f}%</div>
          <div class="hero-label">PCE year-over-year inflation · Latest available observation</div>
          <div class="signal">{state["level"].upper()} · {state["momentum"].upper()}</div>
          <div class="card-note">{default_brief(result.chat_context)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Core PCE", fmt_pct(latest["core_pce"]))
    c2.metric("Fed target", "2.0%")
    c3.metric("6M outlook", fmt_pct(state["forecast_6m"]))

    st.markdown("### Inflation path")
    st.altair_chart(
        plot_forecast(result.history, result.forecast),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">What matters now</div>
              <div class="card-note">
                <b>State:</b> {state["level"]}<br>
                <b>Momentum:</b> {state["momentum"]}<br>
                <b>Outlook:</b> {state["outlook"]}<br>
                <b>Pressure:</b> {state["pressure_score"]:.0f}/100<br>
                <b>Confidence:</b> {state["confidence"]}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">Ask Inflation AI</div>
              <div class="card-note">
                Ask why inflation is high, whether it is moving toward 2%, what is driving
                the forecast, or what could make the outlook wrong.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open AI assistant", use_container_width=True):
            st.session_state["page"] = "AI"
            st.rerun()


def render_forecast(result):
    st.markdown("### Forecast")
    st.caption(
        "Six-month path from the final ensemble. Intervals are calibrated from historical walk-forward forecast errors."
    )

    display = result.forecast.copy()
    display["horizon"] = display["horizon"].map(lambda x: f"{x}M")
    display["point"] = display["point"].map(lambda x: f"{x:.2f}%")
    display["lower_80"] = display["lower_80"].map(lambda x: f"{x:.2f}%")
    display["upper_80"] = display["upper_80"].map(lambda x: f"{x:.2f}%")
    display.columns = ["Horizon", "Forecast", "80% Low", "80% High"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Model validation")
    metrics = result.metrics.copy()
    metrics["MAE"] = metrics["MAE"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    metrics["RMSE"] = metrics["RMSE"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    metrics["Bias"] = metrics["Bias"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    cal = result.calibration
    coverage = cal["empirical_coverage"]
    st.info(
        f"80% conformal target coverage: {coverage:.1%} on {cal['n']} walk-forward 3M forecasts. "
        f"Calibration radius: {cal['radius']:.2f} percentage points."
        if pd.notna(coverage)
        else "Not enough walk-forward forecasts for coverage diagnostics."
    )


def render_drivers(result):
    st.markdown("### What is driving the forecast?")
    st.caption(
        "Importance is a model signal, not proof of economic causation. It reflects the features the 3M XGBoost model relied on most."
    )

    drivers = result.drivers.copy()
    drivers["Importance %"] = drivers["Importance"] * 100
    st.bar_chart(drivers.set_index("Driver")["Importance %"], height=280)

    st.dataframe(
        drivers.assign(
            Importance=drivers["Importance"].map(lambda x: f"{x:.1%}"),
        )[["Driver", "Importance"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Market state")
    state = result.state
    a, b, c = st.columns(3)
    a.metric("Pressure", f'{state["pressure_score"]:.0f}/100')
    b.metric("Momentum", state["momentum"])
    c.metric("Outlook", state["outlook"])


def render_ai(result):
    st.markdown("### Ask Inflation AI")
    st.caption("Gemini 3.5 Flash. Grounded in the current application data and forecast.")

    suggestions = [
        "Why is inflation still above 2%?",
        "Will inflation reach 2% in the next six months?",
        "What is driving inflation right now?",
        "Explain the forecast like I am not an economist.",
    ]

    for q in suggestions:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []

    for message in st.session_state["ai_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about U.S. inflation...")
    pending = st.session_state.pop("pending_question", None)
    question = question or pending

    if question:
        st.session_state["ai_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing the latest model state..."):
                try:
                    answer = ask_gemini(
                        question,
                        result.chat_context,
                        st.session_state["ai_messages"][-7:],
                    )
                except Exception as exc:
                    answer = (
                        f"AI assistant is unavailable right now. {exc}"
                        if st.secrets.get("ENVIRONMENT", "local") != "production"
                        else "AI assistant is temporarily unavailable. Please check the Gemini API configuration."
                    )
                st.markdown(answer)

        st.session_state["ai_messages"].append(
            {"role": "assistant", "content": answer}
        )


def render_data(data, status):
    st.markdown("### Data")
    st.caption(
        "All model inputs are pulled through FRED. Underlying source-of-record labels are retained in the metadata table."
    )
    st.dataframe(status, use_container_width=True, hide_index=True)

    st.markdown("### Quality checks")
    st.dataframe(data_quality(data), use_container_width=True, hide_index=True)

    csv = data.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download processed data",
        csv,
        "us_inflation_intelligence_data.csv",
        "text/csv",
        use_container_width=True,
    )


def main():
    try:
        bundle = load_data()
        data = bundle.data
        status = bundle.status
        result = load_forecast(data)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    page = render_header(result.state)

    if page == "Overview":
        render_overview(data, result)
    elif page == "Forecast":
        render_forecast(result)
    elif page == "Drivers":
        render_drivers(result)
    elif page == "AI":
        render_ai(result)
    else:
        render_data(data, status)

    st.markdown(
        '<div class="footer">US Inflation Intelligence · Built for research, explanation, and disciplined forecasting</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
