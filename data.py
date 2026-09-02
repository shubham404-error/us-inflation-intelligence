from __future__ import annotations

import os
from typing import Final

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

START_DATE: Final[str] = "2000-01-01"
DATA_CUTOFF: Final[str] = "2026-08-31"
FRED_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"

# Deliberately small, economically motivated variable universe.
SERIES = {
    "pcepi": {
        "id": "PCEPI",
        "name": "PCE Price Index",
        "source": "BEA",
        "frequency": "Monthly",
    },
    "core_pce": {
        "id": "PCEPILFE",
        "name": "Core PCE Price Index",
        "source": "BEA",
        "frequency": "Monthly",
    },
    "shelter": {
        "id": "CUSR0000SAH1",
        "name": "CPI Shelter",
        "source": "BLS",
        "frequency": "Monthly",
    },
    "unemployment": {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "source": "BLS",
        "frequency": "Monthly",
    },
    "real_consumption": {
        "id": "DPCERA3M086SBEA",
        "name": "Real Personal Consumption Expenditures",
        "source": "BEA",
        "frequency": "Monthly",
    },
    "inflation_expectations": {
        "id": "MICH",
        "name": "1-Year Inflation Expectations",
        "source": "University of Michigan",
        "frequency": "Monthly",
    },
    "oil": {
        "id": "WTISPLC",
        "name": "WTI Crude Oil Price",
        "source": "EIA / FRED",
        "frequency": "Monthly",
    },
}


def _api_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is missing. Add it to Streamlit Secrets or .env."
        )
    return key


def _monthly_last(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.index = out.index.to_period("M").to_timestamp()
    return out.groupby(level=0).last()


def fetch_series(series_id: str) -> pd.Series:
    params = {
        "api_key": _api_key(),
        "file_type": "json",
        "series_id": series_id,
        "observation_start": START_DATE,
        "observation_end": DATA_CUTOFF,
        "realtime_start": DATA_CUTOFF,
        "realtime_end": DATA_CUTOFF,
    }

    response = requests.get(
        FRED_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()

    if "error_code" in payload:
        raise RuntimeError(
            f"FRED error {payload['error_code']}: "
            f"{payload.get('error_message', 'Unknown error')}"
        )

    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(
            f"No observations returned for FRED series {series_id}."
        )

    frame = pd.DataFrame(observations)
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    frame = (
        frame.dropna(
            subset=["date", "value"]
        )
        .set_index("date")
        .sort_index()
    )

    frame = frame[
        ~frame.index.duplicated(
            keep="last"
        )
    ]

    return frame["value"]


def fetch_model_data() -> pd.DataFrame:
    data = {}

    for key, cfg in SERIES.items():
        data[key] = _monthly_last(
            fetch_series(cfg["id"])
        )

    df = pd.concat(
        data,
        axis=1,
    ).sort_index()

    # Target.
    df["pce_inflation"] = (
        df["pcepi"]
        .pct_change(12)
        * 100
    )

    # Core and shelter inflation.
    df["core_pce_inflation"] = (
        df["core_pce"]
        .pct_change(12)
        * 100
    )

    df["shelter_inflation"] = (
        df["shelter"]
        .pct_change(12)
        * 100
    )

    # Demand.
    df["consumption_growth"] = (
        df["real_consumption"]
        .pct_change(12)
        * 100
    )

    # Energy.
    df["oil_yoy"] = (
        df["oil"]
        .pct_change(12)
        * 100
    )

    df["oil_3m"] = (
        df["oil"]
        .pct_change(3)
        * 100
    )

    # Expectations and labour momentum.
    df["expectations_change"] = (
        df["inflation_expectations"]
        .diff(3)
    )

    df["unemployment_change"] = (
        df["unemployment"]
        .diff(3)
    )

    df = df.loc[
        pd.Timestamp(START_DATE):
        pd.Timestamp(DATA_CUTOFF)
    ].copy()

    return df


def latest_status(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for key, cfg in SERIES.items():
        series = data[key].dropna()

        rows.append(
            {
                "Variable": cfg["name"],
                "FRED ID": cfg["id"],
                "Source": cfg["source"],
                "Frequency": cfg["frequency"],
                "Latest": (
                    series.index[-1].strftime("%Y-%m")
                    if not series.empty
                    else "N/A"
                ),
            }
        )

    return pd.DataFrame(rows)
