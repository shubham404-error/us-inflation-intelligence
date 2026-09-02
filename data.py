from __future__ import annotations

import os
from dataclasses import dataclass
from io import StringIO
from typing import Final

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

START_DATE: Final[str] = "2000-01-01"
DATA_CUTOFF: Final[str] = "2026-08-31"
FRED_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"

# Small, deliberate model universe.
SERIES = {
    "pcepi": {
        "id": "PCEPI",
        "name": "PCE Price Index",
        "source": "BEA",
        "frequency": "Monthly",
        "type": "index",
    },
    "core_pce": {
        "id": "PCEPILFE",
        "name": "Core PCE Price Index",
        "source": "BEA",
        "frequency": "Monthly",
        "type": "index",
    },
    "shelter": {
        "id": "CUSR0000SAH1",
        "name": "CPI Shelter",
        "source": "BLS",
        "frequency": "Monthly",
        "type": "index",
    },
    "unemployment": {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "source": "BLS",
        "frequency": "Monthly",
        "type": "level",
    },
    "real_consumption": {
        "id": "DPCERA3M086SBEA",
        "name": "Real Personal Consumption Expenditures",
        "source": "BEA",
        "frequency": "Monthly",
        "type": "index",
    },
    "inflation_expectations": {
        "id": "MICH",
        "name": "University of Michigan 1-Year Inflation Expectation",
        "source": "University of Michigan",
        "frequency": "Monthly",
        "type": "level",
    },
    "oil": {
        "id": "WTISPLC",
        "name": "WTI Crude Oil Price",
        "source": "EIA / FRED splice",
        "frequency": "Monthly",
        "type": "level",
    },
}


@dataclass(frozen=True)
class FredSeries:
    key: str
    series_id: str
    name: str
    source: str
    frequency: str
    series_type: str


def _get_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FRED_API_KEY is not set.")
    return key


def fetch_series(series_id: str) -> pd.Series:
    params = {
        "api_key": _get_api_key(),
        "file_type": "json",
        "series_id": series_id,
        "observation_start": START_DATE,
        "observation_end": DATA_CUTOFF,
        "realtime_start": DATA_CUTOFF,
        "realtime_end": DATA_CUTOFF,
    }
    response = requests.get(FRED_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(f"No observations returned for {series_id}.")

    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date").sort_index()

    return df["value"].rename(series_id)


def fetch_model_data() -> pd.DataFrame:
    series_map = {}
    for key, cfg in SERIES.items():
        series_map[key] = fetch_series(cfg["id"])

    df = pd.concat(series_map, axis=1)
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()

    # Derived target and economically meaningful transformations.
    df["pce_inflation"] = df["pcepi"].pct_change(12) * 100
    df["core_pce_inflation"] = df["core_pce"].pct_change(12) * 100
    df["shelter_inflation"] = df["shelter"].pct_change(12) * 100
    df["consumption_growth"] = df["real_consumption"].pct_change(12) * 100

    df["oil_yoy"] = df["oil"].pct_change(12) * 100
    df["oil_3m"] = df["oil"].pct_change(3) * 100
    df["expectations_change"] = df["inflation_expectations"].diff(3)
    df["unemployment_change"] = df["unemployment"].diff(3)

    df = df.loc[
        (df.index >= pd.Timestamp(START_DATE))
        & (df.index <= pd.Timestamp(DATA_CUTOFF))
    ].copy()

    return df


def latest_snapshot(data: pd.DataFrame) -> dict:
    target = data.dropna(subset=["pce_inflation"]).copy()
    if target.empty:
        raise RuntimeError("No usable PCE inflation observations.")

    last = target.iloc[-1]
    last_date = target.index[-1]

    rows = []
    for key, cfg in SERIES.items():
        series = data[cfg["id"]].dropna()
        if series.empty:
            latest_date = "N/A"
            latest_value = np.nan
        else:
            latest_date = series.index[-1].strftime("%Y-%m")
            latest_value = series.iloc[-1]

        rows.append(
            {
                "Variable": cfg["name"],
                "FRED ID": cfg["id"],
                "Source": cfg["source"],
                "Frequency": cfg["frequency"],
                "Latest observation": latest_date,
                "Latest value": latest_value,
            }
        )

    return {
        "pce_inflation": float(last["pce_inflation"]),
        "core_pce": float(last["core_pce_inflation"]),
        "unemployment": float(last["unemployment"]),
        "oil": float(last["oil"]),
        "latest_pce_date": last_date.strftime("%Y-%m"),
        "series_status": pd.DataFrame(rows),
    }
