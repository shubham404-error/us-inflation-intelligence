from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
START_DATE = "2000-01-01"
DATA_CUTOFF = "2026-08-31"

# Friendly name -> FRED series metadata.
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
        "name": "Shelter CPI Index",
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
        "name": "University of Michigan Inflation Expectations",
        "source": "University of Michigan / FRED",
        "frequency": "Monthly",
    },
    "oil": {
        "id": "WTISPLC",
        "name": "WTI Spot Price",
        "source": "EIA",
        "frequency": "Monthly",
    },
}


@dataclass
class FetchResult:
    data: pd.DataFrame
    status: pd.DataFrame


def _api_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is missing. Add it to .env locally or Streamlit Secrets in production."
        )
    return key


def fetch_fred_series(series_id: str) -> pd.Series:
    params = {
        "api_key": _api_key(),
        "file_type": "json",
        "series_id": series_id,
        "observation_start": START_DATE,
        "observation_end": DATA_CUTOFF,
        "realtime_start": DATA_CUTOFF,
        "realtime_end": DATA_CUTOFF,
        "limit": 10000,
    }
    response = requests.get(FRED_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    values = []
    for item in payload.get("observations", []):
        value = item.get("value")
        if value in (None, "", "."):
            continue
        values.append((item["date"], float(value)))

    if not values:
        raise RuntimeError(f"No usable FRED observations returned for {series_id}.")

    series = pd.Series(
        data=[v for _, v in values],
        index=pd.to_datetime([d for d, _ in values]),
        name=series_id,
    )
    return series.sort_index()


def build_dataset() -> FetchResult:
    frames = []
    rows = []

    for friendly, meta in SERIES.items():
        series = fetch_fred_series(meta["id"])
        series.name = friendly
        frames.append(series)

        rows.append(
            {
                "series": friendly,
                "fred_id": meta["id"],
                "source": meta["source"],
                "latest_date": series.index.max().date().isoformat(),
                "latest_value": float(series.dropna().iloc[-1]),
                "observations": int(series.notna().sum()),
                "cutoff": DATA_CUTOFF,
            }
        )

    data = pd.concat(frames, axis=1).sort_index()
    data = data.loc[data.index <= pd.Timestamp(DATA_CUTOFF)]

    # Inflation and macro dynamics.
    data["pce_inflation"] = data["pcepi"].pct_change(12) * 100
    data["core_pce_inflation"] = data["core_pce"].pct_change(12) * 100
    data["shelter_inflation"] = data["shelter"].pct_change(12) * 100
    data["consumption_growth"] = data["real_consumption"].pct_change(12) * 100
    data["oil_yoy"] = data["oil"].pct_change(12) * 100
    data["oil_3m"] = data["oil"].pct_change(3) * 100
    data["expectations_change"] = data["inflation_expectations"].diff(3)
    data["unemployment_change"] = data["unemployment"].diff(3)

    # Simple, transparent diagnostics.
    data["inflation_momentum_3m"] = (
        data["pce_inflation"].rolling(3).mean()
        - data["pce_inflation"].rolling(6).mean()
    )

    status = pd.DataFrame(rows)
    status["latest_date"] = pd.to_datetime(status["latest_date"])

    return FetchResult(data=data, status=status)


def latest_metrics(data: pd.DataFrame) -> Dict[str, float]:
    last = data.dropna(subset=["pce_inflation"]).iloc[-1]
    return {
        "pce": float(last["pce_inflation"]),
        "core_pce": float(last["core_pce_inflation"]),
        "shelter": float(last["shelter_inflation"]),
        "unemployment": float(last["unemployment"]),
        "expectations": float(last["inflation_expectations"]),
        "consumption_growth": float(last["consumption_growth"]),
    }


def data_quality(data: pd.DataFrame) -> pd.DataFrame:
    check_cols = [
        "pce_inflation",
        "core_pce_inflation",
        "shelter_inflation",
        "unemployment",
        "consumption_growth",
        "inflation_expectations",
        "oil",
    ]
    rows = []
    for col in check_cols:
        series = data[col].dropna()
        rows.append(
            {
                "field": col,
                "first": series.index.min().date().isoformat() if len(series) else None,
                "latest": series.index.max().date().isoformat() if len(series) else None,
                "missing": int(data[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)
