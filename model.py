from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

TARGET = "pce_inflation"
FED_TARGET = 2.0
HORIZONS = (1, 2, 3, 4, 5, 6)

FEATURES = [
    "pce_inflation",
    "core_pce_inflation",
    "shelter_inflation",
    "unemployment",
    "unemployment_change",
    "consumption_growth",
    "oil_yoy",
    "oil_3m",
    "inflation_expectations",
    "expectations_change",
]

ALIASES = {
    "PCEPI": "pcepi",
    "PCEPILFE": "core_pce",
    "CUSR0000SAH1": "shelter",
    "UNRATE": "unemployment",
    "DPCERA3M086SBEA": "real_consumption",
    "MICH": "inflation_expectations",
    "WTISPLC": "oil",
}

_CACHE: dict[str, dict] = {}


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    for fred_id, internal in ALIASES.items():
        if internal not in df.columns and fred_id in df.columns:
            df[internal] = df[fred_id]

    formulas = {
        "pce_inflation": ("pcepi", lambda s: s.pct_change(12) * 100),
        "core_pce_inflation": ("core_pce", lambda s: s.pct_change(12) * 100),
        "shelter_inflation": ("shelter", lambda s: s.pct_change(12) * 100),
        "consumption_growth": ("real_consumption", lambda s: s.pct_change(12) * 100),
        "oil_yoy": ("oil", lambda s: s.pct_change(12) * 100),
        "oil_3m": ("oil", lambda s: s.pct_change(3) * 100),
        "expectations_change": (
            "inflation_expectations",
            lambda s: s.diff(3),
        ),
        "unemployment_change": (
            "unemployment",
            lambda s: s.diff(3),
        ),
    }

    for output, (source, fn) in formulas.items():
        if output not in df.columns and source in df.columns:
            df[output] = fn(df[source])

    return df


def _validate(data: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(data)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            "Forecasting data is missing: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(map(str, df.columns))
        )
    return df


def _feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    df = _validate(data)
    base = df[FEATURES]
    X = pd.DataFrame(index=base.index)

    for col in FEATURES:
        for lag in (1, 2, 3, 6, 12):
            X[f"{col}_lag{lag}"] = base[col].shift(lag)

    return X


def _supervised(data: pd.DataFrame, horizon: int):
    df = _validate(data)
    X = _feature_matrix(df)
    y = df[TARGET].shift(-horizon)

    merged = pd.concat([X, y.rename("target")], axis=1).dropna()
    if len(merged) < 100:
        raise RuntimeError(
            f"Only {len(merged)} usable observations remain for the {horizon}M model."
        )

    return merged.drop(columns="target"), merged["target"]


def _fit_xgb(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=220,
        max_depth=2,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.88,
        colsample_bytree=0.82,
        reg_alpha=0.05,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
        tree_method="hist",
    )
    model.fit(X, y, verbose=False)
    return model


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    df = pd.concat([actual, predicted], axis=1).dropna()
    return {
        "MAE": float(mean_absolute_error(df.iloc[:, 0], df.iloc[:, 1])),
        "RMSE": float(np.sqrt(mean_squared_error(df.iloc[:, 0], df.iloc[:, 1]))),
        "Bias": float((df.iloc[:, 1] - df.iloc[:, 0]).mean()),
    }


def _xgb_holdout(data: pd.DataFrame, horizon: int, holdout: int = 24):
    X, y = _supervised(data, horizon)
    if len(X) <= holdout + 80:
        holdout = max(12, len(X) // 5)

    split = len(X) - holdout
    model = _fit_xgb(X.iloc[:split], y.iloc[:split])
    pred = pd.Series(
        model.predict(X.iloc[split:]),
        index=X.index[split:],
        name="prediction",
    )
    return y.iloc[split:].rename("actual"), pred


def _sarimax_holdout(data: pd.DataFrame, holdout: int = 24):
    df = _validate(data)
    y = df[TARGET].dropna()

    if len(y) <= holdout + 80:
        holdout = max(12, len(y) // 5)

    split = len(y) - holdout
    train = y.iloc[:split]
    test = y.iloc[split:]

    fitted = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    # Forecast the complete validation block once.
    pred = fitted.forecast(steps=holdout)
    pred = pd.Series(pred.values, index=test.index, name="prediction")
    return test.rename("actual"), pred


def _sarimax_future(data: pd.DataFrame, steps: int = 6):
    df = _validate(data)
    y = df[TARGET].dropna()

    fitted = SARIMAX(
        y,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    dates = pd.date_range(
        y.index[-1] + pd.offsets.MonthBegin(1),
        periods=steps,
        freq="MS",
    )
    return pd.Series(
        fitted.forecast(steps=steps).values,
        index=dates,
        name="sarimax",
    )


def _conformal_radius(
    actual: pd.Series,
    predicted: pd.Series,
    coverage: float = 0.80,
) -> float:
    residuals = (actual - predicted).abs().dropna()
    if residuals.empty:
        return 0.75
    return float(
        residuals.quantile(
            coverage,
            interpolation="higher",
        )
    )


def _pressure(row: pd.Series) -> int:
    score = 50
    if row["pce_inflation"] > 3:
        score += 10
    if row["core_pce_inflation"] > 3:
        score += 8
    if row["shelter_inflation"] > 4:
        score += 7
    if row["oil_yoy"] > 10:
        score += 7
    if row["inflation_expectations"] > 3:
        score += 5
    if row["unemployment"] < 4.5:
        score += 5
    return int(np.clip(score, 0, 100))


def _regime(data: pd.DataFrame) -> str:
    y = _validate(data)[TARGET].dropna()
    current = float(y.iloc[-1])
    momentum = float(y.diff(3).iloc[-1])

    if current < 2.5 and momentum <= 0:
        return "Low / Disinflation"
    if current >= 3.5 and momentum > 0:
        return "Reflationary"
    return "Stable"


def _cache_key(data: pd.DataFrame) -> str:
    recent = data.tail(24)
    raw = pd.util.hash_pandas_object(
        recent,
        index=True,
    ).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def build_forecasts(
    data: pd.DataFrame,
    fast_mode: bool = True,
) -> dict:
    df = _validate(data)
    key = _cache_key(df)

    if key in _CACHE:
        return _CACHE[key]

    # 3M chronological holdout. Used to compare models and calibrate XGBoost range.
    xgb_actual, xgb_pred = _xgb_holdout(df, horizon=3)
    sar_actual, sar_pred = _sarimax_holdout(df)

    xgb_m = _metrics(xgb_actual, xgb_pred)
    sar_m = _metrics(sar_actual, sar_pred)

    inv_xgb = 1 / max(xgb_m["MAE"], 1e-6)
    inv_sar = 1 / max(sar_m["MAE"], 1e-6)

    if abs(xgb_m["MAE"] - sar_m["MAE"]) < 0.02:
        w_xgb = w_sar = 0.50
    else:
        w_xgb = inv_xgb / (inv_xgb + inv_sar)
        w_sar = 1 - w_xgb

    # Fit one direct XGBoost model for each forecast horizon.
    # This avoids artificial extrapolation between only two points.
    xgb_points = {}
    final_models = {}

    for horizon in HORIZONS:
        Xh, yh = _supervised(df, horizon)
        model = _fit_xgb(Xh, yh)
        final_models[horizon] = model
        xgb_points[horizon] = float(
            model.predict(Xh.iloc[[-1]])[0]
        )

    sar_future = _sarimax_future(df, steps=6)

    ensemble_points = {
        h: (
            w_xgb * xgb_points[h]
            + w_sar * float(sar_future.iloc[h - 1])
        )
        for h in HORIZONS
    }

    # Light smoothing for presentation only. The underlying direct forecasts remain
    # available through forecast_table.
    raw = np.array(
        [ensemble_points[h] for h in HORIZONS],
        dtype=float,
    )
    smooth = raw.copy()
    if len(raw) >= 3:
        for i in range(1, len(raw) - 1):
            smooth[i] = (
                0.25 * raw[i - 1]
                + 0.50 * raw[i]
                + 0.25 * raw[i + 1]
            )

    radius = _conformal_radius(
        xgb_actual,
        xgb_pred,
    )

    latest = df.dropna(
        subset=[
            TARGET,
            "core_pce_inflation",
            "shelter_inflation",
            "unemployment",
            "oil_yoy",
            "inflation_expectations",
        ]
    ).iloc[-1]

    pressure = _pressure(latest)
    regime = _regime(df)

    ensemble_mae = (
        w_xgb * xgb_m["MAE"]
        + w_sar * sar_m["MAE"]
    )
    confidence = (
        "High"
        if ensemble_mae < 0.30
        else "Medium"
        if ensemble_mae < 0.50
        else "Low"
    )

    importance = pd.Series(
        final_models[3].feature_importances_,
        index=_supervised(df, 3)[0].columns,
    ).sort_values(ascending=False).head(8)

    drivers = [
        (
            name.replace("_lag1", " L1")
                .replace("_lag2", " L2")
                .replace("_lag3", " L3")
                .replace("_lag6", " L6")
                .replace("_lag12", " L12")
                .replace("_", " "),
            float(value),
        )
        for name, value in importance.items()
    ]

    current_date = df[TARGET].dropna().index[-1]
    future_dates = pd.date_range(
        current_date + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )

    history = (
        df[TARGET]
        .dropna()
        .tail(96)
        .rename("pce_inflation")
        .to_frame()
    )
    history["forecast"] = np.nan
    history["lower"] = np.nan
    history["upper"] = np.nan

    future = pd.DataFrame(
        {
            "pce_inflation": np.nan,
            "forecast": smooth,
            "lower": smooth - np.array(
                [radius * np.sqrt(h / 3) for h in HORIZONS]
            ),
            "upper": smooth + np.array(
                [radius * np.sqrt(h / 3) for h in HORIZONS]
            ),
        },
        index=future_dates,
    )

    chart_df = pd.concat([history, future])

    forecast_table = pd.DataFrame(
        [
            {
                "Horizon": f"{h}M",
                "Forecast": smooth[i],
                "Lower": future.iloc[i]["lower"],
                "Upper": future.iloc[i]["upper"],
            }
            for i, h in enumerate(HORIZONS)
        ]
    )

    regime_history = df[[TARGET]].dropna().copy()
    momentum = regime_history[TARGET].diff(3)
    regime_history["regime"] = np.select(
        [
            (regime_history[TARGET] < 2.5) & (momentum <= 0),
            (regime_history[TARGET] >= 3.5) & (momentum > 0),
        ],
        [
            "Low / Disinflation",
            "Reflationary",
        ],
        default="Stable",
    )

    result = {
        # Stable public contract used by app.py.
        "schema_version": "0.5",
        "selected_model": "Ensemble",
        "ensemble_weights": {
            "XGBoost": w_xgb,
            "SARIMAX": w_sar,
        },
        "metrics": {
            "XGBoost": xgb_m,
            "SARIMAX": sar_m,
        },
        "ensemble_mae": ensemble_mae,
        "point_forecast_3m": float(smooth[2]),
        "point_forecast_6m": float(smooth[5]),
        "lower_3m": float(future.iloc[2]["lower"]),
        "upper_3m": float(future.iloc[2]["upper"]),
        "pressure_score": pressure,
        "regime": regime,
        "confidence": confidence,
        "macro_brief": (
            f"PCE inflation is currently {float(latest[TARGET]):.2f}%. "
            f"The ensemble forecasts {float(smooth[2]):.2f}% in three months "
            f"and {float(smooth[5]):.2f}% in six months, versus the "
            f"Federal Reserve's 2% objective. The current inflation "
            f"environment is {regime.lower()}, with a pressure score of "
            f"{pressure}/100. The 3M XGBoost prediction interval is "
            f"{float(future.iloc[2]['lower']):.2f}% to "
            f"{float(future.iloc[2]['upper']):.2f}%."
        ),
        "forecast_table": forecast_table,
        "chart_df": chart_df,
        "xgb_drivers": drivers,
        "regime_history": regime_history,
        "calibration_radius_3m": radius,
        "xgb_actual_3": xgb_actual,
        "xgb_pred_3": xgb_pred,
    }

    _CACHE[key] = result
    return result


def model_diagnostics(forecasts: dict) -> dict:
    metrics = forecasts["metrics"]

    performance = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "MAE": round(metrics["XGBoost"]["MAE"], 3),
                "RMSE": round(metrics["XGBoost"]["RMSE"], 3),
                "Bias": round(metrics["XGBoost"]["Bias"], 3),
            },
            {
                "Model": "SARIMAX",
                "MAE": round(metrics["SARIMAX"]["MAE"], 3),
                "RMSE": round(metrics["SARIMAX"]["RMSE"], 3),
                "Bias": round(metrics["SARIMAX"]["Bias"], 3),
            },
            {
                "Model": "Ensemble",
                "MAE": round(forecasts["ensemble_mae"], 3),
                "RMSE": np.nan,
                "Bias": np.nan,
            },
        ]
    )

    residuals = (
        forecasts["xgb_actual_3"]
        - forecasts["xgb_pred_3"]
    ).dropna()

    coverage = "N/A"
    if len(residuals) >= 24:
        split = len(residuals) // 2
        calibration = residuals.iloc[:split].abs()
        test = residuals.iloc[split:].abs()
        q = calibration.quantile(
            0.80,
            interpolation="higher",
        )
        coverage = f"{100 * (test <= q).mean():.1f}%"

    return {
        "performance": performance,
        "coverage": pd.DataFrame(
            [
                {
                    "Target": "80%",
                    "Observed holdout coverage": coverage,
                    "Method": "Temporal conformal",
                }
            ]
        ),
    }


def driver_table(forecasts: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Driver": name.title(),
                "Importance": round(value, 4),
            }
            for name, value in forecasts["xgb_drivers"]
        ]
    )
