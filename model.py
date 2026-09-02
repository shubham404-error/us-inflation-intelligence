from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


TARGET = "pce_inflation"
HORIZONS = (1, 3, 6)
FED_TARGET = 2.0

BASE_FEATURES = [
    TARGET,
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

LAGS = (1, 2, 3, 6, 12)


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    state: Dict
    drivers: pd.DataFrame
    history: pd.DataFrame
    calibration: Dict
    chat_context: Dict


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    for col in BASE_FEATURES:
        if col not in data.columns:
            continue
        for lag in LAGS:
            x[f"{col}_lag{lag}"] = data[col].shift(lag)
    return x


def _xgb_model() -> object:
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=350,
            max_depth=2,
            learning_rate=0.025,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.5,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        )
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.035,
        max_leaf_nodes=8,
        l2_regularization=1.0,
        random_state=42,
    )


def _safe_sarimax(train: pd.Series, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=steps)
    mean = np.asarray(pred.predicted_mean, dtype=float)
    conf = np.asarray(pred.conf_int(alpha=0.20), dtype=float)
    return mean, conf


def _fit_direct_xgb(data: pd.DataFrame, horizon: int):
    features = _feature_frame(data)
    y = data[TARGET].shift(-horizon)
    frame = pd.concat([features, y.rename("y")], axis=1).dropna(subset=["y"])

    # Keep the complete feature schema. Tree models can handle missing predictor values,
    # and dropping columns from the prediction row creates XGBoost feature-name mismatches.
    X = frame[features.columns]
    valid = X.notna().any(axis=1)
    X = X.loc[valid]
    y_fit = frame.loc[valid, "y"]

    model = _xgb_model()
    model.fit(X, y_fit)
    return model, features


def _walk_forward_xgb(data: pd.DataFrame, horizon: int, min_train: int = 96) -> pd.DataFrame:
    features = _feature_frame(data)
    target = data[TARGET]
    rows = []

    valid_dates = target.dropna().index

    # Monthly rolling-origin evaluation. Keep the window modest enough for app runtime.
    for i in range(min_train, len(valid_dates) - horizon + 1):
        origin_date = valid_dates[i - 1]
        forecast_date = valid_dates[i + horizon - 1]

        train_mask = features.index <= origin_date
        y_train = target.shift(-horizon)
        fit = pd.concat([features.loc[train_mask], y_train.loc[train_mask].rename("y")], axis=1).dropna()

        if len(fit) < 72:
            continue

        model = _xgb_model()
        model.fit(fit.drop(columns=["y"]), fit["y"])

        if forecast_date not in features.index:
            continue

        train_cols = list(fit.drop(columns=["y"]).columns)
        row = features.loc[[forecast_date]].reindex(columns=train_cols)

        # Preserve the exact training schema. XGBoost and HistGradientBoosting both
        # support missing predictor values, so unavailable latest macro inputs do not
        # delete columns and break the feature contract.
        pred = float(model.predict(row)[0])
        actual = float(target.loc[forecast_date])

        rows.append(
            {
                "origin": origin_date,
                "date": forecast_date,
                "actual": actual,
                "pred": pred,
                "error": actual - pred,
            }
        )

    return pd.DataFrame(rows)


def _walk_forward_sarimax(data: pd.DataFrame, horizon: int, min_train: int = 96) -> pd.DataFrame:
    target = data[TARGET].dropna()
    dates = target.index
    rows = []

    for i in range(min_train, len(dates) - horizon + 1):
        train = target.iloc[:i]
        forecast_date = dates[i + horizon - 1]

        try:
            model = SARIMAX(
                train,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            pred = float(fit.get_forecast(steps=horizon).predicted_mean.iloc[-1])
        except Exception:
            continue

        rows.append(
            {
                "origin": dates[i - 1],
                "date": forecast_date,
                "actual": float(target.loc[forecast_date]),
                "pred": pred,
                "error": float(target.loc[forecast_date]) - pred,
            }
        )

    return pd.DataFrame(rows)


def _metrics(rows: pd.DataFrame, model_name: str, horizon: int) -> Dict:
    if rows.empty:
        return {
            "model": model_name,
            "horizon": horizon,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "n": 0,
        }
    return {
        "model": model_name,
        "horizon": horizon,
        "MAE": float(mean_absolute_error(rows["actual"], rows["pred"])),
        "RMSE": float(mean_squared_error(rows["actual"], rows["pred"]) ** 0.5),
        "Bias": float((rows["pred"] - rows["actual"]).mean()),
        "n": int(len(rows)),
    }


def _direction(m: float) -> str:
    if m <= -0.15:
        return "Moderating"
    if m >= 0.15:
        return "Reaccelerating"
    return "Stable"


def _state(data: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
    target = data[TARGET].dropna()
    current = float(target.iloc[-1])
    m3 = float(target.tail(3).mean() - target.iloc[-4:-1].mean()) if len(target) >= 4 else 0.0
    m6 = float(target.tail(6).mean() - target.iloc[-12:-6].mean()) if len(target) >= 12 else 0.0

    if current > 3.5:
        level = "Elevated"
    elif current > 2.5:
        level = "Above target"
    else:
        level = "Near target"

    pressure = 50.0
    pressure += np.clip((current - FED_TARGET) * 12, -20, 30)
    pressure += np.clip(m3 * 18, -15, 15)

    latest = data.iloc[-1]
    if float(latest.get("core_pce_inflation", current)) > current:
        pressure += 5
    if float(latest.get("shelter_inflation", 0)) > 4:
        pressure += 7
    if float(latest.get("inflation_expectations", 0)) > 3:
        pressure += 5
    pressure = float(np.clip(pressure, 0, 100))

    momentum = _direction(m3)

    f6 = float(forecast.loc[forecast["horizon"] == 6, "point"].iloc[0])
    gap = f6 - FED_TARGET
    if f6 <= 2.5:
        outlook = "Approaching target"
    elif f6 < current - 0.15:
        outlook = "Disinflation"
    elif f6 > current + 0.15:
        outlook = "Reacceleration"
    else:
        outlook = "Sticky"

    confidence = "High" if pressure < 35 or pressure > 75 else "Medium"

    return {
        "current": current,
        "level": level,
        "momentum": momentum,
        "momentum_3m": m3,
        "momentum_6m": m6,
        "outlook": outlook,
        "forecast_6m": f6,
        "gap_to_target": gap,
        "pressure_score": pressure,
        "confidence": confidence,
    }


def _driver_table(data: pd.DataFrame, model) -> pd.DataFrame:
    features = _feature_frame(data)
    cols = list(features.columns)
    latest = features.dropna().iloc[[-1]]
    latest = latest.reindex(columns=cols)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    else:
        importance = np.zeros(len(cols))

    grouped = {}
    for col, imp in zip(cols, importance):
        base = col.rsplit("_lag", 1)[0]
        grouped[base] = grouped.get(base, 0.0) + float(imp)

    labels = {
        TARGET: "Inflation persistence",
        "core_pce_inflation": "Core PCE",
        "shelter_inflation": "Shelter inflation",
        "unemployment": "Unemployment",
        "unemployment_change": "Unemployment change",
        "consumption_growth": "Real consumption growth",
        "oil_yoy": "Oil",
        "oil_3m": "Oil momentum",
        "inflation_expectations": "Inflation expectations",
        "expectations_change": "Expectations change",
    }

    out = pd.DataFrame(
        [
            {"Driver": labels.get(k, k), "Importance": v}
            for k, v in grouped.items()
        ]
    ).sort_values("Importance", ascending=False)

    if out["Importance"].sum() > 0:
        out["Importance"] = out["Importance"] / out["Importance"].sum()
    return out.head(6).reset_index(drop=True)


def _naive_metrics(target: pd.Series) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        actual = target.iloc[h:].to_numpy()
        pred = target.iloc[:-h].to_numpy()
        n = min(len(actual), len(pred))
        actual = actual[:n]
        pred = pred[:n]
        rows.append(
            {
                "model": "Naive",
                "horizon": h,
                "MAE": float(mean_absolute_error(actual, pred)),
                "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
                "Bias": float((pred - actual).mean()),
                "n": int(n),
            }
        )
    return pd.DataFrame(rows)


def run_research_validation(data: pd.DataFrame, max_origins: int = 24) -> pd.DataFrame:
    """Run expensive rolling-origin validation on demand, never during app startup."""
    metric_rows = []
    target = data[TARGET].dropna()
    for h in HORIZONS:
        xgb_eval = _walk_forward_xgb(data, h, min_train=96)
        sarimax_eval = _walk_forward_sarimax(data, h, min_train=96)
        if len(xgb_eval) > max_origins:
            xgb_eval = xgb_eval.tail(max_origins)
        if len(sarimax_eval) > max_origins:
            sarimax_eval = sarimax_eval.tail(max_origins)
        metric_rows.append(_metrics(xgb_eval, "XGBoost", h))
        metric_rows.append(_metrics(sarimax_eval, "SARIMAX", h))
    return pd.concat([_naive_metrics(target), pd.DataFrame(metric_rows)], ignore_index=True)


def build_forecast(data: pd.DataFrame, validation_metrics: pd.DataFrame | None = None) -> ForecastResult:
    """Fast production forecast. Expensive walk-forward validation is intentionally external."""
    data = data.copy()
    target = data[TARGET].dropna()

    # Fit each direct XGBoost model once. These are the production ML forecasts.
    xgb_path = []
    final_xgb_by_h = {}
    for h in range(1, 7):
        model, features = _fit_direct_xgb(data, h)
        final_xgb_by_h[h] = model
        latest_row = features.iloc[[-1]].reindex(columns=features.columns)
        if latest_row.isna().all(axis=1).iloc[0]:
            raise RuntimeError("Latest feature row is unavailable for forecasting.")
        xgb_path.append(float(model.predict(latest_row)[0]))

    # One econometric model produces the complete six-month challenger path.
    sarimax_mean, _ = _safe_sarimax(target, 6)

    # Prefer previously validated weights when supplied. Until research validation is run,
    # use a neutral 50/50 mix rather than pretending one model has already won.
    if validation_metrics is not None and not validation_metrics.empty:
        xgb3 = validation_metrics.query("model == 'XGBoost' and horizon == 3")
        sar3 = validation_metrics.query("model == 'SARIMAX' and horizon == 3")
        xgb_mae = float(xgb3["MAE"].iloc[0]) if not xgb3.empty else np.nan
        sar_mae = float(sar3["MAE"].iloc[0]) if not sar3.empty else np.nan
        if np.isfinite(xgb_mae) and np.isfinite(sar_mae):
            if abs(xgb_mae - sar_mae) < 0.10:
                weights = {"XGBoost": 0.50, "SARIMAX": 0.50}
            elif xgb_mae < sar_mae:
                weights = {"XGBoost": 0.65, "SARIMAX": 0.35}
            else:
                weights = {"XGBoost": 0.35, "SARIMAX": 0.65}
        else:
            weights = {"XGBoost": 0.50, "SARIMAX": 0.50}
    else:
        weights = {"XGBoost": 0.50, "SARIMAX": 0.50}

    ensemble = [
        weights["XGBoost"] * xgb_path[h - 1] + weights["SARIMAX"] * sarimax_mean[h - 1]
        for h in range(1, 7)
    ]

    # Prediction interval. Until a research backtest exists this is explicitly provisional.
    if validation_metrics is not None and not validation_metrics.empty:
        xgb3 = validation_metrics.query("model == 'XGBoost' and horizon == 3")
        radius = float(xgb3["MAE"].iloc[0]) * 1.35 if not xgb3.empty and np.isfinite(xgb3["MAE"].iloc[0]) else 0.45
        calibration_note = "Backtest-calibrated proxy"
    else:
        radius = 0.45
        calibration_note = "Provisional until backtest is run"

    lower = [p - radius * np.sqrt(h / 3) for h, p in enumerate(ensemble, start=1)]
    upper = [p + radius * np.sqrt(h / 3) for h, p in enumerate(ensemble, start=1)]
    forecast = pd.DataFrame(
        {
            "horizon": np.arange(1, 7),
            "point": ensemble,
            "lower_80": lower,
            "upper_80": upper,
        }
    )

    drivers = _driver_table(data, final_xgb_by_h[3])

    hist = data[[TARGET]].dropna().copy()
    hist["momentum_3m"] = hist[TARGET].rolling(3).mean().diff(3)
    hist["state"] = np.select(
        [hist["momentum_3m"] <= -0.15, hist["momentum_3m"] >= 0.15],
        ["Moderating", "Reaccelerating"],
        default="Stable",
    )
    hist = hist.reset_index(names="date")

    state = _state(data, forecast)
    state["weights"] = weights

    if validation_metrics is not None and not validation_metrics.empty:
        xgb3 = validation_metrics.query("model == 'XGBoost' and horizon == 3")
        calibration_radius = float(xgb3["MAE"].iloc[0]) if not xgb3.empty else np.nan
        calibration = {
            "target": 0.80,
            "empirical_coverage": np.nan,
            "radius": calibration_radius,
            "n": 0,
            "note": calibration_note,
        }
    else:
        calibration = {
            "target": 0.80,
            "empirical_coverage": np.nan,
            "radius": radius,
            "n": 0,
            "note": calibration_note,
        }

    metrics = validation_metrics.copy() if validation_metrics is not None else _naive_metrics(target)
    latest_date = data.dropna(subset=[TARGET]).index[-1].date().isoformat()

    chat_context = {
        "latest_data_date": latest_date,
        "fed_target": FED_TARGET,
        "state": state,
        "forecast": forecast.to_dict("records"),
        "drivers": drivers.to_dict("records"),
        "validation": metrics.to_dict("records"),
        "calibration": calibration,
        "latest": data.iloc[-1][
            [
                c for c in [
                    "core_pce_inflation",
                    "shelter_inflation",
                    "unemployment",
                    "consumption_growth",
                    "inflation_expectations",
                    "oil_yoy",
                ] if c in data.columns
            ]
        ].dropna().to_dict(),
    }

    return ForecastResult(
        forecast=forecast,
        metrics=metrics,
        state=state,
        drivers=drivers,
        history=hist,
        calibration=calibration,
        chat_context=chat_context,
    )
