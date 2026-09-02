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


def build_forecast(data: pd.DataFrame) -> ForecastResult:
    data = data.copy()
    target = data[TARGET].dropna()

    # Walk-forward validation.
    validation = {}
    metric_rows = []

    for h in HORIZONS:
        xgb_eval = _walk_forward_xgb(data, h)
        sarimax_eval = _walk_forward_sarimax(data, h)

        validation[("XGBoost", h)] = xgb_eval
        validation[("SARIMAX", h)] = sarimax_eval
        metric_rows.append(_metrics(xgb_eval, "XGBoost", h))
        metric_rows.append(_metrics(sarimax_eval, "SARIMAX", h))

        naive_actual = target.iloc[h:]
        naive_pred = target.iloc[:-h].values
        n = min(len(naive_actual), len(naive_pred))
        naive_rows = pd.DataFrame(
            {
                "actual": naive_actual.iloc[:n].values,
                "pred": naive_pred[:n],
            }
        )
        metric_rows.append(
            {
                "model": "Naive",
                "horizon": h,
                "MAE": float(mean_absolute_error(naive_rows["actual"], naive_rows["pred"])),
                "RMSE": float(mean_squared_error(naive_rows["actual"], naive_rows["pred"]) ** 0.5),
                "Bias": float((naive_rows["pred"] - naive_rows["actual"]).mean()),
                "n": int(n),
            }
        )

    metrics = pd.DataFrame(metric_rows)

    # Final models fitted on all available data.
    predictions = []
    final_xgb_by_h = {}
    for h in HORIZONS:
        model, features = _fit_direct_xgb(data, h)
        final_xgb_by_h[h] = model
        row = features.dropna().iloc[[-1]]
        pred = float(model.predict(row)[0])
        predictions.append({"horizon": h, "xgb": pred})

    # SARIMAX 6M path, used as an econometric challenger.
    sarimax_mean, sarimax_conf = _safe_sarimax(target, 6)

    # Select / blend using 3M validation.
    xgb_3 = validation[("XGBoost", 3)]
    sar_3 = validation[("SARIMAX", 3)]
    xgb_mae = _metrics(xgb_3, "XGBoost", 3)["MAE"]
    sar_mae = _metrics(sar_3, "SARIMAX", 3)["MAE"]

    if not np.isfinite(xgb_mae):
        weights = {"XGBoost": 0.0, "SARIMAX": 1.0}
    elif not np.isfinite(sar_mae):
        weights = {"XGBoost": 1.0, "SARIMAX": 0.0}
    elif abs(xgb_mae - sar_mae) < 0.10:
        weights = {"XGBoost": 0.5, "SARIMAX": 0.5}
    elif xgb_mae < sar_mae:
        weights = {"XGBoost": 0.65, "SARIMAX": 0.35}
    else:
        weights = {"XGBoost": 0.35, "SARIMAX": 0.65}

    xgb_path = []
    for h in range(1, 7):
        model, features = _fit_direct_xgb(data, h)
        row = features.dropna().iloc[[-1]]
        xgb_path.append(float(model.predict(row)[0]))

    ensemble = []
    for h in range(1, 7):
        value = weights["XGBoost"] * xgb_path[h - 1]
        value += weights["SARIMAX"] * sarimax_mean[h - 1]
        ensemble.append(value)

    # Prediction intervals calibrated from 3M XGB walk-forward absolute residuals.
    residuals = np.abs(xgb_3["error"].values) if not xgb_3.empty else np.array([0.4])
    q80 = float(np.quantile(residuals, 0.80))
    lower = []
    upper = []
    for h, point in enumerate(ensemble, start=1):
        radius = q80 * np.sqrt(h / 3)
        lower.append(point - radius)
        upper.append(point + radius)

    forecast = pd.DataFrame(
        {
            "horizon": np.arange(1, 7),
            "point": ensemble,
            "lower_80": lower,
            "upper_80": upper,
        }
    )

    # Driver ranking from 3M XGB.
    drivers = _driver_table(data, final_xgb_by_h[3])

    # Regime-like history based on transparent rolling momentum, not HMM.
    hist = data[[TARGET]].dropna().copy()
    hist["momentum_3m"] = hist[TARGET].rolling(3).mean().diff(3)
    hist["state"] = np.select(
        [
            hist["momentum_3m"] <= -0.15,
            hist["momentum_3m"] >= 0.15,
        ],
        [
            "Moderating",
            "Reaccelerating",
        ],
        default="Stable",
    )
    hist = hist.reset_index(names="date")

    state = _state(data, forecast)
    state["weights"] = weights

    # Empirical 80% interval coverage on the available 3M XGB walk-forward sample.
    coverage = (
        float((np.abs(xgb_3["error"]) <= q80).mean()) if not xgb_3.empty else np.nan
    )
    calibration = {
        "target": 0.80,
        "empirical_coverage": coverage,
        "radius": q80,
        "n": int(len(xgb_3)),
    }

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
                c
                for c in [
                    "core_pce_inflation",
                    "shelter_inflation",
                    "unemployment",
                    "consumption_growth",
                    "inflation_expectations",
                    "oil_yoy",
                ]
                if c in data.columns
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
