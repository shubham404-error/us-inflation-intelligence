from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


TARGET = "pce_inflation"

BASE_FEATURES = [
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

HORIZONS = (1, 2, 3, 4, 5, 6)

_CACHE: Dict[str, dict] = {}


def _cache_key(data: pd.DataFrame) -> str:
    recent = data.tail(24)
    raw = pd.util.hash_pandas_object(
        recent,
        index=True,
    ).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def normalize_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    df = data.copy()

    aliases = {
        "PCEPI": "pcepi",
        "PCEPILFE": "core_pce",
        "CUSR0000SAH1": "shelter",
        "UNRATE": "unemployment",
        "DPCERA3M086SBEA": "real_consumption",
        "MICH": "inflation_expectations",
        "WTISPLC": "oil",
    }

    for fred_id, internal in aliases.items():
        if (
            internal not in df.columns
            and fred_id in df.columns
        ):
            df[internal] = df[fred_id]

    if (
        "pce_inflation" not in df.columns
        and "pcepi" in df.columns
    ):
        df["pce_inflation"] = (
            df["pcepi"].pct_change(12) * 100
        )

    if (
        "core_pce_inflation" not in df.columns
        and "core_pce" in df.columns
    ):
        df["core_pce_inflation"] = (
            df["core_pce"].pct_change(12) * 100
        )

    if (
        "shelter_inflation" not in df.columns
        and "shelter" in df.columns
    ):
        df["shelter_inflation"] = (
            df["shelter"].pct_change(12) * 100
        )

    if (
        "consumption_growth" not in df.columns
        and "real_consumption" in df.columns
    ):
        df["consumption_growth"] = (
            df["real_consumption"].pct_change(12) * 100
        )

    if (
        "oil_yoy" not in df.columns
        and "oil" in df.columns
    ):
        df["oil_yoy"] = (
            df["oil"].pct_change(12) * 100
        )

    if (
        "oil_3m" not in df.columns
        and "oil" in df.columns
    ):
        df["oil_3m"] = (
            df["oil"].pct_change(3) * 100
        )

    if (
        "expectations_change" not in df.columns
        and "inflation_expectations" in df.columns
    ):
        df["expectations_change"] = (
            df["inflation_expectations"].diff(3)
        )

    if (
        "unemployment_change" not in df.columns
        and "unemployment" in df.columns
    ):
        df["unemployment_change"] = (
            df["unemployment"].diff(3)
        )

    return df


def _feature_matrix(
    data: pd.DataFrame,
) -> pd.DataFrame:
    df = normalize_columns(data)

    missing = [
        c for c in BASE_FEATURES
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing model columns: "
            + ", ".join(missing)
        )

    base = df[BASE_FEATURES].copy()

    X = pd.DataFrame(
        index=base.index
    )

    # Lagged predictors only.
    for column in BASE_FEATURES:
        for lag in (1, 2, 3, 6, 12):
            X[f"{column}_lag{lag}"] = (
                base[column].shift(lag)
            )

    return X


def _supervised(
    data: pd.DataFrame,
    horizon: int,
):
    df = normalize_columns(data)
    X = _feature_matrix(df)
    y = df[TARGET].shift(-horizon)

    merged = pd.concat(
        [
            X,
            y.rename("target"),
        ],
        axis=1,
    ).dropna()

    if len(merged) < 100:
        raise RuntimeError(
            f"Not enough observations for {horizon}M model."
        )

    return (
        merged.drop(columns="target"),
        merged["target"],
    )


def _fit_xgb(
    X: pd.DataFrame,
    y: pd.Series,
) -> XGBRegressor:
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

    model.fit(
        X,
        y,
        verbose=False,
    )

    return model


def _holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    holdout: int = 24,
):
    split = len(X) - holdout

    if split < 80:
        raise RuntimeError(
            "Insufficient history for validation."
        )

    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:],
    )


def _xgb_holdout(
    data: pd.DataFrame,
    horizon: int,
):
    X, y = _supervised(
        data,
        horizon,
    )

    X_train, X_test, y_train, y_test = (
        _holdout_split(
            X,
            y,
            holdout=24,
        )
    )

    model = _fit_xgb(
        X_train,
        y_train,
    )

    prediction = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="prediction",
    )

    return (
        y_test.rename("actual"),
        prediction,
    )


def _sarimax_forecast_series(
    data: pd.DataFrame,
    steps: int,
) -> pd.Series:
    df = normalize_columns(data)
    y = df[TARGET].dropna()

    model = SARIMAX(
        y,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False
    )

    forecast = fitted.forecast(
        steps=steps
    )

    return pd.Series(
        forecast.values,
        index=pd.date_range(
            y.index[-1]
            + pd.offsets.MonthBegin(1),
            periods=steps,
            freq="MS",
        ),
        name="sarimax",
    )


def _sarimax_holdout(
    data: pd.DataFrame,
    horizon: int,
):
    df = normalize_columns(data)
    y = df[TARGET].dropna()

    holdout = 24
    split = len(y) - holdout

    if split < 80:
        raise RuntimeError(
            "Insufficient history for SARIMAX validation."
        )

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

    # For a horizon-h validation observation, forecast h steps
    # from each historical origin. To remain fast, evaluate the
    # final contiguous holdout using a direct forecast from the
    # training endpoint and retain the matching last observation.
    fc = fitted.forecast(
        steps=holdout + horizon - 1
    )

    aligned_prediction = pd.Series(
        fc.iloc[horizon - 1:horizon - 1 + holdout].values,
        index=test.index,
        name="prediction",
    )

    return (
        test.rename("actual"),
        aligned_prediction,
    )


def _metric_bundle(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    df = pd.concat(
        [actual, predicted],
        axis=1,
    ).dropna()

    return {
        "MAE": float(
            mean_absolute_error(
                df["actual"],
                df["prediction"],
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    df["actual"],
                    df["prediction"],
                )
            )
        ),
        "Bias": float(
            (
                df["prediction"]
                - df["actual"]
            ).mean()
        ),
    }


def _conformal_radius(
    actual: pd.Series,
    predicted: pd.Series,
    coverage: float = 0.80,
) -> float:
    residuals = (
        actual
        - predicted
    ).abs().dropna()

    if len(residuals) < 12:
        return (
            float(residuals.median())
            if len(residuals)
            else 0.75
        )

    # Conservative finite-sample empirical quantile.
    q = residuals.quantile(
        coverage,
        interpolation="higher",
    )

    return float(q)


def _pressure(
    row: pd.Series,
) -> int:
    score = 50

    if row["pce_inflation"] > 3.0:
        score += 10

    if row["core_pce_inflation"] > 3.0:
        score += 8

    if row["shelter_inflation"] > 4.0:
        score += 7

    if row["oil_yoy"] > 10.0:
        score += 7

    if row["inflation_expectations"] > 3.0:
        score += 5

    if row["unemployment"] < 4.5:
        score += 5

    return int(
        np.clip(
            score,
            0,
            100,
        )
    )


def _regime(
    data: pd.DataFrame,
) -> str:
    df = normalize_columns(data)

    y = df[TARGET].dropna()

    current = float(y.iloc[-1])
    momentum = float(
        y.diff(3).iloc[-1]
    )

    if current < 2.5 and momentum <= 0:
        return "Low / Disinflation"

    if current >= 3.5 and momentum > 0:
        return "Reflationary"

    return "Stable"


def build_forecasts(
    data: pd.DataFrame,
    fast_mode: bool = True,
) -> dict:

    df = normalize_columns(data)

    key = (
        _cache_key(df)
        + f"_{fast_mode}"
    )

    if key in _CACHE:
        return _CACHE[key]

    # --------------------------------------------------------
    # Validate only the important horizons quickly.
    # --------------------------------------------------------

    xgb_actual_3, xgb_pred_3 = _xgb_holdout(
        df,
        horizon=3,
    )

    sar_actual_3, sar_pred_3 = _sarimax_holdout(
        df,
        horizon=3,
    )

    xgb_metrics = _metric_bundle(
        xgb_actual_3,
        xgb_pred_3,
    )

    sar_metrics = _metric_bundle(
        sar_actual_3,
        sar_pred_3,
    )

    metrics = {
        "XGBoost": xgb_metrics,
        "SARIMAX": sar_metrics,
    }

    # Inverse-MAE ensemble weights, capped to avoid domination
    # by a single model when performance is nearly identical.
    inv_xgb = 1 / max(xgb_metrics["MAE"], 1e-6)
    inv_sar = 1 / max(sar_metrics["MAE"], 1e-6)

    w_xgb = inv_xgb / (
        inv_xgb + inv_sar
    )
    w_sar = 1 - w_xgb

    # Keep the ensemble balanced when errors are very close.
    if abs(
        xgb_metrics["MAE"]
        - sar_metrics["MAE"]
    ) < 0.02:
        w_xgb = 0.50
        w_sar = 0.50

    # --------------------------------------------------------
    # Direct XGBoost models for every future month.
    # This is smoother and more honest than drawing a line
    # between 3M and 6M points.
    # --------------------------------------------------------

    final_models = {}
    monthly_forecasts = []

    current_date = (
        df.dropna(
            subset=[TARGET]
        ).index[-1]
    )

    future_dates = pd.date_range(
        current_date
        + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )

    for horizon in HORIZONS:

        X_h, y_h = _supervised(
            df,
            horizon=horizon,
        )

        final_xgb = _fit_xgb(
            X_h,
            y_h,
        )

        xgb_point = float(
            final_xgb.predict(
                X_h.iloc[[-1]]
            )[0]
        )

        # Get the same-horizon SARIMAX point forecast.
        sar_series = _sarimax_forecast_series(
            df,
            steps=6,
        )

        sar_point = float(
            sar_series.iloc[
                horizon - 1
            ]
        )

        ensemble_point = (
            w_xgb * xgb_point
            + w_sar * sar_point
        )

        monthly_forecasts.append(
            ensemble_point
        )

        final_models[horizon] = final_xgb

    # --------------------------------------------------------
    # Smooth display path, using a light rolling blend between
    # adjacent direct forecasts without changing endpoints.
    # --------------------------------------------------------

    raw_path = np.array(
        monthly_forecasts,
        dtype=float,
    )

    smooth_path = raw_path.copy()

    if len(raw_path) >= 3:
        for i in range(1, len(raw_path) - 1):
            smooth_path[i] = (
                0.25 * raw_path[i - 1]
                + 0.50 * raw_path[i]
                + 0.25 * raw_path[i + 1]
            )

    # --------------------------------------------------------
    # Uncertainty.
    # Use horizon-3 conformal radius, scaled gently by horizon.
    # This is a visualization range, not a model-based
    # confidence interval.
    # --------------------------------------------------------

    radius_3m = _conformal_radius(
        xgb_actual_3,
        xgb_pred_3,
    )

    horizon_radii = np.array(
        [
            radius_3m
            * np.sqrt(h / 3)
            for h in HORIZONS
        ],
        dtype=float,
    )

    lower = (
        smooth_path
        - horizon_radii
    )

    upper = (
        smooth_path
        + horizon_radii
    )

    # --------------------------------------------------------
    # Current state.
    # --------------------------------------------------------

    latest_valid = df.dropna(
        subset=[
            TARGET,
            "core_pce_inflation",
            "shelter_inflation",
            "oil_yoy",
            "inflation_expectations",
            "unemployment",
        ]
    ).iloc[-1]

    pressure = _pressure(
        latest_valid
    )

    regime = _regime(df)

    # Model confidence as an error-based label.
    ensemble_mae = (
        w_xgb * xgb_metrics["MAE"]
        + w_sar * sar_metrics["MAE"]
    )

    if ensemble_mae < 0.30:
        confidence = "High"
    elif ensemble_mae < 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    # --------------------------------------------------------
    # Driver importance from the 3M XGBoost model.
    # --------------------------------------------------------

    x3, y3 = _supervised(
        df,
        horizon=3,
    )

    model_3 = final_models[3]

    importance = (
        pd.Series(
            model_3.feature_importances_,
            index=x3.columns,
        )
        .sort_values(
            ascending=False
        )
        .head(8)
    )

    drivers = []

    for name, value in importance.items():

        friendly = (
            name
            .replace("_lag1", " L1")
            .replace("_lag2", " L2")
            .replace("_lag3", " L3")
            .replace("_lag6", " L6")
            .replace("_lag12", " L12")
            .replace("_", " ")
        )

        drivers.append(
            (
                friendly,
                float(value),
            )
        )

    # --------------------------------------------------------
    # Forecast table.
    # --------------------------------------------------------

    rows = []

    for i, horizon in enumerate(HORIZONS):

        rows.append(
            {
                "Horizon": f"{horizon}M",
                "Forecast": smooth_path[i],
                "Lower": lower[i],
                "Upper": upper[i],
            }
        )

    forecast_table = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Regime history.
    # --------------------------------------------------------

    regime_history = df[
        [TARGET]
    ].dropna().copy()

    momentum = (
        regime_history[TARGET]
        .diff(3)
    )

    regime_history[
        "regime"
    ] = np.select(
        [
            (
                (
                    regime_history[TARGET]
                    < 2.5
                )
                & (momentum <= 0)
            ),
            (
                (
                    regime_history[TARGET]
                    >= 3.5
                )
                & (momentum > 0)
            ),
        ],
        [
            "Low / Disinflation",
            "Reflationary",
        ],
        default="Stable",
    )

    # --------------------------------------------------------
    # Chart dataframe.
    # --------------------------------------------------------

    history = (
        df[TARGET]
        .dropna()
        .tail(96)
        .rename("pce_inflation")
        .to_frame()
    )

    future = pd.DataFrame(
        {
            "pce_inflation": np.nan,
            "forecast": smooth_path,
            "lower": lower,
            "upper": upper,
        },
        index=future_dates,
    )

    history["forecast"] = np.nan
    history["lower"] = np.nan
    history["upper"] = np.nan

    chart_df = pd.concat(
        [
            history,
            future,
        ]
    )

    # --------------------------------------------------------
    # Analyst brief.
    # --------------------------------------------------------

    current_pce = float(
        latest_valid[TARGET]
    )

    forecast_3m = float(
        smooth_path[2]
    )

    forecast_6m = float(
        smooth_path[5]
    )

    distance = (
        forecast_3m
        - 2.0
    )

    direction = (
        "higher"
        if forecast_3m > current_pce
        else
        "lower"
        if forecast_3m < current_pce
        else
        "roughly unchanged"
    )

    brief = (
        f"The model expects PCE inflation to be "
        f"{forecast_3m:.2f}% in three months and "
        f"{forecast_6m:.2f}% in six months, versus "
        f"the latest observed rate of "
        f"{current_pce:.2f}%. The near-term direction "
        f"is {direction}, leaving the 3-month estimate "
        f"{distance:+.2f} percentage points from the "
        f"Federal Reserve's 2% objective. The current "
        f"inflation environment is classified as "
        f"{regime.lower()}, with a pressure score of "
        f"{pressure}/100. The 3M XGBoost interval is "
        f"{forecast_3m - radius_3m:.2f}% to "
        f"{forecast_3m + radius_3m:.2f}%."
    )

    result = {
        "selected_model": (
            "Ensemble"
        ),
        "ensemble_weights": {
            "XGBoost": w_xgb,
            "SARIMAX": w_sar,
        },
        "metrics": metrics,
        "ensemble_mae": ensemble_mae,
        "point_forecast_3m": forecast_3m,
        "point_forecast_6m": forecast_6m,
        "lower_3m": forecast_3m - radius_3m,
        "upper_3m": forecast_3m + radius_3m,
        "pressure_score": pressure,
        "regime": regime,
        "confidence": confidence,
        "macro_brief": brief,
        "forecast_table": forecast_table,
        "chart_df": chart_df,
        "xgb_drivers": drivers,
        "regime_history": regime_history,
        "calibration_radius_3m": radius_3m,
        "xgb_actual_3": xgb_actual_3,
        "xgb_pred_3": xgb_pred_3,
    }

    _CACHE[key] = result

    return result


def model_diagnostics(
    forecasts: dict,
) -> dict:

    xgb_a = forecasts[
        "xgb_actual_3"
    ]

    xgb_p = forecasts[
        "xgb_pred_3"
    ]

    residuals = (
        xgb_a
        - xgb_p
    ).dropna()

    coverage_checks = []

    if len(residuals) >= 24:

        split = len(residuals) // 2

        calibration = residuals.iloc[
            :split
        ].abs()

        test = residuals.iloc[
            split:
        ].abs()

        radius = calibration.quantile(
            0.80,
            interpolation="higher",
        )

        coverage_checks = [
            abs_value <= radius
            for abs_value in test
        ]

    empirical_coverage = (
        f"{100 * np.mean(coverage_checks):.1f}%"
        if coverage_checks
        else "N/A"
    )

    perf = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "MAE": round(
                    forecasts["metrics"]["XGBoost"]["MAE"],
                    3,
                ),
                "RMSE": round(
                    forecasts["metrics"]["XGBoost"]["RMSE"],
                    3,
                ),
                "Bias": round(
                    forecasts["metrics"]["XGBoost"]["Bias"],
                    3,
                ),
            },
            {
                "Model": "SARIMAX",
                "MAE": round(
                    forecasts["metrics"]["SARIMAX"]["MAE"],
                    3,
                ),
                "RMSE": round(
                    forecasts["metrics"]["SARIMAX"]["RMSE"],
                    3,
                ),
                "Bias": round(
                    forecasts["metrics"]["SARIMAX"]["Bias"],
                    3,
                ),
            },
            {
                "Model": "Ensemble",
                "MAE": round(
                    forecasts["ensemble_mae"],
                    3,
                ),
                "RMSE": np.nan,
                "Bias": np.nan,
            },
        ]
    )

    coverage = pd.DataFrame(
        [
            {
                "Target": "80%",
                "Observed holdout coverage":
                    empirical_coverage,
                "Method":
                    "Temporal conformal",
            }
        ]
    )

    return {
        "performance": perf,
        "coverage": coverage,
    }


def driver_table(
    forecasts: dict,
) -> pd.DataFrame:

    rows = []

    for name, value in forecasts[
        "xgb_drivers"
    ]:

        rows.append(
            {
                "Driver": name.title(),
                "Importance": round(
                    value,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)
