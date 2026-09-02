from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# ============================================================
# SETTINGS
# ============================================================

TARGET = "pce_inflation"

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

FED_TARGET = 2.0


# ============================================================
# INTERNAL CACHE
# ============================================================

_MODEL_CACHE: Dict[str, dict] = {}


def _data_key(data: pd.DataFrame) -> str:
    """
    Create a lightweight cache key from the latest data state.
    This prevents retraining every time Streamlit reruns.
    """

    recent = data.tail(24).copy()

    raw = pd.util.hash_pandas_object(
        recent,
        index=True,
    ).values.tobytes()

    return hashlib.md5(raw).hexdigest()


# ============================================================
# COLUMN COMPATIBILITY
# ============================================================

def normalize_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Supports both FRED IDs and the project's internal
    friendly column names.
    """

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

    for fred_id, internal_name in aliases.items():

        if (
            internal_name not in df.columns
            and fred_id in df.columns
        ):
            df[internal_name] = df[fred_id]

    # --------------------------------------------------------
    # Derived target / features
    # --------------------------------------------------------

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


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(
    data: pd.DataFrame,
) -> None:

    df = normalize_columns(data)

    missing = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing model columns: "
            + ", ".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(map(str, df.columns))
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_columns(data)

    validate_columns(df)

    base = df[FEATURES].copy()

    X = pd.DataFrame(
        index=base.index
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only lagged values are used.
    # This prevents the current-period target
    # from leaking into the prediction.
    # --------------------------------------------------------

    for column in FEATURES:

        for lag in [1, 3, 6, 12]:

            X[
                f"{column}_lag{lag}"
            ] = base[column].shift(lag)

    return X


# ============================================================
# SUPERVISED DATA
# ============================================================

def supervised_data(
    data: pd.DataFrame,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series]:

    df = normalize_columns(data)

    X = create_features(df)

    # Direct multi-horizon target.
    # Horizon = 3 means predict PCE three months ahead.
    y = df[TARGET].shift(-horizon)

    combined = pd.concat(
        [
            X,
            y.rename("target"),
        ],
        axis=1,
    ).dropna()

    if len(combined) < 100:

        raise RuntimeError(
            f"Only {len(combined)} usable observations "
            f"remain for the {horizon}-month model."
        )

    return (
        combined.drop(
            columns=["target"]
        ),
        combined["target"],
    )


# ============================================================
# XGBOOST
# ============================================================

def fit_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
) -> XGBRegressor:

    model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.80,
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


# ============================================================
# RANDOM FOREST
# ============================================================

def fit_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomForestRegressor:

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=2,
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# XGBOOST CHRONOLOGICAL HOLDOUT
# ============================================================

def xgb_holdout(
    data: pd.DataFrame,
    horizon: int,
    holdout_months: int = 24,
):
    """
    Fast chronological validation.

    One model is trained on the historical training window
    and evaluated against a final contiguous holdout window.

    This is intentionally much faster than fitting a new model
    at every historical month.
    """

    X, y = supervised_data(
        data,
        horizon,
    )

    if len(X) <= holdout_months + 20:
        holdout_months = max(
            12,
            len(X) // 5,
        )

    split = len(X) - holdout_months

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = fit_xgboost(
        X_train,
        y_train,
    )

    predictions = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="prediction",
    )

    actual = y_test.rename(
        "actual"
    )

    return (
        actual,
        predictions,
    )


# ============================================================
# SARIMAX FAST HOLDOUT
# ============================================================

def sarimax_holdout(
    data: pd.DataFrame,
    horizon: int,
    holdout_months: int = 24,
):

    df = normalize_columns(data)

    y = (
        df[TARGET]
        .dropna()
    )

    if len(y) <= holdout_months + 30:
        holdout_months = max(
            12,
            len(y) // 5,
        )

    split = len(y) - holdout_months

    train = y.iloc[:split]
    test = y.iloc[split:]

    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(
            1,
            0,
            1,
            12,
        ),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False
    )

    predictions = fitted.forecast(
        steps=len(test)
    )

    predictions = pd.Series(
        predictions.values,
        index=test.index,
        name="prediction",
    )

    return (
        test.rename("actual"),
        predictions,
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:

    df = pd.concat(
        [
            actual,
            predicted,
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
        }

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


# ============================================================
# CONFORMAL CALIBRATION
# ============================================================

def conformal_radius(
    actual: pd.Series,
    predicted: pd.Series,
    coverage: float = 0.80,
) -> float:

    residuals = (
        actual
        - predicted
    ).abs().dropna()

    if residuals.empty:
        return 0.75

    # Use the chronological holdout residuals.
    return float(
        residuals.quantile(
            coverage,
            interpolation="higher",
        )
    )


# ============================================================
# PRESSURE SCORE
# ============================================================

def pressure_score(
    row: pd.Series,
) -> int:

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

    return int(
        max(
            0,
            min(
                100,
                score,
            ),
        )
    )


# ============================================================
# REGIME
# ============================================================

def detect_regime(
    data: pd.DataFrame,
) -> str:

    df = normalize_columns(data)

    target = (
        df[TARGET]
        .dropna()
    )

    if len(target) < 4:
        return "Insufficient data"

    current = float(
        target.iloc[-1]
    )

    momentum = float(
        target.diff(3).iloc[-1]
    )

    if (
        current < 2.5
        and momentum <= 0
    ):
        return "Low / Disinflationary"

    if (
        current >= 3.5
        and momentum > 0
    ):
        return "Reflationary"

    return "Stable"


# ============================================================
# FINAL FORECAST
# ============================================================

def build_forecasts(
    data: pd.DataFrame,
    fast_mode: bool = True,
) -> dict:

    df = normalize_columns(data)

    validate_columns(df)

    cache_key = (
        _data_key(df)
        + f"_{fast_mode}"
    )

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # ========================================================
    # VALIDATION
    # ========================================================

    xgb_actual_3, xgb_pred_3 = xgb_holdout(
        df,
        horizon=3,
        holdout_months=24,
    )

    sar_actual_3, sar_pred_3 = sarimax_holdout(
        df,
        horizon=3,
        holdout_months=24,
    )

    xgb_metrics = metrics(
        xgb_actual_3,
        xgb_pred_3,
    )

    sar_metrics = metrics(
        sar_actual_3,
        sar_pred_3,
    )

    model_metrics = {
        "XGBoost": xgb_metrics,
        "SARIMAX": sar_metrics,
    }

    # Select model based on chronological holdout MAE.
    if (
        np.isfinite(xgb_metrics["MAE"])
        and (
            xgb_metrics["MAE"]
            <= sar_metrics["MAE"]
        )
    ):
        selected_model = "XGBoost"
    else:
        selected_model = "SARIMAX"

    # ========================================================
    # FINAL XGBOOST 3M
    # ========================================================

    X3, y3 = supervised_data(
        df,
        horizon=3,
    )

    final_xgb_3 = fit_xgboost(
        X3,
        y3,
    )

    point_3m = float(
        final_xgb_3.predict(
            X3.iloc[[-1]]
        )[0]
    )

    conformal_3m = conformal_radius(
        xgb_actual_3,
        xgb_pred_3,
        coverage=0.80,
    )

    # ========================================================
    # FINAL XGBOOST 6M
    # ========================================================

    X6, y6 = supervised_data(
        df,
        horizon=6,
    )

    final_xgb_6 = fit_xgboost(
        X6,
        y6,
    )

    point_6m = float(
        final_xgb_6.predict(
            X6.iloc[[-1]]
        )[0]
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = (
        pd.Series(
            final_xgb_3.feature_importances_,
            index=X3.columns,
        )
        .sort_values(
            ascending=False
        )
        .head(8)
    )

    driver_list = []

    for feature, value in importance.items():

        friendly = (
            feature
            .replace(
                "_lag1",
                " lag 1",
            )
            .replace(
                "_lag3",
                " lag 3",
            )
            .replace(
                "_lag6",
                " lag 6",
            )
            .replace(
                "_lag12",
                " lag 12",
            )
            .replace(
                "_",
                " ",
            )
        )

        driver_list.append(
            (
                friendly,
                float(value),
            )
        )

    # ========================================================
    # CURRENT STATE
    # ========================================================

    target = (
        df[TARGET]
        .dropna()
    )

    latest = target.iloc[-1]

    current_row = df.dropna(
        subset=[
            TARGET,
            "core_pce_inflation",
        ]
    ).iloc[-1]

    p_score = pressure_score(
        current_row
    )

    regime = detect_regime(
        df
    )

    if model_metrics[
        selected_model
    ]["MAE"] < 0.35:

        confidence = "High"

    elif model_metrics[
        selected_model
    ]["MAE"] < 0.55:

        confidence = "Medium"

    else:

        confidence = "Low"

    # ========================================================
    # CHART DATA
    # ========================================================

    history = (
        target
        .tail(84)
        .rename(
            "pce_inflation"
        )
        .to_frame()
    )

    last_history_date = (
        history.index[-1]
    )

    future_dates = pd.date_range(
        last_history_date
        + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )

    # Bridge current → 3M → 6M.
    forecast_path = np.concatenate(
        [
            np.linspace(
                float(latest),
                point_3m,
                3,
            ),
            np.linspace(
                point_3m,
                point_6m,
                3,
            ),
        ]
    )

    lower_path = (
        forecast_path
        - conformal_3m
    )

    upper_path = (
        forecast_path
        + conformal_3m
    )

    future_df = pd.DataFrame(
        {
            "pce_inflation": np.nan,
            "forecast": forecast_path,
            "lower": lower_path,
            "upper": upper_path,
        },
        index=future_dates,
    )

    history[
        "forecast"
    ] = np.nan

    history[
        "lower"
    ] = np.nan

    history[
        "upper"
    ] = np.nan

    chart_df = pd.concat(
        [
            history,
            future_df,
        ]
    )

    # ========================================================
    # FORECAST TABLE
    # ========================================================

    forecast_table = pd.DataFrame(
        [
            [
                "Current",
                float(latest),
                np.nan,
                np.nan,
            ],
            [
                "3M",
                point_3m,
                point_3m
                - conformal_3m,
                point_3m
                + conformal_3m,
            ],
            [
                "6M",
                point_6m,
                np.nan,
                np.nan,
            ],
        ],
        columns=[
            "Horizon",
            "Forecast",
            "Lower",
            "Upper",
        ],
    )

    # ========================================================
    # REGIME HISTORY
    # ========================================================

    regime_history = (
        df[
            [TARGET]
        ]
        .dropna()
        .copy()
    )

    momentum = (
        regime_history[
            TARGET
        ].diff(3)
    )

    regime_history[
        "regime"
    ] = np.select(
        [
            (
                (
                    regime_history[
                        TARGET
                    ] < 2.5
                )
                & (
                    momentum <= 0
                )
            ),
            (
                (
                    regime_history[
                        TARGET
                    ] >= 3.5
                )
                & (
                    momentum > 0
                )
            ),
        ],
        [
            "Low / Disinflationary",
            "Reflationary",
        ],
        default="Stable",
    )

    # ========================================================
    # MACRO BRIEF
    # ========================================================

    distance_to_target = (
        point_3m
        - FED_TARGET
    )

    direction = (
        "higher"
        if point_3m > float(latest)
        else "lower"
        if point_3m < float(latest)
        else "unchanged"
    )

    macro_brief = (
        f"The model forecasts PCE inflation at "
        f"{point_3m:.2f}% in three months and "
        f"{point_6m:.2f}% in six months. "
        f"Compared with the current inflation rate of "
        f"{float(latest):.2f}%, the 3-month forecast is "
        f"{direction}. The 3-month forecast is "
        f"{distance_to_target:+.2f} percentage points "
        f"from the Federal Reserve's 2% objective. "
        f"The current inflation environment is "
        f"{regime.lower()}, with an inflation-pressure "
        f"score of {p_score}/100. "
        f"The XGBoost 80% prediction range for the "
        f"3-month forecast is "
        f"{point_3m - conformal_3m:.2f}% to "
        f"{point_3m + conformal_3m:.2f}%. "
        f"This is a research-model estimate, not an "
        f"official Federal Reserve forecast."
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "selected_model": selected_model,

        "point_forecast_3m":
            point_3m,

        "point_forecast_6m":
            point_6m,

        "lower_3m":
            point_3m
            - conformal_3m,

        "upper_3m":
            point_3m
            + conformal_3m,

        "pressure_score":
            p_score,

        "regime":
            regime,

        "confidence":
            confidence,

        "macro_brief":
            macro_brief,

        "chart_df":
            chart_df,

        "forecast_table":
            forecast_table,

        "xgb_drivers":
            driver_list,

        "regime_history":
            regime_history,

        "raw_metrics":
            model_metrics,

        "xgb_actual":
            xgb_actual_3,

        "xgb_pred":
            xgb_pred_3,

        "sar_actual":
            sar_actual_3,

        "sar_pred":
            sar_pred_3,
    }

    _MODEL_CACHE[
        cache_key
    ] = result

    return result


# ============================================================
# DIAGNOSTICS
# ============================================================

def model_diagnostics(
    forecasts: dict,
) -> dict:

    performance_rows = []

    for model_name, metric in (
        forecasts[
            "raw_metrics"
        ].items()
    ):

        performance_rows.append(
            {
                "Model":
                    model_name,

                "MAE":
                    round(
                        metric["MAE"],
                        4,
                    ),

                "RMSE":
                    round(
                        metric["RMSE"],
                        4,
                    ),

                "Bias":
                    round(
                        metric["Bias"],
                        4,
                    ),

                "Validation":
                    "Chronological holdout",
            }
        )

    performance = (
        pd.DataFrame(
            performance_rows
        )
        .sort_values(
            "MAE"
        )
    )

    # --------------------------------------------------------
    # Conformal coverage on the historical holdout.
    # --------------------------------------------------------

    actual = forecasts[
        "xgb_actual"
    ]

    predicted = forecasts[
        "xgb_pred"
    ]

    residuals = (
        actual
        - predicted
    ).dropna()

    coverage_results = []

    for i in range(
        12,
        len(residuals),
    ):

        calibration = (
            residuals.iloc[
                :i
            ].abs()
        )

        radius = calibration.quantile(
            0.80,
            interpolation="higher",
        )

        covered = (
            abs(
                residuals.iloc[i]
            )
            <= radius
        )

        coverage_results.append(
            covered
        )

    if coverage_results:

        empirical = (
            100
            * np.mean(
                coverage_results
            )
        )

        empirical_text = (
            f"{empirical:.1f}%"
        )

    else:

        empirical_text = "N/A"

    coverage = pd.DataFrame(
        [
            {
                "Model":
                    "XGBoost",

                "Target coverage":
                    "80%",

                "Empirical coverage":
                    empirical_text,

                "Calibration":
                    "Temporal holdout conformal",
            }
        ]
    )

    return {
        "performance":
            performance,

        "coverage":
            coverage,
    }


# ============================================================
# DRIVER TABLE
# ============================================================

def driver_table(
    forecasts: dict,
) -> pd.DataFrame:

    drivers = forecasts[
        "xgb_drivers"
    ]

    if not drivers:

        return pd.DataFrame(
            columns=[
                "Driver",
                "Model importance",
                "Relative strength",
            ]
        )

    max_value = max(
        value
        for _, value
        in drivers
    )

    rows = []

    for name, value in drivers:

        rows.append(
            {
                "Driver":
                    name.title(),

                "Model importance":
                    round(
                        value,
                        4,
                    ),

                "Relative strength":
                    round(
                        value / max_value,
                        2,
                    )
                    if max_value
                    else 0,
            }
        )

    return pd.DataFrame(
        rows
    )
