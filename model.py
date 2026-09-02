from __future__ import annotations

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


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make the model independent of whether an older version of the
    data pipeline produced FRED IDs or friendly internal names.
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

    for fred_name, internal_name in aliases.items():

        if internal_name not in df.columns and fred_name in df.columns:
            df[internal_name] = df[fred_name]

    # Create derived variables if they are missing.
    if "pce_inflation" not in df.columns and "pcepi" in df.columns:
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

def validate_model_data(
    data: pd.DataFrame,
) -> None:

    required = FEATURES

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise KeyError(
            "Forecasting data is missing required columns: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(map(str, data.columns))
        )


# ============================================================
# SUPERVISED DATA
# ============================================================

def supervised_frame(
    data: pd.DataFrame,
    horizon: int,
):
    df = normalize_columns(data)

    validate_model_data(df)

    feature_data = df[FEATURES].copy()

    X = pd.DataFrame(
        index=feature_data.index
    )

    # IMPORTANT:
    # Every predictor is lagged.
    # This prevents current-period information
    # from leaking into the prediction.
    for column in FEATURES:

        for lag in [1, 3, 6, 12]:

            X[
                f"{column}_lag{lag}"
            ] = feature_data[column].shift(lag)

    # Future target.
    y = df[TARGET].shift(-horizon)

    combined = pd.concat(
        [
            X,
            y.rename("target"),
        ],
        axis=1,
    ).dropna()

    if len(combined) < 80:
        raise RuntimeError(
            "Not enough observations after feature engineering."
        )

    X_final = combined.drop(
        columns=["target"]
    )

    y_final = combined["target"]

    return X_final, y_final


# ============================================================
# XGBOOST
# ============================================================

def fit_xgb(
    X: pd.DataFrame,
    y: pd.Series,
) -> XGBRegressor:

    model = XGBRegressor(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# XGBOOST WALK-FORWARD
# ============================================================

def walk_forward_xgb(
    data: pd.DataFrame,
    horizon: int,
    min_train: int = 120,
):

    X, y = supervised_frame(
        data,
        horizon=horizon,
    )

    if len(X) <= min_train:
        raise RuntimeError(
            f"Not enough observations for {horizon}-month XGBoost validation."
        )

    actual = []
    predicted = []
    dates = []

    for i in range(
        min_train,
        len(X),
    ):

        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        X_test = X.iloc[[i]]

        model = fit_xgb(
            X_train,
            y_train,
        )

        prediction = float(
            model.predict(X_test)[0]
        )

        actual.append(
            float(y.iloc[i])
        )

        predicted.append(
            prediction
        )

        dates.append(
            X.index[i]
        )

    return (
        pd.Series(
            actual,
            index=dates,
            name="actual",
        ),
        pd.Series(
            predicted,
            index=dates,
            name="prediction",
        ),
    )


# ============================================================
# SARIMAX WALK-FORWARD
# ============================================================

def walk_forward_sarimax(
    data: pd.DataFrame,
    horizon: int,
    min_train: int = 120,
):

    df = normalize_columns(data)

    if TARGET not in df.columns:
        raise KeyError(
            "PCE inflation target not found."
        )

    y = (
        df[TARGET]
        .dropna()
    )

    actual = []
    predicted = []
    dates = []

    for i in range(
        min_train,
        len(y) - horizon + 1,
    ):

        train = y.iloc[:i]

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

        forecast = fitted.forecast(
            steps=horizon
        )

        target_date = y.index[
            i + horizon - 1
        ]

        actual.append(
            float(
                y.iloc[
                    i + horizon - 1
                ]
            )
        )

        predicted.append(
            float(
                forecast.iloc[-1]
            )
        )

        dates.append(
            target_date
        )

    return (
        pd.Series(
            actual,
            index=dates,
            name="actual",
        ),
        pd.Series(
            predicted,
            index=dates,
            name="prediction",
        ),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
):

    df = pd.concat(
        [actual, predicted],
        axis=1,
    ).dropna()

    return {
        "MAE": float(
            mean_absolute_error(
                df.iloc[:, 0],
                df.iloc[:, 1],
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    df.iloc[:, 0],
                    df.iloc[:, 1],
                )
            )
        ),
        "Bias": float(
            (
                df.iloc[:, 1]
                - df.iloc[:, 0]
            ).mean()
        ),
    }


# ============================================================
# CONFORMAL PREDICTION
# ============================================================

def conformal_radius(
    actual: pd.Series,
    predicted: pd.Series,
    coverage: float = 0.80,
):

    residuals = (
        actual
        .sub(predicted)
        .abs()
        .dropna()
        .tail(60)
    )

    if residuals.empty:
        return 0.75

    return float(
        residuals.quantile(
            coverage,
            interpolation="higher",
        )
    )


# ============================================================
# PRESSURE SCORE
# ============================================================

def calculate_pressure_score(
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

    return max(
        0,
        min(
            100,
            score,
        ),
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

    latest = target.iloc[-1]

    momentum = (
        target.diff(3)
        .iloc[-1]
    )

    if (
        latest < 2.5
        and momentum <= 0
    ):
        return "Low / Disinflationary"

    if (
        latest >= 3.5
        and momentum > 0
    ):
        return "Reflationary"

    return "Stable"


# ============================================================
# MAIN FORECAST FUNCTION
# ============================================================

def build_forecasts(
    data: pd.DataFrame,
) -> dict:

    df = normalize_columns(data)

    validate_model_data(df)

    # --------------------------------------------------------
    # 3 MONTH VALIDATION
    # --------------------------------------------------------

    xgb_actual_3, xgb_pred_3 = (
        walk_forward_xgb(
            df,
            horizon=3,
        )
    )

    sar_actual_3, sar_pred_3 = (
        walk_forward_sarimax(
            df,
            horizon=3,
        )
    )

    xgb_metrics = calculate_metrics(
        xgb_actual_3,
        xgb_pred_3,
    )

    sar_metrics = calculate_metrics(
        sar_actual_3,
        sar_pred_3,
    )

    metrics = {
        "XGBoost": xgb_metrics,
        "SARIMAX": sar_metrics,
    }

    selected_model = min(
        metrics,
        key=lambda name: metrics[name]["MAE"],
    )

    # --------------------------------------------------------
    # FINAL 3M XGBOOST
    # --------------------------------------------------------

    X3, y3 = supervised_frame(
        df,
        horizon=3,
    )

    final_xgb_3 = fit_xgb(
        X3,
        y3,
    )

    point_3m = float(
        final_xgb_3.predict(
            X3.iloc[[-1]]
        )[0]
    )

    radius_3m = conformal_radius(
        xgb_actual_3,
        xgb_pred_3,
        coverage=0.80,
    )

    # --------------------------------------------------------
    # FINAL 6M XGBOOST
    # --------------------------------------------------------

    X6, y6 = supervised_frame(
        df,
        horizon=6,
    )

    final_xgb_6 = fit_xgb(
        X6,
        y6,
    )

    point_6m = float(
        final_xgb_6.predict(
            X6.iloc[[-1]]
        )[0]
    )

    # --------------------------------------------------------
    # IMPORTANCE
    # --------------------------------------------------------

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

    drivers = []

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

        drivers.append(
            (
                friendly,
                float(value),
            )
        )

    # --------------------------------------------------------
    # LATEST OBSERVATION
    # --------------------------------------------------------

    target_data = df.dropna(
        subset=[TARGET]
    )

    latest = target_data.iloc[-1]

    pressure = calculate_pressure_score(
        latest
    )

    regime = detect_regime(
        df
    )

    if metrics[selected_model]["MAE"] < 0.35:
        confidence = "High"

    elif metrics[selected_model]["MAE"] < 0.55:
        confidence = "Medium"

    else:
        confidence = "Low"

    # --------------------------------------------------------
    # HISTORY + FORECAST CHART
    # --------------------------------------------------------

    history = (
        target_data[
            [TARGET]
        ]
        .tail(84)
        .rename(
            columns={
                TARGET:
                "pce_inflation"
            }
        )
    )

    future_dates = pd.date_range(
        history.index[-1]
        + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )

    forecast_values = np.concatenate(
        [
            np.linspace(
                float(latest[TARGET]),
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

    chart_future = pd.DataFrame(
        {
            "pce_inflation": np.nan,
            "forecast": forecast_values,
            "lower": (
                forecast_values
                - radius_3m
            ),
            "upper": (
                forecast_values
                + radius_3m
            ),
        },
        index=future_dates,
    )

    chart_history = history.copy()

    chart_history[
        "forecast"
    ] = np.nan

    chart_history[
        "lower"
    ] = np.nan

    chart_history[
        "upper"
    ] = np.nan

    chart_df = pd.concat(
        [
            chart_history,
            chart_future,
        ]
    )

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    forecast_table = pd.DataFrame(
        [
            [
                "Current",
                float(latest[TARGET]),
                np.nan,
                np.nan,
            ],
            [
                "3M",
                point_3m,
                point_3m - radius_3m,
                point_3m + radius_3m,
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

    # --------------------------------------------------------
    # REGIME HISTORY
    # --------------------------------------------------------

    regime_history = target_data[
        [TARGET]
    ].copy()

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
                & (
                    momentum <= 0
                )
            ),
            (
                (
                    regime_history[TARGET]
                    >= 3.5
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

    # --------------------------------------------------------
    # BRIEF
    # --------------------------------------------------------

    brief = (
        f"The model forecasts PCE inflation at "
        f"{point_3m:.2f}% in three months and "
        f"{point_6m:.2f}% in six months. "
        f"The 3M forecast is "
        f"{point_3m - 2:+.2f} percentage points "
        f"from the Federal Reserve's 2% objective. "
        f"The current inflation environment is "
        f"{regime.lower()}, with an inflation-pressure "
        f"score of {pressure}/100. "
        f"The 80% XGBoost prediction range is "
        f"{point_3m - radius_3m:.2f}% to "
        f"{point_3m + radius_3m:.2f}%, based on "
        f"temporal conformal calibration of historical "
        f"walk-forward errors."
    )

    return {
        "selected_model": selected_model,
        "point_forecast_3m": point_3m,
        "point_forecast_6m": point_6m,
        "lower_3m": point_3m - radius_3m,
        "upper_3m": point_3m + radius_3m,
        "pressure_score": pressure,
        "regime": regime,
        "confidence": confidence,
        "macro_brief": brief,
        "chart_df": chart_df,
        "forecast_table": forecast_table,
        "xgb_drivers": drivers,
        "regime_history": regime_history,
        "raw_metrics": metrics,
        "xgb_actual": xgb_actual_3,
        "xgb_pred": xgb_pred_3,
        "sar_actual": sar_actual_3,
        "sar_pred": sar_pred_3,
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def model_diagnostics(
    forecasts: dict,
) -> dict:

    performance = pd.DataFrame(
        [
            {
                "Model": model_name,
                "MAE": round(
                    metric["MAE"],
                    4,
                ),
                "RMSE": round(
                    metric["RMSE"],
                    4,
                ),
                "Bias": round(
                    metric["Bias"],
                    4,
                ),
            }
            for model_name, metric
            in forecasts["raw_metrics"].items()
        ]
    ).sort_values(
        "MAE"
    )

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
        20,
        len(residuals),
    ):

        calibration = residuals.iloc[
            max(0, i - 60):i
        ].abs()

        q = calibration.quantile(
            0.80,
            interpolation="higher",
        )

        coverage_results.append(
            abs(
                residuals.iloc[i]
            ) <= q
        )

    empirical_coverage = (
        f"{100 * np.mean(coverage_results):.1f}%"
        if coverage_results
        else "N/A"
    )

    coverage = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Target coverage": "80%",
                "Empirical coverage":
                    empirical_coverage,
                "Calibration":
                    "Temporal conformal",
            }
        ]
    )

    return {
        "performance": performance,
        "coverage": coverage,
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
                "Driver": name.title(),
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
