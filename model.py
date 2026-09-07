from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
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

LAGS = (1, 2, 3, 6, 12)

HORIZONS = (1, 2, 3, 4, 5, 6)

TOP_N_FEATURES = 20

# Pooled, coarse walk-forward: 3 independent refit origins, each
# scored on the 12 months immediately after it. Cheaper than a true
# month-by-month walk-forward (~180 refits) but far less noisy than
# a single holdout window.
EVAL_FOLDS = 3
EVAL_FOLD_LEN = 12
EVAL_SPAN = EVAL_FOLDS * EVAL_FOLD_LEN

# Candidate SARIMAX (order, seasonal_order, trend) triples.
# pce_inflation is already a 12-month (YoY) transform of the price
# index, so it's already deseasonalized -- no candidate carries a
# seasonal term (an earlier seasonal candidate was tried and
# consistently failed to converge on this series, never won a
# selection round, and was removed rather than left as dead weight).
# (0,1,0)/trend="n" is a pure random walk: literally the naive
# persistence forecast expressed as an ARIMA model, so if nothing
# else earns its keep, order search can fall back to it. Selection
# happens on a validation slice strictly before EVAL_SPAN, so the
# choice of order can't leak into reported accuracy.
SARIMAX_ORDER_CANDIDATES: List[Tuple[Tuple, Tuple, str]] = [
    ((0, 1, 0), (0, 0, 0, 0), "n"),
    ((0, 1, 0), (0, 0, 0, 0), "c"),
    ((1, 0, 0), (0, 0, 0, 0), "c"),
    ((0, 0, 1), (0, 0, 0, 0), "c"),
    ((1, 0, 1), (0, 0, 0, 0), "c"),
    ((1, 1, 0), (0, 0, 0, 0), "c"),
    ((0, 1, 1), (0, 0, 0, 0), "c"),
    ((2, 0, 1), (0, 0, 0, 0), "c"),
]

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
    columns: Optional[List[str]] = None,
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

    for column in BASE_FEATURES:
        for lag in LAGS:
            X[f"{column}_lag{lag}"] = (
                base[column].shift(lag)
            )

    if columns is not None:
        X = X[columns]

    return X


def _supervised(
    data: pd.DataFrame,
    horizon: int,
    columns: Optional[List[str]] = None,
):
    """
    Builds the XGBoost training frame around the CHANGE in inflation
    (target = level[t+h] - level[t]), not the raw level. A model that
    predicts zero change everywhere degenerates exactly to the naive
    persistence forecast, so this can only underperform naive by the
    model's own error on the change -- it can't be arbitrarily worse
    the way predicting the level directly can.
    """
    df = normalize_columns(data)
    X = _feature_matrix(df, columns=columns)

    level = df[TARGET]
    future_level = level.shift(-horizon)
    delta = future_level - level

    merged = pd.concat(
        [
            X,
            level.rename("base_level"),
            future_level.rename("future_level"),
            delta.rename("target"),
        ],
        axis=1,
    ).dropna()

    if len(merged) < 100:
        raise RuntimeError(
            f"Not enough observations for {horizon}M model."
        )

    X_final = merged.drop(
        columns=["base_level", "future_level", "target"]
    )

    return (
        X_final,
        merged["target"],
        merged["base_level"],
        merged["future_level"],
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


def _screen_features(
    data: pd.DataFrame,
    horizon: int = 3,
    top_n: int = TOP_N_FEATURES,
    exclude_months: int = EVAL_SPAN,
) -> List[str]:
    """
    Rank lagged features by absolute correlation with the CHANGE in
    inflation (what XGBoost is actually trained on) and keep the
    top_n. Screening only ever looks at data before the reported
    evaluation window, so the feature list can't be chosen using the
    period we later score accuracy on.
    """
    X_full = _feature_matrix(data)
    df = normalize_columns(data)
    level = df[TARGET]
    delta = level.shift(-horizon) - level

    merged = pd.concat(
        [X_full, delta.rename("target")],
        axis=1,
    ).dropna()

    if len(merged) <= exclude_months + 20:
        return list(X_full.columns)

    screening_slice = merged.iloc[: len(merged) - exclude_months]

    corr = (
        screening_slice.drop(columns="target")
        .corrwith(screening_slice["target"])
        .abs()
        .dropna()
        .sort_values(ascending=False)
    )

    selected = list(corr.head(top_n).index)

    return selected if selected else list(X_full.columns)


def _select_sarimax_order(
    y: pd.Series,
    candidates: List[Tuple[Tuple, Tuple, str]] = SARIMAX_ORDER_CANDIDATES,
    horizon: int = 3,
    val_len: int = 12,
    exclude_months: int = EVAL_SPAN,
) -> Tuple[Tuple, Tuple, str]:
    """
    Pick the SARIMAX order with the lowest out-of-sample MAE on a
    validation slice sitting strictly before the reported evaluation
    folds, so order selection can't leak into reported accuracy.
    """
    n = len(y)
    val_end = n - exclude_months
    val_start = val_end - val_len

    if val_start < 80:
        return candidates[0]

    train = y.iloc[:val_start]
    test = y.iloc[val_start:val_end]

    best_choice = candidates[0]
    best_mae = np.inf

    for order, seasonal_order, trend in candidates:
        try:
            fitted = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)

            if not fitted.mle_retvals.get("converged", True):
                continue

            fc = fitted.forecast(
                steps=val_len + horizon - 1
            )

            aligned = fc.iloc[
                horizon - 1: horizon - 1 + val_len
            ]

            mae = mean_absolute_error(
                test.values,
                aligned.values,
            )
        except Exception:
            continue

        if mae < best_mae:
            best_mae = mae
            best_choice = (order, seasonal_order, trend)

    return best_choice


def _xgb_pooled_eval(
    data: pd.DataFrame,
    horizon: int,
    columns: List[str],
    folds: int = EVAL_FOLDS,
    fold_len: int = EVAL_FOLD_LEN,
    min_train: int = 100,
):
    X, y_delta, base_level, future_level = _supervised(
        data,
        horizon=horizon,
        columns=columns,
    )

    n = len(X)

    actual_parts = []
    pred_parts = []

    for k in range(folds):
        test_end = n - fold_len * k
        test_start = test_end - fold_len
        train_end = test_start

        if train_end < min_train or test_start < 0:
            continue

        model = _fit_xgb(
            X.iloc[:train_end],
            y_delta.iloc[:train_end],
        )

        X_test = X.iloc[test_start:test_end]

        delta_pred = pd.Series(
            model.predict(X_test),
            index=X_test.index,
        )

        level_pred = (
            base_level.iloc[test_start:test_end]
            + delta_pred
        )

        actual_parts.append(future_level.iloc[test_start:test_end])
        pred_parts.append(level_pred)

    if not actual_parts:
        raise RuntimeError(
            f"Not enough history for {horizon}M XGBoost evaluation."
        )

    actual = pd.concat(actual_parts).sort_index()
    predicted = pd.concat(pred_parts).sort_index()

    # These are indexed by origin date (the row the prediction was
    # made from). Shift to the target date being predicted so this
    # aligns with SARIMAX's target-date-indexed output.
    actual.index = actual.index + pd.DateOffset(months=horizon)
    predicted.index = predicted.index + pd.DateOffset(months=horizon)

    return actual.rename("actual"), predicted.rename("prediction")


def _sarimax_pooled_eval(
    data: pd.DataFrame,
    horizon: int,
    order: Tuple,
    seasonal_order: Tuple,
    trend: str,
    folds: int = EVAL_FOLDS,
    fold_len: int = EVAL_FOLD_LEN,
    min_train: int = 100,
):
    df = normalize_columns(data)
    y = df[TARGET].dropna()
    n = len(y)

    actual_parts = []
    pred_parts = []

    for k in range(folds):
        test_end = n - fold_len * k
        test_start = test_end - fold_len
        train_end = test_start

        if train_end < min_train or test_start < 0:
            continue

        train = y.iloc[:train_end]
        test = y.iloc[test_start:test_end]

        try:
            fitted = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)

            if not fitted.mle_retvals.get("converged", True):
                continue
        except Exception:
            continue

        fc = fitted.forecast(
            steps=fold_len + horizon - 1
        )

        aligned = pd.Series(
            fc.iloc[
                horizon - 1: horizon - 1 + fold_len
            ].values,
            index=test.index,
        )

        actual_parts.append(test)
        pred_parts.append(aligned)

    if not actual_parts:
        raise RuntimeError(
            f"Not enough history for {horizon}M SARIMAX evaluation."
        )

    actual = pd.concat(actual_parts).sort_index()
    predicted = pd.concat(pred_parts).sort_index()

    return actual.rename("actual"), predicted.rename("prediction")


def _naive_pooled_eval(
    data: pd.DataFrame,
    horizon: int,
    folds: int = EVAL_FOLDS,
    fold_len: int = EVAL_FOLD_LEN,
    min_train: int = 100,
):
    """
    The floor every model has to clear: predict that inflation
    `horizon` months from now equals inflation today. Scored on the
    exact same pooled folds as XGBoost/SARIMAX for a fair comparison.
    """
    df = normalize_columns(data)
    y = df[TARGET].dropna()
    n = len(y)

    actual_parts = []
    pred_parts = []

    for k in range(folds):
        test_end = n - fold_len * k
        test_start = test_end - fold_len
        train_end = test_start

        if train_end < min_train or test_start < 0:
            continue

        test_dates = y.index[test_start:test_end]

        actual_parts.append(y.loc[test_dates])
        pred_parts.append(y.shift(horizon).loc[test_dates])

    if not actual_parts:
        raise RuntimeError(
            f"Not enough history for {horizon}M naive evaluation."
        )

    actual = pd.concat(actual_parts).sort_index()
    predicted = pd.concat(pred_parts).sort_index()

    return actual.rename("actual"), predicted.rename("prediction")


def _sarimax_forecast_series(
    data: pd.DataFrame,
    steps: int,
    order: Tuple,
    seasonal_order: Tuple,
    trend: str,
) -> pd.Series:
    df = normalize_columns(data)
    y = df[TARGET].dropna()

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False,
        maxiter=200,
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


def _friendly_name(name: str) -> str:
    return (
        name
        .replace("_lag1", " L1")
        .replace("_lag2", " L2")
        .replace("_lag3", " L3")
        .replace("_lag6", " L6")
        .replace("_lag12", " L12")
        .replace("_", " ")
    )


def _shap_contributions(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    X_today: pd.DataFrame,
    top_n: int = 8,
):
    """
    Real SHAP model contributions (not gain-based feature
    importance) to the predicted CHANGE in inflation. These are
    model attributions, not causal effects.

    Global: mean absolute SHAP value per feature across the
    training frame -- what the model leans on in general.

    Local: signed SHAP value per feature for today's feature row
    -- what's actually pushing the current forecast up or down.
    """
    explainer = shap.TreeExplainer(model)

    train_shap = explainer.shap_values(X_train)

    global_importance = (
        pd.Series(
            np.abs(train_shap).mean(axis=0),
            index=X_train.columns,
        )
        .sort_values(ascending=False)
        .head(top_n)
    )

    global_pairs = [
        (name, float(value))
        for name, value in global_importance.items()
    ]

    today_shap = explainer.shap_values(X_today)[0]

    local_importance = (
        pd.Series(
            today_shap,
            index=X_today.columns,
        )
        .reindex(
            pd.Series(today_shap, index=X_today.columns)
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
    )

    local_pairs = [
        (name, float(value))
        for name, value in local_importance.items()
    ]

    return global_pairs, local_pairs


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
    # Feature screening and SARIMAX order selection. Both only
    # ever see data before the reported evaluation window.
    # --------------------------------------------------------

    selected_features = _screen_features(
        df,
        horizon=3,
    )

    y_raw = df[TARGET].dropna()

    order, seasonal_order, trend = _select_sarimax_order(
        y_raw,
        horizon=3,
    )

    # --------------------------------------------------------
    # Pooled multi-window evaluation: 3 refit origins x 12
    # months each = 36 months of chronological, non-overlapping
    # out-of-sample accuracy, instead of one noisy 24M window.
    # Naive persistence is scored on the same folds as the real
    # floor every model has to beat to earn a place on screen.
    # --------------------------------------------------------

    naive_actual_3, naive_pred_3 = _naive_pooled_eval(
        df,
        horizon=3,
    )

    xgb_actual_3, xgb_pred_3 = _xgb_pooled_eval(
        df,
        horizon=3,
        columns=selected_features,
    )

    sar_actual_3, sar_pred_3 = _sarimax_pooled_eval(
        df,
        horizon=3,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
    )

    naive_metrics = _metric_bundle(
        naive_actual_3,
        naive_pred_3,
    )

    xgb_metrics = _metric_bundle(
        xgb_actual_3,
        xgb_pred_3,
    )

    sar_metrics = _metric_bundle(
        sar_actual_3,
        sar_pred_3,
    )

    # Inverse-MAE weights across all three candidates, including
    # Naive itself. This is shrinkage-toward-persistence: a
    # standard forecast-combination technique, not just averaging
    # for its own sake -- if XGBoost/SARIMAX add no real signal
    # beyond persistence, their weight collapses toward zero and
    # the blend collapses toward Naive; if they do add signal, the
    # blend can beat every individual component.
    inv_naive = 1 / max(naive_metrics["MAE"], 1e-6)
    inv_xgb = 1 / max(xgb_metrics["MAE"], 1e-6)
    inv_sar = 1 / max(sar_metrics["MAE"], 1e-6)

    total_inv = inv_naive + inv_xgb + inv_sar

    w_naive = inv_naive / total_inv
    w_xgb = inv_xgb / total_inv
    w_sar = inv_sar / total_inv

    # --------------------------------------------------------
    # Real ensemble accuracy: blend the three pooled OOS
    # prediction series on their common dates and score the
    # blend itself, rather than averaging the components' own
    # MAEs (which can never show whether blending helped).
    # --------------------------------------------------------

    common_dates = (
        naive_pred_3.index
        .intersection(xgb_pred_3.index)
        .intersection(sar_pred_3.index)
    )

    if len(common_dates):
        ensemble_actual = naive_actual_3.loc[common_dates]
        ensemble_pred = (
            w_naive * naive_pred_3.loc[common_dates]
            + w_xgb * xgb_pred_3.loc[common_dates]
            + w_sar * sar_pred_3.loc[common_dates]
        )
        ensemble_metrics = _metric_bundle(
            ensemble_actual.rename("actual"),
            ensemble_pred.rename("prediction"),
        )
    else:
        ensemble_actual = naive_actual_3
        ensemble_pred = naive_pred_3
        ensemble_metrics = dict(naive_metrics)

    # --------------------------------------------------------
    # Honest model selection: pick whichever of Naive / XGBoost
    # / SARIMAX / Ensemble actually has the lowest pooled OOS
    # MAE, instead of always presenting "Ensemble". Each
    # candidate has to earn its place on screen.
    # --------------------------------------------------------

    metrics = {
        "Naive": naive_metrics,
        "XGBoost": xgb_metrics,
        "SARIMAX": sar_metrics,
        "Ensemble": ensemble_metrics,
    }

    selected_model = min(
        metrics,
        key=lambda name: metrics[name]["MAE"],
    )

    residual_sources = {
        "Naive": (naive_actual_3, naive_pred_3),
        "XGBoost": (xgb_actual_3, xgb_pred_3),
        "SARIMAX": (sar_actual_3, sar_pred_3),
        "Ensemble": (ensemble_actual, ensemble_pred),
    }

    # --------------------------------------------------------
    # Direct XGBoost models for every future month, using the
    # freshest available feature row (today) rather than the
    # training frame's last row -- that row's target horizon is
    # necessarily unobserved and gets dropped by dropna, which
    # would otherwise silently forecast from stale, months-old
    # data. SARIMAX's 6-step path is computed once, not once
    # per horizon.
    # --------------------------------------------------------

    final_models = {}

    naive_path = []
    xgb_path = []
    sar_path = []
    ensemble_path = []

    current_date = (
        df.dropna(
            subset=[TARGET]
        ).index[-1]
    )

    current_pce = float(
        df[TARGET].dropna().iloc[-1]
    )

    future_dates = pd.date_range(
        current_date
        + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )

    sar_series = _sarimax_forecast_series(
        df,
        steps=6,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
    )

    X_today = (
        _feature_matrix(df, columns=selected_features)
        .dropna()
        .iloc[[-1]]
    )

    x3_train = None

    for horizon in HORIZONS:

        X_h, y_delta_h, base_level_h, future_level_h = _supervised(
            df,
            horizon=horizon,
            columns=selected_features,
        )

        final_xgb = _fit_xgb(
            X_h,
            y_delta_h,
        )

        delta_point = float(
            final_xgb.predict(X_today)[0]
        )

        xgb_point = current_pce + delta_point

        sar_point = float(
            sar_series.iloc[
                horizon - 1
            ]
        )

        ensemble_point = (
            w_naive * current_pce
            + w_xgb * xgb_point
            + w_sar * sar_point
        )

        naive_path.append(current_pce)
        xgb_path.append(xgb_point)
        sar_path.append(sar_point)
        ensemble_path.append(ensemble_point)

        final_models[horizon] = final_xgb

        if horizon == 3:
            x3_train = X_h

    paths = {
        "Naive": naive_path,
        "XGBoost": xgb_path,
        "SARIMAX": sar_path,
        "Ensemble": ensemble_path,
    }

    # --------------------------------------------------------
    # Smooth display path, using a light rolling blend between
    # adjacent direct forecasts without changing endpoints.
    # --------------------------------------------------------

    raw_path = np.array(
        paths[selected_model],
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
    # Use the selected model's own pooled residuals (from the
    # 36M evaluation), scaled gently by horizon. This is a
    # visualization range, not a model-based confidence
    # interval.
    # --------------------------------------------------------

    radius_actual, radius_pred = residual_sources[selected_model]

    radius_3m = _conformal_radius(
        radius_actual,
        radius_pred,
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

    selected_mae = metrics[selected_model]["MAE"]

    if selected_mae < 0.30:
        confidence = "High"
    elif selected_mae < 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    # --------------------------------------------------------
    # SHAP explainability from the 3M XGBoost model (illustrative
    # even when XGBoost isn't the selected headline model). SHAP
    # values are model contributions to the predicted CHANGE in
    # inflation, not causal effects.
    # --------------------------------------------------------

    model_3 = final_models[3]

    global_shap, local_shap = _shap_contributions(
        model_3,
        x3_train,
        X_today,
    )

    drivers = [
        (_friendly_name(name), value)
        for name, value in global_shap
    ]

    local_drivers = [
        (_friendly_name(name), value)
        for name, value in local_shap
    ]

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

    beat_naive = (
        "This beats the naive 'no change' benchmark on "
        f"out-of-sample testing ({metrics['Naive']['MAE']:.2f} MAE)."
        if selected_model != "Naive"
        else
        "No model tested currently beats the naive 'no change' "
        f"benchmark ({metrics['Naive']['MAE']:.2f} MAE) on "
        "out-of-sample testing, so that benchmark is shown instead."
    )

    brief = (
        f"The {selected_model} approach expects PCE inflation to be "
        f"{forecast_3m:.2f}% in three months and "
        f"{forecast_6m:.2f}% in six months, versus "
        f"the latest observed rate of "
        f"{current_pce:.2f}%. The near-term direction "
        f"is {direction}, leaving the 3-month estimate "
        f"{distance:+.2f} percentage points from the "
        f"Federal Reserve's 2% objective. The current "
        f"inflation environment is classified as "
        f"{regime.lower()}, with a pressure score of "
        f"{pressure}/100. {beat_naive} The 3M interval is "
        f"{forecast_3m - radius_3m:.2f}% to "
        f"{forecast_3m + radius_3m:.2f}%, from temporal "
        f"conformal calibration."
    )

    result = {
        "selected_model": selected_model,
        "ensemble_weights": {
            "Naive": w_naive,
            "XGBoost": w_xgb,
            "SARIMAX": w_sar,
        },
        "metrics": metrics,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_mae": ensemble_metrics["MAE"],
        "sarimax_order": {
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
        },
        "selected_features": selected_features,
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
        "xgb_local_drivers": local_drivers,
        "regime_history": regime_history,
        "calibration_radius_3m": radius_3m,
        "xgb_actual_3": xgb_actual_3,
        "xgb_pred_3": xgb_pred_3,
        "radius_actual_3": radius_actual,
        "radius_pred_3": radius_pred,
    }

    _CACHE[key] = result

    return result


def model_diagnostics(
    forecasts: dict,
) -> dict:

    radius_actual = forecasts[
        "radius_actual_3"
    ]

    radius_pred = forecasts[
        "radius_pred_3"
    ]

    residuals = (
        radius_actual
        - radius_pred
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

    order = ["Naive", "XGBoost", "SARIMAX", "Ensemble"]

    perf = pd.DataFrame(
        [
            {
                "Model": name,
                "Selected": (
                    "Yes"
                    if name == forecasts["selected_model"]
                    else ""
                ),
                "MAE": round(
                    forecasts["metrics"][name]["MAE"],
                    3,
                ),
                "RMSE": round(
                    forecasts["metrics"][name]["RMSE"],
                    3,
                ),
                "Bias": round(
                    forecasts["metrics"][name]["Bias"],
                    3,
                ),
            }
            for name in order
        ]
    ).sort_values("MAE")

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
    """Global SHAP importance: what the model leans on in general."""

    rows = []

    for name, value in forecasts[
        "xgb_drivers"
    ]:

        rows.append(
            {
                "Driver": name.title(),
                "Mean |SHAP|": round(
                    value,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


def local_driver_table(
    forecasts: dict,
) -> pd.DataFrame:
    """
    Local SHAP contributions for today's forecast specifically --
    signed, in percentage points of the predicted change. Positive
    pushes the forecast up, negative pushes it down. Model
    attribution, not a causal effect.
    """

    rows = []

    for name, value in forecasts[
        "xgb_local_drivers"
    ]:

        rows.append(
            {
                "Driver": name.title(),
                "Contribution (pp)": round(
                    value,
                    4,
                ),
                "Direction": (
                    "Pushes up"
                    if value > 0
                    else "Pushes down"
                ),
            }
        )

    return pd.DataFrame(rows)
