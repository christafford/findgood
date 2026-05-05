"""Walk-forward backtesting framework.

For each trading day in the test window, score all eligible stocks using
only prior data, pick the top 3, and measure actual intraday returns.
"""

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from findgood.db import get_connection
from findgood.features import compute_features

# Base features from the SQL query
BASE_FEATURES = [
    "overnight_gap", "return_1d", "return_5d", "return_10d",
    "volatility_10d", "avg_range_10d", "volume_ratio_1d",
    "up_days_5d", "eod_return_prev",
    "news_count_3d", "news_positive_ratio", "news_negative_ratio",
    "spy_correlation_10d", "beta_spy_10d",
    "sector_correlation_10d", "relative_strength_vs_sector_5d",
    "spy_return_1d", "spy_range_1d",
]

# Interaction features computed from base features
INTERACTION_FEATURES = [
    "return_1d_x_spy_return",        # stock momentum relative to market
    "overnight_gap_x_beta",          # gap adjusted for beta
    "volatility_x_spy_range",        # stock vol in context of market vol
    "rel_strength_x_spy_return",     # sector outperformer when market moves
    "return_1d_x_sector_corr",       # momentum weighted by sector coupling
    "overnight_gap_x_volume_ratio",  # gap with volume confirmation
    "beta_x_spy_return",             # expected move based on beta * market
    "return_1d_minus_beta_x_spy",    # idiosyncratic return (alpha component)
]

FEATURE_COLUMNS = BASE_FEATURES + INTERACTION_FEATURES


def get_trading_days(start: date, end: date) -> list[date]:
    """Return trading days in range (days where we have day_aggs data)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT trade_date FROM day_aggs
                WHERE trade_date >= %s AND trade_date <= %s
                ORDER BY trade_date
            """, (start, end))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features to a DataFrame of base features."""
    df["return_1d_x_spy_return"] = df["return_1d"] * df["spy_return_1d"]
    df["overnight_gap_x_beta"] = df["overnight_gap"] * df["beta_spy_10d"]
    df["volatility_x_spy_range"] = df["volatility_10d"] * df["spy_range_1d"]
    df["rel_strength_x_spy_return"] = df["relative_strength_vs_sector_5d"] * df["spy_return_1d"]
    df["return_1d_x_sector_corr"] = df["return_1d"] * df["sector_correlation_10d"]
    df["overnight_gap_x_volume_ratio"] = df["overnight_gap"] * df["volume_ratio_1d"]
    df["beta_x_spy_return"] = df["beta_spy_10d"] * df["spy_return_1d"]
    df["return_1d_minus_beta_x_spy"] = df["return_1d"] - df["beta_spy_10d"] * df["spy_return_1d"]
    return df


def _features_to_df(features: list[dict]) -> pd.DataFrame:
    """Convert feature dicts to a DataFrame with base + interaction features."""
    df = pd.DataFrame([
        {col: float(f.get(col, 0)) for col in BASE_FEATURES}
        for f in features
    ])
    df = _add_interactions(df)
    return df


def _prepare_training_data(training_data: list[tuple], target_key: str = "intraday_return"):
    """Concatenate all training days into X, y arrays."""
    all_X = []
    all_y = []
    for features, _ in training_data:
        X = _features_to_df(features)
        y = np.array([float(f[target_key]) for f in features])
        all_X.append(X)
        all_y.append(y)

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)

    mask = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def _train_ridge(training_data: list[tuple], target_key: str = "intraday_return"):
    """Train a Ridge regression. Returns (model, scaler)."""
    X, y = _prepare_training_data(training_data, target_key)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    return model, scaler


def _train_lgbm(training_data: list[tuple], target_key: str = "intraday_return"):
    """Train a LightGBM regressor. Returns (model, None)."""
    X, y = _prepare_training_data(training_data, target_key)
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        num_leaves=8,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=200,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbose=-1,
    )
    model.fit(X, y)
    return model, None


# Strategy registry — only ML strategies now
STRATEGIES = {
    "ridge": "ridge",
    "lgbm": "lgbm",
    "ridge_alpha": "ridge",    # predict alpha (return - beta*spy) instead of raw return
    "lgbm_alpha": "lgbm",
    "lgbm_sector_rotation": "lgbm",  # sector rotation: pick stocks outperforming sector while sector lags
}


def run_backtest(start: date, end: date, strategy: str = "lgbm",
                 top_n: int = 3, min_price: float = 1.0,
                 min_avg_volume: float = 100_000) -> dict:
    """Run walk-forward backtest over the date range.

    Strategies:
      ridge / lgbm: predict raw intraday return
      ridge_alpha / lgbm_alpha: predict market-adjusted return
      lgbm_sector_rotation: predict return, but pre-filter to stocks
        outperforming their sector when sector lags the market

    Returns summary dict with daily picks, returns, and aggregate stats.
    """
    trading_days = get_trading_days(start, end)

    if len(trading_days) < 2:
        return {"error": "Need at least 2 trading days"}

    test_days = trading_days[5:]

    # Determine model type and target
    model_type = STRATEGIES.get(strategy)
    if not model_type:
        return {"error": f"Unknown strategy: {strategy}"}

    is_alpha = strategy.endswith("_alpha")
    is_sector_rotation = strategy == "lgbm_sector_rotation"
    train_fn = _train_lgbm if model_type == "lgbm" else _train_ridge
    target_key = "alpha_return" if is_alpha else "intraday_return"

    training_data = []
    min_train_days = 5
    retrain_every = 5
    cached_model = None
    cached_scaler = None
    days_since_train = 0

    daily_results = []
    total_return = 0.0
    winning_days = 0

    for day_idx, trade_date in enumerate(test_days):
        features = compute_features(trade_date, min_price, min_avg_volume)
        if not features:
            continue

        # Compute alpha return for each stock: return - beta * spy_return
        # This is the "today" alpha — only known after the fact (for training)
        spy_ret_today = float(features[0].get("spy_return_1d", 0))  # yesterday's spy
        for f in features:
            beta = float(f.get("beta_spy_10d", 1))
            f["alpha_return"] = float(f["intraday_return"]) - beta * float(f.get("spy_return_1d", 0))

        training_data.append((features, None))

        if day_idx < min_train_days:
            continue

        # Retrain periodically
        if cached_model is None or days_since_train >= retrain_every:
            cached_model, cached_scaler = train_fn(training_data[:-1], target_key)
            days_since_train = 0
        days_since_train += 1
        model, scaler = cached_model, cached_scaler

        # Score all stocks
        X_today = _features_to_df(features)
        mask = X_today.notna().all(axis=1) & np.isfinite(X_today.values).all(axis=1)
        predictions = np.full(len(features), -999.0)
        if mask.any():
            X_pred = scaler.transform(X_today[mask]) if scaler else X_today[mask]
            predictions[mask.values] = model.predict(X_pred)

        for j, f in enumerate(features):
            f["score"] = float(predictions[j])

        # Sector rotation pre-filter: only consider stocks that are
        # outperforming their sector (positive relative strength)
        # when their sector is lagging the market (sector ETF < SPY)
        if is_sector_rotation:
            candidates = [
                f for f in features
                if float(f.get("relative_strength_vs_sector_5d", 0)) > 0
                and f["score"] > -999.0
            ]
            if len(candidates) < top_n:
                candidates = [f for f in features if f["score"] > -999.0]
            scored = sorted(candidates, key=lambda r: r["score"], reverse=True)
        else:
            scored = sorted(features, key=lambda r: r["score"], reverse=True)

        picks = scored[:top_n]

        day_return = sum(float(p["intraday_return"]) for p in picks) / len(picks)
        total_return += day_return
        if day_return > 0:
            winning_days += 1

        best = sorted(features, key=lambda r: float(r["intraday_return"]), reverse=True)[:top_n]
        best_return = sum(float(b["intraday_return"]) for b in best) / len(best)

        market_return = sum(float(f["intraday_return"]) for f in features) / len(features)

        daily_results.append({
            "date": trade_date,
            "picks": [(p["symbol"], float(p["score"]), float(p["intraday_return"])) for p in picks],
            "day_return": day_return,
            "best_possible_return": best_return,
            "market_avg_return": market_return,
            "eligible_stocks": len(features),
        })

    n_days = len(daily_results)
    if n_days == 0:
        return {"error": "No testable days"}

    result = {
        "strategy": strategy,
        "period": f"{test_days[0]} to {test_days[-1]}",
        "trading_days": n_days,
        "total_return": total_return,
        "avg_daily_return": total_return / n_days,
        "winning_days": winning_days,
        "losing_days": n_days - winning_days,
        "win_rate": winning_days / n_days,
        "avg_market_return": sum(d["market_avg_return"] for d in daily_results) / n_days,
        "daily_results": daily_results,
    }

    # Feature importances from final model
    if training_data:
        model, scaler = train_fn(training_data, target_key)
        if model_type == "lgbm":
            importances = model.feature_importances_
            importance = sorted(zip(FEATURE_COLUMNS, importances),
                                key=lambda x: x[1], reverse=True)
        else:
            coefs = model.coef_
            importance = sorted(zip(FEATURE_COLUMNS, coefs),
                                key=lambda x: abs(x[1]), reverse=True)
        result["feature_importance"] = importance

    return result
