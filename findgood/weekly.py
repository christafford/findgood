"""Weekly buy-and-hold prediction.

Buy at Monday open, sell at Friday close. Train on historical weekly returns.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import date, timedelta

from findgood.db import get_connection

FEATURE_COLUMNS = [
    # Momentum
    "return_1w", "return_2w", "return_4w",
    "return_1d_friday",  # how the stock closed on Friday (last trading day)
    # Volatility
    "volatility_20d", "avg_range_20d",
    # Volume
    "volume_ratio_1w",  # last week's volume vs 4-week avg
    # Relative strength
    "rel_strength_spy_1w", "rel_strength_spy_4w",
    "rel_strength_sector_2w",
    # Position
    "dist_from_20d_high", "dist_from_20d_low",
    # Market context
    "spy_return_1w", "spy_volatility_20d",
    # Beta
    "beta_spy_20d",
    # News
    "news_positive_ratio_7d", "news_negative_ratio_7d", "news_count_7d",
    # Interactions
    "return_1w_x_spy_return", "beta_x_spy_return_1w",
    "return_1w_minus_expected",  # idiosyncratic weekly return
]


def _get_week_boundaries(conn) -> list[tuple[date, date]]:
    """Get (monday, friday) pairs for each trading week.

    Returns list of (first_trading_day, last_trading_day) per week.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MIN(trade_date) AS week_start,
                   MAX(trade_date) AS week_end,
                   COUNT(*) AS trading_days
            FROM day_aggs d
            JOIN tickers t ON d.ticker_id = t.id
            WHERE t.symbol = 'SPY'
            GROUP BY date_trunc('week', trade_date)
            HAVING COUNT(*) >= 4  -- need at least 4 trading days (skip holiday weeks)
            ORDER BY week_start
        """)
        return [(row[0], row[1]) for row in cur.fetchall()]


def _compute_weekly_features(as_of_friday: date, min_price: float,
                              min_avg_volume: float) -> list[dict]:
    """Compute features as of a Friday close to predict next week's return."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(WEEKLY_FEATURE_QUERY, {
                "as_of_date": as_of_friday,
                "min_price": min_price,
                "min_avg_volume": min_avg_volume,
            })
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def _get_weekly_returns(week_start: date, week_end: date, conn) -> dict[int, float]:
    """Get (friday_close - monday_open) / monday_open for all tickers in a week."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT mon.ticker_id,
                   CASE WHEN mon.open > 0
                        THEN (fri.close - mon.open) / mon.open
                        ELSE 0 END AS weekly_return
            FROM day_aggs mon
            JOIN day_aggs fri ON mon.ticker_id = fri.ticker_id
            WHERE mon.trade_date = %s
              AND fri.trade_date = %s
              AND mon.open > 0 AND fri.close > 0
        """, (week_start, week_end))
        return {row[0]: float(row[1]) for row in cur.fetchall()}


def backtest_weekly(min_price: float = 10.0, min_avg_volume: float = 500_000,
                    top_n: int = 5) -> dict:
    """Walk-forward weekly backtest: each week, train on prior weeks, pick top N."""
    conn = get_connection()
    weeks = _get_week_boundaries(conn)

    # Need enough history for features (~30 trading days) and training (~20 weeks)
    # Start testing from week 25 onwards
    min_warmup = 25
    if len(weeks) < min_warmup + 5:
        conn.close()
        return {"error": f"Need at least {min_warmup + 5} weeks of data"}

    # Get SPY ticker_id
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM tickers WHERE symbol = 'SPY'")
        spy_id = cur.fetchone()[0]

    test_weeks = weeks[min_warmup:]
    print(f"  Total weeks: {len(weeks)}, testing: {len(test_weeks)}", flush=True)
    print(f"  Period: {test_weeks[0][0]} to {test_weeks[-1][1]}", flush=True)

    results = []
    training_cache = []  # accumulate (features, weekly_returns) per week
    retrain_every = 4  # retrain monthly
    cached_model = None
    cached_scaler = None
    weeks_since_train = 0

    # Pre-build training data from warmup weeks
    print(f"  Building training data from first {min_warmup} weeks...", flush=True)
    for wi in range(min_warmup):
        prev_friday = weeks[wi][1]
        if wi + 1 < len(weeks):
            next_week_start, next_week_end = weeks[wi + 1]
            features = _compute_weekly_features(prev_friday, min_price, min_avg_volume)
            if features:
                weekly_returns = _get_weekly_returns(next_week_start, next_week_end, conn)
                training_cache.append((features, weekly_returns))

    print(f"  Training cache: {len(training_cache)} weeks", flush=True)

    for wi, (week_start, week_end) in enumerate(test_weeks):
        # Features are computed as of the PRIOR Friday
        prev_week_idx = min_warmup + wi - 1
        prev_friday = weeks[prev_week_idx][1]

        # Score stocks for this week
        features = _compute_weekly_features(prev_friday, min_price, min_avg_volume)
        if not features:
            continue

        # Retrain periodically
        if cached_model is None or weeks_since_train >= retrain_every:
            if len(training_cache) >= 10:
                cached_model, cached_scaler = _train_weekly_model(training_cache)
                weeks_since_train = 0
        weeks_since_train += 1

        if cached_model is None:
            # Add to training cache for next time
            weekly_returns = _get_weekly_returns(week_start, week_end, conn)
            training_cache.append((features, weekly_returns))
            continue

        # Predict
        X = _features_to_df(features)
        mask = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1)
        predictions = np.full(len(features), -999.0)
        if mask.any():
            predictions[mask.values] = cached_model.predict(
                cached_scaler.transform(X[mask]))

        for j, f in enumerate(features):
            f["score"] = float(predictions[j])

        scored = [f for f in features if f["score"] > -999.0]
        scored.sort(key=lambda f: f["score"], reverse=True)
        picks = scored[:top_n]

        # Get actual weekly returns
        weekly_returns = _get_weekly_returns(week_start, week_end, conn)

        pick_results = []
        for p in picks:
            actual = weekly_returns.get(p["ticker_id"])
            if actual is not None:
                pick_results.append((p["symbol"], actual))

        if not pick_results:
            training_cache.append((features, weekly_returns))
            continue

        avg_return = np.mean([r for _, r in pick_results])
        spy_return = weekly_returns.get(spy_id, 0)
        alpha = avg_return - spy_return

        if (wi + 1) % 20 == 0 or wi == 0 or wi == len(test_weeks) - 1:
            syms = ", ".join(s for s, _ in pick_results)
            print(f"  [{wi+1}/{len(test_weeks)}] {week_start} "
                  f"picks={avg_return:+.2%} SPY={spy_return:+.2%} "
                  f"alpha={alpha:+.2%}  [{syms}]", flush=True)

        results.append({
            "week_start": week_start,
            "week_end": week_end,
            "picks": pick_results,
            "avg_return": avg_return,
            "spy_return": spy_return,
            "alpha": alpha,
        })

        # Add this week to training cache
        training_cache.append((features, weekly_returns))

    conn.close()

    if not results:
        return {"error": "No testable weeks"}

    avg_pick = np.mean([r["avg_return"] for r in results])
    avg_spy = np.mean([r["spy_return"] for r in results])
    avg_alpha = np.mean([r["alpha"] for r in results])
    win_count = sum(1 for r in results if r["alpha"] > 0)

    # Cumulative returns
    cum_pick = np.prod([1 + r["avg_return"] for r in results]) - 1
    cum_spy = np.prod([1 + r["spy_return"] for r in results]) - 1

    # Feature importance from final model
    importance = None
    if cached_model is not None:
        importance = sorted(zip(FEATURE_COLUMNS, cached_model.coef_),
                            key=lambda x: abs(x[1]), reverse=True)

    return {
        "weeks_tested": len(results),
        "period": f"{results[0]['week_start']} to {results[-1]['week_end']}",
        "avg_weekly_return": avg_pick,
        "avg_spy_weekly_return": avg_spy,
        "avg_weekly_alpha": avg_alpha,
        "cumulative_return": cum_pick,
        "cumulative_spy_return": cum_spy,
        "cumulative_alpha": cum_pick - cum_spy,
        "beat_spy_count": win_count,
        "beat_spy_rate": win_count / len(results),
        "feature_importance": importance,
        "weekly_results": results,
    }


def _train_weekly_model(training_cache: list[tuple]):
    """Train Ridge on accumulated weekly training data."""
    all_X = []
    all_y = []

    for features, weekly_returns in training_cache:
        for f in features:
            actual = weekly_returns.get(f["ticker_id"])
            if actual is not None:
                all_X.append({col: float(f.get(col, 0)) for col in FEATURE_COLUMNS})
                all_y.append(actual)

    X = pd.DataFrame(all_X)
    y = np.array(all_y)

    mask = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    model = Ridge(alpha=5.0)
    model.fit(scaler.fit_transform(X), y)
    return model, scaler


def _features_to_df(features: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {col: float(f.get(col, 0)) for col in FEATURE_COLUMNS}
        for f in features
    ])


WEEKLY_FEATURE_QUERY = """
WITH target AS (
    SELECT %(as_of_date)s::date AS dt  -- the Friday we're scoring from
),

stock_today AS (
    SELECT d.ticker_id, t.symbol, d.close AS close_price, d.open, d.volume
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    JOIN target td ON d.trade_date = td.dt
    WHERE d.close > 0 AND d.open > 0
),

hist AS (
    SELECT d.ticker_id, d.trade_date, d.open, d.high, d.low, d.close, d.volume,
           CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS daily_return,
           ROW_NUMBER() OVER (PARTITION BY d.ticker_id ORDER BY d.trade_date DESC) AS days_ago
    FROM day_aggs d
    JOIN target td ON d.trade_date <= td.dt
      AND d.trade_date >= td.dt - 60
),

stats AS (
    SELECT
        h.ticker_id,

        -- 1-week return (~5 days)
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 5 THEN h.close END), 0) - 1) AS return_1w,
        -- 2-week return
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 10 THEN h.close END), 0) - 1) AS return_2w,
        -- 4-week return
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 20 THEN h.close END), 0) - 1) AS return_4w,

        -- Friday's intraday return
        MAX(CASE WHEN h.days_ago = 1 THEN h.daily_return END) AS return_1d_friday,

        -- Volatility (20-day)
        STDDEV(CASE WHEN h.days_ago <= 20 THEN h.daily_return END) AS volatility_20d,

        -- Average range
        AVG(CASE WHEN h.days_ago <= 20 AND h.close > 0
            THEN (h.high - h.low) / h.close END) AS avg_range_20d,

        -- Volume: last week vs 4-week avg
        AVG(CASE WHEN h.days_ago <= 5 THEN h.volume END) /
            NULLIF(AVG(CASE WHEN h.days_ago <= 20 THEN h.volume END), 0) AS volume_ratio_1w,

        -- 20-day high/low distance
        MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
            NULLIF(MAX(CASE WHEN h.days_ago <= 20 THEN h.high END), 0) - 1 AS dist_from_20d_high,
        MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
            NULLIF(MIN(CASE WHEN h.days_ago <= 20 THEN h.low END), 0) - 1 AS dist_from_20d_low,

        -- Avg volume for filtering
        AVG(CASE WHEN h.days_ago <= 20 THEN h.volume END) AS avg_volume_20d,

        COUNT(*) FILTER (WHERE h.days_ago <= 20) AS trading_days

    FROM hist h
    GROUP BY h.ticker_id
),

spy_hist AS (
    SELECT d.trade_date, d.close AS spy_close,
           CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS spy_return,
           ROW_NUMBER() OVER (ORDER BY d.trade_date DESC) AS days_ago
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    JOIN target td ON d.trade_date <= td.dt AND d.trade_date >= td.dt - 60
    WHERE t.symbol = 'SPY'
),

spy_stats AS (
    SELECT
        (MAX(CASE WHEN days_ago = 1 THEN spy_close END) /
         NULLIF(MAX(CASE WHEN days_ago = 5 THEN spy_close END), 0) - 1) AS spy_return_1w,
        (MAX(CASE WHEN days_ago = 1 THEN spy_close END) /
         NULLIF(MAX(CASE WHEN days_ago = 20 THEN spy_close END), 0) - 1) AS spy_return_4w,
        STDDEV(CASE WHEN days_ago <= 20 THEN spy_return END) AS spy_volatility_20d
    FROM spy_hist
),

stock_beta AS (
    SELECT h.ticker_id,
           REGR_SLOPE(h.daily_return, sp.spy_return) AS beta_spy_20d
    FROM hist h
    JOIN spy_hist sp ON h.trade_date = sp.trade_date
    WHERE h.days_ago <= 20
    GROUP BY h.ticker_id
    HAVING COUNT(*) >= 10
),

sector_rel AS (
    SELECT h.ticker_id,
        COALESCE(
            (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
             NULLIF(MAX(CASE WHEN h.days_ago = 10 THEN h.close END), 0) - 1)
            -
            (MAX(CASE WHEN ed.days_ago = 1 THEN ed.close END) /
             NULLIF(MAX(CASE WHEN ed.days_ago = 10 THEN ed.close END), 0) - 1)
        , 0) AS rel_strength_sector_2w
    FROM hist h
    JOIN ticker_details td ON h.ticker_id = td.ticker_id
    JOIN sector_etfs se ON td.sector = se.sector
    JOIN tickers et ON et.symbol = se.etf_symbol
    JOIN (
        SELECT d.ticker_id, d.trade_date, d.close,
               ROW_NUMBER() OVER (PARTITION BY d.ticker_id ORDER BY d.trade_date DESC) AS days_ago
        FROM day_aggs d
        JOIN target t ON d.trade_date <= t.dt AND d.trade_date >= t.dt - 60
    ) ed ON ed.ticker_id = et.id AND ed.trade_date = h.trade_date
    WHERE h.days_ago <= 10
    GROUP BY h.ticker_id
),

news AS (
    SELECT
        ns.ticker_id,
        COUNT(*) AS news_count_7d,
        COUNT(*) FILTER (WHERE s.label = 'positive')::float /
            GREATEST(COUNT(*), 1) AS news_positive_ratio_7d,
        COUNT(*) FILTER (WHERE s.label = 'negative')::float /
            GREATEST(COUNT(*), 1) AS news_negative_ratio_7d
    FROM news_sentiment ns
    JOIN sentiments s ON ns.sentiment_id = s.id
    JOIN target td ON true
    WHERE ns.published_utc >= (td.dt - 7)::timestamp
      AND ns.published_utc < (td.dt + 1)::timestamp
    GROUP BY ns.ticker_id
)

SELECT
    st.ticker_id,
    st.symbol,
    st.close_price,

    COALESCE(ss.return_1w, 0) AS return_1w,
    COALESCE(ss.return_2w, 0) AS return_2w,
    COALESCE(ss.return_4w, 0) AS return_4w,
    COALESCE(ss.return_1d_friday, 0) AS return_1d_friday,

    COALESCE(ss.volatility_20d, 0) AS volatility_20d,
    COALESCE(ss.avg_range_20d, 0) AS avg_range_20d,

    COALESCE(ss.volume_ratio_1w, 1) AS volume_ratio_1w,

    COALESCE(ss.return_1w, 0) - COALESCE(spy.spy_return_1w, 0) AS rel_strength_spy_1w,
    COALESCE(ss.return_4w, 0) - COALESCE(spy.spy_return_4w, 0) AS rel_strength_spy_4w,
    COALESCE(sr.rel_strength_sector_2w, 0) AS rel_strength_sector_2w,

    COALESCE(ss.dist_from_20d_high, 0) AS dist_from_20d_high,
    COALESCE(ss.dist_from_20d_low, 0) AS dist_from_20d_low,

    COALESCE(spy.spy_return_1w, 0) AS spy_return_1w,
    COALESCE(spy.spy_volatility_20d, 0) AS spy_volatility_20d,

    COALESCE(sb.beta_spy_20d, 1) AS beta_spy_20d,

    COALESCE(nw.news_positive_ratio_7d, 0) AS news_positive_ratio_7d,
    COALESCE(nw.news_negative_ratio_7d, 0) AS news_negative_ratio_7d,
    COALESCE(nw.news_count_7d, 0) AS news_count_7d,

    -- Interactions
    COALESCE(ss.return_1w, 0) * COALESCE(spy.spy_return_1w, 0) AS return_1w_x_spy_return,
    COALESCE(sb.beta_spy_20d, 1) * COALESCE(spy.spy_return_1w, 0) AS beta_x_spy_return_1w,
    COALESCE(ss.return_1w, 0) - COALESCE(sb.beta_spy_20d, 1) * COALESCE(spy.spy_return_1w, 0)
        AS return_1w_minus_expected

FROM stock_today st
JOIN stats ss ON st.ticker_id = ss.ticker_id
LEFT JOIN spy_stats spy ON true
LEFT JOIN stock_beta sb ON st.ticker_id = sb.ticker_id
LEFT JOIN sector_rel sr ON st.ticker_id = sr.ticker_id
LEFT JOIN news nw ON st.ticker_id = nw.ticker_id

WHERE st.close_price >= %(min_price)s
  AND COALESCE(ss.avg_volume_20d, 0) >= %(min_avg_volume)s
  AND ss.trading_days >= 15

ORDER BY st.ticker_id
"""
