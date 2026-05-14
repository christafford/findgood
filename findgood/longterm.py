"""Long-term buy-and-hold stock prediction.

Train on historical 1-year forward returns, predict the best stocks
to buy and hold for a year from the latest available date.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import date, timedelta

from findgood.db import get_connection

FEATURE_COLUMNS = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "volatility_30d", "volatility_90d",
    "rel_strength_spy_1m", "rel_strength_spy_3m", "rel_strength_spy_6m",
    "rel_strength_sector_3m",
    "dist_from_52w_high", "dist_from_52w_low",
    "volume_trend_3m",
    "beta_spy_90d",
    "news_positive_ratio_30d", "news_negative_ratio_30d", "news_count_30d",
]


def _compute_longterm_features(as_of_date: date, min_price: float,
                                min_avg_volume: float) -> list[dict]:
    """Compute long-term features for all eligible stocks as of a given date."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(LONGTERM_FEATURE_QUERY, {
                "as_of_date": as_of_date,
                "min_price": min_price,
                "min_avg_volume": min_avg_volume,
            })
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def _get_training_dates(conn) -> list[date]:
    """Get monthly sample dates for training (first trading day of each month).

    We sample monthly rather than daily to avoid massive training sets
    and reduce autocorrelation in the training data.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (date_trunc('month', trade_date))
                   trade_date
            FROM day_aggs
            WHERE trade_date <= (SELECT MAX(trade_date) FROM day_aggs) - 252
            ORDER BY date_trunc('month', trade_date), trade_date
        """)
        return [row[0] for row in cur.fetchall()]


def _get_forward_returns_bulk(as_of_date: date, conn) -> dict[int, float]:
    """Get 1-year forward close price for all tickers as of a date.

    Finds the trading day closest to 252 trading days later and returns
    a dict of ticker_id -> forward_close_price.
    """
    with conn.cursor() as cur:
        # Find the trading date ~252 days forward
        cur.execute("""
            SELECT trade_date FROM (
                SELECT DISTINCT trade_date FROM day_aggs
                WHERE trade_date > %s
                ORDER BY trade_date
                LIMIT 252
            ) sub
            ORDER BY trade_date DESC
            LIMIT 1
        """, (as_of_date,))
        row = cur.fetchone()
        if not row:
            return {}
        target_date = row[0]

        cur.execute("""
            SELECT ticker_id, close FROM day_aggs
            WHERE trade_date = %s AND close > 0
        """, (target_date,))
        return {r[0]: float(r[1]) for r in cur.fetchall()}


def train_and_predict(min_price: float = 10.0, min_avg_volume: float = 500_000,
                      top_n: int = 10) -> dict:
    """Train on historical 1-year returns, predict best buy-and-hold stocks."""
    conn = get_connection()

    # Get latest date
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(trade_date) FROM day_aggs")
        latest = cur.fetchone()[0]

    # Get monthly training dates (where we can see 1yr forward)
    train_dates = _get_training_dates(conn)
    print(f"  Training dates: {len(train_dates)} months "
          f"({train_dates[0]} to {train_dates[-1]})", flush=True)

    # Build training data
    all_X = []
    all_y = []
    skipped = 0

    for i, td in enumerate(train_dates):
        if (i + 1) % 12 == 0 or i == 0:
            print(f"    [{i+1}/{len(train_dates)}] {td}", flush=True)

        features = _compute_longterm_features(td, min_price, min_avg_volume)
        if not features:
            skipped += 1
            continue

        # Bulk fetch forward prices
        forward_prices = _get_forward_returns_bulk(td, conn)

        matched = 0
        for f in features:
            tid = f["ticker_id"]
            close_now = float(f.get("close_price", 0))
            future_close = forward_prices.get(tid)
            if future_close is not None and close_now > 0:
                fwd_return = (future_close - close_now) / close_now
                X_row = {col: float(f.get(col, 0)) for col in FEATURE_COLUMNS}
                all_X.append(X_row)
                all_y.append(fwd_return)
                matched += 1

        if i == 0:
            print(f"      {len(features)} stocks scored, {matched} with forward data, "
                  f"{len(forward_prices)} forward prices found", flush=True)

    conn.close()

    if not all_X:
        return {"error": "No training data"}

    X = pd.DataFrame(all_X)
    y = np.array(all_y)

    # Clean
    mask = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    print(f"  Training samples: {len(X):,} (skipped {skipped:,} without forward data)")
    print(f"  Avg 1yr return in training: {y.mean():+.1%}")
    print(f"  Median 1yr return: {np.median(y):+.1%}")

    # Train Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=10.0)
    model.fit(X_scaled, y)

    # Feature importance
    importance = sorted(zip(FEATURE_COLUMNS, model.coef_),
                        key=lambda x: abs(x[1]), reverse=True)

    # Predict from latest date
    print(f"\n  Scoring stocks as of {latest}...", flush=True)
    features = _compute_longterm_features(latest, min_price, min_avg_volume)

    if not features:
        return {"error": "No eligible stocks for prediction"}

    X_pred = pd.DataFrame([
        {col: float(f.get(col, 0)) for col in FEATURE_COLUMNS}
        for f in features
    ])
    pred_mask = X_pred.notna().all(axis=1) & np.isfinite(X_pred.values).all(axis=1)

    predictions = np.full(len(features), -999.0)
    if pred_mask.any():
        predictions[pred_mask.values] = model.predict(scaler.transform(X_pred[pred_mask]))

    for j, f in enumerate(features):
        f["predicted_1yr_return"] = float(predictions[j])

    # Filter out leveraged/inverse ETFs
    scored = [f for f in features if f["predicted_1yr_return"] > -999.0
              and not _is_leveraged(f["symbol"])]
    scored.sort(key=lambda f: f["predicted_1yr_return"], reverse=True)

    return {
        "as_of_date": latest,
        "training_samples": len(X),
        "eligible_stocks": len(scored),
        "avg_training_return": float(y.mean()),
        "feature_importance": importance,
        "picks": scored[:top_n],
    }


def _is_leveraged(symbol: str) -> bool:
    """Filter out leveraged/inverse ETF symbols."""
    leveraged_prefixes = ["TQQQ", "SQQQ", "UPRO", "SPXU", "UDOW", "SDOW",
                          "LABU", "LABD", "FNGU", "FNGD", "SOXL", "SOXS",
                          "TNA", "TZA", "NUGT", "DUST", "JNUG", "JDST",
                          "UVXY", "SVXY", "VIXY", "TVIX",
                          "YANG", "YINN", "FAS", "FAZ",
                          "ERX", "ERY", "GUSH", "DRIP",
                          "TECL", "TECS", "CURE", "GDXD", "GDXU",
                          "BNKU", "BNKD", "WEBL", "WEBS",
                          "NAIL", "DRV", "DPST", "RETL",
                          "MIDU", "MIDZ", "SMDD", "URTY",
                          "UCO", "SCO", "BOIL", "KOLD",
                          "AGQ", "ZSL", "UGL", "GLL",
                          "SPXS", "SPXL", "FNGO", "BULZ",
                          "WANT", "HIBL", "HIBS",
                          "TPOR", "DFEN", "PILL", "DUSL",
                          "CWEB", "INDL", "EURL"]
    return symbol in leveraged_prefixes


def backtest_vs_spy(min_price: float = 10.0, min_avg_volume: float = 500_000,
                    top_n: int = 10) -> dict:
    """Walk-forward backtest: quarterly, train on prior data, pick top N,
    measure actual 1yr return vs SPY's 1yr return from same date."""
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM day_aggs")
        earliest, latest = cur.fetchone()

        # Get quarterly test dates: first trading day of each quarter
        # Must have 252+ days of prior data for training AND 252 days forward for evaluation
        cur.execute("""
            SELECT DISTINCT ON (date_trunc('quarter', trade_date))
                   trade_date
            FROM day_aggs
            WHERE trade_date >= %s + 365
              AND trade_date <= %s - 252
            ORDER BY date_trunc('quarter', trade_date), trade_date
        """, (earliest, latest))
        test_dates = [row[0] for row in cur.fetchall()]

        # Get SPY ticker_id
        cur.execute("SELECT id FROM tickers WHERE symbol = 'SPY'")
        spy_id = cur.fetchone()[0]

    print(f"  Test dates: {len(test_dates)} quarters "
          f"({test_dates[0]} to {test_dates[-1]})", flush=True)

    results = []

    for qi, test_date in enumerate(test_dates):
        print(f"  [{qi+1}/{len(test_dates)}] {test_date} ... ", end="", flush=True)

        # Get training dates: monthly samples BEFORE test_date with 1yr forward visible
        conn2 = get_connection()
        with conn2.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (date_trunc('month', trade_date))
                       trade_date
                FROM day_aggs
                WHERE trade_date < %s
                  AND trade_date <= %s - 252
                ORDER BY date_trunc('month', trade_date), trade_date
            """, (test_date, test_date))
            train_dates = [row[0] for row in cur.fetchall()]
        conn2.close()

        if len(train_dates) < 6:
            print("not enough training data, skipping")
            continue

        # Build training data
        all_X = []
        all_y = []
        conn2 = get_connection()
        for td in train_dates:
            features = _compute_longterm_features(td, min_price, min_avg_volume)
            if not features:
                continue
            forward_prices = _get_forward_returns_bulk(td, conn2)
            for f in features:
                tid = f["ticker_id"]
                close_now = float(f.get("close_price", 0))
                future_close = forward_prices.get(tid)
                if future_close is not None and close_now > 0:
                    fwd_return = (future_close - close_now) / close_now
                    all_X.append({col: float(f.get(col, 0)) for col in FEATURE_COLUMNS})
                    all_y.append(fwd_return)
        conn2.close()

        if not all_X:
            print("no training data")
            continue

        X = pd.DataFrame(all_X)
        y = np.array(all_y)
        mask = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(X) < 100:
            print(f"only {len(X)} samples, skipping")
            continue

        # Train
        scaler = StandardScaler()
        model = Ridge(alpha=10.0)
        model.fit(scaler.fit_transform(X), y)

        # Score stocks on test_date
        features = _compute_longterm_features(test_date, min_price, min_avg_volume)
        if not features:
            print("no eligible stocks")
            continue

        X_test = pd.DataFrame([
            {col: float(f.get(col, 0)) for col in FEATURE_COLUMNS}
            for f in features
        ])
        test_mask = X_test.notna().all(axis=1) & np.isfinite(X_test.values).all(axis=1)

        predictions = np.full(len(features), -999.0)
        if test_mask.any():
            predictions[test_mask.values] = model.predict(scaler.transform(X_test[test_mask]))

        for j, f in enumerate(features):
            f["predicted"] = float(predictions[j])

        # Pick top N (exclude leveraged)
        candidates = [f for f in features if f["predicted"] > -999.0
                      and not _is_leveraged(f["symbol"])]
        candidates.sort(key=lambda f: f["predicted"], reverse=True)
        picks = candidates[:top_n]

        if not picks:
            print("no valid picks")
            continue

        # Get actual 1yr forward returns for picks and SPY
        conn2 = get_connection()
        forward_prices = _get_forward_returns_bulk(test_date, conn2)
        conn2.close()

        pick_returns = []
        for p in picks:
            close_now = float(p["close_price"])
            future_close = forward_prices.get(p["ticker_id"])
            if future_close and close_now > 0:
                pick_returns.append((p["symbol"], (future_close - close_now) / close_now))

        if not pick_returns:
            print("no forward data for picks")
            continue

        avg_pick_return = np.mean([r for _, r in pick_returns])

        # SPY return
        spy_now = None
        spy_future = None
        conn2 = get_connection()
        with conn2.cursor() as cur:
            cur.execute("SELECT close FROM day_aggs WHERE ticker_id = %s AND trade_date = %s",
                        (spy_id, test_date))
            row = cur.fetchone()
            if row:
                spy_now = float(row[0])
            spy_future_close = forward_prices.get(spy_id)
        conn2.close()

        spy_return = ((spy_future_close - spy_now) / spy_now) if (spy_now and spy_future_close) else 0

        alpha = avg_pick_return - spy_return

        symbols = ", ".join(s for s, _ in pick_returns[:5])
        if len(pick_returns) > 5:
            symbols += f" +{len(pick_returns)-5} more"
        print(f"picks={avg_pick_return:+.1%} SPY={spy_return:+.1%} alpha={alpha:+.1%}  [{symbols}]")

        results.append({
            "date": test_date,
            "picks": pick_returns,
            "avg_pick_return": avg_pick_return,
            "spy_return": spy_return,
            "alpha": alpha,
            "training_samples": len(X),
        })

    conn.close()

    if not results:
        return {"error": "No testable quarters"}

    avg_pick = np.mean([r["avg_pick_return"] for r in results])
    avg_spy = np.mean([r["spy_return"] for r in results])
    avg_alpha = np.mean([r["alpha"] for r in results])
    win_count = sum(1 for r in results if r["alpha"] > 0)

    return {
        "quarters_tested": len(results),
        "period": f"{results[0]['date']} to {results[-1]['date']}",
        "avg_pick_1yr_return": avg_pick,
        "avg_spy_1yr_return": avg_spy,
        "avg_alpha": avg_alpha,
        "beat_spy_count": win_count,
        "beat_spy_rate": win_count / len(results),
        "quarterly_results": results,
    }


LONGTERM_FEATURE_QUERY = """
WITH target_date AS (
    SELECT %(as_of_date)s::date AS dt
),

-- Stock's close on the target date
stock_today AS (
    SELECT d.ticker_id, t.symbol, d.close AS close_price, d.volume
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    JOIN target_date td ON d.trade_date = td.dt
    WHERE d.close > 0 AND d.open > 0
),

-- Historical prices for computing returns and stats
hist AS (
    SELECT d.ticker_id, d.trade_date, d.open, d.high, d.low, d.close, d.volume,
           CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS daily_return,
           ROW_NUMBER() OVER (PARTITION BY d.ticker_id ORDER BY d.trade_date DESC) AS days_ago
    FROM day_aggs d
    JOIN target_date td ON d.trade_date <= td.dt
      AND d.trade_date >= td.dt - 380  -- ~1.5 years back
),

stock_stats AS (
    SELECT
        h.ticker_id,

        -- Momentum: returns over various windows
        -- 1 month (~21 trading days)
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 21 THEN h.close END), 0) - 1) AS return_1m,
        -- 3 months (~63 trading days)
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 63 THEN h.close END), 0) - 1) AS return_3m,
        -- 6 months (~126 trading days)
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 126 THEN h.close END), 0) - 1) AS return_6m,
        -- 12 months (~252 trading days)
        (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
         NULLIF(MAX(CASE WHEN h.days_ago = 252 THEN h.close END), 0) - 1) AS return_12m,

        -- Volatility
        STDDEV(CASE WHEN h.days_ago <= 30 THEN h.daily_return END) AS volatility_30d,
        STDDEV(CASE WHEN h.days_ago <= 90 THEN h.daily_return END) AS volatility_90d,

        -- 52-week high/low distance
        MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
            NULLIF(MAX(CASE WHEN h.days_ago <= 252 THEN h.high END), 0) - 1 AS dist_from_52w_high,
        MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
            NULLIF(MIN(CASE WHEN h.days_ago <= 252 THEN h.low END), 0) - 1 AS dist_from_52w_low,

        -- Volume trend: avg volume last month / avg volume prior 3 months
        AVG(CASE WHEN h.days_ago <= 21 THEN h.volume END) /
            NULLIF(AVG(CASE WHEN h.days_ago > 21 AND h.days_ago <= 84 THEN h.volume END), 0)
            AS volume_trend_3m,

        -- Average volume for filtering
        AVG(CASE WHEN h.days_ago <= 30 THEN h.volume END) AS avg_volume_30d,

        -- Count of trading days available
        COUNT(*) FILTER (WHERE h.days_ago <= 252) AS trading_days_available

    FROM hist h
    GROUP BY h.ticker_id
),

-- SPY returns for same windows
spy_hist AS (
    SELECT d.trade_date,
           CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS spy_return,
           d.close AS spy_close,
           ROW_NUMBER() OVER (ORDER BY d.trade_date DESC) AS days_ago
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    JOIN target_date td ON d.trade_date <= td.dt AND d.trade_date >= td.dt - 380
    WHERE t.symbol = 'SPY'
),

spy_returns AS (
    SELECT
        (MAX(CASE WHEN days_ago = 1 THEN spy_close END) /
         NULLIF(MAX(CASE WHEN days_ago = 21 THEN spy_close END), 0) - 1) AS spy_return_1m,
        (MAX(CASE WHEN days_ago = 1 THEN spy_close END) /
         NULLIF(MAX(CASE WHEN days_ago = 63 THEN spy_close END), 0) - 1) AS spy_return_3m,
        (MAX(CASE WHEN days_ago = 1 THEN spy_close END) /
         NULLIF(MAX(CASE WHEN days_ago = 126 THEN spy_close END), 0) - 1) AS spy_return_6m
    FROM spy_hist
),

-- Beta vs SPY (90-day)
stock_beta AS (
    SELECT
        h.ticker_id,
        REGR_SLOPE(h.daily_return, sp.spy_return) AS beta_spy_90d
    FROM hist h
    JOIN spy_hist sp ON h.trade_date = sp.trade_date
    WHERE h.days_ago <= 90
    GROUP BY h.ticker_id
    HAVING COUNT(*) >= 30
),

-- Sector relative strength
sector_returns AS (
    SELECT se.sector,
           (MAX(CASE WHEN sp.days_ago = 1 THEN sp.spy_close END) /
            NULLIF(MAX(CASE WHEN sp.days_ago = 63 THEN sp.spy_close END), 0) - 1) AS placeholder
    FROM sector_etfs se
    CROSS JOIN spy_hist sp
    WHERE false  -- placeholder, actual sector calc below
    GROUP BY se.sector
),

stock_sector_strength AS (
    SELECT
        h.ticker_id,
        COALESCE(
            (MAX(CASE WHEN h.days_ago = 1 THEN h.close END) /
             NULLIF(MAX(CASE WHEN h.days_ago = 63 THEN h.close END), 0) - 1)
            -
            (MAX(CASE WHEN ed.days_ago = 1 THEN ed.close END) /
             NULLIF(MAX(CASE WHEN ed.days_ago = 63 THEN ed.close END), 0) - 1)
        , 0) AS rel_strength_sector_3m
    FROM hist h
    JOIN ticker_details td ON h.ticker_id = td.ticker_id
    JOIN sector_etfs se ON td.sector = se.sector
    JOIN tickers et ON et.symbol = se.etf_symbol
    JOIN (
        SELECT d.ticker_id, d.trade_date, d.close,
               ROW_NUMBER() OVER (PARTITION BY d.ticker_id ORDER BY d.trade_date DESC) AS days_ago
        FROM day_aggs d
        JOIN target_date td ON d.trade_date <= td.dt AND d.trade_date >= td.dt - 380
    ) ed ON ed.ticker_id = et.id AND ed.trade_date = h.trade_date
    WHERE h.days_ago <= 63
    GROUP BY h.ticker_id
),

-- News sentiment (30-day window)
news_stats AS (
    SELECT
        ns.ticker_id,
        COUNT(*) AS news_count_30d,
        COUNT(*) FILTER (WHERE s.label = 'positive')::float /
            GREATEST(COUNT(*), 1) AS news_positive_ratio_30d,
        COUNT(*) FILTER (WHERE s.label = 'negative')::float /
            GREATEST(COUNT(*), 1) AS news_negative_ratio_30d
    FROM news_sentiment ns
    JOIN sentiments s ON ns.sentiment_id = s.id
    JOIN target_date td ON true
    WHERE ns.published_utc >= (td.dt - 30)::timestamp
      AND ns.published_utc < (td.dt + 1)::timestamp
    GROUP BY ns.ticker_id
)

SELECT
    st.ticker_id,
    st.symbol,
    st.close_price,

    COALESCE(ss.return_1m, 0) AS return_1m,
    COALESCE(ss.return_3m, 0) AS return_3m,
    COALESCE(ss.return_6m, 0) AS return_6m,
    COALESCE(ss.return_12m, 0) AS return_12m,

    COALESCE(ss.volatility_30d, 0) AS volatility_30d,
    COALESCE(ss.volatility_90d, 0) AS volatility_90d,

    -- Relative strength vs SPY
    COALESCE(ss.return_1m, 0) - COALESCE(spr.spy_return_1m, 0) AS rel_strength_spy_1m,
    COALESCE(ss.return_3m, 0) - COALESCE(spr.spy_return_3m, 0) AS rel_strength_spy_3m,
    COALESCE(ss.return_6m, 0) - COALESCE(spr.spy_return_6m, 0) AS rel_strength_spy_6m,

    COALESCE(sss.rel_strength_sector_3m, 0) AS rel_strength_sector_3m,

    COALESCE(ss.dist_from_52w_high, 0) AS dist_from_52w_high,
    COALESCE(ss.dist_from_52w_low, 0) AS dist_from_52w_low,

    COALESCE(ss.volume_trend_3m, 1) AS volume_trend_3m,

    COALESCE(sb.beta_spy_90d, 1) AS beta_spy_90d,

    COALESCE(nw.news_positive_ratio_30d, 0) AS news_positive_ratio_30d,
    COALESCE(nw.news_negative_ratio_30d, 0) AS news_negative_ratio_30d,
    COALESCE(nw.news_count_30d, 0) AS news_count_30d

FROM stock_today st
JOIN stock_stats ss ON st.ticker_id = ss.ticker_id
LEFT JOIN spy_returns spr ON true
LEFT JOIN stock_beta sb ON st.ticker_id = sb.ticker_id
LEFT JOIN stock_sector_strength sss ON st.ticker_id = sss.ticker_id
LEFT JOIN news_stats nw ON st.ticker_id = nw.ticker_id

WHERE st.close_price >= %(min_price)s
  AND COALESCE(ss.avg_volume_30d, 0) >= %(min_avg_volume)s
  AND ss.trading_days_available >= 200
  -- Exclude ETFs and leveraged products by requiring ticker_details entry
  -- Include stocks with or without sector data (sector features will be 0 if missing)

ORDER BY st.ticker_id
"""
