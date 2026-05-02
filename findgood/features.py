"""Feature engineering for intraday return prediction.

All features for day N use only data available before day N's market open.
The one exception is today's open price, which we observe at the moment of purchase.
"""

from datetime import date
from findgood.db import get_connection


def compute_features(trade_date: date, min_price: float = 1.0,
                     min_avg_volume: float = 100_000) -> list[dict]:
    """Compute all features for every eligible ticker on a given trade_date.

    Filters out penny stocks (< min_price) and illiquid stocks (< min_avg_volume)
    to focus on tradeable names.

    Returns list of dicts, one per ticker, with feature values and the actual
    intraday return (for backtesting).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(FEATURE_QUERY, {
                "trade_date": trade_date,
                "min_price": min_price,
                "min_avg_volume": min_avg_volume,
            })
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


# Single query that computes all features via CTEs.
# This runs entirely in PostgreSQL for performance.
FEATURE_QUERY = """
WITH today AS (
    -- Today's OHLCV (the day we're trading)
    SELECT d.ticker_id, t.symbol,
           d.open, d.high, d.low, d.close, d.volume, d.transactions,
           CASE WHEN d.open > 0
                THEN (d.close - d.open) / d.open
                ELSE 0 END AS intraday_return
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    WHERE d.trade_date = %(trade_date)s
      AND d.open > 0 AND d.close > 0
),

prior AS (
    -- Prior 20 trading days of data
    SELECT d.ticker_id, d.trade_date, d.open, d.high, d.low, d.close,
           d.volume, d.transactions,
           ROW_NUMBER() OVER (PARTITION BY d.ticker_id ORDER BY d.trade_date DESC) AS days_ago
    FROM day_aggs d
    WHERE d.trade_date < %(trade_date)s
      AND d.trade_date >= %(trade_date)s - 30  -- generous window to get 20 trading days
),

prior_stats AS (
    SELECT
        p.ticker_id,

        -- Yesterday's close (for gap calculation)
        MAX(CASE WHEN p.days_ago = 1 THEN p.close END) AS prev_close,
        MAX(CASE WHEN p.days_ago = 1 THEN p.open END) AS prev_open,
        MAX(CASE WHEN p.days_ago = 1 THEN p.volume END) AS prev_volume,

        -- 1-day return
        MAX(CASE WHEN p.days_ago = 1 THEN
            CASE WHEN p.open > 0 THEN (p.close - p.open) / p.open ELSE 0 END
        END) AS return_1d,

        -- 5-day return (close 5 days ago to yesterday's close)
        MAX(CASE WHEN p.days_ago = 5 THEN p.close END) AS close_5d_ago,

        -- 10-day return
        MAX(CASE WHEN p.days_ago = 10 THEN p.close END) AS close_10d_ago,

        -- Volatility: std dev of daily returns over last 10 days
        STDDEV(CASE WHEN p.days_ago <= 10 AND p.open > 0
               THEN (p.close - p.open) / p.open END) AS volatility_10d,

        -- Average daily range over 10 days
        AVG(CASE WHEN p.days_ago <= 10 AND p.close > 0
            THEN (p.high - p.low) / p.close END) AS avg_range_10d,

        -- Average volume over 10 days
        AVG(CASE WHEN p.days_ago <= 10 THEN p.volume END) AS avg_volume_10d,

        -- Average volume over 5 days
        AVG(CASE WHEN p.days_ago <= 5 THEN p.volume END) AS avg_volume_5d,

        -- Count of up days in last 5 days
        SUM(CASE WHEN p.days_ago <= 5 AND p.close > p.open THEN 1 ELSE 0 END) AS up_days_5d,

        -- Count of trading days we have
        COUNT(*) FILTER (WHERE p.days_ago <= 10) AS trading_days_10d

    FROM prior p
    GROUP BY p.ticker_id
),

-- End-of-day minute pattern: average return in last 30 minutes yesterday
eod_pattern AS (
    SELECT
        m.ticker_id,
        CASE WHEN MIN(m.open) > 0
             THEN (MAX(CASE WHEN rn = 1 THEN m.close END) - MIN(CASE WHEN rn = cnt THEN m.open END))
                  / MIN(CASE WHEN rn = cnt THEN m.open END)
             ELSE 0 END AS eod_return
    FROM (
        SELECT m.ticker_id, m.open, m.close, m.window_start,
               ROW_NUMBER() OVER (PARTITION BY m.ticker_id ORDER BY m.window_start DESC) AS rn,
               COUNT(*) OVER (PARTITION BY m.ticker_id) AS cnt
        FROM minute_aggs m
        WHERE m.trade_date = (
            SELECT MAX(d.trade_date) FROM day_aggs d
            WHERE d.trade_date < %(trade_date)s
        )
        AND m.window_start >= (
            SELECT MAX(d.trade_date) FROM day_aggs d
            WHERE d.trade_date < %(trade_date)s
        )::timestamp + interval '15 hours 30 minutes'  -- last 30 min of trading (3:30-4:00 PM ET)
    ) m
    WHERE m.rn <= 30
    GROUP BY m.ticker_id
),

-- News sentiment in the 3 days before trade_date
news_stats AS (
    SELECT
        ns.ticker_id,
        COUNT(*) AS news_count_3d,
        COUNT(*) FILTER (WHERE s.label = 'positive') AS news_positive_3d,
        COUNT(*) FILTER (WHERE s.label = 'negative') AS news_negative_3d,
        COUNT(*) FILTER (WHERE s.label = 'neutral') AS news_neutral_3d
    FROM news_sentiment ns
    JOIN sentiments s ON ns.sentiment_id = s.id
    WHERE ns.published_utc >= (%(trade_date)s - 3)::timestamp
      AND ns.published_utc < %(trade_date)s::timestamp
    GROUP BY ns.ticker_id
)

SELECT
    today.ticker_id,
    today.symbol,
    today.open AS today_open,
    today.intraday_return,

    -- Gap: today's open vs yesterday's close
    CASE WHEN ps.prev_close > 0
         THEN (today.open - ps.prev_close) / ps.prev_close
         ELSE 0 END AS overnight_gap,

    -- Momentum features
    COALESCE(ps.return_1d, 0) AS return_1d,
    CASE WHEN ps.close_5d_ago > 0 AND ps.prev_close IS NOT NULL
         THEN (ps.prev_close - ps.close_5d_ago) / ps.close_5d_ago
         ELSE 0 END AS return_5d,
    CASE WHEN ps.close_10d_ago > 0 AND ps.prev_close IS NOT NULL
         THEN (ps.prev_close - ps.close_10d_ago) / ps.close_10d_ago
         ELSE 0 END AS return_10d,

    -- Volatility & range
    COALESCE(ps.volatility_10d, 0) AS volatility_10d,
    COALESCE(ps.avg_range_10d, 0) AS avg_range_10d,

    -- Volume features
    CASE WHEN ps.avg_volume_10d > 0
         THEN ps.prev_volume / ps.avg_volume_10d
         ELSE 0 END AS volume_ratio_1d,
    COALESCE(ps.avg_volume_10d, 0) AS avg_volume_10d,

    -- Streak
    COALESCE(ps.up_days_5d, 0) AS up_days_5d,

    -- End-of-day pattern
    COALESCE(eod.eod_return, 0) AS eod_return_prev,

    -- News sentiment
    COALESCE(nw.news_count_3d, 0) AS news_count_3d,
    CASE WHEN COALESCE(nw.news_count_3d, 0) > 0
         THEN nw.news_positive_3d::float / nw.news_count_3d
         ELSE 0 END AS news_positive_ratio,
    CASE WHEN COALESCE(nw.news_count_3d, 0) > 0
         THEN nw.news_negative_3d::float / nw.news_count_3d
         ELSE 0 END AS news_negative_ratio

FROM today
JOIN prior_stats ps ON today.ticker_id = ps.ticker_id
LEFT JOIN eod_pattern eod ON today.ticker_id = eod.ticker_id
LEFT JOIN news_stats nw ON today.ticker_id = nw.ticker_id

-- Filter: tradeable stocks only
WHERE today.open >= %(min_price)s
  AND COALESCE(ps.avg_volume_10d, 0) >= %(min_avg_volume)s
  AND ps.trading_days_10d >= 5  -- need enough history

ORDER BY today.ticker_id
"""
