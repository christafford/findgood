"""Feature engineering for intraday return prediction.

All features for day N use only data available before day N's market open.
The one exception is today's open price, which we observe at the moment of purchase.
"""

from datetime import date
from findgood.db import get_connection


def build_eod_cache():
    """Precompute end-of-day returns from minute_aggs into a summary table.

    This avoids scanning the massive minute_aggs table during each feature query.
    The eod_return is the return in the last 30 minutes of trading (3:30-4:00 PM ET).
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                print("  Building eod_cache from minute_aggs...", flush=True)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS eod_cache (
                        ticker_id   integer NOT NULL,
                        trade_date  date NOT NULL,
                        eod_return  numeric(12,8),
                        PRIMARY KEY (ticker_id, trade_date)
                    )
                """)

                # Find dates not yet cached
                cur.execute("""
                    SELECT DISTINCT trade_date FROM day_aggs
                    WHERE trade_date NOT IN (
                        SELECT DISTINCT trade_date FROM eod_cache
                    )
                    ORDER BY trade_date
                """)
                missing_dates = [row[0] for row in cur.fetchall()]

                if not missing_dates:
                    print("  eod_cache is up to date.")
                    return

                print(f"  Computing eod_return for {len(missing_dates)} dates...")

                for i, td in enumerate(missing_dates, 1):
                    if i % 25 == 0 or i == len(missing_dates):
                        print(f"    [{i}/{len(missing_dates)}] {td}", flush=True)

                    # For each date, get the last 30 min bars and compute return
                    # 3:30 PM ET = 19:30 UTC (EST) or 19:30 UTC (EDT depends on season)
                    # Simpler: just take the last 30 bars of the day per ticker
                    cur.execute("""
                        INSERT INTO eod_cache (ticker_id, trade_date, eod_return)
                        SELECT
                            sub.ticker_id,
                            sub.trade_date,
                            CASE WHEN MIN(sub.first_open) > 0
                                 THEN (MAX(sub.last_close) - MIN(sub.first_open)) / MIN(sub.first_open)
                                 ELSE 0 END
                        FROM (
                            SELECT
                                m.ticker_id,
                                m.trade_date,
                                CASE WHEN ROW_NUMBER() OVER w = 1 THEN m.close END AS last_close,
                                CASE WHEN ROW_NUMBER() OVER (PARTITION BY m.ticker_id ORDER BY m.window_start ASC) <=
                                          COUNT(*) OVER (PARTITION BY m.ticker_id) - 29
                                     THEN NULL ELSE m.open END AS first_open_candidate,
                                FIRST_VALUE(m.open) OVER (
                                    PARTITION BY m.ticker_id
                                    ORDER BY m.window_start ASC
                                    ROWS BETWEEN 0 PRECEDING AND 0 FOLLOWING
                                ) AS first_open
                            FROM minute_aggs m
                            WHERE m.trade_date = %s
                            WINDOW w AS (PARTITION BY m.ticker_id ORDER BY m.window_start DESC)
                        ) sub
                        CROSS JOIN LATERAL (
                            SELECT m2.open AS first_open
                            FROM minute_aggs m2
                            WHERE m2.ticker_id = sub.ticker_id
                              AND m2.trade_date = sub.trade_date
                            ORDER BY m2.window_start DESC
                            OFFSET 29 LIMIT 1
                        ) eod_start
                        WHERE sub.last_close IS NOT NULL
                        GROUP BY sub.ticker_id, sub.trade_date
                        ON CONFLICT (ticker_id, trade_date) DO NOTHING
                    """, (td,))
                    conn.commit()

                print("  eod_cache complete.")
    finally:
        conn.close()


def build_eod_cache_v2():
    """Simpler, faster approach: for each (ticker, date), get the open of the
    bar 30 minutes before close and the close of the last bar."""
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS eod_cache (
                        ticker_id   integer NOT NULL,
                        trade_date  date NOT NULL,
                        eod_return  numeric(12,8),
                        PRIMARY KEY (ticker_id, trade_date)
                    )
                """)

                cur.execute("""
                    SELECT DISTINCT d.trade_date FROM day_aggs d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM eod_cache e WHERE e.trade_date = d.trade_date
                    )
                    ORDER BY d.trade_date
                """)
                missing_dates = [row[0] for row in cur.fetchall()]

                if not missing_dates:
                    print("  eod_cache is up to date.")
                    return

                print(f"  Building eod_cache for {len(missing_dates)} dates...", flush=True)

                for i, td in enumerate(missing_dates, 1):
                    if i % 25 == 0 or i == 1 or i == len(missing_dates):
                        print(f"    [{i}/{len(missing_dates)}] {td}", flush=True)

                    cur.execute("""
                        WITH ranked AS (
                            SELECT ticker_id, open, close,
                                   ROW_NUMBER() OVER (PARTITION BY ticker_id ORDER BY window_start DESC) AS rn
                            FROM minute_aggs
                            WHERE trade_date = %s
                        )
                        INSERT INTO eod_cache (ticker_id, trade_date, eod_return)
                        SELECT
                            last_bar.ticker_id,
                            %s,
                            CASE WHEN eod_start.open > 0
                                 THEN (last_bar.close - eod_start.open) / eod_start.open
                                 ELSE 0 END
                        FROM (SELECT ticker_id, close FROM ranked WHERE rn = 1) last_bar
                        JOIN (SELECT ticker_id, open FROM ranked WHERE rn = 30) eod_start
                          ON last_bar.ticker_id = eod_start.ticker_id
                        ON CONFLICT (ticker_id, trade_date) DO NOTHING
                    """, (td, td))
                    conn.commit()

                print("  eod_cache complete.")
    finally:
        conn.close()


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


# Feature query uses precomputed eod_cache instead of scanning minute_aggs.
# Includes sector/index correlation features.
FEATURE_QUERY = """
WITH today AS (
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
    SELECT d.ticker_id, d.trade_date, d.open, d.high, d.low, d.close,
           d.volume, d.transactions,
           ROW_NUMBER() OVER (PARTITION BY d.ticker_id ORDER BY d.trade_date DESC) AS days_ago
    FROM day_aggs d
    WHERE d.trade_date < %(trade_date)s
      AND d.trade_date >= %(trade_date)s - 30
),

prior_stats AS (
    SELECT
        p.ticker_id,
        MAX(CASE WHEN p.days_ago = 1 THEN p.close END) AS prev_close,
        MAX(CASE WHEN p.days_ago = 1 THEN p.open END) AS prev_open,
        MAX(CASE WHEN p.days_ago = 1 THEN p.volume END) AS prev_volume,
        MAX(CASE WHEN p.days_ago = 1 THEN
            CASE WHEN p.open > 0 THEN (p.close - p.open) / p.open ELSE 0 END
        END) AS return_1d,
        MAX(CASE WHEN p.days_ago = 5 THEN p.close END) AS close_5d_ago,
        MAX(CASE WHEN p.days_ago = 10 THEN p.close END) AS close_10d_ago,
        STDDEV(CASE WHEN p.days_ago <= 10 AND p.open > 0
               THEN (p.close - p.open) / p.open END) AS volatility_10d,
        AVG(CASE WHEN p.days_ago <= 10 AND p.close > 0
            THEN (p.high - p.low) / p.close END) AS avg_range_10d,
        AVG(CASE WHEN p.days_ago <= 10 THEN p.volume END) AS avg_volume_10d,
        AVG(CASE WHEN p.days_ago <= 5 THEN p.volume END) AS avg_volume_5d,
        SUM(CASE WHEN p.days_ago <= 5 AND p.close > p.open THEN 1 ELSE 0 END) AS up_days_5d,
        COUNT(*) FILTER (WHERE p.days_ago <= 10) AS trading_days_10d,
        MAX(CASE WHEN p.days_ago = 1 THEN p.trade_date END) AS prev_trade_date
    FROM prior p
    GROUP BY p.ticker_id
),

-- SPY returns for the same prior window (market proxy)
spy_prior AS (
    SELECT d.trade_date,
           CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS spy_return
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    WHERE t.symbol = 'SPY'
      AND d.trade_date < %(trade_date)s
      AND d.trade_date >= %(trade_date)s - 30
),

-- Stock vs SPY correlation over prior 10 days
stock_spy_corr AS (
    SELECT
        p.ticker_id,
        CORR(
            CASE WHEN p.open > 0 THEN (p.close - p.open) / p.open ELSE 0 END,
            sp.spy_return
        ) AS spy_correlation_10d,
        -- Relative strength: stock 5d return minus SPY 5d return
        -- (computed via aggregation)
        REGR_SLOPE(
            CASE WHEN p.open > 0 THEN (p.close - p.open) / p.open ELSE 0 END,
            sp.spy_return
        ) AS beta_spy_10d
    FROM prior p
    JOIN spy_prior sp ON p.trade_date = sp.trade_date
    WHERE p.days_ago <= 10
    GROUP BY p.ticker_id
    HAVING COUNT(*) >= 5
),

-- Sector ETF returns for prior window
sector_etf_prior AS (
    SELECT se.sector, d.trade_date,
           CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS etf_return
    FROM sector_etfs se
    JOIN tickers t ON t.symbol = se.etf_symbol
    JOIN day_aggs d ON d.ticker_id = t.id
    WHERE d.trade_date < %(trade_date)s
      AND d.trade_date >= %(trade_date)s - 30
),

-- Stock vs its sector ETF correlation
stock_sector_corr AS (
    SELECT
        p.ticker_id,
        td.sector,
        CORR(
            CASE WHEN p.open > 0 THEN (p.close - p.open) / p.open ELSE 0 END,
            sep.etf_return
        ) AS sector_correlation_10d,
        -- Relative strength vs sector
        AVG(CASE WHEN p.days_ago <= 5 AND p.open > 0
            THEN (p.close - p.open) / p.open END)
        - AVG(CASE WHEN p.days_ago <= 5 THEN sep.etf_return END)
        AS relative_strength_vs_sector_5d
    FROM prior p
    JOIN ticker_details td ON p.ticker_id = td.ticker_id
    JOIN sector_etf_prior sep ON td.sector = sep.sector AND p.trade_date = sep.trade_date
    WHERE p.days_ago <= 10
    GROUP BY p.ticker_id, td.sector
    HAVING COUNT(*) >= 5
),

-- SPY's return yesterday (market direction)
spy_yesterday AS (
    SELECT
        CASE WHEN d.open > 0 THEN (d.close - d.open) / d.open ELSE 0 END AS spy_return_1d,
        CASE WHEN d.open > 0 THEN (d.high - d.low) / d.close ELSE 0 END AS spy_range_1d
    FROM day_aggs d
    JOIN tickers t ON d.ticker_id = t.id
    WHERE t.symbol = 'SPY'
      AND d.trade_date = (SELECT MAX(trade_date) FROM day_aggs WHERE trade_date < %(trade_date)s)
),

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

    CASE WHEN ps.prev_close > 0
         THEN (today.open - ps.prev_close) / ps.prev_close
         ELSE 0 END AS overnight_gap,

    COALESCE(ps.return_1d, 0) AS return_1d,
    CASE WHEN ps.close_5d_ago > 0 AND ps.prev_close IS NOT NULL
         THEN (ps.prev_close - ps.close_5d_ago) / ps.close_5d_ago
         ELSE 0 END AS return_5d,
    CASE WHEN ps.close_10d_ago > 0 AND ps.prev_close IS NOT NULL
         THEN (ps.prev_close - ps.close_10d_ago) / ps.close_10d_ago
         ELSE 0 END AS return_10d,

    COALESCE(ps.volatility_10d, 0) AS volatility_10d,
    COALESCE(ps.avg_range_10d, 0) AS avg_range_10d,

    CASE WHEN ps.avg_volume_10d > 0
         THEN ps.prev_volume / ps.avg_volume_10d
         ELSE 0 END AS volume_ratio_1d,
    COALESCE(ps.avg_volume_10d, 0) AS avg_volume_10d,

    COALESCE(ps.up_days_5d, 0) AS up_days_5d,

    COALESCE(eod.eod_return, 0) AS eod_return_prev,

    COALESCE(nw.news_count_3d, 0) AS news_count_3d,
    CASE WHEN COALESCE(nw.news_count_3d, 0) > 0
         THEN nw.news_positive_3d::float / nw.news_count_3d
         ELSE 0 END AS news_positive_ratio,
    CASE WHEN COALESCE(nw.news_count_3d, 0) > 0
         THEN nw.news_negative_3d::float / nw.news_count_3d
         ELSE 0 END AS news_negative_ratio,

    -- Sector/index correlation features
    COALESCE(ssc.spy_correlation_10d, 0) AS spy_correlation_10d,
    COALESCE(ssc.beta_spy_10d, 1) AS beta_spy_10d,
    COALESCE(scc.sector_correlation_10d, 0) AS sector_correlation_10d,
    COALESCE(scc.relative_strength_vs_sector_5d, 0) AS relative_strength_vs_sector_5d,
    COALESCE(sy.spy_return_1d, 0) AS spy_return_1d,
    COALESCE(sy.spy_range_1d, 0) AS spy_range_1d

FROM today
JOIN prior_stats ps ON today.ticker_id = ps.ticker_id
LEFT JOIN eod_cache eod ON eod.ticker_id = today.ticker_id
                       AND eod.trade_date = ps.prev_trade_date
LEFT JOIN news_stats nw ON today.ticker_id = nw.ticker_id
LEFT JOIN stock_spy_corr ssc ON today.ticker_id = ssc.ticker_id
LEFT JOIN stock_sector_corr scc ON today.ticker_id = scc.ticker_id
LEFT JOIN spy_yesterday sy ON true

WHERE today.open >= %(min_price)s
  AND COALESCE(ps.avg_volume_10d, 0) >= %(min_avg_volume)s
  AND ps.trading_days_10d >= 5

ORDER BY today.ticker_id
"""
