"""Walk-forward backtesting framework.

For each trading day in the test window, score all eligible stocks using
only prior data, pick the top 3, and measure actual intraday returns.
"""

from datetime import date
from decimal import Decimal
from findgood.db import get_connection
from findgood.features import compute_features


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


def score_stocks(features: list[dict], strategy: str = "momentum_sentiment") -> list[dict]:
    """Score each stock using the named strategy. Returns features sorted by score desc."""
    score_fn = STRATEGIES.get(strategy)
    if not score_fn:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())}")

    for row in features:
        row["score"] = score_fn(row)

    return sorted(features, key=lambda r: r["score"], reverse=True)


# --- Scoring strategies ---
# Each takes a feature dict and returns a float score. Higher = more likely to buy.

def _strategy_momentum_sentiment(f: dict) -> float:
    """Combine short-term momentum with news sentiment and volatility."""
    score = 0.0

    # Short-term momentum: prefer stocks with recent positive returns
    score += float(f.get("return_1d", 0)) * 2.0
    score += float(f.get("return_5d", 0)) * 1.0

    # Overnight gap: small negative gaps can mean mean-reversion opportunity
    gap = float(f.get("overnight_gap", 0))
    if -0.03 < gap < 0.0:
        score += 0.5  # small gap down = potential bounce
    elif gap > 0.05:
        score -= 0.5  # large gap up = may fade

    # News sentiment: positive news = catalyst
    pos_ratio = float(f.get("news_positive_ratio", 0))
    neg_ratio = float(f.get("news_negative_ratio", 0))
    news_count = int(f.get("news_count_3d", 0))
    if news_count > 0:
        score += (pos_ratio - neg_ratio) * 1.5

    # Volatility: prefer moderate volatility (opportunity without chaos)
    vol = float(f.get("volatility_10d", 0))
    if 0.01 < vol < 0.05:
        score += 0.3
    elif vol > 0.08:
        score -= 0.3

    # Volume surge: above-average volume yesterday = attention
    vol_ratio = float(f.get("volume_ratio_1d", 0))
    if vol_ratio > 1.5:
        score += 0.4

    # EOD pattern: strong close yesterday = continuation signal
    score += float(f.get("eod_return_prev", 0)) * 3.0

    return score


def _strategy_mean_reversion(f: dict) -> float:
    """Bet on stocks that dropped recently and may bounce."""
    score = 0.0

    # Prefer recent losers
    score -= float(f.get("return_1d", 0)) * 3.0
    score -= float(f.get("return_5d", 0)) * 1.5

    # But only if they have positive news (catalyst for recovery)
    pos_ratio = float(f.get("news_positive_ratio", 0))
    news_count = int(f.get("news_count_3d", 0))
    if news_count > 0:
        score += pos_ratio * 2.0

    # Negative gap = bounce potential
    gap = float(f.get("overnight_gap", 0))
    if gap < 0:
        score += abs(gap) * 5.0

    # Need volatility for a bounce
    vol = float(f.get("volatility_10d", 0))
    score += vol * 3.0

    # High volume = attention needed for bounce
    vol_ratio = float(f.get("volume_ratio_1d", 0))
    if vol_ratio > 1.5:
        score += 0.5

    return score


def _strategy_breakout(f: dict) -> float:
    """Look for stocks about to break out: volume surge + momentum + news."""
    score = 0.0

    # Strong recent momentum
    r1 = float(f.get("return_1d", 0))
    r5 = float(f.get("return_5d", 0))
    if r1 > 0 and r5 > 0:
        score += r1 * 3.0 + r5 * 1.0

    # Volume surge is key signal
    vol_ratio = float(f.get("volume_ratio_1d", 0))
    if vol_ratio > 2.0:
        score += 1.0
    elif vol_ratio > 1.5:
        score += 0.5

    # Positive gap up = continuation
    gap = float(f.get("overnight_gap", 0))
    if 0 < gap < 0.03:
        score += 0.5

    # News catalyst
    news_count = int(f.get("news_count_3d", 0))
    pos_ratio = float(f.get("news_positive_ratio", 0))
    if news_count > 0 and pos_ratio > 0.6:
        score += 1.0

    # Strong EOD yesterday
    eod = float(f.get("eod_return_prev", 0))
    if eod > 0:
        score += eod * 5.0

    # Higher volatility = bigger breakout potential
    vol = float(f.get("volatility_10d", 0))
    score += vol * 2.0

    return score


STRATEGIES = {
    "momentum_sentiment": _strategy_momentum_sentiment,
    "mean_reversion": _strategy_mean_reversion,
    "breakout": _strategy_breakout,
}


def run_backtest(start: date, end: date, strategy: str = "momentum_sentiment",
                 top_n: int = 3, min_price: float = 1.0,
                 min_avg_volume: float = 100_000) -> dict:
    """Run walk-forward backtest over the date range.

    Returns summary dict with daily picks, returns, and aggregate stats.
    """
    trading_days = get_trading_days(start, end)

    # Need at least the second day to have prior data
    if len(trading_days) < 2:
        return {"error": "Need at least 2 trading days"}

    # Skip the first few days so features have enough history
    test_days = trading_days[5:]  # start after 5 days of warmup

    daily_results = []
    total_return = 0.0
    winning_days = 0

    for trade_date in test_days:
        features = compute_features(trade_date, min_price, min_avg_volume)
        if not features:
            continue

        scored = score_stocks(features, strategy)
        picks = scored[:top_n]

        # Average return of our picks
        day_return = sum(float(p["intraday_return"]) for p in picks) / len(picks)
        total_return += day_return
        if day_return > 0:
            winning_days += 1

        # Also track what the best possible picks would have been
        best = sorted(features, key=lambda r: float(r["intraday_return"]), reverse=True)[:top_n]
        best_return = sum(float(b["intraday_return"]) for b in best) / len(best)

        # And the market average
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

    return {
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
