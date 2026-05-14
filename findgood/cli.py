from datetime import date, timedelta

import click
from findgood import db
from findgood.ingest import ingest_data_type
from findgood.news import fetch_news
from findgood.backtest import run_backtest, STRATEGIES
from findgood.features import build_eod_cache_v2
from findgood.sectors import fetch_ticker_details, init_sector_etfs


@click.group()
def cli():
    """FindGood — stock data downloader and analyzer."""
    pass


@cli.command("init-db")
def init_db():
    """Initialize the database schema (tables, partitions, indexes)."""
    db.init_schema()


@cli.command("download")
@click.option(
    "--data-type", "-t",
    type=click.Choice(["all", "day", "minute", "news"], case_sensitive=False),
    default="all",
    help="Which data to download: day/minute aggregates, news sentiment, or all.",
)
def download(data_type):
    """Download stock data from Massive.com."""
    types_to_fetch = []
    if data_type in ("all", "day"):
        types_to_fetch.append(("day_aggs", "day_aggs_v1"))
    if data_type in ("all", "minute"):
        types_to_fetch.append(("minute_aggs", "minute_aggs_v1"))

    for label, s3_type in types_to_fetch:
        print(f"\n--- {label} ---")
        ingest_data_type(label, s3_type)

    if data_type in ("all", "news"):
        print(f"\n--- news sentiment ---")
        fetch_news()

    print("\nDone.")


@cli.command("status")
def status():
    """Show download progress."""
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT data_type,
                       count(*) as files,
                       sum(rows_inserted) as total_rows,
                       min(trade_date) as earliest,
                       max(trade_date) as latest
                FROM download_log
                GROUP BY data_type
                ORDER BY data_type
            """)
            rows = cur.fetchall()

            if not rows:
                print("No data downloaded yet.")
                return

            print(f"{'Data Type':<15} {'Files':>8} {'Total Rows':>15} {'Earliest':>12} {'Latest':>12}")
            print("-" * 65)
            for data_type, files, total_rows, earliest, latest in rows:
                print(f"{data_type:<15} {files:>8,} {total_rows:>15,} {str(earliest):>12} {str(latest):>12}")

            # News sentiment stats
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'news_sentiment'
                )
            """)
            if cur.fetchone()[0]:
                cur.execute("""
                    SELECT s.label, count(*)
                    FROM news_sentiment ns
                    JOIN sentiments s ON ns.sentiment_id = s.id
                    GROUP BY s.label ORDER BY count(*) DESC
                """)
                sentiments = cur.fetchall()
                if sentiments:
                    cur.execute("SELECT count(*) FROM news_fetch_log WHERE completed = true")
                    news_days = cur.fetchone()[0]
                    cur.execute("SELECT count(*) FROM news_sentiment")
                    news_total = cur.fetchone()[0]
                    print(f"\nNews sentiment: {news_total:,} records across {news_days} days")
                    for label, cnt in sentiments:
                        print(f"  {label}: {cnt:,}")

            # Ticker count
            cur.execute("SELECT count(*) FROM tickers")
            ticker_count = cur.fetchone()[0]
            print(f"\nUnique tickers: {ticker_count:,}")

            # DB size info
            cur.execute("""
                SELECT relname, pg_size_pretty(pg_total_relation_size(oid))
                FROM pg_class
                WHERE relname IN ('day_aggs', 'minute_aggs', 'tickers')
                  AND relkind = 'r'
                ORDER BY relname
            """)
            sizes = cur.fetchall()
            if sizes:
                print(f"\nTable sizes:")
                for name, size in sizes:
                    print(f"  {name}: {size}")

            # For partitioned table, get total
            cur.execute("""
                SELECT pg_size_pretty(sum(pg_total_relation_size(inhrelid)))
                FROM pg_inherits
                WHERE inhparent = 'minute_aggs'::regclass
            """)
            part_size = cur.fetchone()
            if part_size and part_size[0]:
                print(f"  minute_aggs (all partitions): {part_size[0]}")

    finally:
        conn.close()


@cli.command("build-cache")
def build_cache():
    """Precompute eod_cache from minute_aggs (required before backtest)."""
    build_eod_cache_v2()
    print("Done.")


@cli.command("fetch-sectors")
def fetch_sectors():
    """Fetch sector/industry classification for all tickers."""
    init_sector_etfs()
    fetch_ticker_details()


@cli.command("backtest")
@click.option("--strategy", "-s",
              type=click.Choice(list(STRATEGIES.keys()), case_sensitive=False),
              default="lgbm",
              help="Scoring strategy to test.")
@click.option("--top-n", "-n", default=3, help="Number of stocks to pick per day.")
@click.option("--min-price", default=1.0, help="Minimum stock price filter.")
@click.option("--min-volume", default=100_000.0, help="Minimum 10-day avg volume filter.")
@click.option("--days", "-d", default=None, type=int,
              help="Number of days to backtest (default: all available).")
def backtest(strategy, top_n, min_price, min_volume, days):
    """Run walk-forward backtest on historical data."""
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM day_aggs")
            earliest, latest = cur.fetchone()
    finally:
        conn.close()

    if not earliest:
        print("No data available. Run 'findgood download' first.")
        return

    start = earliest
    if days:
        start = max(earliest, latest - timedelta(days=days))

    print(f"Running backtest: {strategy}")
    print(f"  Period: {start} to {latest}")
    print(f"  Top {top_n} picks, min price ${min_price}, min avg volume {min_volume:,.0f}")
    print()

    result = run_backtest(start, latest, strategy, top_n, min_price, min_volume)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Summary
    print(f"{'='*70}")
    print(f"Strategy: {result['strategy']}")
    print(f"Period: {result['period']} ({result['trading_days']} trading days)")
    print(f"{'='*70}")
    print(f"Total return:      {result['total_return']:+.4f} ({result['total_return']*100:+.2f}%)")
    print(f"Avg daily return:  {result['avg_daily_return']:+.6f} ({result['avg_daily_return']*100:+.4f}%)")
    print(f"Avg market return: {result['avg_market_return']:+.6f} ({result['avg_market_return']*100:+.4f}%)")
    alpha = result['avg_daily_return'] - result['avg_market_return']
    print(f"Daily alpha:       {alpha:+.6f} ({alpha*100:+.4f}%)")
    print(f"Win rate:          {result['win_rate']:.1%} ({result['winning_days']}W / {result['losing_days']}L)")

    if "feature_importance" in result:
        print(f"\nFeature importance (learned weights):")
        for feat, coef in result["feature_importance"]:
            bar = "+" * int(min(abs(coef) * 200, 30)) if coef > 0 else "-" * int(min(abs(coef) * 200, 30))
            print(f"  {feat:<25} {coef:>+8.5f}  {bar}")

    print()

    # Daily detail
    print(f"{'Date':<12} {'Return':>9} {'Market':>9} {'Alpha':>9}  Picks")
    print("-" * 80)
    for day in result["daily_results"]:
        dr = day["day_return"]
        mr = day["market_avg_return"]
        picks_str = ", ".join(
            f"{sym} ({ret:+.2%})" for sym, score, ret in day["picks"]
        )
        print(f"{day['date']}  {dr:>+8.4%}  {mr:>+8.4%}  {dr-mr:>+8.4%}  {picks_str}")


@cli.command("compare")
@click.option("--min-price", default=1.0, help="Minimum stock price filter.")
@click.option("--min-volume", default=100_000.0, help="Minimum 10-day avg volume filter.")
def compare(min_price, min_volume):
    """Compare all strategies side-by-side."""
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM day_aggs")
            earliest, latest = cur.fetchone()
    finally:
        conn.close()

    if not earliest:
        print("No data available.")
        return

    print(f"Comparing all strategies: {earliest} to {latest}")
    print(f"Min price ${min_price}, min avg volume {min_volume:,.0f}")
    print()

    results = {}
    for name in STRATEGIES:
        print(f"  Running {name}...", flush=True)
        result = run_backtest(earliest, latest, name, 3, min_price, min_volume)
        if "error" not in result:
            results[name] = result
            alpha = result['avg_daily_return'] - result['avg_market_return']
            print(f"    done: alpha={alpha*100:+.4f}%, win={result['win_rate']:.1%}", flush=True)

    print(f"{'Strategy':<25} {'Total':>9} {'Avg Daily':>11} {'Market':>11} {'Alpha':>11} {'Win%':>7}")
    print("-" * 78)
    for name, r in results.items():
        alpha = r['avg_daily_return'] - r['avg_market_return']
        print(f"{name:<25} {r['total_return']:>+8.4f}  {r['avg_daily_return']*100:>+9.4f}%"
              f"  {r['avg_market_return']*100:>+9.4f}%  {alpha*100:>+9.4f}%  {r['win_rate']:>6.1%}")


@cli.command("predict")
@click.option("--top-n", "-n", default=3, help="Number of stocks to pick.")
@click.option("--min-price", default=10.0, help="Minimum stock price filter.")
@click.option("--min-volume", default=1_000_000.0, help="Minimum 10-day avg volume filter.")
@click.option("--capital", "-c", default=10_000.0, help="Total capital to allocate ($).")
@click.option("--skip-download", is_flag=True, help="Skip downloading new data.")
def predict(top_n, min_price, min_volume, capital, skip_download):
    """Download latest data and generate picks for next trading day.

    Run this each morning before market open. It will:
    1. Download any new day_aggs and news data
    2. Update the eod_cache
    3. Train ridge_alpha on recent data
    4. Output top picks with position sizing
    """
    import numpy as np
    from findgood.features import compute_features, compute_features_forward
    from findgood.backtest import _train_ridge, _features_to_df

    # Step 1: Download latest data
    if not skip_download:
        print("[1/4] Downloading latest data...", flush=True)
        ingest_data_type("day_aggs", "day_aggs_v1")
        fetch_news()
        print()
    else:
        print("[1/4] Skipping download.", flush=True)

    # Step 2: Update eod_cache
    print("[2/4] Updating eod_cache...", flush=True)
    build_eod_cache_v2()

    # Step 3: Train model
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(trade_date) FROM day_aggs")
            latest = cur.fetchone()[0]
            cur.execute("""
                SELECT DISTINCT trade_date FROM day_aggs
                ORDER BY trade_date DESC LIMIT 120
            """)
            train_dates = sorted([row[0] for row in cur.fetchall()])
    finally:
        conn.close()

    if not latest:
        print("No data available. Run 'findgood download' first.")
        return

    print(f"[3/4] Training ridge_alpha on {len(train_dates)} days "
          f"({train_dates[0]} to {train_dates[-1]})...", flush=True)

    all_training = []
    for i, td in enumerate(train_dates, 1):
        if i % 30 == 0:
            print(f"      [{i}/{len(train_dates)}]", flush=True)
        feats = compute_features(td, min_price, min_volume)
        if feats:
            for f in feats:
                beta = float(f.get("beta_spy_10d", 1))
                f["alpha_return"] = float(f["intraday_return"]) - beta * float(f.get("spy_return_1d", 0))
            all_training.append((feats, None))

    if not all_training:
        print("No training data available.")
        return

    model, scaler = _train_ridge(all_training, "alpha_return")
    total_samples = sum(len(t[0]) for t in all_training)
    print(f"      Trained on {total_samples:,} stock-days.", flush=True)

    # Step 4: Score forward-looking features
    print(f"[4/4] Scoring stocks...", flush=True)
    features = compute_features_forward(min_price, min_volume)
    if not features:
        print("No eligible stocks found.")
        return

    # Filter out ETFs and leveraged products — only common stock
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            # Get ticker types from the tickers that have details
            ticker_ids = [f["ticker_id"] for f in features]
            cur.execute("""
                SELECT td.ticker_id, td.sector, td.industry
                FROM ticker_details td
                WHERE td.ticker_id = ANY(%s)
            """, (ticker_ids,))
            details = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    finally:
        conn.close()

    # Attach sector info and filter
    for f in features:
        tid = f["ticker_id"]
        if tid in details:
            f["sector"] = details[tid][0] or "—"
            f["industry"] = details[tid][1] or "—"
        else:
            f["sector"] = "—"
            f["industry"] = "—"

    X = _features_to_df(features)
    mask = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1)

    predictions = np.full(len(features), -999.0)
    if mask.any():
        predictions[mask.values] = model.predict(scaler.transform(X[mask]))

    for j, f in enumerate(features):
        f["score"] = float(predictions[j])

    # Filter: only common stocks (have sector or industry), exclude leveraged/inverse
    from findgood.longterm import _is_leveraged
    scored = [f for f in features
              if f["score"] > -999.0
              and not _is_leveraged(f["symbol"])
              and f["sector"] != "—"]
    scored.sort(key=lambda f: f["score"], reverse=True)

    picks = scored[:top_n]
    per_position = capital / top_n if top_n > 0 else 0

    # Output
    print(f"\n{'='*58}")
    print(f"  FINDGOOD — Daily Picks")
    print(f"{'='*58}")
    print(f"  Data through:  {latest}")
    print(f"  Model:         ridge_alpha (120-day training window)")
    print(f"  Universe:      {len(scored):,} stocks "
          f"(price >= ${min_price}, vol >= {min_volume:,.0f})")
    print(f"{'='*58}")
    print(f"\n  Action: Buy at market open, sell at market close.")
    print(f"  Capital: ${capital:,.2f} | ${per_position:,.2f} per position\n")

    print(f"{'Rank':>4} {'Symbol':>7} {'Score':>9} {'Close':>8} "
          f"{'Shares':>7} {'Sector'}")
    print("-" * 62)
    for i, f in enumerate(picks, 1):
        price = float(f.get("prev_close", 0))
        shares = int(per_position / price) if price > 0 else 0
        cost = shares * price
        print(f"{i:>4} {f['symbol']:>7} {f['score']:>+9.5f} ${price:>7.2f} "
              f"{shares:>7} {f['sector']}")

    if picks:
        total_cost = sum(
            int(per_position / float(p.get("prev_close", 1))) * float(p.get("prev_close", 1))
            for p in picks
        )
        print(f"\n  Total invested: ${total_cost:,.2f} of ${capital:,.2f}")
    print()


@cli.command("buy-and-hold")
@click.option("--top-n", "-n", default=10, help="Number of stocks to pick.")
@click.option("--min-price", default=10.0, help="Minimum stock price filter.")
@click.option("--min-volume", default=500_000.0, help="Minimum 30-day avg volume filter.")
def buy_and_hold(top_n, min_price, min_volume):
    """Predict the best stocks to buy and hold for one year."""
    from findgood.longterm import train_and_predict

    print("FindGood — 1-Year Buy & Hold Predictions")
    print("=" * 50)
    print(f"Filters: price >= ${min_price}, avg vol >= {min_volume:,.0f}")
    print(f"Training on monthly samples with known 1yr forward returns...")
    print()

    result = train_and_predict(min_price, min_volume, top_n)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nFeature importance:")
    for feat, coef in result["feature_importance"]:
        direction = "+" if coef > 0 else "-"
        bar = direction * int(min(abs(coef) * 20, 30))
        print(f"  {feat:<30} {coef:>+8.4f}  {bar}")

    print(f"\n{'=' * 60}")
    print(f"  TOP {top_n} BUY & HOLD PICKS")
    print(f"  As of: {result['as_of_date']}")
    print(f"  Trained on: {result['training_samples']:,} stock-month samples")
    print(f"  Avg 1yr return in training data: {result['avg_training_return']:+.1%}")
    print(f"  Eligible stocks scored: {result['eligible_stocks']:,}")
    print(f"{'=' * 60}")
    print()
    print(f"{'Rank':>4} {'Symbol':>8} {'Predicted 1yr':>14} {'Price':>8} {'Sector':>20}")
    print("-" * 58)

    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            for i, f in enumerate(result["picks"], 1):
                # Get sector for display
                cur.execute("""
                    SELECT td.sector FROM ticker_details td
                    WHERE td.ticker_id = %s
                """, (f["ticker_id"],))
                sector_row = cur.fetchone()
                sector = sector_row[0] if sector_row and sector_row[0] else "—"

                print(f"{i:>4} {f['symbol']:>8} {f['predicted_1yr_return']:>+13.1%}"
                      f" ${float(f['close_price']):>7.2f} {sector:>20}")
    finally:
        conn.close()

    print()


@cli.command("buy-and-hold-backtest")
@click.option("--top-n", "-n", default=10, help="Number of stocks to pick per quarter.")
@click.option("--min-price", default=10.0, help="Minimum stock price filter.")
@click.option("--min-volume", default=500_000.0, help="Minimum 30-day avg volume filter.")
def buy_and_hold_backtest(top_n, min_price, min_volume):
    """Backtest buy-and-hold strategy vs S&P 500 (quarterly, 1yr holding)."""
    from findgood.longterm import backtest_vs_spy

    print("FindGood — Buy & Hold Backtest vs S&P 500")
    print("=" * 55)
    print(f"Pick top {top_n} stocks each quarter, hold for 1 year")
    print(f"Filters: price >= ${min_price}, avg vol >= {min_volume:,.0f}")
    print()

    result = backtest_vs_spy(min_price, min_volume, top_n)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print()
    print(f"{'=' * 55}")
    print(f"  RESULTS: {result['quarters_tested']} quarters tested")
    print(f"  Period: {result['period']}")
    print(f"{'=' * 55}")
    print()
    print(f"  Avg portfolio 1yr return:  {result['avg_pick_1yr_return']:>+7.1%}")
    print(f"  Avg S&P 500 1yr return:    {result['avg_spy_1yr_return']:>+7.1%}")
    print(f"  Avg alpha (vs SPY):        {result['avg_alpha']:>+7.1%}")
    print(f"  Beat SPY:                  {result['beat_spy_count']}/{result['quarters_tested']} "
          f"({result['beat_spy_rate']:.0%})")
    print()

    print(f"{'Quarter':<12} {'Portfolio':>10} {'S&P 500':>10} {'Alpha':>10}")
    print("-" * 45)
    for r in result["quarterly_results"]:
        print(f"{r['date']}  {r['avg_pick_return']:>+9.1%}  {r['spy_return']:>+9.1%}  {r['alpha']:>+9.1%}")
    print()


@cli.command("weekly-backtest")
@click.option("--top-n", "-n", default=5, help="Number of stocks to pick per week.")
@click.option("--min-price", default=10.0, help="Minimum stock price filter.")
@click.option("--min-volume", default=500_000.0, help="Minimum 20-day avg volume filter.")
def weekly_backtest(top_n, min_price, min_volume):
    """Backtest weekly strategy: buy Monday open, sell Friday close, vs S&P 500."""
    from findgood.weekly import backtest_weekly

    print("FindGood — Weekly Buy & Hold Backtest vs S&P 500")
    print("=" * 55)
    print(f"Pick top {top_n} stocks each week, hold Mon-Fri")
    print(f"Filters: price >= ${min_price}, avg vol >= {min_volume:,.0f}")
    print()

    result = backtest_weekly(min_price, min_volume, top_n)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print()
    print(f"{'=' * 55}")
    print(f"  RESULTS: {result['weeks_tested']} weeks tested")
    print(f"  Period: {result['period']}")
    print(f"{'=' * 55}")
    print()
    print(f"  Avg weekly return:       {result['avg_weekly_return']:>+7.2%}")
    print(f"  Avg SPY weekly return:   {result['avg_spy_weekly_return']:>+7.2%}")
    print(f"  Avg weekly alpha:        {result['avg_weekly_alpha']:>+7.2%}")
    print()
    print(f"  Cumulative return:       {result['cumulative_return']:>+7.1%}")
    print(f"  Cumulative SPY return:   {result['cumulative_spy_return']:>+7.1%}")
    print(f"  Cumulative alpha:        {result['cumulative_alpha']:>+7.1%}")
    print()
    print(f"  Beat SPY:  {result['beat_spy_count']}/{result['weeks_tested']} "
          f"({result['beat_spy_rate']:.0%})")

    if result.get("feature_importance"):
        print(f"\n  Feature importance:")
        for feat, coef in result["feature_importance"][:10]:
            print(f"    {feat:<30} {coef:>+8.5f}")

    print()


if __name__ == "__main__":
    cli()
