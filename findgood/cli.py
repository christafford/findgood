from datetime import date, timedelta

import click
from findgood import db
from findgood.ingest import ingest_data_type
from findgood.news import fetch_news
from findgood.backtest import run_backtest, STRATEGIES
from findgood.features import build_eod_cache_v2


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


@cli.command("backtest")
@click.option("--strategy", "-s",
              type=click.Choice(list(STRATEGIES.keys()), case_sensitive=False),
              default="momentum_sentiment",
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
        result = run_backtest(earliest, latest, name, 3, min_price, min_volume)
        if "error" not in result:
            results[name] = result

    print(f"{'Strategy':<25} {'Total':>9} {'Avg Daily':>11} {'Market':>11} {'Alpha':>11} {'Win%':>7}")
    print("-" * 78)
    for name, r in results.items():
        alpha = r['avg_daily_return'] - r['avg_market_return']
        print(f"{name:<25} {r['total_return']:>+8.4f}  {r['avg_daily_return']*100:>+9.4f}%"
              f"  {r['avg_market_return']*100:>+9.4f}%  {alpha*100:>+9.4f}%  {r['win_rate']:>6.1%}")


if __name__ == "__main__":
    cli()
