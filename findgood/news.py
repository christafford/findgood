import requests
from datetime import date, timedelta, datetime, timezone

from findgood import db
from findgood.config import MASSIVE_API_BASE, api_key, LOOKBACK_DAYS


def _fetch_page(session: requests.Session, url: str, params: dict | None = None):
    """Fetch a single page from the news API. Returns (results, next_url)."""
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", []), data.get("next_url")


def _ingest_news_day(session: requests.Session, target_date: date) -> int:
    """Fetch all news articles for a single day and insert sentiment rows."""
    start = datetime(target_date.year, target_date.month, target_date.day,
                     tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    url = f"{MASSIVE_API_BASE}/v2/reference/news"
    params = {
        "published_utc.gte": start.isoformat(),
        "published_utc.lt": end.isoformat(),
        "order": "asc",
        "sort": "published_utc",
        "limit": 1000,
        "apiKey": api_key(),
    }

    total_rows = 0
    page = 0

    while True:
        page += 1
        if page == 1:
            results, next_url = _fetch_page(session, url, params)
        else:
            # next_url doesn't include apiKey, so add it
            results, next_url = _fetch_page(session, next_url, {"apiKey": api_key()})

        if not results:
            break

        rows = _extract_sentiment_rows(results)
        if rows:
            inserted = _insert_sentiment_batch(rows)
            total_rows += inserted

        if page % 10 == 0:
            print(f"    page {page}, {total_rows} sentiments so far ...", flush=True)

        if not next_url:
            break

    return total_rows


def _extract_sentiment_rows(results: list[dict]) -> list[tuple]:
    """Extract (article_id, ticker, sentiment, published_utc) from API results."""
    rows = []
    for article in results:
        article_id = article.get("id")
        published = article.get("published_utc")
        insights = article.get("insights") or []

        if not article_id or not published:
            continue

        for insight in insights:
            ticker = insight.get("ticker")
            sentiment = insight.get("sentiment")
            if ticker and sentiment:
                rows.append((article_id, ticker, sentiment, published))

    return rows


def _insert_sentiment_batch(rows: list[tuple]) -> int:
    """Insert a batch of sentiment rows. Returns count inserted."""
    if not rows:
        return 0

    # Collect unique tickers and sentiments for resolution
    tickers = list({r[1] for r in rows})
    sentiments = list({r[2] for r in rows})

    conn = db.get_connection()
    try:
        with conn:
            ticker_map = db.resolve_ticker_ids(tickers, conn)
            sentiment_map = db.resolve_sentiment_ids(sentiments, conn)

            with conn.cursor() as cur:
                values = []
                for article_id, ticker, sentiment, published in rows:
                    ticker_id = ticker_map.get(ticker)
                    sentiment_id = sentiment_map.get(sentiment)
                    if ticker_id and sentiment_id:
                        values.append(
                            cur.mogrify(
                                "(%s,%s,%s,%s)",
                                (ticker_id, sentiment_id, published, article_id),
                            ).decode()
                        )

                if not values:
                    return 0

                cur.execute(f"""
                    INSERT INTO news_sentiment (ticker_id, sentiment_id, published_utc, article_id)
                    VALUES {",".join(values)}
                    ON CONFLICT (article_id, ticker_id) DO NOTHING
                """)
                return cur.rowcount
    finally:
        conn.close()


def fetch_news():
    """Fetch news sentiment for all days in the lookback window."""
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)

    already_done = db.get_fetched_news_dates()

    pending = []
    current = start
    while current <= today:
        if current not in already_done:
            pending.append(current)
        current += timedelta(days=1)

    if not pending:
        print("  news: all dates already fetched.")
        return

    print(f"  news: {len(pending)} days to fetch "
          f"({LOOKBACK_DAYS + 1 - len(pending)} already done)")

    session = requests.Session()

    for i, target_date in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {target_date} ... ", end="", flush=True)

        try:
            rows = _ingest_news_day(session, target_date)
            # Mark date as complete
            conn = db.get_connection()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO news_fetch_log (fetch_date, rows_fetched, completed)
                            VALUES (%s, %s, true)
                            ON CONFLICT (fetch_date) DO UPDATE
                            SET rows_fetched = EXCLUDED.rows_fetched,
                                completed = true, fetched_at = now()
                        """, (target_date, rows))
            finally:
                conn.close()

            print(f"{rows:,} sentiments")
        except Exception as e:
            print(f"error: {e}")
