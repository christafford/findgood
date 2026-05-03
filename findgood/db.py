import psycopg2
from datetime import date, timedelta
from findgood.config import db_dsn

SCHEMA_SQL = """
-- Normalized ticker symbols
CREATE TABLE IF NOT EXISTS tickers (
    id      serial PRIMARY KEY,
    symbol  text UNIQUE NOT NULL
);

-- Track which S3 files have been ingested
CREATE TABLE IF NOT EXISTS download_log (
    id              serial PRIMARY KEY,
    s3_key          text UNIQUE NOT NULL,
    data_type       text NOT NULL,
    trade_date      date NOT NULL,
    rows_inserted   integer NOT NULL DEFAULT 0,
    file_size_bytes bigint,
    downloaded_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_download_log_data_type ON download_log (data_type);
CREATE INDEX IF NOT EXISTS idx_download_log_trade_date ON download_log (trade_date);

-- Daily OHLCV aggregates
CREATE TABLE IF NOT EXISTS day_aggs (
    ticker_id    integer NOT NULL REFERENCES tickers(id),
    trade_date   date NOT NULL,
    open         numeric(12,4),
    high         numeric(12,4),
    low          numeric(12,4),
    close        numeric(12,4),
    volume       numeric(20,6),
    transactions integer,
    PRIMARY KEY (ticker_id, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_day_aggs_date ON day_aggs (trade_date);

-- Normalized sentiment values
CREATE TABLE IF NOT EXISTS sentiments (
    id      smallserial PRIMARY KEY,
    label   text UNIQUE NOT NULL
);

-- News sentiment per ticker per article
CREATE TABLE IF NOT EXISTS news_sentiment (
    ticker_id     integer NOT NULL REFERENCES tickers(id),
    sentiment_id  smallint NOT NULL REFERENCES sentiments(id),
    published_utc timestamptz NOT NULL,
    article_id    text NOT NULL,
    PRIMARY KEY (article_id, ticker_id)
);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date
    ON news_sentiment (ticker_id, published_utc);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_date
    ON news_sentiment (published_utc);

-- Track news API fetch progress (cursor-based, per-date)
CREATE TABLE IF NOT EXISTS news_fetch_log (
    id           serial PRIMARY KEY,
    fetch_date   date UNIQUE NOT NULL,
    rows_fetched integer NOT NULL DEFAULT 0,
    completed    boolean NOT NULL DEFAULT false,
    fetched_at   timestamptz NOT NULL DEFAULT now()
);

-- Minute OHLCV aggregates (partitioned by month)
CREATE TABLE IF NOT EXISTS minute_aggs (
    ticker_id    integer NOT NULL,
    trade_date   date NOT NULL,
    window_start timestamptz NOT NULL,
    open         numeric(12,4),
    high         numeric(12,4),
    low          numeric(12,4),
    close        numeric(12,4),
    volume       numeric(20,6),
    transactions integer,
    PRIMARY KEY (ticker_id, trade_date, window_start)
) PARTITION BY RANGE (trade_date);
"""


def get_connection():
    return psycopg2.connect(db_dsn())


def init_schema():
    """Create tables, indexes, and monthly partitions for minute_aggs."""
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                if _needs_migration(cur):
                    print("Migrating existing schema to normalized tickers...")
                    _run_migration(cur)
                    _migrate_minute_partitions(cur)
                    print("Migration complete.")
                else:
                    cur.execute(SCHEMA_SQL)
                    _create_minute_partitions(cur)
        print("Database schema initialized.")
    finally:
        conn.close()


def _needs_migration(cur) -> bool:
    """Check if day_aggs exists with a text 'ticker' column (old schema)."""
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'day_aggs' AND column_name = 'ticker'
              AND data_type = 'text'
        )
    """)
    return cur.fetchone()[0]


def _run_migration(cur):
    """Migrate day_aggs from text ticker to integer ticker_id."""
    # 1. Create tickers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            id      serial PRIMARY KEY,
            symbol  text UNIQUE NOT NULL
        )
    """)

    # 2. Populate tickers from existing data
    cur.execute("""
        INSERT INTO tickers (symbol)
        SELECT DISTINCT ticker FROM day_aggs
        ON CONFLICT (symbol) DO NOTHING
    """)

    # 3. Add ticker_id column
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'day_aggs' AND column_name = 'ticker_id'
        )
    """)
    if not cur.fetchone()[0]:
        cur.execute("ALTER TABLE day_aggs ADD COLUMN ticker_id integer")

    # 4. Populate ticker_id
    cur.execute("""
        UPDATE day_aggs SET ticker_id = t.id
        FROM tickers t WHERE day_aggs.ticker = t.symbol
        AND day_aggs.ticker_id IS NULL
    """)

    # 5. Drop old PK, drop ticker column, add new PK and FK
    cur.execute("ALTER TABLE day_aggs DROP CONSTRAINT IF EXISTS day_aggs_pkey")
    cur.execute("ALTER TABLE day_aggs DROP COLUMN IF EXISTS ticker")
    cur.execute("ALTER TABLE day_aggs ALTER COLUMN ticker_id SET NOT NULL")
    cur.execute("ALTER TABLE day_aggs ADD PRIMARY KEY (ticker_id, trade_date)")
    cur.execute("""
        ALTER TABLE day_aggs ADD CONSTRAINT day_aggs_ticker_fk
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
    """)


def _migrate_minute_partitions(cur):
    """Migrate existing minute_aggs partitions from text ticker to ticker_id.

    Since minute_aggs is partitioned and likely empty at migration time,
    the simplest approach is to drop and recreate.
    """
    # Check if minute_aggs has the old 'ticker' column
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'minute_aggs' AND column_name = 'ticker'
              AND data_type = 'text'
        )
    """)
    if not cur.fetchone()[0]:
        _create_minute_partitions(cur)
        return

    # Drop all partitions and parent, then recreate
    cur.execute("""
        SELECT inhrelid::regclass::text FROM pg_inherits
        WHERE inhparent = 'minute_aggs'::regclass
    """)
    partitions = [row[0] for row in cur.fetchall()]
    for part in partitions:
        cur.execute(f"DROP TABLE IF EXISTS {part}")
    cur.execute("DROP TABLE IF EXISTS minute_aggs")

    # Recreate with new schema
    cur.execute("""
        CREATE TABLE minute_aggs (
            ticker_id    integer NOT NULL,
            trade_date   date NOT NULL,
            window_start timestamptz NOT NULL,
            open         numeric(12,4),
            high         numeric(12,4),
            low          numeric(12,4),
            close        numeric(12,4),
            volume       numeric(20,6),
            transactions integer,
            PRIMARY KEY (ticker_id, trade_date, window_start)
        ) PARTITION BY RANGE (trade_date)
    """)
    _create_minute_partitions(cur)


def _create_minute_partitions(cur):
    """Create monthly partitions covering the full lookback window plus 2 months ahead."""
    from findgood.config import LOOKBACK_DAYS
    today = date.today()
    start = date(today.year, today.month, 1) - timedelta(days=LOOKBACK_DAYS + 30)
    start = date(start.year, start.month, 1)

    # Calculate number of months to cover
    months_needed = (LOOKBACK_DAYS // 30) + 4
    current = start
    for _ in range(months_needed):
        part_name = f"minute_aggs_{current.strftime('%Y_%m')}"
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)

        sql = f"""
            CREATE TABLE IF NOT EXISTS {part_name}
            PARTITION OF minute_aggs
            FOR VALUES FROM ('{current.isoformat()}') TO ('{next_month.isoformat()}');
        """
        try:
            cur.execute(sql)
        except psycopg2.errors.InvalidObjectDefinition:
            cur.connection.rollback()
            cur.connection.autocommit = False

        idx_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{part_name}_ticker_date
            ON {part_name} (ticker_id, trade_date);
        """
        cur.execute(idx_sql)

        current = next_month


def get_downloaded_keys(data_type: str) -> set[str]:
    """Return set of S3 keys already ingested for a given data_type."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT s3_key FROM download_log WHERE data_type = %s",
                (data_type,),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def record_download(s3_key: str, data_type: str, trade_date: date,
                    rows_inserted: int, file_size_bytes: int):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO download_log
                       (s3_key, data_type, trade_date, rows_inserted, file_size_bytes)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (s3_key) DO NOTHING""",
                    (s3_key, data_type, trade_date, rows_inserted, file_size_bytes),
                )
    finally:
        conn.close()


def get_fetched_news_dates() -> set[date]:
    """Return set of dates where news fetch is complete."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fetch_date FROM news_fetch_log WHERE completed = true"
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def resolve_sentiment_ids(labels: list[str], conn) -> dict[str, int]:
    """Ensure all sentiment labels exist and return label->id map."""
    if not labels:
        return {}

    with conn.cursor() as cur:
        args = ",".join(cur.mogrify("(%s)", (s,)).decode() for s in set(labels))
        cur.execute(f"""
            INSERT INTO sentiments (label) VALUES {args}
            ON CONFLICT (label) DO NOTHING
        """)
        cur.execute(
            "SELECT label, id FROM sentiments WHERE label = ANY(%s)",
            (list(set(labels)),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def resolve_ticker_ids(symbols: list[str], conn) -> dict[str, int]:
    """Ensure all ticker symbols exist in the tickers table and return a symbol->id map.

    Inserts any new symbols, then returns the full mapping for the given list.
    """
    if not symbols:
        return {}

    with conn.cursor() as cur:
        # Batch upsert all symbols
        args = ",".join(cur.mogrify("(%s)", (s,)).decode() for s in set(symbols))
        cur.execute(f"""
            INSERT INTO tickers (symbol) VALUES {args}
            ON CONFLICT (symbol) DO NOTHING
        """)

        # Fetch IDs for all requested symbols
        cur.execute(
            "SELECT symbol, id FROM tickers WHERE symbol = ANY(%s)",
            (list(set(symbols)),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}
