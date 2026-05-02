import csv
import gzip
import io
import os
import tempfile
from datetime import date, datetime, timezone

from botocore.exceptions import ClientError

from findgood import db
from findgood.downloader import get_s3_client, build_expected_keys, download_file


def _ns_to_date(ns_str: str) -> date:
    """Convert nanosecond unix timestamp string to a date."""
    ts_seconds = int(ns_str) / 1_000_000_000
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).date()


def _ns_to_timestamp_str(ns_str: str) -> str:
    """Convert nanosecond unix timestamp string to ISO timestamp."""
    ts_seconds = int(ns_str) / 1_000_000_000
    dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
    return dt.isoformat()


def _collect_tickers(gz_path: str) -> list[str]:
    """Read the CSV and return all unique ticker symbols."""
    tickers = set()
    with gzip.open(gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.add(row["ticker"])
    return list(tickers)


def _ingest_day_aggs(gz_path: str, trade_date: date, conn) -> int:
    """Parse a day_aggs csv.gz and COPY into day_aggs table. Returns row count."""
    # First pass: collect tickers and resolve IDs
    ticker_map = db.resolve_ticker_ids(_collect_tickers(gz_path), conn)

    # Second pass: build CSV with ticker_id
    buf = io.StringIO()
    writer = csv.writer(buf)
    row_count = 0

    with gzip.open(gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            writer.writerow([
                ticker_map[row["ticker"]],
                trade_date.isoformat(),
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["transactions"],
            ])
            row_count += 1

    buf.seek(0)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _day_aggs_stage (
                ticker_id integer, trade_date date, open numeric(12,4),
                high numeric(12,4), low numeric(12,4), close numeric(12,4),
                volume numeric(20,6), transactions integer
            ) ON COMMIT DROP
        """)
        cur.copy_expert(
            "COPY _day_aggs_stage (ticker_id, trade_date, open, high, low, close, volume, transactions) "
            "FROM STDIN WITH CSV",
            buf,
        )
        cur.execute("""
            INSERT INTO day_aggs (ticker_id, trade_date, open, high, low, close, volume, transactions)
            SELECT ticker_id, trade_date, open, high, low, close, volume, transactions
            FROM _day_aggs_stage
            ON CONFLICT (ticker_id, trade_date) DO NOTHING
        """)

    return row_count


def _ingest_minute_aggs(gz_path: str, trade_date: date, conn) -> int:
    """Parse a minute_aggs csv.gz and COPY into minute_aggs table. Returns row count."""
    # First pass: collect tickers and resolve IDs
    ticker_map = db.resolve_ticker_ids(_collect_tickers(gz_path), conn)

    # Second pass: build CSV with ticker_id
    buf = io.StringIO()
    writer = csv.writer(buf)
    row_count = 0

    with gzip.open(gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = _ns_to_timestamp_str(row["window_start"])
            writer.writerow([
                ticker_map[row["ticker"]],
                trade_date.isoformat(),
                ts_str,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["transactions"],
            ])
            row_count += 1

    buf.seek(0)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _minute_aggs_stage (
                ticker_id integer, trade_date date, window_start timestamptz,
                open numeric(12,4), high numeric(12,4), low numeric(12,4),
                close numeric(12,4), volume numeric(20,6), transactions integer
            ) ON COMMIT DROP
        """)
        cur.copy_expert(
            "COPY _minute_aggs_stage (ticker_id, trade_date, window_start, open, high, low, close, volume, transactions) "
            "FROM STDIN WITH CSV",
            buf,
        )
        cur.execute("""
            INSERT INTO minute_aggs (ticker_id, trade_date, window_start, open, high, low, close, volume, transactions)
            SELECT ticker_id, trade_date, window_start, open, high, low, close, volume, transactions
            FROM _minute_aggs_stage
            ON CONFLICT (ticker_id, trade_date, window_start) DO NOTHING
        """)

    return row_count


def ingest_data_type(data_type_label: str, s3_data_type: str):
    """Download and ingest all missing files for a given data type."""
    s3 = get_s3_client()
    expected = build_expected_keys(s3_data_type)
    already_done = db.get_downloaded_keys(data_type_label)

    pending = [(key, td) for key, td in expected if key not in already_done]

    if not pending:
        print(f"  {data_type_label}: all files already downloaded.")
        return

    print(f"  {data_type_label}: {len(pending)} files to download "
          f"({len(expected) - len(pending)} already done)")

    ingest_fn = _ingest_day_aggs if data_type_label == "day_aggs" else _ingest_minute_aggs

    for i, (s3_key, trade_date) in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {s3_key} ... ", end="", flush=True)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv.gz")
        os.close(tmp_fd)

        try:
            file_size = download_file(s3, s3_key, tmp_path)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                print("not available (holiday?), skipping")
            else:
                print(f"download error: {e}")
            os.unlink(tmp_path)
            continue

        conn = db.get_connection()
        try:
            with conn:
                rows = ingest_fn(tmp_path, trade_date, conn)
            db.record_download(s3_key, data_type_label, trade_date, rows, file_size)
            print(f"{rows:,} rows")
        except Exception as e:
            print(f"ingest error: {e}")
            conn.rollback()
        finally:
            conn.close()
            os.unlink(tmp_path)
