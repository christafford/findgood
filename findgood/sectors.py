"""Fetch and cache sector/industry classification for all tickers."""

import time
import requests
from findgood import db
from findgood.config import MASSIVE_API_BASE, api_key

# Major SPDR sector ETFs mapped to common sector names from the API
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Finance": "XLF",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Health Care": "XLV",
}


def init_sector_etfs():
    """Populate the sector_etfs mapping table."""
    conn = db.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for sector, etf in SECTOR_ETF_MAP.items():
                    cur.execute("""
                        INSERT INTO sector_etfs (sector, etf_symbol)
                        VALUES (%s, %s)
                        ON CONFLICT (sector) DO UPDATE SET etf_symbol = EXCLUDED.etf_symbol
                    """, (sector, etf))
        print(f"  Loaded {len(SECTOR_ETF_MAP)} sector-ETF mappings.")
    finally:
        conn.close()


def fetch_ticker_details():
    """Fetch sector/industry for all tickers missing details from the API."""
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.id, t.symbol FROM tickers t
                LEFT JOIN ticker_details td ON t.id = td.ticker_id
                WHERE td.ticker_id IS NULL
                ORDER BY t.id
            """)
            missing = cur.fetchall()
    finally:
        conn.close()

    if not missing:
        print("  All ticker details already fetched.")
        return

    print(f"  Fetching details for {len(missing)} tickers...", flush=True)

    session = requests.Session()
    batch_size = 100
    fetched = 0
    errors = 0

    for i, (ticker_id, symbol) in enumerate(missing, 1):
        if i % 200 == 0 or i == 1:
            print(f"    [{i}/{len(missing)}] {symbol}...", flush=True)

        try:
            resp = session.get(
                f"{MASSIVE_API_BASE}/v3/reference/tickers/{symbol}",
                params={"apiKey": api_key()},
            )

            if resp.status_code == 404:
                # Ticker not found — store with nulls so we don't retry
                _save_detail(ticker_id, None, None, None, None)
                fetched += 1
                continue

            if resp.status_code == 429:
                # Rate limited — wait and retry
                time.sleep(12)
                resp = session.get(
                    f"{MASSIVE_API_BASE}/v3/reference/tickers/{symbol}",
                    params={"apiKey": api_key()},
                )

            resp.raise_for_status()
            data = resp.json().get("results", {})

            _save_detail(
                ticker_id,
                data.get("sic_code"),
                data.get("sector"),
                data.get("industry"),
                data.get("market_cap"),
            )
            fetched += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error on {symbol}: {e}")
            elif errors == 6:
                print(f"    (suppressing further errors...)")
            # Store with nulls so we don't retry endlessly
            _save_detail(ticker_id, None, None, None, None)

        # Respect rate limits (~100 calls/min for free tier)
        if i % batch_size == 0:
            time.sleep(1)

    print(f"  Done: {fetched} fetched, {errors} errors.")


def _save_detail(ticker_id, sic_code, sector, industry, market_cap):
    conn = db.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ticker_details (ticker_id, sic_code, sector, industry, market_cap)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ticker_id) DO UPDATE SET
                        sic_code = EXCLUDED.sic_code,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        market_cap = EXCLUDED.market_cap,
                        fetched_at = now()
                """, (ticker_id, sic_code, sector, industry, market_cap))
    finally:
        conn.close()
