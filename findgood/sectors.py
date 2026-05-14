"""Fetch and cache sector/industry classification for all tickers."""

import time
import requests
from findgood import db
from findgood.config import MASSIVE_API_BASE, api_key

# Map SIC code ranges to sectors and corresponding ETFs
# Based on standard SIC division structure
SIC_SECTOR_MAP = [
    # (sic_low, sic_high, sector_name, etf_symbol)
    ("0100", "0999", "Agriculture", "XLB"),
    ("1000", "1499", "Mining", "XLE"),
    ("1500", "1799", "Construction", "XLI"),
    ("2000", "3999", "Manufacturing", "XLI"),
    ("4000", "4999", "Transportation & Utilities", "XLU"),
    ("5000", "5199", "Wholesale Trade", "XLY"),
    ("5200", "5999", "Retail Trade", "XLY"),
    ("6000", "6799", "Finance", "XLF"),
    ("7000", "8999", "Services", "XLK"),
    ("9100", "9999", "Public Administration", "XLI"),
]

# Override specific SIC ranges for better sector mapping
SIC_OVERRIDES = {
    # Technology
    "3571": ("Technology", "XLK"),  # Electronic computers
    "3572": ("Technology", "XLK"),  # Computer storage
    "3577": ("Technology", "XLK"),  # Computer peripherals
    "3674": ("Technology", "XLK"),  # Semiconductors
    "3679": ("Technology", "XLK"),  # Electronic components
    "7372": ("Technology", "XLK"),  # Prepackaged software
    "7371": ("Technology", "XLK"),  # Computer services
    "7374": ("Technology", "XLK"),  # Data processing
    "3669": ("Technology", "XLK"),  # Communications equipment
    "3672": ("Technology", "XLK"),  # Printed circuit boards
    "3825": ("Technology", "XLK"),  # Instruments
    # Healthcare
    "2830": ("Healthcare", "XLV"),
    "2833": ("Healthcare", "XLV"),
    "2834": ("Healthcare", "XLV"),  # Pharmaceutical
    "2835": ("Healthcare", "XLV"),
    "2836": ("Healthcare", "XLV"),  # Biological products
    "3841": ("Healthcare", "XLV"),  # Surgical instruments
    "3842": ("Healthcare", "XLV"),
    "3845": ("Healthcare", "XLV"),  # Electromedical
    "5912": ("Healthcare", "XLV"),  # Drug stores
    "8000": ("Healthcare", "XLV"),
    "8011": ("Healthcare", "XLV"),
    "8049": ("Healthcare", "XLV"),
    "8060": ("Healthcare", "XLV"),
    "8062": ("Healthcare", "XLV"),
    "8071": ("Healthcare", "XLV"),
    "8082": ("Healthcare", "XLV"),
    "8090": ("Healthcare", "XLV"),
    "8093": ("Healthcare", "XLV"),
    # Communication Services
    "4812": ("Communication Services", "XLC"),
    "4813": ("Communication Services", "XLC"),
    "4833": ("Communication Services", "XLC"),
    "4841": ("Communication Services", "XLC"),
    "7812": ("Communication Services", "XLC"),
    "7819": ("Communication Services", "XLC"),
    # Energy
    "1311": ("Energy", "XLE"),
    "1381": ("Energy", "XLE"),
    "1382": ("Energy", "XLE"),
    "1389": ("Energy", "XLE"),
    "2911": ("Energy", "XLE"),
    "5171": ("Energy", "XLE"),
    # Real Estate
    "6500": ("Real Estate", "XLRE"),
    "6510": ("Real Estate", "XLRE"),
    "6512": ("Real Estate", "XLRE"),
    "6552": ("Real Estate", "XLRE"),
    "6798": ("Real Estate", "XLRE"),
    # Utilities
    "4911": ("Utilities", "XLU"),
    "4922": ("Utilities", "XLU"),
    "4923": ("Utilities", "XLU"),
    "4924": ("Utilities", "XLU"),
    "4931": ("Utilities", "XLU"),
    "4932": ("Utilities", "XLU"),
    "4941": ("Utilities", "XLU"),
    # Materials
    "2611": ("Materials", "XLB"),
    "2621": ("Materials", "XLB"),
    "2800": ("Materials", "XLB"),
    "2810": ("Materials", "XLB"),
    "2820": ("Materials", "XLB"),
    "2860": ("Materials", "XLB"),
    "2890": ("Materials", "XLB"),
    "3310": ("Materials", "XLB"),
    "3312": ("Materials", "XLB"),
    "3317": ("Materials", "XLB"),
    "3350": ("Materials", "XLB"),
    "3411": ("Materials", "XLB"),
}


def _sic_to_sector(sic_code: str) -> tuple[str, str] | None:
    """Map a SIC code to (sector, etf_symbol)."""
    if not sic_code:
        return None

    # Check specific overrides first
    if sic_code in SIC_OVERRIDES:
        return SIC_OVERRIDES[sic_code]

    # Fall back to range-based mapping
    for low, high, sector, etf in SIC_SECTOR_MAP:
        if low <= sic_code <= high:
            return (sector, etf)

    return None


def init_sector_etfs():
    """Populate the sector_etfs mapping table."""
    # Collect all unique sectors from our mappings
    sectors = {}
    for low, high, sector, etf in SIC_SECTOR_MAP:
        sectors[sector] = etf
    for sic, (sector, etf) in SIC_OVERRIDES.items():
        sectors[sector] = etf

    conn = db.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for sector, etf in sectors.items():
                    cur.execute("""
                        INSERT INTO sector_etfs (sector, etf_symbol)
                        VALUES (%s, %s)
                        ON CONFLICT (sector) DO UPDATE SET etf_symbol = EXCLUDED.etf_symbol
                    """, (sector, etf))
        print(f"  Loaded {len(sectors)} sector-ETF mappings.")
    finally:
        conn.close()


def fetch_ticker_details():
    """Fetch SIC code for all tickers missing details, then derive sector."""
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
        _backfill_sectors()
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
                _save_detail(ticker_id, None, None, None, None)
                fetched += 1
                continue

            if resp.status_code == 429:
                time.sleep(12)
                resp = session.get(
                    f"{MASSIVE_API_BASE}/v3/reference/tickers/{symbol}",
                    params={"apiKey": api_key()},
                )

            resp.raise_for_status()
            data = resp.json().get("results", {})

            sic_code = data.get("sic_code")
            market_cap = data.get("market_cap")
            mapping = _sic_to_sector(sic_code)
            sector = mapping[0] if mapping else None
            industry = data.get("sic_description")

            _save_detail(ticker_id, sic_code, sector, industry, market_cap)
            fetched += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error on {symbol}: {e}")
            elif errors == 6:
                print(f"    (suppressing further errors...)")
            _save_detail(ticker_id, None, None, None, None)

        if i % batch_size == 0:
            time.sleep(1)

    print(f"  Done: {fetched} fetched, {errors} errors.")
    _backfill_sectors()


def _backfill_sectors():
    """Update sector for any tickers that have sic_code but no sector."""
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker_id, sic_code FROM ticker_details
                WHERE sic_code IS NOT NULL AND sector IS NULL
            """)
            to_update = cur.fetchall()

        if not to_update:
            return

        print(f"  Backfilling sector for {len(to_update)} tickers from SIC codes...")
        updated = 0
        with conn:
            with conn.cursor() as cur:
                for ticker_id, sic_code in to_update:
                    mapping = _sic_to_sector(sic_code)
                    if mapping:
                        cur.execute("""
                            UPDATE ticker_details SET sector = %s
                            WHERE ticker_id = %s
                        """, (mapping[0], ticker_id))
                        updated += 1
        print(f"  Updated {updated} tickers with sector data.")
    finally:
        conn.close()


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
