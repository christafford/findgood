import boto3
from datetime import date, timedelta
from findgood.config import s3_config, S3_BUCKET, LOOKBACK_DAYS


def get_s3_client():
    cfg = s3_config()
    return boto3.client("s3", **cfg, region_name="us-east-1")


def build_expected_keys(data_type: str) -> list[tuple[str, date]]:
    """Build list of (s3_key, trade_date) for each trading day in range.

    We generate keys for every weekday in range — S3 will simply 404 for
    holidays, which we handle gracefully.
    """
    s3_prefix = f"us_stocks_sip/{data_type}"
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)

    keys = []
    current = start
    while current <= today:
        # skip weekends
        if current.weekday() < 5:
            key = (
                f"{s3_prefix}/{current.year}/"
                f"{current.month:02d}/{current.isoformat()}.csv.gz"
            )
            keys.append((key, current))
        current += timedelta(days=1)

    return keys


def download_file(s3_client, s3_key: str, dest_path: str) -> int:
    """Download an S3 object to dest_path. Returns file size in bytes."""
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
    size = resp["ContentLength"]
    with open(dest_path, "wb") as f:
        for chunk in resp["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    return size
