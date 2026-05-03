import os
import sys


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return val


def db_dsn() -> str:
    return (
        f"host={_require('DB_HOST')} "
        f"port={_require('DB_PORT')} "
        f"dbname={_require('DB_NAME')} "
        f"user={_require('DB_USER')} "
        f"password={_require('DB_PASSWORD')}"
    )


def s3_config() -> dict:
    return {
        "endpoint_url": _require("MASSIVE_S3_ENDPOINT"),
        "aws_access_key_id": _require("MASSIVE_S3_ACCESS_KEY"),
        "aws_secret_access_key": _require("MASSIVE_S3_SECRET_KEY"),
    }


S3_BUCKET = os.environ.get("MASSIVE_S3_BUCKET", "flatfiles")

MASSIVE_API_BASE = "https://api.massive.com"


def api_key() -> str:
    return _require("MASSIVE_API_KEY")

LOOKBACK_DAYS = 1825
