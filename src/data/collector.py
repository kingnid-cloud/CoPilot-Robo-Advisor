import yfinance as yf
import pandas as pd
from pathlib import Path
import time

class DataCollector:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # robust fetch_price with retries and threads=False
import time
import logging
from pathlib import Path
import yfinance as yf
import pandas as pd
import requests

logger = logging.getLogger(__name__)
CACHE_DIR = Path("cache")

def fetch_price(ticker, period='3y', interval='1d', cache_dir=CACHE_DIR, force=False, max_retries=3, backoff=2.0):
    cache_file = cache_dir / f"{ticker}_price.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # return cached if present and not forcing
    if cache_file.exists() and not force:
        try:
            df = pd.read_parquet(cache_file)
            if df is not None and not df.empty:
                logger.info("Using cached price for %s", ticker)
                return df
        except Exception:
            logger.warning("Failed to read cache for %s, will re-download", ticker)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Use threads=False to avoid many parallel connections that can trigger blocks
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_parquet(cache_file)
                logger.info("Downloaded price for %s (rows=%d)", ticker, len(df))
                return df
            else:
                logger.warning("Empty price DataFrame for %s on attempt %d", ticker, attempt)
                last_exc = RuntimeError("Empty DataFrame")
        except requests.exceptions.RequestException as re:
            last_exc = re
            logger.warning("Network error fetching %s attempt %d: %s", ticker, attempt, re)
        except Exception as e:
            last_exc = e
            logger.warning("Error fetching %s attempt %d: %s", ticker, attempt, e)

        # exponential backoff
        time.sleep(backoff * attempt)

    # final fallback: if cache exists return it even if stale, else raise
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            logger.warning("Returning stale cached data for %s after failures", ticker)
            return df
        except Exception:
            logger.exception("Failed to read fallback cache for %s", ticker)

    # if we get here, no data available
    logger.error("Failed to fetch price for %s after %d attempts. Last error: %s", ticker, max_retries, last_exc)
    raise RuntimeError(f"No price data for {ticker}") from last_exc

