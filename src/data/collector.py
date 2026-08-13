# src/data/collector.py
"""
Robust DataCollector for price and simple fundamentals retrieval.

Features:
- fetch_price: retries, exponential backoff, threads=False for yfinance,
  parquet caching in ./cache, returns DataFrame (empty if unavailable).
- fetch_tickers: reads tickers from a simple CSV or config (fallback list).
- fetch_fundamentals: placeholder that returns a DataFrame of fundamentals;
  extend to call FMP/AlphaVantage/other APIs as needed.
- clear, consistent logging for CI runs.
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger("DataCollector")
logger.setLevel(logging.INFO)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _save_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except Exception:
        logger.exception("Failed to write parquet cache %s", path)

class DataCollector:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_price(self,
                    ticker: str,
                    period: str = "3y",
                    interval: str = "1d",
                    force: bool = False,
                    max_retries: int = 3,
                    backoff: float = 2.0) -> pd.DataFrame:
        """
        Download price history for `ticker`. Returns a DataFrame (may be empty).
        Behavior:
          - Uses cache in cache/{TICKER}_price.parquet
          - Retries on transient errors with exponential backoff
          - Uses threads=False to reduce rate-limit blocks
          - Returns empty DataFrame on persistent failure (downstream code should handle)
        """
        cache_file = self.cache_dir / f"{ticker}_price.parquet"

        # Return cached if present and not forcing
        if cache_file.exists() and not force:
            try:
                df = pd.read_parquet(cache_file)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info("Using cached price for %s (rows=%d)", ticker, len(df))
                    return df
            except Exception:
                logger.warning("Failed to read cache for %s; will attempt download", ticker)

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                # threads=False reduces parallel connections that can trigger blocks
                df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    _save_parquet_safe(df, cache_file)
                    logger.info("Downloaded price for %s (rows=%d) on attempt %d", ticker, len(df), attempt)
                    return df
                else:
                    last_exc = RuntimeError("Empty DataFrame returned")
                    logger.warning("Empty price DataFrame for %s on attempt %d", ticker, attempt)
            except requests.exceptions.RequestException as re:
                last_exc = re
                logger.warning("Network error fetching %s attempt %d: %s", ticker, attempt, re)
            except Exception as e:
                last_exc = e
                logger.warning("Error fetching %s attempt %d: %s", ticker, attempt, e)

            time.sleep(backoff * attempt)

        # Fallback to stale cache if available
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.warning("Returning stale cached data for %s after failures", ticker)
                return df
            except Exception:
                logger.exception("Failed to read fallback cache for %s", ticker)

        logger.error("Failed to fetch price for %s after %d attempts. Last error: %s", ticker, max_retries, last_exc)
        return pd.DataFrame()

    def fetch_tickers(self, path: Optional[Path] = None) -> List[str]:
        """
        Load a list of tickers. Priority:
          1) CSV file at `path` with column 'ticker'
          2) default sample list (S&P 500 sample)
        """
        if path:
            try:
                df = pd.read_csv(path)
                if "ticker" in df.columns:
                    tickers = df["ticker"].dropna().astype(str).str.strip().tolist()
                    if tickers:
                        logger.info("Loaded %d tickers from %s", len(tickers), path)
                        return tickers
            except Exception:
                logger.exception("Failed to read tickers from %s", path)

        # fallback sample list (small)
        sample = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "PG", "NVDA", "TSLA", "META"]
        logger.info("Using fallback sample tickers (%d)", len(sample))
        return sample

    def fetch_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """
        Placeholder fundamentals fetcher.
        Returns a DataFrame with index=ticker and columns for common metrics (ROIC, PEG, GrossMargin).
        Extend this method to call FMP/AlphaVantage/SimFin and cache results.
        """
        rows = []
        for t in tickers:
            # deterministic placeholder values for testing
            base = sum(ord(c) for c in t) % 100
            rows.append({
                "ticker": t,
                "ROIC": (base % 30) / 100.0,         # 0.00 - 0.29
                "PEG": max(0.1, (100 - base) / 50.0),# 0.1 - 2.0
                "GrossMargin": ((base % 60) / 100.0) # 0.00 - 0.59
            })
        df = pd.DataFrame(rows).set_index("ticker")
        logger.info("Built placeholder fundamentals for %d tickers", len(tickers))
        return df

    # convenience: bulk fetch prices with graceful handling
    def fetch_prices_bulk(self, tickers: List[str], period: str = "3y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                results[t] = self.fetch_price(t, period=period, interval=interval)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", t, e)
                results[t] = pd.DataFrame()
        return results
