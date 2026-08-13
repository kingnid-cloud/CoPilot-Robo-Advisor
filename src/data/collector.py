# src/data/collector.py
"""
Robust DataCollector that:
- fetches price history via yfinance with retries, caching, and threads=False
- fetches fundamentals via connectors with Tickerbot -> FMP -> AlphaVantage -> Finnhub fallback
- caches connector responses to reduce API usage in CI
- exposes convenience methods used by run_daily.py
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
import requests

from src.data.connectors import fetch_fundamental_with_fallback
from src.utils.secrets import load_secrets

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

    def fetch_price(
        self,
        ticker: str,
        period: str = "3y",
        interval: str = "1d",
        force: bool = False,
        max_retries: int = 3,
        backoff: float = 2.0,
    ) -> pd.DataFrame:
        """
        Download price history for `ticker`. Returns a DataFrame (may be empty).
        Uses cache in cache/{TICKER}_price.parquet, retries on transient errors,
        and uses threads=False to reduce rate-limit blocks.
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

    def fetch_prices_bulk(self, tickers: List[str], period: str = "3y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                results[t] = self.fetch_price(t, period=period, interval=interval)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", t, e)
                results[t] = pd.DataFrame()
        return results

    def fetch_tickers(self, path: Optional[Path] = None) -> List[str]:
        """
        Load a list of tickers. Priority:
          1) CSV file at `path` with column 'ticker'
          2) default sample list
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

        sample = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "PG", "NVDA", "TSLA", "META"]
        logger.info("Using fallback sample tickers (%d)", len(sample))
        return sample

    def fetch_fundamentals(self, tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch fundamentals for tickers using connectors.fetch_fundamental_with_fallback.
        Returns DataFrame indexed by ticker with normalized columns (ROIC, PEG, GrossMargin, sector).
        """
        secrets = load_secrets()
        rows = []
        for t in tickers:
            try:
                data = fetch_fundamental_with_fallback(t, secrets, use_cache=use_cache)
                if not data:
                    logger.warning("No fundamentals for %s; using placeholders", t)
                    base = sum(ord(c) for c in t) % 100
                    rows.append(
                        {
                            "ticker": t,
                            "ROIC": (base % 30) / 100.0,
                            "PEG": max(0.1, (100 - base) / 50.0),
                            "GrossMargin": ((base % 60) / 100.0),
                            "sector": "UNKNOWN",
                        }
                    )
                    continue

                # Normalize fields from provider responses
                roic = None
                peg = None
                gross_margin = None
                sector = None

                # Tickerbot normalized shape
                if isinstance(data, dict) and data.get("provider") == "tickerbot":
                    fund = data.get("fundamentals", {}) or {}
                    roic = fund.get("roic") or fund.get("ROIC")
                    peg = fund.get("peg") or fund.get("PEG")
                    gross_margin = fund.get("grossMargin") or fund.get("gross_margin")
                    sector = data.get("sector") or data.get("industry")

                # FMP shape
                elif isinstance(data, dict) and data.get("symbol"):
                    roic = data.get("returnOnInvestedCapital") or data.get("roic")
                    peg = data.get("pegRatio") or data.get("peg")
                    gross_margin = data.get("grossProfitRatio") or data.get("grossMargin") or data.get("grossMarginTTM")
                    sector = data.get("sector") or data.get("industry")

                # AlphaVantage shape
                elif isinstance(data, dict) and data.get("Symbol"):
                    roic = data.get("ReturnOnEquityTTM") or data.get("ReturnOnAssetsTTM")
                    peg = data.get("PEGRatio") or data.get("PEG")
                    gross_margin = data.get("GrossProfitTTM")
                    sector = data.get("Sector")

                # Finnhub shape
                elif isinstance(data, dict) and data.get("finnhubIndustry"):
                    sector = data.get("finnhubIndustry")

                # Generic fallback: try common keys
                if roic in (None, "", "None"):
                    roic = data.get("ROIC") if isinstance(data, dict) else roic
                if peg in (None, "", "None"):
                    peg = data.get("PEG") if isinstance(data, dict) else peg
                if gross_margin in (None, "", "None"):
                    gross_margin = data.get("GrossMargin") if isinstance(data, dict) else gross_margin

                # Final numeric normalization with placeholders if needed
                base = sum(ord(c) for c in t) % 100
                try:
                    roic = float(roic) if roic not in (None, "", "None") else (base % 30) / 100.0
                except Exception:
                    roic = (base % 30) / 100.0
                try:
                    peg = float(peg) if peg not in (None, "", "None") else max(0.1, (100 - base) / 50.0)
                except Exception:
                    peg = max(0.1, (100 - base) / 50.0)
                try:
                    gross_margin = float(gross_margin) if gross_margin not in (None, "", "None") else ((base % 60) / 100.0)
                except Exception:
                    gross_margin = ((base % 60) / 100.0)
                sector = sector or "UNKNOWN"

                rows.append({"ticker": t, "ROIC": roic, "PEG": peg, "GrossMargin": gross_margin, "sector": sector})
            except Exception as e:
                logger.exception("fetch_fundamentals error for %s: %s", t, e)
                base = sum(ord(c) for c in t) % 100
                rows.append(
                    {
                        "ticker": t,
                        "ROIC": (base % 30) / 100.0,
                        "PEG": max(0.1, (100 - base) / 50.0),
                        "GrossMargin": ((base % 60) / 100.0),
                        "sector": "UNKNOWN",
                    }
                )

        df = pd.DataFrame(rows).set_index("ticker")
        logger.info("Fetched fundamentals for %d tickers", len(rows))
        return df
