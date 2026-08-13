# src/data/connectors.py
"""
API connector helpers with fallback strategy and simple rate-limit handling.

Functions:
- call_alphavantage
- call_fmp
- call_finnhub
- fetch_fundamental_with_fallback

Each call returns parsed JSON/dict or None on failure.
Cache responses externally (DataCollector) to avoid repeated calls.
"""

from typing import Optional, Dict
import time
import logging
import requests

logger = logging.getLogger("connectors")
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 15.0
RATE_LIMIT_SLEEP = 1.0  # base sleep between calls to avoid bursts

def _safe_get(url: str, params: Dict, timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data
    except requests.exceptions.HTTPError as he:
        logger.warning("HTTP error for %s: %s", url, he)
    except requests.exceptions.RequestException as re:
        logger.warning("Network error for %s: %s", url, re)
    except ValueError as ve:
        logger.warning("Invalid JSON from %s: %s", url, ve)
    return None

def call_alphavantage(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("AlphaVantage key missing")
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
    logger.info("AlphaVantage: requesting overview for %s", symbol)
    data = _safe_get(url, params)
    time.sleep(RATE_LIMIT_SLEEP)
    return data

def call_fmp(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("FMP key missing")
        return None
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    params = {"apikey": api_key}
    logger.info("FMP: requesting profile for %s", symbol)
    data = _safe_get(url, params)
    time.sleep(RATE_LIMIT_SLEEP)
    if isinstance(data, list) and data:
        return data[0]
    return None

def call_finnhub(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("Finnhub key missing")
        return None
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {"symbol": symbol, "token": api_key}
    logger.info("Finnhub: requesting profile for %s", symbol)
    data = _safe_get(url, params)
    time.sleep(RATE_LIMIT_SLEEP)
    return data

def fetch_fundamental_with_fallback(symbol: str, secrets: Dict[str, str]) -> Optional[Dict]:
    """
    Try providers in order: FMP -> AlphaVantage -> Finnhub.
    Return first successful parsed dict or None.
    """
    # FMP
    fmp_key = secrets.get("FMP_KEY")
    if fmp_key:
        try:
            res = call_fmp(symbol, fmp_key)
            if res:
                logger.info("Fetched fundamentals for %s from FMP", symbol)
                return res
        except Exception as e:
            logger.warning("FMP failed for %s: %s", symbol, e)

    # AlphaVantage
    av_key = secrets.get("ALPHAVANTAGE_KEY")
    if av_key:
        try:
            res = call_alphavantage(symbol, av_key)
            if res:
                logger.info("Fetched fundamentals for %s from AlphaVantage", symbol)
                return res
        except Exception as e:
            logger.warning("AlphaVantage failed for %s: %s", symbol, e)

    # Finnhub
    fh_key = secrets.get("FINNHUB_API_KEY")
    if fh_key:
        try:
            res = call_finnhub(symbol, fh_key)
            if res:
                logger.info("Fetched fundamentals for %s from Finnhub", symbol)
                return res
        except Exception as e:
            logger.warning("Finnhub failed for %s: %s", symbol, e)

    logger.error("All fundamental providers failed for %s", symbol)
    return None
