# src/data/connectors.py
"""
API connector helpers with Tickerbot integration and fallback strategy.

Order of preference for fundamentals/enrichment:
  1. Tickerbot (rich metadata, sector/industry, fundamentals)
  2. FinancialModelingPrep (FMP)
  3. AlphaVantage
  4. Finnhub

Behavior:
- Each provider call returns a parsed dict or None.
- fetch_fundamental_with_fallback tries providers in order and returns the first successful result.
- Functions are defensive: they log and return None on errors.
- Caller should cache results to disk to avoid repeated calls in CI.
"""

from typing import Optional, Dict, Any
import time
import logging
import requests
from pathlib import Path
import json

logger = logging.getLogger("connectors")
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 15.0
RATE_LIMIT_SLEEP = 1.0  # base sleep between calls to avoid bursts
CACHE_DIR = Path("cache")  # connectors may use DataCollector's cache; keep consistent
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _safe_get_json(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as he:
        logger.warning("HTTP error for %s: %s", url, he)
    except requests.exceptions.RequestException as re:
        logger.warning("Network error for %s: %s", url, re)
    except ValueError as ve:
        logger.warning("Invalid JSON from %s: %s", url, ve)
    return None

def call_tickerbot(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("Tickerbot key missing")
        return None

    url = "https://api.tickerbot.io/v1/enrich"
    params = {"symbol": symbol}
    headers = {"Authorization": f"Bearer {api_key}"}
    logger.info("Tickerbot: requesting enrichment for %s", symbol)
    data = _safe_get_json(url, params=params, headers=headers)
    time.sleep(RATE_LIMIT_SLEEP)
    if not data:
        return None

    out = {}
    try:
        out["provider"] = "tickerbot"
        out["ticker"] = data.get("ticker") or symbol
        out["name"] = data.get("name") or data.get("companyName")
        out["sector"] = data.get("sector") or data.get("industrySector")
        out["industry"] = data.get("industry") or data.get("industryGroup")
        fundamentals = data.get("fundamentals") or data.get("metrics") or {}
        out["fundamentals"] = fundamentals
        out["ROIC"] = fundamentals.get("roic") or fundamentals.get("ROIC")
        out["PEG"] = fundamentals.get("peg") or fundamentals.get("PEG")
        out["GrossMargin"] = fundamentals.get("grossMargin") or fundamentals.get("gross_margin")
    except Exception as e:
        logger.warning("Tickerbot parse error for %s: %s", symbol, e)
        return data

    return out

def call_fmp(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("FMP key missing")
        return None
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    params = {"apikey": api_key}
    logger.info("FMP: requesting profile for %s", symbol)
    data = _safe_get_json(url, params)
    time.sleep(RATE_LIMIT_SLEEP)
    if isinstance(data, list) and data:
        return data[0]
    return None

def call_alphavantage(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("AlphaVantage key missing")
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
    logger.info("AlphaVantage: requesting overview for %s", symbol)
    data = _safe_get_json(url, params)
    time.sleep(RATE_LIMIT_SLEEP)
    return data

def call_finnhub(symbol: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        logger.debug("Finnhub key missing")
        return None
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {"symbol": symbol, "token": api_key}
    logger.info("Finnhub: requesting profile for %s", symbol)
    data = _safe_get_json(url, params)
    time.sleep(RATE_LIMIT_SLEEP)
    return data

def _cache_result(symbol: str, provider: str, payload: Dict) -> None:
    try:
        p = CACHE_DIR / f"{symbol}_{provider}.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        logger.exception("Failed to write connector cache for %s/%s", symbol, provider)

def _load_cached(symbol: str, provider: str) -> Optional[Dict]:
    p = CACHE_DIR / f"{symbol}_{provider}.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to read connector cache for %s/%s", symbol, provider)
        return None

def fetch_fundamental_with_fallback(symbol: str, secrets: Dict[str, str], use_cache: bool = True) -> Optional[Dict]:
    tb_key = secrets.get("TICKERBOT_API_KEY")
    if tb_key:
        if use_cache:
            cached = _load_cached(symbol, "tickerbot")
            if cached:
                logger.info("Using cached Tickerbot for %s", symbol)
                return cached
        try:
            res = call_tickerbot(symbol, tb_key)
            if res:
                _cache_result(symbol, "tickerbot", res)
                logger.info("Fetched fundamentals for %s from Tickerbot", symbol)
                return res
        except Exception as e:
            logger.warning("Tickerbot failed for %s: %s", symbol, e)

    fmp_key = secrets.get("FMP_KEY")
    if fmp_key:
        if use_cache:
            cached = _load_cached(symbol, "fmp")
            if cached:
                logger.info("Using cached FMP for %s", symbol)
                return cached
        try:
            res = call_fmp(symbol, fmp_key)
            if res:
                _cache_result(symbol, "fmp", res)
                logger.info("Fetched fundamentals for %s from FMP", symbol)
                return res
        except Exception as e:
            logger.warning("FMP failed for %s: %s", symbol, e)

    av_key = secrets.get("ALPHAVANTAGE_KEY")
    if av_key:
        if use_cache:
            cached = _load_cached(symbol, "alphavantage")
            if cached:
                logger.info("Using cached AlphaVantage for %s", symbol)
                return cached
        try:
            res = call_alphavantage(symbol, av_key)
            if res:
                _cache_result(symbol, "alphavantage", res)
                logger.info("Fetched fundamentals for %s from AlphaVantage", symbol)
                return res
        except Exception as e:
            logger.warning("AlphaVantage failed for %s: %s", symbol, e)

    fh_key = secrets.get("FINNHUB_API_KEY")
    if fh_key:
        if use_cache:
            cached = _load_cached(symbol, "finnhub")
            if cached:
                logger.info("Using cached Finnhub for %s", symbol)
                return cached
        try:
            res = call_finnhub(symbol, fh_key)
            if res:
                _cache_result(symbol, "finnhub", res)
                logger.info("Fetched fundamentals for %s from Finnhub", symbol)
                return res
        except Exception as e:
            logger.warning("Finnhub failed for %s: %s", symbol, e)

    logger.error("All fundamental providers failed for %s", symbol)
    return None
