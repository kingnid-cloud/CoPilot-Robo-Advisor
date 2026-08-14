from __future__ import annotations
import argparse
import json
import os
import logging
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import importlib
import requests
import pandas_datareader.data as pdr
from dateutil.relativedelta import relativedelta

logger = logging.getLogger("compare_backtest")
logging.basicConfig(level=logging.INFO)


def make_folds(start: str, end: str, train_years: int, test_months: int):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    folds = []
    current_train_start = s
    while True:
        test_start = current_train_start + relativedelta(years=train_years)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)
        if test_start >= e or test_start > test_end:
            break
        if test_end > e:
            test_end = e
        folds.append({"test_start": test_start.strftime("%Y-%m-%d"), "test_end": test_end.strftime("%Y-%m-%d")})
        current_train_start = current_train_start + relativedelta(months=test_months)
    return folds


def compute_returns_for_fold(tickers, test_start: str, test_end: str):
    """
    Multi-source, robust return computation.
    Tickerbot (v2 history) is tried first. Debug outputs written to outputs/debug/.
    Returns a float (arithmetic mean across tickers if tickers is a list).
    """
    max_attempts = 3
    end_plus1 = (pd.to_datetime(test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickerbot_key = os.environ.get("TICKERBOT_API_KEY", "").strip()
    tickerbot_base = os.environ.get("TICKERBOT_URL", "https://api.tickerbot.io").strip().rstrip("/")
    av_key = os.environ.get("ALPHAVANTAGE_KEY", "").strip()

    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    def compute_from_df(df):
        if df is None or df.empty:
            return None, 0
        # Prefer 'Close' (case-insensitive)
        if "Close" in df:
            ser = df["Close"].dropna()
        elif "close" in df:
            ser = df["close"].dropna()
        else:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                ser = pd.Series(dtype=float)
            else:
                ser = df[numeric_cols[-1]].dropna()
        if ser.empty:
            return None, 0
        entry = ser.iloc[0]; exit = ser.iloc[-1]
        return float((exit / entry) - 1.0), len(ser)

    def save_bytes(path: Path, b: bytes):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b[:2000])
        except Exception:
            logger.exception("Failed to write debug file %s", path)

    def try_tickerbot(ticker):
        if not tickerbot_base:
            logger.info("Tickerbot base URL not set; skipping tickerbot for %s", ticker)
            return None

        # v2/history: use asof + interval only (Tickerbot accepts asof, interval, ticker)
        url = f"{tickerbot_base}/v2/tickers/{ticker}/history"
        params = {"interval": "1d", "asof": test_end}
        headers = {}
        if tickerbot_key:
            headers["Authorization"] = f"Bearer {tickerbot_key}"
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            save_bytes(debug_dir / f"tickerbot_raw_{ticker}_{test_start}_{test_end}.json", resp.content)
            logger.info("Tickerbot checked %s status=%s", url, resp.status_code)
            if resp.status_code != 200:
                logger.info("Tickerbot returned status %s for %s", resp.status_code, url)
                return None
            try:
                j = resp.json()
            except Exception:
                logger.info("Tickerbot response not JSON for %s", url)
                return None

            # Heuristics: find the price-series in common shapes
            prices = None

            # 1) If top-level dict has typical container keys
            if isinstance(j, dict):
                for k in ("series", "data", "prices", "items", "results", "values"):
                    if k in j and (isinstance(j[k], list) or isinstance(j[k], dict)):
                        prices = j[k]
                        break

            # 2) If top-level is a dict and keys look like dates, convert mapping -> list
            if prices is None and isinstance(j, dict):
                date_keys = [k for k in j.keys() if isinstance(k, str) and (k.count("-") == 2 or k.isdigit())]
                if date_keys:
                    # j could map date -> scalar or date -> {close:..., open:...}
                    try:
                        prices = [{"date": k, **(j[k] if isinstance(j[k], dict) else {"close": j[k]})} for k in date_keys]
                    except Exception:
                        prices = None

            # 3) If top-level is already a list (but elements might be scalars)
            if prices is None and isinstance(j, list):
                # If list of dicts, use it directly; if scalars, attempt to find timestamps in j's sibling keys (unlikely)
                if j and isinstance(j[0], dict):
                    prices = j
                else:
                    # leave prices as None for now; we'll log and failback later
                    prices = None

            if prices is None:
                logger.info("Tickerbot v2/history returned JSON but no price list detected; saving for inspection (keys=%s)", list(j.keys())[:8] if isinstance(j, dict) else None)
                return None

            # If prices is a dict (some APIs return object-of-arrays), try converting to list-of-dicts
            if isinstance(prices, dict):
                # try convert mapping -> list
                try:
                    prices = [{"date": k, **(prices[k] if isinstance(prices[k], dict) else {"close": prices[k]})} for k in prices.keys()]
                except Exception:
                    # last resort: wrap dict as single element
                    prices = [prices]

            # Now attempt to build DataFrame; handle scalar-list error gracefully
            try:
                df = pd.DataFrame(prices)
            except ValueError as e:
                msg = str(e)
                logger.info("pd.DataFrame(prices) failed: %s", msg)
                # handle "If using all scalar values, you must pass an index"
                if "If using all scalar values" in msg and isinstance(prices, dict):
                    prices = [{"date": k, "close": v} for k, v in prices.items()]
                    df = pd.DataFrame(prices)
                else:
                    # try converting dict-of-scalars -> list of dicts if possible
                    try:
                        if isinstance(prices, dict):
                            prices = [{"date": k, "close": v} for k, v in prices.items()]
                            df = pd.DataFrame(prices)
                        else:
                            logger.info("Unable to convert prices to DataFrame; saving raw and skipping ticker")
                            return None
                    except Exception as exc2:
                        logger.info("Fallback DataFrame conversion failed: %s", exc2)
                        return None

            # Normalize date/index and 'Close' column
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                except Exception:
                    pass
            if "close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"close": "Close"})
            if "Close" in df.columns:
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

            # Filter to requested window (test_start..test_end) in case API returns extra points
            try:
                df = df[(df.index >= pd.to_datetime(test_start)) & (df.index <= pd.to_datetime(test_end))]
            except Exception:
                pass

            logger.info("Tickerbot for %s via %s returned shape=%s", ticker, url, getattr(df, "shape", None))
            return df
        except Exception as exc:
            logger.info("Tickerbot request failed for %s on %s: %s", ticker, url, exc)
            return None

    def try_ticker_history(ticker):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=test_start, end=end_plus1)
            logger.info("history() for %s returned shape=%s", ticker, None if hist is None else getattr(hist, "shape", None))
            if hist is None or getattr(hist, "shape", (0,))[0] == 0:
                # attempt to save raw Yahoo CSV for debugging
                try:
                    import calendar as _cal
                    period1 = int(_cal.timegm(pd.to_datetime(test_start).timetuple()))
                    period2 = int(_cal.timegm(pd.to_datetime(test_end).timetuple())) + 24 * 3600
                    raw_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
                    r = requests.get(raw_url, timeout=15)
                    save_bytes(debug_dir / f"yf_raw_{ticker}_{test_start}_{test_end}.txt", r.content)
                except Exception:***

