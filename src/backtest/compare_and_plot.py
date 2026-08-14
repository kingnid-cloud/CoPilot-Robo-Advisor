from __future__ import annotations
import argparse
import json
from pathlib import Path
import logging
import math
import time
from datetime import timedelta, datetime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import importlib
from dateutil.relativedelta import relativedelta

logger = logging.getLogger("compare_backtest")
logging.basicConfig(level=logging.INFO)


def make_folds(start: str, end: str, train_years: int, test_months: int):
    """
    Create a sequence of fold dicts with test_start/test_end between start and end.
    Each fold's test period begins after the training window.
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    folds = []

    # initial train end = s + train_years
    current_train_start = s
    while True:
        test_start = current_train_start + relativedelta(years=train_years)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)
        if test_start >= e or test_start > test_end:
            break
        # cap test_end to overall end
        if test_end > e:
            test_end = e
        folds.append({"test_start": test_start.strftime("%Y-%m-%d"), "test_end": test_end.strftime("%Y-%m-%d")})
        # roll forward: move training window forward by the length of the test window
        current_train_start = current_train_start + relativedelta(months=test_months)
    return folds


def compute_returns_for_fold(tickers, test_start: str, test_end: str):
    """
    Robustly compute average return across tickers between test_start and test_end.
    Uses yf.Ticker(...).history with retries and logs diagnostic info.
    Returns a float return (e.g., 0.02 for +2%).
    """
    max_attempts = 3
    end_plus1 = (pd.to_datetime(test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def fetch_single(ticker):
        tk = yf.Ticker(ticker)
        hist = tk.history(start=test_start, end=end_plus1, progress=False)
        return hist

    for attempt in range(1, max_attempts + 1):
        try:
            if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
                rets = []
                for t in tickers:
                    try:
                        hist = fetch_single(t)
                        if hist is None or hist.empty or "Close" not in hist:
                            logger.warning("Ticker %s: no data (attempt %d) for %s-%s; hist shape=%s", t, attempt, test_start, test_end, None if hist is None else getattr(hist, "shape", None))
                            continue
                        close = hist["Close"].dropna()
                        if close.empty:
                            logger.warning("Ticker %s: Close series empty (attempt %d) for %s-%s", t, attempt, test_start, test_end)
                            continue
                        entry = close.iloc[0]; exit = close.iloc[-1]
                        rets.append((exit / entry) - 1.0)
                    except Exception as exc:
                        logger.warning("Ticker %s history exception (attempt %d): %s", t, attempt, exc)
                if not rets:
                    raise RuntimeError("no valid ticker returns found")
                return float(sum(rets) / len(rets))
            else:
                t = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
                hist = fetch_single(t)
                if hist is None or hist.empty or "Close" not in hist:
                    raise RuntimeError(f"No data for ticker {t}")
                close = hist["Close"].dropna()
                if close.empty:
                    raise RuntimeError(f"No close prices for {t}")
                entry = close.iloc[0]; exit = close.iloc[-1]
                return float((exit / entry) - 1.0)
        except Exception as exc:
            logger.warning("Attempt %d: Failed to compute returns for %s (%s-%s): %s", attempt, tickers, test_start, test_end, exc)
            if attempt < max_attempts:
                sleep_s = 2 ** attempt
                logger.info("Sleeping %d seconds before retry...", sleep_s)
                time.sleep(sleep_s)
            else:
                logger.error("All attempts failed for %s (%s-%s); returning 0.0", tickers, test_start, test_end)
                return 0.0


def get_summary_with_adapters(tickers, start, end, train_years, test_months, fred_key=None):
    """
    Try to call an existing walkforward runner if available in src.backtest.walkforward.
    If not present or not usable, create a simple fold-based summary using yfinance.
    """
    # attempt to import and use existing run function or class
    try:
        wf_mod = importlib.import_module("src.backtest.walkforward")
        # prefer a direct run function if present
        for name in ("run_walk_forward", "run_walkforward", "run", "execute", "walk_forward"):
            if hasattr(wf_mod, name):
                fn = getattr(wf_mod, name)
                logger.info("Using walkforward function '%s' from src.backtest.walkforward", name)
                # try to call it; many project-specific run functions have signature similar to:
                # run(tickers, start, end, train_years, test_months, **kwargs)
                try:
                    return fn(tickers=tickers, start=start,**

