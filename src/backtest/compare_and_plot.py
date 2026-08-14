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
    Robustly compute average return across tickers between test_start and test_end.
    Tries in order:
      1) yf.Ticker(...).history()
      2) yf.download(...)
      3) pandas_datareader (stooq) as a last resort
    Returns a float return (e.g., 0.02 for +2%). Logs diagnostic details for each attempt.
    """
    max_attempts = 3
    end_plus1 = (pd.to_datetime(test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def try_ticker_history(ticker):
        tk = yf.Ticker(ticker)
        hist = tk.history(start=test_start, end=end_plus1)
        logger.info("history() for %s returned shape=%s", ticker, None if hist is None else getattr(hist, "shape", None))
        return hist

    def try_yf_download(ticker):
        # yf.download sometimes returns data when Ticker.history doesn't
        df = yf.download(ticker, start=test_start, end=end_plus1, progress=False, threads=False)
        logger.info("yf.download for %s returned shape=%s", ticker, None if df is None else getattr(df, "shape", None))
        return df

    def try_pdr_stooq(ticker):
        try:
            df = pdr.DataReader(ticker, "stooq", start=test_start, end=end_plus1)
            # stooq returns index in descending order; sort by index
            if df is not None and not df.empty:
                df = df.sort_index()
            logger.info("pandas_datareader(stooq) for %s returned shape=%s", ticker, None if df is None else getattr(df, "shape", None))
            return df
        except Exception as exc:
            logger.info("pandas_datareader(stooq) failed for %s: %s", ticker, exc)
            return None

    # helper to compute return from a dataframe that must include 'Close' or an equivalent column
    def compute_from_df(df, ticker):
        if df is None or df.empty:
            return None
        # select Close-like column
        if "Close" in df:
            ser = df["Close"].dropna()
        elif "close" in df:
            ser = df["close"].dropna()
        elif "Close**" in df:  # just in case of weird names
            ser = df.iloc[:, -1].dropna()
        else:
            # try last column
            ser = df.iloc[:, df.shape[1] - 1].dropna() if df.shape[1] > 0 else pd.Series(dtype=float)
        if ser.empty:
            return None
        entry = ser.iloc[0]
        exit = ser.iloc[-1]
        return float((exit / entry) - 1.0)

    # When tickers is a list, compute per-ticker and average; otherwise single ticker flow
    for attempt in range(1, max_attempts + 1):
        try:
            if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
                rets = []
                for t in tickers:
                    # try methods in order
                    for method, fn in (("history", try_ticker_history), ("download", try_yf_download), ("stooq", try_pdr_stooq)):
                        try:
                            df = fn(t)
                        except Exception as exc:
                            logger.warning("Method %s raised for %s (attempt %d): %s", method, t, attempt, exc)
                            df = None
                        r = compute_from_df(df, t)
                        if r is not None:
                            logger.info("Method %s succeeded for %s: return=%0.6f", method, t, r)
                            rets.append(r)
                            break
                        else:
                            logger.info("Method %s returned no usable data for %s (attempt %d)", method, t, attempt)
                    # continue to next ticker
                if not rets:
                    raise RuntimeError("no valid ticker returns found across methods")
                return float(sum(rets) / len(rets))
            else:
                t = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
                for method, fn in (("history", try_ticker_history), ("download", try_yf_download), ("stooq", try_pdr_stooq)):
                    try:
                        df = fn(t)
                    except Exception as exc:
                        logger.warning("Method %s raised for %s (attempt %d): %s", method, t, attempt, exc)
                        df = None
                    r = compute_from_df(df, t)
                    if r is not None:
                        logger.info("Method %s succeeded for %s: return=%0.6f", method, t, r)
                        return r
                    else:
                        logger.info("Method %s returned no usable data for %s (attempt %d)", method, t, attempt)
                raise RuntimeError(f"No data for ticker {t}")
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
    try:
        wf_mod = importlib.import_module("src.backtest.walkforward")
        for name in ("run_walk_forward", "run_walkforward", "run", "execute", "walk_forward"):
            if hasattr(wf_mod, name):
                fn = getattr(wf_mod, name)
                logger.info("Using walkforward function '%s' from src.backtest.walkforward", name)
                try:
                    return fn(tickers=tickers, start=start, end=end, train_years=train_years, test_months=test_months, fred_key=fred_key)
                except TypeError:
                    try:
                        return fn(tickers, start, end, train_years, test_months)
                    except Exception:
                        logger.warning("Found function '%s' but calling it failed; falling back.", name)
                        break
        if hasattr(wf_mod, "WalkForwardBacktester"):
            logger.info("Found WalkForwardBacktester in module; attempting to use it")
            cls = getattr(wf_mod, "WalkForwardBacktester")
            try:
                instance = cls()
                for mname in ("run", "execute", "backtest", "run_walk_forward", "walk_forward"):
                    if hasattr(instance, mname):
                        m = getattr(instance, mname)
                        logger.info("Calling WalkForwardBacktester.%s()", mname)
                        try:
                            return m(tickers=tickers, start=start, end=end, train_years=train_years, test_months=test_months)
                        except TypeError:
                            try:
                                return m(tickers, start, end, train_years, test_months)
                            except Exception:
                                logger.warning("Call to WalkForwardBacktester.%s failed; continuing", mname)
                if hasattr(instance, "simple_report"):
                    logger.info("Using simple_report fallback on WalkForwardBacktester")
                    folds = make_folds(start, end, train_years, test_months)
                    rows = []
                    for f in folds:
                        r = compute_returns_for_fold(tickers, f["test_start"], f["test_end"])
                        rows.append({"combined": r})
                    portfolio_df = pd.DataFrame(rows)
                    rep = instance.simple_report(portfolio_df)
                    out_folds = []
                    for i, f in enumerate(folds):
                        strat_r = float(rep.iloc[i].get("combined", 0.0)) if i < len(rep) else 0.0
                        out_folds.append({"test_start": f["test_start"], "test_end": f["test_end"], "strategy_return": strat_r})
                    return {"folds": out_folds}
            except Exception as exc:
                logger.warning("WalkForwardBacktester use failed: %s", exc)
    except ModuleNotFoundError:
        logger.info("src.backtest.walkforward not found; using builtin simple WF")

    folds = make_folds(start, end, train_years, test_months)
    out_folds = []
    for f in folds:
        strat_r = compute_returns_for_fold(tickers, f["test_start"], f["test_end"])
        out_folds.append({"test_start": f["test_start"], "test_end": f["test_end"], "strategy_return": strat_r})
    return {"folds": out_folds}


def compare_and_plot(tickers, start, end, train_years, test_months, fred_key=None, out_prefix=Path("outputs/backtest")):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = get_summary_with_adapters(tickers, start, end, train_years, test_months, fred_key=fred_key)
    folds = summary.get("folds", [])

    dates = []
    strat_vals = []
    spy_vals = []
    cum_strat = 1.0
    cum_spy = 1.0

    for f in folds:
        ts = f.get("test_start")
        te = f.get("test_end")
        strat_r = f.get("strategy_return")
        if strat_r is None:
            strat_r = 0.0
        try:
            spy_df = yf.download("SPY", start=ts, end=(pd.to_datetime(te) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, threads=False)
            if spy_df is None or spy_df.empty:
                spy_r = 0.0
            else:
                close = spy_df["Close"].dropna()
                if close.empty:
                    spy_r = 0.0
                else:
                    spy_entry = close.iloc[0]
                    spy_exit = close.iloc[-1]
                    spy_r = (spy_exit / spy_entry) - 1.0
        except Exception:
            spy_r = 0.0

        cum_strat *= (1.0 + strat_r)
        cum_spy *= (1.0 + spy_r)
        dates.append(pd.to_datetime(f.get("test_end")))
        strat_vals.append(cum_strat)
        spy_vals.append(cum_spy)

    df = pd.DataFrame({"date": dates, "strategy_cum": strat_vals, "spy_cum": spy_vals}).set_index("date")

    out_json = out_prefix.with_suffix(".json")
    out_png = out_prefix.with_suffix(".png")
    out_csv = out_prefix.with_suffix(".csv")

    try:
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write JSON summary")

    try:
        df.to_csv(out_csv)
    except Exception:
        logger.exception("Failed to write CSV")

    plt.figure(figsize=(10, 6))
    if not df.empty:
        plt.plot(df.index, df["strategy_cum"], label="Strategy (WF)")
        plt.plot(df.index, df["spy_cum"], label="SPY")
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.legend()
    plt.title(f"Walk-forward Strategy vs SPY ({start} to {end})")
    plt.ylabel("Cumulative Return (Growth of $1)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    logger.info("Saved plot to %s", out_png)
    return {"summary": summary, "cumulative_csv": str(out_csv), "plot_png": str(out_png)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", help="Tickers list (space separated) or leave empty to default to SPY", default=["SPY"])
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--train-years", type=int, default=5)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--fred-key", default=None)
    p.add_argument("--out", default="outputs/backtest")
    args = p.parse_args()
    res = compare_and_plot(args.tickers, args.start, args.end, args.train_years, args.test_months, fred_key=args.fred_key, out_prefix=Path(args.out))
    print(res)
