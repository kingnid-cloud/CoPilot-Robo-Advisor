from __future__ import annotations
import argparse
import json
from pathlib import Path
import logging
import math
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
    Compute average return across tickers between test_start and test_end using Close prices.
    Returns a float (e.g., 0.02 for +2%).
    """
    try:
        df = yf.download(tickers, start=test_start, end=(pd.to_datetime(test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, threads=False)
        if df is None or df.empty:
            return 0.0
        # Ensure 'Close' column selection works for single vs multiple tickers
        if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
            close = df["Close"].dropna()
            # get first and last valid date for each ticker
            rets = []
            for t in tickers:
                if t not in close.columns:
                    continue
                ser = close[t].dropna()
                if ser.empty:
                    continue
                entry = ser.iloc[0]
                exit = ser.iloc[-1]
                rets.append((exit / entry) - 1.0)
            if not rets:
                return 0.0
            return float(sum(rets) / len(rets))
        else:
            # single ticker
            # yfinance returns a DataFrame with Close column if multiple columns, or a Series if single ticker
            if "Close" in df:
                ser = df["Close"].dropna()
            else:
                # df may already be the Close series
                ser = df.dropna()
            if ser.empty:
                return 0.0
            entry = ser.iloc[0]
            exit = ser.iloc[-1]
            return float((exit / entry) - 1.0)
    except Exception as exc:
        logger.warning("Failed to compute returns for %s - %s to %s: %s", tickers, test_start, test_end, exc)
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
                    return fn(tickers=tickers, start=start, end=end, train_years=train_years, test_months=test_months, fred_key=fred_key)
                except TypeError:
                    # try positional
                    try:
                        return fn(tickers, start, end, train_years, test_months)
                    except Exception:
                        logger.warning("Found function '%s' but calling it failed; falling back.", name)
                        break
        # if a class WalkForwardBacktester exists, try some common instance methods
        if hasattr(wf_mod, "WalkForwardBacktester"):
            logger.info("Found WalkForwardBacktester in module; attempting to use it")
            cls = getattr(wf_mod, "WalkForwardBacktester")
            try:
                instance = cls()
                # if it has a method that runs the backtest, try those names
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
                # as a last resort, check if it exposes simple_report and we can call that with a placeholder
                if hasattr(instance, "simple_report"):
                    logger.info("Using simple_report fallback on WalkForwardBacktester")
                    # construct a placeholder portfolio_df with one row per test window: combined returns across tickers
                    folds = make_folds(start, end, train_years, test_months)
                    rows = []
                    for f in folds:
                        r = compute_returns_for_fold(tickers, f["test_start"], f["test_end"])
                        rows.append({"combined": r})
                    portfolio_df = pd.DataFrame(rows)
                    rep = instance.simple_report(portfolio_df)
                    # map report rows back into fold-like dicts
                    out_folds = []
                    for i, f in enumerate(folds):
                        strat_r = float(rep.iloc[i].get("combined", 0.0)) if i < len(rep) else 0.0
                        out_folds.append({"test_start": f["test_start"], "test_end": f["test_end"], "strategy_return": strat_r})
                    return {"folds": out_folds}
            except Exception as exc:
                logger.warning("WalkForwardBacktester use failed: %s", exc)
    except ModuleNotFoundError:
        logger.info("src.backtest.walkforward not found; using builtin simple WF")

    # Fallback: build folds and compute returns per fold from tickers directly
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

    # collect cumulative series per fold using fold strategy_return and SPY test returns
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
            # treat as no deployment -> cash yield ~0
            strat_r = 0.0
        # compute SPY return over same period
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

    # save summary and plot
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
