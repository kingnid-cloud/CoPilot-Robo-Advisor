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

        # Strict v2/history call only (asof + interval + limit). Avoid sending 'start' which causes 400.
        url = f"{tickerbot_base}/v2/tickers/{ticker}/history"
        params = {"interval": "1d", "asof": test_end, "limit": 10000}
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
            # Parse common shapes: list-of-dicts or dict with 'series'/'data' keys or date->value mapping
            prices = None
            if isinstance(j, dict):
                for k in ("series", "data", "prices", "items", "results"):
                    if k in j:
                        prices = j[k]
                        break
            if prices is None and isinstance(j, dict):
                date_keys = [k for k in j.keys() if isinstance(k, str) and k.count("-") == 2]
                if date_keys:
                    prices = [{"date": k, **(j[k] if isinstance(j[k], dict) else {"close": j[k]})} for k in date_keys]
            if prices is None and isinstance(j, list):
                prices = j
            if not prices:
                logger.info("Tickerbot v2/history returned no usable prices (keys=%s)", list(j.keys())[:6] if isinstance(j, dict) else None)
                return None
            df = pd.DataFrame(prices)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            if "close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"close": "Close"})
            if "Close" in df.columns:
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
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
                except Exception:
                    pass
            return hist
        except Exception as exc:
            logger.info("Ticker.history() error for %s: %s", ticker, exc)
            # try to capture raw Yahoo response
            try:
                import calendar as _cal
                period1 = int(_cal.timegm(pd.to_datetime(test_start).timetuple()))
                period2 = int(_cal.timegm(pd.to_datetime(test_end).timetuple())) + 24 * 3600
                raw_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
                r = requests.get(raw_url, timeout=15)
                save_bytes(debug_dir / f"yf_raw_{ticker}_{test_start}_{test_end}.txt", r.content)
            except Exception:
                pass
            return None

    def try_yf_download(ticker):
        try:
            try:
                df = yf.download(ticker, start=test_start, end=end_plus1, progress=False, threads=False)
            except TypeError:
                df = yf.download(ticker, start=test_start, end=end_plus1)
            logger.info("yf.download for %s returned shape=%s", ticker, None if df is None else getattr(df, "shape", None))
            if df is None or getattr(df, "shape", (0,))[0] == 0:
                # capture raw CSV
                try:
                    import calendar as _cal
                    period1 = int(_cal.timegm(pd.to_datetime(test_start).timetuple()))
                    period2 = int(_cal.timegm(pd.to_datetime(test_end).timetuple())) + 24 * 3600
                    raw_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
                    r = requests.get(raw_url, timeout=15)
                    save_bytes(debug_dir / f"yf_raw_{ticker}_{test_start}_{test_end}.txt", r.content)
                except Exception:
                    pass
            return df
        except Exception as exc:
            logger.info("yf.download error for %s: %s", ticker, exc)
            # attempt raw fetch for debug
            try:
                import calendar as _cal
                period1 = int(_cal.timegm(pd.to_datetime(test_start).timetuple()))
                period2 = int(_cal.timegm(pd.to_datetime(test_end).timetuple())) + 24 * 3600
                raw_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
                r = requests.get(raw_url, timeout=15)
                save_bytes(debug_dir / f"yf_raw_{ticker}_{test_start}_{test_end}.txt", r.content)
            except Exception:
                pass
            return None

    def try_pdr_stooq(ticker):
        for symbol in (ticker, f"{ticker}.US"):
            try:
                df = pdr.DataReader(symbol, "stooq", start=test_start, end=end_plus1)
                if df is not None and not df.empty:
                    df = df.sort_index()
                logger.info("pandas_datareader(stooq) for %s returned shape=%s", symbol, None if df is None else getattr(df, "shape", None))
                if df is None or getattr(df, "shape", (0,))[0] == 0:
                    # try to capture stooq raw page for debug
                    try:
                        d1 = pd.to_datetime(test_start).strftime("%Y%m%d")
                        d2 = pd.to_datetime(test_end).strftime("%Y%m%d")
                        stooq_url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"
                        r = requests.get(stooq_url, timeout=15)
                        save_bytes(debug_dir / f"stooq_raw_{symbol}_{test_start}_{test_end}.txt", r.content)
                    except Exception:
                        pass
                return df
            except Exception as exc:
                logger.info("pandas_datareader(stooq) failed for %s: %s", symbol, exc)
                try:
                    d1 = pd.to_datetime(test_start).strftime("%Y%m%d")
                    d2 = pd.to_datetime(test_end).strftime("%Y%m%d")
                    stooq_url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"
                    r = requests.get(stooq_url, timeout=15)
                    save_bytes(debug_dir / f"stooq_raw_{symbol}_{test_start}_{test_end}.txt", r.content)
                except Exception:
                    pass
        return None

    def try_alpha_vantage(ticker):
        if not av_key:
            logger.info("AlphaVantage key not set; skipping AlphaVantage for %s", ticker)
            return None
        url = "https://www.alphavantage.co/query"
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker, "outputsize": "full", "apikey": av_key}
        try:
            resp = requests.get(url, params=params, timeout=30)
            save_bytes(debug_dir / f"alphav_raw_{ticker}_{test_start}_{test_end}.txt", resp.content)
            resp.raise_for_status()
            j = resp.json()
            if "Time Series (Daily)" not in j:
                logger.info("AlphaVantage returned no time series for %s: keys=%s", ticker, list(j.keys())[:6])
                return None
            data = j["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # normalized 'Close'
            if "5. adjusted close" in df.columns:
                df = df.rename(columns={"5. adjusted close": "Close"})
                df["Close"] = df["Close"].astype(float)
            elif "4. close" in df.columns:
                df = df.rename(columns={"4. close": "Close"})
                df["Close"] = df["Close"].astype(float)
            else:
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    df = df.rename(columns={numeric_cols[-1]: "Close"})
                else:
                    return None
            df = df[(df.index >= pd.to_datetime(test_start)) & (df.index <= pd.to_datetime(test_end))]
            logger.info("AlphaVantage for %s returned shape=%s", ticker, getattr(df, "shape", None))
            return df
        except Exception as exc:
            logger.info("AlphaVantage request failed for %s: %s", ticker, exc)
            return None

    methods = [
        ("tickerbot", try_tickerbot),
        ("history", try_ticker_history),
        ("download", try_yf_download),
        ("stooq", try_pdr_stooq),
        ("alphavantage", try_alpha_vantage),
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
                rets = []
                for t in tickers:
                    best_ret = None
                    best_len = -1
                    for name, fn in methods:
                        df = None
                        try:
                            df = fn(t)
                        except Exception as exc:
                            logger.info("Method %s raised for %s (attempt %d): %s", name, t, attempt, exc)
                        r, n = compute_from_df(df)
                        if r is not None and n > best_len:
                            best_ret = r
                            best_len = n
                            logger.info("Method %s selected for %s: return=%0.6f (n=%d)", name, t, r, n)
                        else:
                            logger.info("Method %s produced no usable data (n=%d) for %s (attempt %d)", name, n, t, attempt)
                    if best_ret is not None:
                        rets.append(best_ret)
                    else:
                        logger.warning("No valid data found for ticker %s across methods (attempt %d)", t, attempt)
                if not rets:
                    raise RuntimeError("no valid ticker returns found across methods")
                return float(sum(rets) / len(rets))
            else:
                t = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
                best_ret = None
                best_len = -1
                for name, fn in methods:
                    df = None
                    try:
                        df = fn(t)
                    except Exception as exc:
                        logger.info("Method %s raised for %s (attempt %d): %s", name, t, attempt, exc)
                    r, n = compute_from_df(df)
                    if r is not None and n > best_len:
                        best_ret = r
                        best_len = n
                        logger.info("Method %s selected for %s: return=%0.6f (n=%d)", name, t, r, n)
                    else:
                        logger.info("Method %s produced no usable data (n=%d) for %s (attempt %d)", name, n, t, attempt)
                if best_ret is not None:
                    return best_ret
                else:
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
        # compute SPY return over same period using available sources (prefer yf.download for SPY)
        try:
            spy_df = None
            try:
                spy_df = yf.download("SPY", start=ts, end=(pd.to_datetime(te) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, threads=False)
            except TypeError:
                spy_df = yf.download("SPY", start=ts, end=(pd.to_datetime(te) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            if spy_df is None or spy_df.empty or "Close" not in spy_df:
                try:
                    spy_df = yf.Ticker("SPY").history(start=ts, end=(pd.to_datetime(te) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                except Exception:
                    spy_df = None
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
