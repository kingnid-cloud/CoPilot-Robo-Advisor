from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime

import importlib
import concurrent.futures

import pandas as pd
import yfinance as yf
import requests
from dateutil.relativedelta import relativedelta

logger = logging.getLogger("compare_backtest")
logging.basicConfig(level=logging.INFO)


# ---------- fold construction ----------

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
        folds.append(
            {
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            }
        )
        current_train_start = current_train_start + relativedelta(months=test_months)
    return folds


# ---------- helpers ----------

def _save_debug_bytes(path: Path, b: bytes):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b[:2000])
    except Exception:
        logger.exception("Failed to write debug file %s", path)


def _compute_from_df(df: pd.DataFrame | None):
    """
    Normalize a price DataFrame to a 'Close' column and compute simple return.
    """
    if df is None or df.empty:
        return None, 0

    if "Close" not in df and "price" in df:
        df = df.rename(columns={"price": "Close"})
    if "Close" not in df and "close" in df:
        df = df.rename(columns={"close": "Close"})
    if "Close" not in df:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return None, 0
        df = df.rename(columns={numeric_cols[-1]: "Close"})

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    closes = df["Close"].dropna()
    closes = closes[closes > 0]
    if closes.empty or len(closes) < 2:
        return None, 0

    entry = closes.iloc[0]
    exit_ = closes.iloc[-1]
    med = closes.median()
    if entry < 1e-6 or entry < med * 1e-4:
        return None, 0

    ret = float((exit_ / entry) - 1.0)
    return ret, len(closes)


def _parse_prices_from_json(j: object):
    """
    Return a list-of-dicts (rows) or None if not found.

    Handles:
      - dict with keys 'series'|'data'|'prices' -> list
      - dict mapping date->scalar or date->{close:...}
      - list of dicts
    """
    if j is None:
        return None

    if isinstance(j, list):
        if j and isinstance(j[0], dict):
            return j
        return None

    if isinstance(j, dict):
        for k in ("series", "data", "prices", "items", "results", "values"):
            if k in j and isinstance(j[k], (list, dict)):
                return j[k]
        date_keys = [
            k for k in j.keys()
            if isinstance(k, str) and (k.count("-") == 2 or k.isdigit())
        ]
        if date_keys:
            try:
                return [
                    {
                        "date": k,
                        **(j[k] if isinstance(j[k], dict) else {"close": j[k]}),
                    }
                    for k in date_keys
                ]
            except Exception:
                pass
    return None


# ---------- data sources ----------

def _tickerbot_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
    cache_dir = Path("outputs/cache")

    # 1. Try cache first
    cached = load_cached_series(ticker, start, end, cache_dir)
    if cached is not None:
        return _parse_series_json(ticker, cached)

    # 2. Otherwise call Tickerbot
    base = os.environ.get("TICKERBOT_URL", "").rstrip("/")
    key = os.environ.get("TICKERBOT_API_KEY", "")
    if not base:
        return None

    headers = {"Authorization": f"Bearer {key}"} if key else {}

    url = f"{base}/v2/series"
    params = {
        "tickers": ticker,
        "columns": "price",
        "interval": "1d",
        "from": start,
        "to": end,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        raw_json = resp.json()

        # Save raw JSON for debugging
        _save_debug_bytes(debug_dir / f"series_raw_{ticker}_{start}_{end}.json", resp.content)

        # If rate-limited, return None (snapshot fallback will handle it)
        if "error" in raw_json:
            return None

        # Save to cache
        save_cached_series(ticker, start, end, cache_dir, raw_json)

        # Parse
        return _parse_series_json(ticker, raw_json)

    except Exception as exc:
        logger.info("Tickerbot /v2/series parse failed for %s: %s", ticker, exc)
        return None

def _parse_series_json(ticker: str, j: dict) -> pd.DataFrame | None:
    if "data" not in j or ticker not in j["data"]:
        return None

    rows = j["data"][ticker]
    parsed = [{"date": r[0], "Close": r[1]} for r in rows]

    df = pd.DataFrame(parsed)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    return df



def _tickerbot_snapshot_series(
    ticker: str,
    start: str,
    end: str,
    debug_dir: Path,
) -> pd.DataFrame | None:
    """
    Fallback: build a daily series by calling /v2/tickers/{ticker}?asof=YYYY-MM-DD.
    """
    base = os.environ.get("TICKERBOT_URL", "https://api.tickerbot.io").strip().rstrip("/")
    key = os.environ.get("TICKERBOT_API_KEY", "").strip()
    if not base:
        logger.info("Tickerbot base URL not set; skipping snapshot fallback for %s", ticker)
        return None

    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    start_dt = pd.to_datetime(start).normalize()
    end_dt = pd.to_datetime(end).normalize()
    days = (end_dt - start_dt).days + 1
    if days <= 0:
        return None

    max_days = 4000
    if days > max_days:
        logger.info(
            "Requested range %d days too large for snapshot fallback; skipping %s",
            days,
            ticker,
        )
        return None

    rows = []
    url_tick = f"{base}/v2/tickers/{ticker}"
    for d in pd.date_range(start_dt, end_dt, freq="D"):
        dstr = d.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                url_tick,
                params={"asof": dstr},
                headers=headers,
                timeout=10,
            )
            if len(rows) < 3:
                _save_debug_bytes(
                    debug_dir / f"tickerbot_raw_snapshot_sample_{ticker}_{dstr}.json",
                    resp.content,
                )
            if resp.status_code != 200:
                continue
            try:
                j = resp.json()
            except Exception:
                continue
            if isinstance(j, dict) and "data" in j and isinstance(j["data"], dict):
                val = None
                if "price" in j["data"]:
                    val = j["data"]["price"]
                elif "Close" in j["data"]:
                    val = j["data"]["Close"]
                elif "close" in j["data"]:
                    val = j["data"]["close"]
                if val is not None:
                    rows.append({"date": dstr, "Close": val})
            time.sleep(0.05)
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    try:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    except Exception:
        pass
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    return df


def _fmp_series(
    ticker: str,
    start: str,
    end: str,
    debug_dir: Path,
) -> pd.DataFrame | None:
    """
    Fallback: FinancialModelingPrep historical-price-full.
    """
    key = os.environ.get("FMP_KEY", "").strip()
    if not key:
        logger.info("FMP_KEY not set; skipping FMP for %s", ticker)
        return None

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {"from": start, "to": end, "apikey": key}
    try:
        resp = requests.get(url, params=params, timeout=30)
        _save_debug_bytes(
            debug_dir / f"fmp_raw_{ticker}_{start}_{end}.json",
            resp.content,
        )
        if resp.status_code != 200:
            logger.info("FMP returned status %s for %s", resp.status_code, url)
            return None
        j = resp.json()
        if "historical" not in j or not isinstance(j["historical"], list):
            logger.info("FMP returned no historical data for %s", ticker)
            return None
        rows = j["historical"]
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        if "close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"close": "Close"})
        if "Close" in df.columns:
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df
    except Exception as exc:
        logger.info("FMP request failed for %s: %s", ticker, exc)
        return None


def _yf_download_series(
    ticker: str,
    start: str,
    end: str,
    debug_dir: Path,
) -> pd.DataFrame | None:
    """
    Tertiary fallback: yfinance daily prices.
    """
    end_plus1 = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end_plus1,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except TypeError:
            df = yf.download(
                ticker,
                start=start,
                end=end_plus1,
                auto_adjust=True,
            )
        logger.info(
            "yf.download for %s returned shape=%s",
            ticker,
            None if df is None else getattr(df, "shape", None),
        )
        if df is None or df.empty:
            return None
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        if "Close" not in df.columns:
            return None
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df
    except Exception as exc:
        logger.info("yf.download error for %s: %s", ticker, exc)
        return None


# ---------- caching + unified fetch ----------

def _cache_path(ticker: str, start: str, end: str) -> Path:
    cache_dir = Path("outputs/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}_{start}_{end}.parquet"


def _load_cache(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    path = _cache_path(ticker, start, end)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            return df
        except Exception:
            return None
    return None


def _save_cache(ticker: str, start: str, end: str, df: pd.DataFrame):
    path = _cache_path(ticker, start, end)
    try:
        df.to_parquet(path)
    except Exception:
        pass


def get_price_series(
    ticker: str,
    start: str,
    end: str,
    source_mode: str,
    debug_dir: Path,
) -> pd.DataFrame | None:
    """
    Unified fetch with caching and source selection.

    source_mode:
      - 'auto'       : series -> snapshot -> FMP -> yfinance
      - 'tickerbot'  : series only
      - 'snapshot'   : snapshot only
      - 'fmp'        : FMP only
      - 'yf'         : yfinance only
    """
    cached = _load_cache(ticker, start, end)
    if cached is not None and not cached.empty:
        return cached

    methods_auto = [
        ("tickerbot_series", _tickerbot_series),
        ("tickerbot_snapshot", _tickerbot_snapshot_series),
        ("fmp", _fmp_series),
        ("yf_download", _yf_download_series),
    ]

    if source_mode == "tickerbot":
        methods = [("tickerbot_series", _tickerbot_series)]
    elif source_mode == "snapshot":
        methods = [("tickerbot_snapshot", _tickerbot_snapshot_series)]
    elif source_mode == "fmp":
        methods = [("fmp", _fmp_series)]
    elif source_mode == "yf":
        methods = [("yf_download", _yf_download_series)]
    else:
        methods = methods_auto

    df_best = None
    best_len = -1

    for name, fn in methods:
        try:
            df = fn(ticker, start, end, debug_dir)
        except Exception as exc:
            logger.info("Source %s raised for %s: %s", name, ticker, exc)
            df = None
        r, n = _compute_from_df(df)
        if r is not None and n > best_len:
            df_best = df
            best_len = n
            logger.info("Source %s selected for %s (n=%d, ret=%0.6f)", name, ticker, n, r)
        else:
            logger.info("Source %s produced no usable data (n=%d) for %s", name, n, ticker)

    if df_best is not None and not df_best.empty:
        _save_cache(ticker, start, end, df_best)
        return df_best

    return None


# ---------- fold return computation ----------

def compute_returns_for_fold(
    tickers,
    test_start: str,
    test_end: str,
    source_mode: str,
    min_days: int = 30,
) -> dict:
    """
    Compute strategy return and data quality for one fold.
    """
    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
        rets = []
        days_list = []
        for t in tickers:
            df = get_price_series(t, test_start, test_end, source_mode, debug_dir)
            r, n = _compute_from_df(df)
            if r is not None and n >= min_days:
                rets.append(r)
                days_list.append(n)
        if not rets:
            return {"return": 0.0, "days": 0, "missing": 0, "source": "none"}
        avg_ret = float(sum(rets) / len(rets))
        avg_days = int(sum(days_list) / len(days_list))
        expected_days = (pd.to_datetime(test_end) - pd.to_datetime(test_start)).days + 1
        missing = max(0, expected_days - avg_days)
        return {
            "return": avg_ret,
            "days": avg_days,
            "missing": missing,
            "source": source_mode,
        }
    else:
        t = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
        df = get_price_series(t, test_start, test_end, source_mode, debug_dir)
        r, n = _compute_from_df(df)
        if r is None or n < min_days:
            return {"return": 0.0, "days": n or 0, "missing": 0, "source": source_mode}
        expected_days = (pd.to_datetime(test_end) - pd.to_datetime(test_start)).days + 1
        missing = max(0, expected_days - n)
        return {
            "return": r,
            "days": n,
            "missing": missing,
            "source": source_mode,
        }


# ---------- walkforward adapter ----------

def get_summary_with_adapters(
    tickers,
    start: str,
    end: str,
    train_years: int,
    test_months: int,
    fred_key: str | None = None,
    source_mode: str = "auto",
    min_days: int = 30,
    parallel_workers: int = 4,
):
    """
    Adapter around src.backtest.walkforward if present; otherwise use builtin simple WF.
    """
    try:
        wf_mod = importlib.import_module("src.backtest.walkforward")
        for name in ("run_walk_forward", "run_walkforward", "run", "execute", "walk_forward"):
            if hasattr(wf_mod, name):
                fn = getattr(wf_mod, name)
                logger.info(
                    "Using walkforward function '%s' from src.backtest.walkforward",
                    name,
                )
                try:
                    return fn(
                        tickers=tickers,
                        start=start,
                        end=end,
                        train_years=train_years,
                        test_months=test_months,
                        fred_key=fred_key,
                    )
                except TypeError:
                    try:
                        return fn(tickers, start, end, train_years, test_months)
                    except Exception:
                        logger.warning(
                            "Found function '%s' but calling it failed; falling back.",
                            name,
                        )
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
                            return m(
                                tickers=tickers,
                                start=start,
                                end=end,
                                train_years=train_years,
                                test_months=test_months,
                            )
                        except TypeError:
                            try:
                                return m(tickers, start, end, train_years, test_months)
                            except Exception:
                                logger.warning(
                                    "Call to WalkForwardBacktester.%s failed; continuing",
                                    mname,
                                )
                if hasattr(instance, "simple_report"):
                    logger.info("Using simple_report fallback on WalkForwardBacktester")
                    folds = make_folds(start, end, train_years, test_months)
                    rows = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                        futures = {
                            ex.submit(
                                compute_returns_for_fold,
                                tickers,
                                f["test_start"],
                                f["test_end"],
                                source_mode,
                                min_days,
                            ): i
                            for i, f in enumerate(folds)
                        }
                        for fut in concurrent.futures.as_completed(futures):
                            i = futures[fut]
                            res = fut.result()
                            rows.append({"combined": res["return"]})
                    portfolio_df = pd.DataFrame(rows)
                    rep = instance.simple_report(portfolio_df)
                    out_folds = []
                    for i, f in enumerate(folds):
                        strat_r = float(rep.iloc[i].get("combined", 0.0)) if i < len(rep) else 0.0
                        out_folds.append(
                            {
                                "test_start": f["test_start"],
                                "test_end": f["test_end"],
                                "strategy_return": strat_r,
                            }
                        )
                    return {"folds": out_folds}
            except Exception as exc:
                logger.warning("WalkForwardBacktester use failed: %s", exc)
    except ModuleNotFoundError:
        logger.info("src.backtest.walkforward not found; using builtin simple WF")

    folds = make_folds(start, end, train_years, test_months)
    out_folds = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        futures = {
            ex.submit(
                compute_returns_for_fold,
                tickers,
                f["test_start"],
                f["test_end"],
                source_mode,
                min_days,
            ): i
            for i, f in enumerate(folds)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            res = fut.result()
            f = folds[i]
            out_folds.append(
                {
                    "test_start": f["test_start"],
                    "test_end": f["test_end"],
                    "strategy_return": res["return"],
                    "days": res["days"],
                    "missing": res["missing"],
                    "source": res["source"],
                }
            )

    out_folds = sorted(out_folds, key=lambda x: x["test_end"])
    return {"folds": out_folds}


# ---------- compare + plot ----------

def compare_and_plot(
    tickers,
    start: str,
    end: str,
    train_years: int,
    test_months: int,
    fred_key: str | None = None,
    out_prefix: Path | str = Path("outputs/backtest"),
    source_mode: str = "auto",
    min_days: int = 30,
    parallel_workers: int = 4,
):
    """
    Run walk-forward backtest, compare vs SPY/QQQ/VT, and save CSV + PNG + JSON.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = get_summary_with_adapters(
        tickers,
        start,
        end,
        train_years,
        test_months,
        fred_key=fred_key,
        source_mode=source_mode,
        min_days=min_days,
        parallel_workers=parallel_workers,
    )
    folds = summary.get("folds", [])

    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # prefetch benchmarks once
    benchmarks = ["SPY", "QQQ", "VT"]
    bench_series = {}
    for b in benchmarks:
        dfb = get_price_series(b, start, end, source_mode, debug_dir)
        if dfb is not None and not dfb.empty:
            bench_series[b] = dfb["Close"].dropna()

    dates = []
    strat_vals = []
    bench_vals = {b: [] for b in benchmarks}
    cum_strat = 1.0
    cum_bench = {b: 1.0 for b in benchmarks}

    fold_summaries = []

    for f in folds:
        ts = f.get("test_start")
        te = f.get("test_end")
        strat_r = f.get("strategy_return", 0.0)
        days = f.get("days", None)
        missing = f.get("missing", None)
        source_used = f.get("source", source_mode)

        # benchmark returns per fold
        bench_ret = {}
        for b in benchmarks:
            br = 0.0
            try:
                if b in bench_series:
                    s = bench_series[b]
                    mask = (s.index >= pd.to_datetime(ts)) & (s.index <= pd.to_datetime(te))
                    sub = s[mask]
                    if not sub.empty:
                        entry = sub.iloc[0]
                        exit_ = sub.iloc[-1]
                        br = (exit_ / entry) - 1.0
            except Exception:
                br = 0.0
            bench_ret[b] = br

        cum_strat *= (1.0 + strat_r)
        for b in benchmarks:
            cum_bench[b] *= (1.0 + bench_ret[b])

        dates.append(pd.to_datetime(te))
        strat_vals.append(cum_strat)
        for b in benchmarks:
            bench_vals[b].append(cum_bench[b])

        fold_summaries.append(
            {
                "test_start": ts,
                "test_end": te,
                "strategy_return": strat_r,
                "cum_strategy": cum_strat,
                "bench_returns": bench_ret,
                "cum_benchmarks": {b: cum_bench[b] for b in benchmarks},
                "days": days,
                "missing": missing,
                "source": source_used,
            }
        )

    df = pd.DataFrame(
        {
            "date": dates,
            "strategy_cum": strat_vals,
            **{f"{b.lower()}_cum": bench_vals[b] for b in benchmarks},
        }
    )
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)

    # fold summary JSON
    folds_json_path = out_prefix.with_suffix(".folds.json")
    try:
        folds_json_path.write_text(json.dumps(fold_summaries, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write fold summary JSON: %s", exc)

    # plots: equity curves, fold bars, heatmap, rolling Sharpe
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # equity curves
        plt.figure(figsize=(10, 6))
        plt.plot(df["date"], df["strategy_cum"], label="Strategy", linewidth=2)
        for b in benchmarks:
            plt.plot(df["date"], df[f"{b.lower()}_cum"], label=b, linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Cumulative value (start = 1.0)")
        plt.title("Walk-forward backtest vs SPY / QQQ / VT")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix(".equity.png"))
        plt.close()

        # fold returns bar chart
        fold_dates = [pd.to_datetime(f["test_end"]) for f in fold_summaries]
        fold_rets = [f["strategy_return"] for f in fold_summaries]
        plt.figure(figsize=(10, 4))
        plt.bar(fold_dates, fold_rets, width=10)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("Fold end date")
        plt.ylabel("Fold return")
        plt.title("Fold-level strategy returns")
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix(".folds.png"))
        plt.close()

        # data quality heatmap (days vs missing)
        days_arr = np.array([f["days"] or 0 for f in fold_summaries])
        missing_arr = np.array([f["missing"] or 0 for f in fold_summaries])
        quality = np.clip(days_arr / (days_arr + missing_arr + 1e-9), 0.0, 1.0)
        plt.figure(figsize=(10, 2))
        plt.imshow(
            quality.reshape(1, -1),
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(label="Data quality (1 = full)")
        plt.xticks(
            range(len(fold_dates)),
            [d.strftime("%Y-%m") for d in fold_dates],
            rotation=90,
        )
        plt.yticks([])
        plt.title("Fold data quality heatmap")
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix(".quality.png"))
        plt.close()

        # rolling Sharpe over folds (using fold returns)
        rets = np.array(fold_rets)
        window = max(3, min(12, len(rets)))
        roll_sharpe = []
        for i in range(len(rets)):
            if i + 1 < window:
                roll_sharpe.append(np.nan)
            else:
                rwin = rets[i + 1 - window : i + 1]
                mu = np.mean(rwin)
                sigma = np.std(rwin)
                roll_sharpe.append(mu / sigma if sigma > 1e-9 else np.nan)
        plt.figure(figsize=(10, 4))
        plt.plot(fold_dates, roll_sharpe, label=f"Rolling Sharpe (window={window})")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("Fold end date")
        plt.ylabel("Sharpe (approx)")
        plt.title("Rolling Sharpe across folds")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix(".sharpe.png"))
        plt.close()

    except Exception as exc:
        logger.warning("Failed to plot backtest results: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Compare walk-forward backtest vs SPY/QQQ/VT")
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker(s) to backtest (space-separated)",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--train-years",
        type=int,
        required=True,
        help="Training window in years",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        required=True,
        help="Test window in months",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/backtest",
        help="Output prefix (CSV + PNG)",
    )
    parser.add_argument(
        "--fred-key",
        type=str,
        default=None,
        help="Optional FRED API key",
    )
    parser.add_argument(
        "--source-mode",
        type=str,
        default="auto",
        choices=["auto", "tickerbot", "snapshot", "fmp", "yf"],
        help="Data source preference (auto/tickerbot/snapshot/fmp/yf)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=30,
        help="Minimum days of data required per fold",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for fold computation",
    )

    args = parser.parse_args()
    compare_and_plot(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        train_years=args.train_years,
        test_months=args.test_months,
        fred_key=args.fred_key,
        out_prefix=Path(args.out),
        source_mode=args.source_mode,
        min_days=args.min_days,
        parallel_workers=args.workers,
    )


if __name__ == "__main__":
    main()
