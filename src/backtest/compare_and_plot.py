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

# ---------- cached JSON helpers ----------

def load_cached_series(ticker: str, start: str, end: str, cache_dir: Path):
    cache_file = cache_dir / f"{ticker}_{start}_{end}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_series(ticker: str, start: str, end: str, cache_dir: Path, data):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f)

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

# ---------- NEW: local parquet loader ----------

def _local_parquet_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
    p = Path("cache") / f"{ticker}_price.parquet"
    if not p.exists():
        return None

    try:
        df = pd.read_parquet(p)
        if df is None or df.empty:
            return None

        df = df.copy()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        if df.empty:
            return None

        logger.info("Local parquet source selected for %s (rows=%d)", ticker, len(df))
        return df
    except Exception as exc:
        logger.info("Local parquet load failed for %s: %s", ticker, exc)
        return None

# ---------- data sources ----------

def _tickerbot_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
    cache_dir = Path("outputs/cache")

    cached = load_cached_series(ticker, start, end, cache_dir)
    if cached is not None:
        return _parse_series_json(ticker, cached)

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

        _save_debug_bytes(debug_dir / f"series_raw_{ticker}_{start}_{end}.json", resp.content)

        if "error" in raw_json:
            return None

        save_cached_series(ticker, start, end, cache_dir, raw_json)

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

def _tickerbot_snapshot_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
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

def _fmp_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
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

def _yf_download_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
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

    cached = _load_cache(ticker, start, end)
    if cached is not None and not cached.empty:
        return cached

    methods_auto = [
        ("local_parquet", _local_parquet_series),
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
    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Multi-ticker case
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

    # Single ticker case
    else:
        t = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
        df = get_price_series(t, test_start, test_end, source_mode, debug_dir)
        r, n = _compute_from_df(df)

        if r is None or n < min_days:
            return {"return": 0.0, "days": n or 0, "missing": 0, "source": source_mode}

        expected_days = (pd.to_datetime(test_end) - pd.to_datetime(test_start)).days + 1
        missing = max(0, expected_days - n)

        return {
            "return": r
