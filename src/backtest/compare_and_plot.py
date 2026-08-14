from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import time
import importlib

import pandas as pd
import yfinance as yf
import requests
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
        folds.append(
            {
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            }
        )
        current_train_start = current_train_start + relativedelta(months=test_months)
    return folds


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

    # Normalize possible column names
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

    return float((exit_ / entry) - 1.0), len(closes)


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


def _tickerbot_series(
    ticker: str,
    start: str,
    end: str,
    debug_dir: Path,
) -> pd.DataFrame | None:
    """
    Primary source: Tickerbot /v2/series — aligned grid of prices over time.

    Multi-source, robust return computation. Uses Tickerbot /v2/series for ranges first,
    falls back to snapshot-based series if needed.
    """
    base = os.environ.get("TICKERBOT_URL", "https://api.tickerbot.io").strip().rstrip("/")
    key = os.environ.get("TICKERBOT_API_KEY", "").strip()
    if not base:
        logger.info("Tickerbot base URL not set; skipping /v2/series for %s", ticker)
        return None

    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    url = f"{base}/v2/series"
    params = {
        "tickers": ticker,
        "columns": "price",
        "interval": "1d",
        "start": start,
        "end": end,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        _save_debug_bytes(
            debug_dir / f"tickerbot_raw_series_{ticker}_{start}_{end}.json",
            resp.content,
        )
        logger.info("Tickerbot /v2/series %s status=%s", url, resp.status_code)
        if resp.status_code != 200:
            return None
        try:
            j = resp.json()
        except Exception:
            logger.info("Tickerbot /v2/series response not JSON for %s", url)
            return None

        prices = _parse_prices_from_json(j)
        if prices is None:
            logger.info("Tickerbot /v2/series returned 200 but no price list for %s", ticker)
            return None

        if isinstance(prices, dict):
            try:
                prices = [
                    {
                        "date": k,
                        **(
                            prices[k]
                            if isinstance(prices[k], dict)
                            else {"close": prices[k]}
                        ),
                    }
                    for k in prices.keys()
                ]
            except Exception:
                prices = [prices]

        df = pd.DataFrame(prices)
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            except Exception:
                pass
        if "price" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"price": "Close"})
        if "close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"close": "Close"})
        if "Close" in df.columns:
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        try:
            parsed_debug = {
                "source": "series",
                "parsed_shape": getattr(df, "shape", None),
                "columns": list(df.columns),
            }
            if "Close" in df:
                desc = df["Close"].describe().to_dict()
                parsed_debug["close_describe"] = {
                    k: (float(v) if pd.notna(v) else None)
                    for k, v in desc.items()
                }
            (debug_dir / f"tickerbot_parsed_series_{ticker}_{start}_{end}.json").write_text(
                json.dumps(parsed_debug),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to write parsed debug for tickerbot /v2/series")

        return df
    except Exception as exc:
        logger.info("Tickerbot /v2/series request failed for %s: %s", ticker, exc)
        return None


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

    try:
        parsed_debug = {
            "source": "snapshot_series",
            "parsed_shape": getattr(df, "shape", None),
            "sample_rows": rows[:3],
        }
        (debug_dir / f"tickerbot_parsed_snapshotseries_{ticker}_{start}_{end}.json").write_text(
            json.dumps(parsed_debug),
            encoding="utf-8",
        )
    except Exception:
        pass

    logger.info(
        "Built snapshot-based series for %s with %d rows",
        ticker,
        getattr(df, "shape", None)[0] if getattr(df, "shape", None) else 0,
    )
    return df


def _yf_download_series(ticker: str, start: str, end: str, debug_dir: Path) -> pd.DataFrame | None:
    """
    Secondary fallback: yfinance daily prices.
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


def compute_returns_for_fold(tickers, test_start: str, test_end: str) -> float:
    """
    Robust return computation for one fold.

    Order of preference:
      1. Tickerbot /v2/series (daily price grid)
      2. Tickerbot snapshot-per-day series via /v2/tickers/{ticker}?asof=
      3. yfinance daily prices
    """
    debug_dir = Path("outputs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("tickerbot_series", _tickerbot_series),
        ("tickerbot_snapshot", _tickerbot_snapshot_series),
        ("yf_download", _yf_download_series),
    ]

    max_attempts = 3

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
                            df = fn(t, test_start, test_end, debug_dir)
                        except Exception as exc:
                            logger.info(
                                "Method %s raised for %s (attempt %d): %s",
                                name,
                                t,
                                attempt,
                                exc,
                            )
                        r, n = _compute_from_df(df)
                        if r is not None and n > best_len:
                            best_ret = r
                            best_len = n
                            logger.info(
                                "Method %s selected for %s: return=%0.6f (n=%d)",
                                name,
                                t,
                                r,
                                n,
                            )
                        else:
                            logger.info(
                                "Method %s produced no usable data (n=%d) for %s (attempt %d)",
                                name,
                                n,
                                t,
                                attempt,
                            )
                    if best_ret is not None:
                        rets.append(best_ret)
                    else:
                        logger.warning(
                            "No valid data found for ticker %s across methods (attempt %d)",
                            t,
                            attempt,
                        )
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
                        df = fn(t, test_start, test_end, debug_dir)
                    except Exception as exc:
                        logger.info(
                            "Method %s raised for %s (attempt %d): %s",
                            name,
                            t,
                            attempt,
                            exc,
                        )
                    r, n = _compute_from_df(df)
                    if r is not None and n > best_len:
                        best_ret = r
                        best_len = n
                        logger.info(
                            "Method %s selected for %s: return=%0.6f (n=%d)",
                            name,
                            t,
                            r,
                            n,
                        )
                    else:
                        logger.info(
                            "Method %s produced no usable data (n=%d) for %s (attempt %d)",
                            name,
                            n,
                            t,
                            attempt,
                        )
                if best_ret is not None:
                    return best_ret
                else:
                    raise RuntimeError(f"No data for ticker {t}")
        except Exception as exc:
            logger.warning(
                "Attempt %d: Failed to compute returns for %s (%s-%s): %s",
                attempt,
                tickers,
                test_start,
                test_end,
                exc,
            )
            if attempt < max_attempts:
                sleep_s = 2 ** attempt
                logger.info("Sleeping %d seconds before retry...", sleep_s)
                time.sleep(sleep_s)
            else:
                logger.error(
                    "All attempts failed for %s (%s-%s); returning 0.0",
                    tickers,
                    test_start,
                    test_end,
                )
                return 0.0


def get_summary_with_adapters(
    tickers,
    start: str,
    end: str,
    train_years: int,
    test_months: int,
    fred_key: str | None = None,
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
                    for f in folds:
                        r = compute_returns_for_fold(
                            tickers,
                            f["test_start"],
                            f["test_end"],
                        )
                        rows.append({"combined": r})
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
    for f in folds:
        strat_r = compute_returns_for_fold(
            tickers,
            f["test_start"],
            f["test_end"],
        )
        out_folds.append(
            {
                "test_start": f["test_start"],
                "test_end": f["test_end"],
                "strategy_return": strat_r,
            }
        )
    return {"folds": out_folds}


def compare_and_plot(
    tickers,
    start: str,
    end: str,
    train_years: int,
    test_months: int,
    fred_key: str | None = None,
    out_prefix: Path | str = Path("outputs/backtest"),
):
    """
    Run walk-forward backtest, compare vs SPY, and save CSV + PNG.
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
    )
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

        # SPY benchmark over same period — prefer Tickerbot, fallback to yfinance
        spy_r = 0.0
        try:
            debug_dir = Path("outputs/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            spy_df = _tickerbot_series("SPY", ts, te, debug_dir)
            if spy_df is None or spy_df.empty:
                spy_df = _tickerbot_snapshot_series("SPY", ts, te, debug_dir)
            if spy_df is None or spy_df.empty:
                spy_df = _yf_download_series("SPY", ts, te, debug_dir)

            if spy_df is not None and not spy_df.empty:
                close = spy_df["Close"].dropna()
                if not close.empty:
                    spy_entry = close.iloc[0]
                    spy_exit = close.iloc[-1]
                    spy_r = (spy_exit / spy_entry) - 1.0
        except Exception:
            spy_r = 0.0

        cum_strat *= (1.0 + strat_r)
        cum_spy *= (1.0 + spy_r)
        dates.append(pd.to_datetime(te))
        strat_vals.append(cum_strat)
        spy_vals.append(cum_spy)

    df = pd.DataFrame(
        {
            "date": dates,
            "strategy_cum": strat_vals,
            "spy_cum": spy_vals,
        }
    )
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(df["date"], df["strategy_cum"], label="Strategy", linewidth=2)
        plt.plot(df["date"], df["spy_cum"], label="SPY", linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Cumulative value (start = 1.0)")
        plt.title("Walk-forward backtest vs SPY")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix(".png"))
        plt.close()
    except Exception as exc:
        logger.warning("Failed to plot backtest results: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Compare walk-forward backtest vs SPY")
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

    args = parser.parse_args()
    compare_and_plot(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        train_years=args.train_years,
        test_months=args.test_months,
        fred_key=args.fred_key,
        out_prefix=Path(args.out),
    )


if __name__ == "__main__":
    main()
