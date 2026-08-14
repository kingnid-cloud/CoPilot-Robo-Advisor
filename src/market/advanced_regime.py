"""
Advanced regime detection combining DFM, HMM, fuzzy clustering, and market breadth indicators.
This updates the previous module to accept universe price histories and include breadth into indicators.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
import time
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader.data import DataReader
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from src.market.breadth import compute_breadth_from_prices

try:
    import skfuzzy as fuzz
    _HAVE_FUZZY = True
except Exception:
    _HAVE_FUZZY = False

logger = logging.getLogger("AdvancedRegime")
logger.setLevel(logging.INFO)

CACHE_DIR = Path("cache") / "macro"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("cache") / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _cache_df(df: pd.DataFrame, fn: Path) -> None:
    try:
        fn.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fn)
    except Exception:
        logger.exception("Failed to cache macro data to %s", fn)


def _load_cached(fn: Path) -> Optional[pd.DataFrame]:
    if not fn.exists():
        return None
    try:
        return pd.read_parquet(fn)
    except Exception:
        logger.warning("Failed to read cached macro file %s", fn)
        return None


def fetch_macro_series(
    start: str = "2000-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    cache_file = CACHE_DIR / f"macro_{start}_{end}.parquet"
    if use_cache:
        df_cached = _load_cached(cache_file)
        if df_cached is not None:
            logger.info("Loaded macro series from cache %s", cache_file)
            return df_cached

    fred_series = {
        "GDPC1": "RealGDP",
        "CPILFESL": "CoreCPI",
        "UNRATE": "Unemployment",
        "FEDFUNDS": "FedFunds",
        "DGS10": "10Y",
        "VIXCLS": "VIX_FRED",
    }

    dfs = {}
    for code, name in fred_series.items():
        try:
            if fred_api_key:
                import os

                os.environ["FRED_API_KEY"] = fred_api_key
            ser = DataReader(code, "fred", start, end)
            ser = ser.rename(columns={code: name}) if isinstance(ser, pd.DataFrame) else ser.to_frame(name)
            dfs[name] = ser
            time.sleep(0.2)
        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", code, e)
            dfs[name] = None

    if (dfs.get("VIX_FRED") is None) or dfs.get("VIX_FRED") is None:
        try:
            vix = yf.download("^VIX", start=start, end=end, progress=False, threads=False)
            if not vix.empty and "Close" in vix.columns:
                dfs["VIX_FRED"] = vix[["Close"]].rename(columns={"Close": "VIX_FRED"})
            else:
                dfs["VIX_FRED"] = None
        except Exception as e:
            logger.warning("yfinance VIX fetch failed: %s", e)
            dfs["VIX_FRED"] = None

    try:
        spy = yf.download("SPY", start=start, end=end, progress=False, threads=False)
        dfs["SPY_Close"] = spy[["Close"]]
    except Exception as e:
        logger.warning("SPY fetch failed: %s", e)
        dfs["SPY_Close"] = None

    aligned = []
    for name, df in dfs.items():
        if df is None:
            continue
        s = df.copy()
        s.index = pd.to_datetime(s.index)
        try:
            s = s.resample("B").ffill().bfill()
        except Exception:
            s = s.asfreq("B").ffill().bfill()
        s.columns = [name]
        aligned.append(s[[name]])

    if not aligned:
        logger.error("No macro series available")
        return pd.DataFrame()

    big = pd.concat(aligned, axis=1, join="outer").sort_index()
    big = big.fillna(method="ffill").fillna(method="bfill")
    _cache_df(big, cache_file)
    return big


def build_indicator_matrix(macro_df: pd.DataFrame, breadth_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = macro_df.copy()
    out = pd.DataFrame(index=df.index)
    if "SPY_Close" in df.columns:
        close = df["SPY_Close"]
        out["spy_ret_21"] = close.pct_change(21)
        out["spy_ret_63"] = close.pct_change(63)
        out["spy_ret_252"] = close.pct_change(252)
        daily_ret = close.pct_change().dropna()
        out["rv_21"] = daily_ret.rolling(21).std() * np.sqrt(252)
        out["rv_63"] = daily_ret.rolling(63).std() * np.sqrt(252)
    else:
        out["spy_ret_21"] = 0.0
        out["rv_21"] = 0.0

    if "VIX_FRED" in df.columns:
        out["vix"] = df["VIX_FRED"]

    if "RealGDP" in df.columns:
        out["gdp_growth_q4"] = df["RealGDP"].pct_change(4)

    if "CoreCPI" in df.columns:
        out["cpi_yoy"] = df["CoreCPI"].pct_change(12)

    if "Unemployment" in df.columns:
        out["unemployment"] = df["Unemployment"]

    if "FedFunds" in df.columns and "10Y" in df.columns:
        out["term_spread"] = df["10Y"] - df["FedFunds"]
    elif "10Y" in df.columns:
        out["term_spread"] = df["10Y"]

    # merge breadth indicators if provided
    if breadth_df is not None and not breadth_df.empty:
        # align indices
        bf = breadth_df.reindex(out.index).fillna(method="ffill").fillna(method="bfill")
        # pick a subset of breadth features
        for c in ["pct_above_200d", "pct_above_50d", "ad_line", "ad_line_slope_10", "new_highs_52w", "new_lows_52w", "pct_positive_1y"]:
            if c in bf.columns:
                out[c] = bf[c]

    out = out.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    out = out.clip(lower=-10, upper=10)
    return out


def run_dynamic_factor(indicators: pd.DataFrame, n_factors: int = 2, maxiter: int = 1000) -> Tuple[pd.DataFrame, Any]:
    clean = indicators.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if clean.shape[0] < max(100, n_factors * 10):
        scaler = StandardScaler()
        Z = scaler.fit_transform(clean.fillna(method="ffill").fillna(0.0))
        pca = PCA(n_components=n_factors)
        fac = pca.fit_transform(Z)
        fac_df = pd.DataFrame(fac, index=clean.index, columns=[f"factor_{i}" for i in range(n_factors)])
        return fac_df, {"method": "pca_init", "pca": pca, "scaler": scaler}

    try:
        mod = DynamicFactor(clean, k_factors=n_factors, factor_order=1)
        res = mod.fit(maxiter=maxiter, disp=False)
        factors = res.factors
        factors.columns = [f"factor_{i}" for i in range(factors.shape[1])]
        fac_df = pd.DataFrame(factors, index=clean.index)
        return fac_df, res
    except Exception as e:
        logger.exception("DynamicFactor fit failed, falling back to PCA: %s", e)
        scaler = StandardScaler()
        Z = scaler.fit_transform(clean.fillna(method="ffill").fillna(0.0))
        pca = PCA(n_components=n_factors)
        fac = pca.fit_transform(Z)
        fac_df = pd.DataFrame(fac, index=clean.index, columns=[f"factor_{i}" for i in range(n_factors)])
        return fac_df, {"method": "pca_fallback", "pca": pca, "scaler": scaler, "error": str(e)}


def run_hmm(factors: pd.DataFrame, n_states: int = 3, cov_type: str = "full", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    X = factors.fillna(method="ffill").fillna(0.0).values
    if X.shape[0] < 5:
        raise ValueError("Not enough rows for HMM")
    Xd = np.diff(X, axis=0)
    X_stack = np.hstack([X[1:], Xd])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_stack)
    model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=200, random_state=random_state)
    model.fit(Xs)
    states = model.predict(Xs)
    probs = model.predict_proba(Xs)
    return states, probs, {"model": model, "scaler": scaler, "index": factors.index[1:]}


def fuzzy_headwinds(indicators: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    if not _HAVE_FUZZY:
        logger.info("skfuzzy not installed; skipping fuzzy headwinds")
        return {"enabled": False}
    X = indicators.fillna(method="ffill").fillna(0.0).values.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None)
    mem = u[:, -1]
    return {"enabled": True, "centroids": cntr.tolist(), "membership": mem.tolist(), "fpc": float(fpc)}


def summarize_regime(indicators: pd.DataFrame, factors: pd.DataFrame, hmm_states: np.ndarray, hmm_probs: np.ndarray, hmm_meta: Dict[str, Any], fuzzy: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    idx = hmm_meta["index"]
    state_idx = hmm_states
    probs = hmm_probs
    df = pd.DataFrame(index=idx)
    for c in factors.columns:
        df[c] = factors.loc[idx, c]
    ind = indicators.loc[idx]
    df = pd.concat([df, ind], axis=1)
    n_states = probs.shape[1]
    state_stats = {}
    for s in range(n_states):
        mask = state_idx == s
        if mask.sum() == 0:
            state_stats[s] = {"spy_ret": 0.0, "rv": np.inf}
            continue
        spy_mean = float(df.loc[mask, "spy_ret_21"].mean()) if "spy_ret_21" in df.columns else 0.0
        rv = float(df.loc[mask, "rv_21"].mean()) if "rv_21" in df.columns else np.nan
        state_stats[s] = {"spy_ret": spy_mean, "rv": rv, "count": int(mask.sum())}
    ranking = sorted(state_stats.items(), key=lambda kv: (kv[1]["spy_ret"], -kv[1]["rv"]), reverse=True)
    label_map = {}
    labels = ["BULL", "NEUTRAL", "BEAR"]
    for rank, (s, st) in enumerate(ranking):
        lab = labels[min(rank, len(labels) - 1)]
        label_map[s] = lab
    cur_idx = idx[-1]
    cur_state = state_idx[-1]
    cur_state_prob = probs[-1, cur_state]
    cur_vix = float(df.loc[cur_idx, "vix"]) if "vix" in df.columns else None
    base_conf = float(np.clip(cur_state_prob, 0.0, 1.0))
    vix_bonus = 0.0
    if cur_vix is not None:
        if cur_vix < 15:
            vix_bonus = 0.15
        elif cur_vix > 25:
            vix_bonus = -0.25
    ma_slope = None
    if "spy_ret_63" in df.columns and "spy_ret_21" in df.columns:
        ma_slope = float(df["spy_ret_21"].iloc[-1] - df["spy_ret_63"].iloc[-1])
    slope_bonus = 0.0
    if ma_slope is not None:
        if ma_slope > 0.02:
            slope_bonus = 0.1
        elif ma_slope < -0.02:
            slope_bonus = -0.2
    # breadth heuristics
    breadth_penalty = 0.0
    breadth_explanation = {}
    if "pct_above_200d" in df.columns:
        pct200 = float(df.loc[cur_idx, "pct_above_200d"])
        breadth_explanation["pct_above_200d"] = pct200
        if pct200 < 0.15:
            breadth_penalty -= 0.3
        elif pct200 < 0.25:
            breadth_penalty -= 0.15
    if "ad_line_slope_10" in df.columns:
        ad_slope = float(df.loc[cur_idx, "ad_line_slope_10"])
        breadth_explanation["ad_line_slope_10"] = ad_slope
        if ad_slope < -50:
            breadth_penalty -= 0.15
    if "new_highs_52w" in df.columns and "new_lows_52w" in df.columns:
        nh = float(df.loc[cur_idx, "new_highs_52w"])
        nl = float(df.loc[cur_idx, "new_lows_52w"])
        breadth_explanation["new_highs"] = nh
        breadth_explanation["new_lows"] = nl
        if (nh - nl) < 0 and (nh - nl) < -10:
            breadth_penalty -= 0.1
    label = label_map.get(cur_state, "NEUTRAL")
    conf = base_conf + vix_bonus + slope_bonus + breadth_penalty
    if label == "BEAR":
        conf = min(conf, 0.35)
    conf = float(np.clip(conf, 0.0, 1.0))
    explanation = {
        "current_state": int(cur_state),
        "state_label": label,
        "state_prob": float(cur_state_prob),
        "vix": cur_vix,
        "ma_slope_proxy": ma_slope,
        "breadth": breadth_explanation,
        "state_stats": state_stats,
        "label_map": label_map,
    }
    if fuzzy:
        explanation["fuzzy"] = fuzzy
    regime_summary = {"regime_label": label, "regime_confidence": conf, "explanation": explanation}
    return regime_summary, df


def detect_market_regime(
    start: str = "2000-01-01",
    end: Optional[str] = None,
    cache: bool = True,
    fred_api_key: Optional[str] = None,
    df_factors: int = 2,
    hmm_states: int = 3,
    fuzzy_clusters: int = 0,
    save_artifacts: bool = True,
    universe_prices: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    macro = fetch_macro_series(start=start, end=end, use_cache=cache, fred_api_key=fred_api_key)
    if macro.empty:
        return {"summary": {"regime_label": "NEUTRAL", "regime_confidence": 0.3, "explanation": {"reason": "no data"}}, "regime_df": pd.DataFrame(), "models": {}}
    bread = None
    if universe_prices:
        try:
            bread = compute_breadth_from_prices(universe_prices)
        except Exception:
            logger.exception("Breadth computation failed")
            bread = None
    indicators = build_indicator_matrix(macro, breadth_df=bread)
    factors, dfm_meta = run_dynamic_factor(indicators, n_factors=df_factors)
    try:
        states, probs, hmm_meta = run_hmm(factors, n_states=hmm_states)
    except Exception as e:
        logger.exception("HMM failed: %s", e)
        p = PCA(n_components=1)
        pc = p.fit_transform(indicators.fillna(0.0))
        states = (pc[1:, 0] > 0).astype(int)
        probs = np.vstack([np.abs(pc[1:, 0]) / (np.abs(pc[1:, 0]).max() + 1e-6), 1 - (np.abs(pc[1:, 0]) / (np.abs(pc[1:, 0]).max() + 1e-6))]).T
        hmm_meta = {"model": None, "scaler": None, "index": factors.index[1:]}
    fuzzy = None
    if fuzzy_clusters and _HAVE_FUZZY:
        fuzzy = fuzzy_headwinds(indicators, n_clusters=fuzzy_clusters)
    summary, regime_df = summarize_regime(indicators, factors, states, probs, hmm_meta, fuzzy=fuzzy)
    models = {"dfm_meta": dfm_meta, "hmm_meta": hmm_meta, "fuzzy_meta": fuzzy}
    if save_artifacts:
        try:
            art_fn = MODEL_DIR / f"regime_artifacts_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.pkl"
            with art_fn.open("wb") as f:
                pickle.dump({"models": models, "summary": summary}, f)
            logger.info("Saved regime artifacts to %s", art_fn)
        except Exception:
            logger.exception("Failed to save model artifacts")
    return {"summary": summary, "regime_df": regime_df, "models": models}
