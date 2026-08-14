from __future__ import annotations
from typing import Dict, List, Optional
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.normalize import percentile_within_group
from src.utils.curves import piecewise_score
from src.data.connectors import fetch_fundamental_with_fallback
from src.utils.secrets import load_secrets

# For GDP fallback
try:
    from pandas_datareader.data import DataReader
    _HAVE_PDR = True
except Exception:
    _HAVE_PDR = False

logger = logging.getLogger("QualityScorer")
logger.setLevel(logging.INFO)


class QualityScorer:
    def __init__(self, cfg: Dict = None):
        # cfg can contain per-metric breakpoints/scores and weights
        self.cfg = (cfg or {}).get("quality", {})
        # defaults for DCF assumptions will be computed from market data when possible

    def _estimate_growth_from_eps_hist(self, eps_hist: Dict[str, float]) -> Optional[float]:
        """Estimate CAGR from EPS history dict keyed by date/year.
        Require at least 3 non-null points spanning ~3 years.
        """
        try:
            if not eps_hist or not isinstance(eps_hist, dict):
                return None
            ser = pd.Series(eps_hist).dropna().sort_index()
            if ser.shape[0] < 3:
                return None
            # take last and 3-years-ago if available
            # use index as time values — try to coerce to datetime
            try:
                ser.index = pd.to_datetime(ser.index, errors='coerce')
                ser = ser[ser.index.notnull()]
            except Exception:
                pass
            if ser.shape[0] < 3:
                return None
            end = ser.iloc[-1]
            # find value ~3 years earlier
            cutoff = ser.index[-1] - pd.DateOffset(years=3) if hasattr(ser.index, 'dtype') else None
            if cutoff is not None and (ser.index <= cutoff).any():
                start_val = ser[ser.index <= cutoff].iloc[-1]
                years = (ser.index[-1] - ser[ser.index <= cutoff].index[-1]).days / 365.25
                if years <= 0 or start_val == 0:
                    return None
                cagr = (end / start_val) ** (1.0 / years) - 1.0
                return float(cagr)
            # fallback: simple average annual growth over available periods
            vals = ser.values
            n = len(vals) - 1
            if n <= 0 or vals[0] == 0:
                return None
            total_years = n
            cagr = (vals[-1] / vals[0]) ** (1.0 / total_years) - 1.0
            return float(cagr)
        except Exception:
            return None

    def _sector_median_growth(self, sector: str, fundamentals_df: pd.DataFrame) -> Optional[float]:
        try:
            if fundamentals_df is None or 'ticker' in fundamentals_df.columns and 'sector' in fundamentals_df.columns:
                # ensure indexed by ticker
                df = fundamentals_df.copy()
                if 'ticker' in df.columns:
                    df = df.set_index('ticker')
            else:
                df = fundamentals_df
            if sector not in df.columns and 'sector' in df.columns:
                # df already has sector column; filter
                pass
            # find earnings growth in sector
            mask = (df.get('sector') == sector) if 'sector' in df.columns else None
            if mask is None:
                return None
            vals = df.loc[mask, 'earningsGrowth'].dropna() if 'earningsGrowth' in df.columns else pd.Series(dtype=float)
            if vals.empty:
                # try other fields
                vals = df.loc[mask, 'EarningsGrowth'].dropna() if 'EarningsGrowth' in df.columns else pd.Series(dtype=float)
            if vals.empty:
                return None
            return float(vals.median())
        except Exception:
            return None

    def _gdp_growth_fallback(self) -> Optional[float]:
        if not _HAVE_PDR:
            return None
        try:
            ser = DataReader('A191RL1Q225SBEA', 'fred')
            # Attempt to compute recent GDP growth annualized from series
            ser = ser.dropna()
            if ser.empty:
                return None
            # compute YoY growth based on last available
            yoy = ser.pct_change(4).dropna()
            if yoy.empty:
                return None
            return float(yoy.iloc[-1])
        except Exception:
            try:
                ser = DataReader('GDPC1', 'fred')
                ser = ser.dropna()
                if ser.empty:
                    return None
                yoy = ser.pct_change(4).dropna()
                if yoy.empty:
                    return None
                return float(yoy.iloc[-1])
            except Exception:
                return None

    def _compute_dcf(self, ticker: str, fund_row: pd.Series, fundamentals_df: Optional[pd.DataFrame] = None, rf_rate: Optional[float] = None) -> Optional[Dict]:
        """
        Simple DCF estimator using Free Cash Flow (FCF).
        Fallback hierarchy for growth (g):
          1) analyst-provided earningsGrowth
          2) historical earnings growth series -> 3-yr CAGR
          3) sector median earnings growth (from fundamentals_df)
          4) GDP growth fallback (FRED)
          5) last resort small default (0.02)

        Returns dict with fields: implied_price, dcf_ratio, growth_used, growth_source, discount_rate
        Always returns a dict (with noted fallback) if sufficient inputs (FCF and shares/marketcap/price) exist.
        """
        try:
            fcf = None
            shares = None
            price = fund_row.get('price')
            fcf = fund_row.get('FCF') or fund_row.get('freeCashFlow') or fund_row.get('FreeCashFlow') or fund_row.get('fcf')
            shares = fund_row.get('sharesOutstanding') or fund_row.get('shares_outstanding') or fund_row.get('shares')
            market_cap = fund_row.get('market_cap') or fund_row.get('marketCap')
            if shares is None and market_cap and price and price > 0:
                try:
                    shares = market_cap / price
                except Exception:
                    shares = None
            if fcf is None or shares is None or price is None:
                # insufficient for DCF
                return None

            # 1) analyst growth
            g = None
            growth_source = None
            if fund_row.get('earningsGrowth') not in (None, '', 'None'):
                try:
                    g = float(fund_row.get('earningsGrowth'))
                    growth_source = 'analyst'
                except Exception:
                    g = None
            if g is None and fund_row.get('EarningsGrowth') not in (None, '', 'None'):
                try:
                    g = float(fund_row.get('EarningsGrowth'))
                    growth_source = 'analyst'
                except Exception:
                    g = None

            # 2) historical EPS series
            if g is None:
                eps_hist = fund_row.get('EPS_HIST') or fund_row.get('eps_history')
                est = None
                if isinstance(eps_hist, dict):
                    est = self._estimate_growth_from_eps_hist(eps_hist)
                if est is not None:
                    g = est
                    growth_source = 'eps_hist_cagr'

            # 3) sector median
            if g is None and fundamentals_df is not None:
                sector = fund_row.get('sector') or fund_row.get('industry') or None
                if sector:
                    est = self._sector_median_growth(sector, fundamentals_df)
                    if est is not None:
                        g = est
                        growth_source = 'sector_median'

            # 4) GDP fallback
            if g is None:
                est = self._gdp_growth_fallback()
                if est is not None:
                    g = est
                    growth_source = 'gdp'

            # 5) last resort default
            if g is None:
                g = 0.02
                growth_source = 'default_2pct'

            # discount rate
            beta = fund_row.get('Beta') if fund_row.get('Beta') not in (None, '', 'None') else None
            if rf_rate is None:
                rf_rate = float(fund_row.get('rf_rate', 0.02)) if fund_row.get('rf_rate') not in (None, '', 'None') else 0.02
            market_premium = 0.05
            try:
                if beta is not None:
                    r = rf_rate + float(beta) * market_premium
                else:
                    r = 0.08
            except Exception:
                r = 0.08

            # project next 5 years FCF and terminal value
            n = 5
            fcf0 = float(fcf)
            cashflows = [fcf0 * ((1 + g) ** (i + 1)) for i in range(n)]
            terminal_growth = min(g, 0.03)
            if r <= terminal_growth:
                # adjust r slightly above terminal growth to avoid division by zero
                r = terminal_growth + 0.01
            terminal = cashflows[-1] * (1 + terminal_growth) / (r - terminal_growth)
            pv = sum([cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cashflows)]) + terminal / ((1 + r) ** n)
            implied = pv / shares
            dcf_ratio = implied / price if price and price > 0 else None
            return {"implied_price": implied, "dcf_ratio": dcf_ratio, "discount_rate": r, "growth_assumption": float(g), "growth_source": growth_source}
        except Exception:
            logger.exception("DCF computation failed for %s", ticker)
            return None

    def compute_metrics(self, tickers: List[str], prices: Dict[str, pd.DataFrame], fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute quality metrics for a list of tickers.
        - prices: ticker -> price history DataFrame (Close, High, Low)
        - fundamentals: DataFrame indexed by ticker containing available fundamentals
        Returns DataFrame with metrics and a combined quality_score in [0,1].
        """
        rows = []
        secrets = load_secrets()
        # try to get rf rate from FRED if available via fundamentals
        for t in tickers:
            base = fundamentals.loc[t] if t in fundamentals.index else (fundamentals[fundamentals["ticker"] == t].set_index("ticker") if "ticker" in fundamentals.columns else pd.DataFrame()).loc[t] if t in fundamentals.index else None
            def safe_get(df, k):
                try:
                    return df.get(k) if isinstance(df, pd.Series) else None
                except Exception:
                    return None

            roic = safe_get(base, "ROIC")
            peg = safe_get(base, "PEG")
            gross = safe_get(base, "GrossMargin")
            sector = safe_get(base, "sector") or safe_get(base, "industry") or "UNKNOWN"
            ev = safe_get(base, "enterpriseValue") or safe_get(base, "enterprise_value") or None
            ebitda = safe_get(base, "ebitda") or safe_get(base, "EBITDA") or None
            debt_equity = safe_get(base, "debtToEquity") or safe_get(base, "debt_to_equity") or safe_get(base, "D/E") or None
            roe = safe_get(base, "ROE") or safe_get(base, "returnOnEquity") or None
            fcf = safe_get(base, "freeCashFlow") or safe_get(base, "FCF") or None
            pe = safe_get(base, "trailing_pe") or safe_get(base, "trailingPE") or safe_get(base, "pe") or None
            forward_pe = safe_get(base, "forward_pe") or safe_get(base, "forwardPE") or None
            pb = safe_get(base, "priceToBook") or safe_get(base, "pb") or None
            beta = safe_get(base, "beta") or None
            shares = safe_get(base, "sharesOutstanding") or safe_get(base, "shares_outstanding") or None
            price = safe_get(base, "price")

            price_df = prices.get(t)
            stddev = None
            sharpe = None
            if isinstance(price_df, pd.DataFrame) and not price_df.empty:
                close = price_df["Close"].dropna()
                if not close.empty:
                    daily = close.pct_change().dropna()
                    stddev = float(daily.std() * (252 ** 0.5)) if not daily.empty else None
                    ann_ret = float((1 + daily.mean()) ** 252 - 1) if not daily.empty else None
                    if stddev and stddev > 0:
                        sharpe = float((ann_ret - 0.02) / stddev)

            ev_ebitda = None
            try:
                if ev is not None and ebitda is not None:
                    ev_ebitda = float(ev) / float(ebitda) if float(ebitda) != 0 else None
            except Exception:
                ev_ebitda = None

            earnings_growth = safe_get(base, "earningsGrowth") or safe_get(base, "earnings_growth") or None
            operating_margin = safe_get(base, "operatingMargin") or safe_get(base, "operating_margin") or None

            # historical PE percentile (best-effort)
            hist_pe_pct = None
            try:
                earnings_series = None
                if isinstance(base, pd.Series) and base.get("EPS_HIST"):
                    eps_hist = base.get("EPS_HIST")
                    if isinstance(eps_hist, dict):
                        earnings_series = pd.Series(eps_hist)
                hist_pe_pct = None
                if earnings_series is not None:
                    # compute using helper
                    close = price_df["Close"].dropna() if price_df is not None else None
                    if close is not None and not close.empty:
                        eps_ser = earnings_series.reindex(close.index, method='ffill').fillna(method='bfill')
                        pe_series = close / eps_ser.replace({0: np.nan})
                        cur_pe = pe_series.dropna().iloc[-1]
                        hist_pe_pct = float((pe_series.dropna() <= cur_pe).mean())
            except Exception:
                hist_pe_pct = None

            rows.append({
                "ticker": t,
                "sector": sector,
                "ROIC": float(roic) if roic not in (None, "", "None") else None,
                "PEG": float(peg) if peg not in (None, "", "None") else None,
                "GrossMargin": float(gross) if gross not in (None, "", "None") else None,
                "EV_EBITDA": float(ev_ebitda) if ev_ebitda not in (None, "", "None") else None,
                "D_to_E": float(debt_equity) if debt_equity not in (None, "", "None") else None,
                "ROE": float(roe) if roe not in (None, "", "None") else None,
                "FCF": float(fcf) if fcf not in (None, "", "None") else None,
                "PE": float(pe) if pe not in (None, "", "None") else None,
                "ForwardPE": float(forward_pe) if forward_pe not in (None, "", "None") else None,
                "PB": float(pb) if pb not in (None, "", "None") else None,
                "Beta": float(beta) if beta not in (None, "", "None") else None,
                "StdDev": float(stddev) if stddev not in (None, "", "None") else None,
                "Sharpe": float(sharpe) if sharpe not in (None, "", "None") else None,
                "EarningsGrowth": float(earnings_growth) if earnings_growth not in (None, "", "None") else None,
                "OperatingMargin": float(operating_margin) if operating_margin not in (None, "", "None") else None,
                "hist_PE_pct": hist_pe_pct,
            })

        df = pd.DataFrame(rows).set_index("ticker")

        # Cross-sectional percentile within sector for PE, PB, ROIC, GrossMargin, EV/EBITDA
        for col in ["PE", "PB", "ROIC", "GrossMargin", "EV_EBITDA", "D_to_E"]:
            if col in df.columns:
                try:
                    tmp = df.reset_index()
                    tmp = percentile_within_group(tmp, group_col="sector", value_col=col, out_col=f"{col}_pct")
                    df = df.join(tmp.set_index("ticker")[f"{col}_pct"]) if f"{col}_pct" in tmp.columns else df
                except Exception:
                    continue

        # For metrics where lower is better, invert percentile
        for col in ["PE", "PB", "EV_EBITDA", "D_to_E"]:
            pct_col = f"{col}_pct"
            if pct_col in df.columns:
                df[pct_col] = 1.0 - df[pct_col]

        # Include historical PE percentile where available
        # Now apply piecewise scoring; use cfg metrics if present
        metric_cfg = self.cfg.get("metrics", {
            "ROIC_pct": {"breakpoints": [0.0, 0.5, 1.0], "scores": [0.0, 0.5, 1.0], "weight": 1.0},
            "PE_pct": {"breakpoints": [0.0, 0.5, 1.0], "scores": [1.0, 0.5, 0.0], "weight": 1.0},
            "GrossMargin_pct": {"breakpoints": [0.0, 0.5, 1.0], "scores": [0.0, 0.5, 1.0], "weight": 1.0},
            "EV_EBITDA_pct": {"breakpoints": [0.0, 0.5, 1.0], "scores": [1.0, 0.5, 0.0], "weight": 0.8},
        })

        score_components = []
        weights = []
        for metric, mc in metric_cfg.items():
            out_col = f"{metric}_score"
            bp = mc.get("breakpoints", [])
            sc = mc.get("scores", [])
            w = mc.get("weight", 1.0)
            # map metric key to df column (allow _pct suffix)
            src_col = metric if metric in df.columns else (metric.replace("_pct", "") + "_pct")
            if src_col in df.columns:
                def _map_val(x):
                    try:
                        return piecewise_score(float(x), bp, sc)
                    except Exception:
                        return 0.0
                df[out_col] = df[src_col].apply(_map_val)
                score_components.append(out_col)
                weights.append(w)

        total_w = sum(weights) or 1.0
        if score_components:
            df["quality_score"] = sum((df[c] * w for c, w in zip(score_components, weights))) / total_w
        else:
            df["quality_score"] = 0.0

        # DCF computations (best-effort): attach implied_price and dcf_ratio
        dcf_results = {}
        for t in df.index:
            try:
                dcf = self._compute_dcf(t, df.loc[t], fundamentals_df=fundamentals)
                if dcf:
                    dcf_results[t] = dcf
            except Exception:
                logger.exception("DCF failed for %s", t)
        if dcf_results:
            dcf_df = pd.DataFrame.from_dict(dcf_results, orient="index")
            df = df.join(dcf_df)

        # confidence: percent of non-missing numeric metrics
        num_metrics = [c for c in ["ROIC", "PEG", "GrossMargin", "EV_EBITDA", "D_to_E", "ROE", "FCF", "PE"] if c in df.columns]
        df["data_completeness"] = df[num_metrics].notnull().sum(axis=1) / max(1, len(num_metrics))

        logger.info("Computed quality metrics for %d tickers", len(df))
        return df
