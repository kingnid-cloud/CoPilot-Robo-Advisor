# src/scoring/fundamentals.py
from typing import Dict
import pandas as pd
from src.utils.normalize import percentile_within_group
from src.utils.curves import piecewise_score
import logging

logger = logging.getLogger("FundamentalScorer")

class FundamentalScorer:
    def __init__(self, cfg: Dict = None):
        self.cfg = (cfg or {}).get("scoring", {}).get("metrics", {})

    def score_company_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        metrics_df must contain columns: ticker, sector, metric columns (ROIC, PEG, GrossMargin, etc.)
        Returns DataFrame with per-metric scores and combined score.
        """
        df = metrics_df.copy()
        if "sector" not in df.columns:
            df["sector"] = "GLOBAL"

        # compute percentiles for metrics configured to use_percentile
        for metric, mcfg in self.cfg.items():
            if mcfg.get("use_percentile", False):
                pct_col = f"{metric}_pct"
                df = percentile_within_group(df, group_col="sector", value_col=metric, out_col=pct_col)

        # compute per-metric scores
        score_cols = []
        for metric, mcfg in self.cfg.items():
            bp = mcfg.get("breakpoints", [])
            sc = mcfg.get("scores", [])
            col_in = f"{metric}_pct" if mcfg.get("use_percentile", False) else metric
            out_col = f"{metric}_score"
            def _map_val(x):
                try:
                    return piecewise_score(float(x), bp, sc)
                except Exception:
                    return 0.0
            df[out_col] = df[col_in].apply(_map_val)
            score_cols.append((out_col, mcfg.get("weight", 1.0)))

        total_weight = sum(w for _, w in score_cols) or 1.0
        df["fund_score"] = sum(df[c] * w for c, w in score_cols) / total_weight
        return df
