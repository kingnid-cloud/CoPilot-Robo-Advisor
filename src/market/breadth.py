from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("Breadth")
logger.setLevel(logging.INFO)


def compute_breadth_from_prices(prices: Dict[str, pd.DataFrame], min_history: int = 252) -> pd.DataFrame:
    """
    Given a dict ticker -> price DataFrame (with Close), compute market breadth series.
    Returns DataFrame indexed by business days with columns:
      advancers, decliners, ad_line, pct_above_200d, pct_above_50d, new_highs_52w, new_lows_52w, pct_positive_1y
    """
    # collect closes into panel DataFrame
    closes = []
    keys = []
    for t, df in prices.items():
        try:
            if df is None or df.empty or "Close" not in df.columns:
                continue
            s = df["Close"].rename(t).dropna()
            closes.append(s)
            keys.append(t)
        except Exception:
            continue
    if not closes:
        return pd.DataFrame()
    price_df = pd.concat(closes, axis=1).sort_index()
    # ensure business day index
    price_df = price_df.asfreq("B").ffill().bfill()

    # advancers/decliners
    pct_change = price_df.pct_change()
    adv = (pct_change > 0).sum(axis=1)
    dec = (pct_change < 0).sum(axis=1)
    ad_net = adv - dec
    ad_line = ad_net.cumsum()

    # pct above 200d and 50d
    pct_above_200 = (price_df > price_df.rolling(window=200, min_periods=50).mean()).sum(axis=1) / price_df.shape[1]
    pct_above_50 = (price_df > price_df.rolling(window=50, min_periods=20).mean()).sum(axis=1) / price_df.shape[1]

    # new highs/lows 52-week
    rolling_252_max = price_df.rolling(window=252, min_periods=50).max()
    rolling_252_min = price_df.rolling(window=252, min_periods=50).min()
    new_highs = (price_df == rolling_252_max).sum(axis=1)
    new_lows = (price_df == rolling_252_min).sum(axis=1)

    # percent positive 1y
    pct_pos_1y = (price_df.pct_change(252) > 0).sum(axis=1) / price_df.shape[1]

    df = pd.DataFrame(
        {
            "advancers": adv,
            "decliners": dec,
            "ad_net": ad_net,
            "ad_line": ad_line,
            "pct_above_200d": pct_above_200,
            "pct_above_50d": pct_above_50,
            "new_highs_52w": new_highs,
            "new_lows_52w": new_lows,
            "pct_positive_1y": pct_pos_1y,
        },
        index=price_df.index,
    )

    # smooth some series
    df["pct_above_200d_sma5"] = df["pct_above_200d"].rolling(5).mean()
    df["ad_line_slope_10"] = df["ad_line"].diff(10).fillna(0.0)
    return df
