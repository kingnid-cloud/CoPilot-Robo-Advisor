from __future__ import annotations
from typing import Dict, Any, List
from src.data.connectors import fetch_earnings_revisions
from src.utils.secrets import load_secrets
import pandas as pd
import logging

logger = logging.getLogger("RevisionsScorer")
logger.setLevel(logging.INFO)


def compute_revision_metrics(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    For each ticker, fetch earnings/analyst recommendation data and compute:
      - upgrades_count, downgrades_count, net_upgrades, net_upgrade_ratio
      - revision_momentum (normalized -1..1)
      - revision_confidence (based on number of analysts / data completeness)
    Returns DataFrame indexed by ticker with these fields.
    """
    secrets = load_secrets()
    rows = []
    for t in tickers:
        try:
            data = fetch_earnings_revisions(t, secrets, use_cache=use_cache)
            if not data:
                rows.append({"ticker": t, "provider": None, "upgrades": None, "downgrades": None, "net": None, "net_ratio": None, "revision_momentum": 0.0, "revision_confidence": 0.0})
                continue
            prov = data.get("provider")
            payload = data.get("data")
            # Finnhub returns a list of dicts with fields: period, strongBuy, buy, hold, sell, strongSell
            upgrades = 0
            downgrades = 0
            total_analysts = 0
            if prov == "finnhub" and isinstance(payload, list):
                # look at most recent entry
                recent = payload[-1]
                # approximate upgrades as (strongBuy + buy), downgrades as (sell + strongSell)
                upgrades = int(recent.get("buy", 0) + recent.get("strongBuy", 0))
                downgrades = int(recent.get("sell", 0) + recent.get("strongSell", 0))
                total_analysts = upgrades + downgrades + int(recent.get("hold", 0)) + int(recent.get("period", 0) if recent.get("period") else 0)
            elif prov == "fmp" and isinstance(payload, dict):
                # FMP structure may vary; try to extract ratings
                upgrades = None
                downgrades = None
                total_analysts = payload.get("totalRatings") or 0
            else:
                upgrades = None
                downgrades = None
                total_analysts = 0

            net = None
            net_ratio = None
            if upgrades is not None and downgrades is not None:
                net = upgrades - downgrades
                denom = (upgrades + downgrades) if (upgrades + downgrades) != 0 else 1
                net_ratio = (upgrades - downgrades) / denom
            # revision momentum normalized to [-1,1]
            momentum = float(net_ratio) if net_ratio is not None else 0.0
            confidence = min(1.0, float(total_analysts) / 20.0) if total_analysts > 0 else 0.0

            rows.append({"ticker": t, "provider": prov, "upgrades": upgrades, "downgrades": downgrades, "net": net, "net_ratio": net_ratio, "revision_momentum": momentum, "revision_confidence": confidence})
        except Exception:
            logger.exception("Revision fetch failed for %s", t)
            rows.append({"ticker": t, "provider": None, "upgrades": None, "downgrades": None, "net": None, "net_ratio": None, "revision_momentum": 0.0, "revision_confidence": 0.0})
    df = pd.DataFrame(rows).set_index("ticker")
    return df
