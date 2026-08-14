#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

from src.market.advanced_regime import detect_market_regime
from src.utils.io import save_csv, save_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RegimeAgentMain")


def derive_portfolio_controls(summary: dict, params: dict | None = None) -> dict:
    p = params or {}
    base_max_pos = p.get("base_max_position_pct", 0.10)
    base_cash = p.get("base_cash_buffer", 0.02)
    conf = float(summary.get("regime_confidence", 0.5))
    label = summary.get("regime_label", "NEUTRAL").upper()
    controls = {
        "regime_label": label,
        "regime_confidence": conf,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "NORMAL",
        "allow_new_buys": True,
        "target_cash_buffer": base_cash,
        "max_position_pct": base_max_pos,
        "suggested_allocation_scale": 1.0,
        "explanation": summary.get("explanation", {}),
    }
    if label == "BEAR" or conf < 0.40:
        controls["mode"] = "DEFENSIVE"
        controls["allow_new_buys"] = False
        target_cash = min(0.80, 0.20 + max(0.0, 0.4 - conf) * 1.0)
        controls["target_cash_buffer"] = round(target_cash, 3)
        controls["max_position_pct"] = round(max(0.02, base_max_pos * (0.5 * (conf + 0.1))), 4)
        controls["suggested_allocation_scale"] = round(max(0.1, conf + 0.1), 3)
    elif label == "NEUTRAL" or 0.40 <= conf < 0.60:
        controls["mode"] = "NORMAL"
        controls["allow_new_buys"] = conf >= 0.45
        controls["target_cash_buffer"] = round(max(base_cash, 0.05 - (conf - 0.45) * 0.05), 3) if conf < 0.5 else round(base_cash, 3)
        controls["max_position_pct"] = round(max(0.03, base_max_pos * (0.8 + (conf - 0.45))), 4)
        controls["suggested_allocation_scale"] = round(max(0.3, conf + 0.2), 3)
    else:
        controls["mode"] = "AGGRESSIVE"
        controls["allow_new_buys"] = True
        controls["target_cash_buffer"] = round(max(0.0, base_cash * (1.0 - (conf - 0.6) * 0.5)), 3)
        controls["max_position_pct"] = round(min(0.20, base_max_pos * (1.0 + (conf - 0.6))), 4)
        controls["suggested_allocation_scale"] = round(1.0 + (conf - 0.6) * 0.5, 3)
    return controls


def generate_holdings_action(current_holdings: dict, controls: dict) -> dict:
    target_cash = controls["target_cash_buffer"]
    max_pos = controls["max_position_pct"]
    pct_scale = controls["suggested_allocation_scale"]
    total_exposure = sum(v.get("position_pct", 0.0) for v in current_holdings.values())
    current_cash = max(0.0, 1.0 - total_exposure)
    desired_cash = target_cash
    cash_to_raise = max(0.0, desired_cash - current_cash)
    results = {"overall_action": "HOLD", "cash_to_raise": round(cash_to_raise, 4), "per_ticker": {}}
    per = results["per_ticker"]
    if cash_to_raise <= 1e-6 and all(v.get("position_pct", 0.0) <= max_pos + 1e-9 for v in current_holdings.values()):
        results["overall_action"] = "HOLD"
        for t, v in current_holdings.items():
            per[t] = {"current_pct": v.get("position_pct", 0.0), "target_pct": min(v.get("position_pct", 0.0) * pct_scale, max_pos), "action": "HOLD"}
        return results
    exposure = total_exposure if total_exposure > 0 else 1.0
    s = max(0.0, (1.0 - desired_cash) / exposure)
    results["overall_action"] = "PARTIAL_REDUCE" if s < 0.99 else "HOLD"
    for t, v in current_holdings.items():
        curr = v.get("position_pct", 0.0)
        proposed = round(min(curr * s, max_pos), 6)
        action = "HOLD" if proposed >= curr - 1e-6 else "REDUCE"
        per[t] = {"current_pct": round(curr, 6), "target_pct": proposed, "action": action}
    return results


def save_decision_report(base_out: Path, summary: dict, controls: dict, holding_actions: dict, regime_df=None):
    base_out.parent.mkdir(parents=True, exist_ok=True)
    out_json = {"summary": summary, "controls": controls, "holding_actions": holding_actions}
    (base_out).write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    rows = []
    for t, v in holding_actions.get("per_ticker", {}).items():
        rows.append({"ticker": t, **v})
    if rows:
        df = pd.DataFrame(rows)
        save_csv(df, base_out.with_suffix(".csv"))
        save_html(df, base_out.with_suffix(".html"), title=f"Regime Decision {datetime.utcnow().date()}")
    if regime_df is not None and not regime_df.empty:
        try:
            regime_df.tail(10).to_csv(base_out.with_name(base_out.stem + "_regime_tail.csv"))
        except Exception:
            pass
    logger.info("Saved regime decision to %s", base_out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--fred-key", default=None)
    p.add_argument("--n-factors", type=int, default=2)
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--fuzzy", type=int, default=0)
    p.add_argument("--out", default="outputs/regime_decision.json")
    p.add_argument("--holdings-csv", help="CSV with columns ticker,position_pct")
    return p.parse_args()


def main():
    args = parse_args()
    res = detect_market_regime(start=args.start, end=args.end, cache=args.cache, fred_api_key=args.fred_key, df_factors=args.n_factors, hmm_states=args.n_states, fuzzy_clusters=args.fuzzy, save_artifacts=True)
    summary = res.get("summary", {})
    regime_df = res.get("regime_df")
    controls = derive_portfolio_controls(summary)
    current_holdings = {}
    if args.holdings_csv:
        try:
            hdf = pd.read_csv(args.holdings_csv)
            for _, r in hdf.iterrows():
                t = str(r["ticker"]).strip()
                pct = float(r.get("position_pct", 0.0))
                current_holdings[t] = {"position_pct": pct}
        except Exception:
            logger.exception("Failed to load holdings CSV %s", args.holdings_csv)
    holding_actions = generate_holdings_action(current_holdings, controls)
    out_path = Path(args.out)
    save_decision_report(out_path, summary, controls, holding_actions, regime_df=regime_df)
    print(json.dumps({"summary": summary, "controls": controls, "holding_summary": {"total_holdings": len(current_holdings), "current_exposure": round(sum(v["position_pct"] for v in current_holdings.values()),4)}}, indent=2))


if __name__ == "__main__":
    main()
