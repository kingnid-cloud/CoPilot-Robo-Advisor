import pandas as pd

class Allocator:
    def __init__(self, cfg):
        self.cfg = cfg

    def combine_and_allocate(self, fund_scores, tech_scores):
        # fund_scores, tech_scores: dict[ticker] -> dict
        rows = []
        for t in fund_scores.keys():
            f = fund_scores.get(t, {})
            te = tech_scores.get(t, {})
            fund_score = f.get("score", 0.0)
            tech_score = te.get("score", 0.0)
            combined = 0.6 * fund_score + 0.4 * tech_score
            completeness = f.get("completeness", 0.0)
            rows.append({"ticker": t, "fund_score": fund_score, "tech_score": tech_score,
                         "combined": combined, "completeness": completeness})
        df = pd.DataFrame(rows).set_index("ticker").sort_values("combined", ascending=False)
        N = max(self.cfg["allocation"]["min_stocks"], min(self.cfg["allocation"]["max_stocks"], len(df)))
        pick = df.head(N).copy()
        pick["suggested_allocation"] = 1.0 / N
        # enforce max position
        max_pos = self.cfg["allocation"]["max_position_pct"]
        pick["suggested_allocation"] = pick["suggested_allocation"].clip(upper=max_pos)
        # normalize to 1 - cash_buffer
        total = pick["suggested_allocation"].sum()
        cash_buffer = self.cfg["allocation"].get("cash_buffer", 0.02)
        pick["suggested_allocation"] = pick["suggested_allocation"] * ((1.0 - cash_buffer) / total)
        return pick
