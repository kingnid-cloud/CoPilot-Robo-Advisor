# src/scoring/fundamentals.py

"""
Placeholder fundamental scorer.

This file provides a simple class interface used by src/run_daily.py:
    from src.scoring.fundamentals import FundamentalScorer
    fund_scorer = FundamentalScorer()
    fund_scorer.score_placeholder("AAPL")
Replace or extend this class later with real API calls (FMP, AlphaVantage, SimFin).
"""

from typing import Dict

class FundamentalScorer:
    def __init__(self):
        # any initialization for real connectors can go here
        pass

    def score_placeholder(self, ticker: str) -> Dict[str, float]:
        """
        Deterministic placeholder score for demo and testing.
        Returns a dict with keys: score, completeness
        """
        base = sum(ord(c) for c in ticker) % 100
        score = (base / 100.0) * 5.0
        completeness = 0.8
        return {"score": float(score), "completeness": float(completeness)}
