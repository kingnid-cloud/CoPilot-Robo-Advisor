# src/scoring/technicals.py
"""
Technical scorer using the 'ta' package (not pandas_ta).
Computes simple moving averages and RSI and returns a small score dict.
"""

import pandas as pd
from ta.momentum import RSIIndicator
from typing import Dict
import logging

logger = logging.getLogger("TechnicalScorer")

class TechnicalScorer:
    def __init__(self):
        pass

    def score_from_price(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Input: price DataFrame with a 'Close' column.
        Output: dict with keys 'score' and 'signal'.
        """
        if df is None or df.empty:
            return {"score": 0.0, "signal": 0.0}

        close = df["Close"].dropna()
        if len(close) < 50:
            return {"score": 0.0, "signal": 0.0}

        # Simple moving averages
        ma50 = close.rolling(window=50).mean().iloc[-1]
        ma200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else ma50

        # RSI using 'ta'
        try:
            rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
        except Exception:
            rsi = 50.0

        score = 0.0
        if ma50 > ma200:
            score += 2.0
        if rsi < 70:
            score += 1.0

        signal = float(ma50 - ma200)
        return {"score": float(score), "signal": signal}
