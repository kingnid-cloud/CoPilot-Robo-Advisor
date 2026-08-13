import pandas_ta as ta
import pandas as pd

def score_from_price(df):
    if df is None or df.empty:
        return {"score": 0.0, "signal": 0.0}
    close = df['Close'].dropna()
    if len(close) < 50:
        return {"score": 0.0, "signal": 0.0}
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50
    rsi = ta.rsi(close, length=14).iloc[-1]
    score = 0.0
    if ma50 > ma200:
        score += 2.0
    if rsi < 70:
        score += 1.0
    return {"score": score, "signal": (ma50 - ma200)}
