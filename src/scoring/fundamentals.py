# Placeholder fundamental scorer. Replace with real API calls (FMP/AlphaVantage) later.
def score_placeholder(ticker):
    # Simple deterministic placeholder score for demo
    base = sum(ord(c) for c in ticker) % 100
    score = (base / 100.0) * 5.0
    completeness = 0.8
    return {"score": score, "completeness": completeness}
