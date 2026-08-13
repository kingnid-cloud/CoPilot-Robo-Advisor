from src.scoring.fundamentals import score_placeholder
from src.scoring.technicals import score_from_price
import pandas as pd

def test_fund_score():
    s = score_placeholder("AAPL")
    assert "score" in s

def test_tech_score():
    # create minimal price DataFrame
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    df = pd.DataFrame({"Close": [100 + i*0.1 for i in range(60)],
                       "High": [100 + i*0.1 for i in range(60)],
                       "Low": [99 + i*0.1 for i in range(60)]}, index=dates)
    s = score_from_price(df)
    assert "score" in s
