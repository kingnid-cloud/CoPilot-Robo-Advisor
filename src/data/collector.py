import yfinance as yf
import pandas as pd
from pathlib import Path
import time

class DataCollector:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_price(self, ticker, period="2y", interval="1d", force=False):
        cache_file = self.cache_dir / f"{ticker}_price.parquet"
        if cache_file.exists() and not force:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            df.to_parquet(cache_file)
        time.sleep(0.1)
        return df
