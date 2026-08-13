# src/run_daily.py (top of file)
import argparse
import yaml
from pathlib import Path
import pandas as pd
import datetime
import logging
import os

# create logger first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_daily")

# load secrets helper (ensure this file exists: src/utils/secrets.py)
try:
    from src.utils.secrets import load_secrets, missing_secrets
except Exception:
    # If the helper is missing, create a minimal fallback to avoid crashing
    def load_secrets():
        return {}
    def missing_secrets(env):
        return []

# load secrets from environment and warn only about missing names (do NOT log values)
secrets = load_secrets()
missing = missing_secrets(secrets)
if missing:
    logger.warning("The following repository secrets are missing or empty: %s", missing)
else:
    logger.info("All expected secrets are present.")

# Use secrets['ALPHAVANTAGE_KEY'] etc. when calling API clients

from src.data.collector import DataCollector
from src.scoring.fundamentals import FundamentalScorer
from src.scoring.technicals import TechnicalScorer
from src.strategist.allocator import Allocator
from src.backtest.walkforward import WalkForwardBacktester
from src.utils.io import ensure_dirs, save_csv, save_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_daily")

def run(config_path):
    cfg = yaml.safe_load(open(config_path))
    out_dir = Path(cfg.get("report", {}).get("output_dir", "reports"))
    ensure_dirs([out_dir, "cache"])
    today = datetime.date.today().isoformat()
    logger.info("Starting pipeline: %s", today)

    # 1) Data collection (small demo universe)
    tickers = ["AAPL","MSFT","GOOGL","AMZN","JNJ","JPM","PG","NVDA","TSLA","META"]
    dc = DataCollector(cache_dir=cfg["data"]["cache_dir"])
    price_data = {t: dc.fetch_price(t, period=cfg["data"]["price_period"], interval=cfg["data"]["price_interval"]) for t in tickers}

    # 2) Scoring
    fund_scorer = FundamentalScorer()
    tech_scorer = TechnicalScorer()
    fund_scores = {}
    tech_scores = {}
    for t, df in price_data.items():
        fund_scores[t] = fund_scorer.score_placeholder(t)
        tech_scores[t] = tech_scorer.score_from_price(df)

    # 3) Combine and allocate
    allocator = Allocator(cfg)
    combined_df = allocator.combine_and_allocate(fund_scores, tech_scores)

    # 4) Backtest (very small demo)
    backtester = WalkForwardBacktester()
    backtest_report = backtester.simple_report(combined_df)

    # 5) Save reports
    csv_path = out_dir / f"final_portfolio_{today}.csv"
    html_path = out_dir / f"final_portfolio_{today}.html"
    save_csv(combined_df, csv_path)
    save_html(combined_df, html_path, title=f"CoPilot Robo Advisor {today}")

    logger.info("Pipeline finished. Reports: %s, %s", csv_path, html_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)
