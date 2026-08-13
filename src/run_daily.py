# src/run_daily.py
"""
Main pipeline entrypoint for daily run.
- Loads config.yaml
- Loads secrets via src.utils.secrets
- Uses DataCollector to fetch tickers, prices, fundamentals
- Runs TechnicalScorer and FundamentalScorer
- Writes CSV and HTML reports to reports/
- Optionally sends email when ENABLE_EMAIL is true and secrets present
"""
import argparse
import logging
import os
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from src.utils.secrets import load_secrets, missing_secrets, safe_get
from src.data.collector import DataCollector
from src.scoring.technicals import TechnicalScorer
from src.scoring.fundamentals import FundamentalScorer
import smtplib, ssl
from email.message import EmailMessage

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_daily")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def send_email(smtp_user: str, smtp_pass: str, to_addr: str, subject: str, body: str, smtp_host="smtp.gmail.com", smtp_port=587):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        s.ehlo()
        s.starttls(context=context)
        s.ehlo()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
    logger.info("Email sent to %s", to_addr)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def build_reports(final_df: pd.DataFrame, run_date: str) -> (Path, Path):
    csv_path = REPORT_DIR / f"final_portfolio_{run_date}.csv"
    html_path = REPORT_DIR / f"final_portfolio_{run_date}.html"
    final_df.to_csv(csv_path, index=False)
    final_df.to_html(html_path, index=False)
    logger.info("Reports: %s, %s", csv_path, html_path)
    return csv_path, html_path

def run(config_path: str):
    cfg = load_config(config_path)
    secrets = load_secrets()
    missing = missing_secrets(secrets)
    if missing:
        logger.warning("The following repository secrets are missing or empty: %s", missing)
    else:
        logger.info("All expected secrets present.")

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    logger.info("Starting pipeline: %s", run_date)

    # Data collector
    dc = DataCollector()

    # tickers
    tickers = dc.fetch_tickers(Path(cfg.get("data", {}).get("tickers_csv", "")) if cfg.get("data", {}).get("tickers_csv") else None)

    # fetch prices (bulk)
    price_period = cfg.get("data", {}).get("price_period", "3y")
    price_interval = cfg.get("data", {}).get("price_interval", "1d")
    price_data = dc.fetch_prices_bulk(tickers, period=price_period, interval=price_interval)

    # fetch fundamentals
    fundamentals_df = dc.fetch_fundamentals(tickers)
    fundamentals_df = fundamentals_df.reset_index().rename(columns={"index": "ticker"})

    # prepare metrics DataFrame for scoring
    metrics_df = fundamentals_df.copy()
    # ensure sector column exists for percentile grouping
    if "sector" not in metrics_df.columns:
        metrics_df["sector"] = cfg.get("defaults", {}).get("sector", "GLOBAL")

    # scoring
    tech = TechnicalScorer()
    fund = FundamentalScorer(cfg)

    # compute technical scores per ticker
    tech_scores = []
    for t in tickers:
        df_price = price_data.get(t, pd.DataFrame())
        ts = tech.score_from_price(df_price)
        tech_scores.append({"ticker": t, "tech_score": ts.get("score", 0.0), "tech_signal": ts.get("signal", 0.0)})
    tech_df = pd.DataFrame(tech_scores)

    # compute fundamental scores
    fund_scores_df = fund.score_company_metrics(metrics_df)

    # merge results
    merged = fund_scores_df.reset_index().rename(columns={"index": "ticker"}).merge(tech_df, on="ticker", how="left")
    merged["tech_score"] = merged["tech_score"].fillna(0.0)
    merged["final_score"] = merged["fund_score"] * cfg.get("weights", {}).get("fund", 0.7) + merged["tech_score"] * cfg.get("weights", {}).get("tech", 0.3)

    # simple portfolio selection: top N by final_score
    top_n = cfg.get("portfolio", {}).get("top_n", 10)
    final_portfolio = merged.sort_values("final_score", ascending=False).head(top_n)
    final_portfolio = final_portfolio.reset_index(drop=True)

    # write reports
    csv_path, html_path = build_reports(final_portfolio, run_date)

    # optionally send email
    enable_email = str(safe_get(secrets, "ENABLE_EMAIL", cfg.get("email", {}).get("enabled", "false"))).lower() == "true"
    smtp_user = safe_get(secrets, "EMAIL_SMTP_USER", cfg.get("email", {}).get("from_addr"))
    smtp_pass = safe_get(secrets, "EMAIL_SMTP_PASS", safe_get(secrets, "GMAIL_APP_PASSWORD"))
    to_addr = safe_get(secrets, "TO_EMAIL", cfg.get("email", {}).get("to_addr"))

    if enable_email:
        if not smtp_user or not smtp_pass or not to_addr:
            logger.error("Email enabled but SMTP credentials or TO_EMAIL missing. SMTP_USER=%s TO_EMAIL=%s", bool(smtp_user), bool(to_addr))
        else:
            subject = f"Robo Advisor Report {run_date}"
            body = f"Attached are the reports for {run_date}.\nCSV: {csv_path}\nHTML: {html_path}"
            try:
                send_email(smtp_user, smtp_pass, to_addr, subject, body, smtp_host=cfg.get("email", {}).get("smtp_host", "smtp.gmail.com"), smtp_port=cfg.get("email", {}).get("smtp_port", 587))
            except Exception as e:
                logger.exception("Email send failed: %s", e)
    else:
        logger.info("Email disabled by ENABLE_EMAIL flag or config.")

    logger.info("Pipeline finished. Reports: %s, %s", csv_path, html_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)
