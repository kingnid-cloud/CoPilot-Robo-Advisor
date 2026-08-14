# README: Manual Backtest Run

This repository contains a GitHub Actions workflow to run the walk-forward backtest and produce a
Strategy vs SPY cumulative returns plot for a specified date range.

How to run the workflow in GitHub UI
1. Go to your repository on GitHub: https://github.com/kingnid-cloud/CoPilot-Robo-Advisor
2. Click Actions
3. Find the workflow titled "Manual Backtest Run"
4. Click the workflow, then click "Run workflow"
5. Optionally override the inputs (start, end, train_years, test_months). Defaults: 2000-01-01 -> 2025-01-01, train=5, test=6
6. Click the green "Run workflow" button

Artifacts
- When the run completes, download the artifact named `backtest-artifacts`. It contains:
  - outputs/backtest.json
  - outputs/backtest.csv
  - outputs/backtest.png

Secrets
- Ensure FRED_API_KEY is set in Settings → Secrets → Actions for best regime signals.
- Optional keys to improve data completeness: FINNHUB_API_KEY, FMP_KEY, ALPHAVANTAGE_KEY, TICKERBOT_API_KEY, ROIC_API_KEY, SIMFIN_API_KEY, NEWSAPI_KEY
