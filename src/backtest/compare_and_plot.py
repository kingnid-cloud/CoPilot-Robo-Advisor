from __future__ import annotations
import argparse
import json
from pathlib import Path
import logging
import math

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# robust import: bind the actual run function exported by src.backtest.walkforward
import importlib

wf_mod = importlib.import_module("src.backtest.walkforward")
if hasattr(wf_mod, "run_walk_forward"):
    run_walk_forward = wf_mod.run_walk_forward
elif hasattr(wf_mod, "run_walkforward"):
    run_walk_forward = wf_mod.run_walkforward
elif hasattr(wf_mod, "run"):
    run_walk_forward = wf_mod.run
else:
    raise ImportError(
        "walkforward module does not expose a run function; looked for: "
        "run_walk_forward, run_walkforward, run"
    )

logger = logging.getLogger("compare_backtest")
logging.basicConfig(level=logging.INFO)

def compare_and_plot(tickers, start, end, train_years, test_months, fred_key=None, out_prefix=Path("outputs/backtest")):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = run_walk_forward(tickers, start=start, end=end, train_years=train_years, test_months=test_months, cache_macro=True, fred_key=fred_key)
    folds = summary.get("folds", [])

    # collect cumulative series per fold using fold strategy_return and SPY test returns
    dates = []
    strat_vals = []
    spy_vals = []
    cum_strat = 1.0
    cum_spy = 1.0

    for f in folds:
        ts = f.get("test_start")
        te = f.get("test_end")
        strat_r = f.get("strategy_return")
        if strat_r is None:
            # treat as no deployment -> cash yield ~0
            strat_r = 0.0
        # compute SPY return over same period
        try:
            spy_df = yf.download("SPY", start=ts, end=pd.to_datetime(te) + pd.Timedelta(days=1), progress=False, threads=False)
            if spy_df is None or spy_df.empty:
                spy_r = 0.0
            else:
                spy_entry = spy_df['Close'].dropna().iloc[0]
                spy_exit = spy_df['Close'].dropna().iloc[-1]
                spy_r = (spy_exit / spy_entry) - 1.0
        except Exception:
            spy_r = 0.0

        cum_strat *= (1.0 + strat_r)
        cum_spy *= (1.0 + spy_r)
        dates.append(pd.to_datetime(f.get('test_end')))
        strat_vals.append(cum_strat)
        spy_vals.append(cum_spy)

    df = pd.DataFrame({'date': dates, 'strategy_cum': strat_vals, 'spy_cum': spy_vals}).set_index('date')

    # save summary and plot
    out_json = out_prefix.with_suffix('.json')
    out_png = out_prefix.with_suffix('.png')
    out_csv = out_prefix.with_suffix('.csv')

    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    df.to_csv(out_csv)

    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['strategy_cum'], label='Strategy (WF)')
    plt.plot(df.index, df['spy_cum'], label='SPY')
    plt.legend()
    plt.title(f'Walk-forward Strategy vs SPY ({start} to {end})')
    plt.ylabel('Cumulative Return (Growth of $1)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    logger.info('Saved plot to %s', out_png)
    return {'summary': summary, 'cumulative_csv': str(out_csv), 'plot_png': str(out_png)}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', nargs='*', help='Tickers list (space separated) or leave empty to default to SPY', default=['SPY'])
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end', default='2025-01-01')
    p.add_argument('--train-years', type=int, default=5)
    p.add_argument('--test-months', type=int, default=6)
    p.add_argument('--fred-key', default=None)
    p.add_argument('--out', default='outputs/backtest')
    args = p.parse_args()
    res = compare_and_plot(args.tickers, args.start, args.end, args.train_years, args.test_months, fred_key=args.fred_key, out_prefix=Path(args.out))
    print(res)
