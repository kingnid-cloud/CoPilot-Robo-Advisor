#!/usr/bin/env python3
"""
Read the first report in a reports/ folder and produce a concise summary.
Prints the email body to stdout and calls send_report_email.py to send it.
Usage:
  python scripts/format_and_email_summary.py --path reports/ --subject "CoPilot Stock Report"
This script is tolerant of common column name variants.
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import subprocess
import math

# Column mapping helpers
POSSIBLE_COLUMNS = {
    "ticker": ["ticker", "symbol", "tick"],
    "pass_hard_filters": ["pass_hard_filters", "pass_filters", "pass_filters_bool", "passed"],
    "score": ["score", "rank_score", "ranking"],
    "confidence": ["confidence", "conf"],
    "allocation": ["allocation", "suggested_allocation", "alloc_pct"],
    "sector": ["sector", "industry", "gics_sector"],
    "expected_return": ["expected_return", "exp_return", "one_year_return", "expected_1y_return"]
}

def find_column(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

def load_first_report(path):
    p = Path(path)
    if not p.exists():
        return None, "reports path does not exist"
    files = list(p.glob("*.csv")) + list(p.glob("*.json")) + list(p.glob("*.parquet")) + list(p.glob("*.xlsx"))
    if not files:
        return None, "no report files found in reports/"
    f = files[0]
    try:
        if f.suffix == ".csv":
            df = pd.read_csv(f)
        elif f.suffix == ".json":
            df = pd.read_json(f)
        elif f.suffix == ".parquet":
            df = pd.read_parquet(f)
        elif f.suffix == ".xlsx":
            df = pd.read_excel(f)
        else:
            return None, f"unsupported file type: {f.suffix}"
    except Exception as e:
        return None, f"failed to read {f.name}: {e}"
    return (df, f.name), None

def format_percent(x):
    try:
        if pd.isna(x):
            return "N/A"
        if abs(x) <= 1 and not (x > 1 and x < 100):
            return f"{x*100:.1f}%"
        return f"{x:.1f}%" if isinstance(x, (int,float)) else str(x)

    except Exception:
        return str(x)

def build_summary(df, filename):
    # map columns
    col_map = {}
    for canonical, candidates in POSSIBLE_COLUMNS.items():
        found = find_column(df, candidates)
        col_map[canonical] = found

    total_screened = len(df)
    pass_col = col_map["pass_hard_filters"]
    if pass_col:
        passing = df[df[pass_col].astype(bool)]
    else:
        # fallback: treat top N by score if no pass column
        score_col = col_map["score"]
        if score_col:
            passing = df.nlargest(10, score_col)
        else:
            passing = df.head(0)

    num_passing = len(passing)

    # Final portfolio: assume passing rows are the final portfolio if allocation present, else top 10 by score
    alloc_col = col_map["allocation"]
    if alloc_col and num_passing>0:
        final = passing.copy()
    else:
        score_col = col_map["score"]
        if score_col:
            final = df.nlargest(min(10, max(1, num_passing)), score_col)
        else:
            final = df.head(min(10, num_passing))

    # Build final portfolio table rows
    rows = []
    for _, r in final.iterrows():
        ticker = r.get(col_map["ticker"]) if col_map["ticker"] else "N/A"
        score = r.get(col_map["score"]) if col_map["score"] else "N/A"
        conf = r.get(col_map["confidence"]) if col_map["confidence"] else "N/A"
        alloc = r.get(alloc_col) if alloc_col else None
        rows.append({
            "ticker": str(ticker),
            "score": f"{float(score):.3f}" if pd.notna(score) and isinstance(score, (int,float)) else str(score),
            "confidence": f"{float(conf):.2f}" if pd.notna(conf) and isinstance(conf, (int,float)) else str(conf),
            "allocation": format_percent(alloc) if alloc is not None else "N/A"
        })

    # Sector allocations
    sector_col = col_map["sector"]
    if sector_col:
        sector_alloc = (final.groupby(sector_col).apply(lambda g: g[col_map["allocation"]].sum() if col_map["allocation"] else len(g))
                        .sort_values(ascending=False).to_dict())
        # format percentages if allocations are numeric and sum > 0
        total_alloc = sum(v for v in sector_alloc.values() if isinstance(v, (int,float)) and not math.isnan(v))
        formatted_sector = {}
        for k,v in sector_alloc.items():
            if isinstance(v, (int,float)) and total_alloc>0:
                formatted_sector[k] = f"{(v/total_alloc)*100:.1f}%"
            else:
                formatted_sector[k] = str(v)
    else:
        formatted_sector = {}

    # Expected return and risk
    exp_col = col_map["expected_return"]
    if exp_col:
        exp_vals = final[exp_col].dropna().astype(float) if not final.empty else pd.Series(dtype=float)
        expected_1y = f"{exp_vals.mean()*100:.1f}%" if not exp_vals.empty else "N/A"
        risk = f"{exp_vals.std()*100:.1f}%" if not exp_vals.empty else "N/A"
    else:
        expected_1y = "N/A"
        risk = "N/A"

    # Build body
    lines = []
    lines.append(f"CoPilot Stock Report — {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"Total stocks screened: {total_screened}")
    lines.append(f"Number passing hard filters: {num_passing}")
    lines.append("")
    lines.append("Final portfolio")
    lines.append("Ticker | Score | Confidence | Suggested Allocation")
    for r in rows:
        lines.append(f"{r['ticker']} | {r['score']} | {r['confidence']} | {r['allocation']}")
    lines.append("")
    lines.append("Portfolio sector allocations")
    if formatted_sector:
        for k,v in formatted_sector.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("N/A")
    lines.append("")
    lines.append(f"Expected 1-year return (avg): {expected_1y}")
    lines.append(f"Risk assessment (std dev): {risk}")
    body = "\n".join(lines)
    return body

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    (df_and_name, filename), err = load_first_report(args.path)
    if err:
        body = f"No report found or failed to read report: {err}"
        print(body)
        # call email sender with no attachment
        subprocess.run([sys.executable, "scripts/send_report_email.py", "--subject", args.subject, "--body", body, "--file", "/dev/null"], check=False)
        return

    df, filename = df_and_name
    body = build_summary(df, filename)
    print(body)
    # send email
    subprocess.run([sys.executable, "scripts/send_report_email.py", "--subject", args.subject, "--body", body, "--file", "/dev/null"], check=True)

if __name__ == "__main__":
    main()
