# src/utils/normalize.py
import pandas as pd

def percentile_within_group(df: pd.DataFrame, group_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    """
    Compute percentile rank of value_col within each group_col and store in out_col (0-1).
    """
    def pct(s):
        return s.rank(method="average", pct=True)
    df[out_col] = df.groupby(group_col)[value_col].transform(pct)
    df[out_col] = df[out_col].fillna(0.0).astype(float).clip(0.0, 1.0)
    return df
