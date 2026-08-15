import json
import os
from pathlib import Path

def load_cached_series(ticker: str, start: str, end: str, cache_dir: Path):
    cache_file = cache_dir / f"{ticker}_{start}_{end}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_series(ticker: str, start: str, end: str, cache_dir: Path, data):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f)
