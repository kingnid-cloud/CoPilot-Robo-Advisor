# src/utils/secrets.py
"""
Secrets loader for CI and local runs.

- EXPECTED_SECRETS lists all environment variables the pipeline may read.
- load_secrets() returns a dict of present secrets (non-empty).
- missing_secrets(secrets) returns a list of expected names that are missing or empty.
- safe_get(secrets, name, default=None) returns value without raising; useful for optional keys.

IMPORTANT: Do NOT store secret values in source control. Use GitHub Secrets (Settings -> Secrets and variables -> Actions).
"""

import os
from typing import Dict, List, Optional

# All expected secret names used across the pipeline and connectors
EXPECTED_SECRETS = [
    # Email / SMTP
    "EMAIL_SMTP_USER",
    "EMAIL_SMTP_PASS",
    "GMAIL_USER",
    "GMAIL_APP_PASSWORD",
    "TO_EMAIL",

    # Market data / fundamentals / alternative data APIs
    "ALPHAVANTAGE_KEY",
    "FMP_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
    "SIMFIN_API_KEY",
    "ROIC_API_KEY",
    "TICKERBOT_API_KEY",

    # Brokerage / test accounts (if used)
    "RH_TEST_ACCOUNT_NUMBER",
    "RH_TEST_EMAIL",
    "RH_TEST_PASSWORD",

    # Feature flags / runtime toggles
    "ENABLE_LIVE",          # "true" or "false"
    "ENABLE_EMAIL",         # "true" or "false"

    # Optional monitoring / observability
    "SENTRY_DSN",
    "LOGDNA_KEY"
]

# Grouping for convenience (not required, but helpful)
OPTIONAL_SECRETS = [
    "SENTRY_DSN",
    "LOGDNA_KEY",
    "SIMFIN_API_KEY",
    "TICKERBOT_API_KEY"
]

def load_secrets() -> Dict[str, str]:
    """
    Return a dict of secret_name -> value for all EXPECTED_SECRETS that are present and non-empty.
    """
    out: Dict[str, str] = {}
    for name in EXPECTED_SECRETS:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            out[name] = val
    return out

def missing_secrets(secrets: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Return a list of expected secret names that are missing or empty.
    If `secrets` is provided, it will be used as the source; otherwise load_secrets() is called.
    """
    if secrets is None:
        secrets = load_secrets()
    missing = [n for n in EXPECTED_SECRETS if not secrets.get(n)]
    return missing

def safe_get(secrets: Dict[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safe accessor for a secret value; returns default if missing.
    """
    return secrets.get(name) or default
