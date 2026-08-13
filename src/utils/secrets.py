# src/utils/secrets.py
import os
from typing import Dict, List

EXPECTED_SECRETS = [
    "ALPHAVANTAGE_KEY",
    "FMP_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
    "SimFin",
    "ROIC_API_KEY",
    "TICKERBOT_API_KEY",
    "EmailAppPassword",
    "GMAIL_APP_PASSWORD",
    "GMAIL_USER",
    "RH_TEST_ACCOUNT_NUMBER",
    "RH_TEST_EMAIL",
    "RH_TEST_PASSWORD",
    "TO_EMAIL",
    "EMAIL_SMTP_USER",
    "EMAIL_SMTP_PASS",
    "ENABLE_LIVE"
]

def load_secrets() -> Dict[str, str]:
    """
    Return a dict mapping expected secret names to their environment values (or None).
    Do NOT log secret values. Only use presence/absence checks in logs.
    """
    env = {k: os.environ.get(k) for k in EXPECTED_SECRETS}
    return env

def missing_secrets(env: Dict[str, str]) -> List[str]:
    """Return list of secret names that are missing (value is None or empty)."""
    return [k for k, v in env.items() if not v]
