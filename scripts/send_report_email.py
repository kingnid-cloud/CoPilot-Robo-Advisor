#!/usr/bin/env python3
"""
Send a report file via SMTP (Gmail-friendly). Supports multiple secret names:
Preferred: GMAIL_USER, GMAIL_APP_PASSWORD
Fallbacks: EMAIL_SMTP_USER, EMAIL_SMTP_PASS, EMAILAPPPASSWORD
Usage:
  python scripts/send_report_email.py --file reports/daily_screening.csv --subject "CoPilot Stock Report"
"""

import os
import sys
import argparse
import smtplib
from email.message import EmailMessage
from pathlib import Path

def get_smtp_credentials():
    # Preferred Gmail secrets
    user = os.getenv("GMAIL_USER") or os.getenv("EMAIL_SMTP_USER")
    # Accept multiple possible secret names for the app password
    pwd = (
        os.getenv("GMAIL_APP_PASSWORD")
        or os.getenv("EMAILAPPPASSWORD")
        or os.getenv("EMAIL_SMTP_PASS")
    )
    return user, pwd

def send_email(smtp_host, smtp_port, smtp_user, smtp_pass, from_addr, to_addr, subject, body, attachment_path):
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body or "Please find the attached report.")

    if attachment_path:
        p = Path(attachment_path)
        if not p.exists():
            raise FileNotFoundError(f"Attachment not found: {attachment_path}")
        data = p.read_bytes()
        maintype = "application"
        subtype = "octet-stream"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=p.name)

    port = int(smtp_port) if smtp_port else 587
    with smtplib.SMTP(smtp_host, port, timeout=60) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to report file to attach")
    parser.add_argument("--subject", default="CoPilot Stock Report", help="Email subject")
    parser.add_argument("--body", default="", help="Email body text")
    args = parser.parse_args()

    smtp_host = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
    smtp_port = os.getenv("EMAIL_SMTP_PORT", "587")
    smtp_user_env, smtp_pass_env = get_smtp_credentials()
    from_email = os.getenv("FROM_EMAIL") or smtp_user_env
    to_email = os.getenv("TO_EMAIL")

    missing = [k for k,v in [
        ("smtp_user", smtp_user_env),
        ("smtp_pass", smtp_pass_env),
        ("to_email", to_email),
        ("from_email", from_email)
    ] if not v]
    if missing:
        print("Missing required environment variables:", ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    try:
        send_email(
            smtp_host, smtp_port, smtp_user_env, smtp_pass_env,
            from_email, to_email, args.subject, args.body, args.file
        )
        print("Email sent successfully to", to_email)
    except Exception as e:
        print("Failed to send email:", str(e), file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
