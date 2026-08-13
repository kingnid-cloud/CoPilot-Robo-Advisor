# scripts/smtp_test.py
import os, ssl, smtplib
from email.message import EmailMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smtp_test")

smtp_user = os.environ.get("EMAIL_SMTP_USER") or os.environ.get("GMAIL_USER")
smtp_pass = os.environ.get("EMAIL_SMTP_PASS") or os.environ.get("GMAIL_APP_PASSWORD")
to_addr = os.environ.get("TO_EMAIL")

if not smtp_user or not smtp_pass or not to_addr:
    logger.error("Missing env: EMAIL_SMTP_USER, EMAIL_SMTP_PASS/GMAIL_APP_PASSWORD, or TO_EMAIL")
    raise SystemExit(1)

msg = EmailMessage()
msg["Subject"] = "Robo Advisor SMTP test"
msg["From"] = smtp_user
msg["To"] = to_addr
msg.set_content("This is a test email from the GitHub Actions pipeline.")

try:
    logger.info("Connecting to smtp.gmail.com:587")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
        s.set_debuglevel(1)
        s.ehlo()
        s.starttls(context=ssl.create_default_context())
        s.ehlo()
        logger.info("Logging in as %s", smtp_user)
        s.login(smtp_user, smtp_pass)
        logger.info("Sending message to %s", to_addr)
        s.send_message(msg)
    logger.info("SMTP test succeeded: email sent to %s", to_addr)
except Exception as e:
    logger.exception("SMTP test failed: %s", e)
    raise
