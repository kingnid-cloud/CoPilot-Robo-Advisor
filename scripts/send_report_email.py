#!/usr/bin/env python3
"""
Simple SMTP Gmail sender used by the workflow.
Usage:
  python scripts/send_report_email.py --subject "Subject" --body "Body text" --file /dev/null
If file is /dev/null or omitted, no attachment is sent.
Environment variables:
  GMAIL_USER, GMAIL_APP_PASSWORD, FROM_EMAIL, TO_EMAIL
"""
import os
import argparse
import smtplib
from email.message import EmailMessage

def send_email(subject: str, body: str, attachment_path: str | None = None):
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    from_email = os.environ.get("FROM_EMAIL") or gmail_user
    to_email = os.environ.get("TO_EMAIL")

    if not gmail_user or not gmail_pass or not to_email:
        raise SystemExit("Missing GMAIL_USER, GMAIL_APP_PASSWORD, or TO_EMAIL environment variables")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    if attachment_path and attachment_path != "/dev/null":
        try:
            with open(attachment_path, "rb") as f:
                data = f.read()
            import mimetypes
            ctype, encoding = mimetypes.guess_type(attachment_path)
            maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(attachment_path))
        except Exception as e:
            print(f"Warning: failed to attach file {attachment_path}: {e}")

    # Send via Gmail SMTP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.send_message(msg)
    print(f"Email sent successfully to {to_email}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--file", default="/dev/null")
    args = parser.parse_args()
    send_email(args.subject, args.body, args.file)
