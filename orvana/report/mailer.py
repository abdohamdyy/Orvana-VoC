"""Build and send the weekly email. Locally the SMTP server is Mailpit (web UI on :8025), so the
emails can be inspected without a real mail server. Credentials come only from the environment."""
import csv
import hashlib
import io
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

from ..config import settings

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value) -> str:
    """Neutralise CSV/Excel formula injection: a complaint starting with '=' must not run in Excel."""
    s = "" if value is None else str(value)
    return "'" + s if s.startswith(_FORMULA_START) else s


def urgent_csv(rows: list[dict]) -> bytes:
    cols = ["ticket_id", "created_at", "channel", "category", "product_mentioned", "store_mentioned", "reason", "text"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([csv_safe(r.get(c, "")) for c in cols])
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel shows Arabic correctly


def build_message(week_key: str, period: dict, html: str, text: str, urgent_rows: list[dict]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = f"Orvana weekly complaints report {period['start']} to {period['end']}"
    msg["From"] = settings.mail_from
    msg["To"] = settings.mail_to
    msg["Date"] = formatdate(localtime=True)
    # deterministic per (week, content): an accidental resend of the same report collapses in the inbox, but a
    # corrected report gets a new id (Gmail silently drops a message whose Message-ID it has already received).
    # Duplicate sends are prevented by the state DB (State.claim_email), not by this header.
    msg["Message-ID"] = f"<voc-{week_key}-{hashlib.sha256(html.encode('utf-8')).hexdigest()[:12]}@orvana.example>"
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    msg.add_attachment(urgent_csv(urgent_rows), maintype="text", subtype="csv", filename="urgent_tickets.csv")
    return msg


def save_eml(msg: EmailMessage, week_key: str, html: str) -> Path:
    out = settings.outputs_dir / "emails"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{week_key}.html").write_text(html, encoding="utf-8")
    path = out / f"{week_key}.eml"
    path.write_bytes(bytes(msg))
    return path


def send(msg: EmailMessage) -> None:
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
        if settings.smtp_starttls:
            # an explicit context: starttls() without one does NOT verify the server certificate, so a
            # man-in-the-middle could read the SMTP password
            smtp.starttls(context=ssl.create_default_context())
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)
