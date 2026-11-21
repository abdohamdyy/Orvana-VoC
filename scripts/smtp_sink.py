"""Tiny local SMTP server that writes every received email to a folder (no real mail server needed).

    python scripts/smtp_sink.py [--port 1025] [--dir outputs/mailbox]

The Docker setup uses Mailpit (web UI on :8025) for the same purpose; this script is the
dependency-free fallback used when Docker is not available. Uses the stdlib `smtpd` module
(Python <= 3.11).
"""
import argparse
import asyncore
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
import smtpd  # noqa: E402


class Sink(smtpd.SMTPServer):
    def __init__(self, addr, out: Path):
        super().__init__(addr, None, decode_data=False)
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        path = self.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{len(list(self.out.glob('*.eml'))):03d}.eml"
        path.write_bytes(data)
        print(f"received mail from {mailfrom} to {rcpttos} -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1025)
    ap.add_argument("--dir", default="outputs/mailbox")
    args = ap.parse_args()
    Sink(("127.0.0.1", args.port), Path(args.dir))
    print(f"SMTP sink listening on 127.0.0.1:{args.port}, writing to {args.dir}", flush=True)
    asyncore.loop()
