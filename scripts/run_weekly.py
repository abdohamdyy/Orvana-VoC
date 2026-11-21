"""Run the weekly job once, now (what the scheduler does every Monday 08:00 Cairo time).

    python scripts/run_weekly.py            # process + email every pending file in incoming/
    python scripts/run_weekly.py --no-send  # build the reports only
    python scripts/run_weekly.py --force    # reclassify and resend (after a manual check)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.report.weekly import run_pending  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
ap.add_argument("--no-send", action="store_true")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for r in run_pending(force=args.force, send_email=not args.no_send):
    print(json.dumps(r, ensure_ascii=False))
