"""Long-running scheduler (the Docker `scheduler` service): every Monday 08:00 Africa/Cairo.

    python -m orvana.scheduler

- The timezone is explicit (tzdata), so Egypt's daylight-saving changes are handled.
- A run missed while the machine was off is executed on start-up (run_on_start) or within the
  misfire grace window; running more than once is harmless because the job is idempotent.
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import settings
from .report.weekly import run_pending

log = logging.getLogger("scheduler")


def make_trigger() -> CronTrigger:
    return CronTrigger.from_crontab(settings.schedule_cron, timezone=ZoneInfo(settings.timezone))


def job():
    try:
        for r in run_pending():
            log.info("%s", r)
    except Exception:  # keep the scheduler alive; the failure is logged (and alerting would hook here)
        log.exception("weekly job failed")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sched = BlockingScheduler(timezone=ZoneInfo(settings.timezone))
    sched.add_job(job, make_trigger(), id="weekly_report", misfire_grace_time=6 * 3600, coalesce=True, max_instances=1)
    if settings.run_on_start:
        job()
    log.info("next run: %s", make_trigger().get_next_fire_time(None, datetime.now(ZoneInfo(settings.timezone))))
    sched.start()


if __name__ == "__main__":
    main()
