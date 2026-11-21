"""Stage 4 guarantees: no duplicate processing, no duplicate emails, overlapping runs blocked,
correct schedule (Monday 08:00 Cairo, across DST), HTML escaping, numbers recomputed independently."""
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from orvana.report import mailer, weekly
from orvana.report.state import State, run_lock

ROWS = [
    ("T1", "C1", "2026-07-01 10:00:00", "app_chat", "فين الطلب؟ قالولي هيوصل امبارح"),
    ("T1", "C1", "2026-07-01 10:00:00", "app_chat", "فين الطلب؟ قالولي هيوصل امبارح"),  # exact duplicate row
    ("T2", "C2", "2026-07-02 11:00:00", "whatsapp", "<script>alert(1)</script> الأوردر متأخر بقاله 4 أيام"),
    ("T3", None, "2026-07-03 12:00:00", "facebook", ""),  # empty message
    ("T4", "C2", "2026-07-04 13:00:00", "branch", "مواعيد فرع المعادي إيه يوم الجمعة؟"),
]


@pytest.fixture
def week_file(isolated_settings):
    path = isolated_settings.incoming_dir / "tickets_2026-07-01_to_2026-07-07.csv"
    pd.DataFrame(ROWS, columns=["ticket_id", "customer_id", "created_at", "channel", "message_text"]).to_csv(path, index=False)
    return path


@pytest.fixture
def sent(monkeypatch):
    box = []
    monkeypatch.setattr(mailer, "send", lambda msg: box.append(msg))
    return box


def test_twice_means_one_email_and_one_classification(week_file, sent, dummy_backend, monkeypatch):
    calls = []
    real = weekly.classify_file
    monkeypatch.setattr(weekly, "classify_file", lambda *a, **k: (calls.append(1), real(*a, **k)))
    first = weekly.run_pending(backend=dummy_backend, llm=None)
    second = weekly.run_pending(backend=dummy_backend, llm=None)
    assert first[0]["status"] == "emailed"
    assert second[0]["status"].startswith("skipped: already emailed")
    assert len(sent) == 1 and len(calls) == 1


def test_counts_are_exact(week_file, sent, dummy_backend):
    weekly.run_pending(backend=dummy_backend, llm=None)
    facts = weekly.build_report("2026-07-01_to_2026-07-07", State(), None)[0]
    raw = pd.read_csv(week_file)
    unique = raw.drop_duplicates("ticket_id")
    assert facts["volume"]["rows_received"] == len(raw) == 5
    assert facts["volume"]["duplicate_rows_removed"] == len(raw) - len(unique) == 1
    assert facts["volume"]["invalid_messages"] == 1
    assert facts["volume"]["valid_tickets"] == 3
    assert sum(c["count"] for c in facts["categories"]) == facts["volume"]["valid_tickets"]


def test_changed_file_after_send_is_not_resent(week_file, sent, dummy_backend):
    weekly.run_pending(backend=dummy_backend, llm=None)
    week_file.write_text(week_file.read_text(encoding="utf-8") + "T9,C9,2026-07-05 10:00:00,branch,جديد\n", encoding="utf-8")
    res = weekly.run_pending(backend=dummy_backend, llm=None)
    assert "WARNING" in res[0]["status"] and len(sent) == 1


def test_failed_send_can_retry(week_file, dummy_backend, monkeypatch):
    def boom(msg):
        raise ConnectionRefusedError("smtp down")
    monkeypatch.setattr(mailer, "send", boom)
    assert weekly.run_pending(backend=dummy_backend, llm=None)[0]["status"].startswith("FAILED")
    box = []
    monkeypatch.setattr(mailer, "send", lambda m: box.append(m))
    assert weekly.run_pending(backend=dummy_backend, llm=None)[0]["status"] == "emailed" and len(box) == 1


def test_interrupted_send_is_not_resent_automatically(week_file, sent, dummy_backend):
    State().claim_email("2026-07-01_to_2026-07-07", "<x>")  # simulate a crash in the middle of sending
    res = weekly.run_pending(backend=dummy_backend, llm=None)
    assert res[0]["status"].startswith("ALERT") and sent == []


def test_file_still_being_written_is_skipped(week_file, sent, dummy_backend, monkeypatch):
    monkeypatch.setattr(weekly.settings, "file_stable_seconds", 3600)
    assert weekly.run_pending(backend=dummy_backend, llm=None)[0]["status"] == "skipped: file still changing"


def test_overlapping_runs_are_blocked(isolated_settings):
    with run_lock():
        with pytest.raises(RuntimeError):
            with run_lock():
                pass


def test_two_claims_only_one_wins(isolated_settings):
    State()  # create the schema once
    wins = []
    # each thread opens its own connection, like two separate processes would
    threads = [threading.Thread(target=lambda: wins.append(State().claim_email("W", "<id>"))) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(wins) == [False, True]


def test_html_escapes_customer_text(week_file, sent, dummy_backend, monkeypatch):
    monkeypatch.setattr(weekly, "weekly_facts", _with_urgent(weekly.weekly_facts))
    weekly.run_pending(backend=dummy_backend, llm=None)
    html = sent[0].get_body(("html",)).get_content()
    assert "<script>" not in html and "&lt;script&gt;" in html


def _with_urgent(fn):
    """Force T2 into the urgent table so its text is rendered."""
    def wrapper(*a, **k):
        f = fn(*a, **k)
        f["urgent_tickets"].append({"ticket_id": "T2", "created_at": "2026-07-02 11:00", "channel": "whatsapp",
                                    "category": "delivery_delay", "product_mentioned": "", "store_mentioned": "",
                                    "reason": "test", "text": "<script>alert(1)</script>"})
        return f
    return wrapper


@pytest.mark.parametrize("now,expected", [
    (datetime(2026, 7, 15, 12, 0), datetime(2026, 7, 20, 8, 0)),   # summer (EEST, UTC+3)
    (datetime(2026, 12, 2, 9, 0), datetime(2026, 12, 7, 8, 0)),    # winter (EET, UTC+2)
    (datetime(2026, 7, 20, 7, 59), datetime(2026, 7, 20, 8, 0)),   # same Monday, just before
])
def test_schedule_is_monday_8_cairo(now, expected):
    from orvana.scheduler import make_trigger
    tz = ZoneInfo("Africa/Cairo")
    nxt = make_trigger().get_next_fire_time(None, now.replace(tzinfo=tz))
    assert nxt == expected.replace(tzinfo=tz) and nxt.weekday() == 0


SAFETY_ROWS = [
    ("S1", "C1", "2026-07-01 09:00:00", "app_chat",
     "الأوردر متأخر بقاله 5 أيام وكمان شامبو دوف منتهي الصلاحية ولقيته على الرف في فرع المعادي"),
    ("S2", "C2", "2026-07-02 09:00:00", "whatsapp", "الأوردر متأخر بقاله 3 أيام وكمان موبايل سامسونج منتهي الصلاحية"),
    ("S3", "C3", "2026-07-03 09:00:00", "facebook", "النقط بتاعتي منتهية الصلاحية ومحدش رد عليا"),
]


def test_safety_as_second_problem_reaches_the_quality_team(isolated_settings, sent, dummy_backend):
    """Policy §5: ANY safety complaint goes to Quality within 2h, even as the 2nd problem of an urgency-2 ticket.
    An 'expired' phone is listed to confirm, and expired points are not a safety report at all."""
    path = isolated_settings.incoming_dir / "tickets_2026-07-01_to_2026-07-07.csv"
    pd.DataFrame(SAFETY_ROWS, columns=["ticket_id", "customer_id", "created_at", "channel", "message_text"]).to_csv(path, index=False)
    weekly.run_pending(backend=dummy_backend, llm=None)
    facts, summary, _, html, text = weekly.build_report("2026-07-01_to_2026-07-07", State(), None)
    assert facts["urgency"]["3"] == 0  # urgency follows the main problem (the gold), unchanged
    assert facts["safety"] == {"confirmed": 1, "to_confirm": 1, "second_issue": 2}
    by_id = {s["ticket_id"]: s for s in facts["safety_tickets"]}
    assert set(by_id) == {"S1", "S2"} and by_id["S1"]["check"] == "confirmed" and by_id["S2"]["check"] == "needs_confirmation"
    assert all(s["urgency"] == 2 and s["where"] == "second issue" for s in by_id.values())
    assert "Safety reports for the Quality team: 1 confirmed, 1 to confirm" in html and "1 to confirm" in text
    assert "1 حالة سلامة" in summary and "1 بلاغ سلامة يحتاج تأكيد" in summary
    attachment = next(p for p in sent[0].iter_attachments()).get_content()
    attachment = attachment.decode("utf-8-sig") if isinstance(attachment, bytes) else attachment
    assert "S1" in attachment and "S2" in attachment and "S3" not in attachment


def test_starttls_verifies_the_server_certificate(monkeypatch):
    """A real SMTP password must never go over a TLS session whose certificate was not checked."""
    import ssl
    calls = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            calls["context"] = context

        def login(self, user, password):
            calls["login"] = user

        def send_message(self, msg):
            calls["sent"] = True

    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    for k, v in {"smtp_host": "smtp.example.com", "smtp_port": 587, "smtp_starttls": True,
                 "smtp_user": "u@example.com", "smtp_password": "x"}.items():
        monkeypatch.setattr(mailer.settings, k, v)
    mailer.send(mailer.build_message("W", {"start": "a", "end": "b"}, "<p>x</p>", "x", []))
    ctx = calls["context"]
    assert isinstance(ctx, ssl.SSLContext) and ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert calls["login"] == "u@example.com" and calls["sent"]

    calls.clear()
    monkeypatch.setattr(mailer.settings, "smtp_user", "")  # a local sink: no login
    mailer.send(mailer.build_message("W", {"start": "a", "end": "b"}, "<p>x</p>", "x", []))
    assert "login" not in calls and calls["sent"]


def test_message_id_follows_the_week_and_the_content():
    """Same report -> same id (an accidental resend collapses in the inbox); a corrected report -> a new id
    (Gmail drops a message whose id it already has, so the fix would never be seen)."""
    period = {"start": "a", "end": "b"}
    ids = [mailer.build_message("W1", period, html, "t", [])["Message-ID"] for html in ("<p>x</p>", "<p>x</p>", "<p>y</p>")]
    assert ids[0] == ids[1] != ids[2] and ids[0].startswith("<voc-W1-")
