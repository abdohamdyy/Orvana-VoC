"""The weekly job: new file in incoming/ -> classified tickets -> facts -> email. Safe to run any
number of times: a file is classified once, a week is emailed once (see state.py)."""
import hashlib
import json
import logging
import smtplib
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import settings
from ..pipeline import process_frame
from . import mailer
from .render import render
from .state import State, now, run_lock
from .stats import csv_rows, weekly_facts
from .summary_llm import write_summary

log = logging.getLogger("weekly")
REQUIRED = {"ticket_id", "message_text", "created_at", "channel"}


def week_key_from(path: Path) -> str:
    return path.stem.removeprefix("tickets_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_stable(path: Path, seconds: int) -> bool:
    """A file still being copied/uploaded is skipped until it has not changed for `seconds`.
    seconds <= 0 disables the check (file timestamps can be a fraction of a second 'in the future'
    on Windows / across a Docker volume, which made a 0-second check flaky)."""
    return seconds <= 0 or time.time() - path.stat().st_mtime >= seconds


def ollama_up() -> bool:
    try:
        return requests.get(f"{settings.ollama_url}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def production_backends(cache: bool = True):
    """Classifier: fine-tuned model (or TF-IDF if absent) with LLM fallback when Ollama is up.
    Summary LLM: Ollama if reachable, else None (template summary).
    cache=False for anything live (UI, chatbot): no customer text is written to the LLM disk cache."""
    from ..backends import make_cascade
    from ..backends.cascade import CascadeBackend

    try:
        from ..backends.finetuned import FinetunedBackend
        primary, fine_tuned = FinetunedBackend.load(), True
    except Exception as e:  # model files missing: keep working with the baseline
        log.warning("fine-tuned model unavailable (%s); using TF-IDF baseline", e)
        from ..backends.tfidf import TfidfBackend
        primary, fine_tuned = TfidfBackend.load(), False
    if not ollama_up():
        log.warning("Ollama not reachable at %s: no LLM fallback, template summary", settings.ollama_url)
        return primary, None
    from ..backends.llm_ollama import OllamaBackend
    llm = OllamaBackend(cache=cache)
    summary_llm = llm
    if settings.summary_model and settings.summary_model != llm.model:
        try:
            summary_llm = OllamaBackend(model=settings.summary_model, prompt_version="summary", cache=cache)
        except RuntimeError as e:  # summary model not pulled: use the classifier LLM
            log.warning("%s", e)
    # the cascade's routing rule comes from <model_dir>/routing.json (chosen on the suite's dev split)
    classifier = make_cascade(llm) if fine_tuned else CascadeBackend(primary, llm, settings.cascade_threshold)
    return classifier, summary_llm


def read_incoming(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, on_bad_lines="warn", encoding="utf-8")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if "customer_id" not in df:
        df["customer_id"] = None
    df["message_text"] = df.message_text.fillna("")
    return df


def classify_file(path: Path, sha: str, week: str, state: State, backend) -> None:
    df = read_incoming(path)
    rows_received = len(df)
    df = df.drop_duplicates("ticket_id", keep="first")  # exact duplicate rows (12 in week 2)
    records = process_frame(df, backend)
    rows = [{"ticket_id": r.ticket_id, "week_key": week, "created_at": c, "channel": ch,
             "customer_id": cid if isinstance(cid, str) else None, "message_text": t, "record": r.model_dump_json()}
            for r, c, ch, cid, t in zip(records, df.created_at, df.channel, df.customer_id, df.message_text)]
    state.save_tickets(path.name, sha, week, rows_received, rows)
    log.info("classified %s: %d rows, %d unique tickets", path.name, rows_received, len(df))


def model_quality() -> str:
    p = settings.outputs_dir / "metrics" / "model_quality.json"
    return json.loads(p.read_text()).get("category_macro_f1", "n/a") if p.exists() else "n/a"


def build_report(week: str, state: State, llm):
    rows = state.week_tickets(week)
    name, rows_received = state.file_meta(week)
    prev_key = state.previous_week_key(week)
    prev_rows = state.week_tickets(prev_key) if prev_key else None
    stores = pd.read_csv(settings.data_dir / "store_names.csv")
    facts = weekly_facts(week, rows, rows_received, prev_rows, dict(zip(stores.store_id, stores.name_en)))
    summary, source = write_summary(facts, llm)
    html, text = render(facts, summary, source, model_quality())
    return facts, summary, source, html, text


def run_pending(force: bool = False, send_email: bool = True, backend=None, llm=None) -> list[dict]:
    """Process every complete file in incoming/ that has not been emailed yet."""
    results = []
    with run_lock():
        state = State()
        if backend is None:
            backend, llm = production_backends()
        for path in sorted(settings.incoming_dir.glob("*.csv")):
            week, sha = week_key_from(path), sha256(path)
            res = {"file": path.name, "week": week, "at": now()}
            if not is_stable(path, settings.file_stable_seconds):
                res["status"] = "skipped: file still changing"
                results.append(res)
                continue
            row, email = state.file_row(path.name), state.email_status(week)
            changed = row is not None and row[0] != sha
            if email == "sent" and not force:
                res["status"] = "skipped: already emailed" + (" (WARNING: file changed since; use --force)" if changed else "")
                if changed:
                    log.warning("%s changed after its report was sent; not resending without --force", path.name)
                results.append(res)
                continue
            if email == "sending" and not force:
                log.error("week %s: a previous send was interrupted; not resending automatically (check Mailpit, "
                          "then run_weekly.py --force)", week)
                res["status"] = "ALERT: interrupted send, manual check needed"
                results.append(res)
                continue
            if row is None or changed or force:
                classify_file(path, sha, week, state, backend)
            facts, summary, source, html, text = build_report(week, state, llm)
            res.update(summary_source=source, valid_tickets=facts["volume"]["valid_tickets"])
            msg = mailer.build_message(week, facts["period"], html, text, csv_rows(facts))
            if not send_email:
                res["status"] = f"report built, not sent ({mailer.save_eml(msg, week, html)})"
                results.append(res)
                continue
            if force:
                state.db.execute("DELETE FROM emails WHERE week_key=?", (week,))
            if not state.claim_email(week, msg["Message-ID"]):
                res["status"] = "skipped: another run claimed this email"
                results.append(res)
                continue
            eml = mailer.save_eml(msg, week, html)
            try:
                mailer.send(msg)
            except (smtplib.SMTPException, OSError) as e:  # nothing was delivered: allow a retry next run
                state.release_email(week)
                res["status"] = f"FAILED to send: {e}"
                log.error("week %s: send failed: %s", week, e)
                results.append(res)
                continue
            state.mark_sent(week, str(eml))
            res["status"] = "emailed"
            log.info("week %s emailed (%s)", week, eml)
            results.append(res)
    log_path = settings.outputs_dir / "logs" / "weekly_runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results
