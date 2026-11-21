"""Central settings. Values come from environment variables or a .env file (never hard-coded secrets)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # paths
    data_dir: Path = ROOT / "data"
    incoming_dir: Path = ROOT / "incoming"
    state_dir: Path = ROOT / "state"
    outputs_dir: Path = ROOT / "outputs"
    model_dir: Path = ROOT / "models" / "stage3"

    # LLM (Ollama, runs locally)
    # 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and added ~2 s to EVERY call
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b"
    llm_timeout_s: float = 60.0
    cascade_threshold: float = 0.8  # fine-tuned model confidence below this -> ask the LLM
    summary_model: str = ""  # LLM for the weekly Arabic summary; empty = same as llm_model (latency irrelevant here)

    # email (Mailpit locally; real SMTP creds only via env)
    smtp_host: str = "127.0.0.1"
    smtp_port: int = 1025
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = False
    mail_from: str = "voc-report@orvana.example"
    mail_to: str = "management@orvana.example"

    # scheduling
    schedule_cron: str = "0 8 * * mon"  # minute hour day month day_of_week
    timezone: str = "Africa/Cairo"
    run_on_start: bool = True  # catch-up run when the scheduler starts (safe: the job is idempotent)
    file_stable_seconds: int = 60  # ignore files modified less than this ago (still being written)

    # business rules
    double_charge_urgent_egp: int = 1000  # gold: 400/850 -> urgency 2, 1900/1950 -> 3
    max_message_chars: int = 1000

    # chatbot (B3)
    chat_use_llm: bool = True  # False = template replies only (no LLM in the chat at all)
    # how the LLM may word chat replies: template (none) | opener (one empathy sentence) | rewrite (whole reply).
    # template chosen by measurement on dev: same judge scores, 7 ms instead of ~4.5 s (docs/CHATBOT.md)
    chat_phrasing: str = "template"
    chat_answer_threshold: float = 0.33  # retrieval score needed to answer; scripts/calibrate_chat_threshold.py
    chat_session_ttl_s: int = 1800  # idle sessions are forgotten (kept in memory only, never on disk)
    chat_max_turns: int = 30  # after this, hand over to a human
    chat_max_sessions: int = 1000


settings = Settings()
