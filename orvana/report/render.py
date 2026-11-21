"""Turn facts into the email body (HTML + plain text). Jinja autoescape is ON, so customer text
quoted in the urgent-tickets table cannot inject HTML/JS into management's mail client."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]), trim_blocks=True, lstrip_blocks=True)
_text_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=False,
                        lstrip_blocks=True)  # no trim_blocks: keep the line breaks of the plain-text email


def spikes(f: dict, min_abs: int = 10, min_rel: float = 0.5) -> list[dict]:
    """Categories that grew by >= min_abs tickets AND >= 50% vs last week."""
    return [c for c in f["categories"] if c.get("prev_count") is not None
            and c["change_vs_prev"] >= min_abs and c["change_vs_prev"] >= min_rel * max(c["prev_count"], 1)]


def _by_product(groups: list[dict]) -> str:
    """'Samsung x3 (GIZ-05, UPE-03); VoltX x1' from (product, store, count) groups."""
    by_product: dict[str, list] = {}
    for s in groups:
        by_product.setdefault(s["product_mentioned"], []).append(s)
    parts = []
    for prod, rows in sorted(by_product.items(), key=lambda kv: -sum(r["count"] for r in kv[1])):
        stores = sorted({r["store_mentioned"] for r in rows if r["store_mentioned"] != "-"})
        parts.append(f"{prod} x{sum(r['count'] for r in rows)}" + (f" ({', '.join(stores)})" if stores else ""))
    return "; ".join(parts)


def _action(tag: str, tone: str, text: str) -> dict:
    """tone picks the email's colour for the tag: critical | serious | warning | violet | info (always shown with the tag text)."""
    return {"tag": tag, "tone": tone, "text": text}


def actions(f: dict) -> list[dict]:
    """Concrete, policy-based actions derived from the facts (cs_policy_en.md section 5), most urgent first."""
    out = []
    for e in f["emerging_issues"]:
        out.append(_action("New issue", "critical",
                           f"Never seen in the history: {e['tickets']} tickets ({e['share_pct']}% of the week) "
                           f"share the terms {', '.join(e['terms'][:3])}, e.g. \"{e['examples'][0][:110]}\". "
                           f"Assign an owner today (likely team: {next(iter(e['categories']))})."))
    safety = f["safety_by_product_store"]
    if safety:
        out.append(_action("Safety", "critical",
                           f"Expired / overheating: {sum(s['count'] for s in safety)} reports: {_by_product(safety)}. "
                           f"Policy: escalate to the Quality team within 2 hours and pull the product from the shelf."))
    to_confirm = f["safety_to_confirm_by_product_store"]
    if to_confirm:
        out.append(_action("To confirm", "warning",
                           f"{sum(s['count'] for s in to_confirm)} safety reports where the product type rarely has that "
                           f"hazard (e.g. an 'expired' phone, a carton of milk that 'gets very hot'): {_by_product(to_confirm)}. "
                           f"Quality team: call the customer to confirm before pulling stock."))
    for c in spikes(f):
        out.append(_action("Spike", "serious", f"{c['category']}: {c['count']} tickets, {c['change_vs_prev']:+d} vs last week."))
    staff = f["staff_behavior_by_store"][:3]
    if staff:
        out.append(_action("Staff", "violet", "; ".join(f"{s['key']} {s['name']} x{s['count']}" for s in staff)
                           + ". Policy: escalate to the store manager within 24 hours."))
    if f["repeat"]["said_repeat_complaint"]:
        out.append(_action("Call back", "info", f"{f['repeat']['said_repeat_complaint']} customers say this is their "
                                                f"third complaint: call them back first."))
    return out


def kpis(f: dict) -> list[dict]:
    v = f["volume"]
    sub = f"{v['change_vs_prev_pct']:+.1f}% vs last week" if v["change_vs_prev_pct"] is not None else "first week"
    s = f["safety"]
    safety_sub = "2h escalation" + (f", +{s['to_confirm']} to confirm" if s["to_confirm"] else "")
    return [{"label": "Valid complaints", "value": v["valid_tickets"], "sub": sub, "tone": "info"},
            {"label": "Urgent (level 3)", "value": f["urgency"]["3"], "sub": "safety, repeat, double charge", "tone": "serious"},
            {"label": "Safety cases", "value": s["confirmed"], "sub": safety_sub, "tone": "critical"},
            {"label": "Injection attempts", "value": v["injection_attempts"], "sub": "neutralised", "tone": "good"}]


# Email colours (inline styles: mail clients drop <style> blocks). Status steps from the data-viz reference palette;
# every coloured chip also carries its label, so colour never carries meaning alone.
# tone -> (accent, tint background, text on the tint)
TONES = {"critical": ("#d03b3b", "#fbeaea", "#9b2222"), "serious": ("#ec835a", "#fdefe8", "#9a4119"),
         "warning": ("#fab219", "#fff5dc", "#7a5200"), "good": ("#0ca30c", "#e7f5e7", "#1d6b1d"),
         "violet": ("#6250d6", "#efedfb", "#4a3aa7"), "info": ("#2a78d6", "#e8f1fc", "#1c5cab"),
         "neutral": ("#7b8794", "#f0f2f5", "#3e4c59")}


def render(f: dict, summary: str, summary_source: str, model_quality: str = "n/a") -> tuple[str, str]:
    max_share = max([c["share_pct"] for c in f["categories"]] + [0.1])
    ctx = dict(f=f, summary=summary, summary_source=summary_source, kpis=kpis(f), actions=actions(f), tones=TONES,
               max_share=max_share, has_prev=f["volume"]["prev_valid_tickets"] is not None, max_urgent_rows=30,
               max_other_rows=20,  # Gmail clips an email above ~102 KB; the full list is in the CSV
               model_quality=model_quality)
    return _env.get_template("weekly.html.j2").render(**ctx), _text_env.get_template("weekly.txt.j2").render(**ctx)
