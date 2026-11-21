import {
  BadgeCheck, Bot, BookOpenCheck, Headset, MessageSquareText, Package, RefreshCw, Send, ShieldAlert, Store, Ticket, User,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Badge, Card, CategoryBadge, CopyButton, UrgencyBadge } from "../components/ui";
import { ACTIONS, HANDOFF_REASONS, SLOTS } from "../meta";
import type { ChatCase, ChatState, DemoCustomers } from "../types";

interface Msg { id: number; role: "me" | "bot"; text: string; meta?: string; error?: boolean }

const SUGGESTIONS: Record<string, string[]> = {
  "": ["النقط بتنتهي امتى؟", "الشاحن VoltX بيسخن وريحته شياط", "مواعيد فرع المعادي إيه؟", "How long does delivery take?", "عايز أكلم موظف"],
  "DEMO-1": ["طلبي متأخر ولسه ما وصلش", "My order ORV-10010 is late", "el order mt2a5ar", "النقط بتنتهي امتى؟", "عايز أكلم موظف"],
  "DEMO-2": ["رجعت الخلاط ولسه الفلوس مارجعتش", "طلب ORV-10006 متأخر أكتر من 10 أيام", "I returned order ORV-10008, when do I get my money?"],
  "DEMO-3": ["I was charged 1950 EGP twice for order ORV-10010", "اتخصم مني 400 جنيه مرتين", "جالي خلاط غير اللي طلبته"],
};

export function ChatPage({ initial }: { initial?: { say?: string; as?: string } }) {
  const [customer, setCustomer] = useState(initial?.as ?? "DEMO-1");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [state, setState] = useState<ChatState | null>(null);
  const [facts, setFacts] = useState<{ id: string; text: string }[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [demo, setDemo] = useState<DemoCustomers | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const nextId = useRef(1);
  const started = useRef(false);

  useEffect(() => { api.demoCustomers().then(setDemo).catch(() => setDemo(null)); }, []);
  useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight }); }, [messages, sending]);
  useEffect(() => {  // deep link: #/chat?say=...&as=DEMO-1
    if (initial?.say && !started.current) { started.current = true; void send(initial.say); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function push(m: Omit<Msg, "id">) { setMessages((all) => [...all, { ...m, id: nextId.current++ }]); }

  async function send(text: string) {
    const t = text.trim();
    if (!t || sending) return;
    push({ role: "me", text: t });
    setDraft("");
    setSending(true);
    try {
      const r = await api.chat(sessionId, customer, t);
      setSessionId(r.session_id);
      setState(r.state);
      if (r.facts_text.length) {
        setFacts((old) => [...r.facts_text, ...old.filter((f) => !r.facts_text.some((n) => n.id === f.id))]);
      }
      push({ role: "bot", text: r.reply, meta: `${ACTIONS[r.action] ?? r.action} · ${r.source} · ${Math.round(r.latency_ms)} ms` });
    } catch (e) {
      push({ role: "bot", text: (e as Error).message, error: true });
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  async function reset(nextCustomer = customer) {
    if (sessionId) await api.resetChat(sessionId).catch(() => {});
    setSessionId(null); setMessages([]); setState(null); setFacts([]); setCustomer(nextCustomer);
  }

  const orders = demo?.customers.find((c) => c.customer_id === customer)?.orders ?? [];

  return (
    <div className="chat-grid">
      <section className="card chat-card">
        <div className="chat-top">
          <div className="who"><User size={15} /> Logged in as</div>
          <select className="select" value={customer} onChange={(e) => void reset(e.target.value)} aria-label="customer">
            <option value="">Guest (no orders)</option>
            <option value="DEMO-1">DEMO-1</option>
            <option value="DEMO-2">DEMO-2</option>
            <option value="DEMO-3">DEMO-3</option>
          </select>
          <span style={{ flex: 1 }} />
          {state && <StatusBadge state={state} />}
          <button className="btn btn-sm" onClick={() => void reset()}><RefreshCw size={14} /> New chat</button>
        </div>

        <div className="messages" ref={listRef}>
          {messages.length === 0 && (
            <div className="welcome">
              <div className="brand-mark" style={{ width: 52, height: 52, borderRadius: 16 }}><Bot size={26} color="#fff" /></div>
              <h3>Orvana customer service</h3>
              <p className="ink2" style={{ margin: 0 }}>
                Ask about delivery, refunds, returns or Orvana Plus points, or report a problem. The bot replies in your
                language (Egyptian Arabic, Franco or English) and hands over to a colleague when it should.
              </p>
              <div className="chips" style={{ justifyContent: "center" }}>
                {(SUGGESTIONS[customer] ?? SUGGESTIONS[""]).map((s) => (
                  <button key={s} className="chip" dir="auto" onClick={() => void send(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`msg ${m.role === "me" ? "msg-me" : "msg-bot"}`}>
              <div className={`avatar ${m.role === "bot" ? "avatar-bot" : ""}`}>{m.role === "bot" ? <Bot size={16} /> : <User size={16} />}</div>
              <div>
                <div className="bubble" dir="auto" style={m.error ? { background: "var(--critical-soft)", color: "var(--critical-text)" } : undefined}>{m.text}</div>
                {m.meta && <div className="msg-meta">{m.meta}</div>}
              </div>
            </div>
          ))}
          {sending && (
            <div className="msg msg-bot">
              <div className="avatar avatar-bot"><Bot size={16} /></div>
              <div className="bubble typing" aria-label="typing"><span /><span /><span /></div>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); void send(draft); }}>
          <textarea
            ref={inputRef} rows={1} dir="auto" value={draft} placeholder="اكتب رسالتك… / Type a message…"
            onChange={(e) => {
              setDraft(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
            }}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(draft); } }}
          />
          <button className="btn btn-primary" type="submit" disabled={sending || !draft.trim()} aria-label="send">
            <Send size={16} /> Send
          </button>
        </form>
      </section>

      <AgentPanel state={state} facts={facts} customer={customer} orders={orders} today={demo?.today}
        onUseOrder={(id) => { setDraft((d) => (d ? `${d} ${id}` : id)); inputRef.current?.focus(); }} />
    </div>
  );
}

function StatusBadge({ state }: { state: ChatState }) {
  if (state.status === "handed_off") return <Badge tone="warning" icon={Headset}>With a colleague</Badge>;
  if (state.status === "closed") return <Badge tone="good">Closed</Badge>;
  return <Badge tone="accent" icon={MessageSquareText}>Open</Badge>;
}

function AgentPanel({ state, facts, customer, orders, today, onUseOrder }: {
  state: ChatState | null; facts: { id: string; text: string }[]; customer: string;
  orders: { order_id: string; promised: string; delivered: string | null; status: string; payment: string }[];
  today?: string; onUseOrder: (id: string) => void;
}) {
  return (
    <aside className="agent">
      <Card title="Agent panel" icon={Headset} action={<span className="muted small">what the code decided</span>}>
        {!state ? (
          <div className="muted small">Start a conversation: the decisions (case, order check, voucher, escalation,
            hand-over) appear here as they are made. The bot's wording never decides them.</div>
        ) : (
          <dl className="kv" style={{ gridTemplateColumns: "110px 1fr" }}>
            <dt>Status</dt>
            <dd>
              <StatusBadge state={state} />
              {state.handoff_reason && <div className="small ink2" style={{ marginTop: 4 }}>{HANDOFF_REASONS[state.handoff_reason] ?? state.handoff_reason}</div>}
            </dd>
            <dt>Language</dt><dd>{state.lang === "en" ? "English" : "Arabic (Egyptian)"}</dd>
            <dt>Waiting for</dt><dd>{state.awaiting ? SLOTS[state.awaiting] ?? state.awaiting : "—"}</dd>
            <dt>Turns</dt><dd className="num">{state.turns}</dd>
          </dl>
        )}
      </Card>

      {state && state.cases.length > 0 && (
        <Card title={`Cases (${state.cases.length})`} icon={Ticket}>
          <div className="stack" style={{ gap: 10 }}>
            {[...state.cases].reverse().map((c, i) => <CaseCard key={i} c={c} />)}
          </div>
        </Card>
      )}

      {facts.length > 0 && (
        <Card title="Policy applied" icon={BookOpenCheck} action={<span className="muted small">cs_policy_en.md</span>}>
          <div className="stack" style={{ gap: 8 }}>
            {facts.map((f) => <div key={f.id} className="fact" dir="auto">{f.text}</div>)}
          </div>
        </Card>
      )}

      <Card title="Demo orders" icon={Package} action={<Badge tone="warning">DEMO</Badge>}>
        {customer === "" ? (
          <div className="muted small">Guests have no orders: an order number they give is recorded but not looked up.
            Pick DEMO-1…3 to try the order checks.</div>
        ) : (
          <>
            <table className="orders-table">
              <tbody>
                {orders.map((o) => (
                  <tr key={o.order_id}>
                    <td><button className="link-btn mono" onClick={() => onUseOrder(o.order_id)} title="put in the message box">{o.order_id}</button></td>
                    <td className="ink2">due {o.promised}</td>
                    <td className="ink2">{o.delivered ? `delivered ${o.delivered}` : o.status.replace(/_/g, " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="muted small" style={{ marginTop: 8 }}>
              Fictional orders · "today" = {today}. Other customers' orders are treated as not found.
            </div>
          </>
        )}
      </Card>
    </aside>
  );
}

function CaseCard({ c }: { c: ChatCase }) {
  const o = c.outcome;
  return (
    <div className="case">
      <div className="case-head">
        <CategoryBadge category={c.category} showAr={false} />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {c.safety && <Badge tone="critical" icon={ShieldAlert}>Safety</Badge>}
          <UrgencyBadge urgency={c.urgency} />
        </div>
      </div>
      <dl className="kv">
        <dt>Order</dt>
        <dd>{c.order_id ? <><span className="mono">{c.order_id}</span>{" "}
          {c.order_verified ? <Badge tone="good" icon={BadgeCheck}>verified</Badge> : <Badge tone="muted">not verified</Badge>}</> : "—"}</dd>
        {c.store && <><dt>Branch</dt><dd><Store size={13} className="muted" /> {c.store}</dd></>}
        {c.product && <><dt>Brand</dt><dd>{c.product}</dd></>}
      </dl>
      {(o.voucher_egp !== undefined || o.refund_days || o.days_late !== undefined) && (
        <div className="outcomes">
          {o.days_late !== undefined && <Badge tone="neutral">{o.days_late} days late</Badge>}
          {o.voucher_egp !== undefined && (o.voucher_egp > 0
            ? <Badge tone="accent">Voucher {o.voucher_egp} EGP</Badge> : <Badge tone="muted">No voucher</Badge>)}
          {o.can_cancel && <Badge tone="accent">May cancel, full refund</Badge>}
          {o.refund_days ? <Badge tone="accent">Refund within {o.refund_days} working days</Badge> : null}
        </div>
      )}
      <div className="case-head">
        <span className="small muted">{c.done ? "Ticket created" : "Collecting details…"}</span>
        {c.ticket_id && <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span className="mono">{c.ticket_id}</span><CopyButton text={c.ticket_id} label="" />
        </span>}
      </div>
    </div>
  );
}
