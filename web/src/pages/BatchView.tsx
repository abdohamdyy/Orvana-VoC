import {
  CircleCheck, Download, FileUp, Gauge, Inbox, Layers, Search, ShieldAlert, Store, Tag, TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { BarList } from "../components/BarList";
import { Badge, Card, CategoryPill, Spinner, UrgencyBadge } from "../components/ui";
import { CATEGORIES, URGENCY } from "../meta";
import type { BatchResult, BatchRow, Category } from "../types";

const MAX_BYTES = 5_000_000;
const EVAL_COLUMNS = ["ticket_id", "category", "secondary_category", "product_mentioned", "store_mentioned", "sentiment", "urgency"] as const;

function csvCell(v: unknown): string {
  let s = String(v ?? "");
  if (/^[=+\-@\t\r]/.test(s)) s = "'" + s; // CSV/Excel formula injection
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function download(rows: BatchRow[], name: string) {
  const lines = [EVAL_COLUMNS.join(","), ...rows.map((r) => EVAL_COLUMNS.map((c) => csvCell(r[c])).join(","))];
  const url = URL.createObjectURL(new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function BatchView({ initialSample }: { initialSample?: string }) {
  const [result, setResult] = useState<BatchResult | null>(null);
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  async function run(payload: { csv?: string; sample?: string }, label: string) {
    setLoading(true); setError(""); setResult(null); setSource(label);
    try { setResult(await api.batch(payload)); } catch (e) { setError((e as Error).message); } finally { setLoading(false); }
  }
  async function onFile(f: File | undefined) {
    if (!f) return;
    if (f.size > MAX_BYTES) { setError("The file is larger than 5 MB."); return; }
    void run({ csv: await f.text() }, f.name);
  }
  useEffect(() => { if (initialSample) void run({ sample: initialSample }, `sample ${initialSample}`); // deep link
  // eslint-disable-next-line
  }, []);

  return (
    <div className="stack">
      <div className="grid-2">
        <div className={`dropzone ${over ? "over" : ""}`} role="button" tabIndex={0}
          onClick={() => fileRef.current?.click()} onKeyDown={(e) => { if (e.key === "Enter") fileRef.current?.click(); }}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); void onFile(e.dataTransfer.files[0]); }}>
          <FileUp size={30} className="muted" />
          <div><b>Drop a CSV here</b> or click to browse</div>
          <div className="small muted">Needs a <span className="mono">message_text</span> column (<span className="mono">ticket_id</span> optional) · up to 5 MB / 5,000 rows · processed in memory, nothing is stored</div>
          <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => void onFile(e.target.files?.[0])} />
        </div>
        <Card title="Or try a real week" icon={Inbox}>
          <p className="ink2" style={{ marginTop: 0 }}>The two weekly files in <span className="mono">incoming/</span>, as they arrive every Monday.</p>
          <div className="chips">
            <button className="btn" onClick={() => void run({ sample: "week1" }, "incoming week 1 (1–7 Jul 2026)")}>Week 1 · 1–7 Jul</button>
            <button className="btn" onClick={() => void run({ sample: "week2" }, "incoming week 2 (8–14 Jul 2026)")}>Week 2 · 8–14 Jul</button>
          </div>
        </Card>
      </div>

      {loading && <Card><div className="empty"><Spinner /> Classifying {source}…</div></Card>}
      {error && <div className="error">{error}</div>}
      {result && <Dashboard r={result} source={source} />}
    </div>
  );
}

function Dashboard({ r, source }: { r: BatchResult; source: string }) {
  const s = r.summary;
  const pct = (n: number) => (s.rows_received ? `${((100 * n) / s.rows_received).toFixed(1)}% of rows` : "");
  return (
    <div className="stack">
      <div className="toolbar">
        <div className="ink2"><b>{source}</b> · {s.rows_received} tickets in {(s.elapsed_ms / 1000).toFixed(1)} s</div>
        <button className="btn btn-primary" onClick={() => download(r.rows, "predictions.csv")}><Download size={16} /> Download predictions (evaluate.py format)</button>
      </div>
      <div className="kpis">
        <Kpi label="Tickets" icon={Layers} value={s.rows_received} sub="after removing duplicate ids" />
        <Kpi label="Valid complaints" icon={CircleCheck} value={s.valid} sub={pct(s.valid)} />
        <Kpi label="Urgent (3)" icon={ShieldAlert} value={s.urgent} sub="safety · double charge · repeat" />
        <Kpi label="Injection flagged" icon={TriangleAlert} value={s.injection_flagged} sub="removed before classification" />
        <Kpi label="Not a complaint" icon={Inbox} value={s.invalid} sub="empty or pure injection" />
      </div>
      <div className="grid-2">
        <Card title="Complaints by category" icon={Tag} action={<span className="muted small">valid tickets, primary category</span>}>
          <BarList total={s.valid} items={Object.entries(s.by_category).map(([c, n]) => {
            const meta = CATEGORIES[c as Category]; const Icon = meta.icon;
            return { key: c, value: n, label: <><Icon size={14} className="muted" /><span>{meta.en}</span></>, tooltip: `${meta.en} · ${meta.ar}: ${n}` };
          })} />
        </Card>
        <div className="stack">
          <Card title="Urgency" icon={Gauge}>
            <BarList total={s.valid} ranked={false} items={([3, 2, 1] as const).map((u) => ({
              key: String(u), value: s.by_urgency[String(u) as "1" | "2" | "3"],
              label: <span>{u} · {URGENCY[u].label}</span>,
            }))} />
          </Card>
          <div className="grid-2">
            <Card title="Top branches" icon={Store}>
              <BarList compact total={s.valid} items={s.top_stores.map((t) => ({ key: t.code, value: t.count, label: <span title={`${t.en} · ${t.ar}`}>{t.en.replace(" branch", "")}</span>, tooltip: `${t.code} · ${t.en} · ${t.ar}: ${t.count}` }))} />
            </Card>
            <Card title="Top brands" icon={Tag}>
              <BarList compact total={s.valid} items={s.top_products.map((t) => ({ key: t.brand, value: t.count, label: <span>{t.brand}</span> }))} />
            </Card>
          </div>
        </div>
      </div>
      <ResultsTable rows={r.rows} />
    </div>
  );
}

function Kpi({ label, value, sub, icon: Icon }: { label: string; value: number; sub: string; icon: typeof Layers }) {
  return (
    <div className="card kpi">
      <div className="kpi-label"><Icon size={14} /> {label}</div>
      <div className="kpi-value num">{value.toLocaleString()}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );
}

function ResultsTable({ rows }: { rows: BatchRow[] }) {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [urg, setUrg] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [limit, setLimit] = useState(100);
  const shown = useMemo(() => rows.filter((r) =>
    (!cat || r.category === cat) && (!urg || String(r.urgency) === urg) &&
    (!q || r.text.toLowerCase().includes(q.toLowerCase()) || r.ticket_id.toLowerCase().includes(q.toLowerCase()))), [rows, q, cat, urg]);

  return (
    <section className="card">
      <div className="card-head" style={{ paddingBottom: 12, flexWrap: "wrap" }}>
        <h3 className="card-title">All tickets <span className="muted small">({shown.length} of {rows.length})</span></h3>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <div style={{ position: "relative" }}>
            <Search size={14} className="muted" style={{ position: "absolute", insetInlineStart: 10, top: 11 }} />
            <input className="input" style={{ paddingInlineStart: 30 }} placeholder="Search text or id" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <select className="select" value={cat} onChange={(e) => setCat(e.target.value)} aria-label="category">
            <option value="">All categories</option>
            {Object.entries(CATEGORIES).map(([k, v]) => <option key={k} value={k}>{v.en}</option>)}
          </select>
          <select className="select" value={urg} onChange={(e) => setUrg(e.target.value)} aria-label="urgency">
            <option value="">Any urgency</option><option value="3">3 · Urgent</option><option value="2">2 · Normal</option><option value="1">1 · Low</option>
          </select>
        </div>
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>ID</th><th>Message</th><th>Category</th><th>Secondary</th><th>Brand</th><th>Branch</th><th>Urgency</th><th>Flags</th></tr></thead>
          <tbody>
            {shown.slice(0, limit).map((r) => (
              <tr key={r.ticket_id} className="row" onClick={() => setOpen(open === r.ticket_id ? null : r.ticket_id)}>
                <td className="mono">{r.ticket_id}</td>
                <td><div className={`msg-cell ${open === r.ticket_id ? "open" : ""}`} dir="auto">{r.text || <span className="muted">(empty)</span>}</div></td>
                <td>{r.is_valid ? <CategoryPill category={r.category} /> : <span className="muted">—</span>}</td>
                <td>{r.secondary_category ? <CategoryPill category={r.secondary_category} /> : <span className="muted">—</span>}</td>
                <td>{r.product_mentioned || <span className="muted">—</span>}</td>
                <td className="mono">{r.store_mentioned || <span className="muted">—</span>}</td>
                <td>{r.is_valid ? <UrgencyBadge urgency={r.urgency} /> : <span className="muted">—</span>}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  {r.injection_flag && <Badge tone="critical" icon={TriangleAlert}>injection</Badge>}{" "}
                  {!r.is_valid && <Badge tone="muted">not a complaint</Badge>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown.length > limit && (
        <div style={{ padding: 12, textAlign: "center" }}>
          <button className="btn btn-sm" onClick={() => setLimit((l) => l + 200)}>Show more ({shown.length - limit} left)</button>
        </div>
      )}
    </section>
  );
}
