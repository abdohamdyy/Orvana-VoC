import { Check, Copy, type LucideIcon } from "lucide-react";
import { useState, type ReactNode } from "react";
import { CATEGORIES, URGENCY } from "../meta";
import type { Category } from "../types";

export function Card({ title, icon: Icon, action, children, className = "" }: {
  title?: ReactNode; icon?: LucideIcon; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="card-head">
          <h3 className="card-title">{Icon && <Icon size={16} className="muted" />}{title}</h3>
          {action}
        </div>
      )}
      <div className="card-body">{children}</div>
    </section>
  );
}

type Tone = "neutral" | "muted" | "accent" | "good" | "warning" | "critical";
export function Badge({ tone = "neutral", icon: Icon, children, title }: {
  tone?: Tone; icon?: LucideIcon; children: ReactNode; title?: string;
}) {
  return <span className={`badge badge-${tone}`} title={title}>{Icon && <Icon size={12} />}{children}</span>;
}

export function CategoryBadge({ category, size = "md", showAr = true }: {
  category: Category; size?: "md" | "lg"; showAr?: boolean;
}) {
  const meta = CATEGORIES[category];
  const Icon = meta.icon;
  return (
    <span className={`cat ${size === "lg" ? "cat-lg" : ""}`}>
      <span className="cat-icon"><Icon size={size === "lg" ? 20 : 15} /></span>
      <span>
        <div className="cat-en">{meta.en}</div>
        {showAr && <div className="cat-ar" dir="rtl">{meta.ar}</div>}
      </span>
    </span>
  );
}

export function CategoryPill({ category }: { category: Category }) {
  const meta = CATEGORIES[category];
  return <Badge tone="neutral" icon={meta.icon}>{meta.en}</Badge>;
}

export function UrgencyBadge({ urgency }: { urgency: 1 | 2 | 3 }) {
  const u = URGENCY[urgency];
  return <Badge tone={u.tone}>{urgency === 3 ? "▲" : urgency === 1 ? "▽" : "●"} {urgency} · {u.label}</Badge>;
}

export function Meter({ value, label }: { value: number; label?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="meter" title={`${pct.toFixed(0)}%`}>
      <div className="meter-track"><div className="meter-fill" style={{ width: `${pct}%` }} /></div>
      <span className="num small ink2">{label ?? `${pct.toFixed(0)}%`}</span>
    </div>
  );
}

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button className="btn btn-sm btn-ghost" onClick={() => {
      navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1400); }).catch(() => {});
    }}>
      {done ? <Check size={14} /> : <Copy size={14} />} {done ? "Copied" : label}
    </button>
  );
}

export function Spinner() {
  return <span className="spinner" aria-label="loading" />;
}
