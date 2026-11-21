import type { ReactNode } from "react";

// Ranked horizontal bars in ONE hue (dataviz: part-to-whole with more than 6 parts -> ranked bars, one
// hue, largest first; value + share as a direct label; a hover tooltip per bar; the table is the full view).
export interface BarItem { key: string; label: ReactNode; value: number; tooltip?: string }

export function BarList({ items, total, ranked = true, compact = false }: {
  items: BarItem[]; total: number; ranked?: boolean; compact?: boolean;
}) {
  const rows = ranked ? [...items].sort((a, b) => b.value - a.value) : items;
  const max = Math.max(1, ...rows.map((r) => r.value));
  if (!rows.length) return <div className="muted small">Nothing to show.</div>;
  return (
    <div className={`barlist ${compact ? "compact" : ""}`} role="list">
      {rows.map((r) => {
        const share = total ? (100 * r.value) / total : 0;
        return (
          <div className="bar-row" key={r.key} role="listitem">
            <div className="bar-label">{r.label}</div>
            <div className="bar-track"><div className="bar-fill" style={{ width: `${(100 * r.value) / max}%` }} /></div>
            <div className="bar-value num">{r.value}<small>{share.toFixed(0)}%</small></div>
            <div className="tip">{r.tooltip ?? `${r.value} of ${total} (${share.toFixed(1)}%)`}</div>
          </div>
        );
      })}
    </div>
  );
}
