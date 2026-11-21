import { MessageSquare, Moon, Sun, Tags } from "lucide-react";
import { shortModelName as shortName } from "../meta";
import type { Health } from "../types";

export type Page = "chat" | "classify";

export function Sidebar({ page, onNavigate, health, theme, onToggleTheme }: {
  page: Page; onNavigate: (p: Page) => void; health: Health | null; theme: "light" | "dark"; onToggleTheme: () => void;
}) {
  const items: { id: Page; label: string; hint: string; icon: typeof MessageSquare }[] = [
    { id: "chat", label: "Customer chat", hint: "B3", icon: MessageSquare },
    { id: "classify", label: "Classification", hint: "B2", icon: Tags },
  ];
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden><path d="M5 17c2.6 0 2.6-7 5.2-7s2.6 12 5.2 12 2.6-9.6 5.2-9.6 1.9 4.6 3.4 4.6" fill="none" stroke="#fff" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </div>
        <div>
          <div className="brand-name">Orvana</div>
          <div className="brand-sub">Voice of the Customer</div>
        </div>
      </div>
      <div className="nav-label">Workspace</div>
      {items.map(({ id, label, hint, icon: Icon }) => (
        <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => onNavigate(id)}>
          <Icon size={18} /> {label} <span className="nav-hint">{hint}</span>
        </button>
      ))}
      <div className="sidebar-foot">
        <div className="status-line" title={health?.classifier ?? ""}>
          <span className="dot on" /> {health ? shortName(health.classifier) : "connecting…"}
        </div>
        <div className="status-line" title={health?.llm ?? "Ollama is not running: templates and the fine-tuned model only"}>
          <span className={`dot ${health?.llm ? "on" : ""}`} /> {health?.llm ? "Local LLM on" : "Local LLM off"}
        </div>
        <button className="btn btn-sm" onClick={onToggleTheme} aria-label="toggle theme">
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}<span className="hide-sm">{theme === "dark" ? "Light" : "Dark"} mode</span>
        </button>
      </div>
    </aside>
  );
}
