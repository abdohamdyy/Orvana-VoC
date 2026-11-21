import { useEffect, useState } from "react";
import { api } from "./api";
import { Sidebar, type Page } from "./components/Sidebar";
import { ChatPage } from "./pages/ChatPage";
import { ClassifyPage } from "./pages/ClassifyPage";
import type { Health } from "./types";

interface Route { page: Page; params: URLSearchParams }

function parseHash(): Route {
  const [path, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
  return { page: path === "classify" ? "classify" : "chat", params: new URLSearchParams(query) };
}

function storedTheme(): "light" | "dark" {
  try {
    const t = localStorage.getItem("orvana-theme");
    if (t === "light" || t === "dark") return t;
  } catch { /* storage blocked: fall back to the system setting */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const HEADERS: Record<Page, { title: string; sub: string }> = {
  chat: { title: "Customer chat", sub: "The code decides vouchers, refunds, escalation and hand-over; the bot only words them. Nothing is stored." },
  classify: { title: "Classification", sub: "Turn any complaint into a structured record, with the reasons. Or upload a week of complaints." },
};

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [health, setHealth] = useState<Health | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">(storedTheme);

  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => { api.health().then(setHealth).catch(() => setHealth(null)); }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("orvana-theme", theme); } catch { /* not persisted: fine */ }
  }, [theme]);

  const navigate = (p: Page) => { window.location.hash = `#/${p}`; };
  const head = HEADERS[route.page];
  const p = route.params;

  return (
    <div className="app">
      <Sidebar page={route.page} onNavigate={navigate} health={health} theme={theme}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} />
      <main className="main">
        <header className="page-head">
          <div>
            <h1 className="page-title">{head.title}</h1>
            <p className="page-sub">{head.sub}</p>
          </div>
        </header>
        <div className="page-body">
          {route.page === "chat"
            ? <ChatPage key={`chat-${window.location.hash}`} initial={{ say: p.get("say") ?? undefined, as: p.get("as") ?? undefined }} />
            : <ClassifyPage key={`cls-${window.location.hash}`} initial={{ q: p.get("q") ?? undefined, sample: p.get("sample") ?? undefined }} />}
        </div>
      </main>
    </div>
  );
}
