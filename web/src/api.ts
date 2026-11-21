import type { BatchResult, ChatResponse, ClassifyResult, DemoCustomers, Health } from "./types";

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data: unknown;
  try { data = await res.json(); } catch { throw new Error(`Server error (${res.status})`); }
  if (!res.ok || (data && typeof data === "object" && "error" in data)) {
    throw new Error(String((data as { error?: string }).error ?? `Server error (${res.status})`));
  }
  return data as T;
}

export const api = {
  health: () => call<Health>("/api/health"),
  demoCustomers: () => call<DemoCustomers>("/api/demo/customers"),
  classify: (text: string) => call<ClassifyResult>("/api/classify", { text }),
  batch: (payload: { csv?: string; sample?: string }) => call<BatchResult>("/api/classify/batch", payload),
  chat: (session_id: string | null, customer_id: string, message: string) =>
    call<ChatResponse>("/api/chat", { session_id, customer_id, message }),
  resetChat: (session_id: string) => call<{ ok: boolean }>("/api/chat/reset", { session_id }),
};
