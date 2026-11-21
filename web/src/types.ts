export type Category =
  | "delivery_delay" | "wrong_item" | "refund_request" | "product_quality" | "price_complaint"
  | "app_issue" | "staff_behavior" | "loyalty_points" | "inquiry";

export interface TicketRecord {
  ticket_id: string;
  category: Category;
  secondary_category: Category | null;
  product_mentioned: string | null;
  store_mentioned: string | null;
  sentiment: "negative" | "neutral" | "positive";
  urgency: 1 | 2 | 3;
  is_valid: boolean;
  injection_flag: boolean;
  confidence: number;
  model_version: string;
}

export interface Segment { text: string; role: "primary" | "secondary"; category: Category; confidence: number }
export interface StoreName { code: string; ar: string; en: string }

export interface ClassifyResult {
  record: TicketRecord;
  details: {
    cleaned_text: string;
    injection_removed: boolean;
    is_valid: boolean;
    segments: Segment[];
    flags: {
      safety: boolean; repeat: boolean; double_charge_egp: number | null;
      safety_report: { where: "main issue" | "second issue"; check: "confirmed" | "needs_confirmation" } | null;
    };
    urgency_reason: string;
    store: StoreName | null;
  };
  reply: string;
  reply_source: "llm" | "template";
}

export interface BatchRow {
  ticket_id: string; category: Category; secondary_category: Category | ""; product_mentioned: string;
  store_mentioned: string; sentiment: string; urgency: 1 | 2 | 3; is_valid: boolean; injection_flag: boolean;
  confidence: number; text: string;
}

export interface BatchResult {
  sample: string | null;
  rows: BatchRow[];
  summary: {
    rows_received: number; valid: number; invalid: number; injection_flagged: number; urgent: number;
    by_category: Record<string, number>; secondary: Record<string, number>; by_urgency: Record<"1" | "2" | "3", number>;
    top_stores: (StoreName & { count: number })[]; top_products: { brand: string; count: number }[];
    elapsed_ms: number; model_version: string;
  };
}

export interface ChatCase {
  category: Category; order_id: string | null; order_verified: boolean; store: string | null; product: string | null;
  safety: boolean; urgency: 1 | 2 | 3; ticket_id: string | null; done: boolean;
  outcome: { days_late?: number; voucher_egp?: number; can_cancel?: boolean; refund_days?: number | null };
}

export interface ChatState {
  status: "open" | "handed_off" | "closed"; lang: "ar" | "en"; turns: number; awaiting: string | null;
  handoff_reason: string | null; cases: ChatCase[];
}

export interface ChatResponse {
  session_id: string; reply: string; source: "llm" | "template"; action: string; facts: string[];
  facts_text: { id: string; text: string }[]; latency_ms: number; state: ChatState;
}

export interface DemoOrder {
  order_id: string; governorate: string; promised: string; delivered: string | null; status: string;
  payment: string; items: string[];
}
export interface DemoCustomers { today: string; note: string; customers: { customer_id: string; orders: DemoOrder[] }[] }
export interface Health { classifier: string; llm: string | null; chat_llm: boolean; chat_phrasing: string }
