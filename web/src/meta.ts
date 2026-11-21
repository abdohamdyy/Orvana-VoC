import {
  CircleHelp, PackageX, RotateCcw, ShieldAlert, Smartphone, Star, Tag, Truck, UserX, type LucideIcon,
} from "lucide-react";
import type { Category } from "./types";

// Nine categories is more than a colour palette can keep apart, so identity is an icon + a label,
// never colour alone (dataviz rule). Charts use one hue.
export const CATEGORIES: Record<Category, { en: string; ar: string; icon: LucideIcon }> = {
  delivery_delay: { en: "Delivery delay", ar: "تأخير التوصيل", icon: Truck },
  wrong_item: { en: "Wrong item", ar: "منتج غلط", icon: PackageX },
  refund_request: { en: "Refund", ar: "استرداد فلوس", icon: RotateCcw },
  product_quality: { en: "Product quality", ar: "جودة المنتج", icon: ShieldAlert },
  price_complaint: { en: "Price", ar: "السعر", icon: Tag },
  app_issue: { en: "App issue", ar: "مشكلة الأبلكيشن", icon: Smartphone },
  staff_behavior: { en: "Staff behaviour", ar: "سلوك الموظفين", icon: UserX },
  loyalty_points: { en: "Loyalty points", ar: "نقط Orvana Plus", icon: Star },
  inquiry: { en: "Inquiry", ar: "استفسار", icon: CircleHelp },
};

export function shortModelName(name: string): string {
  if (name.startsWith("cascade")) return "e5-small + LLM on disagreement";
  if (name.startsWith("finetuned")) return "fine-tuned e5-small (ONNX)";
  if (name.startsWith("tfidf")) return "TF-IDF baseline";
  if (name.startsWith("ollama")) return "local LLM";
  return name;
}

export const URGENCY: Record<1 | 2 | 3, { label: string; tone: "muted" | "neutral" | "critical" }> = {
  1: { label: "Low", tone: "muted" },
  2: { label: "Normal", tone: "neutral" },
  3: { label: "Urgent", tone: "critical" },
};

export const HANDOFF_REASONS: Record<string, string> = {
  customer_request: "customer asked for a human",
  safety: "safety: escalated to the Quality team",
  urgent: "urgent case",
  repeat_complaint: "repeat complaint",
  no_data: "no data on this topic",
  order_not_found: "order not found twice",
  missing_info: "details still missing",
  customer_insists: "customer insists after the answer",
  repeated_question: "same question again",
  unclear: "still unclear after 2 questions",
  abuse: "abusive messages",
  repeated_injection: "repeated prompt injection",
  turn_limit: "conversation too long",
};

export const SLOTS: Record<string, string> = {
  order_id: "order number", store: "branch", product: "product / brand", details: "more details",
};

export const ACTIONS: Record<string, string> = {
  ask: "asks", answer: "answers", resolve: "resolved", handoff: "hand-over", greet: "greeting", close: "closed",
  clarify: "clarifies", refuse_data: "refused", after_handoff: "with agent", reply: "reply",
};
