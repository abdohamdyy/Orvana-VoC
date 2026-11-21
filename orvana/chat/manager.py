"""Dialogue manager: a deterministic state machine. It decides EVERYTHING that matters (which policy facts
apply, the voucher, the refund time, escalation, hand-over to a human, tickets); respond.py only words it.

One conversation = ChatState. Each customer complaint becomes a Case that collects the details its
category needs (order number, branch, product), at most 2 questions per detail, then is resolved:
policy facts + numbers computed in code + a ticket. Safety, urgent and stuck cases go to a human.
"""
import secrets
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..config import settings
from ..extractors import double_charge_amount, find_product, find_store, safety_check
from ..rules import sentiment
from ..schema import Category, TicketRecord
from .kb import FACT_CONCEPTS
from .nlu import Understanding, hazard_answer, understand
from .orders import DemoOrderService, Order, assess_delay, refund_days
from .respond import Plan, phrase, render, store_name

NEEDS = {"delivery_delay": "order_id", "refund_request": "order_id", "wrong_item": "order_id",
         "staff_behavior": "store", "price_complaint": "store", "product_quality": "product"}
MAX_ASKS = 2


@lru_cache(maxsize=1)
def _brand_types() -> dict[str, str]:
    df = pd.read_csv(settings.data_dir / "products_catalog.csv")
    return {r.brand: r.category for r in df.itertuples()}


def returns_fact(brand: str | None) -> str:
    """Policy §2 by product type (catalog): electrical goods 7 days for a defect + warranty; food and
    detergents only if expired/defective; everything else 14 days."""
    kind = _brand_types().get(brand or "", "")
    return ("returns.electronics" if kind in ("Electronics", "Home & Kitchen")
            else "returns.food" if kind == "FMCG" else "returns.window")


@dataclass
class Case:
    category: str
    text: str
    product: str | None = None
    store: str | None = None
    order_id: str | None = None
    order: Order | None = None
    order_checked: bool = False
    amount: int | None = None  # double-charge amount in EGP
    safety: bool = False
    safety_check: str | None = None  # confirmed | needs_confirmation (the product type rarely has this hazard)
    repeat: bool = False
    urgency: int = 2
    confidence: float = 1.0
    injection: bool = False
    lookup_fails: int = 0
    gave_up: bool = False  # customer cannot / will not give the missing detail
    acked: bool = False
    asks: Counter = field(default_factory=Counter)
    outcome: dict = field(default_factory=dict)  # what the code computed: voucher, refund days ...
    ticket_id: str | None = None
    done: bool = False


@dataclass
class ChatState:
    session_id: str
    customer_id: str | None = None  # from the app login; None = guest (orders cannot be looked up)
    lang: str = "ar"
    turns: int = 0
    status: str = "open"  # open | handed_off | closed
    handoff_reason: str | None = None
    cases: list[Case] = field(default_factory=list)
    awaiting: str | None = None
    strikes: Counter = field(default_factory=Counter)  # abuse, injection, insist, clarify, repeat_answer
    given: Counter = field(default_factory=Counter)  # policy facts already given in answers
    last_bot: str = ""

    @property
    def current(self) -> Case | None:
        return next((c for c in self.cases if not c.done), None)

    def snapshot(self) -> dict:
        return {"status": self.status, "lang": self.lang, "turns": self.turns, "awaiting": self.awaiting,
                "handoff_reason": self.handoff_reason,
                "cases": [{"category": c.category, "order_id": c.order_id, "order_verified": c.order is not None,
                           "store": c.store, "product": c.product, "safety": c.safety,
                           "safety_check": c.safety_check, "urgency": c.urgency,
                           "outcome": c.outcome, "ticket_id": c.ticket_id, "done": c.done} for c in self.cases]}


@dataclass
class TurnResult:
    reply: str
    source: str  # llm | template
    plan: Plan
    draft: str
    problems: list[str]
    latency_ms: float
    state: dict


class ChatBot:
    def __init__(self, backend, llm=None, orders=None, sink=None, phrasing: str | None = None):
        self.backend, self.llm, self.sink = backend, llm, sink
        self.phrasing = phrasing or settings.chat_phrasing  # template | opener | rewrite (see respond.phrase)
        self.orders = orders or DemoOrderService()

    def new_state(self, customer_id: str | None = None, session_id: str | None = None) -> ChatState:
        return ChatState(session_id=session_id or secrets.token_urlsafe(16), customer_id=customer_id or None)

    # ------------------------------------------------------------------ one turn
    def turn(self, st: ChatState, message: str) -> TurnResult:
        t0 = time.perf_counter()
        st.turns += 1
        u = understand(message, self.backend)
        if u.lang:
            st.lang = "en" if u.lang == "en" else "ar"
        plan = Plan(lang=st.lang)
        self._decide(st, u, plan)
        if plan.handoff:
            plan.action = "handoff"
        elif plan.action == "none":
            plan.action = ("resolve" if plan.tickets else "ask" if plan.ask else
                           "answer" if any(p.kind == "answer" for p in plan.parts) else "reply")
        if st.status == "open" and not plan.ask and plan.action not in ("greet", "close"):
            plan.add("anything_else")
        draft = render(plan)
        llm = None if plan.action in ("greet", "close", "after_handoff") else self.llm
        reply, source, problems = phrase(plan, draft, u.text, st.last_bot, llm, self.phrasing, first_turn=st.turns == 1)
        st.last_bot = reply
        return TurnResult(reply, source, plan, draft, problems, (time.perf_counter() - t0) * 1000, st.snapshot())

    def _decide(self, st: ChatState, u: Understanding, plan: Plan) -> None:
        if st.status == "handed_off":
            plan.action = "after_handoff"
            plan.add("after_handoff", ["escalation.first_reply"], ticket_id=self._last_ticket(st))
            return
        st.status = "open"  # a message after "bye" re-opens the chat
        if st.turns > settings.chat_max_turns:
            return self._handoff(st, plan, "turn_limit", u)
        if u.injection:
            st.strikes["injection"] += 1
        if "human" in u.acts:
            return self._handoff(st, plan, "customer_request", u)
        if "abuse" in u.acts:
            st.strikes["abuse"] += 1
            if st.strikes["abuse"] >= 2:
                return self._handoff(st, plan, "abuse", u)
            plan.add("abuse")
        if "data" in u.acts:
            plan.action = "refuse_data"
            return plan.add("refuse_data")
        if u.acts & {"thanks", "bye"} and u.content_words() <= 2:
            return self._close(st, plan)
        if "greet" in u.acts and u.content_words() == 0:
            plan.action = "greet"
            return plan.add("greet")
        slot = st.awaiting
        consumed = self._fill_slot(st, u)
        # the answer to the safety question is about the open case, never a new complaint ("لا هو بس بطل يشتغل")
        if consumed and (slot == "hazard" or u.record is None or u.content_words() <= 3 and "question" not in u.acts):
            return self._advance(st, plan, u)
        if not u.valid:  # empty, a lone insult, or pure prompt injection
            if st.strikes["injection"] >= 3:
                return self._handoff(st, plan, "repeated_injection", u)
            return self._continue_or_clarify(st, plan, u)
        if self._correction(st, u, plan):
            return self._advance(st, plan, u)
        self._new_intent(st, u, plan)
        if st.status == "open":
            self._advance(st, plan, u)

    # ------------------------------------------------------------------ understanding the new message
    def _fill_slot(self, st: ChatState, u: Understanding) -> bool:
        case, slot = st.current, st.awaiting
        if not case or not slot:
            return False
        if slot == "hazard":
            answer = "unsure" if "dont_know" in u.acts else hazard_answer(u.text)
            if answer is None:
                return False  # unclear: asked again (at most MAX_ASKS times), then treated as a hazard
            if answer == "no":
                case.safety, case.safety_check, case.urgency = False, None, 2  # faulty or warranty: a normal case
            else:
                case.safety_check = "confirmed"  # a hazard, or the customer cannot tell: safety first
            st.awaiting = None
            return True
        if slot == "order_id" and u.order_ids:
            case.order_id, case.order_checked = u.order_ids[-1], False
        elif slot == "store" and u.store:
            case.store = u.store
        elif slot == "product" and u.product:
            case.product = u.product
        elif slot != "details" and "dont_know" in u.acts:
            case.gave_up = True
        else:
            return False
        st.awaiting = None
        return True

    def _correction(self, st: ChatState, u: Understanding, plan: Plan) -> bool:
        """'لا مش المعادي، فرع الدقي': a branch correction updates the last case that had a branch."""
        if "correction" not in u.acts or not u.store:
            return False
        case = next((c for c in reversed(st.cases) if c.store and c.store != u.store), None)
        if case is None:
            return False
        case.store = u.store
        plan.add("corrected", store=store_name(u.store, st.lang))
        return True

    def _new_intent(self, st: ChatState, u: Understanding, plan: Plan) -> None:
        if "vague" in u.acts:
            return self._clarify(st, plan, u)
        tau = settings.chat_answer_threshold
        top_score = u.facts[0][1] if u.facts else 0.0
        known = [(f, s) for f, s in u.facts if f.known and s >= tau]
        # "no data, here is a human" only when the message really names that topic (its concept), not on
        # character overlap alone ("hi there" must not land on "are you hiring")
        unknown_top = (bool(u.facts) and not u.facts[0][0].known and top_score >= tau
                       and bool(FACT_CONCEPTS.get(u.facts[0][0].id, set()) & u.concepts))
        answered = False
        # answer what we can, even when part of the question is a topic we have no data for: when the best
        # match is a no-data topic, a known fact is still answered if the question covers all its concepts
        # ("delivery time AND fee": delivery.times = {deliver} is answered, the fee is not)
        if u.facts and not u.facts[0][0].known:
            known = [(f, s) for f, s in known if FACT_CONCEPTS.get(f.id) and FACT_CONCEPTS[f.id] <= u.concepts]
        rec = u.record
        cat = rec.category.value if rec is not None else None
        complaint = (rec is not None and cat != "inquiry" and ("problem" in u.acts or "question" not in u.acts)
                     and u.has_evidence())
        if u.safety:  # safety beats everything: never answered as a question or ignored
            complaint, cat = True, "product_quality"
        elif u.order_ids and st.customer_id and not complaint:  # "where is ORV-10004?" = an order-status request
            complaint, cat = True, cat if cat not in (None, "inquiry") else "delivery_delay"
        # a question, or a statement about a policy topic that is not a complaint ("أرجو معرفة الضمان ...")
        asking = "question" in u.acts or (not complaint and "problem" not in u.acts)
        if asking and known:
            first = known[0]
            facts = [first[0].id] + [f.id for f, s in known[1:2] if s >= 0.9 * first[1]]
            if u.product and facts[0].startswith("returns."):  # the brand decides which return rule applies
                facts = [returns_fact(u.product)]
            if all(st.given[f] for f in facts):  # the same answer again: say so, then hand over
                st.strikes["repeat_answer"] += 1
                if st.strikes["repeat_answer"] >= 2:
                    return self._handoff(st, plan, "repeated_question", u)
                plan.add("as_said")
            plan.add("answer", facts)
            st.given.update(facts)
            answered = True
        if "compensation" in u.acts and "compensation.vouchers" not in plan.fact_ids():
            plan.add("compensation", ["compensation.vouchers"])
            answered = True
        if complaint:
            return self._open_cases(st, u, plan, cat)
        # a no-data topic the question really asks about: it is the best match, or the question has all its concepts
        unknown_asked = any(not f.known and s >= tau and FACT_CONCEPTS.get(f.id, set()) <= u.concepts and FACT_CONCEPTS.get(f.id)
                            for f, s in u.facts)
        if unknown_top or (answered and unknown_asked):
            plan.add("unknown")
            return self._handoff(st, plan, "no_data", u)
        if answered or st.current is not None:  # answered, or keep collecting the open case's details
            return
        if "greet" in u.acts:  # small talk ("Hi! How are you?")
            plan.action = "greet"
            return plan.add("greet")
        self._clarify(st, plan, u)  # off-topic or unclear: say what we can help with; a human after two tries

    def _clarify(self, st: ChatState, plan: Plan, u: Understanding) -> None:
        st.strikes["clarify"] += 1
        if st.strikes["clarify"] > MAX_ASKS:
            return self._handoff(st, plan, "unclear", u)
        plan.action = "clarify"
        if st.strikes["clarify"] == 1:
            plan.add("capabilities")
        self._ask(st, plan, None, "details")

    def _open_cases(self, st: ChatState, u: Understanding, plan: Plan, primary: str) -> None:
        rec = u.record
        cats = [primary] + ([rec.secondary_category.value] if rec is not None and rec.secondary_category else [])
        new = []
        for i, cat in enumerate(cats):
            seg = u.segments[i] if i < len(u.segments) else u.text
            old = next((c for c in st.cases if c.category == cat), None)
            same_order = not u.order_ids or u.order_ids[-1] == (old.order_id if old else None)
            if old is not None and old.done and same_order:  # same issue again after we answered it
                st.strikes["insist"] += 1
                plan.add("noted")
                return self._handoff(st, plan, "customer_insists", u)
            if old is not None and not old.done:
                continue
            product = find_product(seg) or (u.product if i == 0 else None)
            check = safety_check(seg, product)
            case = Case(category=cat, text=seg, injection=u.injection,
                        confidence=rec.confidence if rec is not None else 1.0, product=product,
                        store=find_store(seg) or (u.store if len(cats) == 1 else None),
                        order_id=u.order_ids[-1] if u.order_ids and i == 0 and NEEDS.get(cat) == "order_id" else None,
                        amount=double_charge_amount(seg), safety=check is not None, safety_check=check, repeat=u.repeat,
                        urgency=rec.urgency if rec is not None and i == 0 else 2)
            st.cases.append(case)
            new.append(case)
        if u.safety and new and not any(c.safety for c in new):  # the safety words were outside the segments
            new[0].safety = True
            new[0].safety_check = safety_check(u.text, new[0].product) or "confirmed"

    # ------------------------------------------------------------------ moving cases forward
    def _advance(self, st: ChatState, plan: Plan, u: Understanding) -> None:
        while (case := st.current) is not None and st.status == "open":
            if not case.acked and case.category in Category.__members__ and case.category != "inquiry":
                plan.add("ack", category=case.category)
                case.acked = True
            if case.safety and case.safety_check == "needs_confirmation":
                # an "expired" phone, milk that "gets very hot": ask before escalating as a hazard
                if case.asks["hazard"] < MAX_ASKS:
                    return self._ask(st, plan, case, "hazard")
                case.safety_check = "confirmed"  # still unclear after two tries: safety first
            if case.safety:
                plan.add("safety_first")
                plan.add("escalate_quality", ["escalation.safety"])
                plan.escalate.append("quality")
                case.urgency = 3
                self._ticket(st, case, plan)
                return self._handoff(st, plan, "safety", u)
            if case.order_id and not case.order_checked:
                case.order_checked = True
                case.order = self.orders.lookup(case.order_id, st.customer_id) if st.customer_id else None
                if case.order is None and st.customer_id:
                    case.lookup_fails += 1
                    plan.add("not_found", order_id=case.order_id)
                    case.order_id = None
                    if case.lookup_fails >= MAX_ASKS:
                        self._ticket(st, case, plan)
                        return self._handoff(st, plan, "order_not_found", u)
                elif case.order is None:
                    plan.add("unverified", order_id=case.order_id)
            slot = NEEDS.get(case.category)
            if slot and not getattr(case, slot) and not case.gave_up and case.asks[slot] < MAX_ASKS:
                return self._ask(st, plan, case, slot)
            self._resolve(st, case, plan, u)
        if st.awaiting != "details":
            st.awaiting = None

    def _ask(self, st: ChatState, plan: Plan, case: Case | None, slot: str) -> None:
        if case is not None:
            case.asks[slot] += 1
        st.awaiting, plan.ask = slot, slot

    def _resolve(self, st: ChatState, case: Case, plan: Plan, u: Understanding) -> None:
        cat, o = case.category, case.order
        if cat == "delivery_delay" and o is not None:
            d = assess_delay(o, self.orders.today)
            case.outcome = {"days_late": d.days_late, "voucher_egp": d.voucher_egp, "can_cancel": d.can_cancel}
            facts = ["delivery.one_voucher"] if d.already_compensated else []
            plan.add("delay", facts, order_id=o.order_id, days_late=d.days_late, voucher_egp=d.voucher_egp,
                     can_cancel=d.can_cancel, already_compensated=d.already_compensated,
                     delivered_on=o.delivered.isoformat() if o.delivered else None, promised=o.promised.isoformat())
        elif cat == "delivery_delay":
            plan.add("facts", ["delivery.times"])
        elif cat == "refund_request":
            double = case.amount is not None or (o is not None and o.double_charged)
            days = refund_days(o.payment if o else "", double) if (o or double) else None
            case.outcome = {"refund_days": days}
            fact = ["refund.double_charge"] if double else ([] if o else ["refund.timing"])
            if o is not None and not double:
                plan.add("refund", order_id=o.order_id, payment=o.payment, days=days)
            else:
                plan.add("facts", fact)
            amount = case.amount or (o.total_egp if o is not None and o.double_charged else 0)
            if double and amount >= settings.double_charge_urgent_egp:
                case.urgency = 3
        elif cat == "staff_behavior":
            plan.add("escalate_manager", ["escalation.staff"])
            plan.escalate.append("store_manager")
        elif cat == "product_quality":
            plan.add("facts", [returns_fact(case.product)])
        elif cat == "price_complaint":
            promo = any(w in case.text.lower() for w in ("عرض", "خصم", "promo", "offer", "discount", "3ard"))
            plan.add("facts", ["prices.promos" if promo else "prices.shelf"])
        else:
            plan.add("facts", {"loyalty_points": ["loyalty.earn"], "app_issue": ["app.followup"],
                               "wrong_item": ["wrong_item.replace"]}.get(cat, []))
        self._ticket(st, case, plan)
        if case.urgency == 3 or case.repeat:
            return self._handoff(st, plan, "urgent" if case.urgency == 3 else "repeat_complaint", u)
        if NEEDS.get(cat) and not getattr(case, NEEDS[cat]) and (case.gave_up or case.asks[NEEDS[cat]] >= MAX_ASKS):
            return self._handoff(st, plan, "missing_info", u)

    # ------------------------------------------------------------------ tickets, hand-over, closing
    def _ticket(self, st: ChatState, case: Case, plan: Plan) -> str:
        case.done = True
        if case.ticket_id:
            return case.ticket_id
        case.ticket_id = "CH-" + "".join(secrets.choice(string.ascii_uppercase) for _ in range(6))
        record = TicketRecord(ticket_id=case.ticket_id, category=Category(case.category), product_mentioned=case.product,
                              store_mentioned=case.store, sentiment=sentiment(Category(case.category)), urgency=case.urgency,
                              injection_flag=case.injection, confidence=case.confidence,
                              model_version=f"chat:{getattr(self.backend, 'name', 'none')}")
        if self.sink is not None:
            self.sink.write(st, case, record)
        plan.tickets.append(case.ticket_id)
        plan.add("ticket", ticket_id=case.ticket_id)
        return case.ticket_id

    def _last_ticket(self, st: ChatState) -> str:
        return next((c.ticket_id for c in reversed(st.cases) if c.ticket_id), "")

    def _handoff(self, st: ChatState, plan: Plan, reason: str, u: Understanding) -> None:
        for case in [c for c in st.cases if not c.done]:  # the agent gets everything still open
            self._ticket(st, case, plan)
        if not any(c.ticket_id for c in st.cases):  # e.g. "I want a human" before any complaint
            case = Case(category="inquiry", text=u.text, urgency=1, injection=u.injection)
            st.cases.append(case)
            self._ticket(st, case, plan)
        st.status, st.handoff_reason, st.awaiting = "handed_off", reason, None
        plan.handoff = True
        plan.add("handoff", ["escalation.first_reply"])

    def _close(self, st: ChatState, plan: Plan) -> None:
        for case in [c for c in st.cases if not c.done]:  # leaving with an open case: an agent follows up
            self._ticket(st, case, plan)
        st.status, st.awaiting, plan.action = "closed", None, "close"
        plan.add("close")

    def _continue_or_clarify(self, st: ChatState, plan: Plan, u: Understanding) -> None:
        case = st.current
        if case is not None and st.awaiting and st.awaiting != "details" and case.asks[st.awaiting] < MAX_ASKS:
            return self._ask(st, plan, case, st.awaiting)
        if case is not None and st.awaiting == "hazard":  # no clear answer twice: escalate as a hazard (safety first)
            st.awaiting = None
            return self._advance(st, plan, u)
        self._clarify(st, plan, u)
