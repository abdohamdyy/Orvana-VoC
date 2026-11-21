"""Order lookup + the policy arithmetic the chatbot must never leave to an LLM.

OrderService is the boundary to the real order-management system. DemoOrderService reads the
clearly-labelled fixture data/demo/orders.json (fictional orders) so the demo and tests are
deterministic, including a fixed "today".
"""
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from ..config import settings

ORDER_ID_RE = re.compile(r"(?<![A-Za-z0-9-])(?:ORV\s*[-#]?\s*)?#?(\d{5})(?!\d)", re.IGNORECASE)
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def find_order_ids(text: str) -> list[str]:
    """'ORV-10003', 'orv 10003', '#10003', '١٠٠٠٣' -> ['ORV-10003'] (5-digit ids only, so amounts such as
    1950 and masked phone numbers never look like an order)."""
    return [f"ORV-{d}" for d in ORDER_ID_RE.findall(text.translate(_AR_DIGITS))]


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    governorate: str
    promised: date
    delivered: date | None
    status: str  # processing | out_for_delivery | delivered | returned
    payment: str  # card | wallet | cod
    total_egp: int
    items: tuple[str, ...]
    voucher_issued: bool = False
    double_charged: bool = False


@dataclass(frozen=True)
class DelayOutcome:
    days_late: int
    voucher_egp: int  # 0, 50 or 100
    can_cancel: bool
    already_compensated: bool


def assess_delay(order: Order, today: date) -> DelayOutcome:
    """Policy §1: more than 3 days late -> 50 EGP voucher; more than 7 -> 100 EGP + the right to cancel
    for a full refund; one voucher per order. Lateness is measured against the promised date, up to
    delivery (or today if the order is still on its way)."""
    end = order.delivered or today
    days = max(0, (end - order.promised).days)
    voucher = 100 if days > 7 else 50 if days > 3 else 0
    if order.voucher_issued:
        return DelayOutcome(days, 0, days > 7 and order.delivered is None, voucher > 0)
    return DelayOutcome(days, voucher, days > 7 and order.delivered is None, False)


def refund_days(payment: str, double_charge: bool = False) -> int:
    """Policy §2: card 5 working days, wallet / cash on delivery 10; a double charge 3 after bank confirmation."""
    if double_charge:
        return 3
    return 5 if payment == "card" else 10


class OrderService(Protocol):
    today: date

    def lookup(self, order_id: str, customer_id: str) -> Order | None: ...


class DemoOrderService:
    def __init__(self, path: Path | None = None):
        raw = json.loads((path or settings.data_dir / "demo" / "orders.json").read_text(encoding="utf-8"))
        self.today = date.fromisoformat(raw["today"])
        self._orders = {}
        for o in raw["orders"]:
            self._orders[o["order_id"]] = Order(
                order_id=o["order_id"], customer_id=o["customer_id"], governorate=o["governorate"],
                promised=date.fromisoformat(o["promised"]),
                delivered=date.fromisoformat(o["delivered"]) if o.get("delivered") else None,
                status=o["status"], payment=o["payment"], total_egp=o["total_egp"], items=tuple(o["items"]),
                voucher_issued=o.get("voucher_issued", False), double_charged=o.get("double_charged", False))

    def all_orders(self) -> list[Order]:
        """Demo only: the UI lists the fixture so testers know which order numbers to try."""
        return list(self._orders.values())

    def lookup(self, order_id: str, customer_id: str) -> Order | None:
        """Ownership is checked HERE, in code: another customer's order is indistinguishable from a
        non-existent one, so guessing order numbers reveals nothing."""
        order = self._orders.get(order_id)
        return order if order and customer_id and order.customer_id == customer_id else None
