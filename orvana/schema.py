"""Output schema: one structured record per complaint (see docs/SCHEMA.md)."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    delivery_delay = "delivery_delay"
    wrong_item = "wrong_item"
    refund_request = "refund_request"
    product_quality = "product_quality"
    price_complaint = "price_complaint"
    app_issue = "app_issue"
    staff_behavior = "staff_behavior"
    loyalty_points = "loyalty_points"
    inquiry = "inquiry"


CATEGORIES = [c.value for c in Category]


class Sentiment(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class TicketRecord(BaseModel):
    ticket_id: str
    category: Category
    secondary_category: Optional[Category] = None
    product_mentioned: Optional[str] = None  # brand exactly as in products_catalog.csv
    store_mentioned: Optional[str] = None  # store_id exactly as in store_names.csv
    sentiment: Sentiment
    urgency: int = Field(ge=1, le=3)
    # operational fields (not scored, used by the report and monitoring)
    is_valid: bool = True  # False for empty messages / pure prompt-injection
    injection_flag: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_version: str = ""

    @model_validator(mode="after")
    def _secondary_differs(self):
        if self.secondary_category == self.category:
            self.secondary_category = None
        return self

    def eval_row(self) -> dict:
        """Row in the format expected by evaluate.py (empty string = none)."""
        return {
            "ticket_id": self.ticket_id,
            "category": self.category.value,
            "secondary_category": self.secondary_category.value if self.secondary_category else "",
            "product_mentioned": self.product_mentioned or "",
            "store_mentioned": self.store_mentioned or "",
            "sentiment": self.sentiment.value,
            "urgency": self.urgency,
        }
