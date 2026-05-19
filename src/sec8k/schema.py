"""Pydantic v2 models for 8-K Items, extracted fields, and FastAPI request/response payloads.

Single source of truth for the extraction schema. Any change here is a contract change for
training data, the inference server, the eval harness, and downstream consumers, so it
must land in its own PR with a migration note.
"""

import re
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ITEM_RE = re.compile(r"^\d\.\d\d$")


class Filing8K(BaseModel):
    """Extracted fields from a single SEC 8-K (or 8-K/A) filing."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        json_schema_extra={
            "title": "Filing8K",
            "description": "Extracted SEC 8-K filing fields.",
        },
    )

    form_type: Literal["8-K", "8-K/A"] = Field(
        description="SEC form code; '8-K' for original, '8-K/A' for amendment.",
    )
    filer_company: str = Field(
        min_length=1,
        description="Legal entity name of the filer.",
    )
    filer_ticker: str | None = Field(
        default=None,
        description="Stock ticker symbol if the filer is publicly listed; null otherwise.",
    )
    filer_cik: str = Field(
        pattern=r"^\d{10}$",
        description="10-digit zero-padded SEC Central Index Key.",
    )
    filing_date: date = Field(
        description="Date the 8-K was filed with the SEC.",
    )
    event_date: date | None = Field(
        default=None,
        description=(
            "Date the disclosed event occurred; null if undisclosed. "
            "Must be on or before filing_date."
        ),
    )
    items: list[str] = Field(
        min_length=1,
        description=(
            "SEC 8-K item codes (e.g., '1.01', '5.02'); each matches ^\\d\\.\\d\\d$. "
            "Non-empty; duplicates removed, insertion order preserved."
        ),
    )
    primary_category: Literal[
        "m_and_a",
        "executive_change",
        "financial_results",
        "material_agreement",
        "regulatory",
        "other",
    ] = Field(
        description=(
            "Dominant business event type. "
            "'m_and_a' = merger/acquisition/divestiture; "
            "'executive_change' = appointment/departure/compensation; "
            "'financial_results' = earnings/guidance; "
            "'material_agreement' = entry/termination of a material contract; "
            "'regulatory' = SEC/legal/compliance matter; "
            "'other' = none of the above."
        ),
    )
    counterparties: list[str] = Field(
        default_factory=list,
        description=(
            "Other named parties (acquired company, departing executive, lender, etc.); "
            "empty list if none. Stripped and deduplicated, insertion order preserved."
        ),
    )
    monetary_amount: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Primary dollar (or stated-currency) amount discussed, in whole units "
            "(not cents). Magnitude only; sign is carried by amount_type."
        ),
    )
    currency: Literal["USD", "EUR", "GBP", "CAD", "JPY"] | None = Field(
        default=None,
        description=(
            "ISO currency code for monetary_amount; required when monetary_amount is set."
        ),
    )
    amount_type: (
        Literal[
            "purchase_price",
            "severance",
            "dividend",
            "settlement",
            "loss",
            "gain",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Semantic role of monetary_amount; required when monetary_amount is set. "
            "'purchase_price' for M&A consideration, 'severance' for executive separation, "
            "'dividend' for distributions, 'settlement' for legal payouts, "
            "'loss'/'gain' for one-time charges or credits (magnitude-only), 'other' otherwise."
        ),
    )
    summary: str = Field(
        min_length=1,
        max_length=500,
        description="Factual 1-3 sentence summary of the event (max 500 characters).",
    )
    expected_impact_period: (
        Literal[
            "immediate",
            "current_quarter",
            "current_fiscal_year",
            "future",
            "undisclosed",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "When the financial impact materializes. "
            "'immediate' = recognized at signing/closing of this disclosure; "
            "'current_quarter' = within the quarter of filing_date; "
            "'current_fiscal_year' = within the fiscal year of filing_date; "
            "'future' = later periods; "
            "'undisclosed' = not stated."
        ),
    )

    @field_validator("items", mode="after")
    @classmethod
    def _validate_items(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for idx, code in enumerate(v):
            if not _ITEM_RE.fullmatch(code):
                raise ValueError(f"items[{idx}]={code!r} must match ^\\d\\.\\d\\d$")
            if code not in seen:
                seen.add(code)
                out.append(code)
        return out

    @field_validator("counterparties", mode="after")
    @classmethod
    def _normalize_counterparties(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in v:
            s = raw.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @model_validator(mode="after")
    def _cross_field(self) -> Self:
        if self.event_date is not None and self.event_date > self.filing_date:
            raise ValueError("event_date must be on or before filing_date")
        if self.monetary_amount is not None and (self.currency is None or self.amount_type is None):
            raise ValueError("monetary_amount requires both currency and amount_type")
        return self

    @classmethod
    def empty(cls) -> Self:
        """Sentinel instance signalling 'no extraction' / parser failure to downstream consumers."""
        return cls(
            form_type="8-K",
            filer_company="UNKNOWN",
            filer_cik="0000000000",
            filing_date=date(1970, 1, 1),
            items=["0.00"],
            primary_category="other",
            summary="No extraction.",
        )
