"""Validate Pydantic models in :mod:`sec8k.schema` round-trip against canonical 8-K examples."""

from datetime import date, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from sec8k.schema import Filing8K


def _valid_payload() -> dict[str, Any]:
    return {
        "form_type": "8-K",
        "filer_company": "Acme Corp",
        "filer_ticker": "ACME",
        "filer_cik": "0000320193",
        "filing_date": date(2024, 5, 1),
        "event_date": date(2024, 4, 29),
        "items": ["2.01", "9.01"],
        "primary_category": "m_and_a",
        "counterparties": ["Beta LLC"],
        "monetary_amount": 1_500_000_000.0,
        "currency": "USD",
        "amount_type": "purchase_price",
        "summary": "Acme acquired Beta for $1.5B in cash.",
        "expected_impact_period": "current_quarter",
    }


def test_golden_full_valid_round_trip() -> None:
    payload = _valid_payload()
    model = Filing8K.model_validate(payload)
    reloaded = Filing8K.model_validate_json(model.model_dump_json())
    assert reloaded == model
    assert model.filer_company == "Acme Corp"
    assert model.items == ["2.01", "9.01"]


def test_form_type_rejects_unknown() -> None:
    payload = _valid_payload()
    payload["form_type"] = "8-K/B"
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


@pytest.mark.parametrize("bad_cik", ["123", "12345678901", "abcdefghij", ""])
def test_cik_must_be_10_digits(bad_cik: str) -> None:
    payload = _valid_payload()
    payload["filer_cik"] = bad_cik
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_cik_accepts_zero_padded() -> None:
    payload = _valid_payload()
    payload["filer_cik"] = "0000320193"
    model = Filing8K.model_validate(payload)
    assert model.filer_cik == "0000320193"


def test_items_non_empty() -> None:
    payload = _valid_payload()
    payload["items"] = []
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


@pytest.mark.parametrize("bad_item", ["1.1", "10.01", "a.bc", "1.001", "1-01"])
def test_items_pattern_rejects(bad_item: str) -> None:
    payload = _valid_payload()
    payload["items"] = [bad_item]
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_items_pattern_accepts() -> None:
    payload = _valid_payload()
    payload["items"] = ["1.01", "5.02", "9.99"]
    model = Filing8K.model_validate(payload)
    assert model.items == ["1.01", "5.02", "9.99"]


def test_items_strips_whitespace_before_pattern_check() -> None:
    payload = _valid_payload()
    payload["items"] = [" 1.01 ", "5.02"]
    model = Filing8K.model_validate(payload)
    assert model.items == ["1.01", "5.02"]


def test_items_deduped_order_preserved() -> None:
    payload = _valid_payload()
    payload["items"] = ["5.02", "9.01", "5.02", "2.01"]
    model = Filing8K.model_validate(payload)
    assert model.items == ["5.02", "9.01", "2.01"]


def test_event_date_after_filing_date_rejected() -> None:
    payload = _valid_payload()
    payload["filing_date"] = date(2024, 5, 1)
    payload["event_date"] = date(2024, 5, 1) + timedelta(days=1)
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_event_date_equal_filing_date_ok() -> None:
    payload = _valid_payload()
    payload["event_date"] = payload["filing_date"]
    model = Filing8K.model_validate(payload)
    assert model.event_date == model.filing_date


def test_event_date_none_ok() -> None:
    payload = _valid_payload()
    payload["event_date"] = None
    model = Filing8K.model_validate(payload)
    assert model.event_date is None


@pytest.mark.parametrize(
    ("currency", "amount_type"),
    [
        (None, "purchase_price"),
        ("USD", None),
        (None, None),
    ],
)
def test_monetary_requires_currency_and_amount_type(
    currency: str | None, amount_type: str | None
) -> None:
    payload = _valid_payload()
    payload["currency"] = currency
    payload["amount_type"] = amount_type
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_monetary_all_none_ok() -> None:
    payload = _valid_payload()
    payload["monetary_amount"] = None
    payload["currency"] = None
    payload["amount_type"] = None
    model = Filing8K.model_validate(payload)
    assert model.monetary_amount is None
    assert model.currency is None
    assert model.amount_type is None


def test_monetary_zero_ok() -> None:
    payload = _valid_payload()
    payload["monetary_amount"] = 0.0
    model = Filing8K.model_validate(payload)
    assert model.monetary_amount == 0.0


def test_monetary_negative_rejected() -> None:
    payload = _valid_payload()
    payload["monetary_amount"] = -1.0
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_counterparties_dedup_strip_preserves_order() -> None:
    payload = _valid_payload()
    payload["counterparties"] = ["  Acme  ", "Beta", "Acme", "", "  ", "Beta", "Gamma"]
    model = Filing8K.model_validate(payload)
    assert model.counterparties == ["Acme", "Beta", "Gamma"]


def test_counterparties_empty_ok() -> None:
    payload = _valid_payload()
    payload["counterparties"] = []
    model = Filing8K.model_validate(payload)
    assert model.counterparties == []


def test_counterparties_default_when_omitted() -> None:
    payload = _valid_payload()
    del payload["counterparties"]
    model = Filing8K.model_validate(payload)
    assert model.counterparties == []


def test_summary_whitespace_only_rejected() -> None:
    payload = _valid_payload()
    payload["summary"] = "   "
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_summary_max_length_boundary() -> None:
    payload = _valid_payload()
    payload["summary"] = "x" * 500
    model = Filing8K.model_validate(payload)
    assert len(model.summary) == 500

    payload["summary"] = "x" * 501
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_extra_key_forbidden() -> None:
    payload = _valid_payload()
    payload["hallucinated_field"] = 42
    with pytest.raises(ValidationError):
        Filing8K.model_validate(payload)


def test_empty_returns_valid_sentinel() -> None:
    empty = Filing8K.empty()
    assert empty.filer_ticker is None
    assert empty.event_date is None
    assert empty.monetary_amount is None
    assert empty.currency is None
    assert empty.amount_type is None
    assert empty.expected_impact_period is None
    assert empty.counterparties == []
    assert empty.items == ["0.00"]
    assert empty.filer_cik == "0000000000"
    assert empty.summary == "No extraction."
    assert empty == Filing8K.empty()


def test_empty_round_trips_through_json() -> None:
    empty = Filing8K.empty()
    payload = empty.model_dump_json()
    reloaded = Filing8K.model_validate_json(payload)
    assert reloaded == empty


def test_json_schema_contract() -> None:
    schema = Filing8K.model_json_schema()

    assert schema["additionalProperties"] is False

    required = set(schema["required"])
    expected_required = {
        "form_type",
        "filer_company",
        "filer_cik",
        "filing_date",
        "items",
        "primary_category",
        "summary",
    }
    assert expected_required <= required

    props = schema["properties"]
    assert set(props["form_type"]["enum"]) == {"8-K", "8-K/A"}
    assert set(props["primary_category"]["enum"]) == {
        "m_and_a",
        "executive_change",
        "financial_results",
        "material_agreement",
        "regulatory",
        "other",
    }
    assert props["filer_cik"]["pattern"] == r"^\d{10}$"
    assert props["summary"]["minLength"] == 1
    assert props["summary"]["maxLength"] == 500
    assert props["items"]["minItems"] == 1

    monetary = props["monetary_amount"]
    number_branches = [b for b in monetary.get("anyOf", [monetary]) if b.get("type") == "number"]
    assert number_branches, f"expected a number branch in monetary_amount schema: {monetary!r}"
    assert number_branches[0]["minimum"] == 0
