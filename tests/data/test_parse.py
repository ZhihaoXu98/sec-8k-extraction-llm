"""Parses recorded 8-K fixtures and asserts Item-section extraction is correct."""

from __future__ import annotations

from pathlib import Path

import pytest

from sec8k.data.parse import (
    detect_encoding,
    extract_items,
    html_to_text,
    strip_sgml_envelope,
)


def test_detect_encoding_from_content_type() -> None:
    enc = detect_encoding({"Content-Type": "text/html; charset=ISO-8859-1"}, b"")
    assert enc == "iso-8859-1"


def test_detect_encoding_from_meta() -> None:
    body = b"<html><head><meta charset=\"utf-16\"></head></html>"
    enc = detect_encoding({}, body)
    assert enc == "utf-16"


def test_detect_encoding_fallback_utf8() -> None:
    enc = detect_encoding({}, b"<html></html>")
    assert enc == "utf-8"


def test_strip_sgml_envelope_picks_correct_doc(edgar_fixtures_dir: Path) -> None:
    raw = (edgar_fixtures_dir / "sample_sgml_envelope.txt").read_text()
    inner = strip_sgml_envelope(raw, "8-K")
    assert "Earnings body" in inner
    assert "Press release exhibit body" not in inner
    assert "Item 2.02" in inner


def test_strip_sgml_envelope_no_envelope_passthrough() -> None:
    html = "<html><body>plain</body></html>"
    assert strip_sgml_envelope(html, "8-K") == html


def test_html_to_text_preserves_paragraphs(edgar_fixtures_dir: Path) -> None:
    html = (edgar_fixtures_dir / "sample_earnings.html").read_text()
    text = html_to_text(html)
    assert "\n\n" in text
    # Two paragraphs are separated by blank line.
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 4


def test_html_to_text_strips_nav_and_footer(edgar_fixtures_dir: Path) -> None:
    html = (edgar_fixtures_dir / "sample_earnings.html").read_text()
    text = html_to_text(html)
    assert "Top navigation" not in text
    assert "EDGAR page header banner" not in text
    assert "SEC footer disclaimer" not in text


def test_html_to_text_renders_table_pipes(edgar_fixtures_dir: Path) -> None:
    html = (edgar_fixtures_dir / "sample_earnings.html").read_text()
    text = html_to_text(html)
    assert "Revenue | $90.8 billion" in text


def test_html_to_text_preserves_xbrl_inline_text() -> None:
    html = (
        '<html><body><p>Revenue was '
        '<ix:nonNumeric name="rev">$90.8 billion</ix:nonNumeric> this quarter.</p></body></html>'
    )
    text = html_to_text(html)
    assert "$90.8 billion" in text


def test_html_to_text_drops_script_and_style() -> None:
    html = (
        "<html><head><style>body{color:red}</style>"
        "<script>alert('x')</script></head>"
        "<body><p>Visible.</p></body></html>"
    )
    text = html_to_text(html)
    assert "Visible" in text
    assert "alert" not in text
    assert "color:red" not in text


def test_extract_items_simple() -> None:
    text = "Item 2.02. Results of Operations.\nItem 9.01. Exhibits."
    assert extract_items(text) == ["2.02", "9.01"]


def test_extract_items_case_insensitive() -> None:
    assert extract_items("ITEM 5.02 stuff") == ["5.02"]


def test_extract_items_strips_subsection_letter() -> None:
    assert extract_items("Item 5.02(b) compensatory arrangement") == ["5.02"]


def test_extract_items_deduped_order_preserved() -> None:
    text = "Item 2.01 first. Item 1.01 second. Item 2.01 again. Item 9.01 last."
    assert extract_items(text) == ["2.01", "1.01", "9.01"]


def test_extract_items_empty_when_none() -> None:
    assert extract_items("nothing to see here") == []


@pytest.mark.parametrize(
    ("fixture", "expected_items"),
    [
        ("sample_earnings.html", ["2.02", "9.01"]),
        ("sample_merger.html", ["1.01", "2.01", "9.01"]),
        ("sample_ceo.html", ["5.02"]),
        ("sample_amendment.html", ["4.02", "9.01"]),
    ],
)
def test_extract_items_from_fixtures(
    edgar_fixtures_dir: Path, fixture: str, expected_items: list[str]
) -> None:
    html = (edgar_fixtures_dir / fixture).read_text()
    text = html_to_text(html)
    assert extract_items(text) == expected_items
