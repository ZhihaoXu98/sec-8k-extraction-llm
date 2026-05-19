"""8-K HTML/SGML parser.

Detects Item sections, handles exhibits, normalises text (whitespace, smart quotes,
boilerplate stripping), and produces the structured records consumed downstream.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_CONTENT_TYPE_CHARSET_RE = re.compile(r"charset=([\w-]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset=["']?([\w-]+)""", re.IGNORECASE
)

_SGML_DOCUMENT_RE = re.compile(
    r"<DOCUMENT>(?:(?!</DOCUMENT>).)*?</DOCUMENT>", re.DOTALL | re.IGNORECASE
)
_SGML_TYPE_RE = re.compile(r"<TYPE>\s*([^\r\n<]+)", re.IGNORECASE)
_SGML_TEXT_RE = re.compile(
    r"<TEXT>\s*(.*?)\s*(?:</TEXT>|\Z)", re.DOTALL | re.IGNORECASE
)

_BLOCK_TAGS = (
    "p",
    "div",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
)
_DROP_TAGS = (
    "script",
    "style",
    "iframe",
    "svg",
    "picture",
    "img",
    "source",
    "meta",
    "link",
    "noscript",
)
_BOILERPLATE_CLASS_RE = re.compile(
    r"(nav|navbar|header|footer|sidebar|breadcrumb|pagination|disclaimer|cookie)",
    re.IGNORECASE,
)

_ITEM_RE = re.compile(
    r"item\s+(\d)\.(\d{2})(?:\([a-fA-F]\))?", re.IGNORECASE
)


def detect_encoding(headers: Mapping[str, str], body: bytes) -> str:
    """Determine encoding from HTTP Content-Type, then <meta charset>, then utf-8 fallback."""
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    m = _CONTENT_TYPE_CHARSET_RE.search(content_type)
    if m:
        return m.group(1).lower()
    head = body[:2048]
    m2 = _META_CHARSET_RE.search(head)
    if m2:
        return m2.group(1).decode("ascii", errors="ignore").lower()
    return "utf-8"


def strip_sgml_envelope(text: str, form_type: str) -> str:
    """If a `<SEC-DOCUMENT>` envelope is present, return the `<TEXT>` body for the
    matching `<DOCUMENT><TYPE>{form_type}`. Otherwise return `text` unchanged.
    """
    head = text[:200].upper()
    if "<SEC-DOCUMENT>" not in head and "<SUBMISSION>" not in head:
        return text
    wanted = form_type.upper()
    fallback_text: str | None = None
    for doc_match in _SGML_DOCUMENT_RE.finditer(text):
        block = doc_match.group(0)
        type_match = _SGML_TYPE_RE.search(block)
        if not type_match:
            continue
        doc_type = type_match.group(1).strip().upper()
        text_match = _SGML_TEXT_RE.search(block)
        if not text_match:
            continue
        body = text_match.group(1)
        if doc_type == wanted:
            return body
        if fallback_text is None:
            fallback_text = body
    if fallback_text is not None:
        return fallback_text
    return text


def html_to_text(html: str) -> str:
    """Strip boilerplate and flatten to plain text with paragraph structure preserved."""
    from bs4 import BeautifulSoup
    from bs4.element import NavigableString

    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Whole-document drops (incl. content; never want their text).
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    # XBRL inline tags carry real text wrapped in tag chrome. Keep text, drop tag.
    for tag in list(soup.find_all(True)):
        name = tag.name or ""
        if ":" in name and name.split(":", 1)[0].lower() in ("ix", "xbrli", "xbrldi"):
            tag.unwrap()

    # Boilerplate by class or id.
    for tag in list(soup.find_all(True)):
        tag_attrs = getattr(tag, "attrs", None)
        if not isinstance(tag_attrs, dict):
            continue
        klass = tag_attrs.get("class")
        ident = tag_attrs.get("id")
        attrs: list[str] = []
        if isinstance(klass, list):
            attrs.extend(str(c) for c in klass)
        elif isinstance(klass, str):
            attrs.append(klass)
        if isinstance(ident, str):
            attrs.append(ident)
        if any(_BOILERPLATE_CLASS_RE.search(a) for a in attrs):
            tag.decompose()

    # Render tables to pipe-delimited rows in place.
    for table in list(soup.find_all("table")):
        rendered = _render_table(table)
        table.replace_with(NavigableString(rendered))

    # <br> -> newline.
    for br in list(soup.find_all("br")):
        br.replace_with(NavigableString("\n"))

    # Append a paragraph break after each block tag.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append(NavigableString("\n\n"))

    raw = soup.get_text(separator=" ")

    # Normalize unicode whitespace and collapse runs.
    raw = raw.replace("\xa0", " ").replace("​", "")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _render_table(table: object) -> str:
    """Render a <table> as pipe-delimited rows. `table` is a BeautifulSoup Tag."""
    lines: list[str] = []
    rows = table.find_all("tr")  # type: ignore[attr-defined]
    for row in rows:
        cells_raw = row.find_all(["td", "th"])
        cells = [c.get_text(separator=" ", strip=True) for c in cells_raw]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def extract_items(text: str) -> list[str]:
    """Pull 8-K item codes from filing text, normalized to 'X.YY', deduped, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _ITEM_RE.finditer(text):
        code = f"{match.group(1)}.{match.group(2)}"
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out
