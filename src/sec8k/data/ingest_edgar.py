"""End-to-end EDGAR 8-K ingester.

Streams accessions from the SEC daily index, fetches the primary document, parses
items + plain text, applies per-category quotas, and persists each filing to
``data/raw/edgar/<accession_nodash>.json`` with an append-only manifest at
``data/raw/edgar/manifest.jsonl``.

This module is the single public entry point for the data-ingest layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from sec8k.data.edgar import (
    EdgarClient,
    EdgarError,
    EdgarForbiddenError,
    EdgarNotFoundError,
    IndexRow,
    SubmissionsLookup,
    fetch_submissions,
    full_submission_url,
    iter_8k_accessions,
    normalize_accession,
)
from sec8k.data.parse import extract_items, html_to_text, strip_sgml_envelope

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

ITEM_TO_CATEGORY: dict[str, str] = {
    "2.01": "m_and_a",
    "1.01": "material_agreement",
    "1.02": "material_agreement",
    "2.03": "material_agreement",
    "2.04": "material_agreement",
    "2.02": "financial_results",
    "2.05": "financial_results",
    "2.06": "financial_results",
    "3.01": "regulatory",
    "3.02": "regulatory",
    "3.03": "regulatory",
    "4.01": "regulatory",
    "4.02": "regulatory",
    "5.01": "regulatory",
    "5.03": "regulatory",
    "5.02": "executive_change",
    "5.04": "other",
    "5.05": "other",
    "5.07": "other",
    "5.08": "other",
    "7.01": "other",
    "8.01": "other",
}

CATEGORY_PRIORITY: tuple[str, ...] = (
    "m_and_a",
    "regulatory",
    "financial_results",
    "executive_change",
    "material_agreement",
    "other",
)

DEFAULT_QUOTAS: dict[str, int] = {
    "m_and_a": 100,
    "executive_change": 100,
    "financial_results": 100,
    "material_agreement": 100,
    "regulatory": 100,
    "other": 100,
}


@dataclass
class IngestSummary:
    """Per-run accounting returned by :func:`ingest`."""

    written: dict[str, int]
    final_counts: dict[str, int]
    skipped_existing: int
    skipped_quota: int
    errors: list[tuple[str, str]]
    started_at: datetime
    finished_at: datetime


def assign_primary_category(items: list[str]) -> str:
    """Map a filing's item list to one of the six primary categories.

    9.01 is ignored. Priority hierarchy applied across categories present in items.
    """
    business = [i for i in items if i != "9.01"]
    if not business:
        raise ValueError("items must contain at least one non-9.01 code")
    cats = {ITEM_TO_CATEGORY.get(i, "other") for i in business}
    for cat in CATEGORY_PRIORITY:
        if cat in cats:
            return cat
    return "other"


def assign_primary_item(items: list[str], category: str) -> str:
    """Return the lowest item code in `items` that maps to `category`."""
    candidates = [
        i for i in items if i != "9.01" and ITEM_TO_CATEGORY.get(i, "other") == category
    ]
    if not candidates:
        raise ValueError(f"no items in list map to category {category!r}")
    return sorted(candidates)[0]


def _cik_int(cik_padded: str) -> str:
    """Strip leading zeros for use in EDGAR archive URL paths."""
    n = cik_padded.lstrip("0")
    return n if n else "0"


def _now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


_ACCEPTANCE_DT_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")


def _parse_acceptance_datetime(submission_text: str) -> str | None:
    """Pull <ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS out of the SGML header.

    SEC reports this in US Eastern; we surface a naive ISO 8601 string (consumers know).
    """
    m = _ACCEPTANCE_DT_RE.search(submission_text[:4096])
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt.isoformat()


def _build_filing(
    client: EdgarClient,
    row: IndexRow,
    submissions_cache: dict[str, SubmissionsLookup],
) -> dict[str, Any] | None:
    """Fetch and parse one filing. Return its JSON dict, or None when the filing is skipped."""
    with_dashes, no_dashes = normalize_accession(row.accession_with_dashes)
    cik_int = _cik_int(row.cik)
    submission_url = full_submission_url(cik_int, no_dashes, with_dashes)
    try:
        submission_text = client.get_text(submission_url)
    except EdgarNotFoundError:
        logger.warning("submission 404 for %s", with_dashes)
        return None

    acceptance_dt = _parse_acceptance_datetime(submission_text)
    body = strip_sgml_envelope(submission_text, row.form_type)
    parsed_text = html_to_text(body)

    items = extract_items(parsed_text)
    if not items:
        logger.warning("no item codes extracted from %s", with_dashes)
        return None
    business = [i for i in items if i != "9.01"]
    if not business:
        logger.warning("only 9.01 found in %s; skipping (likely parse miss)", with_dashes)
        return None
    category = assign_primary_category(items)
    primary_item = assign_primary_item(items, category)

    if row.cik not in submissions_cache:
        try:
            submissions_cache[row.cik] = fetch_submissions(client, row.cik)
        except EdgarNotFoundError:
            submissions_cache[row.cik] = SubmissionsLookup(name=row.company)
    sub = submissions_cache[row.cik]
    ticker = sub.tickers[0] if sub.tickers else None
    filer_name = sub.name or row.company

    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": _now_utc_iso(),
        "metadata": {
            "accession_number": with_dashes,
            "accession_number_nodash": no_dashes,
            "cik": row.cik,
            "filer_name": filer_name,
            "ticker": ticker,
            "tickers": sub.tickers,
            "form_type": row.form_type,
            "filing_date": row.filing_date.isoformat(),
            "acceptance_datetime": acceptance_dt,
            "items": items,
            "primary_item": primary_item,
            "primary_category": category,
            "submission_url": submission_url,
            "source": "edgar_daily_index",
        },
        "raw_html": body,
        "parsed_text": parsed_text,
    }


def _manifest_row(filing: dict[str, Any]) -> dict[str, Any]:
    md = filing["metadata"]
    return {
        "accession": md["accession_number"],
        "cik": md["cik"],
        "filer": md["filer_name"],
        "ticker": md.get("ticker"),
        "filing_date": md["filing_date"],
        "items": md["items"],
        "text_length": len(filing.get("parsed_text", "")),
        "primary_category": md["primary_category"],
        "schema_version": filing["schema_version"],
        "fetched_at": filing["fetched_at"],
    }


def _reconcile(out_dir: Path) -> tuple[dict[str, int], set[str]]:
    """Build (counters_by_category, existing_accessions) from manifest + on-disk files.

    Appends manifest rows for orphan JSON files. Drops corrupt manifest lines.
    """
    counters: dict[str, int] = {k: 0 for k in DEFAULT_QUOTAS}
    existing: set[str] = set()
    manifest_path = out_dir / "manifest.jsonl"
    in_manifest: dict[str, str] = {}

    if manifest_path.exists():
        good_lines: list[str] = []
        had_corruption = False
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                had_corruption = True
                continue
            good_lines.append(json.dumps(row, separators=(",", ":")))
            acc = row.get("accession", "")
            cat = row.get("primary_category", "")
            if acc and cat in counters:
                in_manifest[acc] = cat
        if had_corruption:
            content = "\n".join(good_lines) + ("\n" if good_lines else "")
            manifest_path.write_text(content, encoding="utf-8")

    orphans: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            with_dashes, _ = normalize_accession(path.stem)
        except ValueError:
            continue
        existing.add(with_dashes)
        if with_dashes in in_manifest:
            cat = in_manifest[with_dashes]
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metadata = data.get("metadata") or {}
            cat = metadata.get("primary_category", "")
            if cat not in counters:
                continue
            orphans.append(_manifest_row(data))
        counters[cat] += 1

    if orphans:
        with manifest_path.open("a", encoding="utf-8") as f:
            for row in orphans:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    return counters, existing


def ingest(
    start_date: date,
    end_date: date,
    out_dir: Path,
    client: EdgarClient,
    quotas: dict[str, int] | None = None,
    limit: int = 1000,
    dry_run: bool = False,
) -> IngestSummary:
    """Stream EDGAR 8-K filings into ``out_dir``, bounded by quotas and ``limit``.

    On disk: ``<out_dir>/<accession_nodash>.json`` per filing; ``<out_dir>/manifest.jsonl``
    appended once per filing. Resume by re-running; existing files are skipped.
    """
    actual_quotas = dict(DEFAULT_QUOTAS) if quotas is None else dict(quotas)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    counters, existing = _reconcile(out_dir)
    summary = IngestSummary(
        written={k: 0 for k in DEFAULT_QUOTAS},
        final_counts=dict(counters),
        skipped_existing=len(existing),
        skipped_quota=0,
        errors=[],
        started_at=started_at,
        finished_at=started_at,
    )

    submissions_cache: dict[str, SubmissionsLookup] = {}
    manifest_path = out_dir / "manifest.jsonl"
    written_total = 0
    printed = 0

    for row in iter_8k_accessions(client, start_date, end_date):
        if written_total >= limit:
            break
        if dry_run and printed >= 3:
            break
        if not dry_run and all(
            counters[c] >= actual_quotas.get(c, 0) for c in actual_quotas
        ):
            break

        with_dashes, no_dashes = normalize_accession(row.accession_with_dashes)
        if with_dashes in existing and not dry_run:
            continue

        try:
            filing = _build_filing(client, row, submissions_cache)
        except (EdgarNotFoundError, EdgarError, ValueError) as exc:
            summary.errors.append((with_dashes, repr(exc)))
            continue
        except Exception as exc:
            # Best-effort batch: log the unexpected error against the accession and continue.
            logger.exception("unexpected error processing %s", with_dashes)
            summary.errors.append((with_dashes, repr(exc)))
            continue
        if filing is None:
            continue

        category = filing["metadata"]["primary_category"]

        if dry_run:
            print(json.dumps(filing, indent=2))
            printed += 1
            continue

        if counters[category] >= actual_quotas.get(category, 0):
            summary.skipped_quota += 1
            continue

        out_file = out_dir / f"{no_dashes}.json"
        out_file.write_text(json.dumps(filing, indent=2), encoding="utf-8")
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_manifest_row(filing), separators=(",", ":")) + "\n")
        counters[category] += 1
        summary.written[category] += 1
        summary.final_counts[category] = counters[category]
        existing.add(with_dashes)
        written_total += 1

    summary.finished_at = datetime.now(UTC)
    return summary


def _parse_date_or(s: str | None, default: date) -> date:
    if s is None:
        return default
    return date.fromisoformat(s)


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def cli(
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="YYYY-MM-DD. Default: 90 days before --end-date."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="YYYY-MM-DD. Default: today."),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Hard cap on filings written.")
    ] = 1000,
    user_agent: Annotated[
        str | None,
        typer.Option(
            "--user-agent",
            envvar="EDGAR_USER_AGENT",
            help="SEC fair-use UA: 'Your Name your.email@example.com'.",
        ),
    ] = None,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Output directory.")
    ] = Path("data/raw/edgar"),
    rps: Annotated[float, typer.Option("--rps", help="Requests per second to SEC.")] = 8.0,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Fetch 3 filings, print to stdout, write nothing."
        ),
    ] = False,
    quota_m_and_a: Annotated[int, typer.Option("--quota-m-and-a")] = 100,
    quota_executive_change: Annotated[
        int, typer.Option("--quota-executive-change")
    ] = 100,
    quota_financial_results: Annotated[
        int, typer.Option("--quota-financial-results")
    ] = 100,
    quota_material_agreement: Annotated[
        int, typer.Option("--quota-material-agreement")
    ] = 100,
    quota_regulatory: Annotated[int, typer.Option("--quota-regulatory")] = 100,
    quota_other: Annotated[int, typer.Option("--quota-other")] = 100,
    strict_quota: Annotated[
        bool,
        typer.Option(
            "--strict-quota",
            help="Exit non-zero if any category is under quota at end.",
        ),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", envvar="SEC8K_LOG_LEVEL"),
    ] = None,
) -> None:
    """Download SEC 8-K filings from EDGAR into data/raw/edgar/."""
    logging.basicConfig(
        level=(log_level or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not user_agent or not user_agent.strip():
        typer.echo(
            "EDGAR_USER_AGENT is not set. Set it (e.g., in .env) to "
            "'Your Name your.email@example.com' per SEC fair-use policy.",
            err=True,
        )
        raise typer.Exit(code=2)

    today = date.today()
    try:
        end_d = _parse_date_or(end_date, today)
        start_d = _parse_date_or(start_date, end_d - timedelta(days=90))
    except ValueError as exc:
        typer.echo(f"invalid date: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if start_d > end_d:
        typer.echo(f"--start-date {start_d} is after --end-date {end_d}", err=True)
        raise typer.Exit(code=2)
    if (end_d - start_d).days > 365:
        typer.echo("Refusing to ingest more than 365 days in one run.", err=True)
        raise typer.Exit(code=2)

    quotas = {
        "m_and_a": quota_m_and_a,
        "executive_change": quota_executive_change,
        "financial_results": quota_financial_results,
        "material_agreement": quota_material_agreement,
        "regulatory": quota_regulatory,
        "other": quota_other,
    }

    try:
        timeout = float(os.environ.get("SEC8K_HTTP_TIMEOUT", "30.0"))
    except ValueError:
        timeout = 30.0

    try:
        with EdgarClient(user_agent=user_agent, rps=rps, timeout=timeout) as client:
            summary = ingest(
                start_date=start_d,
                end_date=end_d,
                out_dir=out_dir,
                client=client,
                quotas=quotas,
                limit=limit,
                dry_run=dry_run,
            )
    except EdgarForbiddenError as exc:
        typer.echo(f"EDGAR access denied: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Written by category: {summary.written}")
    typer.echo(f"Final counts on disk: {summary.final_counts}")
    typer.echo(f"Skipped (existing): {summary.skipped_existing}")
    typer.echo(f"Skipped (quota): {summary.skipped_quota}")
    typer.echo(f"Errors: {len(summary.errors)}")

    if strict_quota:
        shortfall = {
            c: q for c, q in quotas.items() if summary.final_counts[c] < q
        }
        if shortfall:
            typer.echo(f"Quota shortfall: {shortfall}", err=True)
            raise typer.Exit(code=3)


def main() -> None:
    """Script entry point: load .env and run the typer app."""
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
