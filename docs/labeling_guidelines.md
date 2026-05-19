# Labeling guidelines for SEC 8-K extraction

## Purpose

This is the spec for what "correct" means when labeling 8-K filings against `Filing8K` in `src/sec8k/schema.py`. Every entry in the gold set (`data/gold/v1.jsonl`) must satisfy these rules. The spec is the source of truth — when a label disagrees with the spec, the spec wins. When the spec doesn't tell you what to do, the spec is what gets updated, then the label is re-done.

**This doc has a dual role.** It is (a) the **system prompt** for the Claude Sonnet 4.6 labeler (loaded into every API call with `cache_control: ephemeral`), and (b) the **verification spec** for the Claude Opus 4.7 critic that does the dual-pass review. The two roles share the same text by design — anything the critic uses to flag a Pass-1 label as wrong should be a rule the labeler also saw.

Drift across the gold-set examples destroys intra-Claude consistency (and would destroy inter-annotator agreement if the project ever brought in human labelers). Consistency is what makes the gold set credible as a target distribution for Qwen to learn. Drift is the worst failure mode — worse than being wrong about a specific example.

## How to read this doc

Each of the 14 fields has four parts:

- **Rule** — what the correct value is in the typical case.
- **Boundary** — where the rule stops applying (typically the null/empty conditions).
- **Tiebreaker** — what to do when the rule is genuinely ambiguous. Tiebreakers must be **deterministic** — two passes reading the same filing must arrive at the same answer.
- **Example** — concrete. Real-event references (Microsoft–Activision, Apple's 2024 buyback) are anchors you can verify on EDGAR; specific accession numbers are `[fill in once labeled]` placeholders, to be replaced when a real filing is labeled under this rule.

Throughout this doc, **"labeler"** refers to the Pass-1 Claude Sonnet 4.6 labeler; **"reviewer"** or **"critic"** refers to the Pass-2 Claude Opus 4.7 dual-pass critic; **"annotator"** or **"labelers"** (plural) refers generically to either pass. Where the doc historically said "labeler" or "human", read it as instructions to whichever Claude pass is currently reading. The project owner (a single human) is not in the per-row labeling loop; their role is doc maintenance, design choices, and a 5-10-row sanity spot-check on the final gold set.

**Before reading the field rules**, read the **Concepts cheat-sheet** at the end of this doc if any of the following terms is unfamiliar: 8-K item codes (1.01, 2.01, 5.02, 7.01, 9.01), CIK, Reg FD, impairment, purchase consideration, material agreement. The field rules assume working knowledge of those.

## Verification protocol (LLM dual-pass, not human dual-label)

**Design choice (2026-05-19, week 1).** This is a solo project producing a distillation gold set — Claude Sonnet 4.6 generates labels; Qwen 2.5 7B learns to mimic them. There is **no human dual-labeling step**. The project owner lacks the SEC/finance domain depth that would make a human second pass higher-signal than the LLM first pass; a noisy human-by-the-owner pass would add measurement error rather than ground truth. Instead, the calibration subset is reviewed by a **second, different Claude model** as a critic:

- **Pass 1 (labeler)**: Claude Sonnet 4.6 with this doc as system prompt + the filing text as user prompt + tool-use schema enforcement. Recorded with `provenance.verified_by = "<unset>"`, `provenance.model = "claude-sonnet-4-6"`.
- **Pass 2 (critic)**: Claude Opus 4.7 reading the same filing, this doc as the spec, and Pass-1's label; produces keep/edit/ambig verdicts per field. Recorded with `provenance.verified_by = "claude-opus-4-7"`, `provenance.verification_type = "llm_critical_review"`.
- **Optional Pass 3 (human spot-check)**: project owner spot-checks 5-10 of the 300 for obvious sanity (filer name matches cover page, items list isn't empty, summary is grammatical English, monetary_amount is a number). Marked with `provenance.human_spot_checked = true`. This is a sanity floor, not full calibration.

**What this gets us, and what it doesn't.** The Pass-1↔Pass-2 agreement metric is **intra-Claude consistency**, not human-vs-LLM agreement. That's a weaker signal than what classical inter-annotator-agreement methodology aims for — two different Claudes can share blind spots. We accept this because (a) the gold set's job is to teach Qwen to mimic Claude, so Claude-defined correctness is the right target, and (b) the eval harness downstream provides an independent check (does the fine-tuned Qwen actually extract well on held-out filings?). The kappa-style metrics below remain useful as **internal consistency** signal but should not be reported as inter-annotator-agreement in any external context.

**Targets — intra-Claude consistency on the 50-example calibration subset.** These are the thresholds at which the doc + labeler are considered stable enough to scale to the full 300:

**Fields determined by filing metadata or fixed-format text — target exact agreement = 1.00.** Pass-2 should never disagree here; if it does, either Pass 1 has a bug (e.g., the `filing_date`-vs-`event_date` confusion fixed in `PROMPT_VERSION="v2"`) or this doc is under-specified.
- `form_type` (from SGML `<TYPE>`)
- `filer_cik` (from SGML `<CIK>`)
- `filing_date` (from SGML `<ACCEPTANCE-DATETIME>` — exposed to the labeler via the user prompt as of v2)
- `filer_company` (from cover-page registrant line)
- `filer_ticker` (from cover-page Trading Symbol field, when present)
- `currency` (from filing text — the currency symbol/code is explicit)

**Fields requiring filing-text reading but with a clear extraction target — target exact agreement ≥ 0.95.**
- `items` (set equality; the cover-page list is canonical)
- `event_date` (the body states the date explicitly)
- `monetary_amount` (exact numeric match)
- `counterparties` (set equality with insertion order)

**Subjective categorical fields — target Cohen's kappa ≥ 0.70** ("substantial agreement" per Landis & Koch, 1977), *re-interpreted as intra-Claude consistency*. These fields genuinely require interpretation:
- `primary_category` (priority hierarchy is binding; only Item 1.01 m_and_a/material_agreement and the explicit 8.01-M&A carve-out are substance reads)
- `amount_type` (event → label mapping; borderline cases possible)
- `expected_impact_period` (interpreting timing language)

**Free-text field — qualitative review only.**
- `summary` (no exact-match expectation; review for factual accuracy and adherence to the 1–3 sentence / 500-char constraints)

If the calibration round misses these targets, **stop labeling and revise this doc** — the rule causing the disagreement is the rule that needs sharpening. Do not start the main labeling round until the calibration subset clears the targets.

## Labeling workflow — order of operations

For each filing, work the fields in this order. Earlier fields constrain later ones; doing them out of order causes rework.

1. **Read the filing header** → `form_type`, `filer_cik`, `filing_date`.
2. **Read the cover-page registrant block** → `filer_company`, `filer_ticker`.
3. **Read the cover-page item list** → `items` (every code, in order).
4. **Apply the category-priority table** → `primary_category` (single label).
5. **Read the body section of the primary item** → `event_date`, `counterparties` for the primary event.
6. **Scan the remaining item sections** → add any additional `counterparties`, in body order.
7. **Identify the primary event's headline financial figure** → `monetary_amount`, `currency`, `amount_type` (all three or none).
8. **Determine the impact timing per the filing's own language** → `expected_impact_period`.
9. **Write the summary** → `summary` (1–3 sentences, ≤500 chars, factual paraphrase).
10. **Self-check cross-field consistency** (see "Cross-field consistency checks" below).

If at any step you cannot fill the field without inventing information, leave it null (for nullable fields) or stop and log an edge case (for required fields).

## Cross-field consistency checks

Before declaring a label final, verify:

- `event_date <= filing_date` (schema enforces; this catches date-swap and year-hallucination errors).
- `monetary_amount` set ⇔ `currency` and `amount_type` both set (schema enforces).
- `monetary_amount` is null ⇒ `currency` and `amount_type` are both null.
- `filer_cik` is exactly 10 characters, all digits (schema enforces).
- Every code in `items` matches `^\d\.\d\d$` (schema enforces).
- `primary_category` should be derivable from `items` via the category-priority table, with one read-dependent caveat: **Item 1.01 maps to `m_and_a` OR `material_agreement`** depending on whether the agreement is an M&A / divestiture agreement or a non-M&A agreement, which requires reading the body. So a labeler whose `primary_category` disagrees with the items-table derivation has either (a) chosen the wrong category, (b) omitted the relevant item from `items`, or (c) read 1.01's substance differently than another labeler — option (c) is the legitimate source of disagreement and should surface in calibration kappa.
- `summary` mentions the primary event (the one that determined `primary_category`); secondary events optional if within the 500-char budget.

---

## form_type

**Rule.** Use `"8-K"` for original filings, `"8-K/A"` for amendments. The form code is the `<TYPE>` field in the SGML header and appears on the cover page.

**Boundary.** The SEC defines the form code — the filer doesn't choose it, and we don't infer it. If the header says `"8-K/A"`, it's an amendment regardless of how the body reads; if the header says `"8-K"`, use that even if the body mentions amending an earlier disclosure (the amendment relationship is captured in 8-K/A's own header).

**Tiebreaker.** None needed; the form code is unambiguous from filing metadata.

**Example.** A typical merger announcement → `"8-K"`. A correction filed two weeks later that attaches audited target financials and revises the earlier disclosure → `"8-K/A"`. [Accession: TBD]

---

## filer_company

**Rule.** The filer's legal entity name, exactly as printed on the cover page on the line following "Exact name of registrant as specified in its charter." Preserve the full legal suffix (`Inc.`, `Corp.`, `LLC`, `Holdings`, `plc`) and trailing punctuation.

**Boundary.** Do not substitute the common or brand name. The doing-business-as name belongs in the body text, not in this field. Preserve commas as printed — `"Tesla, Inc."` keeps the comma; `"Apple Inc."` does not have one.

**Tiebreaker.** When an entity has recently changed its legal name (Facebook → Meta Platforms, 2021–2022; Square → Block, 2021), use the name printed on **this specific filing's cover page** — neither the historical name nor today's name. The 8-K announcing a name change is often filed under the old name (the new name's legal effective date typically post-dates the announcement); subsequent filings use the new name. Read the cover; don't assume.

**Example.** Apple's 2024 buyback 8-K (announced May 2, 2024) → `"Apple Inc."` [Accession: TBD]. A filing made by Tesla in 2023 → `"Tesla, Inc."` (with the comma) [Accession: TBD].

---

## filer_ticker

**Rule.** The ticker symbol from the filing's cover-page **`Trading Symbol(s)`** field. Modern 8-K cover pages (post-2019 amendment to Form 8-K) explicitly list trading symbols below the registrant block. That field is the authoritative source.

**Boundary.** Null when:

- The cover page has no `Trading Symbol(s)` field populated (most pre-2019 filings, or current filings from issuers with no registered equity).
- The filer is private (some private REITs and debt-only issuers file 8-Ks under Section 13/15(d)).
- The filer has no equity listing (delisted shell, OTC-only filer).
- The filing entity is a wholly-owned subsidiary with its own CIK but no separate stock listing (only the parent is listed; even if the parent has a ticker, this filer doesn't).

For pre-2019 filings without the explicit field, fall back to SEC's `company_tickers.json` for the filer's CIK. Do not invent a ticker from external sources beyond that.

**Tiebreaker.** When the cover page lists multiple trading symbols (dual-class issuers list both, e.g., Alphabet lists `GOOGL` and `GOOG`; Berkshire Hathaway lists `BRK.A` and `BRK.B`):

- If the filing's body discusses a specific class (e.g., a `Class A` share repurchase), use that class's ticker.
- If the filing is class-neutral, use the ticker that appears **first** on the cover page's `Trading Symbol(s)` line. Filers list them in a consistent order on every filing for the same registrant, so this is reproducible per-issuer (and deterministic across labelers reading the same filing).

**Example.** Apple → `"AAPL"` (single trading symbol on the cover). Alphabet's class-neutral filing → first symbol listed on the cover (typically `"GOOGL"`, but verify per filing). Berkshire Hathaway → first symbol listed on the cover (typically `"BRK.A"`, but verify per filing). A private real-estate fund filing an 8-K under a public-debt indenture → null.

---

## filer_cik

**Rule.** The 10-digit zero-padded Central Index Key from the filing's header (`<CIK>` in the SGML metadata, or the URL path on EDGAR: `/cgi-bin/browse-edgar?CIK=...`). Always pad with leading zeros to exactly 10 characters.

**Boundary.** The CIK is the SEC's permanent unique identifier for a legal entity. One entity → one CIK forever; the ticker can change, the legal name can change, the CIK does not. Subsidiaries have their own CIKs distinct from their parent's. Co-registrant filings (one filing under multiple CIKs) are rare; in those cases, use the primary CIK in the header.

**Tiebreaker.** When the cover-page entity differs from the `<CIK>` header (unusual; usually a co-registrant filing), trust the `<CIK>` header — that's the canonical identifier the SEC indexes against.

**Example.** Apple Inc. → `"0000320193"`. Microsoft Corporation → `"0000789019"`. A 6-digit CIK like `"320193"` must be left-padded to `"0000320193"`; the schema regex `^\d{10}$` rejects unpadded values.

---

## filing_date

**Rule.** The SEC's official acceptance date for the filing, in U.S. Eastern Time. This appears in the `<ACCEPTANCE-DATETIME>` header field and is the timestamp EDGAR shows on the filing index page as "Filed."

**Boundary.** Not the document body's date, not the press release's "for immediate release" date, not the event date. The `filing_date` is when the SEC acknowledged receipt — that's the date that anchors a filing in the public record.

**Tiebreaker.** Filings submitted to EDGAR after 5:30 PM ET on a business day are deemed filed the next business day (per the SEC's EDGAR Filer Manual). Use the acceptance stamp the SEC shows, not the filer's local timestamp or the press release date.

**Example.** An 8-K with `<ACCEPTANCE-DATETIME>20240501145300` → `filing_date = 2024-05-01`. The same filing submitted at `20240501180000` (6:00 PM ET) might be acceptance-stamped May 2 — use the stamp.

---

## event_date

**Rule.** The date the underlying business event actually occurred. Most 8-K bodies state this explicitly: "On [Date], the Company [did X]…"

**Boundary.** Null when:

- The filing has no discrete event date — typical of Item 7.01 (Reg FD) and Item 8.01 (Other Events) where the *filing itself* is the disclosure event.
- The body says "we recently became aware…" without a specific date.
- The content is purely forward-looking ("we expect to announce next quarter…") with no triggering event yet.

The schema validator enforces `event_date <= filing_date`. If your candidate event_date is later than filing_date, you have the wrong date — re-read the body.

**Tiebreaker.** When the body lists multiple dates (e.g., agreement signed May 1, regulatory clearance May 10, closing May 15), use the date of the **triggering event** for the primary item:

| Primary item | Triggering date |
|---|---|
| 1.01 (entry into agreement) | Signing date |
| 1.02 (termination of agreement) | Termination effective date |
| 2.01 (completion of acquisition/disposition) | Closing date |
| 2.02 (results of operations) | Date of the public announcement / press release (not the period being reported) |
| 2.05 (exit/disposal costs) | Date of board commitment to the plan |
| 2.06 (impairment) | Date the impairment was determined / recognized |
| 4.01 (auditor change) | Effective date of dismissal/engagement |
| 4.02 (non-reliance) | Date of audit committee / board determination |
| 5.02 (executive change) | Effective date of departure or appointment |
| 5.03 (articles/bylaws amendment) | Date of board approval or effectiveness |

**Example.** An 8-K filed May 3, 2024, stating "On May 1, 2024, the Company entered into a definitive merger agreement…" → `event_date = 2024-05-01`. A pure Reg FD 8-K announcing a CEO will speak at next week's conference → null (no event has occurred yet; the filing is the event).

---

## items

**Rule.** Every SEC item code the filing explicitly invokes — on the cover-page item list **and** as section headers in the body. Format is exactly `X.XX` (one digit, period, two digits). Duplicates collapse via the schema validator; insertion order is preserved.

**Boundary.** Include items even when secondary — a merger 8-K legitimately covers 1.01, 2.01, 7.01, and 9.01 together. Do **not** include items that appear only as cross-references ("see also Item 2.01 below") without substantive disclosure under that item's own header. Item 9.01 (Financial Statements and Exhibits) is included only when the filing actually attaches exhibits — common for filings with a press release or executed agreement, but absent for some bare disclosures (e.g., a one-paragraph Item 5.02 director resignation with no press release).

**Tiebreaker.** When the cover-page list and body section headers disagree:

- Cover lists items the body does not substantively cover → trust the cover (the filer's formal declaration of scope).
- Body has a substantively-disclosed section the cover omits → log under "Edge cases" and update this rule. (This is rare and usually a filer error; resolve case-by-case.)

**Example.** Merger announcement with cover items 1.01, 2.01, 7.01, 9.01 → `["1.01", "2.01", "7.01", "9.01"]`. A Reg FD 8-K wrapping a press release → `["7.01", "9.01"]`. A bare CEO resignation announcement with no exhibit → `["5.02"]`.

---

## primary_category

**Rule.** A single label characterizing the dominant business event in the filing, drawn from one of six categories. The category each Item code maps to:

| Category | Mapped item codes |
|---|---|
| **m_and_a** | 2.01 (completion of acquisition/disposition); 1.01 *only when the agreement IS an M&A / divestiture agreement* |
| **executive_change** | 5.02 (departure/election/appointment/compensation of directors and officers) |
| **financial_results** | 2.02 (results of operations); 2.05 (exit/disposal costs); 2.06 (material impairment) |
| **material_agreement** | 1.01 (non-M&A definitive agreements: supply, IP licenses, JV agreements, credit facilities); 2.03 (creation of a financial obligation); 2.04 (triggering events accelerating an obligation); 1.02 (termination of a material agreement) |
| **regulatory** | 3.01 (delisting); 3.02 (unregistered equity sales); 3.03 (modification to security holder rights); 4.01 (change of certifying accountant); 4.02 (non-reliance on prior financials); 5.01 (changes in control); 5.03 (articles/bylaws amendments) |
| **other** | 7.01 (Reg FD); 8.01 (Other Events); anything not mapped above |

When a filing's `items` list maps to multiple categories, apply the **category priority** (highest to lowest):

```
m_and_a  >  regulatory  >  financial_results  >  executive_change  >  material_agreement  >  other
```

**Why this ordering?** The hierarchy reflects two considerations:

1. **Structured-extraction signal density.** When `primary_category = m_and_a`, the structured fields (`counterparties`, `monetary_amount`, `amount_type = "purchase_price"`) all carry rich information — losing the M&A label would hide the most data-dense extraction pattern in the gold set. Regulatory events (especially Item 4.02 restatements and Item 3.01 delistings) also have rich structured fields and severe market impact when present, so they come second. Financial results, executive change, and material agreement decrease in average per-filing extraction richness.
2. **Market-impact magnitude.** In event-study literature, M&A target-side abnormal returns and material-restatement abnormal returns are the largest on average; earnings surprises are next; executive turnover is smaller; routine agreement disclosures are smallest. The ordering above is consistent with this empirical pattern, though the priority is primarily about extraction-pattern clarity, not market impact.

The ordering is a **design choice, not a derivation** — an alternative ordering (e.g., putting `regulatory` first because filer-side restatement returns can exceed M&A acquirer returns) is defensible. What matters is that *one* ordering is fixed in this doc and applied consistently across all gold-set examples. If labeling reveals that the chosen ordering causes systematic mis-categorization on real filings, the doc gets updated and the affected gold-set entries get re-labeled.

**Why these six categories and not others?** The six partition the SEC's 8-K item taxonomy into groups that (a) correspond to distinguishable downstream extraction patterns (M&A needs counterparty and purchase price; exec change needs the individual's name; financial results needs the dollar magnitude), and (b) have enough volume each in the EDGAR universe to be learnable by a 7B model. A finer taxonomy (e.g., splitting `material_agreement` into "supply contract" / "credit facility" / "license") would have more learning targets but per-class data scarcity; a coarser taxonomy would lose downstream signal.

**Boundary.** Exactly one label per filing. Categorize by which event drives the primary signal, not by which item section is longest. A long Item 7.01 press release attached to a brief Item 5.02 CEO departure → `executive_change`, not `other` — the substance is governance change; the 7.01 press release is the disclosure mechanism.

**Apply priority strictly when items map to multiple categories.** Substance-based "this filing is really about X" reasoning does NOT override the category-priority hierarchy when the items list contains categories higher up the priority order. The priority order WAS the design choice that resolves substance — applying substance again on top of priority would re-introduce the inconsistency the priority order was created to eliminate. Two exceptions, both explicit and bounded, follow.

**Substance read for Item 1.01 only.** As noted in the items mapping table, Item 1.01 maps to `m_and_a` when the agreement IS an M&A / divestiture agreement, and to `material_agreement` otherwise (supply, IP licenses, JV agreements, credit facilities, SPAC underwriting agreements). This read is required to disambiguate the 1.01 category before priority is applied; once the per-item categories are known, priority governs.

**Substance override for M&A disclosed solely under Item 8.01.** Items 7.01 and 8.01 are mechanisms — they can carry substantive disclosures of any type. When the cover-page items list contains *only* items that map to `other` (e.g., `["8.01"]`, `["7.01", "8.01"]`, `["8.01", "9.01"]`) AND the body substantively describes an **M&A** event (acquisition or disposition completion, business combination, scheme of arrangement, tender offer outcome, intermediate M&A milestone such as shareholder vote results), categorize as `m_and_a`, not `other`. The `items` list stays as the cover lists; only `primary_category` is overridden. **This override applies only to m_and_a** — do not extrapolate to other categories (a 8.01 disclosing operations data is `other`, not `financial_results`). Rationale: losing the M&A label would hide the most data-dense extraction pattern in the gold set, per the priority-ordering rationale above. Filers sometimes use 8.01 for M&A because the foreign-law mechanism (e.g., UK Companies Act scheme of arrangement) or intermediate-milestone nature doesn't fit cleanly under Item 2.01.

**Tiebreaker.** When the priority hierarchy itself ties (e.g., two `regulatory`-mapped items in the same filing), use the **lowest item number** as a deterministic numeric tiebreaker. This is reproducible across labelers without judgment.

**Example.**
- Filing with cover items 1.01 (merger agreement) + 5.02 (new CFO joining at close) + 9.01 → m_and_a beats executive_change in the priority → `m_and_a`.
- Filing with 4.02 (non-reliance) + 2.02 (revised earnings) → regulatory beats financial_results → `regulatory`.
- Filing with 5.02 (CEO departure) + 1.01 (a separately-signed supply agreement same day) → executive_change beats material_agreement → `executive_change`.
- Filing with 7.01 (press release of upcoming conference attendance) + 9.01 → no higher-priority category present → `other`.
- Filing with 1.01 (SPAC IPO underwriting agreement, NOT M&A) + 3.02 (unregistered warrant sale) + 5.02 (initial director appointments) + 5.03 (amended articles) + 8.01 + 9.01 → 1.01 is material_agreement (non-M&A); 3.02 + 5.03 are regulatory; 5.02 is executive_change; 8.01 is other. Priority: regulatory wins → `regulatory`. Do NOT use `material_agreement` based on "the IPO is the substantive event"; the priority rule is binding when multiple categories are present.
- Filing with `["3.02", "8.01", "9.01"]` (private placement warrant issuance bundled with a debt-tranche disclosure) → 3.02 is regulatory, 8.01 is other; priority gives `regulatory`.
- Filing with `["8.01", "9.01"]` substantively describing a UK Companies Act 2006 scheme of arrangement acquisition by a U.S. registrant → m_and_a substance override applies → `m_and_a`.
- Filing with `["8.01"]` disclosing only the shareholder-vote result of a pending acquisition → m_and_a substance override applies → `m_and_a`.

---

## counterparties

**Rule.** Other named parties to the disclosed event:

- M&A → the target (or acquirer, if the filer is being acquired)
- Executive changes → the departing or arriving individual by name
- Material agreements → the contractual counterparty (vendor, lender, JV partner, licensee, customer)
- Settlements → the opposing party / plaintiff / regulator
- Director appointments → the new director by name

**Boundary.** Only parties **named** in the filing — never inferred. No "and other parties" or "an undisclosed buyer." For entities, use the legal name as stated in the body (preserving suffix and punctuation per `filer_company` conventions). For individuals, use the name **as printed in the filing**, including middle initials if printed, but drop honorifics and titles (`Mr.`, `Mrs.`, `Dr.`, `Ph.D.`, `CEO`, `CFO`).

Empty list when no counterparty is named — typical for regular dividend declarations, internal restructurings, bylaw amendments, and qualitative Reg FD content with no other named party.

**Tiebreaker.** When multiple counterparties exist, list them in the order they appear in the **body** of the filing (not the cover-page item list). Do not collapse subsidiaries into their parents — if the filing names "Microsoft Mobile Oy" and "Microsoft Corporation" separately under the same item, both go in. If the body introduces a defined term like `"Buyer"` after a single full-name use, the full legal name is what goes in the list, not the defined term.

**Example.**
- Activision filing the closing merger 8-K with Microsoft as acquirer → `["Microsoft Corporation"]`.
- Executive change filing naming "Timothy D. Cook" as outgoing CEO and "Jane A. Smith" as interim CEO → `["Timothy D. Cook", "Jane A. Smith"]` (middle initials preserved; titles dropped).
- A regular quarterly dividend declaration → `[]`.

---

## monetary_amount

**Rule.** The single most prominent dollar (or stated-currency) figure tied to the disclosed event, as a **non-negative** number in whole units of the currency (not cents). By event type:

- **M&A** → total purchase consideration as headlined (cash + stock at fair value + assumed debt, if the filing gives an aggregate; otherwise cash component)
- **Executive change** → severance package, signing bonus, or new compensation arrangement *if explicitly disclosed*; null otherwise (most appointments don't disclose comp in the 8-K)
- **Settlement** → settlement amount as stated
- **Dividend** → see the dividend-specific convention below
- **Impairment / write-down** → magnitude of the charge (positive; the sign is carried by `amount_type="loss"`)

**Dividend convention.** Dividend disclosures need a specific rule because most dividend 8-Ks state only per-share, not aggregate. Apply this priority order:

1. **If the filing states both per-share AND aggregate together** (e.g., "a special dividend of $0.50 per share, totaling approximately $1.2 billion"), use the aggregate. This is the cleanest case.
2. **If the filing states only per-share** (the typical case for regular quarterly dividends), use the per-share value (e.g., `monetary_amount = 0.25` for a $0.25/share dividend) and **the word `"per share"` must appear in `summary`** so downstream consumers don't naively aggregate.
3. **Never compute aggregate from shares-outstanding looked up elsewhere.** Inferring from external data injects unverifiable information into the label.

This is a deliberate departure from the "whole-units-of-currency-total-impact" framing in the rule above, necessary because dividend disclosures are inherently per-share in 8-K prose.

**All-stock M&A consideration.** When the filing provides a headline dollar figure for an all-stock deal (e.g., "the transaction values Target at approximately $7 billion based on Acquirer's recent stock price"), use that figure. When the filing provides only a share count and conversion ratio without a dollar headline, use null — do not compute from external stock-price data, as that would inject information the filing doesn't disclose.

**Boundary.** Null when no dollar amount is disclosed. Common: most executive appointments without disclosed compensation, governance changes, qualitative Reg FD content, code-of-ethics amendments, auditor changes, routine bylaw modifications.

**Magnitude only** — the sign is carried by `amount_type`. A $500M loss is `500_000_000`, not `-500_000_000`. The schema enforces `ge=0`; negatives are rejected.

**Cross-field constraint.** `monetary_amount` non-null requires both `currency` and `amount_type` non-null (schema enforces). Conversely, if `monetary_amount` is null, both `currency` and `amount_type` must also be null. There's no valid state where one of the three is set and the others aren't.

**Tiebreaker.** When multiple amounts appear, pick the one most directly tied to the *primary* event:

- For acquisitions: total consideration (cash + stock + assumed debt) when given as a single headline figure; otherwise the cash component only; if both an aggregate and a per-share offer price are given, use aggregate.
- For credit facilities (Item 2.03): the facility size, not the initially-drawn amount.
- For multi-component deals ($1B cash + $500M earn-out): the headline total if the filing provides one; otherwise the cash component, with an "earn-out" note in `summary`.
- For settlements with multiple components (cash payment + ongoing obligations): the cash payment if stated as a headline; otherwise null and explain in `summary`.
- For dividends: per the dividend convention above (aggregate if both stated; per-share otherwise).

**Example.**
- Microsoft–Activision merger ($68.7B all-cash deal announced January 2022; closed October 13, 2023) → for whichever 8-K (signing or closing) restates the headline figure, `monetary_amount = 68_700_000_000`, `currency = "USD"`, `amount_type = "purchase_price"`. **Verify against the specific filing's body before citing**; do not assume both the signing and closing 8-Ks repeat the figure identically. [Accession: TBD]
- A $25M severance package for a departing CFO under Item 5.02 → `25_000_000`, `"USD"`, `"severance"`. [Accession: TBD]
- A $200M goodwill impairment under Item 2.06 → `200_000_000`, `"USD"`, `"loss"`. [Accession: TBD]
- A $0.25/share quarterly dividend → `monetary_amount = 0.25`, `currency = "USD"`, `amount_type = "dividend"`, and `summary` includes "per share."
- A new $5B revolving credit facility (Item 2.03) → `5_000_000_000`, `"USD"`, `"other"`.
- An all-stock acquisition where the filing only states "in exchange for X million Acquirer shares" without a dollar headline → `monetary_amount = null` (and `currency`, `amount_type` also null).

---

## currency

**Rule.** The ISO three-letter currency code denominating `monetary_amount`. Use the currency the filing explicitly names: `$` / `USD`, `€` / `EUR`, `£` / `GBP`, `C$` / `CAD`, `¥` / `JPY`.

**Boundary.** Restricted to `{USD, EUR, GBP, CAD, JPY}` per the schema. Null when `monetary_amount` is null. Do **not** default to `USD` for U.S. filers — a U.S. parent's foreign subsidiary may sign agreements denominated in EUR or JPY, and the 8-K may report that currency. Currencies outside this set (CHF, AUD, KRW, etc.) require updating the schema first; do not silently map to USD.

**Tiebreaker.** When the filing discloses a dual-currency amount ("approximately $50 million / €45 million"), use the currency the filing's headline / lead sentence presents **first**. If body and exhibit disagree, the body wins — the 8-K is the disclosure document; exhibits are reference material.

**Example.**
- "We agreed to pay $500 million in cash" → `"USD"`.
- "The Japanese subsidiary entered into a ¥10 billion credit facility" → `"JPY"`.
- A UK-listed registrant declaring a £0.50 per-share dividend → `"GBP"`.
- A Swiss-franc-denominated agreement from a U.S. parent → outside enum; log as edge case and stop.

---

## amount_type

**Rule.** The semantic role of `monetary_amount`:

| Label | Use for |
|---|---|
| **purchase_price** | M&A consideration; asset-purchase consideration |
| **severance** | Termination / separation pay for a departing executive |
| **dividend** | Declared dividend (regular quarterly, special, or one-time) |
| **settlement** | Payment in or out of a litigation, arbitration, or regulatory settlement |
| **loss** | One-time charge, write-down, impairment (magnitude-positive; the label carries the sign) |
| **gain** | One-time positive recognition (gain on sale, contingent gain) |
| **other** | Anything not fitting the above (credit facility size, signing bonus, JV capital commitment, restructuring cost estimate) |

**Boundary.** Null when `monetary_amount` is null (schema enforces the bidirectional constraint). One label per filing. The label refers to the role of *this filing's* primary monetary disclosure, not historical context.

**Tiebreaker.** When an amount could fit two categories, choose the **event-level** label, not the accounting treatment.

- A $500M legal settlement is `settlement`, even though the company will book it as a loss on the income statement — the *event* is the settlement.
- A $200M write-down of a previously-acquired asset is `loss`, not `purchase_price` — the purchase happened in an earlier filing; this filing's event is the impairment.
- A CEO signing bonus paid in cash is `other` (no dedicated label fits perfectly; `severance` is for departure, not arrival).
- A contingent earn-out reaching its trigger and being paid is `purchase_price` (it's deferred M&A consideration).

**Example.**
- Microsoft–Activision $68.7B → `purchase_price`.
- A $25M departing-CFO severance → `severance`.
- A $200M goodwill impairment (Item 2.06) → `loss`.
- A $5B revolving credit facility (Item 2.03) → `other`.
- A $0.25/share quarterly dividend → `dividend`.
- A multi-state regulatory settlement with named opposing parties → `settlement`.

---

## summary

**Rule.** One to three declarative sentences stating *who* did *what*, *when* (if `event_date` is known), and the headline financial figure if applicable. Factually paraphrase the filing's own language. Maximum 500 characters; the schema enforces this via `Field(max_length=500)`.

**Boundary.** Do not include opinion, analyst commentary, market reaction, or forward-looking interpretation unless the filing itself explicitly states it. Do not pad with item codes or filer identification — those are separate fields. Do not verbatim-quote the filing beyond a brief phrase — paraphrase. Stay below 500 chars; if the natural rendering exceeds 500, compress until it fits (drop secondary clauses before dropping the core who/what/when).

**Required content when applicable.**
- If `monetary_amount` is set and represents a per-share value (e.g., dividends), the word `"per share"` must appear in summary — this is how downstream consumers distinguish per-share from aggregate.
- If the filing covers multiple events, the summary describes the primary one (the one driving `primary_category`); secondary events are mentioned only if room allows.

**Tiebreaker.** Lead with the event, not the company. "Acme acquired Beta for $1.5B" is preferred over "Acme Corp announced today that it has entered into a definitive agreement to acquire Beta LLC for approximately $1.5 billion in an all-cash transaction" — the latter spends characters on filing boilerplate that's already captured in `form_type`, `filer_company`, and `event_date`.

**Example.**
- M&A signing: `"On May 1, 2024, Acme Corp entered into a definitive agreement to acquire Beta LLC for $1.5 billion in cash, expected to close in Q3 2024."` (Names the event, the price, and the timing — all from the filing.)
- CEO departure: `"Jane Doe stepped down as CEO of Acme Corp effective May 1, 2024; CFO John Smith is serving as interim CEO while the board conducts a search."`
- Dividend: `"Acme Corp's board declared a quarterly cash dividend of $0.25 per share, payable May 15 to holders of record May 8."` ("per share" is required because monetary_amount=0.25 is the per-share value.)

---

## expected_impact_period

**Rule.** When the financial impact of the disclosed event materializes, per the filing's own language:

| Label | Meaning |
|---|---|
| **immediate** | Already recognized at signing / closing; impact is on the books with this filing |
| **current_quarter** | Within the **filer's fiscal quarter** that contains `filing_date` (not the calendar quarter; many filers have non-calendar fiscal years) |
| **current_fiscal_year** | Within the filer's fiscal year containing `filing_date` |
| **future** | Explicitly later periods, or multi-year impacts not fitting in the current fiscal year |
| **undisclosed** | The filing has a financial impact but does not state timing |

**Boundary.** Use what the filing **states**, not what would be reasonable to infer. A merger filed in May with closing "subject to regulatory approval, expected Q4" → `future`, not `current_fiscal_year`, unless the filing explicitly says recognition will occur this fiscal year. Null when the filing has no financial impact at all (most director appointments without comp; qualitative Reg FD disclosures with no quantified content).

The distinction between `undisclosed` (filing has impact, timing not stated) and null (filing has no financial impact at all) matters for downstream filtering: `undisclosed` is "we have a money event but timing TBD"; null is "no money event."

**Boundary — Item 2.02 earnings releases.** A 2.02 press release announces results for a *just-completed* reporting period (the quarter or fiscal year that ended prior to the filing date). The underlying financial impact is already realized and on the books with this filing. **Use `immediate`.** Do NOT use `undisclosed` (timing IS stated — the just-completed period), `current_quarter` (the just-completed period is typically the PRIOR fiscal quarter, not the one containing `filing_date`), or `current_fiscal_year` (overly broad when `immediate` is the more accurate fit).

**Boundary — governance-only changes.** Auditor changes under Item 4.01 without disclosed audit-fee terms, registered-agent or office changes under Item 5.03, bylaw amendments without disclosed compensation effects, and other governance-only filings → **null**. These changes have financial implications in the abstract (a new auditor will charge fees in future periods) but the FILING itself doesn't disclose a money event, so per the rule above, null is correct. Use `undisclosed` only when the filing references a specific money event whose timing it doesn't pin down (e.g., a settlement amount with no payment date).

**Tiebreaker.** When the filing gives multiple time horizons ("We expect $50M of restructuring costs over the next 18 months, with $20M in Q3"), pick the period covering the **bulk** of the impact (≥50%).

**Example.**
- A same-day-closing tuck-in acquisition with the impact recognized at signing → `immediate`.
- An Item 2.02 8-K announcing Q1 2025 results on April 30, 2025 → `immediate` (Q1 results already realized and on the books at filing).
- A quarterly dividend declared this month, payable next month → `current_quarter`.
- An impairment recorded in the current fiscal year per the filing → `current_fiscal_year`.
- A 5-year supply agreement with no quantified per-period impact → `future`.
- A settlement disclosed without payment-timing language → `undisclosed`.
- A director appointment with no compensation disclosed at all → null.
- An Item 4.01 auditor change with no audit-fee terms disclosed → null.
- An Item 5.03 registered-agent change → null.

---

## Multi-event filings — applying the rules together

When a filing discloses multiple events, apply this hierarchy:

1. **`items`** — list every invoked item code in cover-page order. No collapsing, no selection.
2. **`primary_category`** — apply the category-priority table (m_and_a > regulatory > financial_results > executive_change > material_agreement > other). Within a single category, lowest item number wins.
3. **`event_date`** — the triggering date of the primary item (per the event_date table).
4. **`counterparties`** — every named party across all events, listed in body order. Don't restrict to the primary event.
5. **`monetary_amount` / `currency` / `amount_type`** — describe the **primary event's** headline figure only. Other monetary amounts may be mentioned in `summary` if budget allows.
6. **`summary`** — lead with the primary event; mention secondary events in a second sentence if the 500-char budget allows.
7. **`expected_impact_period`** — describe the impact timing of the primary event.

**Principle.** Secondary events are surfaced through `items` (always) and `summary` (when room allows). Structured monetary fields belong to the primary event so downstream aggregation isn't mixing apples and oranges.

---

## Common labeler mistakes

These are the failure modes that show up most often in initial labeling rounds. Read this list before starting; re-read it after every 50 examples.

1. **Using brand name instead of legal name.** `"Google"` instead of `"Alphabet Inc."`; `"Facebook"` instead of `"Meta Platforms, Inc."`. Always copy from the cover-page registrant line, not from memory.
2. **Forgetting Item 9.01.** When the filing attaches a press release or exhibit, 9.01 IS in `items`. Easy to miss because it's clerical, but it changes downstream models that look for "filings with exhibits."
3. **Treating Item 7.01 wrapper as the primary category.** If the substantive event is earnings (Item 2.02) and Item 7.01 just wraps the press release, primary is `financial_results`. Item 7.01 is a mechanism, not an event.
4. **Confusing `event_date` with `filing_date`.** Most 8-K items must be filed within four business days of the triggering event (Form 8-K General Instruction B.1). So `event_date` is often within the prior week of `filing_date` but not identical. When the body says "On May 1, the Company entered…" and the filing date is May 3, `event_date = 2024-05-01` and `filing_date = 2024-05-03`. Don't collapse them.
5. **Setting `monetary_amount` without setting both `currency` and `amount_type`** (or vice versa). The schema validator catches this, but only at validation time — fix it at labeling time.
6. **Inferring `currency` as USD for U.S. filers.** A U.S. parent's foreign subsidiary may sign agreements in EUR/JPY/GBP; the filing's stated currency wins.
7. **Negative `monetary_amount` for losses.** The schema rejects this. A $50M impairment is `monetary_amount = 50_000_000` with `amount_type = "loss"`. The label carries the sign.
8. **Aggregating dividends without verifying.** Dividend 8-Ks almost always state per-share only. Don't multiply by shares-outstanding from a different source; use the per-share value and annotate "per share" in `summary`.
9. **Listing counterparties not named in the filing.** If the filing says "and other lenders," the unnamed ones are NOT counterparties for this field. Only named entities.
10. **Wrong `event_date` for multi-step transactions.** Use the table under `event_date` to pick the right triggering date for the primary item.
11. **Using "Date of Report" as `filing_date`.** The body's cover-page line "Date of Report (Date of earliest event reported): April 24, 2025" is the *event_date*, not the *filing_date*. The signature block at the end of the 8-K — "Date: April 30, 2025 /s/ [Officer]" — is conventionally the SEC acceptance date (also visible in EDGAR `<ACCEPTANCE-DATETIME>`). Both dates often appear in the same filing; do not collapse them. If they conflict, the EDGAR acceptance stamp wins.
12. **Importing historical / external tickers for `filer_ticker` when the cover says "None".** The cover-page `Trading Symbol(s)` field is the authoritative source for post-2019 filings. When it literally reads "None", `filer_ticker` is null — regardless of whether the issuer was historically listed under a known symbol. Spec only permits the `company_tickers.json` fallback for pre-2019 filings.
13. **Substance-overriding the category priority.** When a filing's items list contains items mapping to multiple categories, priority is binding. The only substance reads allowed are (a) Item 1.01 m_and_a vs material_agreement disambiguation (required because the item itself is ambiguous), and (b) the explicit "M&A solely under Item 8.01" override (when items map only to `other` and body is substantively M&A). Do not substance-override otherwise.
14. **Wrong `expected_impact_period` for Item 2.02 earnings.** A 2.02 announces a *just-completed* period's results, already on the books → `immediate`. Not `undisclosed` (timing IS known — the period just ended), not `current_quarter` (that's the quarter containing filing_date, typically the period AFTER the one being reported).

---

## Edge cases — append as you find them

When labeling, if this doc doesn't tell you what to do:

1. **Stop.** Don't guess. A silent guess becomes an inconsistent label across the 300 examples in the gold set.
2. **Document the case** below as a dated subsection (`### YYYY-MM-DD — [short description]`) with: the filing's accession number, what the filing said, what makes it ambiguous, and what rule has been decided.
3. **Apply the new rule** going forward.
4. **Re-label any earlier examples** the new rule affects. Note in the maintenance log which examples were re-labeled.

This doc is versioned. Every gold-set commit should be reproducible from a specific revision of this doc — if a label changes because the rule changed, the diff in the doc explains why.

### 2026-05-18 — Initial guidelines

First pass. No filing-specific edge cases logged yet.

### 2026-05-19 — Item 2.02 earnings: `expected_impact_period = "immediate"`

Five calibration-set 2.02 earnings filings (accessions `0001051470-25-000124`, `0001217234-25-000036`, `0001214816-25-000112`, `0001410636-25-000084`, `0001466258-25-000121`) surfaced inconsistent labeler picks for `expected_impact_period` (variously `undisclosed`, `current_quarter`). The doc as drafted didn't say which to use. Decision: 2.02 announces results for a *just-completed* reporting period; results are already realized and on the books with the filing; `immediate` is the canonical value. Added explicit Boundary line + example to the `expected_impact_period` rule.

### 2026-05-19 — M&A under Item 8.01 (UK schemes of arrangement; intermediate M&A milestones)

Some M&A disclosures by U.S. registrants land under Item 8.01 rather than Item 2.01. Two patterns observed:
- **Foreign-law mechanism**: a U.S. registrant acquires a UK entity via a Companies Act 2006 scheme of arrangement; the scheme doesn't map cleanly to Item 2.01's "Completion of Acquisition or Disposition of Assets" framing, so the filer uses 8.01. Example: `0001628280-25-020832` (CareTrust REIT acquiring Care REIT plc, shareholder-vote outcome disclosure).
- **Intermediate milestone**: shareholder vote results, regulatory clearance updates, or other progress notes on a pending M&A transaction filed before closing — disclosed under 8.01 because 2.01 only fires at completion.

Decision: extend the `primary_category` rule with a bounded **substance override** — when items map only to `other` AND body substance is M&A, categorize as `m_and_a`. `items` field unchanged (cover canonical). Override applies *only* to M&A, not to other categories. Added explicit subsection + examples to `primary_category` rule.

### 2026-05-19 — Camber Energy `filer_ticker` clarification

Accession `0001477932-25-002309` (Camber Energy, Inc.) cover page lists "Securities registered pursuant to Section 12(b) of the Act: None." Labeler filled `filer_ticker = "CEI"` from external knowledge (Camber's historical NYSE American ticker). Doc already says null when "the filer has no equity listing (delisted shell, OTC-only filer)" — but the labeler treated the historical ticker as authoritative. Reinforcing the existing rule: **when the cover-page `Trading Symbol(s)` field literally reads "None", `filer_ticker` is null. Do not import historical or external tickers; spec forbids external sources beyond `company_tickers.json` for pre-2019 filings, and post-2019 filings should rely on the cover page exclusively.** No rule change needed; flagged as a common-mistake item below.

---

## Open questions / deferred decisions

Decisions raised but not made, pending evidence from the calibration subset. Each entry states the question, why it's deferred rather than decided, and the trigger condition for revisiting.

### 2026-05-19 — Item 5.02 subsection capture in `items`

The schema regex `^\d\.\d\d$` collapses Item 5.02 subsections (a)–(f) into a single `"5.02"` token. Subsection substance matters for `amount_type`: 5.02(b) officer departure often → `severance`; 5.02(e) compensatory arrangement often → `other`; 5.02(c)/(d) appointments often → null. The substance is recoverable via `amount_type` + `summary`, so the digit-pair-level encoding is workable as a default. **Revisit if:** calibration verification surfaces ≥5 cases where 5.02 subsection ambiguity causes label disagreement or labeler confusion. **Resolution paths:** (a) stay at digit-pair (status quo) or (b) extend schema to `^\d\.\d\d[a-f]?$` with corresponding test, prompt, gold-set, and example updates.

---

## Worked example — end-to-end labeling of a hypothetical filing

This example walks through a complete labeling pass to make the workflow concrete. The filing is hypothetical (clearly marked as `Acme/Beta`) but structurally typical of real M&A closing 8-Ks.

**Filing summary (hypothetical):**
- Filed by `Acme Corp` on May 3, 2024 (acceptance timestamp `20240503143000`).
- CIK: `0000123456`. Ticker: `ACME`.
- Cover page lists items: 2.01, 5.02, 7.01, 9.01.
- Body Item 2.01: "On May 1, 2024, Acme Corp completed its previously-announced acquisition of Beta LLC for total consideration of $1.5 billion in cash."
- Body Item 5.02: "In connection with the closing, Jane Doe, the founder of Beta LLC, has been appointed to the Acme board of directors effective May 1, 2024."
- Body Item 7.01: attaches a press release dated May 1, 2024.
- Body Item 9.01: lists the merger agreement and press release as exhibits.

**Working the fields in order:**

1. `form_type = "8-K"` — header says `<TYPE>8-K`.
2. `filer_cik = "0000123456"` — 10 digits.
3. `filing_date = 2024-05-03` — acceptance timestamp.
4. `filer_company = "Acme Corp"` — cover-page registrant line.
5. `filer_ticker = "ACME"` — primary listing.
6. `items = ["2.01", "5.02", "7.01", "9.01"]` — every code on the cover.
7. `primary_category`: items map to m_and_a (2.01), executive_change (5.02), other (7.01), other (9.01). m_and_a wins the category priority → `"m_and_a"`.
8. `event_date = 2024-05-01` — triggering date of the primary item (2.01 → closing date).
9. `counterparties = ["Beta LLC", "Jane Doe"]` — body order; both are named parties to disclosed events.
10. `monetary_amount = 1_500_000_000`; `currency = "USD"`; `amount_type = "purchase_price"` — primary-event headline figure.
11. `expected_impact_period = "immediate"` — the acquisition closed at the event_date, so impact is on the books now.
12. `summary = "On May 1, 2024, Acme Corp completed its acquisition of Beta LLC for $1.5 billion in cash. Beta founder Jane Doe joined Acme's board effective the same day."` (≈155 chars, well under the 500 limit; covers the primary event with the headline figure plus the secondary event since room allows.)
13. Cross-field consistency check: event_date (May 1) ≤ filing_date (May 3) ✓; monetary_amount/currency/amount_type all set ✓; primary_category m_and_a is derivable from items 2.01 ✓.

**What an interviewer might probe, and where the defense lives in this doc:**

- "Why is the primary category `m_and_a` and not `executive_change`, given that the director appointment is also material?" → The category-priority hierarchy (m_and_a > executive_change), justified by event-study evidence of typical market-impact magnitude.
- "Why do you use $1.5B and not, say, the value of the stock issued to the director?" → The monetary_amount tiebreaker for acquisitions: total consideration tied to the primary event.
- "Why is Jane Doe in `counterparties` and not just in `summary`?" → counterparties rule: every named party across all events, in body order.
- "Why isn't the press release date (May 1) the filing_date?" → filing_date is the SEC's acceptance stamp, not the press release date.

---

## Concepts cheat-sheet

Brief definitions for the financial concepts the rules above assume. Read this first if any term in the field rules is unfamiliar.

- **8-K** — A "current report" the SEC requires public companies to file when material events occur between scheduled quarterly (10-Q) or annual (10-K) reports. Each event type corresponds to a numbered Item. The 8-K is *narrative* disclosure of material events; quarterly reports are *periodic* disclosure of financials.
- **8-K/A** — Amendment to a previously-filed 8-K. Used to correct or add information — commonly to attach historical financial statements of an acquired business under Item 9.01, which the original 8-K may have deferred.
- **Item 1.01 / Material Definitive Agreement** — Entry into a contract material to the company (M&A agreements, large supply contracts, credit facilities, IP licenses). Distinct from completion (Item 2.01) — 1.01 is signing.
- **Item 1.02 / Termination of a Material Definitive Agreement** — Counterpart to 1.01. When a previously-disclosed material agreement ends.
- **Item 2.01 / Completion of Acquisition or Disposition of Assets** — Closing of an M&A transaction. The 8-K under 2.01 typically post-dates the 8-K under 1.01 by weeks to months (the regulatory approval period between signing and closing).
- **Item 2.02 / Results of Operations and Financial Condition** — Earnings announcements. Used when the filer publicly announces results before the formal 10-Q or 10-K is filed.
- **Item 2.03 / Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement** — Taking on new debt, signing a credit facility, issuing securities with debt-like features.
- **Item 2.04 / Triggering Events That Accelerate or Increase a Direct Financial Obligation** — When a previously-disclosed obligation has a trigger event (covenant breach, change-of-control acceleration, etc.).
- **Item 2.05 / Costs Associated with Exit or Disposal Activities** — Restructuring announcements; commitment to a plan that will result in material costs.
- **Item 2.06 / Material Impairment** — Filer determines a material asset (most commonly goodwill from a prior acquisition) has lost value and must be written down. Recognized as a non-cash charge.
- **Item 3.01 / Notice of Delisting or Failure to Satisfy a Continued Listing Rule** — Exchange has notified the filer of delisting or listing-rule violation.
- **Item 3.02 / Unregistered Sales of Equity Securities** — Issuance of equity not registered under the Securities Act (private placements, etc.).
- **Item 3.03 / Material Modification to Rights of Security Holders** — Changes affecting existing shareholders' rights (e.g., creating a new class of preferred stock with priority).
- **Item 4.01 / Changes in Registrant's Certifying Accountant** — Auditor change.
- **Item 4.02 / Non-Reliance on Previously Issued Financial Statements** — Restatement disclosure: the audit committee has determined prior financial statements should no longer be relied on.
- **Item 5.01 / Changes in Control of Registrant** — A change of control has occurred (typically via M&A, sometimes via voting agreement).
- **Item 5.02 / Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers** — Executive changes. Subsections (a)–(f): (a) director departure; (b) certain officer resignation/termination; (c) appointment of certain officers; (d) election of new directors; (e) compensatory arrangements of certain officers; (f) salary/bonus determinations not previously disclosed.
- **Item 5.03 / Amendments to Articles of Incorporation or Bylaws** — Charter or bylaw changes.
- **Item 7.01 / Regulation FD Disclosure** — Discloses material non-public information to satisfy Reg FD's fair-disclosure rule. Item 7.01 is the *mechanism* by which a press release is filed alongside an investor call or other selective communication.
- **Item 8.01 / Other Events** — Catch-all for material events not covered by other items. Filers often use 8.01 for share-repurchase authorizations, dividend declarations, regulatory updates that don't trigger 3.x/4.x items.
- **Item 9.01 / Financial Statements and Exhibits** — Attached documents: press releases, executed agreements, audited financials. Appears whenever exhibits are attached, which is most filings but not all.
- **CIK (Central Index Key)** — The SEC's permanent unique ID for a legal entity. 10 digits when zero-padded for display. One entity → one CIK forever; tickers and names can change.
- **Purchase consideration** — Total value transferred in an acquisition: cash paid + market value of stock issued + debt assumed + the fair value of any contingent earn-outs. The headline "deal size" the filing announces.
- **Severance** — Pay to a departing executive beyond accrued salary. Typically cash plus accelerated equity vesting. Disclosed under Item 5.02(e) when material.
- **Impairment** — An accounting recognition that a previously-recorded asset (goodwill, intangibles, PP&E) is worth less than its carrying value. Recognized as a non-cash loss.
- **Reg FD (Regulation Fair Disclosure)** — SEC rule requiring simultaneous disclosure of material non-public information to all market participants. The reason Item 7.01 exists.
- **Material agreement** — A contract whose existence or terms would matter to a reasonable investor. No bright-line dollar threshold; the filer applies judgment subject to SEC enforcement.
- **Counterparty** — In a contract or transaction, the *other* named party — across the table from the filer.
- **Earn-out** — A deferred component of M&A consideration paid only if the acquired business hits specified performance targets post-close. Often disclosed in the original 1.01/2.01 filing; payment triggers may be disclosed in later 8-Ks.

---

## Maintenance log

Append a dated entry every time a rule in this doc changes. Reference the affected gold-set examples by accession number when applicable.

- **2026-05-18** — Initial draft of all 14 field rules + multi-event hierarchy + concepts cheat-sheet + worked example + inter-annotator agreement targets + common-mistakes section + labeling workflow. Drafted in three critical-review passes the same day; revisions in passes 2 and 3 included: replacing the "first-named on cover" tiebreaker for `primary_category` with the category-priority hierarchy; reframing the priority justification as structured-extraction value, not pure market impact; restructuring inter-annotator agreement targets by field-determinism level (1.00 / ≥0.95 / kappa ≥0.70 / qualitative); rebuilding `filer_ticker` around the cover-page `Trading Symbol(s)` field; fixing the Item 2.02 `event_date` rule to use the announcement date, not the period end; resolving the dividend-convention contradiction (aggregate-preferred, per-share-fallback); softening the Microsoft–Activision example to require per-filing verification; acknowledging Item 1.01 substance ambiguity in cross-field consistency checks.
- **2026-05-19** — Tiebreaker review (Step 1.4 of the project plan). Four load-bearing rules explicitly reviewed against their strongest defensible alternatives and affirmed unchanged: `primary_category` priority order (m_and_a > regulatory > financial_results > executive_change > material_agreement > other); dividend `monetary_amount` per-share convention with "per share" required in `summary`; all-stock M&A null rule (no external-price computation when the filing provides no dollar headline); `expected_impact_period` bulk-50% threshold for multi-period impacts. No rule changes; no gold-set re-labeling. Item 5.02 subsection capture in `items` raised and deferred pending calibration evidence — see "Open questions / deferred decisions."
- **2026-05-19 (third update)** — Formalized the **verification protocol** as LLM dual-pass (Claude Sonnet 4.6 labeler + Claude Opus 4.7 critic), not human dual-label. Rewrote the "Inter-annotator agreement targets" section as "Verification protocol" with explicit acknowledgement that the Pass-1↔Pass-2 agreement metric is **intra-Claude consistency** and not the classical inter-annotator-agreement signal. Added optional human-spot-check tier (`provenance.human_spot_checked = true`) as a sanity floor on the final gold set. Updated the "How to read this doc" section to clarify that "labeler"/"reviewer"/"annotator" refer to the two Claude passes, not humans. No field rules changed; no gold-set re-labeling triggered by this update — but the new guidelines SHA *will* trigger a re-label of all 50 calibration rows on the next `--rerun`, since the labeler reads this doc as its system prompt and the doc content changed.
- **2026-05-19 (second update)** — Calibration audit on first 30 of 50 LLM-labeled gold-set examples (sessions `llm-review-2026-05-19-001` + `llm-review-audit-2026-05-19-002` in `data/gold/v1.jsonl`) surfaced three rule gaps and four labeler failure modes. Added:
  - **`expected_impact_period` rule** for Item 2.02 earnings releases (`immediate`) and for governance-only changes — auditor under 4.01, registered agent under 5.03 without compensation effects, bylaw amendments without comp effects — (null). Affected gold-set rows (will need `--rerun`-style re-labeling under the new doc SHA): 0001051470-25-000124, 0001217234-25-000036, 0001214816-25-000112, 0001410636-25-000084, 0001466258-25-000121 (2.02 earnings); 0001477932-25-003004 (auditor change).
  - **`primary_category` rule** — explicit "apply priority strictly when items map to multiple categories" admonition; explicit "M&A solely under Item 8.01" substance override for UK schemes-of-arrangement and intermediate-milestone disclosures. Affected gold-set row: 0001628280-25-020832.
  - **Common labeler mistakes** — added items 11–14 covering: (a) using "Date of Report" as `filing_date` instead of EDGAR acceptance / signature date (3 first-pass errors in calibration: 0000004127-25-000034, 0001104659-25-042501, 0001606498-25-000085; +2 missed by first-pass audit reviewer and caught on second-pass: 0001193125-25-079142, 0001331451-25-000084), (b) importing historical tickers for `filer_ticker` when cover says "None" (0001477932-25-002309, 0001213900-25-036865), (c) substance-overriding the category priority (0001213900-25-036074, 0001193125-25-107501), (d) wrong `expected_impact_period` for 2.02. No retroactive re-label run executed by this update; the next `python scripts/llm_label_gold_set.py --rerun` will pick up the new guidelines SHA and re-label the 50 calibration filings.
