# Analysis: KPI-Ontology-First Approach for Dashboard Rationalization

## The Proposed Approach (Manager's Suggestion)

> **Before** applying data source filters, start from **dashboard visuals** → pick **major KPIs** → align those KPIs with an **ontology** → use that ontology to **compare between reports**.

In other words: **flip the order** — instead of starting from data sources (bottom-up), start from the KPIs/visuals that users actually see (top-down).

---

## Verdict: ✅ Feasible & Valuable — But as a **Complement**, Not a Replacement

This approach is sound and addresses a real gap, but it has trade-offs. Below is the full breakdown.

---

## What This Approach Gets Right

### 1. Business-Intent Alignment
Your current spec starts at the **data source layer** (Stage 0: blocking by schema fingerprint). This is technically optimal for pruning pairs, but it's **infrastructure-centric** — it compares *how* reports are built, not *what business questions they answer*.

The KPI-ontology approach flips this: two dashboards showing "Claims Loss Ratio" and "Loss Experience Ratio" might use completely different data sources, DAX, and visuals, but they answer the **same business question**. An ontology would catch this; a data-source filter would miss it entirely.

### 2. Cross-Source Discovery
Your current Stage 0 groups reports by logical data source. Reports using **different data sources** that track the **same KPI** would never be compared. For example:
- Report A: "Revenue" from SQL Server warehouse
- Report B: "Revenue" from a CSV export of the same warehouse

If schema fingerprints don't match, these never enter the comparison pipeline. A KPI ontology would flag them as semantically identical.

### 3. Stakeholder-Friendly Output
Business users and governance boards care about "we have 14 dashboards tracking Revenue" — not "we have 14 reports sharing 80% schema overlap." The ontology gives you a **business-language clustering** that is immediately actionable for stakeholders.

### 4. Subsumption Becomes Clearer
With a KPI ontology, subsumption is trivial to express: "Dashboard A tracks {Revenue, COGS, Margin}. Dashboard B tracks {Revenue, COGS, Margin, EBITDA, Net Income}. A ⊂ B."

---

## What You'll Miss / Risks

### ⚠️ Risk 1: KPI Extraction Is Harder Than It Sounds

**Problem**: "Pick major KPIs from dashboard visuals" assumes KPIs are cleanly identifiable. In Power BI, a "KPI" might be:
- A **Card visual** showing a DAX measure → easy
- A **column in a matrix** with a complex calculated column → medium
- A **value axis** in a chart with an implicit SUM aggregation → hard
- A **title text** that says "Revenue" but the bound measure is `[Measure_47]` → very hard

**What you need**: An extraction layer that maps `visual → bound measure → semantic KPI name`. This is essentially what your current **Visual Layer (§3.4)** + **DAX Measures Layer (§3.3)** already do, but without the ontology mapping step.

> [!IMPORTANT]
> You'll need an **LLM-assisted KPI extraction** step that goes beyond just reading measure names. It needs to understand that `[SUM_Rev_Q3_Adj]` maps to the ontology concept "Adjusted Revenue."

### ⚠️ Risk 2: Ontology Construction Is a Significant Upfront Investment

**Problem**: Where does the ontology come from?

| Option | Effort | Quality |
|--------|--------|---------|
| Manual curation by business SMEs | High (weeks) | High |
| LLM-generated from measure names across 921 reports | Medium (days) | Medium — needs human review |
| Industry-standard ontology (e.g., FIBO for finance) | Low (import) | Low — too generic, poor coverage |
| Bottom-up clustering of measure names → ontology | Medium | Medium — misses synonyms |

**Recommendation**: Use a **hybrid approach** — LLM-cluster all 921 reports' measure names into candidate KPI concepts, then have SMEs review/refine. This gives you 80% coverage quickly.

### ⚠️ Risk 3: Ontology Granularity Problem

**Problem**: At what level do you define KPI concepts?

```
Too Coarse:                    Too Fine:
"Revenue"                      "Adjusted Net Revenue excl. Tax for APAC Region Q3 FY24"
  → Groups 200 reports           → Groups 1 report
  → Not actionable               → No rationalization value
```

You need a **hierarchical ontology** with at least 3 levels:
```
Domain → Category → Specific KPI
Financial → Revenue → Adjusted Net Revenue
Financial → Revenue → Gross Revenue
Financial → Costs → COGS
Financial → Costs → Operating Expenses
```

### ⚠️ Risk 4: You Lose Structural Precision

**Problem**: Two reports might track the same KPI ("Revenue") but:
- One computes it as `SUM(Sales[Amount])` — correct
- The other computes it as `SUMX(ALL(Sales), Sales[Amount])` — ignores filters, subtly wrong

The KPI ontology would say "same KPI" but the DAX comparison (your current §3.3) would correctly flag them as **semantically different**. Without the structural layer, you'd incorrectly recommend merging them.

### ⚠️ Risk 5: Filters and Context Matter

**Problem**: "Revenue" filtered to APAC vs. "Revenue" filtered to Global are the **same KPI** but **different reports**. The ontology alone can't distinguish them. You still need the **Filter & Slicer Layer (§3.5)** to avoid false positives.

---

## Recommended Hybrid Architecture

Instead of choosing one approach, **combine both**:

```
Current Spec (Bottom-Up):              Manager's Approach (Top-Down):
─────────────────────────               ───────────────────────────────
Data Sources → Schema Match             Visuals → KPI Extraction
     ↓                                       ↓
DAX Measures → AST Compare              KPI → Ontology Mapping
     ↓                                       ↓
Visuals → Type + Binding                 Ontology → Cross-Report Match
     ↓                                       ↓
Filters → Scope Match                   Filter Context Overlay
     ↓                                       ↓
Composite Score                          Business-Intent Score
         ↘                            ↙
          ─────────────────────────────
          │  MERGED RATIONALIZATION   │
          │  Structural + Semantic    │
          │  Score                    │
          ─────────────────────────────
```

### Concrete Integration Points

| Current Spec Layer | KPI Ontology Enhancement |
|---|---|
| **Stage 0: Blocking** | Add **KPI-based blocking**: reports sharing ≥1 ontology concept are candidates even if data sources differ |
| **§3.3 DAX Layer** | After bipartite matching, tag each matched pair with its **ontology concept** for explainability |
| **§3.4 Visual Layer** | Replace raw visual comparison with **KPI-visual alignment**: does this visual serve the same KPI concept? |
| **§4.3 Subsumption** | Express subsumption in ontology terms: "Report A covers 3 of 5 KPI concepts in Report B" |
| **Stage 4: Explainability** | Generate recommendations in ontology language: "14 reports track Revenue — consolidate to 3" |

---

## What's Missing From Both Approaches

Regardless of which approach you use, make sure these aren't overlooked:

### 1. **Usage Analytics** (Neither approach covers this)
A report with 500 daily active users tracking "Revenue" should NOT be merged into a report with 2 users. Your [architecture doc](file:///c:/Users/madhu/Desktop/excelrationlization/architecture_slide_content.md) mentions "usage analytics" in the Discovery Agent but the spec doesn't weight it in scoring.

### 2. **Temporal/Version Drift**
Reports evolve. "Revenue v1" and "Revenue v3" might have the same KPI ontology label but diverged significantly. You need a **freshness signal** (last modified date, refresh frequency).

### 3. **Owner/Consumer Mapping**
Two reports showing the same KPI but owned by different teams (Finance vs. Sales) might intentionally exist. The ontology should include **domain ownership** as a dimension.

### 4. **Data Freshness Differences**
Same KPI, same data source, but one refreshes hourly and the other daily. Merging them loses the hourly granularity. This is partially covered by your Governance Layer (§3.6) but not scored.

---

## Summary Decision Matrix

| Criteria | Data-Source-First (Current) | KPI-Ontology-First (Manager) | Hybrid (Recommended) |
|---|---|---|---|
| Catches structural duplicates | ✅ Excellent | ⚠️ Misses some | ✅ Excellent |
| Catches semantic/business overlaps | ⚠️ Weak across different sources | ✅ Excellent | ✅ Excellent |
| Computational cost | ✅ Cheap blocking | ⚠️ Ontology construction cost | ⚠️ Both costs |
| Stakeholder explainability | ⚠️ Technical | ✅ Business-friendly | ✅ Business-friendly |
| False positive rate | ✅ Low | ⚠️ Higher (same KPI ≠ same report) | ✅ Low |
| False negative rate | ⚠️ Higher (misses cross-source) | ✅ Low | ✅ Low |
| Implementation effort | ✅ Already designed | ⚠️ Needs ontology + extraction | ⚠️ Incremental over current |

---

## Bottom Line

> [!TIP]
> **Tell your manager**: "Yes, this is feasible and valuable. We recommend integrating it as an **additional blocking/matching signal** alongside the data-source cascade — not replacing it. The KPI ontology gives us business-language explainability and catches cross-source overlaps that the structural approach misses. We'll need ~1-2 weeks to build the ontology extraction + alignment layer."

> [!WARNING]
> **Don't skip the structural comparison**. The ontology alone will produce false positives (same KPI name, different computation logic) and miss filter-context differences. You need both layers for accurate rationalization at 921-report scale.
