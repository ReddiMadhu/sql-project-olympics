# Power BI Dashboard Rationalization Engine — Algorithmic Specification

## 1. Problem Statement

Given an estate of **921 Power BI reports** (scaling to 1,000+), with all metadata already extracted into a **normalized relational database** (SQLite for dev, PostgreSQL for production), design and implement a **comparison engine** that:

1. Detects **exact clones** (structurally identical reports)
2. Detects **near-clones** (90%+ identical with minor tweaks)
3. Detects **functional overlap** (reports answering the same business questions, possibly built differently, within the same data source scope)
4. Identifies **subsumption** (Report A is a strict subset of Report B)
5. Produces **graph-based clusters** of related reports
6. Recommends **Decommission / Merge / Keep / Review** for each report
7. Identifies a **golden report** per merge cluster

---

## 2. Comparison Architecture Overview

### 2.1 Atomic Unit of Comparison

- **Report-level**: The entire `.pbix` report is the unit of comparison. Pages and visuals are metadata *within* the report, not independently compared entities.

### 2.2 Comparison Scope

- **Cross-source equivalence** is **in scope** when schema fingerprints indicate logical equivalence (e.g., same data from CSV vs. DB with matching column names and types).
- **Physical source type** (CSV, SQL, API) is not a blocker — logical schema equivalence determines comparability.

### 2.3 Algorithm Strategy: Cascading Multi-Layer Comparison

The engine uses a **cascade filter architecture**: cheap comparisons eliminate non-candidates early, expensive comparisons (LLM, AST parsing) run only on surviving pairs.

```
┌─────────────────────────────────────────────────────────┐
│  Stage 0: Blocking                                      │
│  Group reports by logical data source (schema fingerprint│
│  matching with fuzzy column-name comparison)            │
│  Reduces ~424K pairs → ~50K candidate pairs             │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Coarse Filtering (MinHash + SimHash)          │
│  Per-layer fingerprints eliminate dissimilar pairs       │
│  Reduces ~50K pairs → ~5K candidate pairs               │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Deep Comparison                               │
│  Per-layer scoring with appropriate algorithms          │
│  Produces per-layer similarity scores for ~5K pairs     │
├─────────────────────────────────────────────────────────┤
│  Stage 3: Classification & Clustering                   │
│  Weighted composite → tiered classification             │
│  Graph-based clustering with transitivity               │
│  Golden report selection per cluster                    │
├─────────────────────────────────────────────────────────┤
│  Stage 4: Explainability & Output                       │
│  LLM-generated summaries for stakeholders               │
│  Per-layer breakdown + recommendations                  │
│  Persist results to DB for frontend consumption         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Metadata Layers & Per-Layer Algorithms

Each metadata layer uses a **different comparison algorithm** suited to the data structure of that layer. All layers produce a **0-100% similarity score**.

### 3.1 Data Source Layer (Weight: 25%)

**Elements compared**: Server, database, tables, views, connection type.

**Algorithm**:
- **Fuzzy column-name matching** using Levenshtein distance + synonym detection for cross-source schema matching.
  - `CustomerName` ≈ `Customer_Name` ≈ `cust_name`
  - Threshold: Levenshtein distance ≤ 2 (for names ≥ 5 characters; names < 5 characters require exact match or embedding cosine > 0.85 to avoid false positives on short identifiers like `"ID"` vs `"UK"`), OR embedding cosine similarity > 0.85 for column names.
  - **PostgreSQL acceleration**: In production, use `pg_trgm` extension for trigram-based fuzzy matching directly in SQL. Register the extension (`CREATE EXTENSION IF NOT EXISTS pg_trgm`) and use `similarity()` / `%` operator for column-name matching:
    ```sql
    -- Find fuzzy column-name matches across sources using pg_trgm
    SELECT a.column_name, b.column_name, 
           similarity(a.column_name_normalized, b.column_name_normalized) AS sim
    FROM source_a_columns a
    CROSS JOIN source_b_columns b
    WHERE a.column_name_normalized % b.column_name_normalized  -- GIN index-accelerated
      AND similarity(a.column_name_normalized, b.column_name_normalized) > 0.6;
    ```
  - Create GIN indexes on normalized column names: `CREATE INDEX idx_col_trgm ON columns USING gin (column_name_normalized gin_trgm_ops);`
  - **SQLite fallback**: For dev, implement Levenshtein in Python since SQLite lacks `pg_trgm`. Use embedding cosine similarity as the primary fuzzy matcher in dev.
- **Schema fingerprint**: Hash of `{sorted(column_names_normalized) + data_types}`.
- **Logical source grouping**: Two physically different sources (CSV vs. DB) are treated as the **same logical source** if:
  - Column-name fuzzy match rate > 80% (via `pg_trgm` in prod, embeddings in dev)
  - Data types are compatible (e.g., `VARCHAR` ≈ `TEXT`)
  - DAX expressions reference columns from both sources in similar patterns (cross-validated with DAX layer).
- **Scoring**: Jaccard similarity on normalized table/column sets.

### 3.2 Semantic Model Layer (Weight: 20%)

**Elements compared**: Tables, columns, relationships (topology), hierarchies, KPIs.

**Algorithm**:
- **Table/Column comparison**: Weighted Jaccard on column sets per table. TF-IDF weighting applied (see §4.2).
- **Relationship topology fingerprint**:
  - Encode the relationship graph as an adjacency matrix: `{(table_A, table_B, cardinality, cross_filter_direction)}`.
  - Hash the sorted, canonical representation of this adjacency set.
  - Compare using Jaccard on the edge set (treating each relationship as a set element).
  - Cardinality mismatches (one-to-many vs. many-to-many) reduce the edge similarity by 50%.
  - Cross-filter direction mismatches reduce edge similarity by 25%.
- **Hierarchy comparison**: Jaccard on hierarchy definitions `{hierarchy_name: [level_columns]}`.
- **Sub-scores aggregated**: `0.40 * column_similarity + 0.35 * relationship_similarity + 0.15 * hierarchy_similarity + 0.10 * kpi_similarity`.

### 3.3 DAX Measures Layer (Weight: 30% — CRITICAL)

**Elements compared**: All DAX measures and calculated columns.

**Algorithm — Three-Stage Pipeline**:

#### Stage A: Signature-Based Matching (Fast, Coarse)

For each measure, compute a **signature**:
```
signature = {
    input_columns: sorted([referenced_columns]),
    aggregation_function: primary_aggregation (SUM, AVERAGE, COUNT, etc.)
                          or 'none' for non-aggregate measures,
    output_data_type: resolved_type,
    dependency_subgraph: sorted([referenced_measures])
}
```

**Non-aggregate measures** (e.g., `[Revenue] * 1.1`, `IF(...)`, pure calculated columns): Set `aggregation_function = 'none'`. These are still signature-matched if they share the same input columns, output type, and dependency subgraph — the absence of an aggregation function is itself a matching category.

Two measures are **signature-matched** if:
- Same input columns (after fuzzy name normalization)
- Same aggregation function category (including `'none'`)
- Compatible output data type

#### Stage B: AST-Level Structural Comparison (Medium, Structural)

For signature-matched pairs, parse DAX into an **Abstract Syntax Tree (AST)**:
- Normalize variable names to canonical placeholders (`_var1`, `_var2`).
- Normalize whitespace and formatting.
- Compute **tree edit distance** between normalized ASTs.
- Score: `1 - (edit_distance / max_tree_size)`.

**Rewrite rules — two-tier system** (NOT blanket equivalences):

Rewrite rules are split into **safe** (context-independent, applied unconditionally during AST normalization) and **conditional** (context-dependent, used as routing signals to LLM, never as automatic equivalences).

```python
# Tier 1: SAFE — apply unconditionally during AST normalization, no context dependency
SAFE_REWRITES = [
    ("IF(cond, BLANK(), value)", "IF(NOT(cond), value)"),        # boolean negation only
    ("cond1 && cond2", "AND(cond1, cond2)"),                      # pure syntax variant
    # NOTE: "a + b" → "SUM(a, b)" was REMOVED — DAX SUM() takes a single column
    # argument, not two scalars. There is no safe rewrite for the + operator.
]

# Tier 2: CONDITIONAL — gate on context-safety checks before applying
CONDITIONAL_REWRITES = [
    {
        "pattern": "CALCULATE(SUM(T[C]), FILTER(T, pred))",
        "equivalent_to": "SUMX(FILTER(T, pred), T[C])",
        "preconditions": [
            "no_other_active_filters_in_calling_context",
            "single_table_single_column_predicate",
            "no_all_or_allexcept_in_scope",
        ],
    },
    {
        "pattern": "TOTALYTD(expr, dates)",
        "equivalent_to": "CALCULATE(expr, DATESYTD(dates))",
        "preconditions": [
            "same_date_column",
            "no_fiscal_year_offset_param",
        ],
    },
    # ... additional conditional rules, each with explicit preconditions
]
```

**Routing logic**: Conditional rewrites do NOT reduce edit distance or boost scores. They control whether the pair is routed to LLM review:

```python
def classify_ast_match(measure_a, measure_b, ast_a, ast_b):
    edit_distance = tree_edit_distance(ast_a, ast_b)
    raw_score = 1 - (edit_distance / max_tree_size(ast_a, ast_b))

    # Safe rewrites are already folded into AST normalization (reduce edit distance directly)
    # Conditional rewrites do NOT reduce edit distance — they only adjust routing:
    if matches_conditional_rewrite(ast_a, ast_b) and not preconditions_verified(ast_a, ast_b):
        # Don't upgrade the score. Force this pair into LLM review
        # regardless of where raw_score falls, and flag the reason.
        return {
            "score": raw_score,
            "route": "llm_required",
            "reason": "conditional_rewrite_precondition_unverified"
        }

    # Ambiguity zone (§Stage C): 50-90% AST similarity is too uncertain for
    # deterministic classification — route to LLM for semantic judgment.
    if 0.50 <= raw_score <= 0.90:
        return {
            "score": raw_score,
            "route": "llm_required",
            "reason": "ambiguous_ast_similarity"
        }

    return {"score": raw_score, "route": "standard"}
```

**Consequence for the pipeline**: Stage B no longer silently promotes conditional-rewrite pairs to "equivalent." It routes them to Stage C (LLM) with the specific rewrite rule and unmet precondition included in the prompt, so the LLM is judging a narrowed, well-posed question ("is FILTER predicate `pred` free of interaction with other active filters here?") rather than open-ended equivalence. This costs more LLM calls but removes the correctness risk from the 30%-weighted layer.

**Calibration validation gate**: During calibration (§8), manually verify every conditional-rewrite match in the labeled sample. If any produces a wrong equivalence, that rule's precondition set is insufficient — tighten it before the full run, don't just lower confidence.

#### Stage C: LLM Semantic Judgment (Expensive, Edge Cases)

For pairs where:
- AST similarity is **between 50-90%** (ambiguous zone), OR
- A conditional rewrite was matched but **preconditions could not be verified**, OR
- The DAX parser could not parse the measure (coverage gap — see §10.1)

Send both DAX expressions to **Azure OpenAI** with a structured prompt:
```
You are a DAX expert. Given two DAX measures and their evaluation context,
determine if they are semantically equivalent (will always produce the same 
result given the same filter context).

Measure A: [DAX_A]
Measure A context: [referenced tables, columns, active filters, row context]
[If applicable: This pair matches conditional rewrite pattern X but
precondition Y could not be verified. Focus your analysis on whether Y holds.]

Measure B: [DAX_B]
Measure B context: [referenced tables, columns, active filters, row context]

Consider: CALCULATE context transitions, row context vs filter context,
BLANK propagation, iterator vs aggregator semantics.

Respond with JSON: {
  "equivalent": bool,
  "confidence": float (0-1),
  "edge_cases": ["list of contexts where they might differ"],
  "explanation": "string"
}
```
- **Context-aware caching**: Key the LLM response cache by a composite hash of both the DAX expressions and their corresponding contexts: `sha256(DAX_A + context_fingerprint_A + DAX_B + context_fingerprint_B)`. The context fingerprint is a stable hash of the sorted referenced table names, column names, and active filters. This ensures textually identical expressions analyzed in different model/context topologies do not trigger false cache hits.
- Batch API calls (up to 5 pairs per request to stay within context window limits).
- When a conditional rewrite rule triggered the LLM call, include the specific rule and unmet precondition in the prompt so the LLM judges a **narrowed question**, not open-ended equivalence.

#### Function Contracts and Failure Propagation

```python
def compare_dax_measure_pair(measure_a, measure_b, ast_a, ast_b) -> dict:
    """
    Compares a single pair of measures.
    Returns: {
        'equivalent': bool,
        'score': float,
        'method': str ('exact' | 'signature' | 'ast' | 'llm'),
        'explanation': str,
        'llm_degraded': bool  # True if the LLM call timed out / exhausted retries
    }
    """
    if is_parseable(measure_a) and is_parseable(measure_b):
        ast_match = classify_ast_match(measure_a, measure_b, ast_a, ast_b)
        if ast_match['route'] == 'standard':
            return {
                'equivalent': ast_match['score'] > 0.90,
                'score': ast_match['score'],
                'method': 'ast',
                'explanation': 'Determined via AST tree edit distance.',
                'llm_degraded': False
            }
    
    # Parser coverage gap OR ambiguous AST OR conditional rewrite unverified -> route to LLM
    try:
        result = llm_semantic_judgment(measure_a, measure_b)
        return {
            'equivalent': result['equivalent'],
            'score': 1.0 if result['equivalent'] else 0.0,
            'method': 'llm',
            'explanation': result['explanation'],
            'llm_degraded': False
        }
    except LLMUnavailabilityError:
        # Fallback behaviour: degrade to not-equivalent, flag degradation
        return {
            'equivalent': False,
            'score': 0.0,
            'method': 'llm',
            'explanation': 'LLM unavailable during semantic judgment.',
            'llm_degraded': True
        }

def compare_dax_layer(report_a, report_b) -> dict:
    """
    Compares the full set of DAX measures in report_a and report_b.
    Returns: {
        'score': float,                 # overall bipartite matching score
        'measure_matches': list,        # list of matches stored in measure_equivalences
        'llm_degraded': bool            # True if any constituent pair has llm_degraded=True
    }
    """
```

#### Measure Set Comparison

After individual measure equivalence is established:
- Build a **bipartite matching** between Report A's measures and Report B's measures.
- Use the Hungarian algorithm for optimal assignment.
- **TF-IDF weighting** (see §4.2): Measures appearing in many reports (e.g., generic 'Revenue') contribute less to the similarity score than rare, domain-specific measures.
- **Similarity score** (Jaccard-style, symmetric): `sum(matched_weights) / sum(all_weights_A ∪ all_weights_B)`. When a matched pair’s two sides carry different TF-IDF weights (e.g., `weight_A=1.5`, `weight_B=5.7`), the **per-match weight is `min(weight_A, weight_B)`** (conservative: credit the match at the lower informativeness). The denominator is `sum(all_weights_A) + sum(all_weights_B) - sum(matched_weights)` to avoid double-counting matched items in the union.
- **Subsumption score** (directional, asymmetric): To check "is A ⊂ B?", compute `sum(matched_weights) / sum(all_weights_A)` — i.e., what fraction of A’s weighted content exists in B. To check the reverse ("is B ⊂ A?"), divide by `sum(all_weights_B)`.

> **Zero-weight & Empty-set Guard**: If `sum(all_weights) == 0` (e.g., both reports contain zero measures, or all measures carry zero weight), the similarity and containment scores default to `0.0` rather than raising a division-by-zero exception. If all measures carry zero weight but are identical, the engine falls back to standard unweighted Jaccard/containment (each item treated as weight `1.0`).

> **Design note**: Jaccard is used for the similarity score ("how alike are these?"); the directional subsumption formula is used only in the subsumption check ("is A a subset of B?"). This distinction applies to all set-based comparisons: measures, visuals, filters, and data sources. The `min()` convention for matched-pair weights is consistent across both formulas.

#### Name Normalization

- Use **embedding-based comparison** (Azure OpenAI embeddings) for measure and column names.
- `Total Revenue` ≈ `Revenue Total` ≈ `Rev_Total` — cosine similarity on name embeddings > 0.85 → treat as name match.
- Cache embeddings for all unique measure/column names (one-time computation).

### 3.4 Visual Layer (Weight: 15%)

**Elements compared**: Visual type, bound fields (columns/measures), aggregation applied.

**Algorithm**:
- **Visual type equivalence classes**:
  - `Trend`: bar chart, column chart, line chart, area chart, combo chart
  - `Distribution`: pie chart, donut chart, treemap, funnel
  - `Detail`: table, matrix
  - `KPI`: card, multi-row card, KPI visual, gauge
  - `Spatial`: map, filled map, shape map, ArcGIS
  - `Relationship`: scatter, decomposition tree
  - Two visuals in the same equivalence class with the same bound fields → **equivalent**.
  - Two visuals in different classes with the same bound fields → **partial match** (70% credit).
- **Bound field comparison**: Jaccard on the set of `{field_name, aggregation}` tuples.
- **Layout is ignored**: Only visual type + bound fields matter, not position on the page.
- **Score**: Average pairwise visual equivalence using optimal bipartite matching (Hungarian algorithm).

### 3.5 Filter & Slicer Layer (Weight: 10%)

**Elements compared**: Report-level filters, page-level filters, visual-level filters, slicers, drill-through configurations, bookmarks.

**Algorithm**:
- **Filter comparison**: Jaccard on filter definitions `{target_column, filter_type, filter_value}`.
- **Slicer comparison**: Jaccard on slicer definitions `{target_column, slicer_type}`.
- **Drill-through comparison**: Jaccard on drill-through page definitions.
- **Bookmarks**: Ignored (presentation-level, not functional).
- **Score**: `0.40 * filter_sim + 0.35 * slicer_sim + 0.25 * drillthrough_sim`.

### 3.6 Governance Layer (Weight: 0% — FLAG ONLY)

**Elements flagged**: RLS configurations, refresh schedules, gateway bindings, permission assignments.

**Not scored** — these are **merge risk flags** surfaced in the output:
- `RLS_CONFLICT`: Reports have different RLS rules → merge requires RLS consolidation.
- `REFRESH_MISMATCH`: Reports have different refresh schedules → post-merge decision needed.
- `GATEWAY_DIFFERENT`: Reports use different gateways → infrastructure consideration.
- `PERMISSION_DIVERGENCE`: Reports have different permission sets → access review needed.

---

## 4. Cross-Cutting Algorithmic Concerns

### 4.1 TF-IDF Weighting for Measures

Apply **Term Frequency–Inverse Document Frequency** to measure names/signatures across the report corpus:

```
IDF(measure) = log(N / df(measure)) + 1
```

Where:
- `N` = total number of reports (921)
- `df(measure)` = number of reports containing this measure (or a semantically equivalent measure)

**Effect**: A generic `Revenue` measure (df=200) gets IDF ≈ 2.5, while a specialized `Adjusted EBITDA by Region Q3` (df=3) gets IDF ≈ 6.7. The specialized measure contributes more to the similarity score. The `+ 1` smoothing constant ensures that even ubiquitous measures that appear in all reports (`df = N`) receive a weight of `1.0` rather than `0.0`, preventing denominator sums of 0.

### 4.2 Embedding-Based Name Normalization

- Model: Azure OpenAI `text-embedding-ada-002` (or successor).
- Encode all unique measure names, column names, and table names as embeddings.
- Pre-compute a **name equivalence matrix** using cosine similarity with threshold 0.85.
- Store as a lookup table: `{(name_A, name_B): similarity_score}`.
- One-time cost: ~10K unique names × $0.0001/1K tokens ≈ negligible.

> **Outage fallback**: In the event of Azure OpenAI unavailability during fingerprinting or normalization, name normalization degrades gracefully to exact string match only (or case-insensitive match after removing whitespace/underscores) rather than failing the run.

### 4.3 Subsumption Detection

Report A is **subsumed** by Report B if:
- **90%+ of A’s measures** exist in B (after semantic matching) — using the **directional containment score**: `matched / |A’s measures|`, not Jaccard.
- **90%+ of A’s visuals** exist in B (after visual equivalence matching) — same directional formula.
- **90%+ of A’s filters/slicers** exist in B.
- A’s data sources are a subset of B’s data sources.

Subsumption is **directional**: A ⊂ B does not imply B ⊂ A. The algorithm checks both directions for every candidate pair.

> **Why directional containment, not Jaccard**: Jaccard penalizes size asymmetry. A 10-measure report fully contained in a 100-measure report yields Jaccard ≈ 0.10 but directional containment = 1.0. The 90% subsumption threshold targets the candidate subset’s coverage in the superset, so the directional formula (§3.3) is the correct metric.

#### Subsumption-Golden Reconciliation Rule

When subsumption is detected within a merge cluster, subsumption facts **constrain** what "decommission" can mean — they don't get silently overridden by the usage-weighted golden score.

```python
def build_subsumption_map(member_ids, pairwise_results):
    """Translate stored subsumption ('A⊂B'/'B⊂A', positional to report_a/report_b)
    into directional lookups usable by determine_recommendation.
    Returns: {(x, y): 'first_subset'} meaning x ⊂ y."""
    smap = {}
    for result in pairwise_results:
        if result.subsumption is None:
            continue
        a, b = result.report_a, result.report_b
        if result.subsumption == 'A⊂B':   # a ⊂ b (a is subset of b)
            smap[(a, b)] = 'first_subset'  # queried as (a, b): a is the subset
            smap[(b, a)] = 'second_subset' # queried as (b, a): b contains a
        elif result.subsumption == 'B⊂A':  # b ⊂ a
            smap[(b, a)] = 'first_subset'
            smap[(a, b)] = 'second_subset'
    return smap

def determine_recommendation(member_id, golden_id, pairwise_results,
                             subsumption_map, classification_map):
    """Tier-aware recommendation (§6.2) that respects subsumption constraints.
    classification_map: {(id_a, id_b): classification} built from pairwise_results."""
    if member_id == golden_id:
        return "keep"

    # Divergence check / review gate override: if the pair has been flagged
    # as requiring manual review (divergence or LLM unavailability), the automated
    # decommission/merge content recommendations MUST be gated. Force review.
    classification = classification_map.get(
        (member_id, golden_id),
        classification_map.get((golden_id, member_id))
    )
    if classification == 'review':
        return "review"

    direction = subsumption_map.get((member_id, golden_id))

    if direction == 'first_subset':
        # member ⊂ golden — golden contains everything. Safe to decommission.
        return "decommission"

    if direction == 'second_subset':
        # golden ⊂ member — golden is smaller. Must migrate content first.
        return "merge_content_into_golden"

    # Non-subsumed: use tiered classification from §6.2
    # NOTE: If member has no direct edge to golden (transitive cluster member),
    # classification_map returns None. Fall through to 'review' — this is correct:
    # transitive members lack the pairwise evidence for automated action.
    if classification == 'exact_clone':
        return "decommission"
    elif classification == 'near_clone':
        return "merge"
    elif classification == 'functional_overlap':
        return "review"
    return "review"  # fallback: unclassified, 'unrelated', OR transitive member with no direct edge
```

**Gap analysis output**: For every subsumption recommendation, produce a gap analysis listing exactly what would be lost:

```python
def compute_gap_analysis(decommissioned_id, kept_id):
    """Compute the content in the decommissioned report that is missing from the kept report.
    Used when recommendation is 'merge_content_into_golden': golden (kept) ⊂ member (decommissioned),
    so we iterate over the decommissioned report's content to find what the kept report lacks.
    Returns 'missing_in_golden' as a flat list aligned with content_migration_tasks schema (§9.1)."""
    decommissioned = get_report(decommissioned_id)
    kept = get_report(kept_id)
    missing = []
    for m in decommissioned.measures:
        if not find_equivalent(m, kept.measures):
            missing.append({'type': 'measure', 'id': m.id, 'name': m.name})
    for v in decommissioned.visuals:
        if not find_equivalent(v, kept.visuals):
            missing.append({'type': 'visual', 'id': v.id, 'name': v.title})
    for f in decommissioned.filters:
        if not find_equivalent(f, kept.filters):
            missing.append({'type': 'filter', 'id': f.id, 'name': f.target_column})
    for dt in decommissioned.drillthrough_pages:
        if not find_equivalent(dt, kept.drillthrough_pages):
            missing.append({'type': 'drillthrough', 'id': dt.id, 'name': dt.page_name})
    return {
        'decommissioned_report': decommissioned_id,
        'kept_report': kept_id,
        'coverage': 1 - len(missing) / max(len(decommissioned.all_content), 1),
        'missing_in_golden': missing  # list of {'type', 'id', 'name'}
    }
```

Example output:
```json
{
  "decommissioned_report": "Report A",
  "kept_report": "Report B",
  "coverage": 0.92,
  "missing_in_golden": [
    {"type": "measure", "id": "m_42", "name": "Custom KPI X"},
    {"type": "measure", "id": "m_77", "name": "Adjusted Metric Y"},
    {"type": "visual", "id": "v_12", "name": "Drill-through page 3"}
  ]
}
```

> **Schema alignment**: The `type` field uses singular forms (`'measure'`, `'visual'`, `'filter'`, `'drillthrough'`) matching the `content_migration_tasks.content_type` CHECK constraint in §9.1. The `missing_in_golden` key is used consistently in both §4.3 and §12.

---

## 5. Blocking & Coarse Filtering (Stages 0-1)

### 5.1 Stage 0: Logical Source Blocking

1. Extract all data source connections from the DB.
2. For each source, compute a **schema fingerprint**: `hash(sorted(normalized_column_names) + sorted(data_types))`.
3. Group sources by schema fingerprint similarity:
   - **PostgreSQL (prod)**: Use `pg_trgm` `similarity()` function on concatenated normalized column names. Two sources are in the same block if column-name fuzzy match rate > 80% (consistent with §3.1 and §12), or exact hash match. (Note: the individual column-name `pg_trgm` threshold of 0.6 in §3.1 SQL is a pre-filter for pairwise column name candidates, not the source-grouping threshold.)
   - **SQLite (dev)**: Use Python-side Levenshtein + embedding comparison.
4. Reports are only compared if they share at least one logical source group.
5. Reports with zero source overlap are classified as **unrelated** without further comparison.

> **Acknowledged false-negative risk**: Stage 0 blocking is a hard cut. If two reports use the same underlying data but through sources with different column naming conventions (e.g., one uses `customer_id` and the other uses `cust_num`), and the fuzzy matching threshold (80%) is not met, the pair will be permanently excluded. Unlike Stage 1 (which uses a union of two signals as a recall safety net), Stage 0 has no secondary backstop. **Mitigation**: (1) the embedding-based fuzzy matcher catches synonym-level renames that Levenshtein misses; (2) during calibration (§8), manually inspect a sample of cross-block pairs to estimate the false-negative rate; (3) if the rate is unacceptable, add a secondary cross-block signal (e.g., SimHash on DAX expressions across blocks) as a low-cost backstop.

### 5.2 Stage 1: MinHash + SimHash Coarse Filtering

Within each logical source block:

#### MinHash (for set similarity on measures/columns)

- **Configuration**: 256 permutation functions (`num_perm=256`).
- **Input**: Set of normalized measure signatures per report.
- **LSH banding**: Pin `(b, r)` explicitly using `datasketch`'s low-level constructor, rather than passing `threshold=` which silently derives different `(b, r)` values:

```python
from datasketch import MinHashLSH

NUM_PERM = 256
BANDS = 32
ROWS = 8   # BANDS * ROWS must equal NUM_PERM → 32*8 = 256 ✓

# The S-curve probability of two sets with true Jaccard J being flagged as candidates:
#   P(candidate | J) = 1 - (1 - J^ROWS)^BANDS
# Inflection point (P=0.5) occurs at J ≈ (1/BANDS)^(1/ROWS) ≈ 0.65 for b=32, r=8

- **LSH banding**: Pin `(b, r)` explicitly using `datasketch`'s low-level constructor, rather than passing `threshold=` which silently derives different `(b, r)` values. Since `NUM_PERM = 256` ($2^8$), any divisor split `b * r = 256` must consist of powers of two (e.g., (32, 8), (64, 4), (128, 2), etc.). This severely limits inflection-point tuning (only 9 coarse divisor pairs exist). If finer tuning is required after calibration, `NUM_PERM` should be changed to a number with more divisors (e.g., `NUM_PERM = 240`, which has 20 divisors including 3, 5, 6, 8, 10, 12, 15, 16, 20, 24, etc.).
- **Actual Jaccard inflection point**: **≈ 0.65** (rule-of-thumb approximation via `(1/b)^(1/r)`; the exact P=0.5 crossing from `1-(1-J^8)^32=0.5` is J≈0.62). Since recall > precision is the stated goal (union of MinHash ∪ SimHash, §5.2), a slightly higher inflection is acceptable — SimHash compensates for near-matches that MinHash misses at lower Jaccard values.
- **Alternative**: If a lower inflection (~0.5) is desired after calibration, recompute `(b, r)` using:

```python
def bands_rows_for_threshold(target_j, num_perm=256):
    """Grid-search (b,r) pairs where b*r=num_perm, pick the one whose
    inflection point is closest to target_j (absolute distance, not
    biased to one side). Tie-break: prefer more bands (higher recall)."""
    best = None
    for r in range(1, num_perm + 1):
        if num_perm % r != 0:
            continue
        b = num_perm // r
        inflection = (1 / b) ** (1 / r)
        score = abs(target_j - inflection)  # absolute closest, not only below
        if best is None or score < best[0] or (score == best[0] and b > best[1]):
            best = (score, b, r, inflection)
    return best[1], best[2], best[3]

# Run once during setup, hardcode the result:
b, r, actual_inflection = bands_rows_for_threshold(0.5)
```

#### SimHash (for text similarity on DAX expressions)

- **Configuration**: 256-bit SimHash per report (concatenation of all DAX expressions).
- **Hamming distance threshold**: < 30 bits → candidate pair (30/256 ≈ 88% bit agreement).
- **Purpose**: Catches near-clone reports where DAX was copy-pasted with minor modifications. Complements MinHash because SimHash captures textual similarity while MinHash captures set overlap — a report with reordered but identical measures would score high on MinHash but might miss SimHash, while a copy-pasted report with one renamed measure would score high on SimHash but might miss MinHash.

#### Combined Filtering

A pair survives to Stage 2 if **either** MinHash OR SimHash flags it as a candidate (union, not intersection — to maximize recall since correctness > speed).

---

## 6. Classification & Clustering (Stage 3)

### 6.1 Composite Score Computation

Internal composite for classification (not displayed as primary output):

```
composite = (0.25 × data_source_score) 
          + (0.20 × semantic_model_score) 
          + (0.30 × dax_score) 
          + (0.15 × visual_score) 
          + (0.10 × filter_score)
```

### 6.2 Tiered Classification

| Tier | Composite Score | Label | Action |
|---|---|---|---|
| 1 | > 95% | **Exact Clone** | Decommission (keep one) |
| 2 | > 80% and ≤ 95% | **Near-Clone** | Merge candidate |
| 3 | > 60% and ≤ 80% | **Functional Overlap** | Review for potential merge |
| 4 | ≤ 60% | **Unrelated** | Keep (no action) |

Additional classification:
- **Subsumed** (any tier): If subsumption detected → recommend decommission of the smaller report.
- **Review**: Applied when layer scores are **divergent** (e.g., DAX = 95% but visuals = 30%) — the composite is misleading and human review is needed. Threshold: standard deviation of per-layer scores > 25%.

### 6.3 Graph-Based Clustering

Build a **weighted similarity graph**:
- **Nodes**: Reports
- **Edges**: Pairwise similarity (only for pairs with composite > 60%)
- **Edge weight**: Composite score

**Transitivity & Clustering Scope**: The similarity graph is built using only measured edges with composite > 60%. **No phantom edges are created** — if A-B = 0.9 and B-C = 0.9 but A-C was measured at 0.50 (or not measured), no synthetic A-C edge is added. While a connected component represents the maximum theoretical boundary of related reports, the actual cluster boundaries are determined by the selected clustering algorithm. Louvain, hierarchical, and DBSCAN evaluate graph topology, density, and path strengths natively to partition reports into distinct clusters without relying on manufactured pairwise scores or forcing all path-connected nodes into the same cluster.

**Clustering algorithms** (user-selectable, all three available):

1. **Louvain/Leiden Community Detection** (default):
   - Resolution parameter tunable.
   - Best for discovering natural groupings without specifying K.
   - Recommended for initial analysis.
   - **Input**: Sparse graph `G` (edges > 60% only). Louvain operates natively on sparse graphs.

2. **Hierarchical Agglomerative Clustering**:
   - Produces a dendrogram showing merge hierarchy.
   - Useful for stakeholders to explore "what if we set the threshold at X%?"
   - **Input**: Dense distance matrix built from **all stored `pairwise_scores` rows** (composite > 40%), not just the >60% graph `G`. `distance = 1 - composite_score`. Pairs with no stored pairwise result (below 40% or never compared) default to distance `1.0`. This preserves the 40–60% similarity signal that the dendrogram exploration depends on.
   - Uses average-linkage via `sklearn.cluster.AgglomerativeClustering`.

3. **DBSCAN** (Density-Based):
   - `eps` derived from similarity distribution.
   - **Input**: Same dense distance matrix as hierarchical (from all `pairwise_scores` > 40%), using `distance = 1 - composite_score`. Unmeasured pairs default to distance `1.0`.
   - Naturally identifies **outlier reports** (unique, no merge candidates → auto-classified as "Keep").
   - Useful for identifying truly unique reports.

### 6.4 Golden Report Selection

For each merge cluster, rank reports by:

All inputs are **normalized to [0, 1]** within the cluster before weighting:

```
completeness_norm = (measures + visuals) / max(measures + visuals in cluster, 1)
usage_norm        = log1p(views) / max(log1p(views) in cluster, 1)
freshness_norm    = 1 - (days_since_refresh / max(max_days_since_refresh in cluster, 1))
governance_norm   = count(has_rls, has_perms, ...) / max(total_governance_checks, 1)
recency_norm      = 1 - (days_since_modified / max(max_days_since_modified in cluster, 1))

golden_score = (0.35 × completeness_norm)
             + (0.25 × usage_norm)
             + (0.20 × freshness_norm)
             + (0.10 × governance_norm)
             + (0.10 × recency_norm)
```

> **Design note**: All sub-scores are cluster-relative (normalized against the max within the cluster, not globally). This prevents absolute counts (e.g., 200 measures vs. 10 measures) from dominating alongside 0-1 weights.

**Subsumption override**: When directional subsumption is detected within the cluster (A ⊂ B), the golden selection is constrained:
- If B (superset) has the highest golden_score → B is golden, A is decommissioned. Standard case.
- If A (subset) has the highest golden_score (e.g., A has much higher usage) → the recommendation becomes `MERGE_CONTENT`: migrate B's unique content into A, then decommission B. The golden is A, but the **output explicitly lists content to migrate** from B into A (see §4.3 gap analysis).
- If multiple subsumption relationships exist (A ⊂ B, A ⊂ C), the largest superset is preferred as golden unless golden_score strongly favors a smaller report. This is implemented in `select_golden_report` (§12) using a tie-breaking rule that ranks candidates by the number of cluster members they subsume.

The highest-scoring report is recommended as the **golden report** (keep). Others in the cluster are recommended for decommission or merge into the golden.

### 6.5 Staleness-Based Prioritization

Reports not refreshed in **90+ days** with high similarity to an actively refreshed report receive a **decommission priority boost**:

```
if days_since_refresh > 90 AND max_similarity_to_active_report > 70%
        AND recommendation not in ('keep', 'merge_content_into_golden', 'review'):
    # Don't escalate 'review' — those have unreliable composite scores (divergence check, §6.2).
    # Don't escalate 'keep' or 'merge_content_into_golden' — those have explicit roles.
    recommendation = "strong_decommission"   # lowercase — must match §9.1 CHECK constraint
```

### 6.6 Usage as Separate Input

Usage analytics (view counts, unique viewers, last accessed date) are factored into:
- **Golden report selection** (§6.4) — more-used reports are preferred as the golden.
- **Decommission confidence** — zero-usage reports with high similarity are strong decommission candidates.
- **NOT** into the similarity score itself — usage doesn't affect how *similar* two reports are, only what to *do* about the similarity.

---

## 7. Explainability Engine

### 7.1 Per-Layer Score Breakdown (Primary Output)

Every compared pair produces:

```json
{
  "report_a": "Sales Dashboard Q3",
  "report_b": "Regional Sales Overview",
  "scores": {
    "data_source": { "score": 0.95, "detail": "Same DB, same tables, different connection type (DirectQuery vs Import)" },
    "semantic_model": { "score": 0.88, "detail": "12/14 columns match, relationship topology identical" },
    "dax_measures": { "score": 0.92, "detail": "18/20 measures equivalent (2 unique to Report B)" },
    "visuals": { "score": 0.75, "detail": "6/8 visuals equivalent (Report A has 2 additional KPI cards)" },
    "filters": { "score": 0.60, "detail": "Different slicer configurations (Region vs. Territory)" }
  },
  "composite": 0.85,
  "classification": "Near-Clone",
  "subsumption": "Report A ⊂ Report B (92% subsumed)",
  "governance_flags": ["RLS_CONFLICT", "REFRESH_MISMATCH"],
  "recommendation": "MERGE → Keep Report B, decommission Report A"
}
```

### 7.2 LLM-Generated Summaries

Azure OpenAI generates natural-language explanations where **human-eye-level judgment** is required:

- **DAX equivalence explanations**: "These measures both calculate Year-to-Date Revenue using SUM over the Sales[Amount] column, filtered by a date table. Report A uses CALCULATE with DATESYTD, while Report B uses TOTALYTD — these are functionally identical."
- **Cluster summaries**: "This cluster of 5 reports all analyze regional sales performance using the same underlying Sales_DW dataset. They vary primarily in filter configurations (different regional scopes) and visual layouts."
- **Merge recommendations**: "Recommend merging Reports A, B, and C into Report B (golden). Report B already contains all measures from A and C. The only additions needed are 2 KPI cards from Report A and the drill-through configuration from Report C."

### 7.3 LLM Usage Policy

- Use LLM **only** where deterministic algorithms cannot provide sufficient accuracy:
  - DAX semantic equivalence edge cases (AST similarity 50-90%, or unverifiable conditional rewrite preconditions).
  - DAX parser coverage gaps (unparseable measures route straight to LLM — see §10.1).
  - Name normalization for ambiguous cases.
  - Natural-language explainability summaries.
- **Cache all LLM responses** keyed by context-aware input hash (refer to §3.3 for context-aware hashing rule).
- **Batch API calls** (up to 5 DAX pairs per request to stay within context window limits).

#### LLM Fallback Behavior (Azure OpenAI Unavailability)

Azure OpenAI sits in the critical path for DAX ambiguous-zone judgments (30%-weighted layer) and all explainability output. If the API is unavailable:

- **Retry policy**: Exponential backoff with 3 retries per call, 5s/15s/45s delays. After 3 failures on a single call, circuit-break for 5 minutes.
- **Per-pair degradation**: Individual LLM-required pairs that exhaust retries are classified as `'review'` with `review_reason = 'llm_unavailable'` (persisted in `pairwise_scores.review_reason`). Do not guess equivalence.
- **Batch degradation**: Process remaining pairs, classifying all subsequent LLM-required pairs as `'review'` with `review_reason = 'llm_unavailable'` (no further LLM attempts until the circuit breaker resets).
  - **Persist all results**. Do not discard already-written rows — the per-pair persistence model (§12) makes true all-or-nothing rollback infeasible without a staging-table mechanism that is out of scope for the initial release.
  - **Surface the degradation**: include `run_status` in the `RationalizationResult` so downstream consumers (UI, reports) can display a warning that LLM-dependent classifications are unreliable for this run.
- **Post-run recovery**: Re-process degraded pairs after the outage resolves: `SELECT * FROM pairwise_scores WHERE review_reason = 'llm_unavailable' AND computed_at >= :run_start`.
- **Explainability summaries**: Skip generation; store a placeholder `'LLM unavailable at compute time — rerun Stage 4 to populate'` in the `explanations` table using an UPSERT (`ON CONFLICT (context_type, context_key) DO UPDATE`) to allow subsequent runs to overwrite it when the API recovers.

#### LLM Cost Estimate (Token-Based, Pluggable)

Cost is computed from actual token counts and a **pluggable pricing function**, not hardcoded per-call rates. Pricing changes over time — a hardcoded number goes stale immediately.

```python
def estimate_llm_cost(edge_case_count, cluster_summary_count, merge_rec_count, pricing: dict):
    """
    pricing = {'input_per_1k': ..., 'output_per_1k': ...}  # pulled from current
    provider pricing page at run time, NOT hardcoded in this spec.

    Token counts below are illustrative starting points — MEASURE AND REPLACE
    with real averages from the calibration sample (§8).
    """
    # DAX equivalence prompt: two expressions + table/column context + instructions
    avg_dax_input_tokens = 800      # MEASURE THIS — depends on actual DAX complexity
    avg_dax_output_tokens = 150     # structured JSON response

    # Cluster summary prompt: N member reports' metadata + measure/visual diffs
    avg_cluster_input_tokens = 2000  # scales with cluster size
    avg_cluster_output_tokens = 300

    # Merge recommendation prompt: cluster members + golden report + pairwise diffs
    avg_merge_input_tokens = 2500   # MEASURE THIS — includes gap analysis context
    avg_merge_output_tokens = 500

    dax_cost = edge_case_count * (
        (avg_dax_input_tokens / 1000) * pricing['input_per_1k'] +
        (avg_dax_output_tokens / 1000) * pricing['output_per_1k']
    )
    cluster_cost = cluster_summary_count * (
        (avg_cluster_input_tokens / 1000) * pricing['input_per_1k'] +
        (avg_cluster_output_tokens / 1000) * pricing['output_per_1k']
    )
    merge_cost = merge_rec_count * (
        (avg_merge_input_tokens / 1000) * pricing['input_per_1k'] +
        (avg_merge_output_tokens / 1000) * pricing['output_per_1k']
    )
    return {
        'dax_edge_case_cost': round(dax_cost, 2),
        'cluster_summary_cost': round(cluster_cost, 2),
        'merge_recommendation_cost': round(merge_cost, 2),
        'total': round(dax_cost + cluster_cost + merge_cost, 2),
        'note': 'Token counts are placeholders — replace with measured averages '
                'from the calibration run before quoting to stakeholders'
    }
```

**Illustrative estimate** (using GPT-4o pricing as of early 2025: $5/M input, $15/M output):

| Call Type | Est. Calls | Avg Input Tok | Avg Output Tok | Subtotal |
|---|---|---|---|---|
| DAX equivalence | ~500 edge cases | ~800 | ~150 | ~$3.15 |
| Cluster summaries | ~200 clusters | ~2,000 | ~300 | ~$2.90 |
| Merge recommendations | ~100 clusters | ~2,500 | ~500 | ~$2.00 |
| **Total** | | | | **~$8-12** |

> **Action**: During calibration (§8), log actual prompt/response token counts for every LLM call and compute the real cost from `estimate_llm_cost()`. Do not carry forward "$8-12" as a stated figure — replace it with the function's output against current pricing.

---

## 8. Validation & Calibration Phase

### 8.1 Calibration Protocol

Before running on the full 921-report estate:

1. **Sample selection**: Select **200+ report pairs** spanning all expected similarity tiers. Minimum viable sample per tier:
   - 50+ known exact clones
   - 50+ known near-clones
   - 50+ known functionally overlapping
   - 50+ known unrelated
   
   > **Statistical note**: With 50 samples per tier, a 95% Clopper-Pearson (exact) binomial CI for 85% precision is [72%, 94%]. At 20 samples, it widens to [62%, 97%] — too imprecise to validate the stated acceptance criteria. If 200+ labeled pairs are not available, explicitly state that acceptance criteria are preliminary heuristics to be tightened when more labeled data is collected.
2. **Human labeling**: Require **2+ independent labelers** per pair. Classify each pair as `exact_clone | near_clone | functional_overlap | unrelated`.
   - Report **inter-rater reliability** via Cohen's Kappa (κ). Minimum acceptable: κ ≥ 0.70 (substantial agreement).
   - **Tie-breaking**: If labelers disagree, a third domain expert adjudicates. Pairs with 3-way disagreement are excluded from threshold calibration (labeled `ambiguous`) but retained for qualitative analysis.
   - If κ < 0.70, the tier definitions (§6.2) are ambiguous — refine the labeling rubric before proceeding.
3. **Algorithm run**: Execute the comparison engine on the sample.
4. **Threshold calibration**: 
   - Compute precision/recall for each tier at different threshold values.
   - Select thresholds that maximize F1 score per tier.
   - Adjust the default thresholds (95/80/60) based on calibration results.
5. **Weight calibration**: If per-layer weights are producing counterintuitive results, adjust using the labeled data.
6. **Acceptance criteria**: Proceed to full run only if:
   - Precision > 85% for "Decommission" recommendations
   - Recall > 80% for "Near-Clone" detection
   - No false "Decommission" recommendations on known-unique reports
   - **If criteria are not met**: Adjust layer weights (§6.1) and/or classification thresholds (§6.2) based on the confusion matrix, then re-run on the same calibration sample. **Iteration cap**: Maximum 3 calibration rounds. If acceptance criteria still fail after 3 rounds, escalate to architecture review (the layer weights, comparison algorithms, or tier definitions may be fundamentally misaligned with the corpus characteristics) before proceeding to full run. Do not proceed without explicit stakeholder sign-off on the residual risk.
   - **Overfitting caveat**: Tuning thresholds on the same sample used for evaluation inflates reported precision/recall. To mitigate: split the 200+ labeled pairs into a **tuning set (70%)** and a **held-out validation set (30%)** before calibration begins. Report final acceptance metrics on the held-out set only. If the held-out set is too small for reliable CIs (≤15 per tier), acknowledge the reported metrics as optimistic upper bounds.
7. **DAX rewrite validation**: Manually verify every conditional-rewrite match in the labeled sample. If any produces a wrong equivalence, that rule's precondition set is insufficient — tighten it before the full run.
8. **DAX parser sub-experiment**: Run the same 50-100 pairs through Option C (LLM-only, no AST) and Option D (partial parser + LLM) to measure accuracy/cost difference before committing to the full parser build (see §10.1).
9. **Runtime profiling**: Log actual per-stage timings and compute `estimate_full_runtime()` (see §11) to produce a defensible runtime estimate for stakeholders.

---

## 9. Data Model — Results Schema

### 9.1 Comparison Results Tables

```sql
-- Track execution metadata and LLM degradation status of each batch run
CREATE TABLE batch_runs (
    run_id              SERIAL PRIMARY KEY,
    started_at          TIMESTAMP DEFAULT NOW(),
    completed_at        TIMESTAMP,
    run_status          TEXT CHECK (run_status IN ('success', 'degraded_llm', 'failed')) DEFAULT 'success',
    llm_attempt_count   INTEGER DEFAULT 0,
    llm_failure_count   INTEGER DEFAULT 0
);

-- Per-layer fingerprints (computed once per report, updated on metadata change)
CREATE TABLE report_fingerprints (
    report_id           TEXT PRIMARY KEY,
    data_source_hash    TEXT,          -- Schema fingerprint
    semantic_model_hash TEXT,          -- Column + relationship hash
    dax_minhash         BYTEA,        -- 256-bit MinHash signature
    dax_simhash         BYTEA,        -- 256-bit SimHash
    visual_hash         TEXT,          -- Visual set fingerprint
    filter_hash         TEXT,          -- Filter/slicer fingerprint
    version             INTEGER DEFAULT 1,  -- incremented on recompute (§13)
    source_event        TEXT CHECK (source_event IN ('initial_batch', 'republished', 'manual_trigger')),
    previous_hash_composite TEXT,      -- hash of all six hashes; cheap "did anything change" check
    computed_at         TIMESTAMP DEFAULT NOW()
);

-- Source fingerprints and logical block assignments (§5.1 blocking state)
-- Persisted so incremental path (§13) can call get_logical_block() without recomputing
CREATE TABLE source_fingerprints (
    source_id           TEXT PRIMARY KEY,
    schema_hash         TEXT NOT NULL,  -- hash(sorted(col_names) + data_types)
    report_id           TEXT NOT NULL,  -- which report this source belongs to
    computed_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE source_blocks (
    source_id           TEXT NOT NULL REFERENCES source_fingerprints(source_id) ON DELETE CASCADE,
    block_id            TEXT NOT NULL,
    PRIMARY KEY (source_id, block_id)
);
CREATE INDEX idx_source_blocks_block ON source_blocks (block_id);

-- Pairwise comparison scores (all pairs with composite > 40%, see §12 storage filter)
CREATE TABLE pairwise_scores (
    report_a_id         TEXT NOT NULL,
    report_b_id         TEXT NOT NULL,
    data_source_score   REAL,
    semantic_model_score REAL,
    dax_score           REAL,
    visual_score        REAL,
    filter_score        REAL,
    composite_score     REAL,
    classification      TEXT CHECK (classification IN 
                          ('exact_clone', 'near_clone', 'functional_overlap', 'unrelated', 'review')),
    subsumption         TEXT CHECK (subsumption IN ('A⊂B', 'B⊂A')), -- positional: A=report_a_id, B=report_b_id
    review_reason       TEXT CHECK (review_reason IN
                          ('divergent_scores', 'llm_unavailable')),
                          -- NULL for non-review classifications; enables targeted re-processing
    -- Governance flags stored ONLY in governance_flags table (normalized).
    -- Query via JOIN on (report_a_id, report_b_id). No duplicate array here.
    computed_at         TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (report_a_id, report_b_id),
    CHECK (report_a_id < report_b_id)  -- enforce canonical ordering; Python sorts pair IDs
);

-- Cluster assignments (composite PK enables grouping reports by cluster)
CREATE TABLE clusters (
    cluster_id          INTEGER NOT NULL,  -- assigned by clustering algorithm, NOT auto-increment
    algorithm           TEXT CHECK (algorithm IN ('louvain', 'hierarchical', 'dbscan')),
    report_id           TEXT NOT NULL,
    is_golden           BOOLEAN DEFAULT FALSE,
    golden_score        REAL,
    recommendation      TEXT CHECK (recommendation IN 
                          ('keep', 'decommission', 'merge', 'review', 'strong_decommission',
                           'merge_content_into_golden')),
    computed_at         TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (cluster_id, algorithm, report_id)
);

-- Content migration tasks (tracks what must move when golden ⊂ decommissioned report)
-- NOTE: Migration-task completion (transitioning status from 'pending' to 'done' or 'skipped') 
-- is driven by an external UI or migration process, which is out of scope for this comparison engine.
CREATE TABLE content_migration_tasks (
    source_report_id    TEXT NOT NULL,   -- report being decommissioned
    target_report_id    TEXT NOT NULL,   -- golden report absorbing content
    content_type        TEXT CHECK (content_type IN ('measure', 'visual', 'filter', 'drillthrough')),
    content_id          TEXT NOT NULL,
    content_name        TEXT,            -- human-readable name for the content item
    status              TEXT CHECK (status IN ('pending', 'in_progress', 'done', 'skipped')) DEFAULT 'pending',
    completed_at        TIMESTAMP,      -- set when status transitions to 'done' or 'skipped'
    PRIMARY KEY (source_report_id, target_report_id, content_type, content_id)
);

-- Triggers decommission recommendation once all migration tasks complete.
-- Implementation: scheduled job or webhook that runs:
--   UPDATE clusters SET recommendation = 'decommission'
--   WHERE report_id = :source_report_id AND recommendation = 'merge_content_into_golden'
--     AND NOT EXISTS (
--       SELECT 1 FROM content_migration_tasks
--       WHERE source_report_id = :source_report_id
--         AND status NOT IN ('done', 'skipped')
--     );

-- LLM-generated explanations (cached)
-- NOTE: DAX measure-level LLM explanations are stored in measure_equivalences.llm_explanation,
-- NOT in this table. This table is for cluster/report-level LLM-generated text only.
CREATE TABLE explanations (
    id                  SERIAL PRIMARY KEY,
    context_type        TEXT CHECK (context_type IN ('cluster_summary', 'merge_recommendation')),
    context_key         TEXT NOT NULL,    -- stable hash of sorted member IDs
    explanation         TEXT NOT NULL,
    model_used          TEXT,
    computed_at         TIMESTAMP DEFAULT NOW(),
    UNIQUE (context_type, context_key)
);

-- Per-layer score breakdowns with details
CREATE TABLE score_details (
    report_a_id         TEXT NOT NULL,
    report_b_id         TEXT NOT NULL,
    layer               TEXT CHECK (layer IN 
                          ('data_source', 'semantic_model', 'dax', 'visuals', 'filters')),
    score               REAL,
    detail_json         JSONB,        -- Structured detail per layer
    PRIMARY KEY (report_a_id, report_b_id, layer),
    CHECK (report_a_id < report_b_id)  -- enforce canonical ordering
);

-- Measure-level equivalence (for explainability drill-down)
CREATE TABLE measure_equivalences (
    report_a_id         TEXT NOT NULL,  -- denormalized: owning report of measure_a
    report_b_id         TEXT NOT NULL,  -- denormalized: owning report of measure_b
    measure_a_id        TEXT NOT NULL,
    measure_b_id        TEXT NOT NULL,
    match_method        TEXT CHECK (match_method IN 
                          ('exact', 'signature', 'ast', 'llm')),
    similarity_score    REAL,
    llm_explanation     TEXT,         -- NULL if not LLM-matched
    PRIMARY KEY (report_a_id, report_b_id, measure_a_id, measure_b_id),
    CHECK (report_a_id < report_b_id)  -- enforce canonical report ordering; maps measure_a to report_a, measure_b to report_b
);

-- Governance flags per report pair
CREATE TABLE governance_flags (
    report_a_id         TEXT NOT NULL,
    report_b_id         TEXT NOT NULL,
    flag_type           TEXT CHECK (flag_type IN 
                          ('RLS_CONFLICT', 'REFRESH_MISMATCH', 'GATEWAY_DIFFERENT', 'PERMISSION_DIVERGENCE')),
    detail              TEXT,
    PRIMARY KEY (report_a_id, report_b_id, flag_type),
    CHECK (report_a_id < report_b_id)  -- enforce canonical ordering
);
```

### 9.2 Indexes for Frontend Query Performance

```sql
-- Fast lookup of all comparisons for a specific report
CREATE INDEX idx_pairwise_a ON pairwise_scores (report_a_id);
CREATE INDEX idx_pairwise_b ON pairwise_scores (report_b_id);
CREATE INDEX idx_pairwise_classification ON pairwise_scores (classification);
CREATE INDEX idx_pairwise_composite ON pairwise_scores (composite_score DESC);
CREATE INDEX idx_measure_equiv_report_a ON measure_equivalences (report_a_id);
CREATE INDEX idx_measure_equiv_report_b ON measure_equivalences (report_b_id);

-- Cluster queries (composite PK covers cluster_id + algorithm + report_id lookups)
CREATE INDEX idx_cluster_report ON clusters (report_id);
CREATE INDEX idx_cluster_algo ON clusters (algorithm, cluster_id);
CREATE INDEX idx_cluster_recommendation ON clusters (recommendation);
CREATE INDEX idx_cluster_golden ON clusters (algorithm, is_golden) WHERE is_golden = TRUE;

-- Score detail queries
CREATE INDEX idx_score_detail_layer ON score_details (layer);
```

---

## 10. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Database (Dev) | SQLite | Local development and testing |
| Database (Prod) | PostgreSQL + `pg_trgm` extension | Production; `pg_trgm` for trigram-based fuzzy column-name matching in §3.1/§5.1 |
| Comparison Engine | Python | Scoring, fingerprinting, orchestration |
| Blocking & Data Retrieval | SQL | Source blocking (using `pg_trgm` similarity in prod), fingerprint lookup, result storage |
| MinHash/LSH | `datasketch` library | Coarse filtering; explicit `params=(b,r)` constructor (not `threshold=`), see §5.2 |
| DAX AST Parsing | See §10.1 (Risk Assessment) | Abstract Syntax Tree comparison — **highest-risk dependency** |
| Embeddings | Azure OpenAI `text-embedding-ada-002` | Name normalization, semantic matching |
| LLM | Azure OpenAI GPT-4o | DAX semantic judgment, explainability summaries |
| Clustering | `scikit-learn`, `networkx`, `cdlib` | Louvain/Leiden, hierarchical, DBSCAN |
| Graph Operations | `networkx` | Similarity graph construction, traversal |
| Bipartite Matching | `scipy.optimize.linear_sum_assignment` | Hungarian algorithm for measure/visual matching |
| Frontend Data Layer | PostgreSQL → API → Frontend | Results served via DB queries |

### 10.1 DAX Parser — Risk Assessment & Phased Approach

> **This is the single largest hidden cost in the build.** A parser that correctly models CALCULATE context transitions, VAR scoping, nested EVALUATE blocks, and the full 200+ function surface area is a substantial engineering effort. It must be treated as its own workstream with a phased approach, not a dependency line in the tech stack table.

#### Phase 1: Coverage Audit (Before Committing to a Parsing Strategy)

Run against the real 921-report corpus **before designing the parser**. This determines whether a full DAX grammar is even necessary.

```python
def audit_dax_complexity(all_measures: list) -> dict:
    """Run this ONCE on the real corpus to determine parser scope."""
    return {
        'total_measures': len(all_measures),
        'uses_variables': count_matching(all_measures, r'\bVAR\b'),
        'uses_calculate': count_matching(all_measures, r'\bCALCULATE\s*\('),
        'nesting_depth_distribution': compute_nesting_depths(all_measures),
        'unique_function_calls': extract_unique_functions(all_measures),
        'measures_referencing_measures': count_measure_to_measure_deps(all_measures),
    }
```

#### Phase 2: Build vs. Buy Decision (Driven by Audit Results)

**Option A: Full custom DAX parser (Estimated effort: 4-8 weeks)**
- Write a PEG/ANTLR grammar (e.g., `lark` or `ANTLR`) for DAX.
- Must handle: CALCULATE context transitions, VAR/RETURN scoping, table constructors, row context vs. filter context disambiguation.
- Risk: DAX has undocumented parsing edge cases (measure references that look like column references, disambiguation rules). Microsoft does not publish a formal grammar.
- Benefit: Full control, deterministic, no external dependency.

**Option B: Existing open-source parsers**
- [`pbi-tools`](https://pbi.tools/) extracts model metadata but does not parse DAX into ASTs.
- [`dax-parser`](https://github.com/nicktrog/dax-parser) (JavaScript) — partial coverage, not production-grade.
- [`Tabular Editor`](https://tabulareditor.com/) has an internal parser but is C#/.NET and not library-accessible.
- **No mature, production-quality, library-accessible DAX parser exists as of this writing.**

**Option C: LLM-assisted, no parser (Recommended for initial build)**
- Skip full AST parsing. Instead:
  1. Use signature-based matching (Stage A) as the primary coarse filter.
  2. For signature-matched pairs, use **text normalization** (whitespace, casing, keyword normalization) + Levenshtein/edit distance on normalized DAX text.
  3. Escalate ambiguous pairs to LLM (Stage C) which implicitly performs semantic AST-level comparison.
- Tradeoff: Less precise than true AST comparison, but avoids the parser engineering cost entirely. LLM budget increases by ~200-300 additional calls (~$3-5).
- **This is the recommended approach for the initial build.**

**Option D: Partial parser + LLM (Recommended for production)**
- Build a **partial grammar** covering the top N DAX function patterns identified by the Phase 1 audit (likely 80%+ of real usage per typical enterprise DAX corpora).
- Anything outside the partial grammar's coverage **skips Stage B entirely** and routes straight to Stage C (LLM):

```python
def compare_dax_measure_pair(measure_a: Measure, measure_b: Measure) -> dict:
    """Compare two individual DAX measures.
    Implements the 3-stage pipeline (§3.3)."""
    # Stage A: Signature Matching (Fast Coarse Filter)
    if not signatures_match(measure_a.signature, measure_b.signature):
        return {
            'equivalent': False,
            'score': 0.0,
            'method': 'signature',
            'explanation': 'Signature mismatch: input columns, aggregation, or types differ.',
            'llm_degraded': False,
            'llm_attempts': 0,
            'llm_failures': 0
        }
        
    # Exact Text Matching Short-circuit
    if measure_a.dax_text == measure_b.dax_text:
        return {
            'equivalent': True,
            'score': 1.0,
            'method': 'exact',
            'explanation': 'Exact text match.',
            'llm_degraded': False,
            'llm_attempts': 0,
            'llm_failures': 0
        }

    # Stage B: AST-Level Structural Comparison (Medium, Structural)
    if is_parseable(measure_a.dax_text) and is_parseable(measure_b.dax_text):
        ast_a = parse_dax(measure_a.dax_text)
        ast_b = parse_dax(measure_b.dax_text)
        ast_match = classify_ast_match(measure_a.dax_text, measure_b.dax_text, ast_a, ast_b)
        if ast_match['route'] == 'standard':
            return {
                'equivalent': ast_match['score'] > 0.90,
                'score': ast_match['score'],
                'method': 'ast',
                'explanation': 'Determined via AST tree edit distance.',
                'llm_degraded': False,
                'llm_attempts': 0,
                'llm_failures': 0
            }
            
    # Stage C: LLM Semantic Judgment (Edge Cases, Ambiguous Zone)
    try:
        result = llm_semantic_judgment(measure_a.dax_text, measure_b.dax_text)
        return {
            'equivalent': result['equivalent'],
            'score': 1.0 if result['equivalent'] else 0.0,
            'method': 'llm',
            'explanation': result['explanation'],
            'llm_degraded': False,
            'llm_attempts': 1,
            'llm_failures': 0
        }
    except LLMUnavailabilityError:
        return {
            'equivalent': False,
            'score': 0.0,
            'method': 'llm',
            'explanation': 'LLM unavailable during semantic judgment.',
            'llm_degraded': True,
            'llm_attempts': 1,
            'llm_failures': 1
        }
```

- Estimated effort: 2-3 weeks for the partial parser, using `lark` or `ANTLR` (not regex).

#### Phase 3: Effort & Project Timeline Workstreams

Building the complete rationalization engine involves both one-time engineering build work and recurring operational calibration work.

##### One-Time Engineering Workstreams

| Workstream / Phase | Effort | Dependency |
|---|---|---|
| Phase 1: Coverage audit | 2-3 days | Access to corpus in DB |
| Phase 2: Decision | 1 day | Audit results |
| Option C build (initial parser bypass) | 0 (uses text normalization + LLM) | — |
| Option D build (production partial parser) | 2-3 weeks | Audit results determining grammar scope |
| Option A build (full custom DAX parser) | 4-8 weeks | Only if Option D proves insufficient in calibration |
| Core Scoring & Bipartite-Matching Engine | 1-2 weeks | Database schema completion |
| Clustering & Golden Selection Integration | 1 week | Graph modeling decisions |
| Explainability & Persistence Layer | 1 week | Core engine completion |
| Frontend API / UI Dashboard Integration | 2 weeks | Database schema and query performance indexing |

##### Recurring Operational Calibration Costs

| Operational Item | Human Effort | Frequency | Notes |
|---|---|---|---|
| **Human Labeling Protocol** | 2 labelers × 200 pairs × 10 min/pair = **~67 human-hours** | Every major model / threshold calibration cycle | Require independent scoring, Cohen's Kappa checks (§8.1) |
| **Expert Adjudication** | ~5-15 hours (depends on raters' agreement) | Every major calibration cycle | Resolve raters' disagreements |
| **Precondition Rule Verification** | 10-20 hours | Every major DAX rewrite rule addition | Manual verification of AST equivalences |

**Decision point**: The calibration phase (§8) should run the same 50-100 pairs through Option C and Option D to measure accuracy/cost difference before committing to the full parser build.

---

## 11. Runtime & Bottleneck Analysis

> **Note**: Raw arithmetic operation counts are misleading for runtime estimation — a few million ops complete in milliseconds on modern hardware. The actual bottlenecks are I/O-bound (LLM API calls, DB queries) and DAX parsing complexity. **Do not quote any runtime figure to stakeholders until the calibration phase (§8) produces measured values.** The estimates below are illustrative only.

### 11.1 Profiling-Based Cost Model

Runtime is estimated from **four measured quantities**, populated from the calibration run (§8) rather than guessed up front:

```
runtime_estimate = t_fingerprint + t_deep_compare + t_llm + t_cluster

Where each term is measured empirically on the 50-100 pair calibration sample
and then linearly extrapolated (with a stated confidence interval):

  t_fingerprint  = N_reports × avg(minhash_compute_time + simhash_compute_time)
                   [genuinely sub-second per report — this term is negligible]

  t_deep_compare = P_candidate_pairs × avg(per_layer_score_time)
                   [dominated by DAX AST parsing, NOT tree-edit-distance —
                    profile parse_dax() on the corpus's most complex measures
                    (nested CALCULATE/VAR), not toy expressions]

  t_llm          = E_edge_cases × (avg_latency_per_batched_call / batch_size)
                   [measure real API latency during calibration; batching reduces
                    call count but not per-item latency much below ~1-3s at
                    batch=5 due to token generation time, not network RTT]

  t_cluster      = O(N log N) term — negligible, confirmed by complexity math
```

**Runtime estimation function** (populated after calibration):

```python
def estimate_full_runtime(calibration_metrics: dict, n_full=921):
    """
    calibration_metrics comes from an actual timed run on the calibration sample:
    {
        'avg_fingerprint_time_per_report': float,   # seconds
        'avg_deep_compare_time_per_pair': float,    # seconds, includes AST parse
        'avg_llm_time_per_edge_case': float,        # seconds, post-batching
        'sample_candidate_pair_rate': float,        # candidate_pairs / total_pairs
        'sample_edge_case_rate': float,              # edge_cases / candidate_pairs
    }
    """
    total_pairs = n_full * (n_full - 1) / 2
    p_full = total_pairs * calibration_metrics['sample_candidate_pair_rate']
    e_full = p_full * calibration_metrics['sample_edge_case_rate']

    t_fp = n_full * calibration_metrics['avg_fingerprint_time_per_report']
    t_dc = p_full * calibration_metrics['avg_deep_compare_time_per_pair']
    t_llm = e_full * calibration_metrics['avg_llm_time_per_edge_case']

    total_seconds = t_fp + t_dc + t_llm
    return {
        'estimated_hours': total_seconds / 3600,
        'breakdown': {'fingerprinting': t_fp, 'deep_compare': t_dc, 'llm': t_llm},
        'confidence': 'derived from calibration sample — re-validate if '
                      'P_full deviates >2x from sample rate extrapolation'
    }
```

### 11.2 Illustrative Bottleneck Breakdown (Pre-Calibration Estimates)

| Stage | Bottleneck | Estimated Time | Basis |
|---|---|---|---|
| Stage 0: Blocking | DB queries + `pg_trgm` fuzzy matching | 1-5 minutes | ~5000 sources, GIN-indexed |
| Stage 1: Fingerprinting | CPU-bound, parallelizable | 2-10 minutes | 1000 reports × 256 perms |
| Stage 2: Non-DAX layers | CPU + DB reads | 10-30 minutes | ~5K pairs × 4 layers |
| **Stage 2: DAX comparison** | **Primary bottleneck** | **30-120 minutes** | Depends on parser choice (§10.1) |
| **Stage 2: LLM API calls** | **Network I/O** | **15-60 minutes** | ~500-800 calls, rate-limited |
| Stage 3-4: Clustering + Explanations | Mixed | 10-30 minutes | Clustering trivial; LLM explanations dominate |
| **Total illustrative** | | **1.5-4 hours** | **Replace with measured values from calibration** |

> **Arithmetic note**: The summed low-end per-stage estimates (1+2+10+30+15+10 = 68 min) are below the stated aggregate low bound (1.5 hours = 90 min). The difference accounts for inter-stage overhead (DB I/O, graph construction, result serialization) not itemized in individual rows. These numbers are pre-calibration illustrations — replace with measured values.

### 11.3 Memory Requirements

| Data Structure | Size Estimate | Notes |
|---|---|---|
| MinHash signatures (256 perm × 1000 reports) | ~2 MB | 256 × 8 bytes × 1000 |
| SimHash signatures (256-bit × 1000 reports) | ~32 KB | Negligible |
| Pairwise score matrix (5K pairs × 5 layers) | ~200 KB | Sparse; only candidate pairs stored |
| Name embedding cache (10K × 1536-dim) | ~60 MB | `text-embedding-ada-002` produces 1536-dim vectors |
| LLM response cache | ~5-10 MB | ~800 cached responses × avg 5KB each |
| Similarity graph (NetworkX) | ~5 MB | 1000 nodes, 5K edges with metadata |
| Dense distance matrix (hierarchical/DBSCAN) | ~7 MB | 921×921 float64; built from all `pairwise_scores` > 40% |
| **Total peak memory** | **~80-90 MB** | Fits comfortably in a single process |

### 11.4 Parallelization Opportunities

- **Stage 1**: MinHash/SimHash computation is embarrassingly parallel across reports.
- **Stage 2 (non-LLM)**: Per-layer scoring for different candidate pairs is independent — use `multiprocessing.Pool` or `concurrent.futures`.
- **Stage 2 (LLM)**: Concurrent API calls (5-10 parallel requests, subject to Azure OpenAI rate limits).
- **Stage 4**: LLM explanation generation is independent per cluster — fully parallelizable.

> **Required calibration deliverable**: After running on 50-100 pairs, produce actual timings per stage, feed them to `estimate_full_runtime()`, and present the extrapolation (with confidence interval) before committing to a timeline with stakeholders.

---

## 12. Algorithm Pseudocode

```python
def select_golden_report(member_ids: List[str], weights: dict, pairwise_results: List[PairwiseResult]) -> str:
    """
    Selects the golden report for a cluster.
    If multiple reports have subsumption relationships, prefers the largest superset
    (the one that subsumes the most other members in the cluster) to minimize content migration.
    If no clear superset exists, ranks purely by weighted golden score (completeness, usage, freshness).
    """
    scores = {}
    subsumes_count = {mid: 0 for mid in member_ids}
    
    # 1. Count how many other members each report subsumes
    for r in pairwise_results:
        if r.report_a in member_ids and r.report_b in member_ids:
            if r.subsumption == 'A⊂B':  # B contains A, so B subsumes A
                subsumes_count[r.report_b] += 1
            elif r.subsumption == 'B⊂A': # A contains B, so A subsumes B
                subsumes_count[r.report_a] += 1
                
    # 2. Compute normal golden scores
    for mid in member_ids:
        scores[mid] = compute_weighted_golden_score(mid, weights)
        
    # 3. Sort primarily by number of subsumed members (descending),
    # and secondarily by weighted golden score (descending) as a tie-breaker.
    sorted_members = sorted(
        member_ids,
        key=lambda m: (subsumes_count[m], scores[m]),
        reverse=True
    )
    return sorted_members[0]

@dataclass
class RationalizationResult:
    pairwise_results: List[PairwiseResult]
    clusters: dict
    total_reports: int
    decommission_candidates: int
    merge_clusters: int
    unique_reports: int
    run_status: str  # 'success' or 'degraded_llm'

def rationalize(reports: List[Report], db: Database) -> RationalizationResult:
    # Initialize batch run in database for execution metadata/degradation status
    run_id = db.create_batch_run()
    llm_attempts = 0
    llm_failures = 0
    llm_degraded = False
    
    # ═══════════════════════════════════════════
    # STAGE 0: BLOCKING
    # ═══════════════════════════════════════════
    
    # Compute schema fingerprints for all data sources
    source_fingerprints = {}
    for report in reports:
        for source in report.data_sources:
            fp = schema_fingerprint(
                columns=normalize_column_names(source.columns),
                data_types=source.data_types
            )
            source_fingerprints[source.id] = fp
    
    # Group sources into logical source blocks (fuzzy matching)
    logical_blocks = cluster_sources_by_fingerprint(
        source_fingerprints, 
        fuzzy_threshold=0.80,
        method="levenshtein+embedding"
    )
    
    # Persist source fingerprints and block assignments for incremental path (§13)
    for source_id, fp in source_fingerprints.items():
        db.store_source_fingerprint(source_id, fp)
    for block_id, source_ids in logical_blocks.items():
        for source_id in source_ids:
            db.store_source_block_assignment(source_id, block_id)
    
    # Map reports to blocks
    report_blocks = {}  # block_id → [report_ids]
    for block_id, source_ids in logical_blocks.items():
        report_blocks[block_id] = [
            r.id for r in reports 
            if any(s.id in source_ids for s in r.data_sources)
        ]
    
    # ═══════════════════════════════════════════
    # STAGE 1: COARSE FILTERING
    # ═══════════════════════════════════════════
    
    # Compute per-layer fingerprints (all 6 columns from report_fingerprints table)
    for report in reports:
        report.fingerprints = {
            'data_source_hash': schema_fingerprint(
                columns=normalize_column_names(
                    [c for s in report.data_sources for c in s.columns]
                ),
                data_types=[dt for s in report.data_sources for dt in s.data_types]
            ),
            'semantic_model_hash': hash(sorted(
                report.column_set | report.relationship_set
            )),
            'dax_minhash': compute_minhash(
                report.measure_signatures, num_perm=256
            ),
            'dax_simhash': compute_simhash(
                report.dax_text_concatenated, bits=256
            ),
            'visual_hash': hash(sorted(report.visual_signatures)),
            'filter_hash': hash(sorted(report.filter_definitions))
        }
        db.store_fingerprint(report.id, report.fingerprints)
    
    # LSH candidate generation within each block
    candidate_pairs = set()
    for block_id, report_ids in report_blocks.items():
        block_reports = [r for r in reports if r.id in report_ids]
        
        # MinHash LSH — pin (b, r) explicitly via params= constructor
        # Inflection point ≈ 0.65 (rule-of-thumb; exact P=0.5 crossing ≈ 0.62, see §5.2)
        lsh = MinHashLSH(num_perm=256, params=(32, 8))
        for r in block_reports:
            lsh.insert(r.id, r.fingerprints['dax_minhash'])
        
        for r in block_reports:
            candidates = lsh.query(r.fingerprints['dax_minhash'])
            for c in candidates:
                if c != r.id:
                    candidate_pairs.add(tuple(sorted([r.id, c])))
        
        # SimHash filtering
        for i, r1 in enumerate(block_reports):
            for r2 in block_reports[i+1:]:
                hamming = hamming_distance(
                    r1.fingerprints['dax_simhash'],
                    r2.fingerprints['dax_simhash']
                )
                if hamming < 30:
                    candidate_pairs.add(tuple(sorted([r1.id, r2.id])))
    
    # ═══════════════════════════════════════════
    # STAGE 2: DEEP COMPARISON
    # ═══════════════════════════════════════════
    
    pairwise_results = []
    
    for (id_a, id_b) in candidate_pairs:
        report_a = get_report(id_a)
        report_b = get_report(id_b)
        
        # Per-layer scoring
        # DAX layer returns a rich result with score + measure-level matches
        dax_result = compare_dax_layer(report_a, report_b)
        scores = {
            'data_source': compare_data_sources(report_a, report_b),
            'semantic_model': compare_semantic_models(report_a, report_b),
            'dax': dax_result['score'],                                # report-level DAX score
            'visuals': compare_visuals(report_a, report_b),
            'filters': compare_filters(report_a, report_b)
        }
        # compare_dax_layer() internally runs the 3-stage pipeline
        # (§3.3 Stages A→B→C) per measure pair via compare_dax_measure_pair(),
        # then applies Hungarian bipartite matching (§3.3 Measure Set Comparison)
        # with TF-IDF weighting to produce a single report-level DAX score.
        # Track LLM metrics if any constituent comparison was routed to LLM
        # compare_dax_layer contract returns 'llm_attempts' and 'llm_failures' for the pair
        llm_attempts += dax_result.get('llm_attempts', 0)
        llm_failures += dax_result.get('llm_failures', 0)
        if llm_attempts >= 20 and (llm_failures / llm_attempts) > 0.20:
            llm_degraded = True
            
        # Persist individual measure-pair results to measure_equivalences table:
        for match in dax_result['measure_matches']:
            db.store_measure_equivalence(
                report_a_id=id_a,                                    # denormalized for deletion
                report_b_id=id_b,
                measure_a_id=match['a_id'],  # no independent sort; map to id_a/report_a_id
                measure_b_id=match['b_id'],  # map to id_b/report_b_id
                match_method=match['method'],     # 'exact', 'signature', 'ast', or 'llm'
                similarity_score=match['score'],
                llm_explanation=match.get('explanation')
            )
        
        # Composite score
        composite = (
            0.25 * scores['data_source'] +
            0.20 * scores['semantic_model'] +
            0.30 * scores['dax'] +
            0.15 * scores['visuals'] +
            0.10 * scores['filters']
        )
        
        # Classification
        review_reason = None  # populated only when classification = 'review'
        if composite > 0.95:
            classification = 'exact_clone'
        elif composite > 0.80:
            classification = 'near_clone'
        elif composite > 0.60:
            classification = 'functional_overlap'
        else:
            classification = 'unrelated'
        
        # Divergence check → force review
        layer_scores = list(scores.values())
        if np.std(layer_scores) > 0.25:
            classification = 'review'
            review_reason = 'divergent_scores'
        
        # LLM failure propagation: if compare_dax_layer degraded any measure
        # pairs due to LLM unavailability, override to review
        # Note: If both conditions apply, 'divergent_scores' takes precedence as it is the more actionable signal.
        if dax_result.get('llm_degraded', False) or llm_degraded:
            classification = 'review'
            if review_reason != 'divergent_scores':
                review_reason = 'llm_unavailable'
        
        # Subsumption check
        subsumption = check_subsumption(report_a, report_b, threshold=0.90)
        
        result = PairwiseResult(
            report_a=id_a, report_b=id_b,
            scores=scores, composite=composite,
            classification=classification,
            subsumption=subsumption,
            review_reason=review_reason  # NULL for non-review classifications
        )
        pairwise_results.append(result)
        
        # Store pairwise result and child records only if composite > 40%
        # (matches §9.1 schema comment; prevents orphaned score_details and governance_flags)
        if composite > 0.40:
            db.store_pairwise_result(result)
            
            # Governance flags — stored in normalized governance_flags table, not in PairwiseResult
            # check_governance returns list of {flag_type, detail} dicts
            gov_flags = check_governance(report_a, report_b)
            for flag in gov_flags:
                db.store_governance_flag(id_a, id_b, flag['flag_type'], flag.get('detail'))
            
            # Store per-layer score details for explainability drill-down (§7.1, §9.1)
            for layer_name, layer_score in scores.items():
                db.store_score_detail(
                    report_a_id=id_a, report_b_id=id_b,
                    layer=layer_name, score=layer_score,
                    detail_json=get_layer_detail(report_a, report_b, layer_name)
                )
    
    # ═══════════════════════════════════════════
    # STAGE 3: CLUSTERING
    # ═══════════════════════════════════════════
    
    # Build similarity graph — edges from MEASURED pairwise scores only,
    # no phantom/synthetic edges from transitivity products (see §6.3)
    G = nx.Graph()
    for r in reports:
        G.add_node(r.id, **r.metadata)
    
    for result in pairwise_results:
        if result.composite > 0.60:
            G.add_edge(result.report_a, result.report_b, 
                      weight=result.composite)
    # Clustering algorithms handle transitivity natively through graph topology.
    # No filter_transitive_edges() call — phantom edges removed per §6.3.
    
    # Run all three clustering algorithms.
    # Louvain uses the sparse graph G (edges > 0.60).
    # Hierarchical and DBSCAN use a dense distance matrix built from ALL stored
    # pairwise_scores (> 0.40 floor), preserving the 40-60% signal for dendrogram
    # exploration. Unmeasured pairs default to distance 1.0.
    all_pairwise = db.get_all_pairwise_scores()  # all rows with composite > 40%
    distance_matrix = build_distance_matrix(
        report_ids=[r.id for r in reports],
        pairwise_scores=all_pairwise,
        distance_fn=lambda composite: 1.0 - composite,
        default_distance=1.0  # unmeasured or below-40% pairs
    )
    clusters = {
        'louvain': run_louvain(G),
        'hierarchical': run_hierarchical(distance_matrix),
        'dbscan': run_dbscan(distance_matrix)
    }
    
    # Build classification lookup for tier-aware recommendations (§6.2, §4.3)
    classification_map = {}
    for result in pairwise_results:
        classification_map[(result.report_a, result.report_b)] = result.classification
        classification_map[(result.report_b, result.report_a)] = result.classification
    
    # For each cluster, select golden report
    for algo, cluster_assignments in clusters.items():
        for cluster_id, member_ids in cluster_assignments.items():
            golden = select_golden_report(
                member_ids,
                weights={'completeness': 0.35, 'usage': 0.25, 
                        'freshness': 0.20, 'governance': 0.10, 
                        'recency': 0.10},
                pairwise_results=pairwise_results
            )
            # Build subsumption map for this cluster
            subsumption_map = build_subsumption_map(member_ids, pairwise_results)
            for member_id in member_ids:
                recommendation = determine_recommendation(
                    member_id, golden, pairwise_results,
                    subsumption_map, classification_map  # tier-aware (§6.2)
                )
                
                # Staleness override (§6.5): 90+ days stale with high similarity to an ACTIVE report.
                # Excludes 'review' — pairs flagged for review have unreliable composite scores
                # (divergence check, §6.2), so escalating to strong_decommission is unsafe.
                if recommendation not in ('keep', 'merge_content_into_golden', 'review'):
                    report_meta = get_report(member_id)
                    if (report_meta.days_since_refresh > 90 and
                            get_max_similarity_to_active_report(
                                member_id, pairwise_results, member_ids
                            ) > 0.70):
                        recommendation = 'strong_decommission'
                
                db.store_cluster_assignment(
                    cluster_id, algo, member_id,
                    is_golden=(member_id == golden),
                    recommendation=recommendation
                )
                
                # Populate content_migration_tasks when applicable (§9.1)
                # merge_content_into_golden implies golden ⊂ member (golden is subset).
                # We keep golden, decommission member → find member's content missing in golden.
                if recommendation == 'merge_content_into_golden':
                    gap = compute_gap_analysis(member_id, golden)  # decommissioned, kept
                    for item in gap['missing_in_golden']:
                        db.insert_content_migration_task(
                            source_report_id=member_id,
                            target_report_id=golden,
                            content_type=item['type'],
                            content_id=item['id'],
                            content_name=item['name']
                        )
    
    # ═══════════════════════════════════════════
    # STAGE 4: EXPLAINABILITY
    # ═══════════════════════════════════════════
    
    # Generate LLM explanations for non-trivial clusters
    # Use stable cache keys: hash of sorted member IDs, not arbitrary cluster_id
    for cluster_id, members in clusters['louvain'].items():
        stable_key = hashlib.sha256(
            '|'.join(sorted(members)).encode()
        ).hexdigest()
        
        if len(members) > 1:
            summary = llm_generate_cluster_summary(
                members, pairwise_results
            )
            db.store_explanation('cluster_summary', stable_key, summary, upsert=True)
    
    # Generate merge recommendations for actionable clusters
    for cluster_id, members in clusters['louvain'].items():
        stable_key = hashlib.sha256(
            '|'.join(sorted(members)).encode()
        ).hexdigest()
        
        if len(members) >= 2:
            golden = get_golden_for_cluster(cluster_id, db)
            merge_rec = llm_generate_merge_recommendation(
                members, golden, pairwise_results
            )
            db.store_explanation('merge_recommendation', stable_key, merge_rec, upsert=True)
    
    # Persist the final status of this batch run
    run_status = 'degraded_llm' if llm_degraded else 'success'
    db.complete_batch_run(
        run_id, 
        status=run_status,
        attempts=llm_attempts,
        failures=llm_failures
    )
    
    return RationalizationResult(
        pairwise_results=pairwise_results,
        clusters=clusters,
        total_reports=len(reports),
        decommission_candidates=count_by_recommendation('decommission'),
        merge_clusters=count_clusters_with_multiple_members(),
        unique_reports=count_by_recommendation('keep'),
        run_status=run_status
    )
```

---

## 13. Incremental Re-Run Design

Although the primary use case is a one-time batch rationalization, the schema and architecture are designed to support **incremental re-runs** without retrofitting.

### 13.1 Change Detection Schema

```sql
-- Incremental tracking columns are included in report_fingerprints CREATE TABLE (§9.1).
-- No ALTER TABLE needed — "schema from day one" per §13.3.

-- Event log so incremental runs are auditable, not silent
CREATE TABLE change_events (
    id                  SERIAL PRIMARY KEY,
    report_id           TEXT NOT NULL,
    event_type          TEXT CHECK (event_type IN ('created', 'republished', 'deleted')),
    changed_layers      TEXT[],       -- Which layers changed: ['dax', 'visuals', ...]
    detected_at         TIMESTAMP DEFAULT NOW(),
    processed_at        TIMESTAMP     -- NULL until processed
);
```

### 13.2 Incremental Comparison Strategy

Triggered by a metadata-change event (new report published, existing report republished with modified DAX/visuals, etc.) rather than a full nightly batch.

```python
def incremental_update(changed_report_ids: list, change_events: list, db: Database):
    """
    Scoped recompute: a new/edited report doesn't trigger a full N² re-run.
    change_events: list of {report_id, event_type} from change_events table.
    """
    # Separate deleted reports from created/republished
    deleted_ids = {e['report_id'] for e in change_events if e['event_type'] == 'deleted'}
    active_ids = {rid for rid in changed_report_ids if rid not in deleted_ids}
    
    # Handle deletions first: clean up all references (parent + child tables)
    for report_id in deleted_ids:
        db.delete_pairwise_scores_touching(report_id)
        db.delete_score_details_touching(report_id)       # child of pairwise
        db.delete_governance_flags_touching(report_id)     # child of pairwise
        db.delete_measure_equivalences_for_report(report_id)  # WHERE report_a_id = ? OR report_b_id = ?
        db.delete_cluster_assignments(report_id)
        db.delete_fingerprint(report_id)
        db.delete_content_migration_tasks(report_id)       # WHERE source_report_id = ? OR target_report_id = ?
        db.delete_source_fingerprints(report_id)           # CASCADE deletes source_blocks rows via FK

    # 1. Recompute fingerprints only for active changed reports
    for report_id in active_ids:
        report = get_report(report_id)
        new_fingerprints = compute_fingerprints(report)
        
        # Compute composite hash of all six fingerprints for cheap change-detection (§9.1)
        new_composite_hash = hashlib.sha256(
            (new_fingerprints['data_source_hash'] +
             new_fingerprints['semantic_model_hash'] +
             str(new_fingerprints['dax_minhash']) +
             str(new_fingerprints['dax_simhash']) +
             new_fingerprints['visual_hash'] +
             new_fingerprints['filter_hash']).encode()
        ).hexdigest()
        
        # Retrieve previous metadata to check if anything actually changed
        old_meta = db.get_fingerprint_meta(report_id)  # returns {'previous_hash_composite', 'version'}
        if old_meta and old_meta['previous_hash_composite'] == new_composite_hash:
            continue  # metadata touched but comparison-relevant fields are identical
            
        new_version = (old_meta['version'] + 1) if old_meta else 1
        new_fingerprints['version'] = new_version
        new_fingerprints['previous_hash_composite'] = new_composite_hash
        
        db.store_fingerprint(report_id, new_fingerprints, source_event='republished')

    # 2. Re-run blocking + LSH candidate generation ONLY for active changed reports
    #    against the full existing corpus — not full corpus against itself
    new_candidate_pairs = set()
    for report_id in active_ids:
        block = get_logical_block(report_id)
        block_reports = get_reports_in_block(block)
        new_candidate_pairs |= generate_candidates_for_report(report_id, block_reports)

    # 3. Deep-compare only the new candidate pairs (Stage 2, unchanged logic)
    for pair in new_candidate_pairs:
        result = deep_compare(pair)
        db.upsert_pairwise_result(result)

    # 4. Stale pairwise rows: re-validate pairs involving active changed reports
    stale_pairs = db.get_pairwise_rows_touching(active_ids)
    for pair in stale_pairs:
        if pair not in new_candidate_pairs:
            revalidate_or_retire(pair, db)

    # 5. Invalidate and recompute clusters affected by deletions AND active changes.
    # NOTE: This is an accepted approximation. Community-detection algorithms like Louvain
    # are not decomposable — a single edge change can require merging/splitting clusters
    # not in affected_clusters. The quarterly full re-baseline (§13.3) corrects any drift.
    affected_clusters = db.get_clusters_containing(changed_report_ids)
    for cluster_id in affected_clusters:
        recompute_cluster(cluster_id, db)

    # 6. Mark events as processed
    db.execute(
        "UPDATE change_events SET processed_at = NOW() "
        "WHERE report_id = ANY(%s) AND processed_at IS NULL",
        changed_report_ids
    )

def revalidate_or_retire(pair: tuple, db: Database):
    """
    Checks the status of a retired candidate pair. If its composite score
    falls below the 40% storage threshold, deletes the parent row and all
    associated child data (score_details, governance_flags, measure_equivalences)
    to prevent orphaned database rows.
    """
    id_a, id_b = pair
    result = db.get_pairwise_result(id_a, id_b)
    if result and result.composite < 0.40:
        db.delete_score_details_touching_pair(id_a, id_b)
        db.delete_governance_flags_touching_pair(id_a, id_b)
        db.delete_measure_equivalences_for_pair(id_a, id_b)
        db.delete_pairwise_score_for_pair(id_a, id_b)
```

### 13.3 Operational Model

- **Initial load**: Full "overnight batch" mode (§12 pseudocode) as the first-run path.
- **Steady state**: Incremental path (above) triggered by Power BI publish/refresh webhooks.
- **Periodic re-baseline**: Quarterly full re-run to catch drift the incremental path might miss.
- **Schema from day one**: `computed_at` timestamps, `version` fields, `source_event`, and the `change_events` table are included in the initial schema. The incremental logic is **not built for the initial release** but the schema supports it without migration. This avoids the "retrofit incremental support onto a batch system" anti-pattern.

---

## 14. Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Comparison unit | Report-level | Stakeholders think in reports, not pages/visuals |
| Cross-source matching | In scope with fuzzy schema matching (`pg_trgm` as pre-filter in prod) | Same data from CSV vs. DB should be detected |
| DAX comparison | Hybrid: signature → AST/text-norm → LLM (see §10.1 phased approach) | Best accuracy/cost tradeoff; parser is highest-risk dependency |
| DAX rewrite rules | Two-tier: SAFE (auto-apply) + CONDITIONAL (route to LLM if preconditions unverified) | Context-dependent semantics make blanket rewrites a correctness risk |
| Layout comparison | Ignored | Layout is a presentation choice, not functional |
| Visual type comparison | Equivalence classes | Bar vs. line showing same data = equivalent |
| Similarity score display | Per-layer breakdown (primary) + composite (derived) | Stakeholders need explainability |
| Measure weighting | TF-IDF across corpus | Generic measures shouldn't dominate similarity |
| Name normalization | Embedding-based (Azure OpenAI) | Handles synonyms and abbreviations |
| Clustering | Three algorithms available (Louvain, hierarchical, DBSCAN) | Different algorithms suit different analysis needs |
| Transitivity | Via graph topology (connected components), no phantom edges | Clustering algorithms handle transitivity natively; no synthetic pairwise scores |
| Subsumption tolerance | 90% set inclusion | Allows minor differences |
| Subsumption ↔ Golden | Reconciliation rule: `merge_content_into_golden` when usage favors subset | Prevents silent content loss; tracked via `content_migration_tasks` (§4.3, §6.4, §9.1) |
| RLS/Governance | Flag only, don't score | Merge risk, not similarity indicator |
| Staleness | Prioritization signal for decommission | 90+ days stale + similar = strong decommission |
| Usage | Separate input for recommendations, not similarity | Usage ≠ similarity |
| Classification thresholds | Multi-threshold with calibration | 95/80/60 defaults, tuned via labeled sample |
| LLM usage | Targeted: DAX edge cases + coverage gaps + explainability | Cost control, determinism where possible |
| LLM cost estimate | Token-based via pluggable `estimate_llm_cost()` (§7.3) | Computed from measured token counts × current pricing, not hardcoded |
| Golden report selection | Multi-factor + subsumption reconciliation | Objective, defensible, prevents content loss |
| Output storage | Full persistence to DB | Frontend consumption, auditability |
| Technology | Hybrid SQL/Python; `pg_trgm` as coarse pre-filter (prod), Levenshtein+embeddings as authority | Division of labor: SQL for cheap filtering, Python for scoring |
| LSH configuration | Explicit `params=(32, 8)`, inflection ≈ 0.65 | Pinned `(b, r)` avoids threshold/banding inconsistency; S-curve math documented |
| Run mode | One-time batch; schema supports incremental (§13) via `change_events` + webhooks | Correctness > speed; future-proofed without retrofitting |
| DAX parser | Phased: coverage audit → Option C (initial) → Option D (production) (§10.1) | Highest-risk dependency; audit corpus before committing to parser scope |
| Runtime estimate | Profiling-based `estimate_full_runtime()` from calibration (§11) | Defensible ("measured and extrapolated"), not asserted |
