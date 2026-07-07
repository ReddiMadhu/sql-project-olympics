Here are all 18 questions answered using the actual code and specs from all three repos:

---

## 🧑‍💼 Human Oversight

**Q1. Can any BI/ETL rationalization, migration, or decommissioning recommendation be implemented automatically, or is explicit human approval always required?**

**No action is ever automatic. Explicit human approval is required across all three tools:**

- **RSLI** — `POST /api/execute` requires a valid JWT (authenticated user). The `RiskClassifier` in `risk_classifier.py` hard-blocks High Risk scripts — the UI physically cannot progress to Source Mapping without human action.
- **TB-BI Migration (`bknd-ts`)** — `MigrationOrchestrator.execute()` only runs after a user uploads TML files and explicitly clicks Execute in `MigrationUploadPage.jsx`. Formulas flagged `requires_review = True` (after failing 3 self-heal attempts) are written to the Excel report for mandatory human correction before production use.
- **Excel Rationalization** — `RationalizationEngine.run_rationalization()` produces recommendations stored in `governance_recommendations`. No code path in the engine ever deletes, merges, or modifies any workbook. Decision band `extraction_quality < 0.6` always forces `review` status — never auto-decommission.

---

**Q2. What evidence is captured to demonstrate human review and approval of AI-generated recommendations and migration outputs?**

| Tool | Evidence | Where Stored |
|---|---|---|
| **RSLI** | `username`, `timestamp`, `script_hash` (SHA256), `risk_level`, `execution_status`, column overrides applied | DB `audit_logs` + Azure Blob `overrides.json` + `llm_logs/` |
| **TB-BI Migration** | `migration_id`, per-formula `confidence`, `requires_review` flag, every heal attempt with before/after DAX + validation pass/fail | `migrations.db` + `dax_healing_debug.log` + Excel export |
| **Excel Rationalization** | `action`, `uniqueness_score`, `kpi_overlap_score`, `datasource_overlap_score`, `reasons[]` (LLM justification), `is_real_ai` flag | `governance_recommendations` DB table |

RSLI also provides `GET /api/audit/export?format=csv|json` — a complete downloadable evidence trail for compliance submissions.

---

## ✅ Accuracy & Validation

**Q3. How is the accuracy of AI-generated dashboard migrations and ETL pipeline conversions validated before production deployment?**

**RSLI (ETL validation):**
1. Syntax gate — `ast.parse()` + 2000 line limit
2. Risk gate — AST-based `RiskClassifier` blocks High Risk execution
3. Schema gate — `source_validator.py` catches column additions/deletions/type changes vs. last snapshot; `SemanticColumnMatcher` recommends mappings when drift detected
4. Runtime gate — actual execution against real data, row counts + schema diffs + 5-row sample data captured at every node in the lineage DAG

**TB-BI Migration (formula conversion validation):**
`orchestrator.py` Phase 3.5 runs a **validate → self-heal → re-validate** loop:
```
ThoughtSpot formula → DAX (converter)
  → ValidationEngine.validate() → test slices (pass/fail)
  → If failed: SelfHealingAgent.correct_dax() [LLM] — up to 3 attempts
  → Re-validate after each heal
  → If still failed: confidence capped at 0.5, requires_review = True
```

**Excel Rationalization (recommendation accuracy):**
Deterministic decision bands from `rationalization_coverage.md`:
- `kpi ≥ 0.85 AND ds ≥ 0.85 AND fp_ratio ≥ 0.7` → decommission
- `kpi ≥ 0.50 AND ds ≥ 0.60` → merge
- `uniqueness ≥ 0.40` → keep
- `extraction_quality < 0.6` → always `review` (never auto-decommission)
- Ambiguous band → LLM assessment required; fallback to `review`

---

**Q4. What acceptance criteria or thresholds must be met before migrated assets are approved for use?**

**RSLI:**

| Gate | Criteria |
|---|---|
| Syntax | `ast.parse()` succeeds, ≤ 2000 lines |
| Risk | NOT classified High Risk |
| Source Mapping | All source files mapped (UI enforces) |
| Execution | All nodes complete (`no node_error`) |
| Schema | Output schema matches expected target |
| Audit | Traceable to authenticated `username` |

**TB-BI Migration:**

| Gate | Criteria |
|---|---|
| Parsing | TML files parse without errors |
| Validation | `overall_passed = True` after ≤ 3 heal attempts |
| Review Flag | `requires_review = False` before production |
| Confidence | `≥ 0.9` = auto-accept; `< 0.9` = human review recommended |

**Excel Rationalization:**

| Gate | Criteria |
|---|---|
| Extraction Quality | `extraction_quality_score ≥ 0.6` |
| Risk Flags | No blocking risks (VBA, external links, hidden sheets) |
| Overlap Scores | Must satisfy decision band thresholds |

---

**Q5. How is the correctness of Keep/Merge/Decommission recommendations verified against SME expectations?**

1. **Deterministic Scoring** — `overlap_scorer.py` computes KPI overlap (Jaccard), datasource overlap, fingerprint dedup ratio, and structural similarity with configurable env-var weights (`OVERLAP_WEIGHT_KPI=0.35`, `OVERLAP_WEIGHT_DS=0.25`, etc.)
2. **Risk Guard** — `detect_workbook_risks()` flags VBA macros, external links, hidden sheets — these **block** auto-decommission regardless of overlap scores, recorded in `governance_risks` table
3. **LLM Justification** — `Recommender` calls LLM (3 retries) for human-readable `reasons[]` per recommendation stored in DB
4. **Flag-Only Rule** — per `rationalization_coverage.md`: flag-only risks *"never trigger automatic decommission alone"* — always requires human decision
5. **SME Override** — UI presents recommendations with scores + reasons; SME can change action before any implementation

---

## 🔍 Traceability & Auditability

**Q6. Since traceability metadata is currently not captured, how will users trace a recommendation back to the source dashboard, report, or ETL asset that generated it?**

| Tool | Status | Trace Chain |
|---|---|---|
| **RSLI** | ✅ Fully implemented | `session_id → script_hash (SHA256) → audit_log → Azure Blob (source_files/, llm_logs/, overrides.json)` |
| **TB-BI Migration** | ✅ Implemented | `migration_id → parsed TML model → formula conversions → validation results → healing attempts → PBIP output` (all in `migrations.db`) |
| **Excel Rationalization** | ⚠️ Partial | `governance_recommendations.workbook_id → workbooks → calculated_fields → kpi_cluster_cache`. **LLM call metadata (prompts, model config, response) not yet logged** — explicitly identified as a gap in the repo's own `ai_governance_answers.md` |

For Excel Rationalization, the recommended trace chain to implement:
```
Recommendation
  → workbook_id → workbook name + purpose
  → calculated_fields (KPIs used in scoring)
  → kpi_cluster_cache (canonical groupings)
  → LLM call log (prompt, model, config, response) ← TO BUILD
```

---

**Q7. Are AI recommendations, human overrides, approvals, and final decisions logged for audit purposes?**

| Event | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| AI recommendation generated | ✅ `llm_call` event → DB + Blob | ✅ Per-formula conversion + validation in DB | ✅ `governance_recommendations` table |
| Human override applied | ✅ `validation_override` event | ✅ `requires_review = True` flags it | ⚠️ Not yet implemented |
| Approval / execution trigger | ✅ `script_execute` with `username` | ✅ `migration_id` created on upload action | ⚠️ No user auth layer |
| Final decision / result | ✅ `execution_complete` event | ✅ `update_migration_status("completed")` | ✅ `governance_recommendations.action` |
| High-risk block | ✅ `risk_blocked` event | ✅ `requires_review=True` after failed heals | ✅ `governance_risks` table |
| Audit export | ✅ CSV/JSON via `/api/audit/export` | ✅ Excel report + `dax_healing_debug.log` | ⚠️ Not yet implemented |

---

## 📊 Monitoring & Reliability

**Q8. Since drift monitoring is not implemented, how will degradation in extraction, rationalization, or migration quality be detected over time?**

**Current signals available (not yet automated alerts):**

- **RSLI** — Schema drift IS detected: `source_validator.py` compares every uploaded file against its last snapshot (column additions/deletions/type changes flagged automatically). Rising `execution_status = "failed"` in the Audit Trail signals pipeline degradation. Row count anomalies visible in the lineage DAG.
- **TB-BI Migration** — `requires_review` count per migration saved in DB. Rising counts signal formula quality degradation. `dax_healing_debug.log` shows rising attempt counts if model quality drops.
- **Excel Rationalization** — `extraction_quality_score < 0.6` triggers `review`. Rising `review` rates signal workbook complexity drift. `LLMCaller` retry rate (3 attempts, 60s interval) signals model availability issues.

> ⚠️ **Formal automated drift monitoring (scheduled re-runs, baseline comparison, alerting) is not yet implemented in any tool.** The Excel Rationalization repo's own `ai_governance_answers.md` recommends monthly governance committee reviews of AI output quality metrics.

---

**Q9. What operational metrics will be monitored post-deployment to identify recurring AI errors or quality issues?**

| Metric | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| Success/failure rate | `execution_status` trend | `requires_review` count per migration | `review` action rate vs. portfolio |
| AI quality signal | LLM token usage (`llm_call` events) | Confidence distribution (high/med/low) | LLM retry count per workbook |
| Self-healing effectiveness | N/A | Healing attempts per formula; post-heal pass rate | N/A |
| Risk block rate | `risk_blocked` event count | Formulas `requires_review=True` after 3 heals | `governance_risks` flag count |
| Override frequency | `validation_override` count | Manual corrections post-export | To be implemented |
| Duration/latency | `duration_ms` per node | `elapsed_seconds` per migration | LLM retry interval logs |
| Schema drift signals | Column mismatch count per upload | N/A | `extraction_quality_score` trend |

---

## 🤖 Agentic AI Governance

**Q10. How do the Discovery, Intelligence, Rationalization, and Migration agents interact with one another?**

**Excel Rationalization pipeline:**
```
[Discovery/Extraction]
  ExcelLoader + ProfilingEngine
  → Extracts calculated_fields, fingerprints, datasources into DB
         ↓ (human reviews)
[Intelligence Agent]  RationalizationEngine.run_intelligence()
  → compute_complexity_scores()
  → run_kpi_canonicalization()  [LLM + fuzzy]
  → _enrich_dashboard_metadata()  [LLM: LOB, domain, user_groups]
         ↓ (human validates)
[Rationalization Agent]  RationalizationEngine.run_rationalization()
  → detect_workbook_risks()
  → compute_pairwise_overlaps()
  → compute_uniqueness_scores()
  → Recommender.run()  [deterministic bands + LLM justification]
         ↓ (SME overrides if needed)
[No automated Migration] — workbook merging is out of scope (v2)
```

**TB-BI Migration pipeline (`bknd-ts`):**
```
SpotAppLoader + TMLParser → intermediate_model
LogicGraphBuilder → ReactFlow DAG saved to DB
ThoughtSpotFormulaConverter → DAX per formula
ValidationEngine → test slices per formula
SelfHealingAgent (LLM, ≤3 rounds) → corrected DAX
PBIPGenerator → Power BI project files
ModelEnhancementAgent → enhancement guide
ExcelReportGenerator + DAX + JSON + ZIP export
```

Each stage is sequential and outputs structured data to the next. Human checkpoints sit between major stages.

---

**Q11. What controls exist to prevent an incorrect output from one agent from propagating through downstream agents?**

| Control | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| Human checkpoint | ✅ User reviews parse + risk before clicking Execute | ✅ User reviews Excel report before production deployment | ✅ SME reviews all recommendations |
| Quality gate blocks progression | ✅ High Risk = hard block | ✅ `requires_review=True` after failed heals | ✅ `extraction_quality<0.6` → review only |
| Validate-then-correct loop | ✅ Schema validation before execution | ✅ ValidationEngine → 3 self-heal rounds | ✅ LLM retry (3 attempts) with fallback |
| Session/run isolation | ✅ Each run in `rsli_{session_id}/` dir | ✅ Each migration in unique `migration_id` namespace | ✅ DB-scoped per workbook subset |
| Partial failure visibility | ✅ Failed node shown; downstream = "not reached" | ✅ Failed migration status + error in DB | ✅ `risk_error`, `overlap_error` in summary |
| LLM not used for decisions | ✅ LLM = descriptions only; risk is AST-based | ✅ LLM proposes fix; `ValidationEngine` decides pass/fail | ✅ LLM = justification text only; scores are deterministic |

---

**Q12. Are intermediate agent outputs available for human review before migration or rationalization decisions are finalized?**

**Yes — across all tools:**

| Stage | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| Parse/Extraction | ✅ DAG skeleton shown before execution | ✅ Table/formula counts shown in progress UI | ✅ Extracted workbooks visible in DB/UI |
| Risk/Quality | ✅ Risk badge (green/yellow/red) pre-execution | ✅ `requires_review` flag in Excel export | ✅ `governance_risks` table + quality scores |
| AI recommendations | ✅ Node descriptions labeled `template` vs `llm` | ✅ Full `dax_healing_debug.log` per formula | ✅ `reasons[]` from LLM per recommendation |
| Execution output | ✅ Lineage DAG, sample data, schema diff, LLM logs | ✅ Excel + DAX + JSON + PBIP all downloadable | ✅ Scores + cluster data queryable from DB |
| Enhancement/Post | ✅ Audit Trail expanded row with Blob artifacts | ✅ ModelEnhancementAgent produces guide of manual steps | ✅ Overlap pairs + uniqueness scores |

---

## 🧠 Hallucination & Explainability

**Q13. How do agents interact with one another during discovery, intelligence, rationalization, and migration phases?**

*(See Q10 for full pipelines.)*

On hallucination prevention specifically:
- **RSLI** — `LLMDataSanitizer` strips all sample data values before any LLM call. LLM sees only column names, dtypes, and code — not actual cell values. Even if LLM hallucinated a column name, `source_validator.py` uses the real DataFrame schema.
- **TB-BI** — `SelfHealingAgent` receives the original ThoughtSpot formula + the specific test slice failures — it cannot hallucinate what the source formula said. `ValidationEngine` (deterministic) decides pass/fail; the LLM only proposes a candidate fix.
- **Excel Rationalization** — LLM invoked only for KPI cluster naming, dashboard metadata (LOB/domain), and recommendation justification. All scoring is deterministic. If LLM returns `None` after 3 attempts, `LLMCaller` falls back gracefully; the recommendation is still generated using deterministic scores.

---

**Q14. Can one agent's output automatically influence decisions made by downstream agents?**

**Structured data flows automatically; decisions require deterministic gates or human approval.**

- **What flows automatically:** Structured intermediate model (column names, schema, node graph, formula list) — safe structured data, not free-form LLM output.
- **What does NOT flow automatically:**
  - **RSLI:** LLM descriptions flow to the UI but do NOT affect execution logic or risk scoring.
  - **TB-BI:** Self-healer's corrected DAX flows to PBIP generator — but **only after `ValidationEngine` confirms it passes**. If it never passes, `requires_review = True` prevents bad output from silently reaching production.
  - **Excel Rationalization:** LLM dashboard metadata enrichment flows downstream indirectly. Actual overlap scores are computed from `calculated_fields.fingerprint` and `datasources` — deterministic data, not LLM output.

> **In all three tools, LLM is used for natural language generation only — never for scoring, classification, or routing decisions.** This is the primary hallucination firewall.

---

**Q15. What validation checkpoints exist between agents before recommendations are passed downstream?**

**Type 1 — Automated Technical Validation:**
- **RSLI:** Risk classification → schema validation → runtime execution snapshot comparison
- **TB-BI:** `ValidationEngine` test slices → self-heal loop → re-validation (up to 3 rounds)
- **Excel Rationalization:** `extraction_quality_score` gate → `governance_risks` flag check → decision band thresholds

**Type 2 — Human Review Gates:**
- **RSLI:** SME reviews parse result + risk badge → reviews lineage DAG post-execution
- **TB-BI:** Reviewer inspects Excel report (all `requires_review=True` formulas flagged) → validates PBIP in Power BI before production
- **Excel Rationalization:** SME reviews recommendations with scores + LLM justifications → overrides where needed

**Type 3 — Audit Cross-Checks:**
- **RSLI:** `session_id` links all steps; compare AI recommendation (`llm_logs`) vs. human override (`overrides.json`) vs. final result
- **TB-BI:** `migration_id` links all phases; `dax_healing_debug.log` is a complete forensic trail per formula
- **Excel Rationalization:** `workbook_id` links extraction → intelligence → rationalization; `governance_risks` cross-referenced with recommendations

---

**Q16. How are propagation errors identified when incorrect outputs from one agent affect later stages of the workflow?**

| Detection Mechanism | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| Cascading failure visibility | `node_error` in lineage DAG; downstream = "not reached" | `status = "failed"` + error in DB; partial conversions visible | `risk_error`, `overlap_error`, `recommendation_error` in summary dict |
| Schema/data mismatch | `source_validator.py` catches mismatches pre-execution | `ValidationEngine` test slices catch semantic DAX errors | `extraction_quality_score < 0.6` catches poor extraction |
| Override audit trail | `validation_override`: shows AI said X, human changed to Y | `requires_review=True` + debug log shows what was changed | LLM justification shows AI reasoning (override log = gap) |
| LLM forensics | `llm_logs/*.json` in Azure Blob (full prompt + response) | `dax_healing_debug.log`: formula → initial DAX → each heal → final | LLM call logging (to be implemented) |
| Confidence degradation | Descriptions labeled `template` vs `llm` | Confidence capped at 0.5 after failed heals; `requires_review=True` | LLM retry failure logged by `LLMCaller` |

---

**Q17. Is there a mechanism for reviewers to inspect intermediate outputs produced by each agent?**

**Yes — multiple inspection mechanisms in every tool:**

| Inspection Point | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| Pre-execution | Risk badge, parse result, source mapping slots | Progress UI + logic graph (ReactFlow DAG) | Extracted workbook list, KPI clusters, complexity scores |
| Real-time | SSE streaming: nodes animate gray → orange → green/red | Progress % + stage message | N/A (batch pipeline) |
| Post-execution | Lineage DAG, Node Detail Panel (Metrics, Schema, Sample Data, Code, Description) | Excel report, DAX file, JSON model, PBIP, Enhancement Guide | `governance_recommendations` with scores + reasons |
| Stored artifacts | Azure Blob: `source_files/`, `output_files/`, `llm_logs/`, `overrides.json` | `dax_healing_debug.log`, `migrations.db` | `governance_recommendations`, `governance_risks`, `kpi_cluster_cache` |
| Export | `GET /api/audit/export?format=csv\|json` | ZIP (PBIP + Excel + DAX + JSON) | DB queryable; export = gap |

---

**Q18. Can agent recommendations be overridden at any stage before migration or rationalization actions are approved?**

**Yes — overrides are supported at every stage:**

| Override Point | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| Source mapping | ✅ Reassign column mappings, change dtype casts | ✅ Choose which TML files to include | ✅ Scope `workbook_ids` to include/exclude |
| Risk assessment | ✅ Edit script to remove High Risk patterns, re-parse | N/A | ✅ Risk flags are advisory; human can proceed |
| AI recommendation | ✅ `validation_override` captured in audit | ✅ Manual DAX edit after export | ✅ SME changes action before implementation |
| Post-execution rejection | ✅ Reject output, re-execute with different files | ✅ Re-run migration with corrected TML files | ✅ Re-run with updated workbook set |
| Override evidence | ✅ `overrides.json` in Azure Blob + audit event | ✅ `requires_review=True` + debug log shows edits | ⚠️ Override logging not yet implemented |

---

## Summary

| Area | RSLI | TB-BI Migration | Excel Rationalization |
|---|---|---|---|
| **Human Oversight** | ✅ JWT auth + Risk gate | ✅ Explicit trigger + review flags | ✅ Advisory only; no auto-action |
| **Accuracy & Validation** | ✅ 4-layer: AST, risk, schema, runtime | ✅ Validate → self-heal (3 rounds) | ✅ Deterministic bands + LLM justification |
| **Traceability** | ✅ DB + Blob, SHA256, session chain | ✅ `migration_id` + debug log chain | ⚠️ `workbook_id` exists; LLM call log = gap |
| **Monitoring** | ⚠️ Signals exist; no automated alerts | ⚠️ Confidence trends in DB; no alerts | ⚠️ Quality scores in DB; no alerts |
| **Agentic Governance** | ✅ Sequential + gates + isolation | ✅ 7-phase + validate gates | ✅ 3-phase + risk guards |
| **Hallucination Control** | ✅ LLM = descriptions; PII stripped | ✅ LLM = suggestions; rules decide | ✅ LLM = justification; scores deterministic |
