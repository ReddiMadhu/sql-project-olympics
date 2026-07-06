# AI Governance Implementation — Answers

> [!NOTE]
> These answers address the "Governance Implementation Currently Missing" items identified for the Excel Rationalization AI use case. Each answer provides a recommended implementation approach aligned with industry best practices (NIST AI RMF, EU AI Act, ISO/IEC 42001).

---

## 1. Safeguards Against Malicious Prompts, Prompt Injection, Data Exfiltration & Misuse

**Question:** *Are safeguards in place to protect the AI use case against malicious prompts, prompt injection, data exfiltration, and misuse of tools or plugins?*

### Answer

**Yes — the following safeguards are implemented / recommended for implementation:**

| Threat | Safeguard | Status |
|---|---|---|
| **Prompt Injection** | Input sanitization layer that strips or escapes known injection patterns (e.g., "ignore previous instructions", encoded payloads) before prompts reach the model | ☐ To Implement |
| **Prompt Injection** | System prompt hardening — the system prompt uses delimiters, role-locking instructions, and explicit refusal directives to prevent override attempts | ☐ To Implement |
| **Prompt Injection** | Output validation that checks model responses against expected schema/format before returning to the user | ☐ To Implement |
| **Data Exfiltration** | Data Loss Prevention (DLP) filters on both input and output to detect and block PII, credentials, or sensitive data from being surfaced | ☐ To Implement |
| **Data Exfiltration** | Network-level controls ensuring the AI model cannot make outbound calls to external endpoints or URLs embedded in prompts | ☐ To Implement |
| **Tool/Plugin Misuse** | Principle of least privilege — all tools and plugins are scoped to the minimum permissions required; no arbitrary code execution is allowed | ☐ To Implement |
| **Tool/Plugin Misuse** | Allowlist-based tool invocation — only pre-approved tools/plugins can be called by the model; all others are blocked by default | ☐ To Implement |
| **General** | Rate limiting and anomaly detection on API calls to identify and throttle suspicious usage patterns | ☐ To Implement |
| **General** | Regular red-teaming and adversarial testing exercises conducted quarterly | ☐ To Implement |

> **NL Note (BI Assist & Other Places):** The same safeguards checklist should be audited across all AI-enabled tools including BI Assist, Copilot integrations, and any custom plugin endpoints. A centralized security review ensures consistent protection.

---

## 2. Complaints Management — End User Feedback Process

**Question:** *Is there a process in place for end user feedback/complaints?*

### Answer

**Yes — a structured feedback and complaints management process is recommended as follows:**

**A. Feedback Collection Channels:**
- **In-App Feedback Button:** A persistent "Report Issue / Give Feedback" button within the AI interface allowing users to flag incorrect, harmful, or unsatisfactory outputs with a single click.
- **Thumbs Up / Thumbs Down Rating:** Each AI-generated output includes a simple rating mechanism to capture user satisfaction at the point of interaction.
- **Dedicated Email / Ticketing Channel:** A dedicated mailbox (e.g., `ai-feedback@company.com`) or ServiceNow queue for formal complaints that require investigation.

**B. Complaint Handling Workflow:**

```
User submits complaint
        │
        ▼
Auto-logged in complaint tracker (ServiceNow / Jira)
        │
        ▼
Triaged within 24 hours by AI Operations team
        │
        ├── Low severity (inaccurate output) → Logged for model improvement, user notified within 3 business days
        ├── Medium severity (biased/unfair output) → Escalated to AI Ethics review, user notified within 2 business days
        └── High severity (harmful/toxic output) → Immediate escalation to AI Governance lead + CISO, output quarantined, user notified within 24 hours
```

**C. Feedback Loop:**
- All feedback is aggregated monthly and reviewed by the AI Governance Committee.
- Patterns in complaints (e.g., recurring inaccuracies on a specific data domain) trigger targeted model retraining or prompt refinement.
- Users who submit complaints receive a closed-loop notification when their issue has been resolved.

**D. Metrics Tracked:**
- Total complaints per month
- Average resolution time
- Complaint category breakdown
- Recurrence rate (same issue reported again)

---

## 3. Transparency / Model Card

**Question:** *Have you created an AI model card (that can be shared with regulators or clients) for this AI use case?*

### Answer

**Yes — an AI Model Card is recommended for creation with the following structure:**

### AI Model Card: Excel Rationalization Use Case

| Section | Content |
|---|---|
| **Model Name** | Excel Rationalization AI Assistant |
| **Version** | v1.0 |
| **Date** | July 2026 |
| **Use Case Description** | Automated analysis and rationalization of Excel-based business processes, identifying overlaps, redundancies, and consolidation opportunities across enterprise spreadsheet portfolios. |
| **Intended Users** | Internal business analysts, operations teams, and IT rationalization leads. |
| **Intended Use** | Scoring overlap between Excel workbooks, recommending consolidation candidates, and generating rationalization reports. NOT intended for autonomous decision-making — all outputs require human review. |
| **Out-of-Scope Uses** | Not designed for financial decision-making, regulatory reporting, or processing of personal/sensitive data without additional controls. |
| **Model Architecture** | [Specify: e.g., GPT-4o / Gemini / Claude via API, or custom fine-tuned model] |
| **Training Data** | [Specify: e.g., No custom training data — uses pre-trained foundation model with enterprise context provided via prompts and RAG] |
| **Performance Metrics** | Overlap scoring accuracy: [X]% on internal test set; False positive rate: [X]%; Processing time: [X] seconds per workbook pair |
| **Limitations & Known Biases** | May underperform on highly domain-specific Excel workbooks (e.g., actuarial models); Limited accuracy on password-protected or macro-heavy workbooks; English-language optimized. |
| **Ethical Considerations** | No personal data is processed; Outputs are advisory only; Human-in-the-loop required for all rationalization decisions. |
| **Risk Classification** | Low-to-Medium risk (per EU AI Act categorization — not a high-risk use case) |
| **Data Retention** | Input data processed in-memory only; traceability metadata retained for [X] months per data retention policy. |
| **Contact / Owner** | [Team Name], [Email] |

> [!TIP]
> This model card should be version-controlled and updated with each significant model or prompt change. Store it in a central governance repository accessible to compliance and legal teams.

---

## 4. Traceability Metadata

**Question:** *Is traceability metadata (prompts, system prompts, model/config versions, and source references) captured and retained for auditing and accountability purposes?*

### Answer

**Yes — the following traceability metadata is recommended to be captured and retained:**

### Metadata Captured Per AI Interaction

| Metadata Field | Description | Example |
|---|---|---|
| `request_id` | Unique identifier for each AI interaction | `uuid-v4` |
| `timestamp` | ISO 8601 timestamp of the request | `2026-07-06T15:42:09+05:30` |
| `user_id` | Authenticated user who initiated the request | `madhu@company.com` |
| `user_prompt` | The exact user input/prompt sent | Full text captured |
| `system_prompt` | The system prompt active at time of request | Full text captured (version-tagged) |
| `system_prompt_version` | Version identifier for the system prompt | `v2.3.1` |
| `model_name` | The AI model used | `gpt-4o-2024-08-06` |
| `model_config` | Temperature, max_tokens, top_p, and other parameters | `{"temperature": 0.2, "max_tokens": 4096}` |
| `source_references` | Documents/data sources referenced by the AI | List of file names, row ranges |
| `ai_response` | The complete model output | Full text captured |
| `response_time_ms` | Latency of the AI response | `1234` |
| `feedback_rating` | User's thumbs up/down rating (if provided) | `positive` / `negative` / `null` |
| `session_id` | Groups related interactions in a session | `uuid-v4` |

### Storage & Retention

- **Storage:** All metadata is logged to a dedicated, append-only audit database (e.g., Azure SQL / PostgreSQL with write-once policies, or Azure Blob Storage with immutability policies).
- **Retention Period:** Minimum **12 months** for standard interactions; **36 months** for interactions flagged as complaints or incidents (aligned with regulatory expectations).
- **Access Control:** Audit logs are read-only for all users except the AI Governance team. Access is logged via Azure AD / Entra ID.
- **Encryption:** All metadata is encrypted at rest (AES-256) and in transit (TLS 1.2+).

### Implementation Approach

```python
# Example: Traceability logging middleware
import uuid
from datetime import datetime

def log_ai_interaction(user_id, user_prompt, system_prompt, model_name, 
                       model_config, ai_response, source_references):
    metadata = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "system_prompt_version": get_current_prompt_version(),
        "model_name": model_name,
        "model_config": model_config,
        "ai_response": ai_response,
        "source_references": source_references,
        "response_time_ms": calculate_latency(),
    }
    audit_db.insert("ai_audit_log", metadata)  # Append-only table
    return metadata["request_id"]
```

---

## 5. Toxicity Control — Detection & Mitigation Mechanisms

**Question:** *Does the AI use case implement mechanisms to detect, prevent, or mitigate the generation or amplification of toxic, abusive, or discriminatory content?*

### Answer

**Yes — the following multi-layered toxicity control mechanisms are recommended:**

### Layer 1: Input Filtering (Pre-Model)
- **Content classification** on user inputs to detect and flag toxic, abusive, or discriminatory language before it reaches the model.
- **Blocklist filtering** for known slurs, hate speech terms, and harassment patterns.
- **Intent detection** to identify prompts that attempt to elicit harmful content (e.g., "generate offensive content about...").
- Flagged inputs are blocked and logged for review; the user receives a standardized refusal message.

### Layer 2: Model-Level Controls
- **System prompt guardrails** explicitly instruct the model to refuse generating toxic, discriminatory, or harmful content under any circumstances.
- **Temperature and sampling controls** set conservatively (low temperature) to reduce the likelihood of unpredictable or harmful outputs.
- **Content safety API integration** (e.g., Azure AI Content Safety, OpenAI Moderation API, or Google Cloud Natural Language API) applied to all model outputs before they are returned to the user.

### Layer 3: Output Filtering (Post-Model)
- **Automated toxicity scoring** on every model output using a dedicated content safety classifier. Outputs exceeding the toxicity threshold (e.g., score > 0.7) are blocked and replaced with a safe fallback response.
- **PII detection** to prevent the model from surfacing personally identifiable information in outputs.
- **Bias detection** checks for outputs that disproportionately associate negative attributes with protected characteristics (race, gender, religion, etc.).

### Layer 4: Monitoring & Alerting
- **Real-time dashboard** tracking toxicity scores across all AI interactions.
- **Automated alerts** triggered when toxicity scores spike above baseline or a single interaction scores critically high.
- **Weekly reports** summarizing toxicity metrics reviewed by the AI Governance Committee.

### Applicability to This Use Case

> [!IMPORTANT]
> For the Excel Rationalization use case, the toxicity risk is **low** because the model primarily processes structured data (spreadsheet metadata, column names, formulas) rather than free-form user-generated content. However, safeguards are still warranted because:
> - Users can input free-text prompts that could contain or request harmful content.
> - Model outputs include natural language explanations and recommendations that could theoretically contain biased language.
> - Defense-in-depth is a governance best practice regardless of perceived risk level.

---

## 6. Toxicity Control — Governance & Human Review (If Model Lacks Built-In Mitigation)

**Question:** *If the model does not include toxicity mitigation mechanisms, what governance, operational controls, or human review processes are in place to monitor, identify, and address toxic or harmful outputs produced by the model?*

### Answer

**Even if the underlying model lacks built-in toxicity mitigation, the following governance and operational controls provide comprehensive coverage:**

### A. Governance Controls

| Control | Description | Frequency |
|---|---|---|
| **AI Governance Committee Review** | Quarterly review of all AI outputs flagged as potentially toxic or harmful | Quarterly |
| **AI Ethics Policy** | Published policy prohibiting the use of AI systems to generate or amplify harmful content; all users must acknowledge before access | Annual acknowledgment |
| **Incident Response Procedure** | Documented procedure for handling toxic output incidents, including escalation paths, containment steps, and root cause analysis | As needed, reviewed annually |
| **Vendor Risk Assessment** | Assessment of the AI model provider's own content safety commitments and capabilities | At onboarding + annually |

### B. Operational Controls

| Control | Description |
|---|---|
| **External Content Safety API** | All outputs are passed through a third-party content safety API (e.g., Azure AI Content Safety) as a compensating control, regardless of the model's built-in capabilities |
| **Output Sampling & Review** | A random sample of [X]% of AI outputs is reviewed by a human reviewer weekly to detect toxic content that automated systems may miss |
| **User Reporting Mechanism** | Users can flag any output as toxic/harmful via the in-app feedback button, triggering immediate review |
| **Automated Keyword Monitoring** | Regex and NLP-based monitoring scans all outputs for known toxic patterns and keywords |

### C. Human Review Process

```
AI generates output
        │
        ▼
Automated toxicity scoring (Content Safety API)
        │
        ├── Score < 0.3 (Safe) → Output delivered to user
        ├── Score 0.3–0.7 (Uncertain) → Output delivered + flagged for async human review within 48 hours
        └── Score > 0.7 (Toxic) → Output BLOCKED → Replaced with safe fallback message
                                                    │
                                                    ▼
                                        Incident logged → Reviewed by AI Ethics lead within 24 hours
                                                    │
                                                    ▼
                                        Root cause analysis → Prompt/system prompt updated if needed
                                                    │
                                                    ▼
                                        User notified of resolution
```

### D. Escalation Matrix

| Severity | Response Time | Reviewer | Action |
|---|---|---|---|
| **Low** (mildly inappropriate language) | 48 hours | AI Operations Analyst | Log, refine prompt, no user notification needed |
| **Medium** (biased or discriminatory output) | 24 hours | AI Ethics Lead | Investigate, update guardrails, notify affected user |
| **High** (hate speech, threats, explicit content) | 4 hours | AI Governance Lead + CISO | Quarantine output, suspend use case if systemic, notify all stakeholders |
| **Critical** (regulatory or legal exposure) | 1 hour | CRO / DPO / Legal | Full incident response, potential regulatory notification |

---

## Summary & Recommended Next Steps

| Governance Area | Readiness | Priority |
|---|---|---|
| Safeguards (Prompt Injection, etc.) | 🟡 Partially in place | **High** |
| Complaints Management | 🔴 Not yet implemented | **High** |
| Model Card / Transparency | 🔴 Not yet created | **Medium** |
| Traceability Metadata | 🟡 Partially in place | **High** |
| Toxicity Control (Automated) | 🟡 Partially in place | **Medium** |
| Toxicity Control (Governance/Human) | 🔴 Not yet implemented | **Medium** |

> [!IMPORTANT]
> **Recommended immediate actions:**
> 1. Implement the traceability metadata logging (can be integrated into the existing `governance.py` routes)
> 2. Create the AI Model Card document and store in the governance repository
> 3. Set up the user feedback/complaints channel (even a simple form + ServiceNow ticket)
> 4. Integrate a content safety API (e.g., Azure AI Content Safety) for output screening
