Kill the Quote Spreadsheet:

Autonomous RFx, Dual-LLM Quote Auditor & Gated PO EngineAn enterprise-grade, stateful multi-agent procurement system built to eliminate manual spreadsheet retyping, normalize messy multimodal vendor quotes (PDFs, Word docs, angled photos, emails, spreadsheets), execute deterministic price comparisons, and deliver defensible award decisions through an interactive human-in-the-loop workflow.
The Problem We Are Solving

In enterprise procurement, category buyers routinely spend days manually retyping unstructured vendor bids—whether it's a PDF letterhead with a discount buried in a footnote, a dense narrative Word document, an angled smartphone photo of a rate card, or a shorthand email ("42/kg for 5-ply, 38/kg for 3-ply...").When leadership asks a simple ad-hoc question ("What if we split the award—cheapest per line, but only among vendors who cleared quality compliance?"), the manual cycle starts over. This system automates the entire loop from RFx generation to final audited PO.

System Architecture & Tech StackCode snippetgraph TD
    A[create_rfx_node: Template-Free RFx] --> B[dual_extract_quotes_node: Gemini 3.8 Flash + Groq Llama + Injection Guard]
    B --> C[diff_and_questionnaire_node: Deduplicated Ranges + Confidence Escalation]
    C --> D{{"⏸️ INTERRUPT 1: Multi-Turn Clarification Loop\n(Catches Vague Replies & Demands Rates/PO)"}}
    D -->|Follow-Up Needed| D
    D -->|Clarifications Resolved| E[reconcile_and_score_node\nDuckDB 5-Parameter Best Quotes]
    E --> F{{"⏸️ INTERRUPT 2: Internal Validation Gate\n(Best Quotes Shown, PO Held)"}}
    F -->|Approved| G[generate_po_node\nFinal PO Released]
    G --> H([END])
Agent Orchestration Framework: 

LangGraph (StateGraph) managing stateful multi-node progression with dual human-in-the-loop interrupt breakpoints.State Persistence & Fault Tolerance: SQLite3 (SqliteSaver) persisting every node transition at /tmp/procurement_state.db to enable instant crash recovery and checkpoint history inspection.Primary Reader & Analyst LLM: gemini-3.8-flash (with automatic fallback to gemini-3.1-flash-lite and gemini-3.5-flash-lite) handling multimodal document parsing, RFx generation, and clarification auditing.Secondary Verification LLM: Groq Llama (llama-3.3-70b-versatile) performing independent second-opinion extractions to compute field-level confidence scores.Analytical Calculation Engine: DuckDB (In-Memory SQL) + Pandas for deterministic 30-line normalization, UoM/currency conversions, and L1 split-award ranking.Frontend UI: Streamlit delivering a multi-stage interactive buyer dashboard with a live agent state inspector.

Key Features & Enterprise Guardrails

Flexible, Template-Free RFx Generation:Generates a comprehensive RFx schedule from the buyer's 30-line requirement spreadsheet while explicitly forbidding rigid vendor submission templates. It actively encourages Word, PDF, image, and email submissions.   Dual-LLM Cross-Verification & Confidence Scoring:Both Gemini 3.8 Flash and Groq Llama independently extract structured JSON from vendor files.Field discrepancies and low-confidence extractions ($< 0.85$) automatically trigger Human Escalation flags in the audit dashboard.   Multi-Turn Iterative Clarification Loop:When vendors give vague responses (e.g., "same price as last bid" without actual numbers or PO references), the system evaluates the reply, detects incompleteness, and automatically generates targeted follow-up questions demanding precise rates or prior PO documentation.   Prompt Injection & Security Guardrails:A deterministic Regex + heuristic guardrail scans buyer uploads, vendor quotes, and clarification text boxes at every pipeline stage to block system instruction overrides or unauthorized price manipulation attempts.Deduplicated & Clean Audit Flags:Consecutive missing lines or spec deviations are grouped cleanly into ranges (e.g., "Lines 28–30 missing"), avoiding UI clutter. Vendors in 100% sync require zero clarification.Staged Governance & Gated PO Release:The system presents ranked best quote scorecards and audit defense ledgers first. Purchase Order generation is strictly locked behind an internal stakeholder sign-off validation gate.

Repository Structure

schemas.py: Strongly typed Pydantic data models (BuyerLineItem, VendorQuoteExtraction, ClarificationEvaluation) and LangGraph state contract (ProcurementState).tools_and_models.py: Multimodal file parsers (PyMuPDF for PDFs, python-docx for Word, pandas for Excel/CSV, Pillow for angled smartphone images), prompt injection scanners, and the DualLLMRouter.graph_engine.py: The LangGraph state machine containing all 5 nodes, clarification evaluation logic, DuckDB normalization engine, and SQLite checkpointer configuration.app.py: The Streamlit frontend UI coordinating the 5-step buyer workflow and embedding the Live Agent State Inspector.requirements.txt: Pinned dependencies for deployment.

How to Run & Demo

1. InstallationClone the repository and install dependencies:Bashpip install -r requirements.txt
2. Set Environment Variables / Streamlit SecretsConfigure your API keys in your environment or Streamlit secrets:Bashexport GEMINI_API_KEY="your_google_ai_studio_key"
export GROQ_API_KEY="your_groq_cloud_key"
3. Launch the ApplicationRun the Streamlit UI locally:Bashstreamlit run app.py
4. Step-by-Step Demo Walkthrough
Step 1: Upload your 30-line requirement spreadsheet (00_Buyer_30_Line_Requirements.xlsx) and click Generate Template-Free RFx. Review or download the generated markdown RFx.
Step 2: Upload your 7 multimodal vendor quotation files (.xlsx, .pdf, .docx, angled .png photo, raw email .txt, CSV, and second PDF).
Step 3: Review the Confidence-Score Audit Flags and test the Iterative Clarification Loop (try clicking Simulate Vague Reply to watch the auditor demand exact rates or previous PO proof).
Step 4: Proceed to Ranked Best Quotes and inspect the 5-parameter vendor comparison matrix (Certificate Validity, Landed Cost in INR, Coverage/Spec Compliance, Incoterms/Freight, and Working Capital impact).
Step 5: Complete the Internal Validation Sign-Off checkbox and click Release & Generate Final Purchase Order.Bonus (For Evaluators): Open the Live Agent State Inspector in the sidebar to inspect active SQLite state checkpoints, fault-tolerant execution history, raw LLM JSON outputs, and the Mermaid architecture diagram.# Quote_killer
PO Generation engine
