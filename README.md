# Quote Consolidation, Audit and Award Analysis

An end-to-end procurement assistant that drafts an RFx, reads vendor quotations in any format (PDF, Word, Excel, images, or plain text), audits each quotation, normalises prices into a single comparison, recommends a line-level award, and answers natural-language questions about the results.

Demo: [[Link]([url](https://quotekiller-ncjrslkibk4s7huuyzenu6.streamlit.app/))]
Walkthrough video: Comming Soon

---

## What it does

1. **RFx drafting.** A conversational co-pilot helps the buyer build the request for quotation. A readiness check confirms that required commercial terms (payment terms, freight basis, delivery location, quote validity, response deadline) are present before the RFx can be issued. The item schedule is appended programmatically, so the model writes prose only and quantities cannot be altered.
2. **Dispatch.** The RFx is sent to each vendor. In this version dispatch is simulated and written to an `outbox` folder.
3. **Multi-format reading.** Vendor submissions are parsed from PDF, DOCX, XLSX, CSV, and images. Pages are rasterised for scanned or photographed documents.
4. **Dual-model extraction.** Every submission is read by two independent models in parallel: a primary Gemini model with fallbacks, and a Groq-hosted model that receives images directly when they are supplied. Results are cached by file digest, prompt, and model, so a changed model never returns a stale result.
5. **Agreement check.** Each quoted line is tagged as **Agreed**, **Disputed**, or **Single read** by comparing the effective price per RFx unit across the two readings. Disputed lines are excluded from award until confirmed.
6. **Audit.** For each vendor the system checks the certificate expiry date, the mandatory questionnaire answers, line coverage, specification compliance, freight basis, currency, extraction confidence, and ambiguities. Each finding receives a confidence score and is either auto-cleared (0.85 or above) or escalated to the buyer.
7. **Clarification loop.** Follow-up questions are generated for each vendor. Vendor replies are screened for prompt injection, then evaluated by the model, which updates prices, certificate dates, freight terms, and questionnaire answers. The loop can repeat across rounds, and every round is recorded.
8. **Normalisation and scoring.** Prices are converted to INR, adjusted for unit-of-measure conversion, line discounts, footnote discounts, and percentage freight. Flat freight is added to the vendor total. Outliers are flagged when a price is more than four times or less than a quarter of the line median.
9. **Award.** DuckDB ranks eligible vendors per line and produces an L1 recommendation with the runner-up, the saving against the runner-up, and a *Verified* or *Provisional* basis.
10. **Analyst.** Questions in plain English (for example, "Which vendors cleared the quality gate and what is the cheapest landed price for line 3?") are converted into read-only SQL, validated, executed in DuckDB with external access disabled, and repaired up to three times if they fail. The result is summarised in formal language with caveats.
11. **Purchase order.** A PO summary is generated only after internal sign-off. Until then it is locked.

The workflow is built with LangGraph. It checkpoints state to SQLite and pauses before scoring and before the PO, so a buyer can review each stage.

---

## Architecture

```
app.py                  Streamlit interface: configuration, RFx co-pilot, uploads, results, analyst
graph_engine.py         LangGraph workflow, extraction, audit, scoring, award, analyst agent
tools_and_models.py     Document parsing, LLM router (Gemini + Groq), prompt-injection scan,
                        caching, RFx readiness checks, dispatch simulation
schemas.py              Pydantic models and LangGraph state definition
```

Processing flow:

```
RFx co-pilot → readiness check → dispatch
      ↓
vendor files → parse → dual-model read (parallel) → agreement tags
      ↓
audit (quality gate, coverage, freight, confidence) → clarification rounds
      ↓
[buyer review] → normalise and score → DuckDB L1 award → analyst Q&A
      ↓
[internal sign-off] → purchase order
```

---

## Models

| Role | Provider | Model (configurable) |
|---|---|---|
| Primary reader, audit evaluation, RFx drafting | Google Gemini | `PRIMARY_GEMINI_MODEL`, with fallbacks `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-2.5-flash` |
| Secondary reader (text) and analyst narration | Groq | `GROQ_TEXT_MODEL`, default `openai/gpt-oss-120b` |
| Secondary reader (images) | Groq | `GROQ_VISION_MODEL`, default `qwen/qwen3.8-27b` |

If a Gemini model fails, the next model in the fallback list is tried automatically. If the Groq key is missing, the second reader is skipped and each line is marked *Single read*.

---

## Setup

### Option 1: Local

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-key"
export GROQ_API_KEY="your-groq-key"
export PRIMARY_GEMINI_MODEL="gemini-2.5-flash"
export GROQ_TEXT_MODEL="openai/gpt-oss-120b"
export GROQ_VISION_MODEL="qwen/qwen3.8-27b"
streamlit run app.py --server.port=8501
```

### Option 2: Docker

```bash
docker build -t quote-killer .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY="your-gemini-key" \
  -e GROQ_API_KEY="your-groq-key" \
  -e PRIMARY_GEMINI_MODEL="gemini-2.5-flash" \
  -e GROQ_TEXT_MODEL="openai/gpt-oss-120b" \
  -e GROQ_VISION_MODEL="qwen/qwen3.8-27b" \
  quote-killer
```

Open http://localhost:7860.

### Option 3: Streamlit Community Cloud

1. Push this repository to GitHub.
2. On share.streamlit.io, select the repository and set the main file to `app.py`.
3. Under **Settings → Secrets**, add `GEMINI_API_KEY`, `GROQ_API_KEY`, `PRIMARY_GEMINI_MODEL`, `GROQ_TEXT_MODEL`, and `GROQ_VISION_MODEL`.

### Option 4: Google Colab

Run the notebook cells top to bottom. The final cell launches Streamlit on port 8501. Expose it with cloudflared or ngrok, and set the API keys as environment variables in the cell before launching.

---

## Configuration

| Variable | Purpose | Example |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini key | |
| `GROQ_API_KEY` | Groq key | |
| `PRIMARY_GEMINI_MODEL` | Primary Gemini model | `gemini-2.5-flash` |
| `GROQ_TEXT_MODEL` | Groq text model | `openai/gpt-oss-120b` |
| `GROQ_VISION_MODEL` | Groq vision model | `qwen/qwen3.8-27b` |
| `QK_CACHE_DIR` | Cache location for extraction results | `/tmp/qk_cache` |

The sidebar also lets you enter keys and the primary model at runtime. The questionnaire and RFx requirements are editable in the interface.

---

## Using the application

1. Enter your API keys and primary model in the sidebar.
2. Draft the RFx in the co-pilot, or upload buyer requirements. Resolve any items flagged by the readiness check.
3. Dispatch the RFx to vendors.
4. Upload vendor submissions. Each file can be in any supported format.
5. Review the audit findings and the review queue. Answer clarification questions as they arise and re-run evaluation.
6. Review the comparison and the line-level award.
7. Use the analyst to ask questions about the results.
8. Record internal sign-off to generate the purchase order.

---

## Logging and observability

Every stage writes to two places: the Streamlit console, and an in-memory event list that can be displayed in the interface.

- **Console:** timestamped lines with stage, thread, and key fields. Look for `[read]`, `[audit]`, `[score]`, `[award]`, `[analyst]`, and `[po]`.
- **Event list:** `get_event_log()` returns all recorded events as dictionaries with timestamp, level, stage, message, and fields such as vendor, reader, model, and line ID.
- **Model reads** record which reader and model answered, how long the call took, and whether the result came from cache.
- **Analyst queries** record the question, every SQL attempt, and each error.
- **Failures** include the vendor, reader, and the error message.

To see which API answered for a given vendor, search the Streamlit console for `vendor=<id>` and `reader=`.

---

## Security

- **Prompt injection.** Vendor submissions and clarification replies are scanned for embedded instructions. Matches are logged, the vendor is excluded from award pending review, and a security block is recorded in the audit trail.
- **Untrusted data.** Model prompts state that vendor content is data, not instructions.
- **Read-only analyst.** Generated SQL must be a single `SELECT` or `WITH` statement and must not contain data-modifying or file-access keywords. External access is disabled in DuckDB.
- **Keys.** API keys are read from environment variables or Streamlit Secrets and are not stored in the repository.

---

## Project structure

```
.
├── app.py
├── graph_engine.py
├── tools_and_models.py
├── schemas.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Limitations

- Dispatch to vendors is simulated. No email is sent.
- Exchange rates are supplied by the buyer, not fetched live.
- Free-tier API quotas can throttle batch runs. Rate-limit errors appear as read failures in the log.
- Extraction quality depends on the source. Angled, blurred, or handwritten images receive capped confidence and are escalated for manual checking.
- The cache lives in a local folder. On hosts where the file system resets on restart, cached results are lost.
- The analyst answers only from the extracted comparison. It cannot access the source documents.

---

## Possible extensions

- Send RFx and clarification requests by email, and parse replies automatically.
- Fetch exchange rates from a live source.
- Persist the audit trail and clarification history to a database.
- Add role-based approval before the PO is released.
- Add evaluation tests with labelled vendor quotations to measure extraction accuracy.

---

## Author

Sashikant Sahal
