import os
import tempfile
import streamlit as st
import pandas as pd
from tools_and_models import parse_buyer_requirements, DualLLMRouter
from graph_engine import build_procurement_graph, create_rfx_node, evaluate_vendor_clarifications

st.set_page_config(page_title="Kill the Quote Spreadsheet | AI Procurement Engine", layout="wide")
st.title("⚡ Autonomous RFx, Dual-LLM Quote Auditor & Gated PO Engine")
st.caption("Gemini 3.8 Flash (w/ 3.1 & 3.5 Flash-Lite Fallbacks) + Groq Llama • Confidence Escalation • Prompt Injection Guardrails")

with st.sidebar:
    st.header("🔑 API & Model Configuration")
    gem_key = st.text_input("Google Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    groq_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    primary_gem = st.text_input("Primary Gemini Model", value="gemini-3.8-flash")
    st.info("Fallbacks: `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite`\nSecondary: `llama-3.3-70b-versatile`")

    if gem_key:
        os.environ["GEMINI_API_KEY"] = gem_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    os.environ["PRIMARY_GEMINI_MODEL"] = primary_gem

    st.divider()
    st.subheader("💱 FX Normalization Rates")
    usd_rate = st.number_input("USD to INR", value=84.0)
    eur_rate = st.number_input("EUR to INR", value=91.0)

    st.divider()
    show_inspector = st.checkbox("🛠️ Show Live Agent State & Graph Inspector", value=True)

if "graph" not in st.session_state:
    st.session_state.graph = build_procurement_graph()
if "thread_config" not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": "demo-procurement-v3"}}
if "stage" not in st.session_state:
    st.session_state.stage = 1

# ==============================================================================
# STEP 1: Upload Buyer Requirements & Generate Template-Free RFx
# ==============================================================================
st.subheader("1️⃣ Step 1: Upload 30-Line Requirement & Generate Template-Free RFx")
buyer_file = st.file_uploader("Upload Buyer 30-Line Requirement Spreadsheet (.xlsx / .csv)", type=["xlsx", "xls", "csv"])

if buyer_file and st.button("📝 Generate Template-Free RFx Document", type="primary"):
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner("Scanning buyer file for prompt injections and drafting template-free RFx..."):
            tmp_dir = tempfile.mkdtemp()
            b_path = os.path.join(tmp_dir, buyer_file.name)
            with open(b_path, "wb") as f:
                f.write(buyer_file.getbuffer())

            buyer_reqs, sec_flags = parse_buyer_requirements(b_path)
            st.session_state.buyer_reqs = buyer_reqs
            st.session_state.security_alerts = sec_flags

            rfx_res = create_rfx_node({
                "rfx_id": "AER-2026-101",
                "reference_date": "2026-10-17",
                "buyer_requirements": buyer_reqs
            })
            st.session_state.rfx_doc = rfx_res["rfx_document"]
            st.session_state.stage = 1.5

if st.session_state.get("security_alerts"):
    for alert in st.session_state.security_alerts:
        st.error(f"🛡️ PROMPT INJECTION GUARDRAIL TRIGGERED: {alert}")

if "rfx_doc" in st.session_state:
    st.success(
        "📢 **RFx Ready for Dispatch:** This RFx can be automatically dispatched to all registered vendors in the concerned category directly from this platform. "
        "**For demo purposes, please download the RFx below and submit your 5+ multimodal vendor bids in Step 2.** "
        "*(Notice: Vendors are NOT forced into any standard template—Word, PDF, photos, and emails are actively encouraged!)*"
    )
    with st.expander("📄 Preview Generated Template-Free RFx Document", expanded=True):
        st.markdown(st.session_state.rfx_doc)
    st.download_button(
        "⬇️ Download RFx Document (.md)",
        data=st.session_state.rfx_doc,
        file_name="RFx_AER_2026_101.md",
        mime="text/markdown"
    )
    st.divider()

# ==============================================================================
# STEP 2: Submit Multimodal Vendor Quotes & Run Dual-LLM + Security Audit
# ==============================================================================
if st.session_state.stage >= 1.5:
    st.subheader("2️⃣ Step 2: Submit Vendor Bids (Word, PDF, Angled Photo, Excel, Email)")
    vendor_uploads = st.file_uploader(
        "Upload 5+ Vendor Quotes in Any Format (.docx, .pdf, .png/.jpg, .xlsx, .csv, .txt)",
        type=["pdf", "docx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True
    )

    if vendor_uploads and st.button("🚀 Extract Quotes (Dual-LLM), Compute Confidence & Audit", type="primary"):
        if not os.getenv("GEMINI_API_KEY") or not os.getenv("GROQ_API_KEY"):
            st.error("Please enter both your Gemini API Key and Groq API Key in the sidebar.")
        else:
            with st.spinner("Running Gemini 3.8 Flash + Groq Llama extractions, prompt injection scans, and confidence scoring..."):
                tmp_dir = tempfile.mkdtemp()
                v_files = {}
                for vf in vendor_uploads:
                    v_path = os.path.join(tmp_dir, vf.name)
                    with open(v_path, "wb") as f:
                        f.write(vf.getbuffer())
                    v_files[os.path.splitext(vf.name)[0]] = v_path

                initial_state = {
                    "rfx_id": "AER-2026-101",
                    "reference_date": "2026-10-17",
                    "fx_rates": {"INR": 1.0, "USD": usd_rate, "EUR": eur_rate},
                    "buyer_requirements": st.session_state.buyer_reqs,
                    "rfx_document": st.session_state.rfx_doc,
                    "security_alerts": st.session_state.get("security_alerts", []),
                    "vendor_files": v_files,
                    "gemini_extractions": {},
                    "llama_extractions": {},
                    "discrepancy_log": {},
                    "clarification_questions": {},
                    "clarification_round": 0,
                    "clarification_history": [],
                    "vendor_clarifications": {},
                    "reconciled_quotes": {},
                    "comparison_summary": [],
                    "line_level_awards": [],
                    "defense_audit_log": [],
                    "internal_signoff_approved": False,
                    "internal_approver_name": "",
                    "final_po_document": ""
                }

                # Streams up to Interrupt #1 (before reconcile_and_score_node)
                for _ in st.session_state.graph.stream(initial_state, st.session_state.thread_config):
                    pass
                st.session_state.stage = 2
                st.rerun()

# Pull live state from SQLite Checkpointer
current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if state_vals.get("security_alerts"):
    for alert in state_vals["security_alerts"]:
        st.error(f"🚨 PROMPT INJECTION DETECTED & BLOCKED: {alert}")

# ==============================================================================
# STEP 3: Confidence-Scored Flags & Iterative Multi-Turn Clarification Loop
# ==============================================================================
if st.session_state.stage >= 2 and state_vals.get("discrepancy_log"):
    st.divider()
    st.subheader(f"3️⃣ Step 3: Confidence-Scored Audit Flags & Iterative Vendor Clarification (Round {state_vals.get('clarification_round', 1)})")

    col_d, col_q = st.columns([1.15, 1])

    with col_d:
        st.markdown("**🔍 Deduplicated Issues, Confidence Scores & Human Escalation Routing**")
        st.caption("High Confidence (≥ 0.85) auto-passes. Medium/Low Confidence (< 0.85) or LLM disagreements escalate to human review.")
        for v_id, issue_list in state_vals.get("discrepancy_log", {}).items():
            with st.expander(f"Vendor Audit: {v_id} ({len(issue_list)} flags)", expanded=True):
                if not issue_list:
                    st.success("✅ 100% In Sync (High Confidence ≥ 0.95). Zero discrepancies found.")
                else:
                    df_iss = pd.DataFrame(issue_list)
                    st.dataframe(df_iss, use_container_width=True, hide_index=True)

    with col_q:
        st.markdown("**📨 Selective & Iterative Vendor Clarification Loop**")
        st.caption("Vendors with 0 issues receive NO questions. Vague replies (e.g., 'same price as last bid' without numbers/PO doc) trigger follow-up questions!")

        c_questions = state_vals.get("clarification_questions", {})
        active_vendors_needing_clarification = {v: qs for v, qs in c_questions.items() if len(qs) > 0}

        if not active_vendors_needing_clarification:
            st.success("🎉 All vendors are 100% in sync or all clarifications have been resolved with concrete numbers!")
        else:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🤖 Simulate Vague Reply (Test Follow-Up Loop)"):
                    vague_map = {}
                    for v_id in active_vendors_needing_clarification:
                        vague_map[v_id] = "The missing items not quoted for will have the same price that I quoted in my last bid for the same items."
                    st.session_state.draft_replies = vague_map
            with col_b2:
                if st.button("✅ Simulate Concrete Reply (With Rates & PO Ref)"):
                    concrete_map = {}
                    for v_id in active_vendors_needing_clarification:
                        concrete_map[v_id] = (
                            "Per previous PO #PO-2025-889 (attached), we confirm exact rates: "
                            "Line 14 @ INR 17.50, Line 15 @ INR 15.00, Line 28 @ INR 112.00, Line 29 @ INR 960.00, Line 30 @ INR 4.10. "
                            "Our ISO/FSC certificate has been renewed and is valid until 2028-06-30. Freight is fixed at 6.0%."
                        )
                    st.session_state.draft_replies = concrete_map

        user_edited_replies = {}
        for v_id, qs in c_questions.items():
            if not qs:
                st.success(f"**{v_id}:** ✅ Everything in sync — No clarification required.")
            else:
                st.markdown(f"**Questions Sent to `{v_id}`:**")
                for q in qs:
                    st.warning(f"❓ {q}")
                default_val = st.session_state.get("draft_replies", {}).get(v_id, "")
                user_edited_replies[v_id] = st.text_area(
                    f"Vendor Response from {v_id}:",
                    value=default_val,
                    placeholder="Type vendor reply here (try writing 'same price as last bid' to see the LLM ask for exact rates/PO proof)...",
                    key=f"reply_{v_id}_{state_vals.get('clarification_round', 1)}",
                    height=85
                )

        if active_vendors_needing_clarification:
            if st.button("🔄 Evaluate Vendor Replies (Check If Follow-Up Needed)", type="secondary"):
                with st.spinner("Auditing vendor replies for vague references, concrete rates, and prompt injections..."):
                    updates = evaluate_vendor_clarifications(state_vals, user_edited_replies)
                    st.session_state.graph.update_state(st.session_state.thread_config, updates)
                    st.rerun()

        # Show Multi-Turn Clarification History if any rounds occurred
        if state_vals.get("clarification_history"):
            with st.expander("💬 View Multi-Turn Clarification History", expanded=True):
                for h in state_vals["clarification_history"]:
                    status_icon = "✅ Resolved" if h["is_resolved"] else "⚠️ Incomplete (Follow-Up Triggered)"
                    st.markdown(f"**Round {h['round']} — {h['vendor']} ({status_icon}):**")
                    st.write(f"- *Vendor Said:* \"{h['vendor_reply']}\"")
                    if h["follow_up_generated"]:
                        st.error(f"- *Auditor Follow-Up:* {h['follow_up_generated'][0]}")

    st.divider()
    if st.button("⚖️ Proceed to Code Price Comparison & Generate Best Quote Recommendations", type="primary"):
        with st.spinner("Running DuckDB 5-parameter normalization & ranking best quotes (PO held pending internal validation)..."):
            # Resume graph from Interrupt #1 -> runs reconcile_and_score_node and pauses at Interrupt #2 (before generate_po_node)
            for _ in st.session_state.graph.stream(None, st.session_state.thread_config):
                pass
            st.session_state.stage = 3
            st.rerun()

# ==============================================================================
# STEP 4: Best Quote Recommendations first -> Internal Validation Gate -> PO
# ==============================================================================
current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if st.session_state.stage >= 3 and state_vals.get("comparison_summary"):
    st.divider()
    st.subheader("4️⃣ Step 4: Ranked Best Quotes & 5-Parameter Evaluation Matrix")
    st.info("🔒 **Staged Governance Policy:** Only the recommended best quotes and mathematical audit defense are shown below. Purchase Order (PO) generation is held pending internal stakeholder validation.")

    st.markdown("**📊 Vendor-Level 5-Parameter Comparison Scorecard**")
    st.dataframe(pd.DataFrame(state_vals["comparison_summary"]), use_container_width=True, hide_index=True)

    st.markdown("**🏆 Recommended Best Quotes per Line Item (30 Lines Normalized to DDP INR)**")
    st.dataframe(pd.DataFrame(state_vals["line_level_awards"]), use_container_width=True, hide_index=True)

    with st.expander("🛡️ View Deterministic Code & Dual-LLM Defense Ledger", expanded=True):
        for entry in state_vals.get("defense_audit_log", []):
            st.write(f"- {entry}")

    st.divider()
    st.subheader("5️⃣ Step 5: Internal Validation Gate & Final PO Release")

    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        approver_name = st.text_input("Internal Approver Name & Title", value="Category Head - Packaging Procurement")
        is_approved = st.checkbox("✅ I have reviewed the Dual-LLM audit flags, escalated items, and approve the Best Quote recommendation for PO issuance.")

    with col_g2:
        if st.button("📜 Release & Generate Final Purchase Order (PO)", type="primary", disabled=not is_approved):
            with st.spinner("Recording internal sign-off in SQLite state and releasing Purchase Order..."):
                st.session_state.graph.update_state(
                    st.session_state.thread_config,
                    {
                        "internal_signoff_approved": True,
                        "internal_approver_name": approver_name
                    }
                )
                # Resume from Interrupt #2 -> executes generate_po_node
                for _ in st.session_state.graph.stream(None, st.session_state.thread_config):
                    pass
                st.session_state.stage = 4
                st.rerun()

current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if st.session_state.stage >= 4 and state_vals.get("final_po_document"):
    st.success("✅ Internal Validation Complete — Final Purchase Order Generated!")
    st.markdown(state_vals["final_po_document"])
    st.download_button(
        "⬇️ Download Approved Purchase Order (.md)",
        data=state_vals["final_po_document"],
        file_name="Approved_Purchase_Order.md",
        mime="text/markdown"
    )

# ==============================================================================
# LIVE DEMO INSPECTOR: Showcase State Checkpoints, Agent Graph & Raw LLM Outputs
# ==============================================================================
if show_inspector:
    st.divider()
    st.subheader("🛠️ Live Agent Architecture, SQLite State & Checkpoint Inspector (For Demo)")
    tab_state, tab_history, tab_raw, tab_graph = st.tabs([
        "📦 Active State Checkpoint",
        "📜 Fault-Tolerance Checkpoint History",
        "🔬 Raw Gemini vs. Llama JSON",
        "🗺️ LangGraph State Machine"
    ])

    with tab_state:
        if current_snapshot:
            st.write("**Next Scheduled Graph Node (Breakpoint):**", current_snapshot.next)
            st.write("**Active SQLite Checkpoint ID:**", current_snapshot.config.get("configurable", {}).get("checkpoint_id"))
            st.write("**Clarification Iteration Round:**", state_vals.get("clarification_round", 0))
            st.write("**Internal Sign-Off Status:**", state_vals.get("internal_signoff_approved", False))
            st.write("**State Keys Persisted in `/tmp/procurement_state.db`:**", list(state_vals.keys()))

    with tab_history:
        try:
            history_list = list(st.session_state.graph.get_state_history(st.session_state.thread_config))
            for idx, h in enumerate(history_list):
                chk_id = h.config.get("configurable", {}).get("checkpoint_id", "")[:10]
                st.markdown(f"- **Checkpoint `{chk_id}`** | Next Node: `{h.next}` | Step: `{h.metadata.get('step', idx)}`")
        except Exception:
            st.info("Run Step 2 to populate SQLite checkpoint history.")

    with tab_raw:
        st.json({
            "Gemini_3.8_Flash_Extractions": state_vals.get("gemini_extractions", {}),
            "Groq_Llama_Extractions": state_vals.get("llama_extractions", {})
        })

    with tab_graph:
        st.markdown("""
        ```mermaid
        graph TD
            A[create_rfx_node\nTemplate-Free RFx] --> B[dual_extract_quotes_node\nGemini 3.8 Flash + Groq Llama + Injection Guard]
            B --> C[diff_and_questionnaire_node\nDeduplicated Ranges + Confidence Escalation]
            C --> D{{"⏸️ INTERRUPT 1: Multi-Turn Clarification Loop\n(Catches 'Same as Last Bid' & Demands Rates/PO)"}}
            D -->|Follow-Up Needed| D
            D -->|Clarifications Resolved| E[reconcile_and_score_node\nDuckDB 5-Parameter Best Quotes]
            E --> F{{"⏸️ INTERRUPT 2: Internal Validation Gate\n(Best Quotes Shown, PO Held)"}}
            F -->|Approved| G[generate_po_node\nFinal PO Released]
            G --> H([END])
        ```
        """)
