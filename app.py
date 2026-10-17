import os
import tempfile
import streamlit as st
import pandas as pd
from tools_and_models import parse_buyer_requirements, DualLLMRouter
from graph_engine import build_procurement_graph

st.set_page_config(page_title="Kill the Quote Spreadsheet | AI Procurement Engine", layout="wide")
st.title("⚡ Autonomous RFx, Dual-LLM Quote Auditor & PO Engine")
st.caption("Dual-LLM Verification (Gemini 3.8 Flash + Groq Llama) • Deterministic Code Normalization • Audit Defense")

with st.sidebar:
    st.header("🔑 API & Model Configuration")
    gem_key = st.text_input("Google Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    groq_key = st.text_input("Groq API Key (Llama)", value=os.getenv("GROQ_API_KEY", ""), type="password")
    primary_gem = st.text_input("Primary Gemini Model", value="gemini-3.8-flash")
    
    if gem_key:
        os.environ["GEMINI_API_KEY"] = gem_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    os.environ["PRIMARY_GEMINI_MODEL"] = primary_gem

    st.divider()
    st.subheader("💱 FX Normalization Rates")
    usd_rate = st.number_input("USD to INR", value=84.0)
    eur_rate = st.number_input("EUR to INR", value=91.0)

if "graph" not in st.session_state:
    st.session_state.graph = build_procurement_graph()
if "thread_config" not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": "demo-procurement-001"}}
if "stage" not in st.session_state:
    st.session_state.stage = 1

# --- STEP 1: Upload Buyer Requirement & Generate RFx ---
st.subheader("1️⃣ Step 1: Upload Buyer Requirements & Generate RFx")
buyer_file = st.file_uploader("Upload Buyer 30-Line Requirement Spreadsheet (.xlsx / .csv)", type=["xlsx", "xls", "csv"])

if buyer_file and st.button("📝 Generate RFx Document", type="primary"):
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner("Analyzing 30 line items and drafting RFx with Gemini 3.8 Flash..."):
            tmp_dir = tempfile.mkdtemp()
            b_path = os.path.join(tmp_dir, buyer_file.name)
            with open(b_path, "wb") as f:
                f.write(buyer_file.getbuffer())
            buyer_reqs = parse_buyer_requirements(b_path)

            initial_state = {
                "rfx_id": "AER-2026-101",
                "reference_date": "2026-10-17",
                "fx_rates": {"INR": 1.0, "USD": usd_rate, "EUR": eur_rate},
                "buyer_requirements": buyer_reqs,
                "rfx_document": "",
                "vendor_files": {},
                "gemini_extractions": {},
                "llama_extractions": {},
                "discrepancy_log": {},
                "clarification_questions": {},
                "vendor_clarifications": {},
                "reconciled_quotes": {},
                "comparison_summary": [],
                "line_level_awards": [],
                "defense_audit_log": [],
                "final_po_document": ""
            }

            # Generate RFx node only
            router = DualLLMRouter()
            prompt = f"""Draft a professional, comprehensive Request for Quotation (RFx) # {initial_state['rfx_id']} dated {initial_state['reference_date']}.
Include scope, commercial terms, and summary of {len(buyer_reqs)} line items."""
            rfx_doc, used_model = router.call_gemini_with_fallback(prompt, schema=None)
            
            st.session_state.rfx_doc = f"*Generated via {used_model}*\n\n" + rfx_doc
            st.session_state.buyer_reqs = buyer_reqs
            st.session_state.stage = 1.5

if "rfx_doc" in st.session_state:
    st.success("RFx successfully generated! This can be automatically dispatched to registered category vendors via email/portal from here, or downloaded below for manual distribution.")
    st.markdown(st.session_state.rfx_doc)
    st.download_button(
        "⬇️ Download RFx Document (.md)",
        data=st.session_state.rfx_doc,
        file_name="RFx_Document.md",
        mime="text/markdown"
    )
    st.divider()

# --- STEP 2: Upload 5 Vendor Bids & Run Audit ---
st.subheader("2️⃣ Step 2: Submit 5 Demo Bids (Multimodal Files)")
vendor_uploads = st.file_uploader(
    "Upload Vendor Quotes (Excel, PDF, Word doc, angled Photo .png/.jpg, Email .txt)",
    type=["pdf", "docx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "txt"],
    accept_multiple_files=True
)

if vendor_uploads and st.button("🚀 Process Vendor Bids & Run Dual-LLM Audit", type="primary"):
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("GROQ_API_KEY"):
        st.error("Please enter both your Gemini API Key and Groq API Key in the sidebar.")
    else:
        with st.spinner("Extracting quotes across Gemini 3.8 Flash & Groq Llama, and running audit diffs..."):
            tmp_dir = tempfile.mkdtemp()
            v_files = {}
            for vf in vendor_uploads:
                v_path = os.path.join(tmp_dir, vf.name)
                with open(v_path, "wb") as f:
                    f.write(vf.getbuffer())
                v_name = os.path.splitext(vf.name)[0]
                v_files[v_name] = v_path

            state_update = {
                "rfx_id": "AER-2026-101",
                "reference_date": "2026-10-17",
                "fx_rates": {"INR": 1.0, "USD": usd_rate, "EUR": eur_rate},
                "buyer_requirements": st.session_state.buyer_reqs,
                "rfx_document": st.session_state.rfx_doc,
                "vendor_files": v_files,
            }

            # Run graph up to breakpoint
            for _ in st.session_state.graph.stream(state_update, st.session_state.thread_config):
                pass
            st.session_state.stage = 2
            st.success("Vendor quotes processed and audited!")

current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

# --- STEP 3 & 4: Clarification & PO ---
if st.session_state.stage >= 2 and state_vals.get("clarification_questions"):
    st.divider()
    st.subheader("3️⃣ Dual-LLM Audit Discrepancies & Vendor Clarifications")
    
    col_d, col_q = st.columns([1, 1])
    with col_d:
        st.markdown("**🔍 Audit Flags (Gemini vs. Llama)**")
        for v_id, logs in state_vals.get("discrepancy_log", {}).items():
            with st.expander(f"Flags: {v_id}", expanded=True):
                for l in logs:
                    st.write(f"- {l}")

    with col_q:
        st.markdown("**📨 Vendor Clarification Questionnaire**")
        if st.button("🤖 Simulate Vendor Replies"):
            router = DualLLMRouter()
            sim_replies = {}
            for v_id, qs in state_vals["clarification_questions"].items():
                prompt = f"Act as vendor '{v_id}'. Write a concise 2-sentence commercial reply confirming pricing and valid ISO renewal:\n" + "\n".join(qs)
                reply, _ = router.call_gemini_with_fallback(prompt, schema=None)
                sim_replies[v_id] = reply
            st.session_state.sim_replies = sim_replies

        user_edited_replies = {}
        for v_id, qs in state_vals["clarification_questions"].items():
            st.markdown(f"**To {v_id}:**")
            for q in qs:
                st.caption(f"❓ {q}")
            default_rep = st.session_state.get("sim_replies", {}).get(v_id, "We confirm our quoted rates and valid ISO renewal.")
            user_edited_replies[v_id] = st.text_area(f"Vendor Reply ({v_id})", value=default_rep, key=f"rep_{v_id}", height=70)

    if st.button("⚖️ Calculate Landed Costs & Draft PO", type="primary"):
        with st.spinner("Normalizing prices, checking 5 parameters, and drafting PO..."):
            st.session_state.graph.update_state(st.session_state.thread_config, {"vendor_clarifications": user_edited_replies})
            for _ in st.session_state.graph.stream(None, st.session_state.thread_config):
                pass
            st.session_state.stage = 3

if st.session_state.stage >= 3 and state_vals.get("comparison_summary"):
    st.divider()
    st.subheader("4️⃣ 5-Parameter Vendor Comparison & Draft PO")
    st.dataframe(pd.DataFrame(state_vals["comparison_summary"]), use_container_width=True)
    st.dataframe(pd.DataFrame(state_vals["line_level_awards"]), use_container_width=True)
    st.markdown(state_vals.get("final_po_document", ""))
    st.download_button("⬇️ Download Purchase Order (.md)", data=state_vals.get("final_po_document", ""), file_name="Purchase_Order.md")

    # --- DEMO SHOWCASE EXPANDER: Live Agent State & Graph Inspection ---
with st.sidebar:
    st.divider()
    show_inspector = st.checkbox("🛠️ Show Live Agent Inspector", value=False)

if show_inspector:
    st.divider()
    st.subheader("🛠️ Live Agent Architecture & State Inspector (Demo Mode)")
    
    tab_state, tab_history, tab_graph = st.tabs(["📦 Active State Checkpoint", "📜 Thread History & Fault Tolerance", "🗺️ Agent Graph Workflow"])
    
    with tab_state:
        st.caption("Inspects the exact typed `TypedDict` state stored in the SQLite checkpoint database at `/tmp/procurement_state.db`.")
        current_snap = st.session_state.graph.get_state(st.session_state.thread_config)
        if current_snap:
            st.write("**Next Paused / Target Node:**", current_snap.next)
            st.write("**Current Checkpoint ID:**", current_snap.config.get("configurable", {}).get("checkpoint_id"))
            
            # Show safely filterable key-value summary
            vals = current_snap.values
            if vals:
                st.write("**Stored Data Keys:**", list(vals.keys()))
                if "buyer_requirements" in vals and vals["buyer_requirements"]:
                    st.info(f"Loaded {len(vals['buyer_requirements'])} Buyer Line Items in state.")
                if "vendor_files" in vals and vals["vendor_files"]:
                    st.info(f"Active Vendor Files in State: {list(vals['vendor_files'].keys())}")
        else:
            st.warning("No active state checkpoint found yet. Run an E2E query first.")

    with tab_history:
        st.caption("Demonstrates fault-tolerance: shows that every node invocation is checkpointed so the system can resume from any failure.")
        try:
            history_list = list(st.session_state.graph.get_state_history(st.session_state.thread_config))
            for idx, h in enumerate(history_list):
                step_num = h.metadata.get("step", idx)
                next_node = h.next
                chk_id = h.config.get("configurable", {}).get("checkpoint_id", "")[:8]
                st.markdown(f"**Step {step_num}** | Checkpoint: `{chk_id}` | Next Node: `{next_node}`")
        except Exception as e:
            st.info("Execute Step 1 & 2 to populate checkpoint history.")

    with tab_graph:
        st.caption("Visual representation of the LangGraph State Machine powering your procurement pipeline.")
        st.markdown("""
        ```mermaid
        graph TD
            A[create_rfx_node] --> B[dual_extract_quotes_node]
            B --> C[diff_and_questionnaire_node]
            C --> D{{"INTERRUPT / BREAKPOINT\n(Human Clarification Loop)"}}
            D --> E[reconcile_and_score_node\n(DuckDB 5-Parameter Math)]
            E --> F[generate_po_node]
            F --> G([END])
        ```
        """)
