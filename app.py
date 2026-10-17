import os
import tempfile
import streamlit as st
import pandas as pd
from tools_and_models import parse_buyer_requirements, DualLLMRouter
from graph_engine import build_procurement_graph

st.set_page_config(page_title="Kill the Quote Spreadsheet | AI Procurement Engine", layout="wide")
st.title("⚡ Autonomous RFx, Dual-LLM Quote Auditor & PO Engine")
st.caption("Dual-LLM Verification (Gemini Flash w/ Family Fallback + Groq Llama 4) • Deterministic Code Normalization • Audit Defense")

# Sidebar API Keys & Config
with st.sidebar:
    st.header("🔑 API & Model Configuration")
    gem_key = st.text_input("Google Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    groq_key = st.text_input("Groq API Key (Llama 4)", value=os.getenv("GROQ_API_KEY", ""), type="password")
    primary_gem = st.text_input("Primary Gemini Model", value="gemini-2.5-flash")
    st.info("Fallback Chain: `gemini-2.5-pro` → `gemini-2.5-flash-lite`\nSecondary Model: `meta-llama/llama-4-scout-17b-16e-instruct`")

    if gem_key:
        os.environ["GEMINI_API_KEY"] = gem_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    os.environ["PRIMARY_GEMINI_MODEL"] = primary_gem

    st.divider()
    st.subheader("💱 FX Normalization Rates (to INR)")
    usd_rate = st.number_input("USD to INR", value=84.0)
    eur_rate = st.number_input("EUR to INR", value=91.0)

if "graph" not in st.session_state:
    st.session_state.graph = build_procurement_graph()
if "thread_config" not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": "demo-procurement-001"}}
if "stage" not in st.session_state:
    st.session_state.stage = 1

# --- STEP 1 & 2: Upload Buyer Requirement & 5 Vendor Quotes ---
st.subheader("1️⃣ Upload Buyer Requirements & Vendor Quotes")
col1, col2 = st.columns([1, 1])

with col1:
    buyer_file = st.file_uploader("Upload Buyer 30-Line Requirement (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
with col2:
    vendor_uploads = st.file_uploader(
        "Upload 5 Vendor Quotes (Word .docx, PDF, Angled Image .png/.jpg, Excel, .txt)",
        type=["pdf", "docx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True
    )

if st.button("🚀 Generate RFx, Extract Quotes (Dual-LLM) & Audit Ambiguities", type="primary"):
    if not buyer_file or not vendor_uploads:
        st.error("Please upload both the Buyer Requirement spreadsheet and the Vendor Quote files.")
    elif not os.getenv("GEMINI_API_KEY") or not os.getenv("GROQ_API_KEY"):
        st.error("Please enter both your Gemini API Key and Groq API Key in the sidebar.")
    else:
        with st.spinner("Parsing 30 line items, drafting RFx, and running parallel Gemini + Llama 4 extractions..."):
            tmp_dir = tempfile.mkdtemp()
            b_path = os.path.join(tmp_dir, buyer_file.name)
            with open(b_path, "wb") as f:
                f.write(buyer_file.getbuffer())
            buyer_reqs = parse_buyer_requirements(b_path)

            v_files = {}
            for idx, vf in enumerate(vendor_uploads):
                v_path = os.path.join(tmp_dir, vf.name)
                with open(v_path, "wb") as f:
                    f.write(vf.getbuffer())
                v_name = os.path.splitext(vf.name)[0]
                v_files[v_name] = v_path

            initial_state = {
                "rfx_id": "AER-2026-101",
                "reference_date": "2026-10-17",
                "fx_rates": {"INR": 1.0, "USD": usd_rate, "EUR": eur_rate},
                "buyer_requirements": buyer_reqs,
                "rfx_document": "",
                "vendor_files": v_files,
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

            # Run graph up to the human-in-the-loop questionnaire breakpoint
            for _ in st.session_state.graph.stream(initial_state, st.session_state.thread_config):
                pass
            st.session_state.stage = 2
            st.success("RFx generated, quotes extracted by both LLMs, and ambiguity questionnaires prepared!")

# Fetch current state from SQLite checkpoint
current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if state_vals.get("rfx_document"):
    with st.expander("📄 View Generated RFx & Parsed Buyer Line Items", expanded=False):
        st.markdown(state_vals["rfx_document"])
        st.dataframe(pd.DataFrame(state_vals["buyer_requirements"]), use_container_width=True)

# --- STEP 3: Dual-LLM Diffs & Vendor Clarification Loop ---
if st.session_state.stage >= 2 and state_vals.get("clarification_questions"):
    st.divider()
    st.subheader("2️⃣ Dual-LLM Audit Discrepancies & Vendor Clarification Questionnaire")

    col_d, col_q = st.columns([1, 1])
    with col_d:
        st.markdown("**🔍 Code-Verified Discrepancies (Gemini vs. Llama 4 & Spec Checks)**")
        for v_id, logs in state_vals.get("discrepancy_log", {}).items():
            with st.expander(f"Audit Flags: {v_id}", expanded=True):
                for l in logs:
                    st.write(f"- {l}")

    with col_q:
        st.markdown("**📨 Auto-Generated Vendor Clarification Questionnaires**")
        if st.button("🤖 Simulate Realistic Vendor Replies (Auto-Fill)"):
            router = DualLLMRouter()
            sim_replies = {}
            for v_id, qs in state_vals["clarification_questions"].items():
                prompt = f"Act as vendor '{v_id}'. Write a concise 2-sentence commercial email reply clarifying these buyer questions (confirming exact unit prices, valid ISO renewal or stating inability to meet spec where applicable):\n" + "\n".join(qs)
                reply, _ = router.call_gemini_with_fallback(prompt, schema=None)
                sim_replies[v_id] = reply
            st.session_state.sim_replies = sim_replies

        user_edited_replies = {}
        for v_id, qs in state_vals["clarification_questions"].items():
            st.markdown(f"**To {v_id}:**")
            for q in qs:
                st.caption(f"❓ {q}")
            default_rep = st.session_state.get("sim_replies", {}).get(
                v_id, "We confirm our quoted rates, renewed ISO certificate valid till 2027, and standard conversion factors."
            )
            user_edited_replies[v_id] = st.text_area(f"Vendor Reply ({v_id})", value=default_rep, key=f"rep_{v_id}", height=80)

    if st.button("⚖️ Run Deterministic Price Comparison, Defense Audit & Draft PO", type="primary"):
        with st.spinner("Updating state with vendor clarifications, running DuckDB 30-line normalization, and generating PO..."):
            st.session_state.graph.update_state(
                st.session_state.thread_config,
                {"vendor_clarifications": user_edited_replies}
            )
            for _ in st.session_state.graph.stream(None, st.session_state.thread_config):
                pass
            st.session_state.stage = 3

# --- STEP 4: Final 5-Parameter Scorecard, Defense Log & PO Output ---
current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if st.session_state.stage >= 3 and state_vals.get("comparison_summary"):
    st.divider()
    st.subheader("3️⃣ 5-Parameter Vendor Comparison Matrix")
    st.dataframe(pd.DataFrame(state_vals["comparison_summary"]), use_container_width=True)

    st.subheader("4️⃣ Line-by-Line Normalized Award Table (30 Lines)")
    st.dataframe(pd.DataFrame(state_vals["line_level_awards"]), use_container_width=True)

    col_def, col_po = st.columns([1, 1])
    with col_def:
        st.subheader("🛡️ Code-Verified Decision Defense")
        for d in state_vals.get("defense_audit_log", []):
            st.info(d)

        with st.expander("🔬 View Raw Gemini vs. Llama 4 JSON Extractions"):
            st.json({
                "gemini": state_vals.get("gemini_extractions"),
                "llama4": state_vals.get("llama_extractions")
            })

    with col_po:
        st.subheader("📜 Final Purchase Order (PO)")
        st.markdown(state_vals.get("final_po_document", ""))
        st.download_button(
            "⬇️ Download Purchase Order (.md)",
            data=state_vals.get("final_po_document", ""),
            file_name="Final_Purchase_Order.md",
            mime="text/markdown"
        )
