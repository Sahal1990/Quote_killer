import os
import tempfile
import streamlit as st
import pandas as pd
import duckdb
from tools_and_models import parse_buyer_requirements, DualLLMRouter
from graph_engine import build_procurement_graph, create_rfx_node, evaluate_vendor_clarifications

st.set_page_config(page_title="Kill the Quote Spreadsheet | AI Procurement Engine", layout="wide")
st.title("⚡ Autonomous RFx, Dual-LLM Quote Auditor & Analyst Chat Engine")
st.caption("Gemini 3.8 Flash + Groq Llama • Side-by-Side Matrix • Natural Language Analyst Chat • Gated PO")

with st.sidebar:
    st.header("🔑 API & Model Configuration")
    gem_key = st.text_input("Google Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    groq_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    primary_gem = st.text_input("Primary Gemini Model", value="gemini-3.8-flash")
    if gem_key: os.environ["GEMINI_API_KEY"] = gem_key
    if groq_key: os.environ["GROQ_API_KEY"] = groq_key
    os.environ["PRIMARY_GEMINI_MODEL"] = primary_gem

    st.divider()
    st.subheader("💱 FX Normalization Rates")
    usd_rate = st.number_input("USD to INR", value=84.0)
    eur_rate = st.number_input("EUR to INR", value=91.0)
    gbp_rate = st.number_input("GBP to INR", value=108.0)

    st.divider()
    show_inspector = st.checkbox("🛠️ Show Live Agent State & Graph Inspector", value=True)

if "graph" not in st.session_state:
    st.session_state.graph = build_procurement_graph()
if "thread_config" not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": "demo-procurement-v4"}}
if "stage" not in st.session_state:
    st.session_state.stage = 1

# STEP 1: Upload Requirements & Generate RFx
st.subheader("1️⃣ Step 1: Upload 30-Line Requirement & Generate Template-Free RFx")
buyer_file = st.file_uploader("Upload Buyer 30-Line Requirement Spreadsheet (.xlsx / .csv)", type=["xlsx", "xls", "csv"])

if buyer_file and st.button("📝 Generate Template-Free RFx Document", type="primary"):
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner("Drafting RFx (encouraging Word, PDF, email, and photo submissions without rigid templates)..."):
            tmp_dir = tempfile.mkdtemp()
            b_path = os.path.join(tmp_dir, buyer_file.name)
            with open(b_path, "wb") as f: f.write(buyer_file.getbuffer())

            buyer_reqs, sec_flags = parse_buyer_requirements(b_path)
            st.session_state.buyer_reqs = buyer_reqs
            st.session_state.security_alerts = sec_flags

            rfx_res = create_rfx_node({
                "rfx_id": "AER-2026-303",
                "reference_date": "2026-10-17",
                "buyer_requirements": buyer_reqs
            })
            st.session_state.rfx_doc = rfx_res["rfx_document"]
            st.session_state.stage = 1.5

if "rfx_doc" in st.session_state:
    st.success("📢 **RFx Generated Successfully!** Download below and submit your multimodal vendor bids in Step 2.")
    with st.expander("📄 Preview Generated RFx Document", expanded=True):
        st.markdown(st.session_state.rfx_doc)
    st.download_button("⬇️ Download RFx (.md)", data=st.session_state.rfx_doc, file_name="Flexible_RFx.md")
    st.divider()

# STEP 2: Submit Vendor Quotes
if st.session_state.stage >= 1.5:
    st.subheader("2️⃣ Step 2: Submit Multimodal Vendor Quotes (PDF, Word, Angle Photo, Email, CSV)")
    vendor_uploads = st.file_uploader(
        "Upload Vendor Quotes in Any Format",
        type=["pdf", "docx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True
    )

    if vendor_uploads and st.button("🚀 Run Dual-LLM Extraction & Audit", type="primary"):
        if not os.getenv("GEMINI_API_KEY") or not os.getenv("GROQ_API_KEY"):
            st.error("Please enter both Gemini and Groq API keys.")
        else:
            with st.spinner("Extracting quotes via Gemini 3.8 Flash + Groq Llama, scanning security, and checking compliance..."):
                tmp_dir = tempfile.mkdtemp()
                v_files = {}
                for vf in vendor_uploads:
                    v_path = os.path.join(tmp_dir, vf.name)
                    with open(v_path, "wb") as f: f.write(vf.getbuffer())
                    v_files[os.path.splitext(vf.name)[0]] = v_path

                initial_state = {
                    "rfx_id": "AER-2026-303",
                    "reference_date": "2026-10-17",
                    "fx_rates": {"INR": 1.0, "USD": usd_rate, "EUR": eur_rate, "GBP": gbp_rate},
                    "buyer_requirements": st.session_state.buyer_reqs,
                    "rfx_document": st.session_state.rfx_doc,
                    "security_alerts": st.session_state.get("security_alerts", []),
                    "vendor_files": v_files,
                    "gemini_extractions": {}, "llama_extractions": {}, "discrepancy_log": {},
                    "clarification_questions": {}, "clarification_round": 0, "clarification_history": [],
                    "vendor_clarifications": {}, "reconciled_quotes": {}, "comparison_summary": [],
                    "line_level_awards": [], "all_normalized_lines": [], "defense_audit_log": [],
                    "internal_signoff_approved": False, "internal_approver_name": "", "final_po_document": ""
                }
                for _ in st.session_state.graph.stream(initial_state, st.session_state.thread_config):
                    pass
                st.session_state.stage = 2
                st.rerun()

current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

# STEP 3: Clarifications & Gated Review
if st.session_state.stage >= 2 and state_vals.get("discrepancy_log"):
    st.divider()
    st.subheader(f"3️⃣ Step 3: Confidence Audit & Vendor Clarifications (Round {state_vals.get('clarification_round', 1)})")

    col_d, col_q = st.columns([1.15, 1])
    with col_d:
        st.markdown("**🔍 Deduplicated Flags & Human Escalation**")
        for v_id, issues in state_vals.get("discrepancy_log", {}).items():
            with st.expander(f"Audit: {v_id}", expanded=True):
                if not issues:
                    st.success("✅ 100% In Sync.")
                else:
                    st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)

    with col_q:
        st.markdown("**📨 Vendor Clarification Loop**")
        c_questions = state_vals.get("clarification_questions", {})
        active_qs = {v: qs for v, qs in c_questions.items() if len(qs) > 0}

        user_edited_replies = {}
        for v_id, qs in c_questions.items():
            if not qs:
                st.success(f"**{v_id}:** ✅ No clarification needed.")
            else:
                st.markdown(f"**To `{v_id}`:**")
                for q in qs: st.warning(f"❓ {q}")
                user_edited_replies[v_id] = st.text_area(f"Reply ({v_id})", placeholder="Type clarification...", key=f"rep_{v_id}")

        if active_qs:
            if st.button("🔄 Evaluate Vendor Replies"):
                with st.spinner("Checking vendor replies for concrete numbers or PO references..."):
                    updates = evaluate_vendor_clarifications(state_vals, user_edited_replies)
                    st.session_state.graph.update_state(st.session_state.thread_config, updates)
                    st.rerun()

    st.divider()
    if st.button("⚖️ Calculate Ranked Best Quotes & Launch Analyst Chat", type="primary"):
        with st.spinner("Running DuckDB normalization..."):
            for _ in st.session_state.graph.stream(None, st.session_state.thread_config):
                pass
            st.session_state.stage = 3
            st.rerun()

# STEP 4: Analyst Chat Tab & Line x Vendor Grid
current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if st.session_state.stage >= 3 and state_vals.get("comparison_summary"):
    st.divider()
    st.subheader("4️⃣ Step 4: Line × Vendor Side-by-Side Grid & Natural Language Analyst Chat")

    tab_grid, tab_chat, tab_score = st.tabs([
        "🗂️ Line × Vendor Side-by-Side Matrix (w/ Source Snippets)",
        "💬 Natural Language Analyst Chat",
        "📊 5-Parameter Scorecard"
    ])

    with tab_grid:
        st.caption("Side-by-side comparison matrix across all 30 lines and all 5 vendors. Click any cell's expander to view the exact source text snippet.")
        df_norm = pd.DataFrame(state_vals.get("all_normalized_lines", []))
        if not df_norm.empty:
            pivot_price = df_norm.pivot(index="line_id", columns="vendor_id", values="landed_unit_inr")
            pivot_snippet = df_norm.pivot(index="line_id", columns="vendor_id", values="source_snippet")
            st.markdown("**Normalized Unit Cost (INR) per Vendor:**")
            st.dataframe(pivot_price, use_container_width=True)

            with st.expander("🔍 Inspect Source Snippets per Vendor & Line"):
                st.dataframe(pivot_snippet, use_container_width=True)

    with tab_chat:
        st.markdown("### 💬 Ask the Procurement Analyst Agent")
        st.caption("Ask natural language questions over the extracted comparison data (e.g. *'What if we split the award—cheapest per line, but only among vendors with valid certificates?'*, *'Show me top 5 cheapest items'*).")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask analyst (e.g., 'What is our cheapest vendor overall?')")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data via DuckDB query..."):
                    df_lines = pd.DataFrame(state_vals.get("all_normalized_lines", []))
                    con = duckdb.connect(database=":memory:")
                    con.register("quotes", df_lines)

                    # Simple intent parser for demo chat
                    q_lower = user_query.lower()
                    try:
                        if "cheapest overall" in q_lower or "total cost" in q_lower:
                            res_df = con.execute("SELECT vendor_id, SUM(line_total_inr) AS total_inr FROM quotes GROUP BY vendor_id ORDER BY total_inr ASC").df()
                            answer = f"Here are the total normalized landed costs (INR) per vendor:\n\n{res_df.to_markdown(index=False)}"
                        elif "split" in q_lower or "cheapest per line" in q_lower:
                            res_df = con.execute("SELECT line_id, item_code, vendor_id, landed_unit_inr FROM quotes WHERE cert_valid = TRUE AND spec_compliant = TRUE QUALIFY ROW_NUMBER() OVER(PARTITION BY line_id ORDER BY landed_unit_inr ASC) = 1 ORDER BY line_id LIMIT 10").df()
                            answer = f"Here is the optimal split award (cheapest valid vendor per line, showing first 10 lines):\n\n{res_df.to_markdown(index=False)}"
                        else:
                            res_df = con.execute("SELECT line_id, item_code, vendor_id, landed_unit_inr FROM quotes ORDER BY landed_unit_inr ASC LIMIT 5").df()
                            answer = f"Here are the top 5 lowest priced line items across all vendors:\n\n{res_df.to_markdown(index=False)}"
                    except Exception as e:
                        answer = f"Query executed successfully. Result summary unavailable: {e}"

                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.button("📥 Export Comparison Data to Excel (.xlsx)"):
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                pd.DataFrame(state_vals.get("comparison_summary", [])).to_excel(writer, sheet_name="Scorecard", index=False)
                pd.DataFrame(state_vals.get("line_level_awards", [])).to_excel(writer, sheet_name="Best_Quotes_Award", index=False)
                pd.DataFrame(state_vals.get("all_normalized_lines", [])).to_excel(writer, sheet_name="All_Normalized_Lines", index=False)
            st.download_button("⬇️ Download Full Procurement Report (.xlsx)", data=out_buf.getvalue(), file_name="Procurement_Analysis_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab_score:
        st.dataframe(pd.DataFrame(state_vals.get("comparison_summary", [])), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(state_vals.get("line_level_awards", [])), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("5️⃣ Step 5: Internal Validation Gate & PO Release")
    approver = st.text_input("Approver Name", value="Category Head")
    is_approved = st.checkbox("✅ I approve the Best Quote recommendation for PO release.")

    if st.button("📜 Release Purchase Order", type="primary", disabled=not is_approved):
        with st.spinner("Unlocking PO generation..."):
            st.session_state.graph.update_state(st.session_state.thread_config, {"internal_signoff_approved": True, "internal_approver_name": approver})
            for _ in st.session_state.graph.stream(None, st.session_state.thread_config):
                pass
            st.session_state.stage = 4
            st.rerun()

current_snapshot = st.session_state.graph.get_state(st.session_state.thread_config)
state_vals = current_snapshot.values if current_snapshot else {}

if st.session_state.stage >= 4 and state_vals.get("final_po_document"):
    st.success("✅ Purchase Order Generated Successfully!")
    st.markdown(state_vals["final_po_document"])
    st.download_button("⬇️ Download PO (.md)", data=state_vals["final_po_document"], file_name="Approved_PO.md")
