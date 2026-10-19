import io
import os
import time
import uuid
import queue
import tempfile
import threading
from datetime import date

import pandas as pd
import streamlit as st

from schemas import DEFAULT_QUESTIONNAIRE
from tools_and_models import parse_buyer_requirements, DualLLMRouter
from graph_engine import (build_procurement_graph, stream_rfx, evaluate_vendor_clarifications,
                          AnalystAgent, set_progress_hook)

st.set_page_config(page_title="Quote Consolidation and Award Analysis", layout="wide")
st.title("Quote Consolidation, Audit and Award Analysis")
st.caption("Drafts the RFx, reads vendor responses in any format, consolidates them into one comparison, "
           "and supports natural-language analysis through to a defensible award.")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")
    gem_key = st.text_input("Google Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    groq_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    primary_gem = st.text_input("Primary Gemini model", value=os.getenv("PRIMARY_GEMINI_MODEL", "gemini-3.8-flash"))
    if gem_key:
        os.environ["GEMINI_API_KEY"] = gem_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    os.environ["PRIMARY_GEMINI_MODEL"] = primary_gem

    st.divider()
    st.subheader("Reference data")
    ref_date = st.date_input("Audit date", value=date.today()).isoformat()
    usd_rate = st.number_input("USD to INR", value=84.0)
    eur_rate = st.number_input("EUR to INR", value=91.0)
    gbp_rate = st.number_input("GBP to INR", value=108.0)

    st.divider()
    st.subheader("Quality questionnaire")
    q_df = st.data_editor(pd.DataFrame(DEFAULT_QUESTIONNAIRE), hide_index=True, use_container_width=True, num_rows="dynamic", key="qn_editor")
    questionnaire = q_df.dropna(subset=["id", "text"]).to_dict("records")

RFX_ID = "AER-2026-303"

if "graph" not in st.session_state:
    st.session_state.graph = build_procurement_graph()
    st.session_state.thread_config = {"configurable": {"thread_id": f"session-{uuid.uuid4().hex[:8]}"}}
    st.session_state.stage = 1
    st.session_state.chat_history = []
cfg = st.session_state.thread_config


def run_graph(payload, status):
    """Runs the graph in a worker thread and relays progress to the interface as it happens."""
    graph, q, err = st.session_state.graph, queue.Queue(), {}
    set_progress_hook(q.put)

    def worker():
        try:
            for _ in graph.stream(payload, cfg):
                pass
        except Exception as e:
            err["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        while not q.empty():
            status.write(q.get())
        time.sleep(0.15)
    while not q.empty():
        status.write(q.get())
    set_progress_hook(None)
    if "e" in err:
        raise err["e"]


def titled(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).replace("_", " ").title())


def snapshot():
    snap = st.session_state.graph.get_state(cfg)
    return snap.values if snap else {}


# ----------------------------------------------------------------------------- step 1
st.subheader("Step 1. Requirements and RFx")
buyer_file = st.file_uploader("Buyer requirement schedule (.xlsx or .csv)", type=["xlsx", "xls", "csv"])
if buyer_file and "buyer_reqs" not in st.session_state:
    path = os.path.join(tempfile.mkdtemp(), buyer_file.name)
    with open(path, "wb") as f:
        f.write(buyer_file.getbuffer())
    st.session_state.buyer_reqs, st.session_state.security_alerts = parse_buyer_requirements(path)

if "buyer_reqs" in st.session_state:
    with st.expander(f"Requirement schedule ({len(st.session_state.buyer_reqs)} lines)", expanded=False):
        st.dataframe(titled(pd.DataFrame(st.session_state.buyer_reqs)), use_container_width=True, hide_index=True)
    for a in st.session_state.security_alerts:
        st.warning(a)
    if st.button("Draft RFx", type="primary"):
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GROQ_API_KEY"):
            st.error("Enter at least one API key in the sidebar.")
        else:
            with st.container(border=True):
                st.session_state.rfx_doc = st.write_stream(
                    stream_rfx(st.session_state.buyer_reqs, RFX_ID, ref_date, questionnaire, DualLLMRouter()))
            st.session_state.stage = max(st.session_state.stage, 1.5)

if "rfx_doc" in st.session_state:
    st.download_button("Download RFx (.md)", data=st.session_state.rfx_doc, file_name="RFx.md")
    st.divider()

# ----------------------------------------------------------------------------- step 2
if st.session_state.stage >= 1.5:
    st.subheader("Step 2. Vendor responses")
    uploads = st.file_uploader("Upload vendor responses in any format", accept_multiple_files=True,
                               type=["pdf", "docx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "txt", "eml"])
    if uploads and st.button("Read and audit responses", type="primary"):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("A Gemini API key is required.")
        else:
            tmp, files = tempfile.mkdtemp(), {}
            for vf in uploads:
                p = os.path.join(tmp, vf.name)
                with open(p, "wb") as f:
                    f.write(vf.getbuffer())
                files[os.path.splitext(vf.name)[0]] = p
            state0 = {
                "rfx_id": RFX_ID, "reference_date": ref_date,
                "fx_rates": {"INR": 1.0, "USD": usd_rate, "EUR": eur_rate, "GBP": gbp_rate},
                "buyer_requirements": st.session_state.buyer_reqs, "questionnaire": questionnaire,
                "rfx_document": st.session_state.get("rfx_doc", ""), "security_alerts": st.session_state.get("security_alerts", []),
                "vendor_files": files, "gemini_extractions": {}, "llama_extractions": {}, "discrepancy_log": {},
                "clarification_questions": {}, "clarification_round": 0, "clarification_history": [],
                "vendor_clarifications": {}, "reconciled_quotes": {}, "comparison_summary": [], "line_level_awards": [],
                "all_normalized_lines": [], "review_queue": [], "defense_audit_log": [],
                "internal_signoff_approved": False, "internal_approver_name": "", "final_po_document": ""}
            with st.status("Reading vendor submissions in parallel", expanded=True) as status:
                run_graph(state0, status)
                status.update(label="All submissions read and audited", state="complete", expanded=False)
            st.session_state.stage = 2
            st.rerun()

state_vals = snapshot()

# ----------------------------------------------------------------------------- step 3
if st.session_state.stage >= 2 and state_vals.get("discrepancy_log"):
    st.divider()
    rnd = state_vals.get("clarification_round", 1)
    st.subheader(f"Step 3. Audit findings and vendor clarifications (round {rnd})")
    for a in state_vals.get("security_alerts", []):
        st.error(a)
    col_d, col_q = st.columns([1.2, 1])
    with col_d:
        st.markdown("**Findings by vendor**")
        for vid, issues in state_vals["discrepancy_log"].items():
            with st.expander(f"{vid} ({len(issues)} finding{'s' if len(issues) != 1 else ''})", expanded=bool(issues)):
                if issues:
                    st.dataframe(titled(pd.DataFrame(issues)), use_container_width=True, hide_index=True)
                else:
                    st.success("No findings. Both readings agree.")
    with col_q:
        st.markdown("**Vendor clarification loop**")
        replies, active = {}, False
        for vid, qs in state_vals.get("clarification_questions", {}).items():
            if not qs:
                st.success(f"{vid}: no clarification required.")
                continue
            active = True
            st.markdown(f"**To {vid}**")
            for q in qs:
                st.info(q)
            replies[vid] = st.text_area(f"Reply from {vid}", key=f"rep_{vid}_{rnd}", placeholder="Paste the vendor reply")
        if active and st.button("Evaluate vendor replies"):
            with st.spinner("Evaluating replies"):
                upd = evaluate_vendor_clarifications(state_vals, replies)
                st.session_state.graph.update_state(cfg, upd, as_node="diff_and_questionnaire_node")
            st.rerun()
    st.divider()
    if st.button("Compute comparison and award recommendation", type="primary"):
        with st.status("Normalising units and currency, applying discounts and freight", expanded=False) as status:
            run_graph(None, status)
            status.update(label="Comparison ready", state="complete")
        st.session_state.stage = 3
        st.rerun()

state_vals = snapshot()


# ----------------------------------------------------------------------------- step 4
def styled_matrix(df: pd.DataFrame):
    idx = ["line_id", "item_code"]
    price = df.pivot_table(index=idx, columns="vendor_id", values="landed_unit_inr", aggfunc="min")
    elig = df.assign(e=df["award_eligible"].astype(int)).pivot_table(index=idx, columns="vendor_id", values="e", aggfunc="max").reindex_like(price).fillna(0).astype(bool)
    flag = df.assign(f=(df["review_flags"] != "").astype(int)).pivot_table(index=idx, columns="vendor_id", values="f", aggfunc="max").reindex_like(price).fillna(0).astype(bool)
    best = price.where(elig).min(axis=1)

    def colour(_):
        s = pd.DataFrame("", index=price.index, columns=price.columns)
        s[price.eq(best, axis=0) & elig] = "background-color: #d9ead3"
        s[flag] = "background-color: #fff2cc"
        return s
    return price.style.apply(colour, axis=None).format("{:,.2f}", na_rep="Not quoted")


def build_report(sv) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(sv.get("comparison_summary", [])).to_excel(w, sheet_name="Vendor_Scorecard", index=False)
        pd.DataFrame(sv.get("line_level_awards", [])).to_excel(w, sheet_name="Award_Recommendation", index=False)
        pd.DataFrame(sv.get("all_normalized_lines", [])).to_excel(w, sheet_name="Normalised_Lines", index=False)
        pd.DataFrame(sv.get("review_queue", [])).to_excel(w, sheet_name="Review_Queue", index=False)
        pd.DataFrame({"Audit record": sv.get("defense_audit_log", [])}).to_excel(w, sheet_name="Audit_Record", index=False)
    return buf.getvalue()


def render_result(m, key):
    if m.get("sql"):
        with st.expander("Query used"):
            st.code(m["sql"], language="sql")
            for a in m.get("assumptions", []):
                st.caption(a)
    df = m.get("df")
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        ch = m.get("chart") or {}
        if ch.get("type") in ("bar", "line") and ch.get("x") in df.columns and ch.get("y") in df.columns:
            plot = df.set_index(ch["x"])[ch["y"]]
            (st.bar_chart if ch["type"] == "bar" else st.line_chart)(plot)
        st.download_button("Download this result (.csv)", df.to_csv(index=False), file_name="analysis_result.csv", key=f"dl_{key}")


if st.session_state.stage >= 3 and state_vals.get("comparison_summary"):
    st.divider()
    st.subheader("Step 4. Comparison and analyst")
    df_norm = pd.DataFrame(state_vals.get("all_normalized_lines", []))
    tab_grid, tab_review, tab_chat, tab_score = st.tabs(["Comparison matrix", "Review queue", "Analyst", "Scorecard and award"])

    with tab_grid:
        st.caption("Normalised landed unit cost in INR. Green marks the lowest eligible price on a line; amber marks a quotation carrying review flags.")
        if not df_norm.empty:
            st.dataframe(styled_matrix(df_norm), use_container_width=True)
            with st.expander("Source evidence by vendor and line"):
                st.dataframe(titled(df_norm[["line_id", "vendor_id", "raw_price", "currency", "quoted_uom", "read_agreement", "confidence", "source_snippet"]]),
                             use_container_width=True, hide_index=True)

    with tab_review:
        rq = pd.DataFrame(state_vals.get("review_queue", []))
        if rq.empty:
            st.success("No quotations carry review flags.")
        else:
            st.caption("Lines requiring buyer attention. Lines marked ineligible are excluded from the award until resolved.")
            st.dataframe(titled(rq), use_container_width=True, hide_index=True)

    with tab_chat:
        st.caption("Ask questions over the extracted data. Each answer shows the query, the resulting table and a grounded summary.")
        suggestions = ["Which vendor has the lowest total landed cost, and how many lines does each quote?",
                       "Split the award: cheapest eligible vendor per line, only among vendors who cleared the quality questionnaire.",
                       "Show the ten lines with the widest price spread across vendors as a chart.",
                       "Which awarded lines are provisional and why?"]
        cols = st.columns(len(suggestions))
        picked = None
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sug_{i}", use_container_width=True):
                picked = s
        for i, m in enumerate(st.session_state.chat_history):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    render_result(m, i)
        question = st.chat_input("Ask a question about the quotations") or picked
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                agent = AnalystAgent(state_vals)
                with st.status("Analysing", expanded=False) as s:
                    res = agent.ask(question, [{"q": h["q"], "sql": h.get("sql", "")} for h in st.session_state.chat_history if h["role"] == "assistant"])
                    s.update(label="Query executed", state="complete")
                render_result(res, "new")
                text = st.write_stream(agent.narrate(question, res))
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": text, "q": question, **{k: res[k] for k in ("sql", "df", "chart", "assumptions")}})

    with tab_score:
        st.dataframe(titled(pd.DataFrame(state_vals["comparison_summary"])), use_container_width=True, hide_index=True)
        aw = pd.DataFrame(state_vals.get("line_level_awards", []))
        if not aw.empty:
            st.markdown(f"**Recommended split award: INR {aw['line_total_inr'].sum():,.2f} across {aw['recommended_vendor'].nunique()} vendor(s)**")
            st.dataframe(titled(aw.drop(columns=["source_snippet"], errors="ignore")), use_container_width=True, hide_index=True)
        with st.expander("Audit record"):
            for e in state_vals.get("defense_audit_log", []):
                st.markdown(f"- {e}")
        st.download_button("Download full report (.xlsx)", build_report(state_vals), file_name="Procurement_Analysis_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.subheader("Step 5. Internal approval and purchase order")
    approver = st.text_input("Approver name", value="Category Head")
    ok = st.checkbox("I have reviewed the recommendation, including provisional lines, and approve release.")
    if st.button("Release purchase order", type="primary", disabled=not ok):
        st.session_state.graph.update_state(cfg, {"internal_signoff_approved": True, "internal_approver_name": approver}, as_node="reconcile_and_score_node")
        with st.status("Generating purchase order", expanded=False) as status:
            run_graph(None, status)
            status.update(label="Purchase order generated", state="complete")
        st.session_state.stage = 4
        st.rerun()

state_vals = snapshot()
if st.session_state.stage >= 4 and state_vals.get("final_po_document"):
    st.success("Purchase order generated.")
    st.markdown(state_vals["final_po_document"])
    st.download_button("Download purchase order (.md)", state_vals["final_po_document"], file_name="Purchase_Order.md")
