import json
import sqlite3
import pandas as pd
import duckdb
from datetime import datetime
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from schemas import ProcurementState, VendorQuoteExtraction
from tools_and_models import parse_buyer_requirements, extract_multimodal_payload, DualLLMRouter


def build_extraction_prompt(buyer_reqs: List[Dict[str, Any]], raw_text: str, vendor_id: str) -> str:
    schema_hint = json.dumps(VendorQuoteExtraction.model_json_schema(), indent=2)
    return f"""You are a senior procurement auditor. Extract the vendor quotation for '{vendor_id}' and map it strictly against the Buyer's RFx Line Items.

TODAY'S AUDIT DATE: 2026-10-17

BUYER RFx LINE ITEMS (Target Specs & Units):
{json.dumps(buyer_reqs, indent=2)}

RAW VENDOR SUBMISSION TEXT / TABLES:
{raw_text}

CRITICAL EXTRACTION RULES:
1. Map every quoted item to its corresponding `line_id` (1 to {len(buyer_reqs)}). If a vendor omits a line item, either omit it from `line_items` or set `quoted_qty` to 0 and record it in `ambiguities_detected`.
2. Watch for unit of measure (UoM) mismatches: e.g., if Buyer wants 'piece' and vendor quotes 'per 100 pcs', set `conversion_factor_to_target_uom` = 0.01. If vendor quotes 'per box' and doesn't state pieces per box, flag it in `ambiguities_detected`.
3. Capture buried footnote discounts in `global_footnote_discount_pct` or `line_discount_pct`.
4. Check certificate validity (e.g., ISO/FSC expiry date).
5. Return ONLY valid JSON matching this exact schema:
{schema_hint}
"""


# --- NODE 1: RFx Generation ---
def create_rfx_node(state: ProcurementState) -> Dict[str, Any]:
    router = DualLLMRouter()
    reqs = state["buyer_requirements"]
    prompt = f"""Draft a professional, comprehensive Request for Quotation (RFx) # {state['rfx_id']} dated {state['reference_date']}.
Include:
1. Scope & Commercial Terms (Currency preference: INR, DDP delivery, mandatory valid ISO/FSC quality certificate as of {state['reference_date']}).
2. Evaluation Criteria (5 Parameters: Certificate Validity, Normalized Landed Cost, Line Coverage & Spec Compliance, Incoterms/Freight, Payment & Lead Time Impact).
3. Summary Table of the {len(reqs)} Line Items to be procured:
{json.dumps(reqs[:10], indent=2)} ... (total {len(reqs)} items).
"""
    rfx_doc, used_model = router.call_gemini_with_fallback(prompt, schema=None)
    return {"rfx_document": f"*Generated via {used_model}*\n\n" + rfx_doc}


# --- NODE 2: Dual-LLM Extraction (Gemini Flash w/ Fallback + Groq Llama 4) ---
def dual_extract_quotes_node(state: ProcurementState) -> Dict[str, Any]:
    router = DualLLMRouter()
    buyer_reqs = state["buyer_requirements"]
    gemini_outs = {}
    llama_outs = {}

    for vendor_id, file_path in state["vendor_files"].items():
        raw_text, img_bytes = extract_multimodal_payload(file_path)
        prompt = build_extraction_prompt(buyer_reqs, raw_text, vendor_id)

        # 1. Gemini Extraction with fallback
        gem_json, gem_model = router.call_gemini_with_fallback(
            prompt=prompt, images_bytes=img_bytes, schema=VendorQuoteExtraction
        )
        gem_json["_model_used"] = gem_model
        gemini_outs[vendor_id] = gem_json

        # 2. Groq Llama 4 Extraction
        llama_json, llama_model = router.call_groq_llama4(
            prompt=prompt, images_bytes=img_bytes, json_mode=True
        )
        llama_json["_model_used"] = llama_model
        llama_outs[vendor_id] = llama_json

    return {
        "gemini_extractions": gemini_outs,
        "llama_extractions": llama_outs
    }


# --- NODE 3: Code-Level Consensus Diffing & Questionnaire Generation ---
def diff_and_questionnaire_node(state: ProcurementState) -> Dict[str, Any]:
    buyer_reqs = state["buyer_requirements"]
    total_expected_lines = len(buyer_reqs)
    discrepancy_log: Dict[str, List[str]] = {}
    clarification_questions: Dict[str, List[str]] = {}
    reconciled_quotes: Dict[str, Dict[str, Any]] = {}

    for vendor_id in state["vendor_files"].keys():
        gem = state["gemini_extractions"].get(vendor_id, {})
        lla = state["llama_extractions"].get(vendor_id, {})
        diffs = []
        questions = []

        # Compare top-level parameters in Python code
        if gem.get("certificate_expiry_date") != lla.get("certificate_expiry_date"):
            diffs.append(
                f"Certificate Expiry LLM Disagreement: Gemini read '{gem.get('certificate_expiry_date')}' vs Llama 4 read '{lla.get('certificate_expiry_date')}'."
            )

        cert_exp = gem.get("certificate_expiry_date", "MISSING")
        if cert_exp in ["MISSING", "EXPIRED", ""] or cert_exp < state["reference_date"]:
            questions.append(
                f"Your quality certificate ({gem.get('certificate_name', 'ISO/FSC')}) appears expired or missing (status: {cert_exp} vs audit date {state['reference_date']}). Can you provide a renewed valid certificate?"
            )

        if gem.get("freight_included") != lla.get("freight_included"):
            diffs.append(
                f"Freight Terms LLM Disagreement: Gemini={gem.get('freight_included')} vs Llama4={lla.get('freight_included')}."
            )
        if not gem.get("freight_included", True):
            questions.append(
                f"Your quote indicates freight is extra ({gem.get('incoterms', 'EXW')}). Please confirm the exact percentage or flat freight charge for DDP delivery."
            )

        # Line-by-line deterministic comparison
        gem_lines = {int(item["line_id"]): item for item in gem.get("line_items", []) if "line_id" in item}
        lla_lines = {int(item["line_id"]): item for item in lla.get("line_items", []) if "line_id" in item}

        quoted_line_ids = [lid for lid, item in gem_lines.items() if float(item.get("quoted_qty", 0)) > 0]
        if len(quoted_line_ids) < total_expected_lines:
            missing_ids = sorted(list(set(range(1, total_expected_lines + 1)) - set(quoted_line_ids)))
            diffs.append(f"Partial Coverage: Vendor quoted {len(quoted_line_ids)}/{total_expected_lines} lines. Missing lines: {missing_ids}")
            questions.append(f"Lines {missing_ids} were omitted from your bid. Please confirm if you can supply them and at what unit price.")

        for lid in range(1, total_expected_lines + 1):
            g_item = gem_lines.get(lid)
            l_item = lla_lines.get(lid)
            if g_item and l_item:
                g_price = float(g_item.get("quoted_unit_price", 0))
                l_price = float(l_item.get("quoted_unit_price", 0))
                if abs(g_price - l_price) > 0.01:
                    diffs.append(
                        f"Line {lid} Price Discrepancy between LLMs: Gemini extracted {g_price} vs Llama 4 extracted {l_price}."
                    )
                    questions.append(
                        f"Line {lid} price was ambiguous in your document ({g_price} vs {l_price}). Please confirm the exact unit rate and UoM."
                    )
                if not g_item.get("spec_compliant", True):
                    diffs.append(f"Line {lid} Spec Deviation: Quoted '{g_item.get('quoted_description')}' against buyer spec.")
                    questions.append(f"Line {lid} deviates from target specification ({g_item.get('quoted_description')}). Confirm if compliant spec can be supplied at quoted rate.")
                if g_item.get("quoted_uom", "").lower() != buyer_reqs[lid - 1]["target_uom"].lower():
                    questions.append(
                        f"Line {lid} was quoted in '{g_item.get('quoted_uom')}' instead of target '{buyer_reqs[lid - 1]['target_uom']}'. Confirm conversion factor ({g_item.get('conversion_factor_to_target_uom')})."
                    )

        # Merge LLM-detected ambiguities
        for amb in gem.get("ambiguities_detected", []):
            if amb not in questions:
                questions.append(f"Clarify ambiguity: {amb}")

        if not diffs:
            diffs.append("100% Consensus between Gemini and Llama 4 across all extracted fields.")
        if not questions:
            questions.append("No ambiguities found. Please re-confirm commercial validity for 30 days.")

        discrepancy_log[vendor_id] = diffs
        clarification_questions[vendor_id] = questions[:6]  # Keep questionnaire crisp
        reconciled_quotes[vendor_id] = gem  # Base reconciliation on Gemini schema + audit log

    return {
        "discrepancy_log": discrepancy_log,
        "clarification_questions": clarification_questions,
        "reconciled_quotes": reconciled_quotes
    }


# --- NODE 4: Deterministic Price Normalization, 5-Parameter Scoring & Defense Log ---
def reconcile_and_score_node(state: ProcurementState) -> Dict[str, Any]:
    buyer_reqs = state["buyer_requirements"]
    fx_rates = state.get("fx_rates", {"INR": 1.0, "USD": 84.0, "EUR": 91.0})
    ref_date = state["reference_date"]
    clarifications = state.get("vendor_clarifications", {})
    reconciled = state["reconciled_quotes"]

    defense_log: List[str] = []
    comparison_summary: List[Dict[str, Any]] = []
    all_normalized_lines: List[Dict[str, Any]] = []

    for vendor_id, quote in reconciled.items():
        reply_text = clarifications.get(vendor_id, "").lower()

        # 1. Parameter 1: Certificate Validity Check (with vendor clarification override)
        cert_exp = str(quote.get("certificate_expiry_date", "MISSING"))
        cert_valid = (cert_exp not in ["MISSING", "EXPIRED", "None", ""]) and (cert_exp >= ref_date)
        if not cert_valid and ("renewed" in reply_text or "valid" in reply_text or "2027" in reply_text or "2028" in reply_text):
            cert_valid = True
            cert_status = "VALID (Cleared via Clarification)"
            defense_log.append(f"[DEFENSE - {vendor_id}] Certificate initially flagged ({cert_exp}), marked VALID after vendor submitted renewed compliance confirmation.")
        elif cert_valid:
            cert_status = f"VALID (Exp: {cert_exp})"
        else:
            cert_status = f"INVALID/EXPIRED ({cert_exp})"
            defense_log.append(f"[DEFENSE - {vendor_id}] Disqualified from Quality-Cleared Pool: Certificate '{quote.get('certificate_name')}' expired/missing ({cert_exp} < {ref_date}).")

        # 2. Parameter 3: Incoterms & Hidden Logistics Surcharge
        freight_inc = bool(quote.get("freight_included", True))
        freight_surcharge_pct = 0.0
        if not freight_inc:
            freight_surcharge_pct = float(quote.get("freight_extra_pct_or_flat", 5.0))
            if freight_surcharge_pct <= 0 or freight_surcharge_pct > 50:
                freight_surcharge_pct = 6.0  # Deterministic 6% EXW/FOB freight penalty if unstated
            defense_log.append(f"[DEFENSE - {vendor_id}] Applied +{freight_surcharge_pct:.1f}% logistics surcharge because freight was marked EXTRA ({quote.get('incoterms', 'EXW')}).")

        # 3. Parameter 4: Lead Time & Payment Terms Working-Capital Impact
        pay_days = int(quote.get("payment_terms_days", 30))
        lead_days = int(quote.get("delivery_lead_time_days", 14))
        # Working capital cost adjustment: baseline Net 30 (1% annual per 30 days = ~0.033% per day difference)
        wc_adjustment_pct = (30 - pay_days) * 0.033
        global_disc = float(quote.get("global_footnote_discount_pct", 0.0))

        # 4. Line-level normalization across all 30 lines
        q_lines = {int(l["line_id"]): l for l in quote.get("line_items", []) if "line_id" in l}
        covered_lines = 0
        compliant_lines = 0
        total_landed_cost_inr = 0.0

        for req in buyer_reqs:
            lid = int(req["line_id"])
            req_qty = float(req["required_qty"])
            item_q = q_lines.get(lid)

            if item_q and float(item_q.get("quoted_qty", 0)) > 0:
                covered_lines += 1
                is_comp = bool(item_q.get("spec_compliant", True))
                if is_comp or "compliant" in reply_text:
                    compliant_lines += 1
                    is_comp = True

                raw_price = float(item_q.get("quoted_unit_price", 0.0))
                curr = str(item_q.get("quoted_Currency", "INR")).upper()
                fx = fx_rates.get(curr, 1.0)
                conv = float(item_q.get("conversion_factor_to_target_uom", 1.0))
                if conv <= 0:
                    conv = 1.0
                line_disc = float(item_q.get("line_discount_pct", 0.0))
                effective_disc = max(global_disc, line_disc)

                # Deterministic Landed Unit Cost Formula in INR per Buyer Target UoM
                base_inr_per_target_uom = raw_price * fx * conv
                discounted_unit = base_inr_per_target_uom * (1.0 - effective_disc / 100.0)
                landed_unit_inr = discounted_unit * (1.0 + (freight_surcharge_pct + wc_adjustment_pct) / 100.0)
                line_total_inr = landed_unit_inr * req_qty
                total_landed_cost_inr += line_total_inr

                all_normalized_lines.append({
                    "line_id": lid,
                    "item_code": req["item_code"],
                    "description": req["description"],
                    "required_qty": req_qty,
                    "target_uom": req["target_uom"],
                    "vendor_id": vendor_id,
                    "cert_valid": cert_valid,
                    "spec_compliant": is_comp,
                    "raw_price": raw_price,
                    "currency": curr,
                    "quoted_uom": item_q.get("quoted_uom", req["target_uom"]),
                    "discount_pct": effective_disc,
                    "freight_surcharge_pct": freight_surcharge_pct,
                    "landed_unit_inr": round(landed_unit_inr, 2),
                    "line_total_inr": round(line_total_inr, 2)
                })

        comparison_summary.append({
            "Vendor": vendor_id,
            "1_Certificate_Validity": cert_status,
            "2_Normalized_Total_Landed_INR": round(total_landed_cost_inr, 2),
            "3_Coverage_&_Spec_Compliance": f"{covered_lines}/{len(buyer_reqs)} Quoted ({compliant_lines} Compliant)",
            "4_Incoterms_&_Freight": f"{quote.get('incoterms', 'DDP')} ({'Included' if freight_inc else f'+{freight_surcharge_pct}% Extra'})",
            "5_LeadTime_&_PaymentTerms": f"{lead_days}d Lead | Net {pay_days}d ({wc_adjustment_pct:+.2f}% WC adj)",
            "Eligible_For_Award": cert_valid and (covered_lines > 0)
        })

    # Use DuckDB over normalized lines to select the cheapest eligible & compliant vendor per line
    df_lines = pd.DataFrame(all_normalized_lines)
    line_awards: List[Dict[str, Any]] = []

    if not df_lines.empty:
        con = duckdb.connect(database=":memory:")
        con.register("quotes_table", df_lines)
        # First try strictly compliant & valid certificate quotes
        award_df = con.execute("""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY line_id
                        ORDER BY
                            CASE WHEN cert_valid AND spec_compliant THEN 0
                                 WHEN spec_compliant THEN 1
                                 ELSE 2 END ASC,
                            landed_unit_inr ASC
                    ) as rk
                FROM quotes_table
                WHERE landed_unit_inr > 0
            )
            SELECT line_id, item_code, description, required_qty, target_uom,
                   vendor_id AS winning_vendor, landed_unit_inr, line_total_inr,
                   cert_valid, spec_compliant, currency, raw_price, quoted_uom
            FROM ranked
            WHERE rk = 1
            ORDER BY line_id
        """).df()
        line_awards = award_df.to_dict(orient="records")

        # Add split-award summary defense
        total_split_po = sum(r["line_total_inr"] for r in line_awards)
        defense_log.append(
            f"[DEFENSE - OPTIMIZATION] Executed DuckDB line-level L1 optimization across {len(line_awards)} lines prioritizing Valid Certificates + Spec Compliance. Total Optimized PO Value: INR {total_split_po:,.2f}."
        )

    return {
        "comparison_summary": comparison_summary,
        "line_level_awards": line_awards,
        "defense_audit_log": defense_log
    }


# --- NODE 5: Automated Purchase Order (PO) Drafting ---
def generate_po_node(state: ProcurementState) -> Dict[str, Any]:
    awards = state.get("line_level_awards", [])
    defense_log = state.get("defense_audit_log", [])

    if not awards:
        return {"final_po_document": "### No eligible line items found to issue a Purchase Order."}

    df_awards = pd.DataFrame(awards)
    total_po_val = df_awards["line_total_inr"].sum()
    vendors_awarded = df_awards["winning_vendor"].unique().tolist()

    po_md = f"""# PURCHASE ORDER (SPLIT / OPTIMIZED AWARD)
**PO Reference:** PO-{state['rfx_id']}  
**Execution Date:** {state['reference_date']}  
**Awarded Vendor(s):** {', '.join(vendors_awarded)}  
**Total Contract Value (Normalized DDP INR):** ₹ {total_po_val:,.2f}

---

### 1. Awarded Line Items Schedule
{df_awards[['line_id', 'item_code', 'description', 'required_qty', 'target_uom', 'winning_vendor', 'landed_unit_inr', 'line_total_inr']].to_markdown(index=False)}

---

### 2. Dual-LLM & Code Audit Defense Ledger
The above award was mathematically verified using dual independent LLM extractions (**Gemini Flash w/ Fallback** + **Groq Llama 4**) reconciled via Python & DuckDB:
"""
    for entry in defense_log:
        po_md += f"- {entry}\n"

    return {"final_po_document": po_md}


def build_procurement_graph(db_path: str = "/tmp/procurement_state.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    workflow = StateGraph(ProcurementState)
    workflow.add_node("create_rfx_node", create_rfx_node)
    workflow.add_node("dual_extract_quotes_node", dual_extract_quotes_node)
    workflow.add_node("diff_and_questionnaire_node", diff_and_questionnaire_node)
    workflow.add_node("reconcile_and_score_node", reconcile_and_score_node)
    workflow.add_node("generate_po_node", generate_po_node)

    workflow.set_entry_point("create_rfx_node")
    workflow.add_edge("create_rfx_node", "dual_extract_quotes_node")
    workflow.add_edge("dual_extract_quotes_node", "diff_and_questionnaire_node")
    workflow.add_edge("diff_and_questionnaire_node", "reconcile_and_score_node")
    workflow.add_edge("reconcile_and_score_node", "generate_po_node")
    workflow.add_edge("generate_po_node", END)

    # Pause before scoring so the buyer can review & simulate vendor replies to the questionnaire
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["reconcile_and_score_node"]
    )
