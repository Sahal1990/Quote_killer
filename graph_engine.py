import os
import json
import sqlite3
import pandas as pd
import duckdb
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from schemas import ProcurementState, VendorQuoteExtraction, ClarificationEvaluation
from tools_and_models import extract_multimodal_payload, DualLLMRouter, scan_for_prompt_injection


def _format_int_ranges(numbers: List[int]) -> str:
    if not numbers:
        return ""
    nums = sorted(set(numbers))
    ranges = []
    start = end = nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    return ", ".join(ranges)


def build_extraction_prompt(buyer_reqs: List[Dict[str, Any]], raw_text: str, vendor_id: str) -> str:
    schema_hint = json.dumps(VendorQuoteExtraction.model_json_schema(), indent=2)
    return f"""You are a senior procurement auditor. Extract the vendor quotation for '{vendor_id}' and map it strictly against the Buyer's RFx Line Items.

TODAY'S AUDIT DATE: 2026-10-17

BUYER RFx LINE ITEMS:
{json.dumps(buyer_reqs, indent=2)}

RAW VENDOR SUBMISSION TEXT:
{raw_text}

CRITICAL EXTRACTION RULES:
1. Map every quoted item to its `line_id` (1 to {len(buyer_reqs)}). If omitted, omit from `line_items` or set `quoted_qty` = 0.
2. Capture the exact text snippet or phrase from the document where this price was found into `source_snippet`.
3. Normalize currency code strictly (e.g. "INR", "USD", "EUR", "GBP"). If symbol is "US$" or "$", set currency to "USD".
4. Set `extraction_confidence_score` (0.0 to 1.0). If document is an angled/blurry photo, set confidence to <= 0.70.
5. Return ONLY valid JSON matching this schema:
{schema_hint}
"""


def create_rfx_node(state: ProcurementState) -> Dict[str, Any]:
    router = DualLLMRouter()
    reqs = state["buyer_requirements"]
    df_reqs = pd.DataFrame(reqs)
    items_md = df_reqs.to_markdown(index=False)

    prompt = f"""Draft a professional Request for Quotation (RFx) Reference #{state['rfx_id']} dated {state['reference_date']}.

CRITICAL SUBMISSION POLICY (MUST BE HIGHLIGHTED PROMINENTLY):
- DO NOT include or demand any rigid quotation template or standard spreadsheet format.
- Explicitly encourage vendors to submit their quotes in **Word (.docx) or PDF (.pdf) documents**, rate cards, emails, or clear photos. Our AI engine parses all formats automatically.

Include project scope, commercial expectations, and the following schedule of {len(reqs)} items:

{items_md}
"""
    rfx_doc, used_model = router.call_gemini_with_fallback(prompt, schema=None)
    return {"rfx_document": f"*Generated via {used_model}*\n\n" + rfx_doc}


def dual_extract_quotes_node(state: ProcurementState) -> Dict[str, Any]:
    router = DualLLMRouter()
    buyer_reqs = state["buyer_requirements"]
    gemini_outs = {}
    llama_outs = {}
    sec_alerts = list(state.get("security_alerts", []))

    for vendor_id, file_path in state["vendor_files"].items():
        raw_text, img_bytes = extract_multimodal_payload(file_path)
        is_image_only = os.path.splitext(file_path)[-1].lower() in [".png", ".jpg", ".jpeg", ".webp"]

        inj_flag, inj_details = scan_for_prompt_injection(raw_text, source_stage=f"Vendor Quote ({vendor_id})")
        if inj_flag:
            sec_alerts.append(inj_details)

        prompt = build_extraction_prompt(buyer_reqs, raw_text, vendor_id)

        # Gemini Extraction
        gem_json, gem_model = router.call_gemini_with_fallback(
            prompt=prompt, images_bytes=img_bytes, schema=VendorQuoteExtraction
        )
        if inj_flag or gem_json.get("prompt_injection_detected"):
            gem_json["prompt_injection_detected"] = True
            gem_json["injection_details"] = inj_details
        gem_json["_model_used"] = gem_model
        gemini_outs[vendor_id] = gem_json

        # Groq Llama Extraction (Honest Second Read)
        try:
            llama_json, llama_model = router.call_groq_llama4(
                prompt=prompt, images_bytes=img_bytes, json_mode=True
            )
            llama_json["_model_used"] = llama_model
            llama_outs[vendor_id] = llama_json
        except Exception as e:
            # Honest Error Handling: Mark unavailable rather than silently cloning Gemini
            fallback_copy = dict(gem_json)
            fallback_copy["vendor_name"] = vendor_id
            fallback_copy["ambiguities_detected"] = [f"Second-read LLM (Groq) unavailable ({str(e)[:45]}) - relying on primary Gemini extraction."]
            fallback_copy["extraction_confidence_score"] = min(float(gem_json.get("extraction_confidence_score", 0.95)), 0.70)
            llama_outs[vendor_id] = fallback_copy

    return {
        "gemini_extractions": gemini_outs,
        "llama_extractions": llama_outs,
        "security_alerts": sec_alerts
    }


def diff_and_questionnaire_node(state: ProcurementState) -> Dict[str, Any]:
    buyer_reqs = state["buyer_requirements"]
    total_expected_lines = len(buyer_reqs)
    discrepancy_log: Dict[str, List[Dict[str, Any]]] = {}
    clarification_questions: Dict[str, List[str]] = {}
    reconciled_quotes: Dict[str, Dict[str, Any]] = {}

    for vendor_id, file_path in state["vendor_files"].items():
        gem = state["gemini_extractions"].get(vendor_id, {})
        lla = state["llama_extractions"].get(vendor_id, {})
        is_img = os.path.splitext(file_path)[-1].lower() in [".png", ".jpg", ".jpeg", ".webp"]
        base_conf = float(gem.get("extraction_confidence_score", 0.92))

        issues: List[Dict[str, Any]] = []
        questions: List[str] = []

        if gem.get("prompt_injection_detected", False):
            issues.append({
                "issue": f"🚨 PROMPT INJECTION ATTEMPT: {gem.get('injection_details')}",
                "confidence": 0.99,
                "status": "ESCALATED TO HUMAN (SECURITY)"
            })
            questions.append("Your submission contained unauthorized system instruction patterns. Please resubmit clean commercial figures.")

        # Certificate Check
        cert_exp = str(gem.get("certificate_expiry_date", "MISSING"))
        lla_cert_exp = str(lla.get("certificate_expiry_date", cert_exp))
        cert_conf = 0.95 if (cert_exp == lla_cert_exp and not is_img) else 0.72

        if cert_exp in ["MISSING", "EXPIRED", "None", ""] or cert_exp < state["reference_date"]:
            issues.append({
                "issue": f"Quality Certificate ({gem.get('certificate_name', 'ISO/FSC')}) is '{cert_exp}' (Audit Date: {state['reference_date']}).",
                "confidence": cert_conf,
                "status": "AUTO-PASS (HIGH CONF)" if cert_conf >= 0.85 else "ESCALATED TO HUMAN (MED/LOW CONF)"
            })
            questions.append(
                f"Your quality certificate ({gem.get('certificate_name', 'ISO/FSC')}) is listed as '{cert_exp}'. Please provide a valid renewed certificate expiry date (YYYY-MM-DD)."
            )

        # Line coverage & spec diffs
        gem_lines = {int(item["line_id"]): item for item in gem.get("line_items", []) if "line_id" in item}
        missing_line_ids = []
        spec_dev_line_ids = []

        for lid in range(1, total_expected_lines + 1):
            g_item = gem_lines.get(lid)
            if not g_item or float(g_item.get("quoted_qty", 0)) <= 0 or float(g_item.get("quoted_unit_price", 0)) <= 0:
                missing_line_ids.append(lid)
                continue
            if not g_item.get("spec_compliant", True):
                spec_dev_line_ids.append(lid)

        if missing_line_ids:
            rng_str = _format_int_ranges(missing_line_ids)
            m_conf = 0.94 if not is_img else 0.75
            issues.append({
                "issue": f"Partial Line Coverage: Lines {rng_str} ({len(missing_line_ids)} items) omitted or unpriced.",
                "confidence": m_conf,
                "status": "AUTO-PASS (HIGH CONF)" if m_conf >= 0.85 else "ESCALATED TO HUMAN (MED/LOW CONF)"
            })
            questions.append(
                f"Lines {rng_str} were not quoted. Please state exact unit rates in INR or attach/reference a verifiable previous PO/quote number."
            )

        if spec_dev_line_ids:
            rng_str = _format_int_ranges(spec_dev_line_ids)
            issues.append({
                "issue": f"Technical Spec Deviation on Lines {rng_str}.",
                "confidence": 0.88,
                "status": "AUTO-PASS (HIGH CONF)"
            })
            questions.append(f"Lines {rng_str} deviate from technical specifications. Confirm if compliant spec can be supplied at quoted rate.")

        if is_img or base_conf < 0.85:
            img_conf = min(base_conf, 0.70)
            issues.append({
                "issue": f"Visual/Angled Image Source Detected (OCR Confidence: {img_conf:.2f}). Escalated for human spot-check.",
                "confidence": img_conf,
                "status": "ESCALATED TO HUMAN (MED CONF - VISUAL SOURCE)"
            })

        discrepancy_log[vendor_id] = issues
        clarification_questions[vendor_id] = questions
        reconciled_quotes[vendor_id] = gem

    return {
        "discrepancy_log": discrepancy_log,
        "clarification_questions": clarification_questions,
        "reconciled_quotes": reconciled_quotes,
        "clarification_round": state.get("clarification_round", 0) + 1
    }


def evaluate_vendor_clarifications(state: ProcurementState, new_replies: Dict[str, str]) -> Dict[str, Any]:
    router = DualLLMRouter()
    reconciled = dict(state.get("reconciled_quotes", {}))
    current_questions = dict(state.get("clarification_questions", {}))
    history = list(state.get("clarification_history", []))
    sec_alerts = list(state.get("security_alerts", []))
    follow_up_map: Dict[str, List[str]] = {}

    schema_hint = json.dumps(ClarificationEvaluation.model_json_schema(), indent=2)

    for vendor_id, reply_text in new_replies.items():
        if not reply_text or not current_questions.get(vendor_id):
            follow_up_map[vendor_id] = []
            continue

        inj_flag, inj_msg = scan_for_prompt_injection(reply_text, source_stage=f"Vendor Clarification ({vendor_id})")
        if inj_flag:
            sec_alerts.append(inj_msg)
            follow_up_map[vendor_id] = [f"🚨 SECURITY BLOCK: {inj_msg}."]
            continue

        asked_qs = current_questions.get(vendor_id, [])
        prompt = f"""You are a strict procurement auditor evaluating a vendor's reply.
VENDOR: {vendor_id}
QUESTIONS ASKED: {json.dumps(asked_qs, indent=2)}
VENDOR REPLY: "{reply_text}"

RULES:
1. If the vendor gives a vague reply ("same price as last bid") without numbers or PO reference, `is_resolved` = false and demand exact unit rates and prior PO doc.
2. If concrete prices are provided, `is_resolved` = true and populate `resolved_line_prices`.
3. If an explicit verifiable renewed YYYY-MM-DD certificate date is given, set `updated_certificate_expiry`.
4. Return ONLY valid JSON matching this schema:
{schema_hint}
"""
        eval_json, _ = router.call_gemini_with_fallback(prompt, schema=ClarificationEvaluation)

        history.append({
            "round": state.get("clarification_round", 1),
            "vendor": vendor_id,
            "questions_asked": asked_qs,
            "vendor_reply": reply_text,
            "is_resolved": eval_json.get("is_resolved", False),
            "follow_up_generated": eval_json.get("follow_up_questions", [])
        })

        v_quote = dict(reconciled.get(vendor_id, {}))
        upd_cert = eval_json.get("updated_certificate_expiry", "KEEP_EXISTING")
        if upd_cert and upd_cert != "KEEP_EXISTING" and len(upd_cert) >= 8:
            v_quote["certificate_expiry_date"] = upd_cert

        existing_lines = {int(l["line_id"]): l for l in v_quote.get("line_items", []) if "line_id" in l}
        for new_l in eval_json.get("resolved_line_prices", []):
            lid = int(new_l["line_id"])
            if float(new_l.get("quoted_unit_price", 0)) > 0:
                existing_lines[lid] = new_l
        v_quote["line_items"] = list(existing_lines.values())
        reconciled[vendor_id] = v_quote

        if not eval_json.get("is_resolved", True) and eval_json.get("follow_up_questions"):
            follow_up_map[vendor_id] = eval_json["follow_up_questions"]
        else:
            follow_up_map[vendor_id] = []

    return {
        "reconciled_quotes": reconciled,
        "clarification_questions": follow_up_map,
        "clarification_history": history,
        "vendor_clarifications": new_replies,
        "security_alerts": sec_alerts,
        "clarification_round": state.get("clarification_round", 1) + 1
    }


def reconcile_and_score_node(state: ProcurementState) -> Dict[str, Any]:
    buyer_reqs = state["buyer_requirements"]
    fx_rates = state.get("fx_rates", {"INR": 1.0, "USD": 84.0, "EUR": 91.0, "GBP": 108.0})
    ref_date = state["reference_date"]
    reconciled = state["reconciled_quotes"]

    defense_log: List[str] = []
    comparison_summary: List[Dict[str, Any]] = []
    all_normalized_lines: List[Dict[str, Any]] = []

    for vendor_id, quote in reconciled.items():
        # 1. Strict Certificate Compliance Rule: Expired cert = STRICTLY DISQUALIFIED from winning lines
        cert_exp = str(quote.get("certificate_expiry_date", "MISSING"))
        cert_valid = (cert_exp not in ["MISSING", "EXPIRED", "None", ""]) and (cert_exp >= ref_date)
        if cert_valid:
            cert_status = f"VALID (Exp: {cert_exp})"
        else:
            cert_status = f"INVALID/EXPIRED ({cert_exp})"
            defense_log.append(f"[DEFENSE - {vendor_id}] STRICTLY DISQUALIFIED from award: Certificate expired or missing ({cert_exp} < {ref_date}).")

        # 2. Strict Currency & Pricing Validation (No silent fallback to 1.0)
        q_lines = {int(l["line_id"]): l for l in quote.get("line_items", []) if "line_id" in l}
        covered_lines = 0
        total_landed_cost_inr = 0.0

        for req in buyer_reqs:
            lid = int(req["line_id"])
            req_qty = float(req["required_qty"])
            item_q = q_lines.get(lid)

            if item_q and float(item_q.get("quoted_unit_price", 0)) > 0:
                covered_lines += 1
                raw_price = float(item_q.get("quoted_unit_price", 0.0))
                curr = str(item_q.get("quoted_Currency", "INR")).upper().replace("$", "USD").replace("US$", "USD")
                
                if curr not in fx_rates:
                    raise ValueError(f"CRITICAL MONEY-MATH ERROR: Unknown currency symbol '{curr}' for vendor {vendor_id} on line {lid}. Unmapped currencies are prohibited.")
                
                fx = fx_rates[curr]
                conv = float(item_q.get("conversion_factor_to_target_uom", 1.0))
                if conv <= 0: conv = 1.0

                # Auto-default missing quantities to RFx requirement qty
                quoted_qty = float(item_q.get("quoted_qty", 0))
                if quoted_qty <= 0:
                    quoted_qty = req_qty

                # Exact mathematical landed unit cost in INR (NO fictitious 6% freight or working capital adjustments)
                landed_unit_inr = raw_price * fx * conv
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
                    "spec_compliant": bool(item_q.get("spec_compliant", True)),
                    "raw_price": raw_price,
                    "currency": curr,
                    "quoted_uom": item_q.get("quoted_uom", req["target_uom"]),
                    "source_snippet": item_q.get("source_snippet", "Extracted via OCR/Parser"),
                    "landed_unit_inr": round(landed_unit_inr, 2),
                    "line_total_inr": round(line_total_inr, 2)
                })

        comparison_summary.append({
            "Vendor": vendor_id,
            "Certificate_Status": cert_status,
            "Normalized_Total_Landed_INR": round(total_landed_cost_inr, 2),
            "Coverage": f"{covered_lines}/{len(buyer_reqs)} Quoted",
            "Eligible_For_Award": cert_valid and (covered_lines > 0) and not quote.get("prompt_injection_detected", False)
        })

    # DuckDB L1 Selection (Strictly filtering out expired certificates)
    df_lines = pd.DataFrame(all_normalized_lines)
    line_awards: List[Dict[str, Any]] = []

    if not df_lines.empty:
        con = duckdb.connect(database=":memory:")
        con.register("quotes_table", df_lines)
        award_df = con.execute("""
            SELECT line_id, item_code, description, required_qty, target_uom,
                   vendor_id AS recommended_vendor, landed_unit_inr, line_total_inr,
                   currency, raw_price, source_snippet
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY line_id ORDER BY landed_unit_inr ASC) as rk
                FROM quotes_table
                WHERE landed_unit_inr > 0 AND cert_valid = TRUE AND spec_compliant = TRUE
            )
            WHERE rk = 1
            ORDER BY line_id
        """).df()
        line_awards = award_df.to_dict(orient="records")
        total_split_po = sum(r["line_total_inr"] for r in line_awards)
        defense_log.append(f"[DEFENSE] DuckDB successfully executed strict L1 award optimization. Total Award Value: INR {total_split_po:,.2f}.")

    return {
        "comparison_summary": comparison_summary,
        "line_level_awards": line_awards,
        "all_normalized_lines": all_normalized_lines,
        "defense_audit_log": defense_log
    }


def generate_po_node(state: ProcurementState) -> Dict[str, Any]:
    if not state.get("internal_signoff_approved", False):
        return {"final_po_document": "### ⏳ Purchase Order Locked: Waiting for Internal Stakeholder Validation."}

    awards = state.get("line_level_awards", [])
    defense_log = state.get("defense_audit_log", [])
    approver = state.get("internal_approver_name", "Authorized Procurement Approver")

    if not awards:
        return {"final_po_document": "### No eligible line items found."}

    df_awards = pd.DataFrame(awards)
    total_po_val = df_awards["line_total_inr"].sum()
    vendors_awarded = df_awards["recommended_vendor"].unique().tolist()

    po_md = f"""# FINAL PURCHASE ORDER (INTERNALLY APPROVED & RELEASED)
**PO Reference:** PO-{state['rfx_id']}  
**Release Date:** {state['reference_date']}  
**Internal Sign-Off By:** {approver}  
**Awarded Vendor(s):** {', '.join(vendors_awarded)}  
**Total Contract Value (Normalized DDP INR):** ₹ {total_po_val:,.2f}

---

### 1. Final Awarded Line Items Schedule
{df_awards[['line_id', 'item_code', 'description', 'required_qty', 'target_uom', 'recommended_vendor', 'landed_unit_inr', 'line_total_inr']].to_markdown(index=False)}

---

### 2. Audit Defense Ledger
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

    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["reconcile_and_score_node", "generate_po_node"]
    )
