from typing import List, Dict, Optional, TypedDict, Any
from pydantic import BaseModel, Field

class BuyerLineItem(BaseModel):
    line_id: int
    item_code: str
    description: str
    required_qty: float
    target_uom: str
    max_lead_time_days: int

class ExtractedQuoteLine(BaseModel):
    line_id: int
    quoted_description: str
    spec_compliant: bool
    quoted_qty: float
    quoted_unit_price: float
    quoted_Currency: str
    quoted_uom: str
    conversion_factor_to_target_uom: float = 1.0
    line_discount_pct: float = 0.0
    source_snippet: str = Field(default="", description="Exact text snippet from vendor document where this price/line was found")

class VendorQuoteExtraction(BaseModel):
    vendor_name: str
    certificate_name: str
    certificate_expiry_date: str
    incoterms: str
    freight_included: bool
    freight_extra_pct_or_flat: float = 0.0
    payment_terms_days: int = 30
    delivery_lead_time_days: int = 14
    global_footnote_discount_pct: float = 0.0
    ambiguities_detected: List[str] = []
    line_items: List[ExtractedQuoteLine] = []
    prompt_injection_detected: bool = Field(default=False)
    injection_details: str = Field(default="None")
    extraction_confidence_score: float = Field(default=0.95)

class ClarificationEvaluation(BaseModel):
    is_resolved: bool = Field(description="False if vendor gave vague reply like 'same as last bid' without actual numbers or PO/quote doc reference")
    follow_up_questions: List[str] = Field(description="Specific follow-up questions asking for exact unit rates or prior PO/quote document")
    updated_certificate_expiry: str = Field(description="YYYY-MM-DD if vendor provided explicit verifiable renewed date, else KEEP_EXISTING")
    updated_freight_extra_pct: float = Field(description="Confirmed freight percentage if clarified, else -1.0")
    resolved_line_prices: List[ExtractedQuoteLine] = Field(description="Line items where vendor provided concrete unit prices")

class ProcurementState(TypedDict):
    rfx_id: str
    reference_date: str
    fx_rates: Dict[str, float]
    buyer_requirements: List[Dict[str, Any]]
    rfx_document: str
    security_alerts: List[str]
    vendor_files: Dict[str, str]
    gemini_extractions: Dict[str, Dict[str, Any]]
    llama_extractions: Dict[str, Dict[str, Any]]
    discrepancy_log: Dict[str, List[Dict[str, Any]]]
    clarification_questions: Dict[str, List[str]]
    clarification_round: int
    clarification_history: List[Dict[str, Any]]
    vendor_clarifications: Dict[str, str]
    reconciled_quotes: Dict[str, Dict[str, Any]]
    comparison_summary: List[Dict[str, Any]]
    line_level_awards: List[Dict[str, Any]]
    all_normalized_lines: List[Dict[str, Any]]  # Raw normalized flat table for Analyst chat & export
    defense_audit_log: List[str]
    internal_signoff_approved: bool
    internal_approver_name: str
    final_po_document: str
