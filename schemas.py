from typing import List, Dict, TypedDict, Any
from pydantic import BaseModel, Field

# Default buyer questionnaire. Editable in the application sidebar.
DEFAULT_QUESTIONNAIRE: List[Dict[str, Any]] = [
    {"id": "Q1", "mandatory": True, "text": "Does the vendor hold a valid ISO 9001 (or equivalent) quality certification? State the certificate number and expiry date."},
    {"id": "Q2", "mandatory": True, "text": "Can all quoted items be delivered within the lead times stated in the RFx?"},
    {"id": "Q3", "mandatory": False, "text": "Does the vendor accept the payment terms stated in the RFx?"},
    {"id": "Q4", "mandatory": True, "text": "Will a batch-wise quality inspection report accompany every dispatch?"},
    {"id": "Q5", "mandatory": False, "text": "Is the vendor able to supply pre-production samples for approval?"},
]


class BuyerLineItem(BaseModel):
    line_id: int
    item_code: str
    description: str
    required_qty: float
    target_uom: str
    max_lead_time_days: int


class QuestionnaireAnswer(BaseModel):
    question_id: str
    answer: str = "NOT_ANSWERED"  # YES | NO | PARTIAL | NOT_ANSWERED
    evidence: str = ""


class ExtractedQuoteLine(BaseModel):
    line_id: int
    quoted_description: str = ""
    spec_compliant: bool = True
    quoted_qty: float = 0.0
    quoted_unit_price: float = 0.0
    quoted_currency: str = "INR"
    quoted_uom: str = ""
    conversion_factor_to_target_uom: float = Field(default=1.0, description="Multiplier converting a price per quoted UoM into a price per RFx UoM")
    line_discount_pct: float = 0.0
    confidence: float = 0.9
    source_snippet: str = Field(default="", description="Exact text from the vendor document where this price was found")


class VendorQuoteExtraction(BaseModel):
    vendor_name: str = ""
    certificate_name: str = "MISSING"
    certificate_expiry_date: str = "MISSING"
    incoterms: str = "UNSPECIFIED"
    freight_terms: str = "UNSPECIFIED"  # INCLUDED | EXTRA_PCT | EXTRA_FLAT | EXTRA_UNQUANTIFIED | UNSPECIFIED
    freight_extra_pct: float = 0.0
    freight_extra_flat_inr: float = 0.0
    payment_terms_days: int = 30
    delivery_lead_time_days: int = 14
    global_footnote_discount_pct: float = 0.0
    questionnaire: List[QuestionnaireAnswer] = []
    ambiguities_detected: List[str] = []
    line_items: List[ExtractedQuoteLine] = []
    prompt_injection_detected: bool = False
    injection_details: str = "None"
    extraction_confidence_score: float = 0.95


class ClarificationEvaluation(BaseModel):
    is_resolved: bool = Field(default=False, description="False if the reply is vague (for example 'same as last bid') without figures or a document reference")
    follow_up_questions: List[str] = []
    updated_certificate_expiry: str = "KEEP_EXISTING"
    updated_freight_extra_pct: float = -1.0
    resolved_line_prices: List[ExtractedQuoteLine] = []
    questionnaire_updates: List[QuestionnaireAnswer] = []


class ProcurementState(TypedDict):
    rfx_id: str
    reference_date: str
    fx_rates: Dict[str, float]
    buyer_requirements: List[Dict[str, Any]]
    questionnaire: List[Dict[str, Any]]
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
    all_normalized_lines: List[Dict[str, Any]]
    review_queue: List[Dict[str, Any]]
    defense_audit_log: List[str]
    internal_signoff_approved: bool
    internal_approver_name: str
    final_po_document: str
