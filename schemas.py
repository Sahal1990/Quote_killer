from typing import List, Dict, Optional, TypedDict, Any
from pydantic import BaseModel, Field

class BuyerLineItem(BaseModel):
    line_id: int = Field(description="Line item number (1 to 30)")
    item_code: str = Field(description="SKU or Item Code")
    description: str = Field(description="Technical specification, e.g., 5-ply corrugated box")
    required_qty: float = Field(description="Quantity needed")
    target_uom: str = Field(description="Requested Unit of Measure, e.g., piece, box, kg")
    max_lead_time_days: int = Field(description="Maximum acceptable delivery lead time in days")

class ExtractedQuoteLine(BaseModel):
    line_id: int = Field(description="Matched line item number from RFx (1 to 30)")
    quoted_description: str = Field(description="What the vendor actually quoted")
    spec_compliant: bool = Field(description="True if matches buyer spec, False if deviation (e.g. 3-ply instead of 5-ply)")
    quoted_qty: float = Field(description="Quantity quoted by vendor (0 if line omitted)")
    quoted_unit_price: float = Field(description="Raw unit price as written in quote")
    quoted_Currency: str = Field(description="Currency code: INR, USD, EUR")
    quoted_uom: str = Field(description="Unit of measure quoted: piece, box, per_100_pcs, kg")
    conversion_factor_to_target_uom: float = Field(
        description="Multiplier to convert quoted_uom to buyer target_uom. E.g., if quoted per 100 pcs and buyer wants per piece, set to 0.01. Default 1.0."
    )
    line_discount_pct: float = Field(description="Line-specific or footnote discount percentage (0 to 100)")

class VendorQuoteExtraction(BaseModel):
    vendor_name: str = Field(description="Name of the quoting vendor")
    certificate_name: str = Field(description="ISO 9001, FSC, BIS, or None")
    certificate_expiry_date: str = Field(description="Expiry date in YYYY-MM-DD format, or 'MISSING' / 'EXPIRED'")
    incoterms: str = Field(description="DDP, CIF, EXW, FOB, or UNKNOWN")
    freight_included: bool = Field(description="True if freight is included in unit price, False if freight is extra")
    freight_extra_pct_or_flat: float = Field(description="If freight is extra, estimated percentage or flat surcharge noted, else 0.0")
    payment_terms_days: int = Field(description="Net payment days, e.g., 0 for Advance, 30 for Net 30, 60 for Net 60")
    delivery_lead_time_days: int = Field(description="Quoted delivery time in days")
    global_footnote_discount_pct: float = Field(description="Overall discount buried in footnotes or text (0 to 100)")
    ambiguities_detected: List[str] = Field(description="Any missing lines, unclear UoMs, blurry text, or unstated terms")
    line_items: List[ExtractedQuoteLine] = Field(description="List of extracted line items")

class ProcurementState(TypedDict):
    rfx_id: str
    reference_date: str
    fx_rates: Dict[str, float]
    buyer_requirements: List[Dict[str, Any]]
    rfx_document: str
    vendor_files: Dict[str, str]                  # vendor_id -> file_path
    gemini_extractions: Dict[str, Dict[str, Any]] # vendor_id -> VendorQuoteExtraction dict
    llama_extractions: Dict[str, Dict[str, Any]]  # vendor_id -> VendorQuoteExtraction dict
    discrepancy_log: Dict[str, List[str]]         # vendor_id -> list of LLM-vs-LLM or spec diffs
    clarification_questions: Dict[str, List[str]] # vendor_id -> questions to send
    vendor_clarifications: Dict[str, str]         # vendor_id -> vendor's reply text
    reconciled_quotes: Dict[str, Dict[str, Any]]  # vendor_id -> harmonized quote dict
    comparison_summary: List[Dict[str, Any]]      # Vendor-level 5-parameter scorecard
    line_level_awards: List[Dict[str, Any]]       # 30-line comparison & L1 winner per line
    defense_audit_log: List[str]                  # Code-verified defense statements
    final_po_document: str                        # Final Markdown PO
