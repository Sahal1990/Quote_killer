import os
import re
import io
import json
import base64
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
from typing import Dict, Any, Tuple, List
from google import genai
from google.genai import types
from groq import Groq

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACKS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
GROQ_LLAMA4_MODEL = os.getenv("GROQ_LLAMA4_MODEL", "llama-3.3-70b-versatile").strip()

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"system\s+override",
    r"disregard\s+(system|previous)\s+prompt",
    r"you\s+are\s+now\s+an?\s+unrestricted",
    r"forget\s+all\s+prior\s+rules",
    r"award\s+(the\s+)?(contract|po|order)\s+to",
    r"mark\s+this\s+vendor\s+as\s+l1",
    r"override\s+price\s+to\s+0",
    r"<\s*script\s*>",
    r"drop\s+table",
]

def _clean_key(key: str) -> str:
    if not key:
        return ""
    return key.strip().strip('"').strip("'")

def scan_for_prompt_injection(text_content: str, source_stage: str = "Input") -> Tuple[bool, str]:
    if not text_content:
        return False, "None"
    lower_text = text_content.lower()
    for pattern in INJECTION_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            return True, f"[{source_stage}] Blocked suspicious instruction override pattern: '{match.group(0)}'"
    return False, "None"

def parse_buyer_requirements(file_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    df = df.dropna(how="all").reset_index(drop=True)
    records = []
    security_flags = []

    for idx, row in df.iterrows():
        row_str = " ".join(str(v) for v in row.values)
        inj_detected, inj_msg = scan_for_prompt_injection(row_str, source_stage=f"Buyer Requirement Row {idx+1}")
        if inj_detected:
            security_flags.append(inj_msg)

        row_dict = {str(k).strip().lower(): v for k, v in row.to_dict().items()}
        records.append({
            "line_id": int(row_dict.get("line_id", row_dict.get("s.no", row_dict.get("id", idx + 1)))),
            "item_code": str(row_dict.get("item_code", row_dict.get("sku", row_dict.get("code", f"ITEM-{idx+1:02d}")))),
            "description": str(row_dict.get("description", row_dict.get("item", row_dict.get("specification", list(row_dict.values())[0])))),
            "required_qty": float(row_dict.get("required_qty", row_dict.get("qty", row_dict.get("quantity", 100.0)))),
            "target_uom": str(row_dict.get("target_uom", row_dict.get("uom", row_dict.get("unit", "piece")))).lower(),
            "max_lead_time_days": int(row_dict.get("max_lead_time_days", row_dict.get("lead_time", 14)))
        })
    return records, security_flags

def extract_multimodal_payload(file_path: str) -> Tuple[str, List[bytes]]:
    ext = os.path.splitext(file_path)[-1].lower()
    extracted_text = f"--- SOURCE FILE: {os.path.basename(file_path)} ---\n"
    images_bytes: List[bytes] = []

    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            extracted_text += f"\n[Page {page_num + 1}]\n" + page.get_text("text")
            pix = page.get_pixmap(dpi=150)
            images_bytes.append(pix.tobytes("png"))
        doc.close()
    elif ext == ".docx":
        doc = Document(file_path)
        for p in doc.paragraphs:
            if p.text.strip():
                extracted_text += p.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                extracted_text += " | ".join(cell.text.strip() for cell in row.cells) + "\n"
    elif ext in [".xlsx", ".xls", ".csv"]:
        df_dict = pd.read_excel(file_path, sheet_name=None) if ext != ".csv" else {"Sheet1": pd.read_csv(file_path)}
        for sheet_name, df in df_dict.items():
            extracted_text += f"\n[Sheet: {sheet_name}]\n" + df.to_markdown(index=False) + "\n"
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            buf = io.BytesIO()
            img_rgb.save(buf, format="PNG")
            images_bytes.append(buf.getvalue())
        extracted_text += "[Visual Quote Image Attached - Inspect image carefully for angled text, prices, and footnotes.]\n"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            extracted_text += f.read()

    return extracted_text, images_bytes

class DualLLMRouter:
    def __init__(self, gemini_api_key: str = None, groq_api_key: str = None):
        self.gemini_key = _clean_key(gemini_api_key or os.getenv("GEMINI_API_KEY", ""))
        self.groq_key = _clean_key(groq_api_key or os.getenv("GROQ_API_KEY", ""))
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def call_gemini_with_fallback(self, prompt: str, images_bytes: List[bytes] = None, schema=None) -> Tuple[Any, str]:
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is missing or empty.")
        primary = os.getenv("PRIMARY_GEMINI_MODEL", GEMINI_MODEL).strip()
        models_to_try = list(dict.fromkeys([primary] + GEMINI_FALLBACKS))
        errors_log = []

        contents = [prompt]
        if images_bytes:
            for img_b in images_bytes[:2]:
                contents.append(types.Part.from_bytes(data=img_b, mime_type="image/png"))

        config_kwargs = {"temperature": 0.1}
        if schema:
            config_kwargs["response_mime_type"] = "application/json"

        for model_name in models_to_try:
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name, contents=contents, config=types.GenerateContentConfig(**config_kwargs)
                )
                text_out = getattr(response, "text", None) or ""
                text_out = text_out.strip()
                if schema:
                    if text_out.startswith("```"):
                        text_out = text_out.split("```")[1]
                        if text_out.startswith("json"):
                            text_out = text_out[4:]
                    return json.loads(text_out.strip()), model_name
                return text_out, model_name
            except Exception as e:
                errors_log.append(f"[{model_name}]: {str(e)}")
                continue
        raise RuntimeError("All Gemini models failed: " + " | ".join(errors_log))

    def call_groq_llama4(self, prompt: str, images_bytes: List[bytes] = None, json_mode: bool = True) -> Tuple[Any, str]:
        if not self.groq_client:
            raise ValueError("GROQ_API_KEY is missing or empty.")
        model_to_use = os.getenv("GROQ_LLAMA4_MODEL", GROQ_LLAMA4_MODEL).strip()
        
        # If images are present, append OCR/vision note for Groq text model
        text_prompt = prompt
        if images_bytes:
            text_prompt += "\n[Note: This vendor quote was provided via an image/photo attachment. Extract carefully from OCR text or mark confidence as visual-source if ambiguous.]"

        kwargs = {"model": model_to_use, "messages": [{"role": "user", "content": text_prompt}], "temperature": 0.1}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self.groq_client.chat.completions.create(**kwargs)
        raw_text = completion.choices[0].message.content.strip()
        if json_mode:
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            return json.loads(raw_text.strip()), model_to_use
        return raw_text, model_to_use
