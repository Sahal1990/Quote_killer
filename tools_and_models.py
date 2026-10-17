import os
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
from schemas import VendorQuoteExtraction

# Configurable Model Strings (Supports user override + automatic Gemini family fallback)
PRIMARY_GEMINI_MODEL = os.getenv("PRIMARY_GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash-lite"]
GROQ_LLAMA4_MODEL = os.getenv("GROQ_LLAMA4_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def parse_buyer_requirements(file_path: str) -> List[Dict[str, Any]]:
    """Reads the buyer's 30-line-item spreadsheet (.xlsx or .csv) and standardizes columns."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    df = df.dropna(how="all").reset_index(drop=True)
    records = []
    for idx, row in df.iterrows():
        # Flexible column mapping so your existing 30-line sheet works out of the box
        row_dict = {str(k).strip().lower(): v for k, v in row.to_dict().items()}
        records.append({
            "line_id": int(row_dict.get("line_id", row_dict.get("s.no", row_dict.get("id", idx + 1)))),
            "item_code": str(row_dict.get("item_code", row_dict.get("sku", row_dict.get("code", f"ITEM-{idx+1:02d}")))),
            "description": str(row_dict.get("description", row_dict.get("item", row_dict.get("specification", list(row_dict.values())[0])))),
            "required_qty": float(row_dict.get("required_qty", row_dict.get("qty", row_dict.get("quantity", 100.0)))),
            "target_uom": str(row_dict.get("target_uom", row_dict.get("uom", row_dict.get("unit", "piece")))).lower(),
            "max_lead_time_days": int(row_dict.get("max_lead_time_days", row_dict.get("lead_time", 14)))
        })
    return records


def extract_multimodal_payload(file_path: str) -> Tuple[str, List[bytes]]:
    """Extracts raw text/tables and image bytes from PDF, DOCX, XLSX, PNG/JPG, or TXT."""
    ext = os.path.splitext(file_path)[-1].lower()
    extracted_text = f"--- SOURCE FILE: {os.path.basename(file_path)} ---\n"
    images_bytes: List[bytes] = []

    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            extracted_text += f"\n[Page {page_num + 1}]\n" + page.get_text("text")
            # Render page as image in case of scanned PDF or angled photo inside PDF
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
    """Manages Gemini Primary + Fallback family routing and Groq Llama 4 parallel execution."""
    def __init__(self, gemini_api_key: str = None, groq_api_key: str = None):
        self.gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def call_gemini_with_fallback(self, prompt: str, images_bytes: List[bytes] = None, schema=None) -> Tuple[Any, str]:
        """Attempts primary Gemini model, automatically falling back to backup Gemini models on traffic/quota errors."""
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not set.")

        models_to_try = [PRIMARY_GEMINI_MODEL] + FALLBACK_GEMINI_MODELS
        last_error = None

        contents = [prompt]
        if images_bytes:
            for img_b in images_bytes[:2]:  # Cap at 2 images for speed/token safety
                contents.append(types.Part.from_bytes(data=img_b, mime_type="image/png"))

        config_kwargs = {"temperature": 0.1}
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema

        for model_name in models_to_try:
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                if schema:
                    return json.loads(response.text), model_name
                return response.text, model_name
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    def call_groq_llama4(self, prompt: str, images_bytes: List[bytes] = None, json_mode: bool = True) -> Tuple[Any, str]:
        """Calls Llama 4 Scout on Groq with vision + JSON schema enforcement."""
        if not self.groq_client:
            raise ValueError("GROQ_API_KEY is not set.")

        user_content = [{"type": "text", "text": prompt}]
        if images_bytes:
            b64_img = base64.b64encode(images_bytes[0]).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_img}"}
            })

        kwargs = {
            "model": GROQ_LLAMA4_MODEL,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.1,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self.groq_client.chat.completions.create(**kwargs)
        raw_text = completion.choices[0].message.content
        if json_mode:
            return json.loads(raw_text), GROQ_LLAMA4_MODEL
        return raw_text, GROQ_LLAMA4_MODEL
