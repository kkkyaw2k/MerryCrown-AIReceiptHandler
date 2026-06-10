import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Optional

import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from pipelinetest.common_helpers import make_schema_strict, optimize_receipt_image, safe_file_name


BASE_DIR = Path(__file__).resolve().parent
OPTIMIZED_IMAGE_DIR = BASE_DIR / "optimized_images"
STAGE1_MODEL = "google/gemini-2.5-flash"  # Replace with your chosen model for stage 1 transcription
STAGE2_MODEL = "openai/gpt-5.4-nano"  # Replace with your chosen model for stage 2 parsing
MANUAL_IMAGE_PATH = r"Z:\Project REPO\MerryCrown-Backend\pipelinetest\ParseAI\optimized_images\photo2.jpg"


class Item(BaseModel):
    item_name: str = Field(
        description="The actual product name. Remove employee/staff names that appear near the item.",
    )
    item_code: Optional[str] = Field(default=None, description="SKU, barcode, or item code if present.")
    quantity: int = Field(description="The quantity shown in the QTY column.")
    price: float = Field(description="The printed unit price for one item.")
    line_subtotal: float = Field(
        description="The printed line total before discounts. Use this to cross-check quantity * price.",
    )
    discount: float = Field(
        default=0.0,
        description=(
            "Per-unit discount for this item. If the receipt shows a discount subtotal or stacked discount lines "
            "beneath the item, combine those line discounts first, then divide by quantity to store the per-unit value."
        ),
    )
    remarks: Optional[str] = Field(default=None, description="Extra notes, including staff or promo context.")

    @property
    def subtotal(self) -> float:
        return self.quantity * self.price

    @property
    def discounted_line_total(self) -> float:
        return self.line_subtotal - (self.quantity * self.discount)


class AutoReceipt(BaseModel):
    store_name: str = Field(description="The name of the store, merchant, or vendor.")
    receipt_no: Optional[str] = Field(default=None, description="Receipt number, invoice ID, or transaction number.")
    receipt_date: Optional[str] = Field(default=None, description="Date of the receipt, ideally in YYYY-MM-DD format.")
    customer_id: int = Field(default=0, description="Customer ID if explicitly printed; default to 0.")
    items: list[Item] = Field(default_factory=list, description="The extracted receipt line items.")
    total: float = Field(description="The final grand total paid as stated on the receipt.")
    tax: float = Field(default=0.09, description="Tax rate applied to the receipt, default 9%.")
    remarks: Optional[str] = Field(default=None, description="General remarks or extra context.")
    amounts_are_post_tax: bool = Field(
        default=False,
        description="True when item prices already include tax. False when tax is added at the bottom.",
    )
    is_balanced: bool = Field(
        default=False,
        description="True when line items, discounts, tax, and total all reconcile within a small tolerance.",
    )

    @property
    def gross_subtotal(self) -> float:
        return sum(item.line_subtotal for item in self.items)

    @property
    def total_discount(self) -> float:
        return sum(item.discount * item.quantity for item in self.items)

    @property
    def net_subtotal(self) -> float:
        return self.gross_subtotal - self.total_discount

    @property
    def expected_total(self) -> float:
        if self.amounts_are_post_tax:
            return self.net_subtotal
        return self.net_subtotal * (1 + self.tax)


STRICT_AUTO_RECEIPT_SCHEMA = make_schema_strict(AutoReceipt.model_json_schema())


def build_stage1_prompt() -> str:
    return """You are an expert financial document layout auditor. Analyze this single receipt image and produce a precise markdown transcription.

CRITICAL RULES:
1. Column alignment matters. Keep QTY, UNIT PRICE, and LINE SUBTOTAL in their correct vertical columns. Never borrow digits from nearby item codes or category prefixes.
2. Remove human names from `item_name` when they clearly represent staff or customer identities. Keep that context in `remarks` instead.
3. Preserve every printed discount line and promo note. If the receipt shows stacked discount lines below a single item, capture each line separately under that item in the markdown instead of collapsing or averaging them.
4. Identify whether prices are post-tax or pre-tax. Explain the GST/tax strategy in a short analysis section.
5. If the math does not fully reconcile, state that in the markdown and do not invent values.

Output markdown only, with these sections:
### Store Name
### Receipt Metadata
### GST/Tax Strategy Analysis
### Items Table
Use a table with columns: Product Name | Qty | Unit Price | Line Subtotal | Per-Unit Discount | Remarks
### Explicit Discount Figures
### Totals & Payment
### Math Notes
"""


def build_stage2_prompt() -> str:
    return """You are a receipt schema enforcement service. Convert the markdown transcription into the exact JSON schema.

Rules:
1. Use the explicit markdown values first. Do not invent prices, quantities, totals, or dates.
2. Keep `discount` as a per-unit value. If the markdown shows a discount subtotal, or multiple discount rows below the same item, combine those line discounts into one total discount first, then divide by quantity.
3. Set `amounts_are_post_tax` according to the markdown tax analysis.
4. Set `is_balanced` to true only when the items, discounts, tax treatment, and final total reconcile within a 0.02 tolerance.
5. If the markdown is ambiguous, preserve the best explicit values and mark `is_balanced` false.
6. Output only JSON matching the schema.
"""


def detect_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/jpeg"


def prepare_image_for_stage1(image_path: Path) -> Path:
    OPTIMIZED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    optimized_path = OPTIMIZED_IMAGE_DIR / f"{safe_file_name(image_path.stem)}.jpg"
    if not optimized_path.exists():
        optimize_receipt_image(str(image_path), str(optimized_path))
    return optimized_path


def stage1_transcribe_to_markdown(client: OpenAI, image_path: Path) -> str:
    optimized_path = prepare_image_for_stage1(image_path)
    base64_image = base64.b64encode(optimized_path.read_bytes()).decode("utf-8")

    response = client.chat.completions.create(
        model=STAGE1_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_stage1_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{detect_mime_type(optimized_path)};base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
    )

    markdown_text = (response.choices[0].message.content or "").strip()
    if not markdown_text:
        raise RuntimeError("Stage 1 transcription returned no markdown text.")
    return markdown_text


def stage2_parse_markdown_to_receipt(client: OpenAI, markdown_text: str) -> AutoReceipt:
    response = client.chat.completions.create(
        model=STAGE2_MODEL,
        messages=[
            {"role": "system", "content": build_stage2_prompt()},
            {"role": "user", "content": f"Markdown transcription:\n\n{markdown_text}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AutoReceipt",
                "strict": True,
                "schema": STRICT_AUTO_RECEIPT_SCHEMA,
            },
        },
        temperature=0.0,
    )

    json_text = (response.choices[0].message.content or "").strip()
    if not json_text:
        raise RuntimeError("Stage 2 schema parsing returned no JSON text.")
    return AutoReceipt.model_validate_json(json_text)


def evaluate_receipt(receipt: AutoReceipt) -> dict[str, float | bool]:
    expected_total = receipt.expected_total
    delta = abs(receipt.total - expected_total)
    is_balanced = delta <= 0.02
    receipt.is_balanced = is_balanced
    return {
        "gross_subtotal": receipt.gross_subtotal,
        "total_discount": receipt.total_discount,
        "net_subtotal": receipt.net_subtotal,
        "expected_total": expected_total,
        "delta": delta,
        "is_balanced": is_balanced,
    }


def save_markdown(markdown_text: str, image_path: Path) -> Path:
    output_path = BASE_DIR / f"transcribed_receipt_{safe_file_name(STAGE1_MODEL)}_{safe_file_name(image_path.stem)}.md"
    output_path.write_text(markdown_text, encoding="utf-8")
    return output_path


def save_receipt_json(receipt: AutoReceipt, image_path: Path) -> Path:
    output_path = BASE_DIR / f"final_receipt_object_{safe_file_name(STAGE1_MODEL)}_{safe_file_name(STAGE2_MODEL)}_{safe_file_name(image_path.stem)}.json"
    output_path.write_text(json.dumps(receipt.model_dump(), indent=4), encoding="utf-8")
    return output_path


def print_pipeline_summary(image_path: Path, markdown_path: Path, receipt_path: Path, receipt: AutoReceipt, validation: dict[str, float | bool]) -> None:
    print()
    print("==========================================")
    print("SINGLE RECEIPT PIPELINE COMPLETE")
    print("==========================================")
    print(f"Stage 1 model: {STAGE1_MODEL}")
    print(f"Stage 2 model: {STAGE2_MODEL}")
    print(f"Source image: {image_path}")
    print(f"Markdown saved to: {markdown_path}")
    print(f"JSON saved to: {receipt_path}")
    print()
    print(f"Store: {receipt.store_name}")
    print(f"Receipt total: {receipt.total:.2f}")
    print(f"Gross subtotal: {validation['gross_subtotal']:.2f}")
    print(f"Discount total: {validation['total_discount']:.2f}")
    print(f"Expected total: {validation['expected_total']:.2f}")
    print(f"Delta: {validation['delta']:.2f}")
    print(f"Balanced: {validation['is_balanced']}")
    print(f"Items extracted: {len(receipt.items)}")


def process_single_receipt(image_path: str) -> AutoReceipt:
    dotenv.load_dotenv()

    source_image = Path(image_path).expanduser().resolve()
    if not source_image.exists():
        raise FileNotFoundError(f"Image not found: {source_image}")

    openrouter_key = os.getenv("OPENROUTER_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_KEY is not set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    markdown_text = stage1_transcribe_to_markdown(client, source_image)
    markdown_path = save_markdown(markdown_text, source_image)

    receipt = stage2_parse_markdown_to_receipt(client, markdown_text)
    validation = evaluate_receipt(receipt)
    receipt_path = save_receipt_json(receipt, source_image)

    print_pipeline_summary(source_image, markdown_path, receipt_path, receipt, validation)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-receipt 2-stage transcription pipeline.")
    parser.add_argument("image_path", nargs="?", help="Path to the receipt image to parse.")
    args = parser.parse_args()

    image_path = MANUAL_IMAGE_PATH.strip() or args.image_path
    if not image_path:
        raise RuntimeError("Set MANUAL_IMAGE_PATH at the top of the file or pass an image_path argument.")

    try:
        receipt = process_single_receipt(image_path)
        print()
        print("Parsed receipt object:")
        print(receipt.model_dump_json(indent=2))
    except Exception as error:
        print(f"\n❌ Pipeline failed: {error}")


if __name__ == "__main__":
    main()