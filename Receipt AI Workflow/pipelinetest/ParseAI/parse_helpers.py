import json
import os
import sys
from glob import glob
from typing import List, Optional

from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)
GROUPING_DIR = os.path.join(PIPELINE_ROOT, "GroupingAI")
GROUPING_IMAGES_DIR = os.path.join(GROUPING_DIR, "images")
GROUPING_OPTIMIZED_DIR = os.path.join(GROUPING_DIR, "optimized_images")
TRANSCRIPTION_OPTIMIZED_DIR = os.path.join(SCRIPT_DIR, "optimized_images")
TRANSCRIPTION_OUTPUT_DIR = SCRIPT_DIR

if PIPELINE_ROOT not in sys.path:
    sys.path.append(PIPELINE_ROOT)

from common_helpers import (  # noqa: E402
    encode_image,
    make_schema_strict,
    natural_sort_key,
    optimize_receipt_image,
    safe_file_name,
)


class Item(BaseModel):
    item_name: str = Field(description="The clean name of the product. Remove employee/staff designations.")
    item_code: Optional[str] = Field(None, description="SKU, barcode, or item code if available.")
    quantity: int = Field(description="Total units purchased. Keep an eye on vertical column alignment.")
    discount: float = Field(
        0.0,
        description="Per-unit discount for this item. If the receipt shows a total line discount, divide by quantity to store the per-unit value. If multiple discounts are listed for the same line, combine them first before dividing by quantity.",
    )
    price: float = Field(description="The single unit price of the item.")
    remarks: Optional[str] = Field(None, description="Any promotional notes, like 'Staff 15%' or 'GWP'.")

    @property
    def subtotal(self) -> float:
        return self.quantity * self.price


class AutoReceipt(BaseModel):
    store_name: str = Field(description="Name of the vendor/merchant.")
    receipt_no: Optional[str] = Field(None, description="Invoice or receipt reference number.")
    receipt_date: Optional[str] = Field("", description="Date in YYYY-MM-DD format if visible.")
    customer_id: int = Field(0, description="Customer identification number. Defaults to 0 if missing.")
    items: List[Item] = Field(default_factory=list, description="Array of extracted line items.")
    total: float = Field(description="The final total amount printed on the receipt.")
    tax: float = Field(0.09, description="The tax rate (e.g. 0.09 for 9%).")
    commission: float = Field(0.05, description="The commission calculated on the subtotal. Default is 0.05.")
    remarks: Optional[str] = Field(None, description="General receipt comments, refund policies, or extra context.")
    amounts_are_post_tax: bool = Field(
        description="True if individual item prices ALREADY include the GST/tax. False if item prices are pre-tax and tax is added at the bottom."
    )
    is_balanced: bool = Field(
        description="Math validation flag. Returns True if the line items perfectly sum to the subtotal and add up to the final total."
    )


class TransactionGroup(BaseModel):
    transaction_index: int = Field(description="1-based transaction grouping index.")
    image_filenames: List[str] = Field(description="The filenames that belong to this grouped transaction.")
    reason: str = Field(description="Why these images were grouped together.")


class TransactionGroups(BaseModel):
    transactions: List[TransactionGroup] = Field(default_factory=list)


def list_image_files(image_dir: str) -> List[str]:
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    files = []
    for entry in os.listdir(image_dir):
        full_path = os.path.join(image_dir, entry)
        if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in allowed_extensions:
            files.append(full_path)

    return sorted(files, key=lambda path: natural_sort_key(os.path.basename(path)))


def load_grouped_transactions(manual_json: str | None = None) -> List[TransactionGroup]:
    if manual_json is not None:
        manual_json = manual_json.strip()
        if not manual_json:
            raise ValueError("manual_json was provided but is empty.")

        raw_data = json.loads(manual_json)
        if isinstance(raw_data, dict) and "transactions" in raw_data:
            return [TransactionGroup.model_validate(item) for item in raw_data["transactions"]]
        if isinstance(raw_data, list):
            return [TransactionGroup.model_validate(item) for item in raw_data]

        raise ValueError("manual_json must be a JSON array or an object with a transactions field.")

    candidates = sorted(
        glob(os.path.join(GROUPING_DIR, "grouped_transactions*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No grouping file found in {GROUPING_DIR}. Run the grouping stage first."
        )

    latest_grouping_file = candidates[0]
    with open(latest_grouping_file, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if isinstance(raw_data, dict) and "transactions" in raw_data:
        return [TransactionGroup.model_validate(item) for item in raw_data["transactions"]]
    if isinstance(raw_data, list):
        return [TransactionGroup.model_validate(item) for item in raw_data]

    raise ValueError(f"Unexpected grouping file format in {latest_grouping_file}")


def build_transcription_prompt(group: TransactionGroup) -> str:
    filenames_block = "\n".join(f"- {name}" for name in group.image_filenames)
    return f"""You are an expert financial document layout auditor. Analyze the attached sequence of receipt images, which are continuous page segments of the same single transaction. Perform an incredibly precise visual transcription.

CRITICAL RULES:
1. **Continuous Scroll Handling**: Treat these images as a single continuous document stitched top-to-bottom. Read across page breaks smoothly. If an item line begins at the bottom of Image 1 and its discount rows wrap to the top of Image 2, associate them with the same item.
2. **Column Alignment Security**: Ensure you track vertical columns safely. Do not let surrounding numeric item codes infect the 'Quantity', 'Unit Price', or 'Total' values.
3. **Staff/Identity Scraping**: Strip out human names that denote who bought the item or who got a staff perk (e.g., 'Nor Senen'). Remove them completely from the product name line.
4. **Raw Multi-Line Discount Capture**: Items often have multiple stacked discount lines beneath them (e.g., 'STR Staff 15%', 'UP1800FSK', 'GSTSAVING'). You MUST capture every single negative value line individually under its respective item. Do NOT combine them, do NOT average them, and do NOT perform any mathematical division. Simply transcribe the exact raw names and negative amounts as printed.
5. **GST/Tax Strategy Analysis**: Identify if prices are inclusive of GST (look for 'GST INCL' or 'GST Gross' tags) or if tax is added explicitly at the very bottom of the second page.

Output a beautifully structured markdown transcription containing:
- Store Name (from Page 1)
- Receipt Metadata (Date, Receipt No, etc.)
- A detailed item list where each item block looks exactly like this template:
  * ITEM: [Actual Cleaned Product Name]
  * QTY: [Quantity] | UNIT_PRICE: [Unit Price] | LINE_TOTAL: [Line Total]
  * APPLIED_DISCOUNTS:
    - [Discount Name 1]: [Negative Value 1]
    - [Discount Name 2]: [Negative Value 2]
- The final payment totals, subtotal, and GST summaries found at the end of the final page.

Transcription grouping context:
- transaction_index: {group.transaction_index}
- reason: {group.reason}
"""


def build_parse_prompt(group: TransactionGroup) -> str:
    return build_transcription_prompt(group)


def resolve_image_path(filename: str) -> str:
    candidates = [
        os.path.join(GROUPING_IMAGES_DIR, filename),
        os.path.join(TRANSCRIPTION_OPTIMIZED_DIR, filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find image for filename: {filename}")


def prepare_image_for_transcription(image_path: str) -> str:
    os.makedirs(TRANSCRIPTION_OPTIMIZED_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    optimized_path = os.path.join(TRANSCRIPTION_OPTIMIZED_DIR, f"{safe_file_name(base_name)}.jpg")
    if not os.path.exists(optimized_path):
        optimize_receipt_image(image_path, optimized_path)
    return optimized_path


def prepare_image_for_parse(image_path: str) -> str:
    return prepare_image_for_transcription(image_path)
