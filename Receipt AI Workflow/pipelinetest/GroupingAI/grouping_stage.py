import base64
import json
import os
import re
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
import dotenv
dotenv.load_dotenv()

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
OPTIMIZED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimized_images")
stage1_client_model = "google/gemini-2.5-flash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import PIL.Image
import PIL.ImageEnhance

def optimize_receipt_image(image_path: str, target_path: str):
    img = PIL.Image.open(image_path)
    
    # 1. Downscale maximum bound to 1200px (Legible for OCR text, but drops token grids massively)
    img.thumbnail((1200, 1200), PIL.Image.Resampling.LANCZOS)
    
    # 2. Convert to Grayscale (removes extra color layer data overhead)
    img = img.convert("L")
    
    # 3. Boost contrast to make text pop out sharply against backdrops
    enhancer = PIL.ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0) 
    
    img.save(target_path, "JPEG", quality=80)

def safe_file_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


MODEL_TAG = safe_file_name(stage1_client_model)


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


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


def build_grouping_prompt(image_filenames: List[str]) -> str:
    filenames_block = "\n".join(f"- {name}" for name in image_filenames)
    return f"""You are a document triage assistant. The user has uploaded receipt images. Some images are completely independent receipts from separate stores. Others are sequential continuations (Page 1, Page 2, Page 3) of the same physical long receipt.

Analyze the visual layouts, text headers, transaction numbers, and item flows across the sequence. Group them into distinct chronological logical transactions.

Return JSON only. No markdown, no code fences, no commentary.

Output schema:
{{
  "transactions": [
    {{
      "transaction_index": 1,
      "image_filenames": ["img_01.jpg", "img_02.jpg"],
      "reason": "img_02 is a continuous printout matching the items and font layout of img_01 from Shilla Duty Free"
    }}
  ]
}}

Rules:
1. Group images that belong to the same physical receipt, including page continuations.
2. Treat the images as a single continuous document stitched top-to-bottom when needed.
3. Read across page breaks smoothly. If a line begins at the bottom of one image and wraps to the top of the next, consider them part of the same transaction.
4. Extract store metadata from whichever page contains it, usually the first page.
5. Preserve the original image filenames exactly as given.

Image filename order:
{filenames_block}"""


class TransactionGroup(BaseModel):
    transaction_index: int = Field(description="1-based transaction grouping index.")
    image_filenames: List[str] = Field(description="The filenames that belong to this grouped transaction.")
    reason: str = Field(description="Why these images were grouped together.")


class TransactionGroups(BaseModel):
    transactions: List[TransactionGroup] = Field(default_factory=list)
# ==========================================
# 1. INITIALIZE OPENROUTER CLIENT
# ==========================================
# A single API key and billing profile covers both Anthropic and OpenAI models.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def make_schema_strict(schema: dict) -> dict:
    """Recursively force strict JSON-schema compatibility for provider validation.

    OpenAI-compatible strict schemas require every property to be listed in
    `required`, even when the field is nullable. We preserve nullability and
    make every object disallow extra keys.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                schema["required"] = list(properties.keys())
        for key, value in schema.items():
            if isinstance(value, (dict, list)):
                make_schema_strict(value)
    elif isinstance(schema, list):
        for item in schema:
            make_schema_strict(item)
    return schema

def process_receipt_pipeline(image_dir: str = IMAGE_DIR) -> List[TransactionGroup]:
    image_paths = list_image_files(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No receipt images found in {image_dir}")

    os.makedirs(OPTIMIZED_DIR, exist_ok=True)

    optimized_image_paths = []
    image_filenames = []
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        optimized_path = os.path.join(OPTIMIZED_DIR, f"{safe_file_name(base_name)}.jpg")
        if not os.path.exists(optimized_path):
            optimize_receipt_image(image_path, optimized_path)
        optimized_image_paths.append(optimized_path)
        image_filenames.append(os.path.basename(image_path))

    print(f"--- STEP 1: Grouping {len(optimized_image_paths)} receipt images via {stage1_client_model} ---")
    grouping_prompt = build_grouping_prompt(image_filenames)

    content = [{"type": "text", "text": grouping_prompt}]
    for optimized_image_path in optimized_image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(optimized_image_path)}"},
            }
        )

    response = client.chat.completions.create(
        model=stage1_client_model,
        messages=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TransactionGroupsSchema",
                "strict": True,
                "schema": make_schema_strict(TransactionGroups.model_json_schema()),
            },
        },
        temperature=0.1,
    )

    json_string = response.choices[0].message.content
    parsed_groups = TransactionGroups.model_validate_json(json_string)

    output_path = os.path.join(BASE_DIR, f"grouped_transactions_{MODEL_TAG}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([group.model_dump() for group in parsed_groups.transactions], f, indent=2)

    return parsed_groups.transactions


def print_grouping_summary(grouped_transactions: List[TransactionGroup], output_path: str) -> None:
    print()
    print("==========================================")
    print("DOCUMENT GROUPING COMPLETE")
    print("==========================================")
    print(f"Model: {stage1_client_model}")
    print(f"Transactions found: {len(grouped_transactions)}")
    print(f"Saved to: {output_path}")

    for group in grouped_transactions:
        print()
        print(f"Transaction {group.transaction_index}")
        print(f"Images: {', '.join(group.image_filenames)}")
        print(f"Reason: {group.reason}")
    


# ==========================================
# 4. EXECUTION DRIVER
# ==========================================
if __name__ == "__main__":
    try:
        grouped_transactions = process_receipt_pipeline(IMAGE_DIR)
        output_path = os.path.join(BASE_DIR, f"grouped_transactions_{MODEL_TAG}.json")
        print_grouping_summary(grouped_transactions, output_path)

    except Exception as error:
        print(f"\n❌ Pipeline failed: {error}")

##TODO : stage 2 is not getting the correct discount amount