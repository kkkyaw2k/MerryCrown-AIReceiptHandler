import base64
import json
import os
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
import dotenv
dotenv.load_dotenv()

filename = r"Z:\Project REPO\MerryCrown-Frontend\WhatsApp Image 2026-04-25 at 4.09.40 PM.jpeg"
stage1_client_model = "google/gemini-3.1-flash-lite"  
stage2_client_model = "openai/gpt-5.4-nano"
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
# ==========================================
# 1. INITIALIZE OPENROUTER CLIENT
# ==========================================
# A single API key and billing profile covers both Anthropic and OpenAI models.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
)

# ==========================================
# 2. DEFINING THE PYDANTIC SCHEMAS
# ==========================================
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


class AutoReceipts(BaseModel):
    receipts: List[AutoReceipt] = Field(
        default_factory=list,
        description="All receipts extracted from the source text or image. Return one item per receipt.",
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

# ==========================================
# 3. PIPELINE EXECUTION FUNCTION
# ==========================================
def process_receipt_pipeline(image_path: str) -> AutoReceipts:
    transcript_path = os.path.join(BASE_DIR, f"transcribed_receipt_{safe_file_name(stage1_client_model)}.md")
    if os.path.exists(transcript_path):
        print("--- STEP 1: Reusing saved markdown transcription ---")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcribed_text = f.read()
        print("[Saved transcription loaded successfully]\n")
    else:
        optimized_image_path = os.path.join(
            BASE_DIR,
            f"optimized_{safe_file_name(os.path.splitext(os.path.basename(image_path))[0])}.jpg",
        )
        #optimize_receipt_image(image_path, optimized_image_path)
        base64_image = encode_image(image_path)

        # --------------------------------------------------
        # STEP 1: VISION PROCESSING (Claude 3.5 Sonnet / Sonnet 4.6)
        # --------------------------------------------------
        print(f"--- STEP 1: Running Image-to-Text Transcription via {stage1_client_model} ---")

        vision_prompt = """
        You are an expert financial document layout auditor. Analyze the attached receipt image and perform an incredibly precise transcription.

        CRITICAL RULES:
        1. **Column Alignment Security**: Receipts contain multi-line entries. Ensure you track the vertical columns safely. Do not let surrounding numeric item codes infect the 'Quantity' or 'Unit Price' values.
        2. **Staff/Identity Scraping**: Strip out human names that denote who bought the item or who got a staff perk (e.g. 'Khoo Wei Lian', 'Thin Zar Moe Htoo'). Remove them completely from the product name line.
        3. **GST/Tax Strategy Analysis**: Analyze how tax is distributed:
           - Check if prices are inclusive of GST (Post-Tax): Look for markers like 'GST Gross', 'Tax Inc', 'GST INCL', or tax codes next to items indicating tax was already built into the unit price.
           - Check if prices are exclusive of GST (Pre-Tax): Look for a net subtotal, with GST calculated as a separate block added underneath to form the grand total.
          4. **Item Breakdown**: For every item, explicitly capture the item name, quantity, unit price, and discount.
          5. **Discount Sub-Lines**: If the receipt has multiple discount rows under the same item, list each discount sub-line separately in the markdown transcription before you summarize them.
          6. **Discount Scope Check**: The `discount` field must store the per-unit discount, not the line-total discount. First combine any stacked discounts for the same line into one total discount, then divide by quantity using `per_unit_discount = total_discount / quantity`. If the receipt already shows a per-item discount, keep it as-is.

          Output a beautifully structured markdown transcription containing:
          - Store Name
          - Receipt Metadata
          - A clear markdown table of items with columns for item name, quantity, unit price, discount, and remarks
          - A separate markdown section that lists each discount sub-line exactly as it appears on the receipt
          - The final payment total
        """

        step1_response = client.chat.completions.create(
            model=stage1_client_model,  # This will automatically use the correct model based on the API key's billing profile
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.1
        )

        transcribed_text = step1_response.choices[0].message.content
        print("[Step 1 Transcription Finished Successfully]\n")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)

    # --------------------------------------------------
    # STEP 2: SCHEMA CONSOLIDATION (GPT-4o via Strict JSON Schema)
    # --------------------------------------------------
    print(f"--- STEP 2: Executing Strict Schema Mapping via {stage2_client_model} ---")
    
    schema_system_prompt = """
    You are a data serialization microservice. Your job is to convert raw receipt summaries into structured JSON matching a strict schema definition.
    
    Enforcement Rules:
    - If the input contains more than one receipt, extract each one separately and return them all in a top-level `receipts` array.
    - Default tax is 0.09 (9%) and default commission is 0.05 (5%) based on subtotal.
    - Set `amounts_are_post_tax` to True if item unit prices already included the tax amount. Set to False if the tax was strictly added to a net subtotal at the bottom.
    - Check the math: Sum (quantity * price) for items. Verify if it aligns with subtotal metrics and the overall total. Set `is_balanced` to true if the calculations balance out within a small 0.02 delta.
    - For discounts, the JSON field must contain the per-unit discount. If multiple discounts are listed for the same item, combine them into one total discount first, then divide by quantity: `discount = total_discount / quantity`. If the receipt already shows a per-unit discount, keep that value directly.
    - Output must be purely raw JSON matching the required schema structure.
    """

    # Generate schema dict and make it strictly compatible with OpenAI specifications
    pydantic_schema = AutoReceipts.model_json_schema()
    strict_pydantic_schema = make_schema_strict(pydantic_schema)

    openrouter_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "AutoReceiptsSchema",
            "strict": True,
            "schema": strict_pydantic_schema # Now includes required additionalProperties keys
        }
    }

    step2_response = client.chat.completions.create(
        model=stage2_client_model,
        messages=[
            {"role": "system", "content": schema_system_prompt},
            {"role": "user", "content": f"Structure this markdown text according to the target schema:\n\n{transcribed_text}"}
        ],
        response_format=openrouter_response_format,
        temperature=0.1
    )
    
    json_string = step2_response.choices[0].message.content
    print("[Step 2 Schema Mapping Complete]")
    # --------------------------------------------------
    # STEP 3: CONVERT TO PYDANTIC OBJECT
    # --------------------------------------------------
    parsed_receipt_object = AutoReceipts.model_validate_json(json_string)
    output_path = os.path.join(
        BASE_DIR,
        f"final_receipt_object_{safe_file_name(stage1_client_model)}_{safe_file_name(stage2_client_model)}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_receipt_object.model_dump(), f, indent=4)
    return parsed_receipt_object
    


# ==========================================
# 4. EXECUTION DRIVER
# ==========================================
if __name__ == "__main__":
    try:
        receipt_result = process_receipt_pipeline(filename)
        
        print("\n==========================================")
        print("🎉 OBJECT PARSED SUCCESSFULLY")
        print("==========================================")
        print(f"Receipts Extracted: {len(receipt_result.receipts)}")

        for receipt_index, receipt in enumerate(receipt_result.receipts, 1):
            print("\n------------------------------------------")
            print(f"Receipt {receipt_index}")
            print("------------------------------------------")
            print(f"Store:         {receipt.store_name}")
            print(f"Grand Total:   ${receipt.total:.2f}")
            print(f"Are Items Post-Tax?: {receipt.amounts_are_post_tax}")
            print(f"Math Balanced?:     {receipt.is_balanced}")
            print(f"Extracted Items Count: {len(receipt.items)}")

            for i, item in enumerate(receipt.items, 1):
                print(f"  └─ Item {i}: {item.item_name} (Qty: {item.quantity} x ${item.price}) -> Subtotal: ${item.subtotal:.2f}")

    except Exception as error:
        print(f"\n❌ Pipeline failed: {error}")

##TODO : stage 2 is not getting the correct discount amount