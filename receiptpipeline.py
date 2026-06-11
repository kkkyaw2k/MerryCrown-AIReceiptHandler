from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import logging
from model import UploadRequest, DraftReceiptCreate

load_dotenv()

with open("stage_config.json", "r") as f:
    stage_configs = json.load(f)

def openaiclient():
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

def extract_json_from_text(text: str):
    """Utility to clean up markdown code blocks if the model returns them."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def group_images_stage1(client: OpenAI, img_urls: list[str]) -> list[list[str]]:
    # Skip Stage 1 completely if only 1 image
    if len(img_urls) == 1:
        return [img_urls]
    
    content_list = [{"type": "text", "text": f"Group these images into logical receipts. The valid URLs are: {json.dumps(img_urls)}"}]
    for url in img_urls:
        content_list.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": stage_configs["stage1"]["prompt"]},
        {"role": "user", "content": content_list}
    ]
    
    response = client.chat.completions.create(
        model=stage_configs["stage1"]["model"],
        messages=messages
    )
    
    try:
        groups = extract_json_from_text(response.choices[0].message.content)
        if isinstance(groups, list):
            return groups
    except Exception as e:
        logging.error(f"Stage 1 grouping failed: {e}. Defaulting to one large group.")
    
    return [img_urls]

def transcribe_stage2(client: OpenAI, group_urls: list[str]) -> str:
    content_list = []
    for url in group_urls:
        content_list.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": stage_configs["stage2"]["prompt"]},
        {"role": "user", "content": content_list}
    ]
    
    response = client.chat.completions.create(
        model=stage_configs["stage2"]["model"],
        messages=messages
    )
    return response.choices[0].message.content

def parse_stage3(client: OpenAI, markdown_text: str) -> dict:
    messages = [
        {"role": "system", "content": stage_configs["stage3"]["prompt"]},
        {"role": "user", "content": markdown_text}
    ]
    
    response = client.chat.completions.create(
        model=stage_configs["stage3"]["model"],
        messages=messages
    )
    
    try:
        return extract_json_from_text(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Stage 3 parsing failed: {e}")
        return {}

async def process_receipts_background(request: UploadRequest, supabase):
    client = openaiclient()
    
    print(f"\n--- Started processing batch: {request.batch_id} | Total Images: {len(request.img_urls)} ---")
    
    # Stage 1: Group
    print("Executing Stage 1: Grouping images...")
    groups = group_images_stage1(client, request.img_urls)
    print(f"Stage 1 Complete: Found {len(groups)} logical receipt(s).")
    
    for i, group in enumerate(groups, 1):
        try:
            print(f"\n>> Processing Receipt {i}/{len(groups)} (Images: {len(group)})")
            
            # Stage 2: Transcribe
            print(f"  [{i}/{len(groups)}] Executing Stage 2: Transcribing to Markdown...")
            raw_transcribe = transcribe_stage2(client, group)
            if "NOT_A_RECEIPT" in raw_transcribe.strip() or raw_transcribe.strip() == "NOT_A_RECEIPT":
                print(f"  [{i}/{len(groups)}] Image is not a receipt. Silently ignoring.")
                continue
            
            print(f"  [{i}/{len(groups)}] Stage 2 Complete.")
            
            # Stage 3: Parse
            print(f"  [{i}/{len(groups)}] Executing Stage 3: Parsing to JSON Schema...")
            raw_receipt = parse_stage3(client, raw_transcribe)
            print(f"  [{i}/{len(groups)}] Stage 3 Complete.")
            
            # Construct DraftReceiptCreate
            # We unpack raw_receipt but safely ensure required fields are there
            draft_receipt = DraftReceiptCreate(
                **raw_receipt,
                customer_id=request.customer_id,
                batch_id=request.batch_id,
                img_url=group,
                raw_transcribe=raw_transcribe,
                raw_receipt=raw_receipt,
                status="pending_review"
            )
            
            # Insert into Supabase
            print(f"  [{i}/{len(groups)}] Saving Draft Receipt to Supabase...")
            supabase.table("draft_receipt").insert(
                draft_receipt.model_dump(exclude={"batch_id"}, exclude_none=True)
            ).execute()
            print(f"  [{i}/{len(groups)}] Successfully saved to Supabase!")
            
        except Exception as e:
            print(f"  [{i}/{len(groups)}] ERROR: Failed to process group: {e}")
            logging.error(f"Failed to process group {group}: {e}")
            
    print(f"--- Finished processing batch: {request.batch_id} ---\n")