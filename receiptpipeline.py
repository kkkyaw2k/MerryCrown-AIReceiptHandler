from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json
import logging
import asyncio
from model import UploadRequest, DraftReceiptCreate

load_dotenv()

with open("stage_config.json", "r") as f:
    stage_configs = json.load(f)

def openaiclient():
    return AsyncOpenAI(
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

async def group_images_stage1(client: AsyncOpenAI, img_urls: list[str]) -> list[list[str]]:
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
    
    response = await client.chat.completions.create(
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

async def transcribe_stage2(client: AsyncOpenAI, group_urls: list[str]) -> str:
    content_list = []
    for url in group_urls:
        content_list.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": stage_configs["stage2"]["prompt"]},
        {"role": "user", "content": content_list}
    ]
    
    response = await client.chat.completions.create(
        model=stage_configs["stage2"]["model"],
        messages=messages
    )
    return response.choices[0].message.content

async def parse_stage3(client: AsyncOpenAI, markdown_text: str) -> dict:
    messages = [
        {"role": "system", "content": stage_configs["stage3"]["prompt"]},
        {"role": "user", "content": markdown_text}
    ]
    
    response = await client.chat.completions.create(
        model=stage_configs["stage3"]["model"],
        messages=messages
    )
    
    try:
        return extract_json_from_text(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Stage 3 parsing failed: {e}")
        return {}

async def process_single_group(i: int, total_groups: int, group: list[str], client: AsyncOpenAI, request: UploadRequest, supabase):
    try:
        logging.info(f">> Processing Receipt {i}/{total_groups} (Images: {len(group)})")
        
        # Stage 2: Transcribe
        logging.info(f"  [{i}/{total_groups}] Executing Stage 2: Transcribing to Markdown...")
        raw_transcribe = await transcribe_stage2(client, group)
        if "NOT_A_RECEIPT" in raw_transcribe.strip() or raw_transcribe.strip() == "NOT_A_RECEIPT":
            logging.info(f"  [{i}/{total_groups}] Image is not a receipt. Silently ignoring.")
            return
        
        logging.info(f"  [{i}/{total_groups}] Stage 2 Complete.")
        
        # Stage 3: Parse
        logging.info(f"  [{i}/{total_groups}] Executing Stage 3: Parsing to JSON Schema...")
        raw_receipt = await parse_stage3(client, raw_transcribe)
        logging.info(f"  [{i}/{total_groups}] Stage 3 Complete.")
        
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
        logging.info(f"  [{i}/{total_groups}] Saving Draft Receipt to Supabase...")
        
        def insert_to_supabase():
            return supabase.table("draft_receipt").insert(
                draft_receipt.model_dump(exclude={"batch_id"}, exclude_none=True)
            ).execute()
            
        await asyncio.to_thread(insert_to_supabase)
        logging.info(f"  [{i}/{total_groups}] Successfully saved to Supabase!")
        
    except Exception as e:
        logging.error(f"  [{i}/{total_groups}] Failed to process group {group}: {e}")

async def process_receipts_background(request: UploadRequest, supabase):
    client = openaiclient()
    
    logging.info(f"--- Started processing batch: {request.batch_id} | Total Images: {len(request.img_urls)} ---")
    
    # Stage 1: Group
    logging.info("Executing Stage 1: Grouping images...")
    groups = await group_images_stage1(client, request.img_urls)
    logging.info(f"Stage 1 Complete: Found {len(groups)} logical receipt(s).")
    
    # Process all groups concurrently
    tasks = [
        process_single_group(i, len(groups), group, client, request, supabase)
        for i, group in enumerate(groups, 1)
    ]
    
    if tasks:
        await asyncio.gather(*tasks)
            
    logging.info(f"--- Finished processing batch: {request.batch_id} ---")