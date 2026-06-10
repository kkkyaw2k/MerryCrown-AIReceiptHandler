import json
import os
from typing import List

import dotenv
from openai import OpenAI

from grouping_helpers import (
    BASE_DIR,
    IMAGE_DIR,
    OPTIMIZED_DIR,
    TransactionGroup,
    TransactionGroups,
    build_grouping_prompt,
    encode_image,
    list_image_files,
    make_schema_strict,
    optimize_receipt_image,
    safe_file_name,
)

dotenv.load_dotenv()

stage1_client_model = "google/gemini-2.5-flash"
MODEL_TAG = safe_file_name(stage1_client_model)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
)

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