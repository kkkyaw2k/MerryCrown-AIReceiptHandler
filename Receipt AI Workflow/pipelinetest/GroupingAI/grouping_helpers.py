import os
import sys
from typing import List

from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from common_helpers import (  # noqa: E402
    BASE_DIR,
    IMAGE_DIR,
    OPTIMIZED_DIR,
    encode_image,
    list_image_files,
    make_schema_strict,
    optimize_receipt_image,
    safe_file_name,
)


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
