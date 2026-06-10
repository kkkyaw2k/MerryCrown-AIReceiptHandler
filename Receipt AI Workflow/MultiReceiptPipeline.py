import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import dotenv
from markitdown import MarkItDown
from openai import OpenAI

from pipelinetest.common_helpers import (
    optimize_receipt_image,
    list_image_files,
    safe_file_name,
)

from SingleReceiptPipeline import AutoReceipt, STRICT_AUTO_RECEIPT_SCHEMA

BASE_DIR = Path(__file__).resolve().parent
OPTIMIZED_IMAGE_DIR = BASE_DIR / "optimized_images"

# Models by stage (adjust as needed)
STAGE1_GROUP_MODEL = "google/gemini-2.5-flash"
STAGE2_MODEL = "google/gemini-3.1-flash-lite"
STAGE3_MODEL = "openai/gpt-5.4-nano"  # used as a reasoning/schema-matching model


def prepare_and_optimize(image_paths: List[str]) -> List[Path]:
    OPTIMIZED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    optimized = []
    for p in image_paths:
        src = Path(p).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        out = OPTIMIZED_IMAGE_DIR / f"{safe_file_name(src.stem)}.jpg"
        if not out.exists():
            optimize_receipt_image(str(src), str(out))
        optimized.append(out)
    return optimized


def build_grouping_prompt(filenames: List[str]) -> str:
    return (
        "You are an assistant that groups uploaded receipt images into logical receipts.\n"
        "Given the list of filenames for images uploaded by a single user action, return a JSON array of groups.\n"
        "Each group is an array of filenames that belong to the same physical receipt (for example, multiple photos of a long receipt).\n"
        "Return only valid JSON, e.g. [[\"img1.jpg\", \"img2.jpg\"],[\"img3.jpg\"]].\n\n"
        f"Filenames:\n{json.dumps(filenames)}\n\n"  # provide filenames for reference
        "Rules:\n"
        "- Group by obvious continuity hints: sequential filenames, similar store names, or explicit duplicate content.\n"
        "- If unsure, put the file alone in its own group.\n"
        "- Output must be pure JSON array of arrays of filenames."
    )


def group_images_by_receipt(client: OpenAI, optimized_paths: List[Path]) -> List[List[Path]]:
    filenames = [p.name for p in optimized_paths]
    resp = client.chat.completions.create(
        model=STAGE1_GROUP_MODEL,
        messages=[{"role": "user", "content": build_grouping_prompt(filenames)}],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        groups = json.loads(text)
        mapped = []
        for group in groups:
            mapped.append([p for p in optimized_paths if p.name in group])
        # filter empty groups
        return [g for g in mapped if g]
    except Exception:
        # Fallback: treat each image as its own group
        return [[p] for p in optimized_paths]


def build_stage2_multi_prompt() -> str:
    return (
        "You are an expert OCR and transcription assistant. Given one or more optimized receipt images, "
        "produce a concise markdown transcription for each image. Output a JSON object mapping filename -> markdown text.\n"
        "Do not invent numbers; transcribe what's visible. Use clear item tables where possible.\n"
    )


def transcribe_images_batch(client: OpenAI, image_paths: List[Path]) -> dict:
    """Transcribe multiple images in a single model call when possible. Returns dict filename->markdown."""
    payload_parts = []
    for p in image_paths:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        payload_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "filename": p.name}})

    messages = [
        {"role": "user", "content": build_stage2_multi_prompt()},
        {"role": "user", "content": payload_parts},
    ]

    resp = client.chat.completions.create(model=STAGE2_MODEL, messages=messages, temperature=0.1)
    content = (resp.choices[0].message.content or "").strip()
    # Expecting JSON mapping filename->markdown. Try to parse; otherwise return raw text under a single key.
    try:
        return json.loads(content)
    except Exception:
        # Fallback: put entire response under a combined key
        return {"combined": content}

#def transcribe_image_single(client: OpenAI, image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    res = MarkItDown(
    enable_plugins=True,
    llm_client=client,
    llm_model="gemini-2.5-flash",
    )
    return res.convert(image_path).text_content

def transcribe_image_single(client: OpenAI, image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    resp = client.chat.completions.create(
        model=STAGE2_MODEL,
        messages=[
            {"role": "user", "content": build_stage2_multi_prompt()},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "filename": image_path.name}}]},
        ],
        temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()


def stage3_reason_and_match(client: OpenAI, markdowns: List[str]) -> List[dict]:
    """Take a list of markdown transcriptions (for one logical receipt) and convert to list of AutoReceipt-compatible JSON objects."""
    system = (
        "You are a schema-enforcement engine. Convert the provided markdown transcriptions into JSON objects matching the AutoReceipt schema. "
        "Return a JSON array of objects. Use explicit values only and set `is_balanced` according to reconciliation rules."
    )
    user_content = "\n\n----\n\n".join(markdowns)
    # Some providers reject complex JSON Schema with $ref. Request plain JSON and validate locally.
    resp = client.chat.completions.create(
        model=STAGE3_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        temperature=0.0,
    )

    text = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        # Try to extract JSON blob from text if the model added commentary
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            parsed = json.loads(text[start:end])
        except Exception as e:
            raise RuntimeError(f"Stage3 parsing failed, could not decode JSON: {e}\n{text}")

    if not isinstance(parsed, list):
        raise RuntimeError(f"Stage3 parsing failed, expected JSON array but got: {type(parsed)}\n{text}")

    def _first_present(*values):
        for value in values:
            if value is not None and value != "":
                return value
        return None

    def _coerce_int(value, default: int = 1) -> int:
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except Exception:
            return default

    def _coerce_float(value, default: float = 0.0) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _normalize_receipt(raw: dict) -> dict:
        # Map common vendor keys to our AutoReceipt schema
        out: dict = {}
        out["store_name"] = raw.get("store_name") or raw.get("merchant_name") or raw.get("merchant") or raw.get("file_name") or "Unknown"
        out["receipt_no"] = raw.get("receipt_no") or raw.get("receipt_number") or raw.get("ticket_number")
        out["receipt_date"] = raw.get("receipt_date") or raw.get("date") or None
        out["customer_id"] = _coerce_int(raw.get("customer_id"), default=0)

        items_out = []
        for it in raw.get("items", []):
            name = _first_present(it.get("item_name"), it.get("description"), it.get("product"), it.get("name"))
            qty = _coerce_int(_first_present(it.get("quantity"), it.get("qty"), it.get("count")), default=1)

            amount = _first_present(it.get("line_subtotal"), it.get("amount"), it.get("line_total"), it.get("subtotal"))
            raw_price = _first_present(it.get("price"), it.get("unit_price"), it.get("unit_price_amount"))

            if raw_price is None and amount is not None:
                price = _coerce_float(amount, default=0.0) / max(qty, 1)
            else:
                price = _coerce_float(raw_price, default=0.0)

            if amount is None:
                line_subtotal = price * qty
            else:
                line_subtotal = _coerce_float(amount, default=price * qty)

            discount = _coerce_float(_first_present(it.get("discount"), it.get("discount_amount"), it.get("discount_total")), default=0.0)
            items_out.append({
                "item_name": name,
                "item_code": it.get("item_code") or it.get("sku"),
                "quantity": qty,
                "price": price,
                "line_subtotal": line_subtotal,
                "discount": discount,
                "remarks": it.get("remarks") or it.get("note"),
            })

        out["items"] = items_out

        # totals
        totals = raw.get("totals") or raw.get("summary") or {}
        total_val = raw.get("total") or (totals.get("total") if isinstance(totals, dict) else None)
        if total_val is None:
            total_val = totals.get("total_including_tax") if isinstance(totals, dict) else None
        out["total"] = _coerce_float(total_val, default=0.0)

        # tax: prefer provided tax rate percent in items or totals
        tax_rate = raw.get("tax")
        if tax_rate is None:
            # look for tax_rate_percent in first item
            for it in raw.get("items", []):
                if it.get("tax_rate_percent") is not None:
                    try:
                        tax_rate = float(it.get("tax_rate_percent")) / 100.0
                    except Exception:
                        tax_rate = None
                    break
        out["tax"] = _coerce_float(tax_rate, default=0.09)

        out["remarks"] = raw.get("remarks") or raw.get("description")
        out["amounts_are_post_tax"] = bool(raw.get("amounts_are_post_tax", False))
        out["is_balanced"] = bool(raw.get("is_balanced", False))
        out["total_items"] = sum(_coerce_int(i.get("quantity"), default=0) for i in items_out)
        out["commission"] = _coerce_float(raw.get("commission"), default=0.0)

        return out

    validated = []
    for idx, item in enumerate(parsed):
        try:
            if isinstance(item, str):
                parsed_item = json.loads(item)
            else:
                parsed_item = item

            normalized = _normalize_receipt(parsed_item)
            # validate with AutoReceipt
            item_obj = AutoReceipt.model_validate(normalized)
            # recompute balance locally
            validation = {
                "expected_total": item_obj.expected_total if hasattr(item_obj, "expected_total") else None
            }
            validated.append(item_obj.model_dump())
        except Exception as e:
            # include item JSON for debugging
            raise RuntimeError(f"Validation failed for receipt index {idx}: {e}\nItem: {json.dumps(parsed_item)}")

    return validated


def process_multireceipt(images: List[str]) -> List[dict]:
    dotenv.load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_KEY is not set.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

    optimized = prepare_and_optimize(images)

    # If more than one optimized image, use grouping stage; otherwise single-image path
    groups = group_images_by_receipt(client, optimized) if len(optimized) > 1 else [[p] for p in optimized]

    all_receipts = []

    for group in groups:
        # For each group, transcribe images (batch if multiple)
        if len(group) == 1:
            md = transcribe_image_single(client, group[0])
            markdowns = [md]
        else:
            try:
                mapping = transcribe_images_batch(client, group)
                # mapping: filename->markdown
                markdowns = [mapping.get(p.name, "") for p in group]
            except Exception:
                # Fallback: transcribe one-by-one in parallel
                markdowns = []
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futures = {ex.submit(transcribe_image_single, client, p): p for p in group}
                    for fut in as_completed(futures):
                        try:
                            markdowns.append(fut.result())
                        except Exception as e:
                            markdowns.append(f"TRANSCRIPTION_FAILED: {e}")

        # Stage 3: reason and match into schema
        receipts = stage3_reason_and_match(client, markdowns)
        all_receipts.extend(receipts)

    # Persistence/upload is handled by the REST API layer (not in this AI pipeline module).
    # Save results locally
    out_path = BASE_DIR / "multi_receipt_results.json"
    out_path.write_text(json.dumps(all_receipts, indent=2), encoding="utf-8")
    print(f"Saved {len(all_receipts)} receipts to: {out_path}")
    return all_receipts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-receipt pipeline over a folder or list of images.")
    parser.add_argument("image_dir", nargs="?", help="Path to a folder of images or single image path.")
    args = parser.parse_args()

    if not args.image_dir:
        raise RuntimeError("Pass a folder path or an image path to process.")

    path = Path(args.image_dir)
    if path.is_dir():
        images = list_image_files(str(path))
    else:
        images = [str(path)]

    try:
        receipts = process_multireceipt(images)
        print(json.dumps(receipts, indent=2))
    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
