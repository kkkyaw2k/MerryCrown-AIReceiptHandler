import base64
import os
import re
from typing import List

import PIL.Image
import PIL.ImageEnhance


def safe_file_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def process_images(image_paths):
    processed_images = []
    for path in image_paths:
        # Placeholder for image processing logic
        processed_image = f"Processed {path}"
        processed_images.append(processed_image)
    return processed_images


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def optimize_receipt_image(image_path: str, target_path: str) -> None:
    img = PIL.Image.open(image_path)
    img.thumbnail((1200, 1200), PIL.Image.Resampling.LANCZOS)
    img = img.convert("L")
    enhancer = PIL.ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img.save(target_path, "JPEG", quality=80)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def make_schema_strict(schema: dict) -> dict:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                schema["required"] = list(properties.keys())
        for value in schema.values():
            if isinstance(value, (dict, list)):
                make_schema_strict(value)
    elif isinstance(schema, list):
        for item in schema:
            make_schema_strict(item)
    return schema


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
