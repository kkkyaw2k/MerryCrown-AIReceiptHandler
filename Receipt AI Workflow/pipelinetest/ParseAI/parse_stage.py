import asyncio
import os
from typing import List

import dotenv
from openai import AsyncOpenAI

from parse_helpers import (
	TRANSCRIPTION_OUTPUT_DIR,
	GROUPING_IMAGES_DIR,
	TransactionGroup,
	build_transcription_prompt,
	encode_image,
	load_grouped_transactions,
	prepare_image_for_transcription,
	resolve_image_path,
	safe_file_name,
)

dotenv.load_dotenv()

TRANSCRIPTION_MODEL = "google/gemini-2.5-flash"
MODEL_TAG = safe_file_name(TRANSCRIPTION_MODEL)

# Paste a single grouped transaction JSON array here to run a one-transaction test.
# Leave USE_MANUAL_TRANSCRIPTION_GROUPS = False to load the latest grouped file instead.
MANUAL_TRANSCRIPTION_GROUPS_JSON = r'''
[
	{
		"transaction_index": 1,
		"image_filenames": ["photo1-1.jpg", "photo1-2.jpg"],
		"reason": "photo1-2 is a direct continuation of the item list and totals from photo1-1 for The Shilla Duty Free."
	}
]
'''
USE_MANUAL_TRANSCRIPTION_GROUPS = False

client = AsyncOpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key=os.getenv("OPENROUTER_KEY"),
)


async def transcribe_group_async(group: TransactionGroup, semaphore: asyncio.Semaphore) -> dict:
	async with semaphore:
		prompt = build_transcription_prompt(group)
		content = [{"type": "text", "text": prompt}]

		for filename in group.image_filenames:
			source_path = resolve_image_path(filename)
			optimized_path = prepare_image_for_transcription(source_path)
			content.append(
				{
					"type": "image_url",
					"image_url": {"url": f"data:image/jpeg;base64,{encode_image(optimized_path)}"},
				}
			)

		response = await client.chat.completions.create(
			model=TRANSCRIPTION_MODEL,
			messages=[{"role": "user", "content": content}],
			temperature=0.1,
		)

		transcript_text = response.choices[0].message.content or ""
		return {
			"transaction_index": group.transaction_index,
			"image_filenames": group.image_filenames,
			"reason": group.reason,
			"transcript_text": transcript_text,
		}


async def transcribe_transactions_concurrently(groups: List[TransactionGroup]) -> List[dict]:
	max_concurrency = int(os.getenv("TRANSCRIPTION_STAGE_MAX_CONCURRENCY", "4"))
	semaphore = asyncio.Semaphore(max_concurrency)

	tasks = [transcribe_group_async(group, semaphore) for group in groups]
	transcripts = await asyncio.gather(*tasks)
	return list(transcripts)


def save_transcripts(transcripts: List[dict]) -> List[str]:
	os.makedirs(TRANSCRIPTION_OUTPUT_DIR, exist_ok=True)
	output_paths: List[str] = []
	for transcript in transcripts:
		transaction_index = transcript["transaction_index"]
		output_path = os.path.join(
			TRANSCRIPTION_OUTPUT_DIR,
			f"transcribed_receipt_{MODEL_TAG}_transaction_{transaction_index}.md",
		)
		with open(output_path, "w", encoding="utf-8") as handle:
			handle.write(transcript["transcript_text"])
		output_paths.append(output_path)
	return output_paths


def print_transcription_summary(transcripts: List[dict], output_paths: List[str]) -> None:
	print()
	print("==========================================")
	print("TRANSCRIPTION STAGE COMPLETE")
	print("==========================================")
	print(f"Model: {TRANSCRIPTION_MODEL}")
	print(f"Transactions transcribed: {len(transcripts)}")
	for path in output_paths:
		print(f"Saved to: {path}")

	for index, transcript in enumerate(transcripts, 1):
		print()
		print(f"Transaction {index}")
		print(f"Images: {', '.join(transcript['image_filenames'])}")
		print(f"Reason: {transcript['reason']}")
		print(f"Transcript characters: {len(transcript['transcript_text'])}")


async def main() -> None:
	manual_json = MANUAL_TRANSCRIPTION_GROUPS_JSON if USE_MANUAL_TRANSCRIPTION_GROUPS else None
	groups = load_grouped_transactions(manual_json)
	transcripts = await transcribe_transactions_concurrently(groups)
	output_paths = save_transcripts(transcripts)
	print_transcription_summary(transcripts, output_paths)


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except Exception as error:
		print(f"\n❌ Transcription stage failed: {error}")
