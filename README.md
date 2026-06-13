# MerryCrown AI Receipt Handler

![Receipt AI Pipeline](https://github.com/user-attachments/assets/ec13bafb-eb70-4093-80e4-31ca6ee56821)

**MerryCrown AI Receipt Handler** is a robust, asynchronous API service designed to automatically process, group, transcribe, and parse receipt images into structured JSON data. It leverages advanced Large Language Models via OpenRouter to meticulously analyze receipt documents and persist them as drafts in Supabase.

---

## 🚀 Features

- **Asynchronous Processing:** Long-running AI pipelines run in the background via FastAPI's `BackgroundTasks`, enabling fast API response times.
- **Smart Image Grouping:** Intelligently groups multiple uploaded photos (like front/back or split receipts) into logical singular receipt entities.
- **Precision Transcription:** Extracts exact column-aligned markdown from financial documents, handling tricky layouts, discounts, and tax figures.
- **Strict JSON Parsing:** Enforces a rigid JSON schema required by downstream systems. Includes specific edge-case logic (e.g., standardizing tax-inclusive pricing for Shilla Duty Free receipts).
- **Secure Integration:** Authenticated endpoints utilizing `x-api-key` headers.
- **Database Ready:** Seamlessly inserts extracted data directly into your Supabase `draft_receipt` table.

---

## 🧠 The 3-Stage AI Pipeline

This service uses a highly specialized 3-stage approach to ensure accuracy:

### Stage 1: Grouping (`google/gemini-3.1-flash-lite`)
Analyzes a batch of uploaded images and determines which images belong together to form a single physical receipt. Completely ignores irrelevant images (like selfies or landscapes).

### Stage 2: Transcription (`google/gemini-3.1-flash-lite`)
Acts as an expert financial document layout auditor. It takes the grouped images and produces a highly accurate Markdown transcription. It preserves vertical column alignment, explicit discount lines, and provides a GST/Tax strategy analysis.

### Stage 3: Schema Enforcement (`openai/gpt-4o-mini`)
Converts the transcribed Markdown into strict, structured JSON. It maps fields to specific keys, normalizes dates, handles per-unit discount math, and applies complex tax logic depending on the store type.

---

## 🛠 Tech Stack

- **Framework:** FastAPI (Python)
- **AI Integration:** OpenRouter API (Gemini 3.1 & GPT-4o-mini)
- **Database:** Supabase (PostgreSQL)
- **Environment:** Uvicorn, Python-dotenv

---

## 📡 API Endpoints

### `POST /receipt/upload`
Accepts a batch of receipt image URLs to be processed. Requires the `x-api-key` header.
**Payload:**
```json
{
  "batch_id": "string",
  "customer_id": "string",
  "img_urls": ["url1", "url2", "url3"]
}
```
**Response:**
```json
{
  "message": "Receipt processing started",
  "batch_id": "string",
  "status": "accepted"
}
```

### `GET /`
Secure root endpoint. Requires `x-api-key`.

### `GET /health`
Public health check endpoint for monitoring uptime.

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd MerryCrown-AIReceiptHandler
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```ini
   OPENROUTER_API_KEY=your_openrouter_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_anon_or_service_key
   API_KEY=your_secure_api_key
   ```

4. **Run the Application Locally:**
   You can run the provided batch script:
   ```bash
   start_local.bat
   ```
   Or run it manually via Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 1000 --reload
   ```

---

*Built for the MerryCrown ecosystem to automate financial data entry.*
