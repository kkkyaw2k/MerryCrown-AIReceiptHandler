## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## MerryCrown system overview (hand-written)

Part of three repos in `Z:\Project REPO`: `MerryCrown-Backend` (FastAPI + Supabase, calls this service), `MerryCrown-AIReceiptHandler` (this repo), `MerryCrown-Frontend` (React PWA in `new version/app/`). Cross-repo graph: `Z:\Project REPO\graphify-filtered\graphify-out\graph.html` (no cross-repo edges; links below are from reading the code).

### What this service does
- `POST /receipt/upload` (header `x-api-key`) receives `{customer_id, batch_id, img_urls}` from the Backend's `/receipts/ai-process`, returns "accepted" immediately, and processes in a FastAPI BackgroundTask (`receiptpipeline.process_receipts_background`).
- Pipeline (models/prompts live in `stage_config.json`, calls go through OpenRouter via `OPENROUTER_API_KEY`):
  1. Stage 1 group images that belong to the same receipt and drop non-receipt images (skipped if only 1 image).
  2. Stage 2 transcribe each group to markdown.
  3. Stage 3 parse the markdown into the receipt JSON schema.
  Results are inserted into Supabase as draft/pending receipts, which the frontend then reviews and approves through the Backend.
- Public `GET /health` (for cron keep-alive); `GET /` needs the API key.
- `langgraph-plan.md` describes a planned migration of the pipeline to LangGraph (not implemented).

### Run / env
- `start_local.bat` -> `uvicorn main:app --reload --port 8001`.
- Env: `SUPABASE_URL`, `SUPABASE_KEY`, `API_KEY` (must equal the Backend's `AI_HANDLER_API_KEY`), `OPENROUTER_API_KEY`. Never commit real values.
- CORS allows localhost:3000/5173 and the Vercel backend origin. Logs go to stdout (appears to be hosted on Render).

### Deployment (verified live 2026-11-10)
- Hosted on Render: `https://merrycrown-aireceipthandler.onrender.com` (uvicorn behind Cloudflare). `/health` returns 200 publicly; `/` returns 401 without `x-api-key`, as in the code. Only the unauthenticated surface was checked; the OCR pipeline itself was not exercised.
- Backend (caller): `https://merry-crown-backend.vercel.app`.

### Operational notes (from the owner)
- The OCR pipeline works in production.
- The Render instance is kept awake by a 10-minute cron job pinging `/health`, so cold starts should not be expected.
