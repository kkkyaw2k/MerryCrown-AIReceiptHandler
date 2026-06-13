from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from dotenv import load_dotenv
import supabase
from fastapi import BackgroundTasks
from model import DraftReceiptCreate, UploadRequest
import logging

load_dotenv()

# Configure logging to output to stdout (visible in Render logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MerryCrown AI Receipt Handler")

supabase = supabase.create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Uses a header 'x-api-key' to authenticate requests
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

#================ CORS Setup =================

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://merry-crown-backend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#================ Routes ==================

# This route is SECURE. It requires the 'x-api-key' header.
@app.get("/", dependencies=[Depends(get_api_key)])
async def root():
    return {"message": "MerryCrown AI Receipt Handler API is running securely!"}

# Health checks are usually kept PUBLIC so cronjob can ping it to keep server active
@app.get("/health")
async def health():
    return {"status": "ok"}

#================ API Routes =================
@app.post("/receipt/upload", dependencies=[Depends(get_api_key)])
async def upload(request: UploadRequest, background_tasks: BackgroundTasks):
    from receiptpipeline import process_receipts_background
    background_tasks.add_task(process_receipts_background, request, supabase)
    return {
        "message": "Receipt processing started",
        "batch_id": request.batch_id,
        "status": "accepted"
    }





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=1000, reload=True)