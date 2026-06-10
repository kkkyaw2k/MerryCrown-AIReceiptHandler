from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(title="MerryCrown AI Receipt Handler")

# ==========================================
# SECURITY: API Key Setup
# ==========================================
API_KEY_NAME = "x-api-key"
# Render will use the environment variable if set, otherwise falls back to this string locally
EXPECTED_API_KEY = os.getenv("API_KEY", "my-super-secret-dev-key")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

# ==========================================
# CORS Setup
# ==========================================
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

# ==========================================
# Routes
# ==========================================

# This route is now SECURE. It requires the 'x-api-key' header.
@app.get("/", dependencies=[Depends(get_api_key)])
async def root():
    return {"message": "MerryCrown AI Receipt Handler API is running securely!"}

# Health checks are usually kept PUBLIC so Render can monitor if the server is alive
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=1000, reload=True)