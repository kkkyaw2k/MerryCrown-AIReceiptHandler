from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="MerryCrown AI Receipt Handler")

# Define the domains that are allowed to make requests to this API
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://merry-crown-backend.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MerryCrown AI Receipt Handler API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=1000, reload=True)