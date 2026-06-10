from fastapi import FastAPI
import uvicorn

app = FastAPI(title="MerryCrown AI Receipt Handler")

@app.get("/")
async def root():
    return {"message": "MerryCrown AI Receipt Handler API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)