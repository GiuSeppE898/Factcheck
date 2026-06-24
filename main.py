from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.insert(0, "src")
from pipeline import fact_check

app = FastAPI(title="Fact-Checking API", version="1.0")

class ArticoloRequest(BaseModel):
    testo: str
    k: int = 3

@app.post("/fact-check")
def fact_check_endpoint(req: ArticoloRequest):
    return fact_check(req.testo, k=req.k)

@app.get("/health")
def health():
    return {"status": "ok"}

# AVVIO CON uvicorn main:app --reload