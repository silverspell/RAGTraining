from fastapi import FastAPI, HTTPException
from utils import ask_gemini
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

app = FastAPI()

@dataclass
class AskRequest:
    query: str
    source_id: str | None = None

@dataclass
class AskResponse:
    answer: str

@app.post("/ask")
async def ask_endpoint(request: AskRequest) -> AskResponse:
    answer = ask_gemini(request.query, request.source_id)
    return AskResponse(answer=answer)