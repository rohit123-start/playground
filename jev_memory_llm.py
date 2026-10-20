"""POST /memory-llm: a free-form chat-completion call to the memory pipeline's own model
(openai/gpt-5.6-luna, via OpenRouter — same model ai-platform's app/rag/extraction.py hardcodes
as MEMORY_EXTRACTOR_MODEL), for trying out system-prompt wording without touching the real
extractor or its DB writes. Non-streaming, single call, no history, no separate user message:
the system prompt alone is the whole request."""
import os
import time

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
MEMORY_LLM_MODEL = os.getenv("MEMORY_LLM_MODEL", "openai/gpt-5.6-luna")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Set OPEN_ROUTER_API_KEY environment variable")
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


class MemoryLLMRequest(BaseModel):
    system_prompt: str


class MemoryLLMResponse(BaseModel):
    result: str
    model: str
    latency_ms: int


@router.post("/memory-llm", response_model=MemoryLLMResponse)
async def memory_llm(req: MemoryLLMRequest) -> MemoryLLMResponse:
    if not req.system_prompt.strip():
        raise HTTPException(status_code=422, detail="system_prompt must not be empty")
    client = _get_client()
    messages = [{"role": "system", "content": req.system_prompt}]
    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=MEMORY_LLM_MODEL,
            messages=messages,
            stream=False,
        )
    except APIError as err:
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {err}")
    latency_ms = round((time.perf_counter() - start) * 1000)
    text = resp.choices[0].message.content or ""
    return MemoryLLMResponse(result=text, model=MEMORY_LLM_MODEL, latency_ms=latency_ms)
