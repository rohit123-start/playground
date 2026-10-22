"""POST /query-rewriter: a free-form chat-completion call to the query-rewriter's own model
(openai/gpt-6-luna, via OpenRouter), for trying out the query-rewriter system-prompt wording
without touching any real retrieval pipeline. Non-streaming, single call: a system prompt plus
one optional user-turn message (latest_message / recent_turns, however the tester wants to
format it) — same shape as jev_memory_llm.py / jev_summary_updater.py."""
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
QUERY_REWRITER_MODEL = os.getenv("QUERY_REWRITER_MODEL", "openai/gpt-6-luna")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Set OPEN_ROUTER_API_KEY environment variable")
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


class QueryRewriterRequest(BaseModel):
    system_prompt: str
    user_input: str = ""  # latest_message / recent_turns, as one blob


class QueryRewriterResponse(BaseModel):
    result: str
    model: str
    latency_ms: int


@router.post("/query-rewriter", response_model=QueryRewriterResponse)
async def query_rewriter(req: QueryRewriterRequest) -> QueryRewriterResponse:
    if not req.system_prompt.strip():
        raise HTTPException(status_code=422, detail="system_prompt must not be empty")
    client = _get_client()
    messages = [{"role": "system", "content": req.system_prompt}]
    if req.user_input.strip():
        messages.append({"role": "user", "content": req.user_input})
    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=QUERY_REWRITER_MODEL,
            messages=messages,
            stream=False,
        )
    except APIError as err:
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {err}")
    latency_ms = round((time.perf_counter() - start) * 1000)
    text = resp.choices[0].message.content or ""
    return QueryRewriterResponse(result=text, model=QUERY_REWRITER_MODEL, latency_ms=latency_ms)
