"""POST /summary-updater: a free-form chat-completion call to the conversation-summarizer's own
model (openai/gpt-6-luna, via OpenRouter), for trying out the summary-updater system-prompt
wording without touching any real chat or its DB writes. Non-streaming, single call: a system
prompt plus one optional user-turn message (<current_summary>/<decision_log>/<messages_to_fold>,
however the tester wants to format it) — same shape as jev_memory_llm.py, just with a second,
separate input field for the data the prompt itself says it "receives"."""
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
SUMMARY_UPDATER_MODEL = os.getenv("SUMMARY_UPDATER_MODEL", "openai/gpt-6-luna")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Set OPEN_ROUTER_API_KEY environment variable")
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


class SummaryUpdaterRequest(BaseModel):
    system_prompt: str
    user_input: str = ""  # <current_summary> / <decision_log> / <messages_to_fold>, as one blob


class SummaryUpdaterResponse(BaseModel):
    result: str
    model: str
    latency_ms: int


@router.post("/summary-updater", response_model=SummaryUpdaterResponse)
async def summary_updater(req: SummaryUpdaterRequest) -> SummaryUpdaterResponse:
    if not req.system_prompt.strip():
        raise HTTPException(status_code=422, detail="system_prompt must not be empty")
    client = _get_client()
    messages = [{"role": "system", "content": req.system_prompt}]
    if req.user_input.strip():
        messages.append({"role": "user", "content": req.user_input})
    start = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=SUMMARY_UPDATER_MODEL,
            messages=messages,
            stream=False,
        )
    except APIError as err:
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {err}")
    latency_ms = round((time.perf_counter() - start) * 1000)
    text = resp.choices[0].message.content or ""
    return SummaryUpdaterResponse(result=text, model=SUMMARY_UPDATER_MODEL, latency_ms=latency_ms)
