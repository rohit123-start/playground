"""POST /jev-prompt-scan-open-router: model/tool routing + context routing with typesafe/jev
through the OpenRouter Decisions API. Two independent choice questions, one call."""
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas import SelectRequest

load_dotenv()

router = APIRouter()

OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("JEV_OPEN_ROUTER_AI_MODEL")  # OpenRouter model id

# Option names are the returned values.
questions = {
    "model_routing": {
        "type": "choice",
        "instructions": (
            "Does this request require a specific model or tool, or can the default model handle it?"
        ),
        "criteria": {
            "default_model": (
                "The default LLM can fulfill the request without a specialized model or tool."
            ),
            "specific_model_or_tool": (
                "The request requires a specialized model or tool, such as web search, image "
                "generation, code execution, audio processing, video processing, etc."
            ),
        },
    },
    "context_routing": {
        "type": "choice",
        "instructions": "Does this request require additional context?",
        "criteria": {
            "context_yes": (
                "The answer requires information provided before the current request, such as "
                "previous conversation, previous tool results, previously processed files, "
                "memory, or earlier outputs."
            ),
            "context_no": (
                "The request can be fulfilled using only the current request and its direct "
                "inputs. This includes files, attachments, and their metadata provided with the "
                "current request. Current-request inputs are not context. For unclear, "
                "meaningless, random, or self-contained input, choose context_no."
            ),
        },
    },
}


class PromptScanRequest(SelectRequest):
    metadata: dict | None = None  # optional extra info (attachments, history, ...) given to JEV


class Decision(BaseModel):
    choice: str
    confidence: float | None = None
    probabilities: dict[str, float]


class PromptScanResponse(BaseModel):
    model_routing: Decision
    context_routing: Decision
    latency_ms: int


def _decision(name: str, answer) -> Decision:
    if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
        raise HTTPException(status_code=502, detail=f"Missing/invalid answer for {name}: {answer}")
    probs = {k: round(float(v), 4) for k, v in answer["probabilities"].items()}
    choice = answer.get("choice") or max(probs, key=probs.get)
    conf = answer.get("confidence")
    return Decision(
        choice=choice,
        confidence=round(float(conf), 4) if isinstance(conf, (int, float)) else None,
        probabilities=probs,
    )


@router.post("/jev-prompt-scan-open-router", response_model=PromptScanResponse)
async def jev_prompt_scan_open_router(req: PromptScanRequest) -> PromptScanResponse:
    if not OPENROUTER_MODEL or not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set JEV_OPEN_ROUTER_AI_MODEL and OPEN_ROUTER_API_KEY environment variables",
        )
    state = {"user_request": req.user_query}
    if req.metadata:
        state["metadata"] = req.metadata
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.post(
                OPENROUTER_DECISIONS_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": OPENROUTER_MODEL, "state": state, "questions": questions},
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenRouter {r.status_code}: {r.text}")

    answers = r.json().get("answers")
    if not isinstance(answers, dict):
        raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {r.text}")
    return PromptScanResponse(
        model_routing=_decision("model_routing", answers.get("model_routing")),
        context_routing=_decision("context_routing", answers.get("context_routing")),
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
