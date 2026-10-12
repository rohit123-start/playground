"""POST /jev-prompt-scan-open-router: model routing plus history/memory/file/media checks with typesafe/jev
through the OpenRouter Decisions API. Independent questions, one call."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import openrouter
from question_store import QuestionStore
from schemas import SelectRequest

load_dotenv()

router = APIRouter()

OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("JEV_OPEN_ROUTER_AI_MODEL")  # OpenRouter model id

# Option names are the returned values. The API has only choice/score/noul types, so the yes/no
# questions are choices with "true"/"false" options. Questions live in questions_jev_prompt_scan.json.
# The file is re-read when it changes, so edits made in the playground apply without a restart.
question_store = QuestionStore(Path(__file__).parent / "questions_jev_prompt_scan.json")


class PromptScanRequest(SelectRequest):
    metadata: dict | None = None  # optional extra info (attachments, history, ...) given to JEV


class Decision(BaseModel):
    choice: str  # booleans come back as "true"/"false"
    confidence: float | None = None
    probabilities: dict[str, float]


class PromptScanResponse(BaseModel):
    decisions: dict[str, Decision]  # keyed by question name
    cached: bool  # served from the in-process cache, no OpenRouter call
    latency_ms: int


def _decision(name: str, answer) -> Decision:
    if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
        raise HTTPException(status_code=502, detail=f"Missing/invalid answer for {name}: {answer}")
    probs = {str(k).lower(): round(float(v), 4) for k, v in answer["probabilities"].items()}
    choice = answer.get("choice")
    choice = str(choice).lower() if choice is not None else max(probs, key=probs.get)
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
    questions = question_store.get()
    start = time.perf_counter()
    body, cached = await openrouter.post_decision(
        OPENROUTER_DECISIONS_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, state, questions
    )
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {json.dumps(body)}")
    return PromptScanResponse(
        decisions={name: _decision(name, answers.get(name)) for name in questions},
        cached=cached,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
