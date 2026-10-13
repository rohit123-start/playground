"""POST /jev-tool-model-routing: tool model routing with typesafe/jev through the OpenRouter
Decisions API. Routes requests to appropriate tools and models based on capability requirements.
One call determines which tool is needed and which model should execute it."""
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

# Questions for tool model routing, including capability detection and model selection.
# The file is re-read when it changes, so edits made in the playground apply without a restart.
question_store = QuestionStore(Path(__file__).parent / "questions_jev_tool_model_routing.json")


class ToolModelRoutingRequest(SelectRequest):
    metadata: dict | None = None  # optional extra info (attachments, history, ...) given to JEV


class Decision(BaseModel):
    choice: str | None  # model name or None for noul type questions
    confidence: float | None = None
    probabilities: dict[str, float]
    type: str  # "noul" or "choice"


class ToolModelRoutingResponse(BaseModel):
    decisions: dict[str, Decision]  # keyed by tool/capability name
    cached: bool  # served from the in-process cache, no OpenRouter call
    latency_ms: int


def _decision(name: str, answer, question_type: str) -> Decision:
    if not isinstance(answer, dict):
        raise HTTPException(status_code=502, detail=f"Missing/invalid answer for {name}: {answer}")

    if question_type == "noul":
        # A noul answer carries one float under the "noul" key: the probability the
        # question's "true" criterion holds, e.g. {"type": "noul", "noul": 0.93}.
        if not isinstance(answer.get("noul"), (int, float)):
            raise HTTPException(status_code=502, detail=f"Missing/invalid answer for {name}: {answer}")
        p_true = round(float(answer["noul"]), 4)
        probs = {"true": p_true, "false": round(1 - p_true, 4)}
        choice = "true" if p_true >= 0.5 else None
        conf = max(probs.values())
    else:
        # For choice type questions, expect probabilities field
        if not isinstance(answer.get("probabilities"), dict):
            raise HTTPException(status_code=502, detail=f"Missing/invalid answer for {name}: {answer}")
        probs = {str(k).lower(): round(float(v), 4) for k, v in answer["probabilities"].items()}
        choice = answer.get("choice")
        choice = str(choice).lower() if choice is not None else max(probs, key=probs.get)
        conf = answer.get("confidence")
        conf = round(float(conf), 4) if isinstance(conf, (int, float)) else None

    return Decision(
        choice=choice,
        confidence=conf,
        probabilities=probs,
        type=question_type,
    )


@router.post("/jev-tool-model-routing", response_model=ToolModelRoutingResponse)
async def jev_tool_model_routing(req: ToolModelRoutingRequest) -> ToolModelRoutingResponse:
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

    # Build decisions with type information from questions
    decisions = {}
    for name in questions:
        question_type = questions[name].get("type", "choice")
        decisions[name] = _decision(name, answers.get(name), question_type)

    return ToolModelRoutingResponse(
        decisions=decisions,
        cached=cached,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
