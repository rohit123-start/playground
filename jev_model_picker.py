"""POST /jev-model-picker-{low,medium,high}: pick models for a prompt with typesafe/jev through the
OpenRouter Decisions API. One call asks one choice question per capability (answer, search, image,
video generation/editing, audio generation/editing); each answer is a model of the tier or "none"."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import openrouter
from jev_prompt_scan import Decision, PromptScanRequest, _decision

load_dotenv()

router = APIRouter()

OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("JEV_OPEN_ROUTER_AI_MODEL")  # OpenRouter model id

BASE_DIR = Path(__file__).parent
TIERS = ("low", "medium", "high")

# Built from model_tiers/<tier>.yaml by build_model_picker_questions.py.
questions: dict[str, dict] = {
    tier: json.loads(
        (BASE_DIR / f"questions_jev_model_picker_{tier}.json").read_text(encoding="utf-8")
    )
    for tier in TIERS
}


class ModelPickResponse(BaseModel):
    tier: str
    selected: dict[str, str | None]  # capability -> chosen model, or None when "none" won
    decisions: dict[str, Decision]  # capability -> full decision with probabilities
    cached: bool  # served from the in-process cache, no OpenRouter call
    latency_ms: int


def _make_handler(tier: str):
    async def pick(req: PromptScanRequest) -> ModelPickResponse:
        if not OPENROUTER_MODEL or not OPENROUTER_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Set JEV_OPEN_ROUTER_AI_MODEL and OPEN_ROUTER_API_KEY environment variables",
            )
        state = {"user_request": req.user_query}
        if req.metadata:
            state["metadata"] = req.metadata
        start = time.perf_counter()
        body, cached = await openrouter.post_decision(
            OPENROUTER_DECISIONS_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, state, questions[tier]
        )
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {json.dumps(body)}")
        decisions = {name: _decision(name, answers.get(name)) for name in questions[tier]}
        return ModelPickResponse(
            tier=tier,
            selected={n: (None if d.choice == "none" else d.choice) for n, d in decisions.items()},
            decisions=decisions,
            cached=cached,
            latency_ms=round((time.perf_counter() - start) * 1000),
        )

    return pick


for _tier in TIERS:
    router.add_api_route(
        f"/jev-model-picker-{_tier}",
        _make_handler(_tier),
        methods=["POST"],
        response_model=ModelPickResponse,
        name=f"jev_model_picker_{_tier}",
    )
