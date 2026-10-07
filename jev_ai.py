"""POST /jev-open-router-tool: tool selection with typesafe/jev through the OpenRouter Decisions API."""
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from schemas import SelectedTool, SelectRequest, SelectResponse

load_dotenv()

router = APIRouter()

OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("JEV_OPEN_ROUTER_AI_MODEL")  # OpenRouter model id
THRESHOLD = float(os.getenv("DECIDER_THRESHOLD", "0.65"))

# Native noul/choice questions for the OpenRouter Decisions API.
with open(Path(__file__).with_name("questions_jev_open_router_tool.json"), encoding="utf-8") as f:
    questions = json.load(f)


def _probability(answer) -> float | None:
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, dict):
        for key in ("probability", "noul", "score"):
            if isinstance(answer.get(key), (int, float)):
                return float(answer[key])
        probs = answer.get("probabilities")
        if isinstance(probs, dict):
            for key in ("true", "yes"):
                if key in probs:
                    return float(probs[key])
    return None


def pick_top(scores: dict[str, float]) -> list[tuple[str, float]]:
    """At most one tool: the highest score at or above THRESHOLD; "none" winning means no tool."""
    eligible = {k: v for k, v in scores.items() if v >= THRESHOLD}
    if not eligible:
        return []
    top = max(eligible, key=eligible.get)
    return [] if top == "none" else [(top, eligible[top])]


@router.post("/jev-open-router-tool", response_model=SelectResponse)
async def jev_open_router_tool(req: SelectRequest) -> SelectResponse:
    if not OPENROUTER_MODEL or not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set JEV_OPEN_ROUTER_AI_MODEL and OPEN_ROUTER_API_KEY environment variables",
        )
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.post(
                OPENROUTER_DECISIONS_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": OPENROUTER_MODEL,
                    "state": {"user_request": req.user_query},
                    "questions": questions,
                },
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenRouter {r.status_code}: {r.text}")

    answers = r.json().get("answers")
    scores = {}
    for k in questions:
        p = _probability(answers.get(k)) if isinstance(answers, dict) else None
        if p is not None:
            scores[k] = p
    if not scores:
        raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {r.text}")
    return SelectResponse(
        tools_selected=[SelectedTool(tool=k, score=round(v, 4)) for k, v in pick_top(scores)]
    )
