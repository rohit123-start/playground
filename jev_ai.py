"""POST /jev-open-router-tool: tool selection with typesafe/jev through the OpenRouter Decisions API.
Returns every tool with its probability of being needed (no threshold), highest first."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

import openrouter
from schemas import SelectedTool, SelectRequest, SelectResponse

load_dotenv()

router = APIRouter()

OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("JEV_OPEN_ROUTER_AI_MODEL")  # OpenRouter model id

# Native noul/choice questions for the OpenRouter Decisions API.
_QUESTIONS_FILE = Path(__file__).with_name("questions_jev_open_router_tool.json")
questions = json.loads(_QUESTIONS_FILE.read_text(encoding="utf-8")) if _QUESTIONS_FILE.exists() else {}


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


@router.post("/jev-open-router-tool", response_model=SelectResponse)
async def jev_open_router_tool(req: SelectRequest) -> SelectResponse:
    if not OPENROUTER_MODEL or not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set JEV_OPEN_ROUTER_AI_MODEL and OPEN_ROUTER_API_KEY environment variables",
        )
    body, _ = await openrouter.post_decision(
        OPENROUTER_DECISIONS_URL,
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        {"user_request": req.user_query},
        questions,
    )
    answers = body.get("answers")
    scores = {}
    for k in questions:
        p = _probability(answers.get(k)) if isinstance(answers, dict) else None
        if p is not None:
            scores[k] = p
    if not scores:
        raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {json.dumps(body)}")
    return SelectResponse(
        tools_selected=[
            SelectedTool(tool=k, score=round(v, 4))
            for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
    )
