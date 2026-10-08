"""POST /jev-model-picker-{low,medium,high}: pick a model for a prompt with typesafe/jev through the
OpenRouter Decisions API. One call asks two choice questions (task, model); the final model is the
highest-probability model among those in the tier that handle the chosen task."""
import json
import os
import time
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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

# Questions come from questions_jev_model_picker_<tier>.json (built by build_model_picker_questions.py);
# the model -> handled-tasks map comes from the same tier YAML.
questions: dict[str, dict] = {}
handles: dict[str, dict[str, set[str]]] = {}
for _tier in TIERS:
    questions[_tier] = json.loads(
        (BASE_DIR / f"questions_jev_model_picker_{_tier}.json").read_text(encoding="utf-8")
    )
    _data = yaml.safe_load((BASE_DIR / "model_tiers" / f"{_tier}.yaml").read_text(encoding="utf-8"))
    handles[_tier] = {name: set(m["handles"]) for name, m in _data["models"].items()}


class Candidate(BaseModel):
    model: str
    probability: float


class ModelPickResponse(BaseModel):
    tier: str
    model: str
    task: Decision
    candidates: list[Candidate]  # models in the tier that handle the task, best first
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
        try:
            async with httpx.AsyncClient(timeout=60) as http:
                r = await http.post(
                    OPENROUTER_DECISIONS_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={"model": OPENROUTER_MODEL, "state": state, "questions": questions[tier]},
                )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {e}")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OpenRouter {r.status_code}: {r.text}")

        answers = r.json().get("answers")
        if not isinstance(answers, dict):
            raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {r.text}")
        task = _decision("task", answers.get("task"))
        model_probs = _decision("model", answers.get("model")).probabilities

        candidates = sorted(
            (
                Candidate(model=m, probability=p)
                for m, p in model_probs.items()
                if task.choice in handles[tier].get(m, ())
            ),
            key=lambda c: c.probability,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(
                status_code=502, detail=f"No {tier} model handles task {task.choice!r}"
            )
        return ModelPickResponse(
            tier=tier,
            model=candidates[0].model,
            task=task,
            candidates=candidates,
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
