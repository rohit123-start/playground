"""POST /voyage-intent-embed: intent detection by embedding similarity (voyageai/voyage-4-lite via OpenRouter).

The example prompts in prompt_intent_examples.json are embedded once at startup (see the
lifespan in main.py). Each request embeds the user query and picks the intent whose
examples are closest to it.
"""
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas import SelectRequest

load_dotenv()

router = APIRouter()

OPENROUTER_EMBEDDINGS_URL = os.getenv("OPENROUTER_EMBEDDINGS_URL", "https://openrouter.ai/api/v1/embeddings")
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
MODEL_NAME = "voyageai/voyage-4-lite"
TOP_K = 3  # an intent's score is the mean of its K most similar examples
BATCH_SIZE = 64

_examples: list[tuple[str, str]] = []  # (intent, prompt)
_example_vectors: np.ndarray | None = None  # (n_examples, dim), unit length


async def _embed(texts: list[str]) -> np.ndarray:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Set OPEN_ROUTER_API_KEY environment variable")
    vectors = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            resp = await client.post(
                OPENROUTER_EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": MODEL_NAME, "input": texts[i : i + BATCH_SIZE]},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"OpenRouter error {resp.status_code}: {resp.text[:500]}")
            data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            vectors.extend(d["embedding"] for d in data)
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


async def load() -> None:
    """Embed every example prompt. Called once at server startup."""
    global _examples, _example_vectors
    with open(Path(__file__).with_name("prompt_intent_examples.json"), encoding="utf-8") as f:
        intents = json.load(f)["intents"]
    _examples = [(intent, p) for intent, prompts in intents.items() for p in prompts]
    _example_vectors = await _embed([p for _, p in _examples])


class IntentResponse(BaseModel):
    intent: str
    score: float  # similarity of the winning intent
    scores: dict[str, float]  # similarity of every intent
    matched_example: str  # the closest example prompt
    latency_ms: float  # embedding + matching time


@router.post("/voyage-intent-embed", response_model=IntentResponse)
async def voyage_intent_embed(req: SelectRequest) -> IntentResponse:
    if _example_vectors is None:
        raise HTTPException(status_code=503, detail="Example embeddings are not loaded yet")
    start = time.perf_counter()
    sims = _example_vectors @ (await _embed([req.user_query]))[0]
    scores = {}
    for intent in {i for i, _ in _examples}:
        idx = [n for n, (i, _) in enumerate(_examples) if i == intent]
        scores[intent] = float(np.sort(sims[idx])[-TOP_K:].mean())
    best = max(scores, key=scores.get)
    closest = int(np.argmax(sims))
    latency_ms = (time.perf_counter() - start) * 1000
    return IntentResponse(
        intent=best,
        score=round(scores[best], 4),
        scores={k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
        matched_example=_examples[closest][1],
        latency_ms=round(latency_ms, 1),
    )
