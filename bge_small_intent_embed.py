"""POST /bge-small-intent-embed: intent detection by embedding similarity (BAAI/bge-small-en-v1.5).

The example prompts in prompt_intent_examples.json are embedded once at startup (eager
loading, see the lifespan in main.py). Each request embeds the user query and picks the
intent whose examples are closest to it.
"""
import json
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastembed import TextEmbedding
from pydantic import BaseModel

from schemas import SelectRequest

router = APIRouter()

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 3  # an intent's score is the mean of its K most similar examples

_model: TextEmbedding | None = None
_examples: list[tuple[str, str]] = []  # (intent, prompt)
_example_vectors: np.ndarray | None = None  # (n_examples, dim), unit length


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(_model.embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load() -> None:
    """Load the model and embed every example prompt. Called once at server startup."""
    global _model, _examples, _example_vectors
    with open(Path(__file__).with_name("prompt_intent_examples.json"), encoding="utf-8") as f:
        intents = json.load(f)["intents"]
    _examples = [(intent, p) for intent, prompts in intents.items() for p in prompts]
    _model = TextEmbedding(MODEL_NAME)
    _example_vectors = _embed([p for _, p in _examples])


class IntentResponse(BaseModel):
    intent: str
    score: float  # similarity of the winning intent
    scores: dict[str, float]  # similarity of every intent
    matched_example: str  # the closest example prompt
    latency_ms: float  # embedding + matching time


@router.post("/bge-small-intent-embed", response_model=IntentResponse)
async def bge_small_intent_embed(req: SelectRequest) -> IntentResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded yet")
    start = time.perf_counter()
    sims = _example_vectors @ _embed([req.user_query])[0]
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
