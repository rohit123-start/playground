"""POST /reranker-compare: cross-encoder reranking of a semantic-search result set, scored for
accuracy (precision@K / perfect separation, same ground-truth shape as jev_embeddings.py's
/semantic-search) and timed — the stage that sits ON TOP of embedding-based retrieval, not a
replacement for it.

- local: Xenova/ms-marco-MiniLM-L-6-v2 via fastembed's TextCrossEncoder, in-process, no network
  call. Same checkpoint ai-platform's app/rerank.py already uses for L2 retrieval.
- openrouter: qwen/qwen3-reranker-8b via OpenRouter's dedicated /api/v1/rerank endpoint (NOT
  /chat/completions — rerank models 400 there with a pointer to this endpoint). Confirmed live:
  OpenRouter serves it through Fireworks as "accounts/fireworks/models/qwen3-reranker-8b".
"""
import os
import time

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

LOCAL_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
OPENROUTER_RERANK_MODEL = os.getenv("OPENROUTER_RERANK_MODEL", "qwen/qwen3-reranker-8b")

_local_model = None  # fastembed TextCrossEncoder, loaded lazily (first call pays the load cost)


def _get_local_model():
    global _local_model
    if _local_model is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _local_model = TextCrossEncoder(model_name=LOCAL_RERANK_MODEL)
    return _local_model


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    relevant: list[str] = []  # ground-truth subset of `documents` considered relevant to `query`


class RerankResult(BaseModel):
    document: str
    score: float
    relevant: bool


class RerankSideResult(BaseModel):
    model: str
    latency_ms: int
    results: list[RerankResult] = []
    precision_at_k: "float | None" = None
    perfect_separation: "bool | None" = None
    error: "str | None" = None


class RerankResponse(BaseModel):
    local: RerankSideResult
    openrouter: RerankSideResult


def _accuracy(results: list[RerankResult], relevant: set[str]) -> tuple["float | None", "bool | None"]:
    k = len(relevant)
    if k == 0:
        return None, None
    top_k = results[:k]
    precision_at_k = sum(1 for r in top_k if r.relevant) / k
    perfect_separation = all(r.relevant for r in top_k)
    return round(precision_at_k, 4), perfect_separation


def _run_local(query: str, documents: list[str], relevant: set[str]) -> RerankSideResult:
    start = time.perf_counter()
    try:
        model = _get_local_model()
        scores = list(model.rerank(query, documents))  # same order as `documents`, raw logits, NOT sorted
        pairs = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        results = [RerankResult(document=doc, score=round(float(s), 4), relevant=doc in relevant) for doc, s in pairs]
        latency_ms = round((time.perf_counter() - start) * 1000)
        precision_at_k, perfect_separation = _accuracy(results, relevant)
        return RerankSideResult(
            model=LOCAL_RERANK_MODEL, latency_ms=latency_ms, results=results,
            precision_at_k=precision_at_k, perfect_separation=perfect_separation,
        )
    except Exception as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return RerankSideResult(model=LOCAL_RERANK_MODEL, latency_ms=latency_ms, error=str(err))


async def _run_openrouter(query: str, documents: list[str], relevant: set[str]) -> RerankSideResult:
    start = time.perf_counter()
    if not OPENROUTER_API_KEY:
        return RerankSideResult(model=OPENROUTER_RERANK_MODEL, latency_ms=0, error="Set OPEN_ROUTER_API_KEY environment variable")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                OPENROUTER_RERANK_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={"model": OPENROUTER_RERANK_MODEL, "query": query, "documents": documents, "top_n": len(documents)},
            )
        latency_ms = round((time.perf_counter() - start) * 1000)
        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            return RerankSideResult(model=OPENROUTER_RERANK_MODEL, latency_ms=latency_ms, error=detail)
        data = resp.json()
        results = [
            RerankResult(
                document=item["document"]["text"], score=round(float(item["relevance_score"]), 6),
                relevant=item["document"]["text"] in relevant,
            )
            for item in data["results"]
        ]
        precision_at_k, perfect_separation = _accuracy(results, relevant)
        return RerankSideResult(
            model=OPENROUTER_RERANK_MODEL, latency_ms=latency_ms, results=results,
            precision_at_k=precision_at_k, perfect_separation=perfect_separation,
        )
    except Exception as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return RerankSideResult(model=OPENROUTER_RERANK_MODEL, latency_ms=latency_ms, error=str(err))


@router.post("/reranker-compare", response_model=RerankResponse)
async def reranker_compare(req: RerankRequest) -> RerankResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    documents = [doc for doc in (d.strip() for d in req.documents) if doc]
    if not documents:
        raise HTTPException(status_code=422, detail="at least one document is required")
    relevant = {doc for doc in (d.strip() for d in req.relevant) if doc and doc in documents}
    local_result = _run_local(req.query, documents, relevant)
    openrouter_result = await _run_openrouter(req.query, documents, relevant)
    return RerankResponse(local=local_result, openrouter=openrouter_result)
