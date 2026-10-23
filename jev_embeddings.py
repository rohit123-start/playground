"""POST /embeddings-latency: runs one piece of text through two embedding models and times
each — the comparison behind ai-platform's "should we switch embedding models" question
(app/embeddings.py). Not a routing decision, just a side-by-side latency/shape check:

- local: BAAI/bge-small-en-v1.5 via fastembed/ONNX, in-process, no network call. This is
  ai-platform's current model (app/embeddings.py's MODEL_NAME) — loaded here the same way,
  so the local number reflects the real steady-state cost once the model is warm.
- openrouter: qwen/qwen3-embedding-8b via OpenRouter's /embeddings endpoint (same
  AsyncOpenAI + base_url pattern as jev_memory_llm.py's chat-completions call).

Both run on every request so a cold/first-call outlier on either side is visible, not hidden
by only-once warmup; the UI distinguishes "first call" from "steady state" by letting the
caller mash the button and compare.
"""
import os
import time

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# Same model id ai-platform's app/embeddings.py locks EMBED_DIM/the pgvector column to.
LOCAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

_openrouter_client: AsyncOpenAI | None = None
_local_model = None  # fastembed.TextEmbedding, loaded lazily (first call pays the load cost)


def _get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Set OPEN_ROUTER_API_KEY environment variable")
    if _openrouter_client is None:
        _openrouter_client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _openrouter_client


def _get_local_model():
    """Lazy singleton, same pattern as app/embeddings.py's _get_model() — the ~130MB model is
    downloaded/loaded once per process and reused; only the FIRST call anywhere in this
    process pays that cost, which is exactly the "first call" vs. "steady state" split the
    UI is trying to show."""
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding
        _local_model = TextEmbedding(model_name=LOCAL_MODEL_NAME)
    return _local_model


class EmbeddingsLatencyRequest(BaseModel):
    text: str


class EmbeddingSideResult(BaseModel):
    model: str
    latency_ms: int
    dimensions: "int | None" = None
    error: "str | None" = None


class EmbeddingsLatencyResponse(BaseModel):
    local: EmbeddingSideResult
    openrouter: EmbeddingSideResult


class SemanticSearchRequest(BaseModel):
    query: str
    documents: list[str]
    relevant: list[str] = []  # ground-truth subset of `documents` considered relevant to `query`


class SemanticSearchResult(BaseModel):
    document: str
    similarity: float
    relevant: bool


class SemanticSearchSideResult(BaseModel):
    model: str
    latency_ms: int
    results: list[SemanticSearchResult] = []
    # precision@K, K = len(relevant): fraction of the top-K ranked docs that are truly relevant.
    # None when no ground truth was supplied, so the UI can tell "not scored" from "scored 0".
    precision_at_k: "float | None" = None
    perfect_separation: "bool | None" = None  # every relevant doc ranked above every irrelevant one
    error: "str | None" = None


def _run_local(text: str) -> EmbeddingSideResult:
    start = time.perf_counter()
    try:
        model = _get_local_model()
        vec = list(model.embed([text]))[0]
        latency_ms = round((time.perf_counter() - start) * 1000)
        return EmbeddingSideResult(model=LOCAL_MODEL_NAME, latency_ms=latency_ms, dimensions=len(vec))
    except Exception as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return EmbeddingSideResult(model=LOCAL_MODEL_NAME, latency_ms=latency_ms, error=str(err))


async def _run_openrouter(text: str) -> EmbeddingSideResult:
    start = time.perf_counter()
    try:
        client = _get_openrouter_client()
        resp = await client.embeddings.create(model=OPENROUTER_EMBEDDING_MODEL, input=text)
        latency_ms = round((time.perf_counter() - start) * 1000)
        return EmbeddingSideResult(
            model=OPENROUTER_EMBEDDING_MODEL, latency_ms=latency_ms,
            dimensions=len(resp.data[0].embedding),
        )
    except HTTPException as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return EmbeddingSideResult(model=OPENROUTER_EMBEDDING_MODEL, latency_ms=latency_ms, error=err.detail)
    except APIError as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return EmbeddingSideResult(model=OPENROUTER_EMBEDDING_MODEL, latency_ms=latency_ms, error=str(err))


@router.post("/embeddings-latency", response_model=EmbeddingsLatencyResponse)
async def embeddings_latency(req: EmbeddingsLatencyRequest) -> EmbeddingsLatencyResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    # Local runs on the event loop thread (fastembed/ONNX is sync, CPU-bound) — fine for a
    # one-off playground call; run it first so its cost isn't hidden behind the OpenRouter
    # round trip's own wall time.
    local_result = _run_local(req.text)
    openrouter_result = await _run_openrouter(req.text)
    return EmbeddingsLatencyResponse(local=local_result, openrouter=openrouter_result)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _ranked(documents: list[str], relevant: set[str], query_vec, doc_vecs) -> list[SemanticSearchResult]:
    sims = [(doc, _cosine(list(query_vec), list(vec))) for doc, vec in zip(documents, doc_vecs)]
    sims.sort(key=lambda pair: pair[1], reverse=True)
    return [
        SemanticSearchResult(document=doc, similarity=round(sim, 4), relevant=doc in relevant)
        for doc, sim in sims
    ]


def _score(results: list[SemanticSearchResult], relevant: set[str]) -> tuple["float | None", "bool | None"]:
    """precision@K (K = number of ground-truth relevant docs) and whether every relevant doc
    outranked every irrelevant one — the two things that matter for retrieval, not just latency."""
    k = len(relevant)
    if k == 0:
        return None, None
    top_k = results[:k]
    precision_at_k = sum(1 for r in top_k if r.relevant) / k
    perfect_separation = all(r.relevant for r in top_k)
    return round(precision_at_k, 4), perfect_separation


def _run_local_search(query: str, documents: list[str], relevant: set[str]) -> SemanticSearchSideResult:
    start = time.perf_counter()
    try:
        model = _get_local_model()
        vectors = list(model.embed([query, *documents]))
        latency_ms = round((time.perf_counter() - start) * 1000)
        results = _ranked(documents, relevant, vectors[0], vectors[1:])
        precision_at_k, perfect_separation = _score(results, relevant)
        return SemanticSearchSideResult(
            model=LOCAL_MODEL_NAME, latency_ms=latency_ms, results=results,
            precision_at_k=precision_at_k, perfect_separation=perfect_separation,
        )
    except Exception as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return SemanticSearchSideResult(model=LOCAL_MODEL_NAME, latency_ms=latency_ms, error=str(err))


async def _run_openrouter_search(query: str, documents: list[str], relevant: set[str]) -> SemanticSearchSideResult:
    start = time.perf_counter()
    try:
        client = _get_openrouter_client()
        resp = await client.embeddings.create(model=OPENROUTER_EMBEDDING_MODEL, input=[query, *documents])
        vectors = [item.embedding for item in resp.data]
        latency_ms = round((time.perf_counter() - start) * 1000)
        results = _ranked(documents, relevant, vectors[0], vectors[1:])
        precision_at_k, perfect_separation = _score(results, relevant)
        return SemanticSearchSideResult(
            model=OPENROUTER_EMBEDDING_MODEL, latency_ms=latency_ms, results=results,
            precision_at_k=precision_at_k, perfect_separation=perfect_separation,
        )
    except HTTPException as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return SemanticSearchSideResult(model=OPENROUTER_EMBEDDING_MODEL, latency_ms=latency_ms, error=err.detail)
    except APIError as err:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return SemanticSearchSideResult(model=OPENROUTER_EMBEDDING_MODEL, latency_ms=latency_ms, error=str(err))


class SemanticSearchResponse(BaseModel):
    local: SemanticSearchSideResult
    openrouter: SemanticSearchSideResult


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(req: SemanticSearchRequest) -> SemanticSearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    documents = [doc for doc in (d.strip() for d in req.documents) if doc]
    if not documents:
        raise HTTPException(status_code=422, detail="at least one document is required")
    relevant = {doc for doc in (d.strip() for d in req.relevant) if doc and doc in documents}
    local_result = _run_local_search(req.query, documents, relevant)
    openrouter_result = await _run_openrouter_search(req.query, documents, relevant)
    return SemanticSearchResponse(local=local_result, openrouter=openrouter_result)
