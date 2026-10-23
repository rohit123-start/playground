"""POST /pipeline-test: the full retrieval pipeline at corpus scale, not a single query/doc pair
— semantic search over a ~50-page synthetic corpus narrows to a semantic_top_k shortlist, the
cross-encoder reranks ONLY that shortlist down to rerank_top_k, and each stage is timed and
scored against ground truth separately, so you can see where a stage *loses* a relevant doc
(falls outside the semantic shortlist) vs. where it correctly *demotes* one (survives the
shortlist but reranked out of the final top-N).

Two full pipelines, same shape:
- local: BAAI/bge-small-en-v1.5 embeddings (jev_embeddings.py) -> Xenova/ms-marco-MiniLM-L-6-v2
  cross-encoder (jev_reranker.py), both in-process/ONNX, no network call.
- openrouter: qwen/qwen3-embedding-8b embeddings -> qwen/qwen3-reranker-8b rerank, both via
  OpenRouter (embeddings endpoint + the dedicated /api/v1/rerank endpoint).

GET /pipeline-corpus hands back a fresh deterministic (seeded) synthetic corpus: ~50 pages of
unrelated filler chunks plus a handful of near-miss distractors and exactly 3 true-relevant
chunks about one planted fact, scattered at random positions — so "find the fact in the haystack"
is actually being tested, not a 6-document toy case.
"""
import random
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jev_embeddings import (
    LOCAL_MODEL_NAME,
    OPENROUTER_EMBEDDING_MODEL,
    _get_local_model as _embed_local_model,
    _get_openrouter_client,
    _ranked,
)
from jev_reranker import (
    LOCAL_RERANK_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_RERANK_MODEL,
    OPENROUTER_RERANK_URL,
    _get_local_model as _rerank_local_model,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Synthetic ~50-page corpus
# ---------------------------------------------------------------------------

_FILLER_TOPICS = [
    "The recipe calls for simmering the tomato sauce for forty minutes, stirring occasionally so "
    "the garlic doesn't catch on the bottom of the pan. Fresh basil goes in at the very end.",
    "Voyager 1 is now over 15 billion miles from Earth, still sending back faint telemetry across "
    "interstellar space after more than four decades in flight.",
    "The 1986 World Cup final is remembered less for the score than for the two goals that defined "
    "a generation of commentary about the sport.",
    "Golden retrievers shed heavily twice a year, and a weekly brushing routine keeps the loose "
    "undercoat from matting during the transition.",
    "Pruning tomato suckers early in the season redirects the plant's energy into fewer, larger "
    "fruit instead of a tangle of competing stems.",
    "The second movement of the symphony shifts from a minor key into a brighter, almost playful "
    "theme before the brass section reintroduces the original motif.",
    "Budget airlines often charge separately for seat selection, checked bags, and even printed "
    "boarding passes, so the advertised fare rarely matches what you pay at checkout.",
    "Walking for thirty minutes a day at a brisk pace is associated with measurable improvements "
    "in resting heart rate within about six weeks.",
    "The museum's new wing houses pottery shards dated to roughly 1200 BCE, recovered from a "
    "coastal excavation site over three separate digging seasons.",
    "A light frost in early October can still allow a second harvest of hardy greens like kale "
    "and spinach if the bed is covered overnight.",
    "The city's old tram network was decommissioned in the 1960s, and most of the original rail "
    "lines now sit paved over beneath the current bus routes.",
    "Sourdough starter needs feeding roughly once a day at room temperature, or once a week if "
    "kept in the refrigerator between bakes.",
    "The chess opening was considered too passive for competitive play until a string of "
    "unexpected wins in the 1990s revived interest in it.",
    "Migratory geese typically fly in a V formation to reduce drag for all but the lead bird, "
    "which rotates position throughout the journey.",
    "The documentary follows three beekeepers across a single season, from the first spring "
    "inspection to the final honey extraction in late summer.",
    "A standard marathon course measures 26.2 miles, a distance fixed by the 1908 London Olympics "
    "rather than any round number.",
    "The library's rare books room keeps its humidity below fifty percent year-round to slow the "
    "degradation of nineteenth-century paper stock.",
    "Volcanic soil in the region is unusually rich in potassium, which is part of why the local "
    "vineyards produce such a distinctive mineral finish.",
    "The bridge's suspension cables are inspected every five years by technicians who rappel down "
    "the main span rather than using scaffolding.",
    "Competitive crossword constructors often start with the long theme answers before filling in "
    "the shorter connecting words around them.",
]

_NEAR_MISS_TOPICS = [
    "We're considering moving off Supabase to a custom Node service sometime next quarter, but "
    "nothing has been decided and it isn't scheduled yet.",
    "The frontend team finished migrating the dashboard to Next.js 14 last sprint, separate from "
    "any backend changes.",
    "Someone asked in standup whether we should evaluate DynamoDB as an alternative, but the team "
    "agreed to revisit that conversation after launch.",
    "The infra channel has an open thread about Redis caching in front of the database, which is "
    "unrelated to which backend service we're actually running.",
    "An old design doc from last year proposed a Firebase backend, but that plan was shelved "
    "before any of it was built.",
]

_RELEVANT_FACTS = [
    "We decided to use Supabase as our backend for now.",
    "Actually, let's go mobile first instead of web. Also drop the Supabase decision, we haven't picked a backend yet.",
    "The team confirmed in Friday's sync that Supabase stays the backend through the v1 launch.",
]

PAGES = 50
CHUNKS_PER_PAGE = 3
DEFAULT_SEED = 42
QUERY = "What backend are we using?"


def _generate_corpus(seed: int = DEFAULT_SEED) -> tuple[str, list[str], list[str]]:
    rng = random.Random(seed)
    total = PAGES * CHUNKS_PER_PAGE
    pool = list(_FILLER_TOPICS) * (total // len(_FILLER_TOPICS) + 1)
    rng.shuffle(pool)
    filler = pool[: total - len(_NEAR_MISS_TOPICS) - len(_RELEVANT_FACTS)]
    documents = filler + list(_NEAR_MISS_TOPICS) + list(_RELEVANT_FACTS)
    rng.shuffle(documents)
    # Tag each chunk with its (shuffled) page/section position, same way a real page-chunked
    # document would be labeled, so duplicated filler text isn't byte-identical across the corpus.
    tagged = [f"[p{i // CHUNKS_PER_PAGE + 1}.{i % CHUNKS_PER_PAGE + 1}] {doc}" for i, doc in enumerate(documents)]
    relevant_tagged = [tagged[documents.index(fact)] for fact in _RELEVANT_FACTS]
    return QUERY, tagged, relevant_tagged


@router.get("/pipeline-corpus")
async def pipeline_corpus(seed: int = DEFAULT_SEED):
    query, documents, relevant = _generate_corpus(seed)
    return {"query": query, "documents": documents, "relevant": relevant, "pages": PAGES}


# ---------------------------------------------------------------------------
# Pipeline: semantic search (whole corpus) -> cross-encoder rerank (shortlist only)
# ---------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    query: str
    documents: list[str]
    relevant: list[str] = []
    semantic_top_k: int = 20
    rerank_top_k: int = 3


class PipelineDoc(BaseModel):
    document: str
    semantic_similarity: float
    rerank_score: "float | None" = None
    relevant: bool


class PipelineSideResult(BaseModel):
    embed_model: str
    rerank_model: str
    corpus_size: int
    embed_latency_ms: int
    rerank_latency_ms: int
    total_latency_ms: int
    semantic_recall_at_k: "float | None" = None  # of all relevant docs, fraction that survived into the shortlist
    final_precision: "float | None" = None  # of the final top-N, fraction that are truly relevant
    final_recall: "float | None" = None  # of all relevant docs, fraction present in the final top-N
    final_results: list[PipelineDoc] = []
    error: "str | None" = None


class PipelineResponse(BaseModel):
    local: PipelineSideResult  # bge-small embed + Xenova rerank, both local
    openrouter: PipelineSideResult  # qwen3-embedding-8b embed + qwen3-reranker-8b rerank, both hosted
    mixed: PipelineSideResult  # bge-small embed (local) + qwen3-reranker-8b rerank (hosted)


def _recall(subset_relevant_flags: list[bool], relevant_count: int) -> "float | None":
    if relevant_count == 0:
        return None
    return round(sum(1 for f in subset_relevant_flags if f) / relevant_count, 4)


def _embed_and_shortlist_local(query: str, documents: list[str], relevant: set[str], semantic_top_k: int):
    """Returns (shortlist docs, sim_by_doc, semantic_recall_at_k, embed_latency_ms)."""
    t0 = time.perf_counter()
    embed_model = _embed_local_model()
    vectors = list(embed_model.embed([query, *documents]))
    embed_latency_ms = round((time.perf_counter() - t0) * 1000)
    ranked = _ranked(documents, relevant, vectors[0], vectors[1:])
    shortlist = ranked[:semantic_top_k]
    semantic_recall_at_k = _recall([r.relevant for r in shortlist], len(relevant))
    sim_by_doc = {r.document: r.similarity for r in shortlist}
    return [r.document for r in shortlist], sim_by_doc, semantic_recall_at_k, embed_latency_ms


async def _embed_and_shortlist_openrouter(query: str, documents: list[str], relevant: set[str], semantic_top_k: int):
    t0 = time.perf_counter()
    client = _get_openrouter_client()
    resp = await client.embeddings.create(model=OPENROUTER_EMBEDDING_MODEL, input=[query, *documents])
    vectors = [item.embedding for item in resp.data]
    embed_latency_ms = round((time.perf_counter() - t0) * 1000)
    ranked = _ranked(documents, relevant, vectors[0], vectors[1:])
    shortlist = ranked[:semantic_top_k]
    semantic_recall_at_k = _recall([r.relevant for r in shortlist], len(relevant))
    sim_by_doc = {r.document: r.similarity for r in shortlist}
    return [r.document for r in shortlist], sim_by_doc, semantic_recall_at_k, embed_latency_ms


def _rerank_local(query: str, subset_docs: list[str]):
    """Returns (ranked (doc, score) pairs desc, rerank_latency_ms)."""
    t0 = time.perf_counter()
    rerank_model = _rerank_local_model()
    scores = list(rerank_model.rerank(query, subset_docs))
    rerank_latency_ms = round((time.perf_counter() - t0) * 1000)
    reranked = sorted(zip(subset_docs, scores), key=lambda pair: pair[1], reverse=True)
    return reranked, rerank_latency_ms


async def _rerank_openrouter(query: str, subset_docs: list[str]):
    """Returns (ranked (doc, score) pairs desc, rerank_latency_ms). Raises on a non-200 response."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as hc:
        rr = await hc.post(
            OPENROUTER_RERANK_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": OPENROUTER_RERANK_MODEL, "query": query, "documents": subset_docs, "top_n": len(subset_docs)},
        )
    rerank_latency_ms = round((time.perf_counter() - t0) * 1000)
    if rr.status_code != 200:
        detail = rr.text
        try:
            detail = rr.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise RuntimeError(detail)
    data = rr.json()
    reranked = [(item["document"]["text"], float(item["relevance_score"])) for item in data["results"]]
    return reranked, rerank_latency_ms


def _finish(
    embed_model: str, rerank_model: str, corpus_size: int, embed_latency_ms: int, rerank_latency_ms: int,
    start_total: float, semantic_recall_at_k: "float | None", sim_by_doc: dict, reranked: list, rerank_top_k: int,
    relevant: set[str],
) -> PipelineSideResult:
    final_top = reranked[:rerank_top_k]
    final_results = [
        PipelineDoc(
            document=doc, semantic_similarity=round(sim_by_doc.get(doc, 0.0), 4),
            rerank_score=round(float(score), 6), relevant=doc in relevant,
        )
        for doc, score in final_top
    ]
    final_precision = round(sum(1 for d in final_results if d.relevant) / len(final_results), 4) if final_results else None
    final_recall = _recall([d.relevant for d in final_results], len(relevant))
    total_latency_ms = round((time.perf_counter() - start_total) * 1000)
    return PipelineSideResult(
        embed_model=embed_model, rerank_model=rerank_model, corpus_size=corpus_size,
        embed_latency_ms=embed_latency_ms, rerank_latency_ms=rerank_latency_ms, total_latency_ms=total_latency_ms,
        semantic_recall_at_k=semantic_recall_at_k, final_precision=final_precision, final_recall=final_recall,
        final_results=final_results,
    )


def _failed(embed_model: str, rerank_model: str, corpus_size: int, start_total: float, err: Exception) -> PipelineSideResult:
    total_latency_ms = round((time.perf_counter() - start_total) * 1000)
    return PipelineSideResult(
        embed_model=embed_model, rerank_model=rerank_model, corpus_size=corpus_size,
        embed_latency_ms=0, rerank_latency_ms=0, total_latency_ms=total_latency_ms, error=str(err),
    )


def _run_local_pipeline(
    query: str, documents: list[str], relevant: set[str], semantic_top_k: int, rerank_top_k: int,
) -> PipelineSideResult:
    start_total = time.perf_counter()
    try:
        subset_docs, sim_by_doc, semantic_recall_at_k, embed_latency_ms = _embed_and_shortlist_local(
            query, documents, relevant, semantic_top_k,
        )
        reranked, rerank_latency_ms = _rerank_local(query, subset_docs)
        return _finish(
            LOCAL_MODEL_NAME, LOCAL_RERANK_MODEL, len(documents), embed_latency_ms, rerank_latency_ms,
            start_total, semantic_recall_at_k, sim_by_doc, reranked, rerank_top_k, relevant,
        )
    except Exception as err:
        return _failed(LOCAL_MODEL_NAME, LOCAL_RERANK_MODEL, len(documents), start_total, err)


async def _run_openrouter_pipeline(
    query: str, documents: list[str], relevant: set[str], semantic_top_k: int, rerank_top_k: int,
) -> PipelineSideResult:
    start_total = time.perf_counter()
    if not OPENROUTER_API_KEY:
        return _failed(
            OPENROUTER_EMBEDDING_MODEL, OPENROUTER_RERANK_MODEL, len(documents), start_total,
            RuntimeError("Set OPEN_ROUTER_API_KEY environment variable"),
        )
    try:
        subset_docs, sim_by_doc, semantic_recall_at_k, embed_latency_ms = await _embed_and_shortlist_openrouter(
            query, documents, relevant, semantic_top_k,
        )
        reranked, rerank_latency_ms = await _rerank_openrouter(query, subset_docs)
        return _finish(
            OPENROUTER_EMBEDDING_MODEL, OPENROUTER_RERANK_MODEL, len(documents), embed_latency_ms, rerank_latency_ms,
            start_total, semantic_recall_at_k, sim_by_doc, reranked, rerank_top_k, relevant,
        )
    except Exception as err:
        return _failed(OPENROUTER_EMBEDDING_MODEL, OPENROUTER_RERANK_MODEL, len(documents), start_total, err)


async def _run_mixed_pipeline(
    query: str, documents: list[str], relevant: set[str], semantic_top_k: int, rerank_top_k: int,
) -> PipelineSideResult:
    """bge-small (local) retrieval feeding qwen3-reranker-8b (hosted) reranking — the combination
    that keeps the fast/free embedding stage and only pays the network cost where it showed a real
    accuracy gain in jev_pipeline.py's local-vs-openrouter test."""
    start_total = time.perf_counter()
    if not OPENROUTER_API_KEY:
        return _failed(
            LOCAL_MODEL_NAME, OPENROUTER_RERANK_MODEL, len(documents), start_total,
            RuntimeError("Set OPEN_ROUTER_API_KEY environment variable"),
        )
    try:
        subset_docs, sim_by_doc, semantic_recall_at_k, embed_latency_ms = _embed_and_shortlist_local(
            query, documents, relevant, semantic_top_k,
        )
        reranked, rerank_latency_ms = await _rerank_openrouter(query, subset_docs)
        return _finish(
            LOCAL_MODEL_NAME, OPENROUTER_RERANK_MODEL, len(documents), embed_latency_ms, rerank_latency_ms,
            start_total, semantic_recall_at_k, sim_by_doc, reranked, rerank_top_k, relevant,
        )
    except Exception as err:
        return _failed(LOCAL_MODEL_NAME, OPENROUTER_RERANK_MODEL, len(documents), start_total, err)


@router.post("/pipeline-test", response_model=PipelineResponse)
async def pipeline_test(req: PipelineRequest) -> PipelineResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    documents = [doc for doc in (d.strip() for d in req.documents) if doc]
    if not documents:
        raise HTTPException(status_code=422, detail="at least one document is required")
    if req.semantic_top_k < 1 or req.rerank_top_k < 1:
        raise HTTPException(status_code=422, detail="semantic_top_k and rerank_top_k must be >= 1")
    relevant = {doc for doc in (d.strip() for d in req.relevant) if doc and doc in documents}
    local_result = _run_local_pipeline(req.query, documents, relevant, req.semantic_top_k, req.rerank_top_k)
    openrouter_result = await _run_openrouter_pipeline(req.query, documents, relevant, req.semantic_top_k, req.rerank_top_k)
    mixed_result = await _run_mixed_pipeline(req.query, documents, relevant, req.semantic_top_k, req.rerank_top_k)
    return PipelineResponse(local=local_result, openrouter=openrouter_result, mixed=mixed_result)
