from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

import bge_small_intent_embed
import jev_ai
import qwen8b_intent_embed
import qwen_intent_embed
import voyage_intent_embed

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager loading: load the embedding model and embed the example prompts at startup.
    bge_small_intent_embed.load()
    await voyage_intent_embed.load()
    await qwen_intent_embed.load()
    await qwen8b_intent_embed.load()
    yield


app = FastAPI(title="Tool Selector", lifespan=lifespan)
app.include_router(jev_ai.router)  # POST /jev-open-router-     
app.include_router(bge_small_intent_embed.router)  # POST /bge-small-intent-embed
app.include_router(voyage_intent_embed.router)  # POST /voyage-intent-embed
app.include_router(qwen_intent_embed.router)  # POST /qwen-intent-embed
app.include_router(qwen8b_intent_embed.router)  # POST /qwen-8b-intent-embed


@app.get("/playground", include_in_schema=False)
async def playground_tool():
    return FileResponse(BASE_DIR / "playground_tool.html")


@app.get("/playground/intent", include_in_schema=False)
async def playground_intent():
    return FileResponse(BASE_DIR / "playground_intent.html")


@app.get("/playground/voyage", include_in_schema=False)
async def playground_voyage():
    return FileResponse(BASE_DIR / "playground_voyage.html")


@app.get("/playground/qwen", include_in_schema=False)
async def playground_qwen():
    return FileResponse(BASE_DIR / "playground_qwen.html")


@app.get("/playground/qwen8b", include_in_schema=False)
async def playground_qwen8b():
    return FileResponse(BASE_DIR / "playground_qwen8b.html")


@app.get("/playground/examples", include_in_schema=False)
async def playground_examples():
    return FileResponse(BASE_DIR / "prompt_intent_examples.json")


@app.get("/playground/questions", include_in_schema=False)
async def playground_questions():
    return FileResponse(BASE_DIR / "questions_jev_open_router_tool.json")
