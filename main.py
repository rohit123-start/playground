from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

import bge_small_intent_embed
import jev_ai

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager loading: load the embedding model and embed the example prompts at startup.
    bge_small_intent_embed.load()
    yield


app = FastAPI(title="Tool Selector", lifespan=lifespan)
app.include_router(jev_ai.router)  # POST /jev-open-router-tool
app.include_router(bge_small_intent_embed.router)  # POST /bge-small-intent-embed


@app.get("/playground", include_in_schema=False)
async def playground():
    return FileResponse(BASE_DIR / "playground.html")


@app.get("/playground/examples", include_in_schema=False)
async def playground_examples():
    return FileResponse(BASE_DIR / "prompt_intent_examples.json")
