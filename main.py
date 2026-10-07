from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

import jev_ai

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Tool Selector")
app.include_router(jev_ai.router)  # POST /jev-open-router-tool


@app.get("/playground", include_in_schema=False)
async def playground():
    return FileResponse(BASE_DIR / "playground.html")


@app.get("/playground/examples", include_in_schema=False)
async def playground_examples():
    return FileResponse(BASE_DIR / "prompt_intent_examples.json")
