from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

import jev_ai
import jev_model_picker
import jev_prompt_scan

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Tool Selector")
app.include_router(jev_ai.router)  # POST /jev-open-router-tool
app.include_router(jev_prompt_scan.router)  # POST /jev-prompt-scan-open-router
app.include_router(jev_model_picker.router)  # POST /jev-model-picker-{low,medium,high}


@app.get("/playground", include_in_schema=False)
async def playground_tool():
    return FileResponse(BASE_DIR / "playground_tool.html")


@app.get("/playground/prompt-scan", include_in_schema=False)
async def playground_prompt_scan():
    return FileResponse(BASE_DIR / "playground_prompt_scan.html")


@app.get("/playground/model-picker", include_in_schema=False)
async def playground_model_picker():
    return FileResponse(BASE_DIR / "playground_model_picker.html")


@app.get("/playground/questions", include_in_schema=False)
async def playground_questions():
    return FileResponse(BASE_DIR / "questions_jev_open_router_tool.json")
