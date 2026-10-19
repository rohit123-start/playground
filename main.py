from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse

import audio_processing
import file_processing
import image_processing
import jev_ai
import jev_model_picker
import jev_prompt_scan
import jev_tool_model_routing
import video_processing
import video_splitter

BASE_DIR = Path(__file__).parent

# Playground pages are revalidated on every load (never served from the browser cache), and the
# questions JSON is never cached, so edits and new page versions show up without a hard refresh.
NO_CACHE = {"Cache-Control": "no-cache"}
NO_STORE = {"Cache-Control": "no-store"}


def _page(name: str) -> FileResponse:
    return FileResponse(BASE_DIR / name, headers=NO_CACHE)

app = FastAPI(title="Tool Selector")
app.include_router(jev_ai.router)  # POST /jev-open-router-tool
app.include_router(jev_prompt_scan.router)  # POST /jev-prompt-scan-open-router
app.include_router(jev_model_picker.router)  # POST /jev-model-picker-{low,medium,high}
app.include_router(jev_tool_model_routing.router)  # POST /jev-tool-model-routing
app.include_router(video_splitter.router)  # POST /video-splitter/split
app.include_router(video_processing.router)  # POST /video-processing/process
app.include_router(file_processing.router)  # POST /file-processing/process
app.include_router(image_processing.router)  # POST /image-processing/process
app.include_router(audio_processing.router)  # POST /audio-processing/process


@app.get("/playground", include_in_schema=False)
async def playground_tool():
    return _page("playground_tool.html")


@app.get("/playground/prompt-scan", include_in_schema=False)
async def playground_prompt_scan():
    return _page("playground_prompt_scan.html")


@app.get("/playground/model-picker", include_in_schema=False)
async def playground_model_picker():
    return _page("playground_model_picker.html")


@app.get("/playground/tool-model-routing", include_in_schema=False)
async def playground_tool_model_routing():
    return _page("playground_tool_model_routing.html")


@app.get("/playground/video-splitter", include_in_schema=False)
async def playground_video_splitter():
    return _page("playground_video_splitter.html")


@app.get("/playground/video-processing", include_in_schema=False)
async def playground_video_processing():
    return _page("playground_video_processing.html")


@app.get("/playground/file-processing", include_in_schema=False)
async def playground_file_processing():
    return _page("playground_file_processing.html")


@app.get("/playground/image-processing", include_in_schema=False)
async def playground_image_processing():
    return _page("playground_image_processing.html")


@app.get("/playground/audio-processing", include_in_schema=False)
async def playground_audio_processing():
    return _page("playground_audio_processing.html")


# Questions JSON shown and edited in the playground pages. PUT validates, saves, and applies
# the new questions to the next request (no restart).
@app.get("/playground/questions", include_in_schema=False)
async def playground_questions(response: Response):
    response.headers.update(NO_STORE)
    return jev_ai.question_store.get()


@app.put("/playground/questions", include_in_schema=False)
async def save_playground_questions(body: dict):
    return jev_ai.question_store.put(body)


@app.get("/playground/prompt-scan/questions", include_in_schema=False)
async def playground_prompt_scan_questions(response: Response):
    response.headers.update(NO_STORE)
    return jev_prompt_scan.question_store.get()


@app.put("/playground/prompt-scan/questions", include_in_schema=False)
async def save_playground_prompt_scan_questions(body: dict):
    return jev_prompt_scan.question_store.put(body)


def _picker_store(tier: str):
    if tier not in jev_model_picker.question_stores:
        raise HTTPException(status_code=404, detail="Unknown tier")
    return jev_model_picker.question_stores[tier]


@app.get("/playground/model-picker/questions/{tier}", include_in_schema=False)
async def playground_model_picker_questions(tier: str, response: Response):
    response.headers.update(NO_STORE)
    return _picker_store(tier).get()


@app.put("/playground/model-picker/questions/{tier}", include_in_schema=False)
async def save_playground_model_picker_questions(tier: str, body: dict):
    return _picker_store(tier).put(body)


@app.get("/playground/tool-model-routing/questions", include_in_schema=False)
async def playground_tool_model_routing_questions(response: Response):
    response.headers.update(NO_STORE)
    return jev_tool_model_routing.question_store.get()


@app.put("/playground/tool-model-routing/questions", include_in_schema=False)
async def save_playground_tool_model_routing_questions(body: dict):
    return jev_tool_model_routing.question_store.put(body)
