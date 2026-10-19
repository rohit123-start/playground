"""POST /image-processing/process: standalone image analysis — direct port of ai-platform's
app/libraries/image_analysis.py, which is a fixed-prompt Qwen Omni vision call and genuinely
nothing more for a standalone image upload. The richer pipeline ai-platform actually has around
images (size/area-fraction filtering, trivial-icon skipping, per-file caps) only applies to
EMBEDDED images found while ingesting a PDF/DOCX/PPTX/XLSX — that's already ported into
file_processing.py's document branches. There's no separate resize/compress/format-conversion
step for a direct image upload in ai-platform, so this doesn't invent one either.

What this section adds beyond a single analyze_image() call: support for MULTIPLE images in one
request, run concurrently (asyncio.gather) with each image's own Omni latency/tokens tracked —
the same pattern already established for every other Omni-backed playground section — plus a
size guard against the same real Qwen Omni per-item limit video_splitter.py found empirically
(OMNI_MAX_BYTES), since ai-platform's own images are small document-embedded ones and never hit
it, but an arbitrary direct upload here easily could."""
import asyncio
import io
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image as PILImage

import omni_video
from file_processing import IMAGE_EXTS, _ext
from video_splitter import OMNI_MAX_BYTES

router = APIRouter()


def _dimensions(data: bytes) -> tuple[int | None, int | None]:
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            return im.width, im.height
    except Exception:
        return None, None


async def _analyze_one(filename: str, data: bytes, mime: str) -> dict:
    width, height = _dimensions(data)
    size_bytes = len(data)
    base = {"filename": filename, "mime": mime, "size_bytes": size_bytes, "width": width, "height": height}

    if size_bytes > OMNI_MAX_BYTES:
        return {**base, "text": None, "usage": None, "latency_ms": None,
                "error": f"image too large for Omni ({size_bytes / 1_000_000:.1f} MB > 15 MB)"}
    try:
        text, usage, latency_ms = await omni_video.analyze_image(data, mime=mime)
        return {**base, "text": text, "usage": usage, "latency_ms": latency_ms, "error": None}
    except omni_video.OmniError as err:
        return {**base, "text": None, "usage": None, "latency_ms": None, "error": str(err)}


@router.post("/image-processing/process")
async def process_images(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    jobs: list[tuple[str, bytes, str]] = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        mime = f.content_type if (f.content_type or "").startswith("image/") else IMAGE_EXTS.get(_ext(f.filename), "image/png")
        jobs.append((f.filename or "image", data, mime))
    if not jobs:
        raise HTTPException(status_code=400, detail="No non-empty image files uploaded")

    task_start = time.perf_counter()
    results = await asyncio.gather(*(_analyze_one(name, data, mime) for name, data, mime in jobs))
    latency_ms = round((time.perf_counter() - task_start) * 1000)

    omni_latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    tokens_total = sum(r["usage"]["total_tokens"] for r in results if r["usage"])

    return {
        "image_count": len(results),
        "omni_configured": omni_video.is_configured(),
        "omni_key_env": omni_video.OMNI_API_KEY_ENV,
        "omni_tokens_total": tokens_total,
        # Sum: total Omni compute across every image — not wall clock, since all images in the
        # request run concurrently. Max: the single slowest call, closer to the actual wait.
        "omni_latency_ms": sum(omni_latencies),
        "omni_latency_max_ms": max(omni_latencies) if omni_latencies else 0,
        "latency_ms": latency_ms,
        "images": results,
    }
