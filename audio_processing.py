"""POST /audio-processing/process: standalone audio transcription — direct port of ai-platform's
app/libraries/audio_processing.py, a fixed-prompt Qwen Omni call with no preprocessing or
chunking in that codebase (confirmed: app/rag/ingest.py sends the 'audio' kind whole, same as
image/video — "standalone media has no text fallback").

Unlike image_processing.py though, audio commonly exceeds Omni's real per-item size limit just
from running a few minutes long — far more likely than for a single photo — so this reuses
video_processing.py's bitrate-targeted audio compressor as a retry-on-oversize safety net (same
pattern video_splitter.py / video_processing.py already use for their own oversized chunks),
rather than just rejecting the file outright the way image_processing.py does for images, where
hitting that limit is rare enough that a clear error beats the added complexity.

AUDIO_MAX_BYTES is calibrated separately from video_splitter.py's OMNI_MAX_BYTES (~15 MB, for
video/image): live testing against the real OpenRouter/Qwen endpoint found audio starts failing
with "the audio is empty" once its base64 payload crosses 8 MiB (confirmed by bisection — 8,385,432
b64 bytes succeeded, 8,428,344 failed, bracketing the 8*1024*1024 boundary), giving a raw-byte
ceiling of 6,291,456 — well under the video/image limit, so audio needs its own constant.

Supports multiple audio files per request, run concurrently, each with its own Omni
latency/tokens tracked — the pattern already established for every other Omni-backed section."""
import asyncio
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import omni_video
from file_processing import AUDIO_EXTS, _ext
from video_processing import _compress_audio_for_omni, _probe_media

AUDIO_MAX_BYTES = 6_000_000  # real ceiling is 6,291,456 (8 MiB of base64) — margin for encoding overhead
AUDIO_COMPRESS_TARGET_MB = 5.0

router = APIRouter()


async def _process_one(path: Path, filename: str, audio_fmt: str, tmp_dir: Path) -> dict:
    """Never raises: every failure mode (unreadable file, compression failure, Omni failure)
    becomes this item's own error instead of failing the whole batch — same fail-open rule every
    other Omni-backed section in this playground uses."""
    size_bytes = path.stat().st_size
    base = {"filename": filename, "size_bytes": size_bytes, "duration": None}

    try:
        info = await _probe_media(path)
        duration = info["duration"]
        base["duration"] = round(duration, 1) if duration else None
    except Exception as err:
        return {**base, "text": None, "usage": None, "latency_ms": None, "error": f"could not read audio info: {err}"}

    send_path, send_fmt = path, audio_fmt
    if size_bytes > AUDIO_MAX_BYTES:
        target_mb = AUDIO_COMPRESS_TARGET_MB
        for _ in range(3):  # retry at a lower target if a hard-to-compress file overshoots
            try:
                send_path = await _compress_audio_for_omni(send_path, max(duration, 0.5), tmp_dir, target_mb)
            except Exception as err:
                return {**base, "text": None, "usage": None, "latency_ms": None,
                        "error": f"audio too large for Omni ({size_bytes / 1_000_000:.1f} MB > 6 MB) and compression failed: {err}"}
            send_fmt = "mp3"
            sent_size = send_path.stat().st_size
            if sent_size <= AUDIO_MAX_BYTES:
                break
            target_mb *= 0.8 * AUDIO_MAX_BYTES / sent_size
        else:
            sent_mb = send_path.stat().st_size / 1_000_000
            return {**base, "text": None, "usage": None, "latency_ms": None,
                    "error": f"still too large for Omni after compression ({sent_mb:.1f} MB > 6 MB)"}

    try:
        text, usage, latency_ms = await omni_video.transcribe_audio(send_path.read_bytes(), audio_format=send_fmt)
        return {**base, "text": text, "usage": usage, "latency_ms": latency_ms, "error": None}
    except omni_video.OmniError as err:
        return {**base, "text": None, "usage": None, "latency_ms": None, "error": str(err)}


@router.post("/audio-processing/process")
async def process_audios(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    with tempfile.TemporaryDirectory(prefix="audio-processing-") as tmp:
        tmp_dir = Path(tmp)
        jobs: list[tuple[Path, str, str]] = []
        for i, f in enumerate(files):
            data = await f.read()
            if not data:
                continue
            filename = f.filename or f"audio_{i}"
            audio_fmt = AUDIO_EXTS.get(_ext(filename), "mp3")
            path = tmp_dir / f"{i:02d}_{filename}"  # index prefix avoids collisions on same-name uploads
            path.write_bytes(data)
            jobs.append((path, filename, audio_fmt))
        if not jobs:
            raise HTTPException(status_code=400, detail="No non-empty audio files uploaded")

        task_start = time.perf_counter()
        results = await asyncio.gather(*(_process_one(path, name, fmt, tmp_dir) for path, name, fmt in jobs))
        latency_ms = round((time.perf_counter() - task_start) * 1000)

    omni_latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    tokens_total = sum(r["usage"]["total_tokens"] for r in results if r["usage"])

    return {
        "audio_count": len(results),
        "omni_configured": omni_video.is_configured(),
        "omni_key_env": omni_video.OMNI_API_KEY_ENV,
        "omni_tokens_total": tokens_total,
        # Sum: total Omni compute across every file — not wall clock, since all files in the
        # request run concurrently. Max: the single slowest call, closer to the actual wait.
        "omni_latency_ms": sum(omni_latencies),
        "omni_latency_max_ms": max(omni_latencies) if omni_latencies else 0,
        "latency_ms": latency_ms,
        "audios": results,
    }
