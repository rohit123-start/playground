"""POST /video-splitter/split: splits an uploaded video into parts using ffmpeg stream copy
(-c copy). No re-encoding happens, so it's fast and lossless, but cuts snap to the nearest
preceding keyframe rather than the exact requested time. The ffmpeg binary comes from
imageio-ffmpeg, so no system install is required. Each part is then optionally transcribed with
Qwen Omni (see omni_video.py) so the playground can show what's actually in each clip."""
import asyncio
import base64
import re
import tempfile
import time
import zipfile
from enum import Enum
from io import BytesIO
from pathlib import Path

import imageio_ffmpeg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import omni_video

router = APIRouter()

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
MIME_BY_SUFFIX = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".m4v": "video/mp4",
    ".mkv": "video/x-matroska", ".webm": "video/webm", ".avi": "video/x-msvideo",
}

# Two separate DashScope (Qwen Omni) limits apply to the base64 data-uri payload, confirmed
# empirically against the live API:
#   1. A generic JSON-parser cap (Jackson StreamReadConstraints): 28,000,000 chars.
#   2. DashScope's own, tighter check: "Exceeded limit on max bytes per data-uri item: 20971520"
#      (exactly 20 MiB) — measured on the base64 STRING itself, not the raw file.
# #2 is the binding one. Base64 inflates raw bytes by 4/3, so the raw-video ceiling is
# 20,971,520 * 3/4 = 15,728,640 bytes (15 MiB). Stay comfortably under it.
OMNI_MAX_BYTES = 15_000_000
OMNI_COMPRESS_TARGET_MB = 13.0


class SplitMode(str, Enum):
    PARTS = "parts"  # value = number of equal-length parts (>= 2)
    DURATION = "duration"  # value = seconds per part (> 0)
    TIMESTAMPS = "timestamps"  # value = comma-separated cut points in seconds
    SIZE = "size"  # value = target max MB per part (> 0)


async def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_EXE, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, stderr.decode("utf-8", "replace")


async def _probe_duration(path: Path) -> float:
    # No output file is given, so ffmpeg prints the input's info to stderr and exits non-zero.
    _, stderr = await _run_ffmpeg(["-i", str(path)])
    m = DURATION_RE.search(stderr)
    if not m:
        raise HTTPException(status_code=400, detail="Could not read video duration (unsupported or corrupt file)")
    hours, minutes, seconds = m.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _segments(mode: SplitMode, value: str, duration: float) -> list[tuple[float, float]]:
    if mode is SplitMode.PARTS:
        try:
            parts = int(value)
        except ValueError:
            raise HTTPException(status_code=400, detail="parts must be an integer")
        if parts < 2:
            raise HTTPException(status_code=400, detail="parts must be at least 2")
        step = duration / parts
        bounds = [i * step for i in range(parts + 1)]
    elif mode is SplitMode.DURATION:
        try:
            seg_len = float(value)
        except ValueError:
            raise HTTPException(status_code=400, detail="duration must be a number")
        if seg_len <= 0:
            raise HTTPException(status_code=400, detail="duration must be > 0")
        bounds = []
        t = 0.0
        while t < duration:
            bounds.append(t)
            t += seg_len
        bounds.append(duration)
    else:
        try:
            cuts = sorted({float(x) for x in value.split(",") if x.strip()})
        except ValueError:
            raise HTTPException(status_code=400, detail="timestamps must be comma-separated numbers")
        cuts = [c for c in cuts if 0 < c < duration]
        if not cuts:
            raise HTTPException(status_code=400, detail="timestamps must include at least one cut strictly between 0 and the video's duration")
        bounds = [0.0, *cuts, duration]

    bounds[-1] = duration  # avoid float drift leaving a near-zero trailing segment
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


async def _cut_segment(input_path: Path, start: float, end: float, out_path: Path) -> None:
    args = [
        "-y", "-ss", f"{start:.3f}", "-i", str(input_path),
        "-t", f"{end - start:.3f}", "-map", "0", "-c", "copy",
        "-avoid_negative_ts", "make_zero", str(out_path),
    ]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not out_path.exists():
        raise HTTPException(status_code=500, detail=f"ffmpeg failed cutting {out_path.name}: {stderr[-800:]}")


async def _split_by_size(
    input_path: Path, duration: float, target_bytes: float, tmp_dir: Path, stem: str, suffix: str,
) -> list[tuple[Path, float, float]]:
    """Cuts segments sized to land near target_bytes each. Stream copy can't predict a cut's
    output size ahead of time — it depends on local bitrate, not just duration — so each cut is
    measured after the fact and re-cut smaller if it overshoots, then the next segment starts
    from where this one actually ended (never from the original estimate)."""
    total_bytes = input_path.stat().st_size
    avg_bps = total_bytes / duration if duration > 0 else 0
    local_bps = avg_bps  # updated from each cut's actual rate, so estimates track regional
    out: list[tuple[Path, float, float]] = []       # bitrate shifts instead of reverting to the
    start, i = 0.0, 0                                # file-wide average every segment.
    while start < duration - 1e-6:
        i += 1
        bps = local_bps if local_bps > 0 else avg_bps
        seg_len = (target_bytes / bps) if bps > 0 else (duration - start)
        end = min(start + seg_len, duration)
        out_path = tmp_dir / f"{stem}_part{i:02d}{suffix}"
        for _ in range(5):  # shrink and re-cut if this stream-copied slice overshoots target
            await _cut_segment(input_path, start, end, out_path)
            actual = out_path.stat().st_size
            # Accept once it fits, or once it can't shrink further — NOT just because the
            # estimate happened to reach the file's end (a stale bitrate estimate can land there
            # even when there's plenty of oversized content left to split into more parts).
            if actual <= target_bytes or end - start <= 0.5 + 1e-6:
                break
            end = max(start + (end - start) * 0.85 * (target_bytes / actual), start + 0.5)
        if end > start:
            local_bps = out_path.stat().st_size / (end - start)
        out.append((out_path, start, end))
        start = end
    return out


async def _compress_for_omni(path: Path, duration: float, tmp_dir: Path, target_mb: float) -> Path:
    """Re-encodes path down to roughly target_mb so it clears OMNI_MAX_BYTES. Sized from the
    clip's own known duration (start/end from _segments), so unlike the reference this was
    ported from, it never shells out to ffprobe — imageio-ffmpeg doesn't bundle it."""
    target_bitrate = max(int(target_mb * 8_000_000 / duration) - 64_000, 100_000)  # reserve 64kbps for audio
    out_path = tmp_dir / f"{path.stem}_compressed.mp4"
    args = [
        "-y", "-i", str(path),
        "-c:v", "libx264", "-preset", "fast",
        "-b:v", str(target_bitrate), "-maxrate", str(target_bitrate), "-bufsize", str(target_bitrate * 2),
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not out_path.exists():
        raise RuntimeError(stderr[-800:])
    return out_path


async def _transcribe(path: Path, mime: str, duration: float, tmp_dir: Path) -> tuple[str | None, str | None, bool]:
    """Returns (text, error, compressed) — exactly one of text/error is set. Never raises: a
    transcription failure on one clip shouldn't fail the whole split (same fail-open rule
    ai-platform's ingest uses). Compression only affects what's sent to Omni — the zip keeps the
    original lossless stream-copied part."""
    send_path, send_mime, compressed = path, mime, False
    size = path.stat().st_size
    if size > OMNI_MAX_BYTES:
        target_mb = OMNI_COMPRESS_TARGET_MB
        for _ in range(3):  # retry at a lower target if a hard-to-compress clip overshoots
            try:
                send_path = await _compress_for_omni(path, duration, tmp_dir, target_mb)
            except Exception as err:
                return None, f"clip too large for Omni OCR ({size / 1_000_000:.1f} MB > 15 MB) and compression failed: {err}", False
            send_mime, compressed = "video/mp4", True
            sent_size = send_path.stat().st_size
            if sent_size <= OMNI_MAX_BYTES:
                break
            target_mb *= 0.8 * OMNI_MAX_BYTES / sent_size
        else:
            sent_mb = send_path.stat().st_size / 1_000_000
            return None, f"still too large for Omni OCR after compression ({sent_mb:.1f} MB > 15 MB) — split into more/smaller parts", True

    try:
        return await omni_video.transcribe_video(send_path.read_bytes(), mime=send_mime), None, compressed
    except omni_video.OmniError as err:
        return None, str(err), compressed


@router.post("/video-splitter/split")
async def split_video(
    file: UploadFile = File(...),
    mode: SplitMode = Form(...),
    value: str = Form(...),
    transcribe: bool = Form(True),
):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    stem = Path(file.filename or "video").stem
    mime = MIME_BY_SUFFIX.get(suffix.lower(), "video/mp4")
    do_transcribe = transcribe and omni_video.is_configured()

    with tempfile.TemporaryDirectory(prefix="video-splitter-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / f"input{suffix}"
        with input_path.open("wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)

        task_start = time.perf_counter()
        duration = await _probe_duration(input_path)

        out_paths: list[Path] = []
        if mode is SplitMode.SIZE:
            try:
                target_mb = float(value)
            except ValueError:
                raise HTTPException(status_code=400, detail="size must be a number")
            if target_mb <= 0:
                raise HTTPException(status_code=400, detail="size must be > 0")
            cuts = await _split_by_size(input_path, duration, target_mb * 1_000_000, tmp_dir, stem, suffix)
            segments = [(s, e) for _, s, e in cuts]
            out_paths = [p for p, _, _ in cuts]
        else:
            segments = _segments(mode, value, duration)
            for i, (start, end) in enumerate(segments, start=1):
                out_path = tmp_dir / f"{stem}_part{i:02d}{suffix}"
                await _cut_segment(input_path, start, end, out_path)
                out_paths.append(out_path)

        split_latency_ms = round((time.perf_counter() - task_start) * 1000)

        omni_start = time.perf_counter()
        if do_transcribe:
            results = await asyncio.gather(*(
                _transcribe(p, mime, end - start, tmp_dir) for p, (start, end) in zip(out_paths, segments)
            ))
        else:
            results = [(None, None, False)] * len(out_paths)
        omni_latency_ms = round((time.perf_counter() - omni_start) * 1000)

        # Transcript .txt files ride along in the zip next to each part, named to match it, so
        # the download is self-contained — not just a dependency on reading the JSON response.
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for out_path, (text, error, _compressed) in zip(out_paths, results):
                zf.write(out_path, arcname=out_path.name)
                if do_transcribe:
                    content = text if text else (f"[Omni OCR error: {error}]" if error else "[empty transcript]")
                    zf.writestr(out_path.stem + ".txt", content)

        segs_info = []
        for i, ((start, end), out_path, (text, error, compressed)) in enumerate(zip(segments, out_paths, results), start=1):
            segs_info.append({
                "index": i, "start": round(start, 3), "end": round(end, 3),
                "filename": out_path.name, "size_bytes": out_path.stat().st_size,
                "text": text, "error": error, "omni_compressed": compressed,
            })

        return {
            "duration": round(duration, 3),
            "split_latency_ms": split_latency_ms,
            "omni_latency_ms": omni_latency_ms,
            "latency_ms": split_latency_ms + omni_latency_ms,
            "omni_configured": omni_video.is_configured(),
            "transcribed": do_transcribe,
            "segments": segs_info,
            "zip_filename": f"{stem}_split.zip",
            "zip_base64": base64.b64encode(zip_buf.getvalue()).decode("ascii"),
        }
