"""POST /video-splitter/split: splits an uploaded video into parts using ffmpeg stream copy
(-c copy). No re-encoding happens, so it's fast and lossless, but cuts snap to the nearest
preceding keyframe rather than the exact requested time. The ffmpeg binary comes from
imageio-ffmpeg, so no system install is required. Each part is then optionally transcribed with
Qwen Omni (see omni_video.py) so the playground can show what's actually in each clip."""
import asyncio
import base64
import bisect
import math
import os
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
KEYFRAME_RE = re.compile(r"pts_time:(-?\d+\.?\d*)")
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
# Not just a smaller upload: fewer frames for Omni to analyze is normally the bigger latency win
# (resolution alone mostly just shrinks the file). 2 fps / 720p are reasonable starting defaults,
# not universal ones — fast motion needs more frames, small on-screen text may need more
# resolution — but this is an automated pipeline with no way to make that per-video judgment call.
OMNI_COMPRESS_FPS = 2
OMNI_COMPRESS_MAX_WIDTH = 1280
OMNI_COMPRESS_MAX_HEIGHT = 720


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


async def _keyframe_times(path: Path) -> list[float]:
    """Sorted keyframe (I-frame) timestamps, always including 0.0 (every stream starts on one).
    -skip_frame nokey makes ffmpeg skip decoding every non-key frame, so this is one fast pass
    over the file's keyframe headers, not a full decode."""
    args = ["-skip_frame", "nokey", "-i", str(path), "-vf", "showinfo", "-f", "null", os.devnull]
    _, stderr = await _run_ffmpeg(args)
    times = {0.0}
    for m in KEYFRAME_RE.finditer(stderr):
        t = float(m.group(1))
        if t >= 0:
            times.add(t)
    return sorted(times)


def _snap_down(t: float, keyframes: list[float]) -> float:
    """The real start time a stream-copy -ss seek to t produces: the keyframe at or before t —
    matching ffmpeg's own seek behavior, so our bookkeeping matches reality exactly instead of
    drifting from whatever ffmpeg actually cut."""
    idx = bisect.bisect_right(keyframes, t + 1e-6) - 1
    return keyframes[max(idx, 0)]


def _segments(mode: SplitMode, value: str, duration: float, keyframes: list[float]) -> list[tuple[float, float]]:
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

    # Snap every internal boundary to the keyframe stream-copy will actually cut at (0 and
    # duration are always valid: every stream starts on a keyframe, and duration is the file's
    # true end). Boundaries that collapse onto the same keyframe are merged — the real part
    # count can come out lower than requested if cut points fall within one keyframe interval;
    # that's an honest reflection of what stream copy can actually produce, not a bug.
    snapped = [0.0]
    for b in bounds[1:-1]:
        s = _snap_down(b, keyframes)
        if s > snapped[-1] + 1e-3:
            snapped.append(s)
    snapped.append(duration)
    return [(snapped[i], snapped[i + 1]) for i in range(len(snapped) - 1)]


async def _cut_segment(input_path: Path, start: float, end: float, out_path: Path) -> None:
    # Keyframe timestamps are often repeating decimals (e.g. 128 frames / 30fps = 8.533333...).
    # Formatting at millisecond precision (.3f) used to ROUND THAT DOWN below the true keyframe
    # (8.533333 -> "8.533", which is LESS than the keyframe) — ffmpeg's "-ss" then seeks to the
    # PREVIOUS keyframe a whole GOP earlier, silently duplicating that GOP's content into this
    # segment. Microsecond precision (.6f) plus a small upward nudge keeps -ss unambiguously at
    # or after the intended keyframe, never below it.
    args = [
        "-y", "-ss", f"{start + 1e-4:.6f}", "-i", str(input_path),
        "-t", f"{end - start:.6f}", "-map", "0", "-c", "copy",
        "-avoid_negative_ts", "make_zero", str(out_path),
    ]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not out_path.exists():
        raise HTTPException(status_code=500, detail=f"ffmpeg failed cutting {out_path.name}: {stderr[-800:]}")


async def _split_by_size(
    input_path: Path, duration: float, target_bytes: float, tmp_dir: Path, stem: str, suffix: str,
    keyframes: list[float],
) -> list[tuple[Path, float, float]]:
    """Cuts into ceil(total_size / target_bytes) parts — the minimum that can satisfy the target
    — dividing the REMAINING duration evenly across the REMAINING part count at each step, rather
    than estimating each part's length independently from the last part's bitrate (the latter
    accumulates drift over several parts into a wasted undersized leftover part).

    Every candidate boundary is chosen directly from `keyframes`, so `start` is always exactly
    where stream copy will really cut (never a snapped-and-drifted approximation of it). That
    matters beyond precision: an unaligned boundary makes the next part silently start earlier
    than labeled (duplicating content with the part before it), which both inflates total output
    size and corrupts the byte-rate estimate this function uses to plan ahead. With boundaries
    pinned to real keyframes there is no overlap, so the estimate stays honest.

    A short slice also carries a disproportionate I-frame cost relative to the file-wide average
    (every slice pays for its own leading keyframe, which the average smooths out over many of
    them) — so even an exact fair-share estimate tends to overshoot slightly and need a shrink.
    FAIR_SHARE_MARGIN pre-emptively undershoots each estimate to absorb that, so the common case
    needs no shrink at all. When a shrink is needed, it steps to the PREVIOUS keyframe rather
    than an arbitrary smaller duration — again, so every attempt is a real, non-overlapping cut."""
    FAIR_SHARE_MARGIN = 0.98
    remaining_bytes_est = input_path.stat().st_size
    out: list[tuple[Path, float, float]] = []
    start, start_idx, i = 0.0, 0, 0
    while start < duration - 1e-6:
        i += 1
        remaining_duration = duration - start
        parts_left = max(1, math.ceil(remaining_bytes_est / target_bytes)) if target_bytes > 0 else 1
        target_time = start + FAIR_SHARE_MARGIN * remaining_duration / parts_left

        # The keyframe at/before target_time, but always at least one keyframe past `start` so
        # we make forward progress even when the estimate lands short of the next keyframe.
        idx = bisect.bisect_right(keyframes, min(target_time, duration) + 1e-6) - 1
        idx = max(idx, start_idx + 1)
        if idx < len(keyframes) and keyframes[idx] < duration - 1e-6:
            end_idx, end = idx, keyframes[idx]
        else:
            end_idx, end = len(keyframes), duration  # no keyframe left before the file's true end

        out_path = tmp_dir / f"{stem}_part{i:02d}{suffix}"
        for _ in range(5):  # shrink to the previous keyframe if this slice overshoots target
            await _cut_segment(input_path, start, end, out_path)
            actual = out_path.stat().st_size
            # Accept once it fits, or once there's no earlier keyframe left to shrink to — NOT
            # just because the estimate happened to reach the file's end (a stale estimate can
            # land there even when there's plenty of oversized content left to split further).
            if actual <= target_bytes or end >= duration - 1e-3 or end_idx <= start_idx + 1:
                break
            end_idx -= 1
            end = keyframes[end_idx]

        # Once it fits, greedily extend to the next keyframe(s) while it still fits. Rounding
        # down to a keyframe means the fair-share estimate tends to land one keyframe short of
        # what the target would actually allow — without this, a well-behaved file would
        # systematically need one extra part beyond the true minimum.
        for _ in range(20):
            if actual > target_bytes or end >= duration - 1e-6:
                break
            if end_idx + 1 < len(keyframes) and keyframes[end_idx + 1] < duration - 1e-6:
                cand_idx, cand_end = end_idx + 1, keyframes[end_idx + 1]
            else:
                cand_idx, cand_end = len(keyframes), duration
            await _cut_segment(input_path, start, cand_end, out_path)
            cand_actual = out_path.stat().st_size
            if cand_actual > target_bytes:
                await _cut_segment(input_path, start, end, out_path)  # revert to the last good cut
                break
            end_idx, end, actual = cand_idx, cand_end, cand_actual

        if end > start:
            local_rate = actual / (end - start)
            remaining_bytes_est = max(local_rate * (duration - end), 0.0)
        out.append((out_path, start, end))
        start, start_idx = end, end_idx
    return out


async def _reduce_for_omni(path: Path, tmp_dir: Path) -> Path:
    """Caps fps and resolution with a single fast CRF pass (no bitrate target to chase) —
    applied to EVERY clip sent to Omni, not just ones over OMNI_MAX_BYTES. The size threshold
    only guards what Omni's API will *accept*; it says nothing about how long Omni *takes* to
    look at a clip, which scales with frame count more than byte count (see OMNI_COMPRESS_FPS
    above). A small clip under the size cap was still being sent at full fps/resolution and
    taking just as long as an oversized one — this is what actually fixes that."""
    vf = (
        f"fps={OMNI_COMPRESS_FPS},"
        f"scale=w='min({OMNI_COMPRESS_MAX_WIDTH},iw)':h='min({OMNI_COMPRESS_MAX_HEIGHT},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    out_path = tmp_dir / f"{path.stem}_omniready.mp4"
    args = [
        "-y", "-i", str(path), "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart", str(out_path),
    ]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not out_path.exists():
        raise RuntimeError(stderr[-800:])
    return out_path


async def _compress_for_omni(path: Path, duration: float, tmp_dir: Path, target_mb: float) -> Path:
    """Re-encodes path for Omni: caps fps and resolution, not just bitrate (see OMNI_COMPRESS_FPS
    above), and uses two-pass H.264 so the result lands close to target_mb on the first try —
    cutting how often the caller's retry loop needs to re-run the whole encode. Sized from the
    clip's own known duration (start/end from the split step), so unlike the reference this was
    ported from, it never shells out to ffprobe or re-decodes the file to count frames —
    imageio-ffmpeg doesn't bundle ffprobe, and we already know the duration for free."""
    audio_bitrate = 64_000
    target_total_bitrate = int(target_mb * 1_000_000 * 8 / duration * 0.94)  # 6% muxing/container margin
    video_bitrate = max(target_total_bitrate - audio_bitrate, 100_000)

    vf = (
        f"fps={OMNI_COMPRESS_FPS},"
        f"scale=w='min({OMNI_COMPRESS_MAX_WIDTH},iw)':h='min({OMNI_COMPRESS_MAX_HEIGHT},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    out_path = tmp_dir / f"{path.stem}_compressed.mp4"
    passlog = str(tmp_dir / f"{path.stem}_passlog")

    pass1 = [
        "-y", "-i", str(path), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-b:v", str(video_bitrate),
        "-pass", "1", "-passlogfile", passlog, "-an", "-f", "null", os.devnull,
    ]
    code, stderr = await _run_ffmpeg(pass1)
    if code != 0:
        raise RuntimeError(stderr[-800:])

    pass2 = [
        "-y", "-i", str(path), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-b:v", str(video_bitrate),
        "-pass", "2", "-passlogfile", passlog,
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart", str(out_path),
    ]
    code, stderr = await _run_ffmpeg(pass2)
    if code != 0 or not out_path.exists():
        raise RuntimeError(stderr[-800:])
    return out_path


async def _transcribe(
    path: Path, mime: str, duration: float, tmp_dir: Path,
) -> tuple[str | None, str | None, bool, dict | None, int | None]:
    """Returns (text, error, compressed, usage, omni_latency_ms) — exactly one of text/error is
    set. Never raises: a transcription failure on one clip shouldn't fail the whole split (same
    fail-open rule ai-platform's ingest uses). The reduced/compressed copy only affects what's
    sent to Omni — the zip keeps the original lossless stream-copied part. usage/latency are
    None on error (compression failures and timing before a dropped stream connection aren't
    worth threading through)."""
    original_mb = path.stat().st_size / 1_000_000
    try:
        send_path, send_mime, compressed = await _reduce_for_omni(path, tmp_dir), "video/mp4", True
    except Exception:
        send_path, send_mime, compressed = path, mime, False  # fall back to the original on failure

    size = send_path.stat().st_size
    if size > OMNI_MAX_BYTES:
        target_mb = OMNI_COMPRESS_TARGET_MB
        for _ in range(3):  # retry at a lower target if a hard-to-compress clip overshoots
            try:
                send_path = await _compress_for_omni(send_path, duration, tmp_dir, target_mb)
            except Exception as err:
                return None, f"clip too large for Omni OCR ({original_mb:.1f} MB > 15 MB) and compression failed: {err}", False, None, None
            send_mime, compressed = "video/mp4", True
            sent_size = send_path.stat().st_size
            if sent_size <= OMNI_MAX_BYTES:
                break
            target_mb *= 0.8 * OMNI_MAX_BYTES / sent_size
        else:
            sent_mb = send_path.stat().st_size / 1_000_000
            return None, f"still too large for Omni OCR after compression ({sent_mb:.1f} MB > 15 MB) — split into more/smaller parts", True, None, None

    try:
        text, usage, latency_ms = await omni_video.transcribe_video(send_path.read_bytes(), mime=send_mime)
        return text, None, compressed, usage, latency_ms
    except omni_video.OmniError as err:
        return None, str(err), compressed, None, None


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
        keyframes = await _keyframe_times(input_path)

        out_paths: list[Path] = []
        if mode is SplitMode.SIZE:
            try:
                target_mb = float(value)
            except ValueError:
                raise HTTPException(status_code=400, detail="size must be a number")
            if target_mb <= 0:
                raise HTTPException(status_code=400, detail="size must be > 0")
            cuts = await _split_by_size(input_path, duration, target_mb * 1_000_000, tmp_dir, stem, suffix, keyframes)
            segments = [(s, e) for _, s, e in cuts]
            out_paths = [p for p, _, _ in cuts]
        else:
            segments = _segments(mode, value, duration, keyframes)
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
            results = [(None, None, False, None, None)] * len(out_paths)
        omni_latency_ms = round((time.perf_counter() - omni_start) * 1000)

        # Transcript .txt files ride along in the zip next to each part, named to match it, so
        # the download is self-contained — not just a dependency on reading the JSON response.
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for out_path, (text, error, _compressed, _usage, _chunk_latency_ms) in zip(out_paths, results):
                zf.write(out_path, arcname=out_path.name)
                if do_transcribe:
                    content = text if text else (f"[Omni OCR error: {error}]" if error else "[empty transcript]")
                    zf.writestr(out_path.stem + ".txt", content)

        segs_info = []
        for i, ((start, end), out_path, (text, error, compressed, usage, chunk_latency_ms)) in enumerate(zip(segments, out_paths, results), start=1):
            segs_info.append({
                "index": i, "start": round(start, 3), "end": round(end, 3),
                "filename": out_path.name, "size_bytes": out_path.stat().st_size,
                "text": text, "error": error, "omni_compressed": compressed,
                "omni_latency_ms": chunk_latency_ms, "omni_tokens": usage,
            })

        tokens_total = sum(s["omni_tokens"]["total_tokens"] for s in segs_info if s["omni_tokens"])

        return {
            "duration": round(duration, 3),
            "split_latency_ms": split_latency_ms,
            "omni_latency_ms": omni_latency_ms,
            "latency_ms": split_latency_ms + omni_latency_ms,
            "omni_configured": omni_video.is_configured(),
            "omni_key_env": omni_video.OMNI_API_KEY_ENV,
            "omni_tokens_total": tokens_total,
            "transcribed": do_transcribe,
            "segments": segs_info,
            "zip_filename": f"{stem}_split.zip",
            "zip_base64": base64.b64encode(zip_buf.getvalue()).decode("ascii"),
        }
