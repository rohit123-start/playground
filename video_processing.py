"""POST /video-processing/process: a preprocessing pipeline for long video/audio, distinct from
video_splitter.py's lossless stream-copy splitter. Where that one keeps parts lossless for
download, this one is built for feeding Qwen Omni: validates a duration cap, scales video down to
720p and re-encodes while splitting, splits audio-only input with stream copy when possible, runs
every chunk through Omni, then restores a single original timeline by shifting each chunk's
[MM:SS] markers by its own real start time.

Latency: the chunk re-encode keeps original fps (720p-capped) because it's also what the ZIP
hands back to the user to watch — dropping fps there would make the download a slideshow. Before
a video chunk is actually sent to Omni, a SEPARATE 2fps/720p copy is made just for that call via
video_splitter's _reduce_for_omni — the zipped chunk a person downloads is never fps-reduced.

Video chunks are cut ONE AT A TIME (not via ffmpeg's segment muxer, which only hands back
completed chunks after processing the ENTIRE file) specifically so cutting and Omni can overlap:
chunk 1 starts its Omni call as soon as it's cut, while chunk 2 is still being cut, instead of
waiting for every chunk to exist before any Omni call begins. Re-encoding (not stream copy) means
each cut is frame-accurate regardless of keyframes, so this needs none of video_splitter.py's
keyframe-snapping machinery. Cutting itself is capped at CUT_CONCURRENCY concurrent ffmpeg
processes — unlike Omni calls (I/O-bound, waiting on the network), ffmpeg re-encodes are CPU-bound,
so running too many at once would oversubscribe the machine rather than speed anything up.
Audio-only input keeps the simpler segment-muxer+stream-copy batch approach: it's already fast
(no re-encode), so there's no meaningful split phase to overlap with anything."""
import asyncio
import base64
import math
import re
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path

import imageio_ffmpeg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import omni_video
from video_splitter import OMNI_COMPRESS_TARGET_MB, OMNI_MAX_BYTES
from video_splitter import _compress_for_omni as _compress_video_for_omni
from video_splitter import _reduce_for_omni as _reduce_video_for_omni

router = APIRouter()

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
VIDEO_STREAM_RE = re.compile(r"Video:.*?(\d{2,5})x(\d{2,5})")
AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?Audio:")
TIMESTAMP_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[–-](\d{1,3}):(\d{2}))?\]")

MAX_DURATION_SECONDS = 30 * 60  # reject anything longer before spending time processing it
DEFAULT_SEGMENT_SECONDS = 180
CUT_CONCURRENCY = 2  # concurrent ffmpeg re-encodes — CPU-bound, unlike the Omni calls alongside them
# OMNI_MAX_BYTES/OMNI_COMPRESS_TARGET_MB and the video compressor are imported from
# video_splitter.py rather than re-derived — it already found DashScope's real "max bytes per
# data-uri item" limit empirically (see its own comment on OMNI_MAX_BYTES) and a proven two-pass,
# fps/resolution-capped compressor to fit under it. A 720p/crf28 chunk from this pipeline's own
# split step rarely hits that limit, but a long or high-motion segment still can.


async def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_EXE, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, stderr.decode("utf-8", "replace")


async def _probe_media(path: Path) -> dict:
    """duration/has_video/has_audio/width/height via ffmpeg's own -i banner — no ffprobe,
    imageio-ffmpeg doesn't bundle it."""
    _, stderr = await _run_ffmpeg(["-i", str(path)])
    m = DURATION_RE.search(stderr)
    if not m:
        raise HTTPException(status_code=400, detail="Could not read media info (unsupported or corrupt file)")
    hours, minutes, seconds = m.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    vid = VIDEO_STREAM_RE.search(stderr)
    return {
        "duration": duration,
        "has_video": vid is not None,
        "has_audio": AUDIO_STREAM_RE.search(stderr) is not None,
        "width": int(vid.group(1)) if vid else None,
        "height": int(vid.group(2)) if vid else None,
    }


def _parse_segment_list(list_path: Path, tmp_dir: Path) -> list[tuple[Path, float, float]]:
    chunks = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        name, start, end = line.rsplit(",", 2)
        chunks.append((tmp_dir / name, float(start), float(end)))
    return chunks


def _plan_video_segments(duration: float, segment_seconds: int) -> list[tuple[float, float]]:
    """Equal-ish segment_seconds-wide boundaries, last one taking the remainder. No keyframe
    snapping needed here (unlike video_splitter.py): these chunks are re-encoded, not
    stream-copied, so ffmpeg cuts exactly where asked regardless of the source's keyframes."""
    bounds = []
    t = 0.0
    while t < duration:
        bounds.append(t)
        t += segment_seconds
    bounds.append(duration)
    bounds = sorted(set(bounds))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


async def _cut_video_chunk(input_path: Path, start: float, end: float, out_path: Path) -> None:
    args = [
        "-y", "-ss", f"{start:.3f}", "-i", str(input_path), "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", "scale=-2:720",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        str(out_path),
    ]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not out_path.exists():
        raise HTTPException(status_code=500, detail=f"ffmpeg failed cutting {out_path.name}: {stderr[-800:]}")


async def _split_audio_chunks(
    input_path: Path, tmp_dir: Path, segment_seconds: int, suffix: str,
) -> list[tuple[Path, float, float]]:
    list_path = tmp_dir / "chunks.csv"
    args = [
        "-y", "-i", str(input_path),
        "-f", "segment", "-segment_time", str(segment_seconds), "-c", "copy",
        "-segment_list", str(list_path), "-segment_list_type", "csv",
        str(tmp_dir / f"chunk_%03d{suffix}"),
    ]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not list_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed splitting audio (stream copy may not support this format): {stderr[-800:]}",
        )
    return _parse_segment_list(list_path, tmp_dir)


def _fmt_mmss(total_seconds: float) -> str:
    total_seconds = max(0, round(total_seconds))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _shift_timestamps(text: str, offset: float) -> str:
    """Rewrites every [MM:SS] / [MM:SS–MM:SS] marker Omni produced (relative to its own
    clip) to be relative to the ORIGINAL file, by adding the chunk's real start time. Safe to
    regex here specifically because we control the prompt's output format (omni_video.PROMPT /
    AUDIO_PROMPT both mandate this exact bracket notation)."""
    def repl(m):
        mm1, ss1, mm2, ss2 = m.groups()
        t1 = _fmt_mmss(int(mm1) * 60 + int(ss1) + offset)
        if mm2 is not None:
            t2 = _fmt_mmss(int(mm2) * 60 + int(ss2) + offset)
            return f"[{t1}–{t2}]"
        return f"[{t1}]"
    return TIMESTAMP_RE.sub(repl, text)


async def _compress_audio_for_omni(path: Path, duration: float, tmp_dir: Path, target_mb: float) -> Path:
    """Re-encodes an oversized audio chunk down to roughly target_mb. Audio has no fps/resolution
    to cut, so this is just a bitrate-targeted re-encode (mirrors video_splitter's bitrate math,
    minus the video-specific knobs)."""
    bitrate = max(int(target_mb * 1_000_000 * 8 / duration * 0.94), 32_000)
    out_path = tmp_dir / f"{path.stem}_compressed.mp3"
    args = ["-y", "-i", str(path), "-c:a", "libmp3lame", "-b:a", str(bitrate), str(out_path)]
    code, stderr = await _run_ffmpeg(args)
    if code != 0 or not out_path.exists():
        raise RuntimeError(stderr[-800:])
    return out_path


async def _process_chunk(
    path: Path, start: float, end: float, kind: str, audio_format: str, tmp_dir: Path,
) -> tuple[str | None, str | None, dict | None, int | None]:
    """Returns (text, error, usage, omni_latency_ms) — exactly one of text/error is set. Never
    raises: one bad chunk shouldn't fail the whole file (same fail-open rule video_splitter.py's
    _transcribe uses). usage/latency are None on error.

    Video chunks are always run through _reduce_video_for_omni first — fewer frames for Omni to
    analyze, not just a smaller file (see this module's docstring) — before the size check, so a
    chunk over Qwen's real size limit (OMNI_MAX_BYTES) almost never needs the heavier two-pass
    compressor below; when it still does, that's retried at a lower target if a hard-to-compress
    chunk overshoots, same pattern video_splitter.py's _transcribe uses."""
    send_path, send_audio_format = path, audio_format
    original_mb = path.stat().st_size / 1_000_000

    if kind == "video":
        try:
            send_path = await _reduce_video_for_omni(send_path, tmp_dir)
        except Exception:
            pass  # fall back to the original (still resolution-capped by the split step)

    size = send_path.stat().st_size
    if size > OMNI_MAX_BYTES:
        duration = max(end - start, 0.5)
        target_mb = OMNI_COMPRESS_TARGET_MB
        compress = _compress_video_for_omni if kind == "video" else _compress_audio_for_omni
        for _ in range(3):  # retry at a lower target if a hard-to-compress chunk overshoots
            try:
                send_path = await compress(send_path, duration, tmp_dir, target_mb)
            except Exception as err:
                return None, f"chunk too large for Omni ({original_mb:.1f} MB > 15 MB) and compression failed: {err}", None, None
            send_audio_format = "mp3"
            sent_size = send_path.stat().st_size
            if sent_size <= OMNI_MAX_BYTES:
                break
            target_mb *= 0.8 * OMNI_MAX_BYTES / sent_size
        else:
            sent_mb = send_path.stat().st_size / 1_000_000
            return None, f"still too large for Omni after compression ({sent_mb:.1f} MB > 15 MB) — try a shorter segment length", None, None

    try:
        data = send_path.read_bytes()
        text, usage, latency_ms = await (
            omni_video.transcribe_video(data, mime="video/mp4") if kind == "video"
            else omni_video.transcribe_audio(data, audio_format=send_audio_format)
        )
        return _shift_timestamps(text, start), None, usage, latency_ms
    except omni_video.OmniError as err:
        return None, str(err), None, None


async def _cut_and_process_video_chunk(
    sem: asyncio.Semaphore, input_path: Path, i: int, start: float, end: float,
    tmp_dir: Path, do_process: bool,
) -> tuple[Path, float, float, str | None, str | None, dict | None, int | None, int]:
    """Cuts one video chunk (capped at CUT_CONCURRENCY concurrent ffmpeg processes), then
    immediately starts its Omni call — so when this runs alongside other chunks via
    asyncio.gather, chunk 1's Omni call overlaps with chunk 2 (or 3, 4...) still being cut,
    instead of every chunk waiting for the whole file to finish splitting first. Returns
    (out_path, start, end, text, error, usage, omni_latency_ms, cut_latency_ms)."""
    out_path = tmp_dir / f"chunk_{i:03d}.mp4"
    cut_start = time.perf_counter()
    async with sem:
        await _cut_video_chunk(input_path, start, end, out_path)
    cut_latency_ms = round((time.perf_counter() - cut_start) * 1000)

    if do_process:
        text, error, usage, omni_latency_ms = await _process_chunk(out_path, start, end, "video", "mp3", tmp_dir)
    else:
        text, error, usage, omni_latency_ms = None, f"Omni not configured (set {omni_video.OMNI_API_KEY_ENV})", None, None
    return out_path, start, end, text, error, usage, omni_latency_ms, cut_latency_ms


@router.post("/video-processing/process")
async def process_video(
    file: UploadFile = File(...),
    segment_seconds: int = Form(DEFAULT_SEGMENT_SECONDS),
):
    if segment_seconds <= 0:
        raise HTTPException(status_code=400, detail="segment_seconds must be > 0")

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    stem = Path(file.filename or "input").stem

    with tempfile.TemporaryDirectory(prefix="video-processing-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / f"input{suffix}"
        with input_path.open("wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)

        task_start = time.perf_counter()
        info = await _probe_media(input_path)
        if info["duration"] > MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Video is {info['duration'] / 60:.1f} min, over the {MAX_DURATION_SECONDS // 60}-minute limit",
            )
        probe_latency_ms = round((time.perf_counter() - task_start) * 1000)

        kind = "video" if info["has_video"] else "audio"
        do_process = omni_video.is_configured()
        cut_latencies: dict[int, int] = {}

        # process_latency_ms times this whole block (split + Omni together) and is the
        # authoritative total for "how long did the real work take" — split_latency_ms and
        # omni_latency_ms below are sub-measurements for display, not necessarily additive to
        # it: for video they overlap (see module docstring), so neither alone nor their sum is
        # wall clock; for audio they're still genuinely sequential, so they do add up to it.
        process_start = time.perf_counter()
        if kind == "video":
            segments = _plan_video_segments(info["duration"], segment_seconds)
            sem = asyncio.Semaphore(CUT_CONCURRENCY)
            raw = await asyncio.gather(*(
                _cut_and_process_video_chunk(sem, input_path, i, start, end, tmp_dir, do_process)
                for i, (start, end) in enumerate(segments, start=1)
            ))
            chunks = [(p, s, e) for p, s, e, *_ in raw]
            results = [(text, error, usage, omni_ms) for _p, _s, _e, text, error, usage, omni_ms, _cut_ms in raw]
            cut_latencies = {i: cut_ms for i, (*_r, cut_ms) in enumerate(raw, start=1)}
            split_latency_ms = sum(cut_latencies.values())  # total ffmpeg cut work, not wall clock
            omni_latency_ms = round((time.perf_counter() - process_start) * 1000)  # == process_latency_ms
        elif info["has_audio"]:
            split_start = time.perf_counter()
            chunks = await _split_audio_chunks(input_path, tmp_dir, segment_seconds, suffix)
            split_latency_ms = round((time.perf_counter() - split_start) * 1000)

            omni_start = time.perf_counter()
            audio_format = suffix.lstrip(".").lower() or "mp3"
            if do_process:
                results = await asyncio.gather(*(
                    _process_chunk(path, start, end, kind, audio_format, tmp_dir) for path, start, end in chunks
                ))
            else:
                results = [(None, f"Omni not configured (set {omni_video.OMNI_API_KEY_ENV})", None, None)] * len(chunks)
            omni_latency_ms = round((time.perf_counter() - omni_start) * 1000)
        else:
            raise HTTPException(status_code=400, detail="File has neither a video nor an audio stream")
        process_latency_ms = round((time.perf_counter() - process_start) * 1000)

        chunks_info = []
        merged_parts = []
        for i, ((path, start, end), (text, error, usage, chunk_latency_ms)) in enumerate(zip(chunks, results), start=1):
            chunks_info.append({
                "index": i, "start": round(start, 3), "end": round(end, 3),
                "filename": path.name, "size_bytes": path.stat().st_size,
                "text": text, "error": error,
                "omni_latency_ms": chunk_latency_ms, "omni_tokens": usage,
                "cut_latency_ms": cut_latencies.get(i),
            })
            merged_parts.append(f"=== Chunk {i} ({_fmt_mmss(start)}–{_fmt_mmss(end)}) ===\n{text or f'[{error}]'}")
        merged_text = "\n\n".join(merged_parts)

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, _start, _end in chunks:
                zf.write(path, arcname=path.name)
            zf.writestr(f"{stem}_merged_transcript.txt", merged_text)

        tokens_total = sum(c["omni_tokens"]["total_tokens"] for c in chunks_info if c["omni_tokens"])

        return {
            "duration": round(info["duration"], 3),
            "kind": kind,
            "width": info["width"],
            "height": info["height"],
            "probe_latency_ms": probe_latency_ms,
            "split_latency_ms": split_latency_ms,
            "omni_latency_ms": omni_latency_ms,
            "process_latency_ms": process_latency_ms,
            "latency_ms": probe_latency_ms + process_latency_ms,
            "omni_configured": do_process,
            "omni_key_env": omni_video.OMNI_API_KEY_ENV,
            "omni_tokens_total": tokens_total,
            "chunks": chunks_info,
            "merged_text": merged_text,
            "zip_filename": f"{stem}_processed.zip",
            "zip_base64": base64.b64encode(zip_buf.getvalue()).decode("ascii"),
        }
