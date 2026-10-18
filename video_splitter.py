"""POST /video-splitter/split: splits an uploaded video into parts using ffmpeg stream copy
(-c copy). No re-encoding happens, so it's fast and lossless, but cuts snap to the nearest
preceding keyframe rather than the exact requested time. The ffmpeg binary comes from
imageio-ffmpeg, so no system install is required."""
import asyncio
import json
import re
import tempfile
import zipfile
from enum import Enum
from io import BytesIO
from pathlib import Path

import imageio_ffmpeg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

router = APIRouter()

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class SplitMode(str, Enum):
    PARTS = "parts"  # value = number of equal-length parts (>= 2)
    DURATION = "duration"  # value = seconds per part (> 0)
    TIMESTAMPS = "timestamps"  # value = comma-separated cut points in seconds


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


@router.post("/video-splitter/split")
async def split_video(
    file: UploadFile = File(...),
    mode: SplitMode = Form(...),
    value: str = Form(...),
):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    stem = Path(file.filename or "video").stem

    with tempfile.TemporaryDirectory(prefix="video-splitter-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / f"input{suffix}"
        with input_path.open("wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)

        duration = await _probe_duration(input_path)
        segments = _segments(mode, value, duration)

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (start, end) in enumerate(segments, start=1):
                out_path = tmp_dir / f"{stem}_part{i:02d}{suffix}"
                args = [
                    "-y", "-ss", f"{start:.3f}", "-i", str(input_path),
                    "-t", f"{end - start:.3f}", "-map", "0", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", str(out_path),
                ]
                code, stderr = await _run_ffmpeg(args)
                if code != 0 or not out_path.exists():
                    raise HTTPException(status_code=500, detail=f"ffmpeg failed on part {i}: {stderr[-800:]}")
                zf.write(out_path, arcname=out_path.name)

        info = json.dumps({"duration": round(duration, 3), "segments": [[round(s, 3), round(e, 3)] for s, e in segments]})
        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}_split.zip"',
                "X-Split-Info": info,
            },
        )
