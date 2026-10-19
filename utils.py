"""Shared video-compression helpers for shrinking a clip down to what Qwen Omni will accept/
process quickly. Originally part of video_splitter.py; pulled out here because video_processing.py
and image_processing.py depend on them too and the splitter itself has been removed."""
import asyncio
import os
from pathlib import Path

import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

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


async def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_EXE, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, stderr.decode("utf-8", "replace")


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
    clip's own known duration (start/end from the split step), so this never shells out to
    ffprobe or re-decodes the file to count frames — imageio-ffmpeg doesn't bundle ffprobe, and
    the duration is already known for free."""
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
