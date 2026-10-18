"""Qwen Omni video-to-text, ported from ai-platform's app/rag/omni.py (video kind) and
app/libraries/video_processing.py (the video PROMPT). Used to transcribe each clip the video
splitter produces. Transport: Qwen Cloud (DashScope) OpenAI-compatible endpoint, same as
ai-platform's `qwen` app. Omni models only answer streaming requests and are asked for text
output only, so the reply is accumulated from `delta.content`."""
import base64
import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger("video_splitter.omni")

OMNI_BASE_URL = os.environ.get("OMNI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
OMNI_API_KEY_ENV = "QWEN_API_KEY"
OMNI_MODEL = os.environ.get("OMNI_MODEL", "qwen3.8-omni-flash")
_MIN_PLAUSIBLE_KEY_LENGTH = 20

PROMPT = (
    "You convert a video clip into a complete text record. Another AI will use only your record "
    "to answer any future question about this clip, and it will never see or hear it. Anything "
    "you leave out is lost.\n"
    "\n"
    "Rules:\n"
    "- Do not answer questions, give opinions, or summarise. Only record what is seen and heard.\n"
    "- Timestamps are relative to the start of this clip, in MM:SS.\n"
    "- Transcribe speech word for word in its original language. Do not clean it up or "
    "paraphrase.\n"
    "- Copy on-screen text exactly, inside quotes.\n"
    "- If something is unclear, write [unclear]. Never guess.\n"
    "- Do not identify real people by name. Label them Person 1, Person 2, and so on, and keep "
    "the same labels throughout.\n"
    "- Write descriptions in English.\n"
    "\n"
    "Output these sections, in this order, and skip any that do not apply:\n"
    "\n"
    "TIMELINE\n"
    "One entry every 10 to 30 seconds, or at every scene change, in the form [MM:SS–MM:SS] "
    "followed by what is on screen: setting, people, actions, objects, camera movement.\n"
    "\n"
    "SPEECH\n"
    "[MM:SS] Speaker label: exact words. One line per turn.\n"
    "\n"
    "ON-SCREEN TEXT\n"
    "[MM:SS] \"exact text\", and where it appears (caption, title, sign, screen).\n"
    "\n"
    "SOUND AND MUSIC\n"
    "[MM:SS] What is heard: music (genre, instruments, mood, vocals), sound effects, background "
    "noise.\n"
    "\n"
    "PEOPLE\n"
    "Each person label with appearance and clothing, so they can be recognised across the clip."
)


class OmniError(Exception):
    pass


class OmniNotConfigured(OmniError):
    pass


_client: AsyncOpenAI | None = None


def is_configured() -> bool:
    key = os.environ.get(OMNI_API_KEY_ENV)
    return bool(key) and len(key.strip()) >= _MIN_PLAUSIBLE_KEY_LENGTH


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get(OMNI_API_KEY_ENV)
        if not is_configured():
            raise OmniNotConfigured(f"{OMNI_API_KEY_ENV} is not configured")
        _client = AsyncOpenAI(api_key=api_key, base_url=OMNI_BASE_URL)
    return _client


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def transcribe_video(data: bytes, *, mime: str = "video/mp4") -> str:
    """OCR / transcribe one video clip with Qwen Omni. Raises OmniError on failure."""
    client = _get_client()
    content = [
        {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{_b64(data)}"}},
        {"type": "text", "text": PROMPT},
    ]
    parts: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=OMNI_MODEL,
            messages=[{"role": "user", "content": content}],
            modalities=["text"],
            stream=True,
            temperature=0,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
    except OmniError:
        raise
    except Exception as err:
        logger.exception("omni call failed (model=%s)", OMNI_MODEL)
        raise OmniError(f"{type(err).__name__}: {err}") from err
    return "".join(parts).strip()
