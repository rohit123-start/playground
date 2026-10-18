"""Qwen Omni video/audio/page/image-to-text, ported from ai-platform's app/rag/omni.py (all four
kinds) and app/libraries/{video,audio,file}_processing.py + image_analysis.py (the PROMPTs). Used
to transcribe each clip the video splitter and video processing pipeline produce, and each
rendered page / embedded image file_processing.py sends to Omni. Omni models only answer
streaming requests and are asked for text output only, so the reply is accumulated from
`delta.content`.

Two interchangeable transports, picked by OMNI_PROVIDER (env var, default "openrouter"):
  - "openrouter": OPEN_ROUTER_API_KEY, already used everywhere else in this project, routes to
    the same model (qwen/qwen3.8-omni-flash) pay-as-you-go — no "free quota" wall to hit.
  - "dashscope": Qwen Cloud direct (QWEN_API_KEY), same as ai-platform's `qwen` app.
Both speak the OpenAI-compatible chat.completions schema and accept the same video_url content
block, confirmed against OpenRouter's own docs — but their input_audio.data field differs:
DashScope accepts a data: URI prefix, OpenRouter requires raw base64 with NO prefix. That's the
one place this module branches on provider."""
import asyncio
import base64
import logging
import os
import time

from openai import AsyncOpenAI

logger = logging.getLogger("video_splitter.omni")

_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "model_env": "OMNI_OPENROUTER_MODEL",
        "default_model": "qwen/qwen3.8-omni-flash",
        "audio_data_prefix": "",  # raw base64, no data: URI
    },
    "dashscope": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
        "model_env": "OMNI_MODEL",
        "default_model": "qwen3.8-omni-flash",
        "audio_data_prefix": "data:;base64,",
    },
}

OMNI_PROVIDER = os.environ.get("OMNI_PROVIDER", "openrouter").strip().lower()
_cfg = _PROVIDERS.get(OMNI_PROVIDER, _PROVIDERS["openrouter"])

OMNI_BASE_URL = os.environ.get("OMNI_BASE_URL", _cfg["base_url"])
OMNI_API_KEY_ENV = _cfg["api_key_env"]
OMNI_MODEL = os.environ.get(_cfg["model_env"], _cfg["default_model"])
_AUDIO_DATA_PREFIX = _cfg["audio_data_prefix"]
_MIN_PLAUSIBLE_KEY_LENGTH = 20
# Retries for a failed Omni call on that SAME chunk/page/image — content-moderation false
# positives in particular have already shown non-deterministic behavior across otherwise
# identical calls in testing, so a second attempt genuinely can succeed where the first didn't.
# Never applies to OmniNotConfigured (raised by _get_client, before any retry loop even starts)
# — a missing/invalid key won't fix itself between attempts.
OMNI_MAX_RETRIES = 2
OMNI_RETRY_DELAY_SECONDS = 1.0

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

AUDIO_PROMPT = (
    "You convert an audio file into a complete text record. Another AI will use only your record "
    "to answer any future question about this audio, and it will never hear it. Anything you "
    "leave out is lost.\n"
    "\n"
    "Rules:\n"
    "- Do not answer questions, give opinions, or summarise. Only record what is heard.\n"
    "- Timestamps are relative to the start of this file, in MM:SS.\n"
    "- Transcribe speech word for word in its original language, including repetitions and "
    "filler words. Do not clean it up or paraphrase.\n"
    "- If a word is unclear, write [unclear]. Never guess.\n"
    "- Label speakers Speaker 1, Speaker 2, and so on, and keep the same labels throughout. Do "
    "not guess names unless a speaker says their own name.\n"
    "- Write descriptions in English.\n"
    "\n"
    "Output these sections, in this order, and skip any that do not apply:\n"
    "\n"
    "OVERVIEW\n"
    "What kind of audio this is (conversation, voice note, song, podcast, lecture, sound clip), "
    "the number of speakers, the languages spoken, and the total length.\n"
    "\n"
    "SPEECH\n"
    "[MM:SS] Speaker label: exact words. One line per turn.\n"
    "\n"
    "SPEAKERS\n"
    "Each speaker label with voice description: approximate age range, tone, accent if clear, "
    "and speaking style.\n"
    "\n"
    "MUSIC\n"
    "[MM:SS–MM:SS] Genre, instruments, tempo (slow, medium or fast), key mood, and whether "
    "there are vocals. Transcribe sung lyrics only if they are clearly audible.\n"
    "\n"
    "OTHER SOUNDS\n"
    "[MM:SS] Background noise, sound effects, environment (street, room, car)."
)

PAGE_PROMPT = (
    "You convert one page of a document into a complete text record. Another AI will use only "
    "your record to answer any future question about this page, and it will never see it. "
    "Anything you leave out is lost.\n"
    "\n"
    "Rules:\n"
    "- Do not answer questions, give opinions, or summarise. Only record what is on the page.\n"
    "- Copy all text exactly as written, in normal reading order. Keep the original language "
    "and spelling. Do not put body text in quotes; just reproduce it.\n"
    "- Keep the structure: headings as headings, lists as lists, paragraphs as paragraphs.\n"
    "- If something is unreadable, write [unreadable]. Never guess.\n"
    "- Write descriptions in English.\n"
    "\n"
    "Output these sections, in this order, and skip any that do not apply:\n"
    "\n"
    "TEXT\n"
    "The full text of the page in reading order, with its structure kept. Include headers, "
    "footers, page numbers, captions and footnotes.\n"
    "\n"
    "TABLES\n"
    "Every table rebuilt as a markdown table, with all rows, columns and values exactly as "
    "shown.\n"
    "\n"
    "CHARTS\n"
    "For each chart: type, title, axes, legend, and every data value or label shown.\n"
    "\n"
    "IMAGES AND DIAGRAMS\n"
    "For each photo, diagram or illustration: what it shows, any text inside it (in quotes), "
    "and where it sits on the page.\n"
    "\n"
    "OTHER MARKS\n"
    "Stamps, signatures (record only that one is present), handwriting (transcribed), "
    "highlights, logos."
)

IMAGE_PROMPT = (
    "You convert an image into a complete text record. Another AI will use only your record to "
    "answer any future question about this image, and it will never see the image. Anything you "
    "leave out is lost.\n"
    "\n"
    "Rules:\n"
    "- Do not answer questions, give opinions, or summarise. Only record what is there.\n"
    "- Copy every visible word, number, price, label and symbol exactly as written, inside "
    "quotes. Keep the original language and spelling, including mistakes.\n"
    "- If something is unreadable, write [unreadable]. Never guess.\n"
    "- Do not identify real people by name. Describe them instead.\n"
    "- Write descriptions in English.\n"
    "\n"
    "Output these sections, in this order, and skip any that do not apply:\n"
    "\n"
    "TEXT\n"
    "Every piece of visible text in reading order, each in quotes, with where it appears.\n"
    "\n"
    "OBJECTS\n"
    "Everything visible: what it is, how many, colour, size, and position (for example \"top "
    "left\", \"second shelf from the bottom\").\n"
    "\n"
    "PEOPLE\n"
    "How many, approximate age range, clothing, expression, pose, and what each is doing.\n"
    "\n"
    "CHARTS AND TABLES\n"
    "Chart type, title, axes, legend, and every value. Tables row by row, as a markdown table.\n"
    "\n"
    "SCENE\n"
    "Setting, indoor or outdoor, lighting, time of day if visible, style (photo, screenshot, "
    "drawing, document), and overall layout.\n"
    "\n"
    "OTHER DETAILS\n"
    "Anything notable not covered above: logos, damage, watermarks, unusual items."
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


async def _call_omni_once(client: AsyncOpenAI, content: list[dict]) -> tuple[str, dict | None]:
    """One attempt, no retry — raises OmniError on failure. Split out of _call_omni so the retry
    loop there can call this fresh each time without re-running _get_client()."""
    parts: list[str] = []
    usage = None
    try:
        stream = await client.chat.completions.create(
            model=OMNI_MODEL,
            messages=[{"role": "user", "content": content}],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
            temperature=0,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
    except OmniError:
        raise
    except Exception as err:
        logger.exception("omni call failed (model=%s)", OMNI_MODEL)
        raise OmniError(f"{type(err).__name__}: {err}") from err
    return "".join(parts).strip(), usage


async def _call_omni(content: list[dict]) -> tuple[str, dict | None, int]:
    """Returns (text, usage, latency_ms). usage is {"prompt_tokens", "completion_tokens",
    "total_tokens"} when the provider reports it — both OpenRouter and DashScope are
    OpenAI-compatible and honor stream_options.include_usage, which puts a populated `usage` on
    the final (otherwise-empty) stream chunk — else None. latency_ms times the WHOLE call
    including any retries, so a per-chunk caller's reported latency reflects what it actually
    waited, not just the final successful attempt.

    Retries up to OMNI_MAX_RETRIES times on failure, for this SAME chunk/page/image — see
    OMNI_MAX_RETRIES' own comment for why that's worth doing here. OmniNotConfigured is raised by
    _get_client() below, outside this loop, so a missing/invalid key is never retried."""
    client = _get_client()
    start = time.perf_counter()
    last_err: OmniError | None = None
    for attempt in range(OMNI_MAX_RETRIES + 1):
        try:
            text, usage = await _call_omni_once(client, content)
            latency_ms = round((time.perf_counter() - start) * 1000)
            return text, usage, latency_ms
        except OmniError as err:
            last_err = err
            if attempt < OMNI_MAX_RETRIES:
                logger.warning(
                    "omni call failed (attempt %d/%d), retrying: %s",
                    attempt + 1, OMNI_MAX_RETRIES + 1, err,
                )
                await asyncio.sleep(OMNI_RETRY_DELAY_SECONDS)
    raise last_err


async def transcribe_video(data: bytes, *, mime: str = "video/mp4") -> tuple[str, dict | None, int]:
    """OCR / transcribe one video clip with Qwen Omni. Returns (text, usage, latency_ms) —
    see _call_omni. Raises OmniError on failure."""
    content = [
        {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{_b64(data)}"}},
        {"type": "text", "text": PROMPT},
    ]
    return await _call_omni(content)


async def transcribe_audio(data: bytes, *, audio_format: str = "mp3") -> tuple[str, dict | None, int]:
    """Transcribe one audio clip with Qwen Omni. Returns (text, usage, latency_ms) — see
    _call_omni. Raises OmniError on failure."""
    content = [
        {"type": "input_audio", "input_audio": {"data": f"{_AUDIO_DATA_PREFIX}{_b64(data)}", "format": audio_format}},
        {"type": "text", "text": AUDIO_PROMPT},
    ]
    return await _call_omni(content)


async def transcribe_page(data: bytes, *, mime: str = "image/png") -> tuple[str, dict | None, int]:
    """OCR one rendered document page (e.g. a scanned PDF page) with Qwen Omni. Returns
    (text, usage, latency_ms) — see _call_omni. Raises OmniError on failure."""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(data)}"}},
        {"type": "text", "text": PAGE_PROMPT},
    ]
    return await _call_omni(content)


async def analyze_image(data: bytes, *, mime: str = "image/png") -> tuple[str, dict | None, int]:
    """OCR + describe one standalone image (or an embedded image pulled out of a document)
    with Qwen Omni. Returns (text, usage, latency_ms) — see _call_omni. Raises OmniError on
    failure."""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(data)}"}},
        {"type": "text", "text": IMAGE_PROMPT},
    ]
    return await _call_omni(content)
