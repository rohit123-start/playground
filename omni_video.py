"""Qwen Omni video/audio-to-text, ported from ai-platform's app/rag/omni.py (video + audio kinds)
and app/libraries/video_processing.py + audio_processing.py (the PROMPTs). Used to transcribe
each clip the video splitter and video processing pipeline produce. Omni models only answer
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
import base64
import logging
import os

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


async def _call_omni(content: list[dict]) -> str:
    client = _get_client()
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


async def transcribe_video(data: bytes, *, mime: str = "video/mp4") -> str:
    """OCR / transcribe one video clip with Qwen Omni. Raises OmniError on failure."""
    content = [
        {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{_b64(data)}"}},
        {"type": "text", "text": PROMPT},
    ]
    return await _call_omni(content)


async def transcribe_audio(data: bytes, *, audio_format: str = "mp3") -> str:
    """Transcribe one audio clip with Qwen Omni. Raises OmniError on failure."""
    content = [
        {"type": "input_audio", "input_audio": {"data": f"{_AUDIO_DATA_PREFIX}{_b64(data)}", "format": audio_format}},
        {"type": "text", "text": AUDIO_PROMPT},
    ]
    return await _call_omni(content)
