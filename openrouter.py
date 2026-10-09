"""Shared OpenRouter Decisions client: request headers (incl. OpenRouter response caching) and an
in-process TTL cache so identical requests are answered without calling OpenRouter again."""
import hashlib
import json
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

CACHE_ENABLED = os.getenv("OPENROUTER_CACHE", "true").lower() != "false"  # OpenRouter-side headers
CACHE_TTL = os.getenv("OPENROUTER_CACHE_TTL", "3600")  # seconds
APP_CACHE_ENABLED = os.getenv("APP_CACHE", "true").lower() != "false"  # in-process cache
APP_CACHE_MAX = int(os.getenv("APP_CACHE_MAX", "1000"))

_cache: dict[str, tuple[float, dict]] = {}  # key -> (expires_at, response body)


def headers(api_key: str) -> dict[str, str]:
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if CACHE_ENABLED:
        h["X-OpenRouter-Cache"] = "true"
        h["X-OpenRouter-Cache-TTL"] = CACHE_TTL
    return h


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _get(key: str) -> dict | None:
    hit = _cache.get(key)
    if not hit:
        return None
    if hit[0] < time.monotonic():
        del _cache[key]
        return None
    return hit[1]


def _put(key: str, body: dict) -> None:
    now = time.monotonic()
    for k in [k for k, (exp, _) in _cache.items() if exp < now]:  # drop expired
        del _cache[k]
    while len(_cache) >= APP_CACHE_MAX:  # then oldest-inserted
        del _cache[next(iter(_cache))]
    _cache[key] = (now + float(CACHE_TTL), body)


async def post_decision(url: str, api_key: str, model: str, state: dict, questions: dict) -> tuple[dict, bool]:
    """POST to the Decisions API; returns (response body, served_from_cache). Raises HTTPException."""
    payload = {"model": model, "state": state, "questions": questions}
    key = _key(payload)
    if APP_CACHE_ENABLED and (body := _get(key)) is not None:
        return body, True
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.post(url, headers=headers(api_key), json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenRouter {r.status_code}: {r.text}")
    body = r.json()
    if APP_CACHE_ENABLED and isinstance(body.get("answers"), dict):  # never cache bad responses
        _put(key, body)
    return body, False
