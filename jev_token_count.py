"""POST /token-count: counts the tokens a prompt would cost, via tiktoken — purely local,
no API call, no model response. `model` picks which encoding tiktoken uses
(encoding_for_model), falling back to cl100k_base for anything tiktoken doesn't recognize
(e.g. a newer/third-party model id) rather than failing the request."""
import tiktoken
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DEFAULT_ENCODING = "cl100k_base"

# tiktoken doesn't know "gpt-6-luna" (ai-platform's memory-pipeline model, via OpenRouter —
# see app/rag/extraction.py's MEMORY_EXTRACTOR_MODEL) or the bare "gpt-6-luna" id
# app/providers/registry.py's `gpt` entry uses, so encoding_for_model() would silently fall
# back to the older cl100k_base and undercount. tiktoken DOES resolve "gpt-5.6-luna" to
# o200k_base already, via its own internal "gpt-5*" prefix rule (tiktoken.model.
# MODEL_PREFIX_TO_ENCODING) — gpt-6-luna is that same lineage's successor, so o200k_base is
# the right encoding here too; pin it explicitly rather than waiting on a tiktoken release
# that recognizes "gpt-6*".
MODEL_ENCODING_OVERRIDES = {
    "gpt-6-luna": "o200k_base",
    "openai/gpt-6-luna": "o200k_base",
}

# A handful of common model ids for the dropdown — tiktoken.encoding_for_model also accepts
# any other model name it recognizes; this list is just what the UI offers, not a limit.
MODEL_CHOICES = [
    "gpt-4o", "gpt-4.1", "gpt-4", "gpt-3.5-turbo",
    "gpt-6-luna",
    "text-embedding-3-small", "text-embedding-3-large",
]

_encoders: dict[str, "tiktoken.Encoding"] = {}


def _encoding_for(model: str) -> "tiktoken.Encoding":
    if model not in _encoders:
        override = MODEL_ENCODING_OVERRIDES.get(model)
        if override is not None:
            _encoders[model] = tiktoken.get_encoding(override)
        else:
            try:
                _encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                if DEFAULT_ENCODING not in _encoders:
                    _encoders[DEFAULT_ENCODING] = tiktoken.get_encoding(DEFAULT_ENCODING)
                _encoders[model] = _encoders[DEFAULT_ENCODING]
    return _encoders[model]


class TokenCountRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"


class TokenCountResponse(BaseModel):
    tokens: int
    model: str
    encoding: str


@router.post("/token-count", response_model=TokenCountResponse)
async def token_count(req: TokenCountRequest) -> TokenCountResponse:
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    enc = _encoding_for(req.model)
    return TokenCountResponse(tokens=len(enc.encode(req.prompt)), model=req.model, encoding=enc.name)
