"""POST /token-count: counts the tokens a prompt would cost, via tiktoken — purely local,
no API call, no model response. `model` picks which encoding tiktoken uses
(encoding_for_model), falling back to cl100k_base for anything tiktoken doesn't recognize
(e.g. a newer/third-party model id) rather than failing the request."""
import tiktoken
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DEFAULT_ENCODING = "cl100k_base"

# A handful of common model ids for the dropdown — tiktoken.encoding_for_model also accepts
# any other model name it recognizes; this list is just what the UI offers, not a limit.
MODEL_CHOICES = [
    "gpt-4o", "gpt-4.1", "gpt-4", "gpt-3.5-turbo",
    "text-embedding-3-small", "text-embedding-3-large",
]

_encoders: dict[str, "tiktoken.Encoding"] = {}


def _encoding_for(model: str) -> "tiktoken.Encoding":
    if model not in _encoders:
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
