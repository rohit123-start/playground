"""POST /jev-prompt-scan-open-router: model routing plus history/memory/file/media checks with typesafe/jev
through the OpenRouter Decisions API. Independent questions, one call."""
import json
import os
import time

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import openrouter
from schemas import SelectRequest

load_dotenv()

router = APIRouter()

OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("JEV_OPEN_ROUTER_AI_MODEL")  # OpenRouter model id

# Option names are the returned values. The API has only choice/score/noul types, so the yes/no\n# questions are choices with "true"/"false" options.
questions = {
    "model_routing": {
        "type": "choice",
        "instructions": (
            "Does this request require a specialized model or tool, "
            "or can the default LLM handle it adequately?"
        ),
        "criteria": {
            "default_model": (
                "The default LLM can fulfill the entire request "
                "adequately, considering its complexity, reasoning, "
                "quality, and capability requirements."
            ),
            "specific_model_or_tool": (
                "The request requires a specialized model or tool "
                "because the default LLM cannot adequately fulfill "
                "the entire request. Consider task complexity, "
                "reasoning requirements, coding difficulty, quality "
                "requirements, and specialized capabilities. This "
                "includes complex reasoning, advanced coding, web "
                "research, image generation, video generation, and "
                "other specialized tasks. Evaluate the entire request, "
                "even when a dedicated file or media processing "
                "question also applies."
            ),
        },
    },
    "conversation_history": {
        "type": "choice",
        "instructions": (
            "Is previous conversation history required to fulfill "
            "the current request?"
        ),
        "criteria": {
            "true": (
                "The request requires information from previous messages, "
                "earlier decisions, previous conversation turns, or "
                "earlier assistant responses."
            ),
            "false": (
                "The request can be fulfilled without information "
                "from previous conversation history."
            ),
        },
    },
    "memory_context": {
        "type": "choice",
        "instructions": (
            "Is saved user memory required to fulfill the current request?"
        ),
        "criteria": {
            "true": (
                "The request requires relevant information from persistent "
                "user memory, such as saved preferences, decisions, "
                "or previously stored details."
            ),
            "false": (
                "The request can be fulfilled without retrieving "
                "saved user memory."
            ),
        },
    },
    "previous_files_context": {
        "type": "choice",
        "instructions": (
            "Are previously uploaded or processed files required "
            "to fulfill the current request?"
        ),
        "available_previous_files": (
            "Dynamically populated with available file metadata "
            "from the database for the current user."
        ),
        "criteria": {
            "true": (
                "The request requires information from one or more "
                "previously uploaded or processed files. Use the user "
                "request and available file metadata to identify "
                "potentially relevant files. Metadata may include file IDs, "
                "filenames, file types, descriptions, extracted content "
                "summaries, and processing status."
            ),
            "false": (
                "The request does not require information from previously "
                "uploaded or processed files. Files provided with the "
                "current message are handled through current-message "
                "input processing and semantic search."
            ),
        },
    },
    "file_processing": {
        "type": "choice",
        "instructions": (
            "Does the current request require processing a newly "
            "provided non-image, non-audio, non-video file?"
        ),
        "criteria": {
            "true": (
                "The request requires reading, extracting, analyzing, "
                "summarizing, comparing, or transforming a newly "
                "provided PDF, Word document, spreadsheet, text file, "
                "CSV, ZIP archive, or another supported non-media file. "
                "Determine this from the request and available file "
                "metadata. The mere presence of a file is insufficient."
            ),
            "false": (
                "No newly provided non-media file requires processing, "
                "or the request can be fulfilled without processing "
                "the newly provided file. Also choose false when the "
                "required content has already been processed and is "
                "available in stored context."
            ),
        },
    },
    "image_analysis": {
        "type": "choice",
        "instructions": (
            "Does the current request require analysis of a newly "
            "provided image?"
        ),
        "criteria": {
            "true": (
                "The request requires understanding, interpreting, "
                "describing, comparing, or extracting information "
                "from a newly provided image. Determine this from "
                "the request and available image metadata."
            ),
            "false": (
                "No newly provided image requires analysis, the "
                "required information has already been processed "
                "and stored, or the image is only an input to a "
                "generation or editing task that does not require "
                "separate analysis."
            ),
        },
    },
    "audio_processing": {
        "type": "choice",
        "instructions": (
            "Does the current request require processing newly "
            "provided audio?"
        ),
        "criteria": {
            "true": (
                "The request requires transcribing, analyzing, "
                "summarizing, extracting information from, or "
                "converting newly provided audio. Determine this "
                "from the request and available audio metadata."
            ),
            "false": (
                "No newly provided audio requires processing, the "
                "required information has already been processed "
                "and stored, or the request is to generate or modify "
                "audio rather than process existing audio."
            ),
        },
    },
    "video_processing": {
        "type": "choice",
        "instructions": (
            "Does the current request require processing a newly "
            "provided video?"
        ),
        "criteria": {
            "true": (
                "The request requires analyzing, summarizing, "
                "transcribing, extracting information from, or "
                "converting a newly provided video. Determine this "
                "from the request and available video metadata."
            ),
            "false": (
                "No newly provided video requires processing, the "
                "required information has already been processed "
                "and stored, or the request is to generate or modify "
                "video rather than process existing video."
            ),
        },
    },
}


class PromptScanRequest(SelectRequest):
    metadata: dict | None = None  # optional extra info (attachments, history, ...) given to JEV


class Decision(BaseModel):
    choice: str  # booleans come back as "true"/"false"
    confidence: float | None = None
    probabilities: dict[str, float]


class PromptScanResponse(BaseModel):
    decisions: dict[str, Decision]  # keyed by question name
    cached: bool  # served from the in-process cache, no OpenRouter call
    latency_ms: int


def _decision(name: str, answer) -> Decision:
    if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
        raise HTTPException(status_code=502, detail=f"Missing/invalid answer for {name}: {answer}")
    probs = {str(k).lower(): round(float(v), 4) for k, v in answer["probabilities"].items()}
    choice = answer.get("choice")
    choice = str(choice).lower() if choice is not None else max(probs, key=probs.get)
    conf = answer.get("confidence")
    return Decision(
        choice=choice,
        confidence=round(float(conf), 4) if isinstance(conf, (int, float)) else None,
        probabilities=probs,
    )


@router.post("/jev-prompt-scan-open-router", response_model=PromptScanResponse)
async def jev_prompt_scan_open_router(req: PromptScanRequest) -> PromptScanResponse:
    if not OPENROUTER_MODEL or not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Set JEV_OPEN_ROUTER_AI_MODEL and OPEN_ROUTER_API_KEY environment variables",
        )
    state = {"user_request": req.user_query}
    if req.metadata:
        state["metadata"] = req.metadata
    start = time.perf_counter()
    body, cached = await openrouter.post_decision(
        OPENROUTER_DECISIONS_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, state, questions
    )
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise HTTPException(status_code=502, detail=f"Unrecognised OpenRouter response: {json.dumps(body)}")
    return PromptScanResponse(
        decisions={name: _decision(name, answers.get(name)) for name in questions},
        cached=cached,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
