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
            "Can the default LLM fulfill the user's entire request, "
            "or does the request require an additional specialized "
            "model or tool?"
        ),
        "criteria": {
            "default_model": (
                "Choose when the default LLM can fulfill the request "
                "adequately. Selecting file_processing, image_analysis, "
                "audio_processing, or video_processing alone does not "
                "require specific_model_or_tool. These dedicated "
                "processing requirements are handled independently. "
                "Choose default_model when no additional specialized "
                "model or tool is required."
            ),
            "specific_model_or_tool": (
                "Choose when the request requires an additional "
                "specialized model or tool beyond any applicable "
                "dedicated file or media processing. Examples include "
                "web search, image generation, video generation, "
                "code execution, advanced coding, or complex reasoning "
                "that the default LLM cannot adequately handle. "
                "Evaluate the actual requirements, not merely the "
                "topic, file type, or task length."
            ),
        },
    },

    "conversation_history": {
        "type": "choice",
        "instructions": (
            "Does the request require information from earlier "
            "messages in this conversation?"
        ),
        "criteria": {
            "true": (
                "Choose only when answering requires earlier user "
                "messages, assistant responses, decisions, or "
                "discussion details. Examples include 'What did we "
                "discuss earlier?' and 'Continue the plan we made.'"
            ),
            "false": (
                "Choose when the request can be answered without "
                "retrieving earlier conversation messages. A reference "
                "to a previous file alone does not automatically "
                "require conversation history if the file can be "
                "identified through previous_files_context."
            ),
        },
    },

    "memory_context": {
        "type": "choice",
        "instructions": (
            "Does the request require information from persistent "
            "saved user memory?"
        ),
        "criteria": {
            "true": (
                "Choose only when saved preferences, personal settings, "
                "or previously stored user details are needed. Examples "
                "include 'Use my saved preferences' or 'What pricing "
                "model did I previously decide on?' when that decision "
                "is stored in memory."
            ),
            "false": (
                "Choose when persistent user memory is not needed. "
                "Do not select this merely because the request refers "
                "to earlier conversation messages or uploaded files."
            ),
        },
    },

    "previous_files_context": {
        "type": "choice",
        "instructions": (
            "Does the request require information from a file uploaded "
            "or processed before the current message?"
        ),
        "available_previous_files": (
            "Dynamically supplied file metadata from the database "
            "for the current user. Use the available metadata to "
            "identify files relevant to the request."
        ),
        "criteria": {
            "true": (
                "Choose when the request requires a previously uploaded "
                "or processed file. Match the request against available "
                "file metadata, including file ID, filename, file type, "
                "description, extracted content summary, and processing "
                "status. For example, 'Summarize the contract I uploaded "
                "yesterday' requires this context when the matching "
                "contract is available."
            ),
            "false": (
                "Choose when no previously uploaded or processed file "
                "is required. Do not select this merely because a file "
                "exists in the database. Newly attached files belong "
                "to current-message input processing."
            ),
        },
    },

    "file_processing": {
        "type": "choice",
        "instructions": (
            "Does the request require processing a newly provided "
            "non-image, non-audio, non-video file?"
        ),
        "criteria": {
            "true": (
                "Choose when the request requires reading, extracting, "
                "analyzing, summarizing, comparing, or transforming a "
                "newly provided PDF, document, spreadsheet, CSV, text "
                "file, ZIP archive, or another supported non-media file. "
                "Use the request and available file metadata to identify "
                "the required operation. The presence of a file alone "
                "is insufficient."
            ),
            "false": (
                "Choose when no newly provided non-media file needs "
                "processing, or when the request can be fulfilled "
                "without processing the file. Previously processed "
                "content available in stored context does not require "
                "processing the original file again unless the request "
                "requires a new operation on that file."
            ),
        },
    },

    "image_analysis": {
        "type": "choice",
        "instructions": (
            "Does the request require analyzing a newly provided image?"
        ),
        "criteria": {
            "true": (
                "Choose when the request requires understanding, "
                "describing, interpreting, comparing, or extracting "
                "information from a newly provided image."
            ),
            "false": (
                "Choose when no newly provided image requires analysis, "
                "or the image is only an input to an image generation "
                "or editing operation that does not require separate "
                "analysis."
            ),
        },
    },

    "audio_processing": {
        "type": "choice",
        "instructions": (
            "Does the request require processing newly provided audio?"
        ),
        "criteria": {
            "true": (
                "Choose when the request requires transcribing, "
                "analyzing, summarizing, or extracting information "
                "from newly provided audio."
            ),
            "false": (
                "Choose when no newly provided audio requires processing, "
                "or the request is to generate or modify audio rather "
                "than process existing audio."
            ),
        },
    },

    "video_processing": {
        "type": "choice",
        "instructions": (
            "Does the request require processing a newly provided video?"
        ),
        "criteria": {
            "true": (
                "Choose when the request requires analyzing, summarizing, "
                "transcribing, or extracting information from a newly "
                "provided video. Select this for the video as a whole, "
                "including its visual content and relevant audio."
            ),
            "false": (
                "Choose when no newly provided video requires processing, "
                "or the request is to generate or modify video rather "
                "than process existing video."
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
