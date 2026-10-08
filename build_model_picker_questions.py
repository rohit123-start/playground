"""Convert model_tiers/{low,medium,high}.yaml into questions_jev_model_picker_<tier>.json.

Run after editing a YAML (needs PyYAML, which is only a build-time dependency):
    python build_model_picker_questions.py
"""
import json
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
TIERS = ("low", "medium", "high")

# One-line hints that disambiguate overlapping task categories (the YAML only has keywords).
TASK_HINTS = {
    "multimodal": "Understanding or analyzing an existing image, photo, video, audio or screenshot the user provides.",
    "long_docs": "Working over a long or uploaded document, transcript, contract or paper.",
    "extraction": "Classifying, extracting, tagging or labeling content, often in high volume.",
    "image_generation": "Creating a brand-new image from a description.",
    "image_editing": "Modifying an existing image.",
    "video_generation": "Creating a brand-new video from a description or image.",
    "video_editing": "Modifying an existing video clip.",
    "audio_generation": "Creating speech, narration, music or songs.",
    "audio_editing": "Modifying an existing song or audio track.",
    "research": "Needs fresh or web-sourced information with sources.",
}


def task_question(tasks: dict[str, list[str]], models: dict[str, dict]) -> dict:
    covered = {t for m in models.values() for t in m["handles"]}
    criteria = {}
    for task, keywords in tasks.items():
        if task not in covered:  # no model in this tier can do it
            continue
        text = "Request involves: " + ", ".join(keywords) + "."
        if task in TASK_HINTS:
            text += " " + TASK_HINTS[task]
        criteria[task] = text
    return {
        "type": "choice",
        "instructions": "Which single task category best matches what the user is asking for?",
        "criteria": criteria,
    }


def model_question(models: dict[str, dict]) -> dict:
    criteria = {}
    for name, m in models.items():
        feats = m["features"]
        criteria[name] = (
            f"Handles: {', '.join(m['handles'])}. Difficulty: {m['difficulty']}. "
            f"Thinking: {str(feats['thinking']).lower()}, deep thinking: {str(feats['deep_thinking']).lower()}. "
            f"Best for: {'; '.join(m['best_for'])}."
        )
    return {
        "type": "choice",
        "instructions": (
            "Which model is the best fit for this request, considering the task, how difficult "
            "it is, and whether it needs reasoning?"
        ),
        "criteria": criteria,
    }


def main() -> None:
    for tier in TIERS:
        data = yaml.safe_load((BASE_DIR / "model_tiers" / f"{tier}.yaml").read_text(encoding="utf-8"))
        models = data["models"]
        questions = {
            "task": task_question(data["tasks"], models),
            "model": model_question(models),
        }
        out = BASE_DIR / f"questions_jev_model_picker_{tier}.json"
        out.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{tier}: {len(models)} models, {len(questions['task']['criteria'])} tasks -> {out.name}")


if __name__ == "__main__":
    main()
