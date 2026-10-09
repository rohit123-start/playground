"""Convert model_tiers/{low,medium,high}.yaml into questions_jev_model_picker_<tier>.json.

Each YAML already holds the OpenRouter Decisions questions under `choices`; this just validates
them and writes them out as JSON. Run after editing a YAML (needs PyYAML, build-time only):
    python build_model_picker_questions.py
"""
import json
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
TIERS = ("low", "medium", "high")


def main() -> None:
    for tier in TIERS:
        data = yaml.safe_load((BASE_DIR / "model_tiers" / f"{tier}.yaml").read_text(encoding="utf-8"))
        questions = data["choices"]
        for name, q in questions.items():
            assert q["type"] == "choice", f"{tier}.{name}: type must be 'choice'"
            assert q["instructions"], f"{tier}.{name}: missing instructions"
            assert "none" in q["criteria"], f"{tier}.{name}: missing 'none' option"
            assert all(isinstance(v, str) and v for v in q["criteria"].values()), f"{tier}.{name}: empty criterion"
        out = BASE_DIR / f"questions_jev_model_picker_{tier}.json"
        out.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = ", ".join(f"{n}={len(q['criteria']) - 1}" for n, q in questions.items())
        print(f"{tier}: {len(questions)} questions ({summary}) -> {out.name}")


if __name__ == "__main__":
    main()
