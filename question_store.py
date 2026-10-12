"""A questions JSON file that can be edited while the server runs.

get() re-reads the file whenever it changes on disk, so edits apply to the next request without a
restart. put() validates a new version and writes it atomically. Used by the playground editors.
"""
import json
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException

QUESTION_TYPES = ("choice", "noul", "score")


class QuestionStore:
    def __init__(self, path: Path, require_none: bool = False):
        self.path = path
        self.require_none = require_none  # model picker: every choice question needs a "none" option
        self._mtime = None
        self._data: dict = {}
        self.get()  # fail fast at startup if the file is missing or invalid

    def get(self) -> dict:
        mtime = self.path.stat().st_mtime_ns
        if mtime != self._mtime:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if self._mtime is None:
                    raise
                # Mid-edit or invalid JSON on disk: keep serving the last good version.
                print(f"[questions] keeping previous {self.path.name}, reload failed: {exc}")
                return self._data
            self._data, self._mtime = data, mtime
        return self._data

    def validate(self, new: object) -> dict:
        def bad(msg: str):
            raise HTTPException(status_code=422, detail=msg)

        if not isinstance(new, dict) or not new:
            bad("Questions must be a non-empty JSON object")
        for name, q in new.items():
            if not isinstance(q, dict):
                bad(f"{name}: must be an object")
            if q.get("type") not in QUESTION_TYPES:
                bad(f"{name}: type must be one of {', '.join(QUESTION_TYPES)}")
            if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
                bad(f"{name}: missing instructions")
            crit = q.get("criteria")
            if crit is not None:
                if not isinstance(crit, dict) or not crit:
                    bad(f"{name}: criteria must be a non-empty object")
                if not all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
                           for k, v in crit.items()):
                    bad(f"{name}: every criteria option needs a name and a description")
                if self.require_none and q["type"] == "choice" and "none" not in crit:
                    bad(f"{name}: criteria must include a 'none' option")
            elif q["type"] == "choice":
                bad(f"{name}: a choice question needs criteria")
        return new

    def put(self, new: object) -> dict:
        new = self.validate(new)
        newline = "\r\n" if b"\r\n" in self.path.read_bytes() else "\n"  # keep the file's line endings
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as f:
                f.write(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)  # atomic: a request never sees a half-written file
        except BaseException:
            os.unlink(tmp)
            raise
        return self.get()
