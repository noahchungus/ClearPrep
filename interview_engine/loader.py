"""Loads the hand-written content files that ship inside the package (YAML).

Content lives in `interview_engine/data/` and is deliberately separate from code:
editing a question, cue phrase or exemplar never requires touching Python.
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"


@functools.lru_cache(maxsize=None)
def load_yaml(relative: str):
    with open(DATA_DIR / relative, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_question_files() -> list[dict]:
    out: list[dict] = []
    for path in sorted((DATA_DIR / "questions").glob("*.yaml")):
        out.extend(load_yaml(f"questions/{path.name}") or [])
    return out
