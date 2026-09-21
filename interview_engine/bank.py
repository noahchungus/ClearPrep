"""The question bank: hand-written YAML entries loaded into simple dataclasses."""
from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, field

from .loader import load_question_files, load_yaml

QUESTION_TYPES = ("opener", "motivation", "behavioral", "situational", "technical")
PROFILE_FOR_TYPE = {"opener": "intro", "motivation": "star", "behavioral": "star", "situational": "situational", "technical": "technical"}


@dataclass
class Question:
    id: str
    text: str
    type: str
    competency: str
    families: list[str]
    difficulty: int
    strong_answer: list[str] = field(default_factory=list)
    concepts: list[list[str]] = field(default_factory=list)
    profile: str = ""
    rephrase: str = ""
    generated: bool = False
    skill: str = ""

    def __post_init__(self) -> None:
        if not self.profile:
            self.profile = PROFILE_FOR_TYPE.get(self.type, "star")

    def to_dict(self) -> dict:
        return asdict(self)


def competencies() -> dict:
    return load_yaml("competencies.yaml")


@functools.lru_cache(maxsize=1)
def load_bank() -> tuple[Question, ...]:
    out = []
    for raw in load_question_files():
        out.append(
            Question(
                id=raw["id"], text=raw["text"], type=raw["type"], competency=raw["competency"],
                families=list(raw["families"]), difficulty=int(raw["difficulty"]),
                strong_answer=list(raw.get("strong_answer", [])),
                concepts=[[str(a) for a in g] for g in raw.get("concepts", [])],
                profile=raw.get("profile", ""), rephrase=raw.get("rephrase", ""),
            )
        )
    return tuple(out)


def validate_bank(questions=None) -> list[str]:
    """Return a list of human-readable problems (empty list = the content files are consistent)."""
    from .roles import families

    questions = questions if questions is not None else load_bank()
    comps, fams = competencies(), families()
    problems, seen = [], set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.type not in QUESTION_TYPES:
            problems.append(f"{q.id}: unknown type {q.type!r}")
        if q.competency not in comps:
            problems.append(f"{q.id}: unknown competency {q.competency!r}")
        for fam in q.families:
            if fam not in fams:
                problems.append(f"{q.id}: unknown family {fam!r}")
        if q.difficulty not in (1, 2, 3):
            problems.append(f"{q.id}: difficulty must be 1-3")
        if len(q.strong_answer) < 2:
            problems.append(f"{q.id}: needs at least 2 strong_answer points")
        if q.type == "technical" and len(q.concepts) < 3:
            problems.append(f"{q.id}: technical questions need at least 3 concept groups")
        if q.profile not in ("star", "situational", "technical", "why_org", "intro"):
            problems.append(f"{q.id}: bad profile {q.profile!r}")
    return problems


def fill(text: str, org: str, role: str) -> str:
    return text.replace("{org}", org.strip() or "your organization").replace("{role}", role.strip() or "this role")
