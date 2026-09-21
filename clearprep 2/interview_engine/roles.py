"""Map a job title (and optionally a job description) to a role family using keyword rules."""
from __future__ import annotations

from dataclasses import dataclass

from .lexicons import phrase_matcher
from .loader import load_yaml


@dataclass(frozen=True)
class FamilyGuess:
    family: str
    label: str
    source: str  # "title", "job description", "default" or "user"
    scores: dict


def families() -> dict:
    return load_yaml("roles.yaml")["families"]


def family_label(family: str) -> str:
    return families().get(family, {}).get("label", family.title())


def infer_family(title: str, jd_text: str = "") -> FamilyGuess:
    fams = families()
    scores: dict[str, int] = {}
    last_pos: dict[str, int] = {}
    for fam, cfg in fams.items():
        kws = [str(k) for k in cfg.get("keywords", [])]
        if not kws:
            continue
        rx = phrase_matcher(kws)
        found = list(rx.finditer(title or ""))
        # a multi-word keyword is more specific than a single word, so it scores higher
        scores[fam] = sum(len(m.group().split()) for m in found)
        last_pos[fam] = max((m.start() for m in found), default=-1)
    # ties go to the family whose keyword appears last: in "Data Security Analyst" the later word is the domain
    best = max(scores.items(), key=lambda kv: (kv[1], last_pos[kv[0]]), default=("general", 0))
    if best[1] > 0:
        return FamilyGuess(best[0], family_label(best[0]), "title", scores)
    if jd_text.strip():
        from .personalize import extract_skills  # local import: personalize depends on this module

        counts: dict[str, int] = {}
        for hit in extract_skills(jd_text, family=None):
            if hit["kind"] == "soft":
                continue
            for fam in hit["families"]:
                if fam != "general":
                    counts[fam] = counts.get(fam, 0) + hit["count"]
        if counts:
            fam = max(counts.items(), key=lambda kv: kv[1])[0]
            return FamilyGuess(fam, family_label(fam), "job description", counts)
    return FamilyGuess("general", family_label("general"), "default", scores)


def seniority_warning(title: str) -> str | None:
    cfg = load_yaml("roles.yaml")
    if phrase_matcher([str(k) for k in cfg["internship_keywords"]]).search(title or ""):
        return None
    if phrase_matcher([str(k) for k in cfg["senior_keywords"]]).search(title or ""):
        return "This title looks more senior than an internship or entry-level role. The questions are aimed at students and new graduates, so difficulty may feel light."
    return None
