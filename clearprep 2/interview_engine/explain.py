"""Builds the "How scoring works" page content from the analyzer itself, so the docs cannot drift."""
from __future__ import annotations

from .analyzer import DIMENSIONS, IDEAL_WORDS, IDEAL_WPM, PRESENT_THRESHOLD, PROFILES, STAR_WEIGHTS, WEIGHTS, Context, analyze
from .loader import load_yaml

PROFILE_LABELS = {
    "star": "Behavioral (STAR) answers", "situational": "Situational (“what would you do”)", "technical": "Technical explanations",
    "why_org": "“Why this organization / role?”", "intro": "“Tell me about yourself” / background", "ask_us": "Your questions for the interviewer",
}
# Checks that only appear when triggered (they subtract points and have no fixed maximum).
DEDUCTIONS = {"star": ["Vague wording"], "situational": ["Vague wording"], "technical": ["Vague wording"], "why_org": ["Generic phrasing", "Vague wording"],
              "intro": ["Vague wording"], "ask_us": ["Too early for pay/perks", "Answerable from their website"]}


def scoring_overview() -> dict:
    """Run the analyzer on a placeholder answer with every optional input present to list all checks."""
    ctx_kwargs = dict(question_text="Tell me about a time you led a team", competency="leadership", competency_cues=["lead"], concepts=[["x"], ["y"], ["z"]],
                      org="Org", role="Role", jd_skills={"Skill": []}, about_keywords=["mission"], duration_seconds=60.0, spoken=True)
    profiles = []
    for prof in PROFILES:
        res = analyze("placeholder " * 60, Context(profile=prof, **ctx_kwargs))
        dims = []
        for d in DIMENSIONS:
            checks = [{"label": c["label"], "possible": c["possible"]} for c in res["dimensions"][d]["checks"] if c["possible"] > 0]
            dims.append({"name": d, "checks": checks})
        lo, hi = IDEAL_WORDS[prof]
        profiles.append({"key": prof, "label": PROFILE_LABELS[prof], "dims": dims, "ideal": (lo, hi), "deductions": DEDUCTIONS.get(prof, [])})
    lex, cues = load_yaml("lexicons.yaml"), load_yaml("star_cues.yaml")
    return {
        "weights": {d: round(WEIGHTS[d] * 100) for d in DIMENSIONS}, "star_weights": STAR_WEIGHTS, "present_threshold": PRESENT_THRESHOLD, "ideal_wpm": IDEAL_WPM,
        "profiles": profiles,
        "lexicons": {"Filler words": lex["filler_words"] + lex["filler_phrases"], "Hedging phrases": lex["hedges"], "Weak-ownership phrases": lex["weak_phrases"],
                     "Vague wording": lex["vague_phrases"], "Generic “why us” phrases": lex["generic_org_phrases"], "Postpone until later": lex["premature_topics"]},
        "cues": {k: {"zone": cues[k]["zone"], "patterns": cues[k]["patterns"]} for k in ("situation", "task", "action", "result")},
    }
