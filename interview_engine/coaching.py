"""Presentation helpers for feedback: what to fix first, fill-in-the-blank scaffolds, exemplars.

There is no text generation anywhere. Feedback is assembled from the analyzer's own checks and
from hand-written content in data/*.yaml.
"""
from __future__ import annotations

from .analyzer import STAR_LABELS
from .bank import competencies
from .loader import load_yaml
from .text import sentences


def lost_points(analysis: dict) -> list[dict]:
    """Every check that lost points, biggest loss first, with its dimension attached."""
    rows = []
    for dim, d in analysis["dimensions"].items():
        for c in d["checks"]:
            lost = c["possible"] - c["earned"]
            if lost > 0.4 and c["possible"] > 0:
                rows.append({"dimension": dim, "label": c["label"], "lost": round(lost, 1), "possible": c["possible"], "detail": c["detail"], "evidence": c["evidence"]})
            elif c["possible"] == 0 and c["earned"] < 0:
                rows.append({"dimension": dim, "label": c["label"], "lost": round(-c["earned"], 1), "possible": 0, "detail": c["detail"], "evidence": c["evidence"]})
    rows.sort(key=lambda r: -r["lost"])
    return rows


def top_fixes(analysis: dict, n: int = 3) -> list[dict]:
    return lost_points(analysis)[:n]


def strengths(analysis: dict, n: int = 3) -> list[dict]:
    rows = []
    for dim, d in analysis["dimensions"].items():
        for c in d["checks"]:
            if c["possible"] > 0 and c["earned"] >= 0.85 * c["possible"] and c["possible"] >= 15:
                rows.append({"dimension": dim, "label": c["label"], "detail": c["detail"], "weight": c["possible"]})
    rows.sort(key=lambda r: -r["weight"])
    return rows[:n]


def scaffold(question: dict, analysis: dict) -> list[dict]:
    """Fill-in-the-blank prompts for the STAR parts that are missing or weak. Asks for the user's own facts."""
    if analysis["profile"] != "star" or not analysis.get("star"):
        return []
    comp = competencies().get(question.get("competency"), {})
    sc = comp.get("scaffold", {})
    order = [("situation", "Situation"), ("task", "Task"), ("action", "Action"), ("result", "Result")]
    rows = []
    for key, label in order:
        status = analysis["star"][key]["status"]
        if status != "present":
            rows.append({"key": key, "label": label, "status": status, "prompt": sc.get(key, "")})
    return rows


def full_scaffold(question: dict) -> list[dict]:
    sc = competencies().get(question.get("competency"), {}).get("scaffold", {})
    return [{"key": k, "label": STAR_LABELS[k], "prompt": sc.get(k, "")} for k in ("situation", "task", "action", "result")]


def exemplar_for(question: dict) -> dict | None:
    """The hand-written exemplar for this question, split into sentences and tagged with the part each plays."""
    ex = load_yaml("exemplars.yaml")
    prof = question.get("profile")
    key = {"intro": "intro", "why_org": "why_org", "situational": "situational", "technical": "technical"}.get(prof) or question.get("competency")
    if key == "technical" and question.get("competency") not in ("technical", None) and prof == "technical":
        key = "technical"
    entry = ex.get(key) or ex.get("problem_solving")
    if not entry:
        return None
    text = " ".join(str(entry["answer"]).split())
    org, role = question.get("_org", ""), question.get("_role", "")
    text = text.replace("{role}", role or "this role").replace("{org}", org or "this organization")
    sents = [s.text for s in sentences(text)]
    parts = entry.get("parts", {}) or {}
    labels: dict[int, str] = {}
    names = {"S": "Situation", "T": "Task", "A": "Action", "R": "Result"}
    for k, v in parts.items():
        idxs = v if isinstance(v, list) else [v]
        for i in idxs:
            labels[int(i) - 1] = names.get(k, str(k).replace("_", " ").title())
    return {"scenario": entry.get("scenario", ""), "note": entry.get("note", ""), "sentences": [{"text": s, "label": labels.get(i, "")} for i, s in enumerate(sents)]}


def competency_info(competency: str) -> dict:
    c = competencies().get(competency, {})
    return {"label": c.get("label", competency), "looks_for": c.get("looks_for", ""), "rephrase": c.get("rephrase", ""), "probe": c.get("probe", "")}
