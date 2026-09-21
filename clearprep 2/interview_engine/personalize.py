"""Turn the pasted job description / about-company text into usable interview signals.

Company-specific content comes ONLY from text the user pasted. Nothing here looks anything up
or asserts a fact about an employer.
"""
from __future__ import annotations

import functools
import math
import re
from collections import Counter

from .lexicons import get_lexicons, phrase_matcher
from .loader import load_yaml
from .text import STOPWORDS, content_tokens, stem, words

REQUIREMENT_HEADS = re.compile(r"(requirements?|qualifications?|what you('|’)ll need|must have|you have|skills|preferred|nice to have|minimum)", re.IGNORECASE)
SOFT_TO_COMPETENCY = {
    "Communication": "communication", "Collaboration": "teamwork", "Problem solving": "problem_solving",
    "Attention to detail": "problem_solving", "Time management": "deadlines", "Customer service": "communication",
    "Leadership": "leadership", "Adaptability": "adaptability", "Documentation": "communication",
}


@functools.lru_cache(maxsize=1)
def _skill_table() -> list[dict]:
    return [dict(s, aliases=[str(a) for a in (s.get("aliases") or [])], name=str(s["name"])) for s in load_yaml("skills.yaml")["skills"]]


@functools.lru_cache(maxsize=1)
def _skill_res() -> list:
    return [(s, phrase_matcher([s["name"]] + s["aliases"])) for s in _skill_table()]


def extract_skills(jd_text: str, family: str | None = None, limit: int = 12) -> list[dict]:
    """Find skills from the built-in lexicon in the pasted job description.

    Ranking: mention count, plus a boost when the mention sits under a requirements-style heading,
    plus a small boost for skills that belong to the detected role family.
    """
    if not jd_text or not jd_text.strip():
        return []
    # locate spans belonging to "requirements"-type sections (heading line to the next blank line)
    req_spans, pos = [], 0
    for block in re.split(r"\n\s*\n", jd_text):
        i = jd_text.find(block, pos)
        first = block.strip().split("\n")[0] if block.strip() else ""
        if REQUIREMENT_HEADS.search(first) and len(first) < 80:
            req_spans.append((i, i + len(block)))
        pos = i + len(block)
    hits = []
    for skill, rx in _skill_res():
        found = list(rx.finditer(jd_text))
        if not found:
            continue
        in_req = sum(1 for m in found if any(a <= m.start() < b for a, b in req_spans))
        score = len(found) + 0.5 * in_req + (0.5 if family and family in skill["families"] else 0)
        hits.append({"name": skill["name"], "kind": skill["kind"], "families": skill["families"], "aliases": skill["aliases"], "count": len(found), "in_requirements": in_req > 0, "score": score})
    hits.sort(key=lambda h: (-h["score"], h["name"]))
    return hits[:limit]


@functools.lru_cache(maxsize=1)
def _background_df() -> tuple[Counter, int]:
    """Document frequencies over the hand-written content, used as the IDF background corpus."""
    from .bank import load_bank

    docs = [" ".join([q.text] + q.strong_answer) for q in load_bank()]
    docs += [str(v.get("answer", "")) for v in load_yaml("exemplars.yaml").values() if isinstance(v, dict)]
    df: Counter = Counter()
    for d in docs:
        df.update({stem(t) for t in content_tokens(d)})
    return df, len(docs)


def extract_keywords(text: str, top_n: int = 10, exclude: set[str] | None = None) -> list[str]:
    """TF-IDF keyword extraction (terms frequent in `text` but rare in the background corpus)."""
    if not text or not text.strip():
        return []
    L = get_lexicons()
    boiler = {stem(w) for w in L.raw["jd_boilerplate"]}
    excl = {stem(e) for e in (exclude or set())}
    df, n_docs = _background_df()
    toks = [t for t in content_tokens(text) if len(t) > 3]
    tf = Counter(stem(t) for t in toks)
    surface: dict[str, Counter] = {}
    for t in toks:
        surface.setdefault(stem(t), Counter())[t] += 1
    scored = []
    for s, c in tf.items():
        if s in boiler or s in excl:
            continue
        idf = math.log((n_docs + 1) / (df.get(s, 0) + 1)) + 1
        scored.append((c * idf, s))
    scored.sort(reverse=True)
    return [surface[s].most_common(1)[0][0] for _, s in scored[:top_n]]


def extract_about_keywords(about_text: str, org: str = "", top_n: int = 14) -> list[str]:
    """Distinctive words from the user's pasted about-company text (values words first)."""
    if not about_text or not about_text.strip():
        return []
    L = get_lexicons()
    org_stems = {stem(t) for t in content_tokens(org)}
    values = {stem(v) for v in L.raw["values_words"]}
    seen: dict[str, str] = {}
    for w in words(about_text):
        t = w.text.lower()
        if stem(t) in values and stem(t) not in seen and stem(t) not in org_stems:
            seen[stem(t)] = t
    for k in extract_keywords(about_text, top_n=top_n, exclude=org_stems):
        seen.setdefault(stem(k), k)
    return list(seen.values())[:top_n]


def extract_duties(jd_text: str, limit: int = 6) -> list[str]:
    """Bullet lines from the posting, quoted back verbatim (they are the user's own pasted text)."""
    out = []
    for line in (jd_text or "").splitlines():
        m = re.match(r"^\s*(?:[-•*·▪●◦]|\d+[.)])\s*(.+)$", line)
        if m and 20 <= len(m.group(1).strip()) <= 180:
            out.append(m.group(1).strip())
    return out[:limit]


def skill_questions(skills: list[dict], limit: int = 3) -> list[dict]:
    """Template-generated, skill-specific questions built from what the JD actually mentions."""
    out = []
    for hit in skills:
        if len(out) >= limit:
            break
        name, kind = hit["name"], hit["kind"]
        base = {"generated": True, "skill": name, "families": ["general"], "difficulty": 2}
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if kind == "tool":
            out.append({**base, "id": f"skill-{slug}-1", "type": "behavioral", "profile": "star", "competency": "problem_solving",
                        "text": f"This role mentions {name}. Tell me about a time you used it to solve a real problem.",
                        "strong_answer": [f"Names the specific {name} features or tasks you used.", "Describes the problem and your steps.", "Gives a concrete result."]})
        elif kind == "practice":
            out.append({**base, "id": f"skill-{slug}-1", "type": "behavioral", "profile": "star", "competency": "problem_solving",
                        "text": f"Describe a time you applied {name} in a project, class or job.",
                        "strong_answer": [f"Explains what {name} meant in that context.", "Shows your specific actions.", "Gives an outcome."]})
        else:
            comp = SOFT_TO_COMPETENCY.get(name, "problem_solving")
            out.append({**base, "id": f"skill-{slug}-1", "type": "behavioral", "profile": "star", "competency": comp,
                        "text": f"The posting emphasizes {name.lower()}. Give me an example of when you showed it.",
                        "strong_answer": [f"Uses a concrete example of {name.lower()}.", "Shows your own actions and the result."]})
    return out


def questions_to_ask(org: str, role: str, skills: list[dict], about_keywords: list[str], duties: list[str]) -> list[dict]:
    """Questions for the candidate to ask the interviewer. Templates + only the user's pasted facts."""
    cfg = load_yaml("ask_us.yaml")
    out = [{"text": q["text"].replace("{org}", org or "your organization").replace("{role}", role or "this role"), "why": q["why"]} for q in cfg["base"]]
    if skills:
        names = [s["name"] for s in skills if s["kind"] != "soft"][:3]
        if names:
            out.insert(0, {"text": f"Which of {', '.join(names)} would someone in this role use most in the first few weeks, and what would help me get up to speed?", "why": "Grounded in your pasted job description; shows you read it."})
    for kw in about_keywords[:2]:
        if kw in {v for v in get_lexicons().raw["values_words"]}:
            out.insert(1, {"text": f"In the material I read about {org or 'the organization'}, {kw} came up a lot. How does that show up in the {role or 'team'}'s day-to-day work?", "why": f"Uses “{kw}” from the about-text you pasted. Only ask if that text is accurate and current."})
            break
    for duty in duties[:1]:
        out.append({"text": f"The posting mentions “{duty[:110]}”. What does a good first project in that area look like?", "why": "Quotes a line from the posting you pasted."})
    return out
