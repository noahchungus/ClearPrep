"""Deterministic, explainable answer analyzer.

No machine-learning model and no text generation: every number is produced by
named checks over lexicons and regular expressions (see data/lexicons.yaml and
data/star_cues.yaml). Each dimension score is

    100 * sum(points earned) / sum(points possible)

over a list of checks, and every check carries a plain-English reason plus the
exact spans of the user's own text that triggered it.

The analyzer scores STRUCTURE and DELIVERY signals. It cannot judge whether a
story is impressive or a technical statement is factually correct.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .lexicons import get_lexicons, phrase_matcher
from .text import STOPWORDS, Span, content_stems, content_tokens, sentences, snippet, stem, words

DIMENSIONS = ("relevance", "structure", "specificity", "concision", "confidence")
WEIGHTS = {"relevance": 0.25, "structure": 0.30, "specificity": 0.20, "concision": 0.10, "confidence": 0.15}
STAR_WEIGHTS = {"situation": 20, "task": 15, "action": 35, "result": 30}
STAR_LABELS = {"situation": "Situation", "task": "Task", "action": "Action", "result": "Result"}
PRESENT_THRESHOLD = 1.0  # position-weighted cue weight needed to call a STAR component "present"

# Ideal answer length (words) per answer profile. ~130-150 spoken words per minute.
IDEAL_WORDS = {
    "star": (90, 250),
    "situational": (60, 200),
    "technical": (50, 200),
    "why_org": (40, 130),
    "intro": (60, 170),
    "ask_us": (20, 120),
}
PROFILES = tuple(IDEAL_WORDS)
IDEAL_WPM = (110, 170)
QUESTION_STOP = {"tell", "time", "describe", "give", "example", "explain", "walk", "through", "talk", "share", "think", "would", "could", "situation"}

HIGHLIGHT_PRIORITY = ["filler", "hedge", "weak", "vague", "generic", "premature", "number", "star-r", "star-a", "star-t", "star-s", "match", "verb", "we"]


@dataclass
class Context:
    """Everything the analyzer needs to know about the question being answered."""

    question_text: str = ""
    profile: str = "star"
    competency: str | None = None
    competency_cues: list[str] = field(default_factory=list)
    concepts: list[list[str]] = field(default_factory=list)
    org: str = ""
    role: str = ""
    jd_skills: dict[str, list[str]] = field(default_factory=dict)  # canonical -> aliases
    about_keywords: list[str] = field(default_factory=list)
    duration_seconds: float | None = None
    spoken: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "Context":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- helpers


def _interp(x: float, zero_at: float, full_at: float, possible: float) -> float:
    """Linear ramp: 0 points at `zero_at`, all points at `full_at` (works for either direction)."""
    if zero_at == full_at:
        return possible if x >= full_at else 0.0
    t = (x - zero_at) / (full_at - zero_at)
    return possible * max(0.0, min(1.0, t))


def _spans(text: str, pattern: re.Pattern, limit: int = 8) -> list[Span]:
    return [Span(m.start(), m.end(), m.group()) for m in pattern.finditer(text)][:limit]


def _all_spans(text: str, pattern: re.Pattern) -> list[Span]:
    return [Span(m.start(), m.end(), m.group()) for m in pattern.finditer(text)]


def _check(label: str, earned: float, possible: float, detail: str, evidence: list[Span] | None = None) -> dict:
    return {
        "label": label,
        "earned": round(earned, 1),
        "possible": round(possible, 1),
        "detail": detail,
        "evidence": [s.to_dict() for s in (evidence or [])[:6]],
    }


def _dimension(checks: list[dict]) -> dict:
    possible = sum(c["possible"] for c in checks)
    earned = sum(c["earned"] for c in checks)
    score = 0 if possible <= 0 else max(0, min(100, round(100 * earned / possible)))
    return {"score": score, "checks": checks}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _names(spans: list[Span], limit: int = 4) -> str:
    seen: list[str] = []
    for s in spans:
        t = s.text.strip().lower()
        if t not in seen:
            seen.append(t)
    shown = ", ".join(f"“{t}”" for t in seen[:limit])
    return shown + ("…" if len(seen) > limit else "")


# --------------------------------------------------------------------------- feature extraction


class Features:
    """Everything measured from the answer text, computed once."""

    def __init__(self, text: str, ctx: Context) -> None:
        L = get_lexicons()
        self.text = text
        self.ctx = ctx
        self.n_chars = max(len(text), 1)
        self.words = words(text)
        self.wc = len(self.words)
        self.sents = sentences(text)
        self.sent_wcs = [len(words(s.text)) for s in self.sents]
        self.avg_sent = (sum(self.sent_wcs) / len(self.sent_wcs)) if self.sent_wcs else 0.0
        self.max_sent = max(self.sent_wcs) if self.sent_wcs else 0
        self.no_punct = self.wc >= 60 and len(self.sents) <= 1
        self.stems = content_stems(text)
        self.low = text.lower()

        # numbers / quantities (bare 4-digit years are context, not results)
        nums = []
        for m in L.number_re.finditer(text):
            raw = m.group().strip()
            if re.fullmatch(r"(19|20)\d{2}", raw):
                continue
            nums.append(Span(m.start(), m.end(), raw))
        nums += _all_spans(text, L.number_word_re)
        nums.sort(key=lambda s: s.start)
        self.numbers = _dedupe_overlaps(nums)

        # delivery signals
        self.fillers = _all_spans(text, L.filler_re) + _like_fillers(text)
        self.hedges = _all_spans(text, L.hedge_re)
        self.weak = _all_spans(text, L.weak_re)
        self.vague = _all_spans(text, L.vague_re)
        self.generic_org = _all_spans(text, L.generic_org_re)
        per100 = 100 / max(self.wc, 1)
        self.filler_rate = len(self.fillers) * per100
        self.hedge_rate = len(self.hedges) * per100

        # ownership ("I" vs "we")
        self.i_spans = _all_spans(text, L.first_person_i_re)
        self.we_spans = _all_spans(text, L.first_person_we_re)
        tot = len(self.i_spans) + len(self.we_spans)
        self.personal_ratio = (len(self.i_spans) / tot) if tot else None

        # verbs the speaker attributes to themselves
        self.i_verbs: list[Span] = []
        for m in L.i_action_re.finditer(text):
            v = m.group(1).lower()
            if v in L.strong_verbs or v in L.irregular_past or (v.endswith("ed") and len(v) > 4 and v not in STOPWORDS and v not in L.stative_verbs):
                self.i_verbs.append(Span(m.start(1), m.end(1), m.group(1)))
        # verbs in a series after the first "I <verb>" ("I gathered the lists, printed a copy, and showed everyone")
        for sent in self.sents:
            first = next((v for v in self.i_verbs if sent.start <= v.start < sent.end), None)
            if first:
                for m in re.finditer(r"(?:,|\band)\s+(?:then\s+)?([a-z]+)\b", sent.text[first.end - sent.start :], re.IGNORECASE):
                    v = m.group(1).lower()
                    if v in L.strong_verbs or v in L.irregular_past or (v.endswith("ed") and len(v) > 4 and v not in STOPWORDS and v not in L.stative_verbs):
                        a = first.end + m.start(1)
                        self.i_verbs.append(Span(a, a + len(v), m.group(1)))
        self.distinct_i_verbs = {s.text.lower() for s in self.i_verbs}
        self.strong_verb_spans = [Span(w.start, w.end, w.text) for w in self.words if w.text.lower() in L.strong_verbs]
        self.conditional_steps = [m.group(1).lower() for m in re.finditer(r"\bi(?:'d|’d| would|'ll|’ll| will)\s+(?:then\s+|first\s+|also\s+)?([a-z]+)", text, re.IGNORECASE)]

        # STAR cues
        self.star = self._star(L)
        self.learning = [Span(m.start(), m.end(), m.group()) for rx in L.learning_res for m in rx.finditer(text)]

        # named details: tools/skills from the job description, concepts, capitalised names
        self.jd_hits = self._jd_hits()
        self.concept_hits, self.concept_missing = self._concepts()
        self.proper_nouns = self._proper_nouns(L)

        # relevance signals
        comp_stems = {stem(c): c for c in ctx.competency_cues}
        self.comp_hits = [w for w in self.words if stem(w.text) in comp_stems]
        self.comp_distinct = {stem(w.text) for w in self.comp_hits}
        q_stems = {s for s in content_stems(ctx.question_text) if s not in {stem(x) for x in QUESTION_STOP}}
        self.q_overlap = (len(q_stems & self.stems) / len(q_stems)) if q_stems else 0.0
        self.org_hits = self._org_hits()
        self.role_hits = self._role_hits()
        self.about_hits = [w for w in self.words if stem(w.text) in {stem(k) for k in ctx.about_keywords} and w.text.lower() not in STOPWORDS]
        self.about_distinct = {stem(w.text) for w in self.about_hits}

        # structure signals
        self.sequence = _spans(text, L.sequence_re, 12)
        self.reasoning = _spans(text, L.reasoning_re, 12)
        self.verification = _spans(text, L.verification_re, 12)
        self.stakeholders = _spans(text, L.stakeholder_re, 12)
        self.examples = _spans(text, L.example_re, 8)
        self.details = _spans(text, L.detail_re, 8)

        # repetition
        self.repeated_words, self.repeated_trigrams = self._repetition()

        # pacing (only meaningful when the answer was spoken and timed)
        self.wpm = None
        if ctx.spoken and ctx.duration_seconds and ctx.duration_seconds >= 10 and self.wc >= 10:
            self.wpm = round(self.wc / (ctx.duration_seconds / 60))

    # ---- STAR
    def _star(self, L) -> dict:
        out = {}
        for comp, cfg in L.star.items():
            lo, hi = cfg["zone"]
            cues: list[tuple[Span, float]] = []
            for rx in cfg["res"]:
                for m in rx.finditer(self.text):
                    sp = Span(m.start(), m.end(), m.group())
                    if any(sp.start < c.end and c.start < sp.end for c, _ in cues):
                        continue
                    pos = m.start() / self.n_chars
                    w = 1.0 if lo <= pos <= hi else 0.5
                    if self._in_hypothetical(sp, L):
                        w *= 0.5  # "I would..." / "I always..." describes a habit or hypothetical, not a story
                    cues.append((sp, w))
            weight = sum(w for _, w in cues)
            extra = ""
            if comp == "action":
                n_v = len(self.distinct_i_verbs)
                bonus = 0.0 if n_v == 0 else min(2.0, 1.0 + 0.5 * (n_v - 1))  # one first-person action verb is enough to call it an action
                if bonus:
                    weight += bonus
                    extra = f" plus {_plural(len(self.distinct_i_verbs), 'first-person action verb')}"
            if comp == "result":
                late_numbers = [n for n in self.numbers if n.start / self.n_chars >= 0.5]
                if late_numbers:
                    weight += 0.5
                    extra = " plus a number in the closing half"
            out[comp] = {"weight": weight, "cues": [c for c, _ in cues], "extra": extra}
        for comp, d in out.items():
            w = d["weight"]
            if w >= PRESENT_THRESHOLD:
                d["status"], d["factor"] = "present", min(1.0, 0.6 + 0.2 * (w - 1))
            elif w > 0:
                d["status"], d["factor"] = "weak", 0.3
            else:
                d["status"], d["factor"] = "missing", 0.0
        return out

    def _in_hypothetical(self, sp: Span, L) -> bool:
        for sent in self.sents:
            if sent.start <= sp.start < sent.end:
                return bool(L.hypothetical_re.search(sent.text))
        return False

    def _jd_hits(self) -> dict[str, list[Span]]:
        hits: dict[str, list[Span]] = {}
        for canon, aliases in self.ctx.jd_skills.items():
            rx = phrase_matcher([canon] + list(aliases), inflect=True)
            found = _all_spans(self.text, rx)
            if found:
                hits[canon] = found
        return hits

    def _concepts(self):
        hits, missing = {}, []
        for group in self.ctx.concepts:
            rx = phrase_matcher(group, inflect=True)
            found = _all_spans(self.text, rx)
            if found:
                hits[group[0]] = found
            else:
                missing.append(group[0])
        return hits, missing

    def _proper_nouns(self, L) -> list[Span]:
        starts = {s.start for s in self.sents}
        out = []
        for w in self.words:
            t = w.text
            if len(t) > 1 and t[0].isupper() and w.start not in starts and t not in L.proper_noun_ignore and not t.isupper():
                out.append(Span(w.start, w.end, t))
            elif len(t) >= 2 and t.isupper() and t not in {"I", "A"} and t.isalpha() and w.start not in starts:
                out.append(Span(w.start, w.end, t))  # acronyms: SQL, AWS
        return out

    def _org_hits(self) -> list[Span]:
        org = self.ctx.org.strip()
        if not org:
            return []
        return _all_spans(self.text, phrase_matcher([org]))

    def _role_hits(self) -> list[Span]:
        role_stems = {stem(t) for t in content_tokens(self.ctx.role)}
        return [Span(w.start, w.end, w.text) for w in self.words if stem(w.text) in role_stems]

    def _repetition(self):
        toks = [w.text.lower() for w in self.words]
        counts = Counter(stem(t) for t in toks if t not in STOPWORDS and len(t) > 3)
        repeated_words = {k: v for k, v in counts.items() if v >= 4}
        tri = Counter()
        for i in range(len(toks) - 2):
            g = tuple(toks[i : i + 3])
            if sum(1 for t in g if t not in STOPWORDS) >= 2:
                tri[g] += 1
        repeated_trigrams = {" ".join(k): v for k, v in tri.items() if v >= 2}
        return repeated_words, repeated_trigrams


def _like_fillers(text: str) -> list[Span]:
    """'like' used as a filler, not as a comparison/verb: ', like,' / 'was like' / 'so like'."""
    pat = re.compile(
        r"(?:,\s*like\b(?=\s*,|\s+(?:i|we|it|the|a|um|you)\b))|\b(?:so|and|but|just|was|were|it's|it’s)\s+like\b(?!\s+(?:a|an|the|my|our|this|that|you|to|when|how|as)\b)",
        re.IGNORECASE,
    )
    out = []
    for m in pat.finditer(text):
        i = m.group().lower().rfind("like")
        out.append(Span(m.start() + i, m.start() + i + 4, "like"))
    return out


def _dedupe_overlaps(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- shared checks


def _concision(f: Features) -> dict:
    lo, hi = IDEAL_WORDS[f.ctx.profile]
    checks = []
    if f.wc < lo:
        pts = _interp(f.wc, 0.3 * lo, lo, 60)
        detail = f"{f.wc} words is short for this kind of answer (aim for {lo}–{hi}). Short answers rarely have room for a specific example and a result."
    elif f.wc > hi:
        pts = _interp(f.wc, 1.8 * hi, hi, 60)
        detail = f"{f.wc} words is long (aim for {lo}–{hi}, roughly {round(lo/140,1)}–{round(hi/140,1)} minutes spoken). Cut background and keep the action and result."
    else:
        pts, detail = 60, f"{f.wc} words sits inside the {lo}–{hi} word target."
    checks.append(_check("Length", pts, 60, detail))

    if f.no_punct:
        checks.append(_check("Sentence length", 0, 25, f"{f.wc} words with almost no sentence breaks — it reads as one run-on. (Dictation often drops punctuation; add periods to see this score properly.)"))
    else:
        pts = _interp(f.avg_sent, 38, 22, 25)
        longest = max(zip(f.sent_wcs, f.sents), default=(0, None), key=lambda t: t[0])
        ev = [Span(longest[1].start, longest[1].end, longest[1].text[:120])] if longest[0] > 40 and longest[1] else []
        detail = f"Average sentence is {f.avg_sent:.0f} words" + (f"; the longest runs {longest[0]} words." if longest[0] > 40 else ". Short, clear sentences are easy to follow when spoken." if pts >= 25 else ". Try breaking long sentences in two.")
        checks.append(_check("Sentence length", pts, 25, detail, ev))

    n_rep = len(f.repeated_words) + len(f.repeated_trigrams)
    if n_rep == 0:
        checks.append(_check("Repetition", 15, 15, "No repeated key words or phrases."))
    else:
        items = list(f.repeated_words) + list(f.repeated_trigrams)
        checks.append(_check("Repetition", max(0, 15 - 5 * n_rep), 15, f"Repeated: {', '.join('“' + i + '”' for i in items[:4])}. Vary wording or merge the repeated points."))

    if f.wpm is not None:
        pts = min(_interp(f.wpm, 70, IDEAL_WPM[0], 20), _interp(f.wpm, 210, IDEAL_WPM[1], 20))
        note = "in the comfortable range" if IDEAL_WPM[0] <= f.wpm <= IDEAL_WPM[1] else ("a little fast — slow down and pause between points" if f.wpm > IDEAL_WPM[1] else "slow — fine for care, but watch for long pauses")
        checks.append(_check("Speaking pace", pts, 20, f"{f.wpm} words per minute ({note}; target {IDEAL_WPM[0]}–{IDEAL_WPM[1]})."))
    return _dimension(checks)


def _confidence(f: Features, with_ownership: bool = True) -> dict:
    checks = []
    if with_ownership:
        full = 0.4 if f.ctx.competency in ("teamwork", "leadership") else 0.6
        if f.personal_ratio is None:
            checks.append(_check("Ownership (I vs we)", 10, 30, "No first-person statements found. Say what YOU did (“I organized…”), not just what happened."))
        else:
            pts = _interp(f.personal_ratio, 0.15, full, 30)
            detail = f"{len(f.i_spans)} personal (“I/my/me”) vs {len(f.we_spans)} group (“we/our/us”) references — {round(100*f.personal_ratio)}% personal."
            if f.personal_ratio < 0.3 and f.we_spans:
                detail += " Interviewers are hiring you, so make your own contribution unmistakable."
            checks.append(_check("Ownership (I vs we)", pts, 30, detail, f.we_spans if f.personal_ratio < 0.5 else []))
    pts = _interp(f.hedge_rate, 6, 1, 30)
    checks.append(_check("Hedging", pts, 30,
                         (f"{_plural(len(f.hedges), 'hedge')} ({_names(f.hedges)}) — {f.hedge_rate:.1f} per 100 words. State results plainly instead of softening them." if f.hedges else "No hedging phrases (“I think”, “maybe”, “sort of”)."), f.hedges))
    pts = _interp(f.filler_rate, 5, 0.5, 25)
    checks.append(_check("Filler words", pts, 25,
                         (f"{_plural(len(f.fillers), 'filler')} ({_names(f.fillers)}) — {f.filler_rate:.1f} per 100 words. Pause silently instead." if f.fillers else "No filler words detected."), f.fillers))
    pts = max(0, 15 - 5 * len(f.weak))
    checks.append(_check("Ownership language", pts, 15,
                         (f"Weak-ownership phrases: {_names(f.weak)}. Swap for a direct verb (“helped with the report” → “wrote sections 2 and 3 of the report”)." if f.weak else "No weak-ownership phrases (“helped with”, “was involved in”)."), f.weak))
    return _dimension(checks)


def _vague_deduction(f: Features) -> dict | None:
    if not f.vague:
        return None
    return _check("Vague wording", -min(20, 4 * len(f.vague)), 0, f"Vague words weaken specifics: {_names(f.vague)}. Replace with the actual thing, number or name.", f.vague)


def _quant_check(f: Features, possible: float, what: str = "results") -> dict:
    n = len(f.numbers)
    pts = 0 if n == 0 else (possible * 0.6 if n == 1 else possible)
    detail = (f"{_plural(n, 'number')} found ({_names(f.numbers)}). Numbers make {what} concrete." if n else f"No numbers. Add at least one measurable detail: how many, how much, how long, how often, what %.")
    return _check("Numbers & measurable detail", pts, possible, detail, f.numbers)


def _named_check(f: Features, possible: float, include_concepts: bool = True) -> dict:
    named: dict[str, Span] = {}
    for canon, sp in f.jd_hits.items():
        named.setdefault(canon.lower(), sp[0])
    if include_concepts:
        for canon, sp in f.concept_hits.items():
            named.setdefault(canon.lower(), sp[0])
    for sp in f.proper_nouns:
        named.setdefault(sp.text.lower(), sp)
    n = len(named)
    pts = possible * min(n, 3) / 3
    detail = (f"{_plural(n, 'named detail')}: {_names(list(named.values()))}. Naming tools, people’s roles, courses or places makes a story believable." if n else "No named tools, courses, places or systems. Name what you used or where it happened.")
    return _check("Named details (tools, places, courses)", pts, possible, detail, list(named.values()))


def _verb_check(f: Features, possible: float) -> dict:
    n = len(f.distinct_i_verbs)
    pts = possible * min(n, 4) / 4
    detail = (f"{_plural(n, 'first-person action verb')}: {_names(f.i_verbs)}. Strong verbs show what you actually did." if n else "No “I + action verb” statements found (“I built…”, “I organized…”). Interviewers listen for what YOU did.")
    return _check("Action verbs", pts, possible, detail, f.i_verbs)


def _star_check(f: Features, comp: str) -> dict:
    d = f.star[comp]
    weight = STAR_WEIGHTS[comp]
    label = STAR_LABELS[comp]
    ev = d["cues"][:4]
    if d["status"] == "present":
        detail = f"{label} detected: {_names(ev) or 'action statements'}{d['extra']}."
    elif d["status"] == "weak":
        detail = f"{label} is only weakly signalled ({_names(ev)}). Make it explicit."
    else:
        detail = MISSING_STAR_HINT[comp]
    return _check(label, weight * d["factor"], weight, detail, ev)


MISSING_STAR_HINT = {
    "situation": "No situation detected: nothing clearly places the story in a real time and place. Open with when and where (“Last semester in my statistics class…”). (The detector can miss unusual phrasing; if you did say it, use a clearer time marker.)",
    "task": "No task or goal detected. Say what you were responsible for or trying to achieve (“My job was to…”).",
    "action": "No clear actions detected. Walk through what YOU did, step by step (“First I…, then I…”).",
    "result": "No clear result detected. End with the outcome: what changed, by how much, and what you learned. (Phrases like “as a result”, “we ended up”, or a number make it unmistakable.)",
}


# --------------------------------------------------------------------------- profiles


def _relevance_star(f: Features, situational: bool = False) -> dict:
    c = []
    if f.ctx.competency_cues:
        n = len(f.comp_distinct)
        c.append(_check("Stays on the competency", 30 * min(n, 3) / 3, 30,
                        (f"Uses language tied to {f.ctx.competency or 'the topic'} ({_names(f.comp_hits)})." if n else f"Little language tied to {f.ctx.competency or 'the competency'} — make the connection explicit (e.g. {', '.join(f.ctx.competency_cues[:3])})."), f.comp_hits))
    c.append(_check("Answers the question asked", 20 * min(f.q_overlap / 0.5, 1), 20,
                    f"{round(100*f.q_overlap)}% of the question’s key words appear in your answer." + ("" if f.q_overlap >= 0.5 else " Echo the question’s terms so the link is obvious.")))
    if situational:
        n = len({s.text.lower() for s in f.stakeholders})
        c.append(_check("Considers other people", 25 * min(n, 2) / 2, 25,
                        (f"Mentions who you would involve ({_names(f.stakeholders)})." if n else "No mention of who you’d talk to or involve (manager, teammate, customer…). Situational answers should show communication."), f.stakeholders))
    else:
        s, a = f.star["situation"]["status"], f.star["action"]["status"]
        earned = 25 if "present" in (s, a) else (12 if "weak" in (s, a) else 0)
        c.append(_check("Gives a real example", earned, 25,
                        "You describe a concrete situation or your own actions." if earned == 25 else "The answer stays general. Interviewers want one real example, not a philosophy."))
    if f.ctx.jd_skills:
        n = len(f.jd_hits)
        c.append(_check("Connects to the job description", 15 * min(n, 2) / 2, 15,
                        (f"Mentions skills from the posting: {_names([v[0] for v in f.jd_hits.values()])}." if n else "Doesn’t touch any skill from the pasted job description. Where honest, tie your example to one."), [v[0] for v in f.jd_hits.values()]))
    return _dimension(c)


def _structure_star(f: Features) -> dict:
    checks = [_star_check(f, k) for k in ("situation", "task", "action", "result")]
    if f.ctx.competency in ("failure", "conflict"):
        got = bool(f.learning)
        checks.append(_check("Reflection", 10 if got else 0, 10,
                             ("Shows what you learned or would change: " + _names(f.learning) if got else "No reflection. For this kind of question, say what you learned or would do differently."), f.learning))
    return _dimension(checks)


def _specificity_star(f: Features) -> dict:
    checks = [_quant_check(f, 30), _named_check(f, 25), _verb_check(f, 20)]
    s = f.star["situation"]
    t = f.star["task"]
    best = max(s["factor"], t["factor"])
    checks.append(_check("Concrete setting", 15 * (1 if best >= 0.6 else 0.5 if best > 0 else 0), 15,
                         "The setting (when/where/what) is stated." if best >= 0.6 else "The setting is missing or vague. When and where did this happen?", (s["cues"] + t["cues"])[:3]))
    v = _vague_deduction(f)
    if v:
        checks.append(v)
    return _dimension(checks)


def _relevance_situational(f: Features) -> dict:
    return _relevance_star(f, situational=True)


def _structure_situational(f: Features) -> dict:
    steps = len(set(f.conditional_steps))
    n_seq = len({s.text.lower() for s in f.sequence})
    n_ver = len({s.text.lower() for s in f.verification})
    n_rea = len({s.text.lower() for s in f.reasoning})
    checks = [
        _check("Ordered steps", 25 * min(n_seq, 3) / 3, 25, (f"Sequencing words ({_names(f.sequence)}) make the plan easy to follow." if n_seq else "No sequencing (“first… then… finally”). Lay out your plan as steps."), f.sequence),
        _check("Says what you would do", 20 * min(steps, 3) / 3, 20, (f"{_plural(steps, 'distinct “I would…” action')}: {', '.join(sorted(set(f.conditional_steps))[:4])}." if steps else "No “I would…” statements. Situational answers should commit to actions (“I would first ask…”)."), []),
        _check("Follows through / verifies", 25 * min(n_ver, 2) / 2, 25, (f"Includes follow-through ({_names(f.verification)})." if n_ver else "No follow-through. Say how you’d check it worked (confirm, follow up, document, review)."), f.verification),
        _check("Explains the reasoning", 20 * min(n_rea, 2) / 2, 20, (f"Gives reasons ({_names(f.reasoning)})." if n_rea else "No reasoning words. Say WHY you’d take each step (“because…”, “so that…”)."), f.reasoning),
    ]
    return _dimension(checks)


def _specificity_situational(f: Features) -> dict:
    steps = len(set(f.conditional_steps))
    checks = [_quant_check(f, 15, "your plan"), _named_check(f, 25), _check("Concrete actions", 30 * min(steps + len(f.distinct_i_verbs), 4) / 4, 30, f"{_plural(steps + len(f.distinct_i_verbs), 'concrete action')} stated. Specific actions beat “I’d handle it professionally”.")]
    v = _vague_deduction(f)
    if v:
        checks.append(v)
    return _dimension(checks)


def _relevance_technical(f: Features) -> dict:
    total = len(f.ctx.concepts)
    checks = []
    if total:
        got = len(f.concept_hits)
        target = max(1, round(total * 0.6))
        pts = 60 * min(got / target, 1)
        detail = f"Touches {got} of {total} expected ideas" + (f" ({_names([v[0] for v in f.concept_hits.values()])})." if got else ".")
        if f.concept_missing:
            detail += f" Not mentioned: {', '.join(f.concept_missing[:5])}."
        detail += " (Checks that ideas are mentioned, not that they are correct.)"
        checks.append(_check("Covers the key ideas", pts, 60, detail, [v[0] for v in f.concept_hits.values()]))
    checks.append(_check("Answers the question asked", 15 * min(f.q_overlap / 0.5, 1), 15, f"{round(100*f.q_overlap)}% of the question’s key words appear in your answer."))
    if f.ctx.jd_skills:
        n = len(f.jd_hits)
        checks.append(_check("Connects to the job description", 15 * min(n, 2) / 2, 15, (f"Mentions {_names([v[0] for v in f.jd_hits.values()])} from the posting." if n else "No skill from the pasted job description appears."), [v[0] for v in f.jd_hits.values()]))
    return _dimension(checks)


def _structure_technical(f: Features) -> dict:
    first = f.sents[0] if f.sents else None
    direct = bool(first) and len(words(first.text)) <= 30 and any(first.start <= sp.start < first.end for v in f.concept_hits.values() for sp in v)
    if not f.ctx.concepts:
        direct = bool(first) and len(words(first.text)) <= 30
    n_rea = len({s.text.lower() for s in f.reasoning})
    checks = [
        _check("Direct first sentence", 25 if direct else 8, 25, "You lead with a short, direct answer." if direct else "Lead with a one-sentence direct answer (definition or decision), then explain."),
        _check("Explains why / trade-offs", 30 * min(n_rea, 2) / 2, 30, (f"Reasoning present ({_names(f.reasoning)})." if n_rea else "No reasoning words. Explain why, and mention a trade-off or alternative."), f.reasoning),
        _check("Gives an example", 25 if f.examples else 0, 25, (f"Includes an example ({_names(f.examples)})." if f.examples else "No example. “For instance, in my project…” proves you have used the idea."), f.examples),
        _check("Ordered explanation", 20 * min(len({s.text.lower() for s in f.sequence}), 2) / 2, 20, ("Uses ordering words to walk through the idea." if f.sequence else "Add ordering (“first… then…”) so the explanation can be followed."), f.sequence),
    ]
    return _dimension(checks)


def _specificity_technical(f: Features) -> dict:
    hits = len(f.concept_hits)
    complexity = re.findall(r"\bO\([^)]*\)|\b\d+\s?(?:ms|s|gb|mb|kb|rows|requests|users)\b", f.text, re.IGNORECASE)
    checks = [
        _check("Precise terminology", 30 * min(hits, 4) / 4 if f.ctx.concepts else 15, 30, (f"Uses precise terms: {_names([v[0] for v in f.concept_hits.values()])}." if hits else "Few precise technical terms. Use the exact names of the concepts and tools.") if f.ctx.concepts else "No expected terms configured for this question.", [v[0] for v in f.concept_hits.values()]),
        _named_check(f, 20, include_concepts=False),
        _check("Numbers / sizes / complexity", 15 if (f.numbers or complexity) else 0, 15, "Includes concrete numbers or complexity." if (f.numbers or complexity) else "No numbers, sizes or complexity notes (e.g. O(n), 200 ms, 10k rows).", f.numbers),
        _check("Grounded in experience", 20 if (f.examples or f.i_verbs) else 0, 20, "Ties the idea to something you built or did." if (f.examples or f.i_verbs) else "Nothing ties this to your own work. Mention where you used it."),
    ]
    v = _vague_deduction(f)
    if v:
        checks.append(v)
    return _dimension(checks)


def _relevance_why(f: Features) -> dict:
    reason_words = re.compile(r"\b(because|drawn|interested|excited|appeal|align|value|mission|admire|impress|respect|attracted|want to|goal|hope to)\b", re.IGNORECASE)
    reasons = _all_spans(f.text, reason_words)
    checks = [
        _check("Gives actual reasons", 25 * min(len(reasons), 2) / 2, 25, (f"States reasons ({_names(reasons)})." if reasons else "No stated reasons. Say “I’m drawn to X because Y”."), reasons),
        _check("Names the organization", 25 if f.org_hits else 0, 25, ("Names the organization." if f.org_hits else f"Never says “{f.ctx.org or 'the organization'}”. A named reason sounds researched."), f.org_hits),
        _check("Connects to the role", 25 * min(len(f.role_hits) + len(f.jd_hits), 2) / 2, 25, (f"Connects to the role ({_names(f.role_hits + [v[0] for v in f.jd_hits.values()])})." if (f.role_hits or f.jd_hits) else "Doesn’t connect to the role itself. Say what about THIS work interests you."), f.role_hits + [v[0] for v in f.jd_hits.values()]),
    ]
    if f.ctx.about_keywords:
        n = len(f.about_distinct)
        checks.append(_check("Uses the company text you pasted", 25 * min(n, 3) / 3, 25, (f"Echoes specifics from the about-text you provided ({_names(f.about_hits)})." if n else "None of the specifics from your pasted about-text appear. Pick one that’s true for you and mention it."), f.about_hits))
    return _dimension(checks)


def _structure_why(f: Features) -> dict:
    org_part = bool(f.org_hits) or len(f.about_distinct) >= 2
    role_part = bool(f.role_hits or f.jd_hits)
    you_part = bool(re.search(r"\b(my|i've|i have|i studied|i built|i learned|i led|coursework|class|project|experience|background|skills)\b", f.text, re.IGNORECASE)) and (bool(f.i_spans) )
    checks = [
        _check("Reason about the organization", 35 if org_part else 0, 35, "Gives a reason specific to the organization." if org_part else "No organization-specific reason found."),
        _check("Reason about the role", 30 if role_part else 0, 30, "Gives a reason tied to the role/work." if role_part else "No role-specific reason found."),
        _check("Ties it to you", 35 if you_part else 0, 35, "Connects the reasons to your background or goals." if you_part else "Doesn’t connect to your own background or goals. Say what you bring or want to learn."),
    ]
    return _dimension(checks)


def _specificity_why(f: Features) -> dict:
    g = len(f.generic_org)
    checks = [_named_check(f, 30), _quant_check(f, 15, "your background"), _verb_check(f, 15)]
    if g:
        checks.append(_check("Generic phrasing", -min(30, 10 * g), 0, f"Generic praise that could apply to any employer: {_names(f.generic_org)}. Replace with something only true of this organization.", f.generic_org))
    else:
        checks.append(_check("Avoids generic praise", 20, 20, "No stock phrases like “great company” or “fast-paced”."))
    v = _vague_deduction(f)
    if v:
        checks.append(v)
    return _dimension(checks)


def _relevance_intro(f: Features) -> dict:
    connect = f.role_hits + [v[0] for v in f.jd_hits.values()]
    checks = [
        _check("Connects to the role", 60 * min(len(connect), 2) / 2, 60, (f"Links your background to the role ({_names(connect)})." if connect else "Never connects back to the role. End by linking your story to why you fit this position."), connect),
        _check("Shows real background", 40 if (f.i_verbs or f.numbers or f.proper_nouns) else 0, 40, "Mentions concrete background." if (f.i_verbs or f.numbers or f.proper_nouns) else "Very little concrete background (school, projects, jobs)."),
    ]
    return _dimension(checks)


def _structure_intro(f: Features) -> dict:
    L = get_lexicons()
    present = _spans(f.text, L.intro_present_re)
    past = _spans(f.text, L.intro_past_re) or ([f.i_verbs[0]] if len(f.i_verbs) >= 2 else [])
    future = _spans(f.text, L.intro_future_re)
    checks = [
        _check("Present: who you are now", 30 if present else 0, 30, (f"Present-tense intro ({_names(present)})." if present else "Start with who you are now (“I’m a second-year CS student at…”)."), present),
        _check("Past: relevant background", 35 if past else 0, 35, (f"Background covered ({_names(past)})." if past else "Add one or two relevant experiences or projects."), past),
        _check("Future: why this role", 35 if future else 0, 35, (f"Looks forward ({_names(future)})." if future else "Close with where you’re headed and why this role fits."), future),
    ]
    return _dimension(checks)


def _specificity_intro(f: Features) -> dict:
    checks = [_named_check(f, 30), _quant_check(f, 20, "your background"), _verb_check(f, 25)]
    v = _vague_deduction(f)
    if v:
        checks.append(v)
    return _dimension(checks)


def _ask_questions(f: Features) -> list[Span]:
    out = []
    pos = 0
    for part in re.split(r"(?<=[?])\s+|\n+", f.text):
        seg = part.strip()
        if seg:
            i = f.text.find(seg, pos)
            out.append(Span(i, i + len(seg), seg))
            pos = i + len(seg)
    return [s for s in out if len(words(s.text)) >= 3]


def _relevance_ask(f: Features) -> dict:
    L = get_lexicons()
    ref = f.org_hits + f.role_hits + [v[0] for v in f.jd_hits.values()] + f.about_hits
    fwd_re = phrase_matcher(["success", "first 90 days", "first few months", "challenges", "growth", "learn", "mentor", "team", "next steps", "expectations", "priorities", "culture", "day-to-day", "day to day", "projects", "training", "onboarding"])
    fwd = _spans(f.text, fwd_re)
    prem = _all_spans(f.text, L.premature_re)
    lazy = _all_spans(f.text, L.lazy_re)
    checks = [
        _check("Refers to this org / role", 40 * min(len(ref), 2) / 2, 40, (f"Anchored in this job ({_names(ref)})." if ref else "Nothing anchors these questions to this organization or role."), ref),
        _check("Forward-looking topics", 30 * min(len({s.text.lower() for s in fwd}), 2) / 2, 30, (f"Asks about success, growth or the work ({_names(fwd)})." if fwd else "Ask about success in the role, the team, training or challenges."), fwd),
    ]
    if prem:
        checks.append(_check("Too early for pay/perks", -20, 0, f"Save {_names(prem)} for the recruiter or offer stage.", prem))
    if lazy:
        checks.append(_check("Answerable from their website", -30, 0, "This is answerable from a quick search and signals no research.", lazy))
    return _dimension(checks)


def _structure_ask(f: Features) -> dict:
    qs = _ask_questions(f)
    n = len(qs)
    if n == 0:
        pts_n, detail_n = 0, "No questions found. Prepare 2–4."
    elif n == 1:
        pts_n, detail_n = 20, "Only one question. Prepare 2–4 so you never run out."
    elif n <= 5:
        pts_n, detail_n = 40, f"{n} questions is a good number."
    else:
        pts_n, detail_n = 25, f"{n} questions is a lot — pick your best 3–4."
    open_re = re.compile(r"^(how|what|why|which|where|when|could you|can you tell|tell me|describe|walk me)", re.IGNORECASE)
    opens = [q for q in qs if open_re.match(q.text.strip())]
    frac = len(opens) / n if n else 0
    return _dimension([
        _check("Number of questions", pts_n, 40, detail_n),
        _check("Open-ended", 40 * min(frac / 0.6, 1), 40, f"{len(opens)} of {n} are open-ended (how/what/why) — these start conversations." if n else "—", opens),
        _check("Phrased as questions", 20 if f.text.count("?") >= max(1, n - 1) and n else 0, 20, "Each is phrased as a question." if f.text.count("?") >= max(1, n - 1) and n else "Phrase each as a real question ending with “?”."),
    ])


def _specificity_ask(f: Features) -> dict:
    ref = f.about_hits + [v[0] for v in f.jd_hits.values()]
    return _dimension([
        _check("Uses details you researched", 40 * min(len(ref), 2) / 2, 40, (f"References specifics ({_names(ref)})." if ref else "Weave in one detail from the job posting or company text."), ref),
        _named_check(f, 30, include_concepts=False),
        _check("Sounds like you", 30 if f.i_spans else 10, 30, "Uses your own voice (“I’m curious how…”)." if f.i_spans else "Add a personal frame: “I’m curious how…”, “I’d like to understand…”."),
    ])


PROFILE_FUNCS = {
    "star": (_relevance_star, _structure_star, _specificity_star),
    "situational": (_relevance_situational, _structure_situational, _specificity_situational),
    "technical": (_relevance_technical, _structure_technical, _specificity_technical),
    "why_org": (_relevance_why, _structure_why, _specificity_why),
    "intro": (_relevance_intro, _structure_intro, _specificity_intro),
    "ask_us": (_relevance_ask, _structure_ask, _specificity_ask),
}


# --------------------------------------------------------------------------- gaps + highlights


def _gaps(f: Features, dims: dict) -> list[str]:
    ctx = f.ctx
    lo, hi = IDEAL_WORDS[ctx.profile]
    g = []
    if f.wc < 0.5 * lo:
        g.append("too_short")
    if f.wc > 1.3 * hi:
        g.append("too_long")
    if ctx.profile == "star":
        if f.star["result"]["status"] != "present":
            g.append("no_result")
        if f.star["situation"]["status"] != "present" and f.star["task"]["status"] != "present":
            g.append("no_example")
        if f.star["action"]["status"] != "present":
            g.append("no_action")
        if ctx.competency in ("failure", "conflict") and not f.learning:
            g.append("no_learning")
    if ctx.profile in ("star", "situational", "intro") and f.personal_ratio is not None and f.personal_ratio < 0.15 and len(f.we_spans) >= 2:
        g.append("all_we")
    if ctx.profile in ("star", "situational") and dims["specificity"]["score"] < 35 and "no_example" not in g:
        g.append("vague")
    if ctx.profile == "technical" and ctx.concepts and len(f.concept_hits) < max(1, round(0.4 * len(ctx.concepts))):
        g.append("missing_concepts")
    if ctx.profile == "why_org" and f.generic_org and len(f.about_distinct) <= 1:
        g.append("generic_why")
    if ctx.profile == "why_org" and not f.org_hits:
        g.append("no_org")
    return g


def _highlights(f: Features) -> list[dict]:
    raw: list[tuple[str, Span, str]] = []
    for s in f.fillers:
        raw.append(("filler", s, "Filler word"))
    for s in f.hedges:
        raw.append(("hedge", s, "Hedging phrase"))
    for s in f.weak:
        raw.append(("weak", s, "Weak-ownership phrase"))
    for s in f.vague:
        raw.append(("vague", s, "Vague wording"))
    for s in f.generic_org:
        raw.append(("generic", s, "Generic phrase"))
    for s in f.numbers:
        raw.append(("number", s, "Number / measurable detail"))
    for comp, key in (("result", "star-r"), ("action", "star-a"), ("task", "star-t"), ("situation", "star-s")):
        for s in f.star[comp]["cues"]:
            raw.append((key, s, f"{STAR_LABELS[comp]} cue"))
    for canon, spans in list(f.jd_hits.items()) + list(f.concept_hits.items()):
        for s in spans:
            raw.append(("match", s, f"Matches “{canon}”"))
    for s in f.about_hits:
        raw.append(("match", s, "Echoes your pasted company text"))
    for s in f.i_verbs:
        raw.append(("verb", s, "Action verb"))
    for s in f.we_spans:
        raw.append(("we", s, "Group reference (we/our/us)"))
    order = {k: i for i, k in enumerate(HIGHLIGHT_PRIORITY)}
    raw.sort(key=lambda t: (order[t[0]], t[1].start))
    taken: list[tuple[int, int]] = []
    out = []
    for kind, sp, note in raw:
        if any(sp.start < e and s < sp.end for s, e in taken):
            continue
        taken.append((sp.start, sp.end))
        out.append({"start": sp.start, "end": sp.end, "kind": kind, "note": note})
    out.sort(key=lambda d: d["start"])
    return out


def _metrics(f: Features) -> dict:
    return {
        "words": f.wc,
        "sentences": len(f.sents),
        "avg_sentence_words": round(f.avg_sent, 1),
        "numbers": len(f.numbers),
        "personal_ratio": None if f.personal_ratio is None else round(f.personal_ratio, 2),
        "fillers": len(f.fillers),
        "hedges": len(f.hedges),
        "filler_per_100": round(f.filler_rate, 1),
        "hedge_per_100": round(f.hedge_rate, 1),
        "action_verbs": len(f.distinct_i_verbs),
        "wpm": f.wpm,
        "question_overlap": round(f.q_overlap, 2),
    }


# --------------------------------------------------------------------------- public API


def analyze(text: str, ctx: Context | dict | None = None) -> dict:
    """Score one answer. Returns a JSON-serialisable dict; see docs/scoring for the schema."""
    if not isinstance(ctx, Context):
        ctx = Context.from_dict(ctx)
    if ctx.profile not in PROFILES:
        raise ValueError(f"unknown profile {ctx.profile!r}; expected one of {PROFILES}")
    text = (text or "").replace("\r\n", "\n").strip()
    f = Features(text, ctx)
    rel_fn, str_fn, spec_fn = PROFILE_FUNCS[ctx.profile]
    dims = {
        "relevance": rel_fn(f),
        "structure": str_fn(f),
        "specificity": spec_fn(f),
        "concision": _concision(f),
        "confidence": _confidence(f, with_ownership=ctx.profile in ("star", "situational", "intro", "why_org")),
    }
    overall = round(sum(dims[d]["score"] * WEIGHTS[d] for d in DIMENSIONS))
    result = {
        "profile": ctx.profile,
        "text": text,
        "overall": overall,
        "dimensions": dims,
        "metrics": _metrics(f),
        "gaps": _gaps(f, dims),
        "highlights": _highlights(f),
        "star": {k: {"status": v["status"], "evidence": [s.to_dict() for s in v["cues"][:4]]} for k, v in f.star.items()} if ctx.profile == "star" else None,
        "jd": None,
        "concepts": None,
    }
    if ctx.jd_skills:
        result["jd"] = {"matched": sorted(f.jd_hits), "missing": [k for k in ctx.jd_skills if k not in f.jd_hits]}
    if ctx.concepts:
        result["concepts"] = {"matched": list(f.concept_hits), "missing": f.concept_missing}
    return result
