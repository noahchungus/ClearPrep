"""Small, dependency-free text utilities shared by the whole engine.

Everything here is deterministic and works on the original string offsets so the
UI can highlight evidence directly in the user's own words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*|\d+(?:[.,]\d+)*")
SENT_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its just me more
    most my no nor not of off on once only or other our ours out over own same she should so some
    such than that the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your
    yours yourself really get got go went one two""".split()
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


def words(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]


def word_count(text: str) -> int:
    return sum(1 for m in WORD_RE.finditer(text) if m.group()[0].isalpha() or m.group()[0].isdigit())


def sentences(text: str) -> list[Span]:
    """Split into sentences, keeping offsets into `text`."""
    out: list[Span] = []
    pos = 0
    for m in SENT_END_RE.finditer(text):
        seg = text[pos : m.start()]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            out.append(Span(pos + lead, pos + len(seg.rstrip()), seg.strip()))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        out.append(Span(pos + lead, pos + len(tail.rstrip()), tail.strip()))
    return out


def stem(word: str) -> str:
    """Very light suffix stripping. Not linguistically perfect, but stable and explainable."""
    w = word.lower().strip("'’-").replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf, repl in (("ies", "y"), ("ations", ""), ("ation", ""), ("ings", ""), ("ing", ""),
                      ("edly", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + repl
    return w


def content_stems(text: str) -> set[str]:
    return {stem(w.text) for w in words(text) if w.text.lower() not in STOPWORDS and len(w.text) > 2 and w.text[0].isalpha()}


def content_tokens(text: str) -> list[str]:
    return [w.text.lower() for w in words(text) if w.text.lower() not in STOPWORDS and len(w.text) > 2 and w.text[0].isalpha()]


def snippet(text: str, start: int, end: int, radius: int = 60) -> str:
    """A short readable excerpt around a span (used as evidence in the UI)."""
    a, b = max(0, start - radius), min(len(text), end + radius)
    s = text[a:b].replace("\n", " ").strip()
    return ("…" if a > 0 else "") + s + ("…" if b < len(text) else "")
