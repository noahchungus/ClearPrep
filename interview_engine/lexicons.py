"""Compiled lexicons/patterns built from the YAML content files."""
from __future__ import annotations

import functools
import re

from .loader import load_yaml


def _phrase_re(phrases: list[str], inflect: bool = False) -> re.Pattern:
    """Whole-word/phrase matcher, longest phrases first so 'you know what i mean' wins.

    inflect=True also accepts a plural/verb ending (s, es, ed, ing) on the last word, used for concept and
    skill matching so "collisions" matches "collision".
    """
    ordered = sorted({p.lower() for p in phrases}, key=len, reverse=True)
    tail = r"(?:s|es|ed|ing)?" if inflect else ""
    parts = [r"(?<![A-Za-z'’])" + re.escape(p).replace(r"\ ", r"\s+") + tail + r"(?![A-Za-z])" for p in ordered]
    return re.compile("|".join(parts), re.IGNORECASE)


class Lexicons:
    def __init__(self) -> None:
        lex = load_yaml("lexicons.yaml")
        cues = load_yaml("star_cues.yaml")
        self.raw = lex
        self.cues_raw = cues
        self.filler_re = _phrase_re(lex["filler_words"] + lex["filler_phrases"])
        self.hedge_re = _phrase_re(lex["hedges"])
        self.weak_re = _phrase_re(lex["weak_phrases"])
        self.vague_re = _phrase_re(lex["vague_phrases"])
        self.generic_org_re = _phrase_re(lex["generic_org_phrases"])
        self.premature_re = _phrase_re(lex["premature_topics"])
        self.lazy_re = _phrase_re(lex["lazy_questions"])
        self.stakeholder_re = _phrase_re(lex["stakeholder_words"])
        self.verification_re = _phrase_re(lex["verification_words"])
        self.sequence_re = _phrase_re(lex["sequence_words"])
        self.reasoning_re = _phrase_re(lex["reasoning_words"])
        self.example_re = _phrase_re(lex["example_words"])
        self.detail_re = _phrase_re(lex["detail_markers"])
        self.intro_present_re = _phrase_re(lex["intro_present"])
        self.intro_past_re = _phrase_re(lex["intro_past"])
        self.intro_future_re = _phrase_re(lex["intro_future"])
        self.conditional_i_re = re.compile(r"\bi(?:'d|’d| would|'ll|’ll| will| can| could| might)\b", re.IGNORECASE)
        self.strong_verbs = frozenset(v.lower() for v in lex["strong_verbs"])
        self.stative_verbs = frozenset(v.lower() for v in lex["stative_verbs"])
        self.hypothetical_re = _phrase_re(lex["hypothetical_markers"])
        self.irregular_past = frozenset(v.lower() for v in lex["irregular_past"])
        self.proper_noun_ignore = frozenset(lex["proper_noun_ignore"])
        self.star = {
            comp: {
                "zone": tuple(cues[comp]["zone"]),
                "res": [re.compile(p, re.IGNORECASE) for p in cues[comp]["patterns"]],
            }
            for comp in ("situation", "task", "action", "result")
        }
        self.learning_res = [re.compile(p, re.IGNORECASE) for p in cues["learning"]["patterns"]]
        self.number_word_re = re.compile(cues["number_words"][0] if isinstance(cues["number_words"], list) else cues["number_words"], re.IGNORECASE)
        self.first_person_i_re = re.compile(r"\b(?:i|i'm|i’m|i've|i’ve|i'd|i’d|i'll|i’ll|me|my|mine|myself)\b", re.IGNORECASE)
        self.first_person_we_re = re.compile(r"\b(?:we|we're|we’re|we've|we’ve|we'd|we’d|we'll|we’ll|us|our|ours|ourselves)\b", re.IGNORECASE)
        self.i_action_re = re.compile(r"\bi\s+(?:then\s+|also\s+|quickly\s+|immediately\s+|personally\s+|first\s+|finally\s+)?([a-z]+)\b", re.IGNORECASE)
        self.number_re = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?\s?(?:%|percent|k\b|x\b|m\b|million|thousand|hours?|hrs?|days?|weeks?|months?|minutes?|mins?|seconds?)?", re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def get_lexicons() -> Lexicons:
    return Lexicons()


def phrase_matcher(phrases: list[str], inflect: bool = False) -> re.Pattern:
    return _phrase_re(phrases, inflect)
