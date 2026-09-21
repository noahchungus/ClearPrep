"""Rule-based follow-up questions triggered by gaps the analyzer found.

Each rule is (gap code, profiles it applies to, question template, plain-English reason).
Rules are checked in priority order and at most one follow-up is asked per question, the way a
real interviewer would probe the weakest point first.
"""
from __future__ import annotations

RULES: list[dict] = [
    {"code": "too_long", "profiles": ("star", "situational", "technical", "why_org", "intro"),
     "text": "That was a lot of detail. Can you summarize it in 30 seconds: what happened, what you did, and the result?",
     "reason": "Your answer ran long. Interviewers value being able to give the short version on demand."},
    {"code": "too_short", "profiles": ("star", "situational", "technical", "why_org", "intro"),
     "text": "Can you tell me more about that? Walk me through what was happening and what you specifically did.",
     "reason": "Your answer was very short, so there was not enough to evaluate."},
    {"code": "no_example", "profiles": ("star",),
     "text": "Can you walk me through one specific example: when and where it happened, and who was involved?",
     "reason": "No real situation was detected; the answer stayed general."},
    {"code": "all_we", "profiles": ("star", "situational", "intro"),
     "text": "You said “we” quite a bit. What was your personal contribution?",
     "reason": "Almost every statement was about the group, so your own role was unclear."},
    {"code": "no_action", "profiles": ("star",),
     "text": "What specific steps did you personally take?",
     "reason": "No clear first-person actions were found."},
    {"code": "no_result", "profiles": ("star",),
     "text": "What was the outcome, and how did you measure it?",
     "reason": "No clear result was detected in the story."},
    {"code": "no_learning", "profiles": ("star",),
     "text": "What did you learn from that, and what would you do differently now?",
     "reason": "For this kind of question interviewers expect reflection, and none was found."},
    {"code": "vague", "profiles": ("star", "situational"),
     "text": "Can you be more specific? Give me a number or a concrete detail that shows what that looked like.",
     "reason": "The answer was heavy on vague wording and light on specifics."},
    {"code": "missing_concepts", "profiles": ("technical",),
     "text": "Where does “{concept}” fit into your answer?",
     "reason": "A key idea for this question was not mentioned."},
    {"code": "generic_why", "profiles": ("why_org",),
     "text": "What specifically about {org} draws you, beyond the role itself?",
     "reason": "The reasons given could apply to almost any employer."},
    {"code": "no_org", "profiles": ("why_org",),
     "text": "Why {org} rather than another organization in this space?",
     "reason": "You never named the organization, so the answer did not show a specific choice."},
]


def select_followup(analysis: dict, question: dict, profile: dict) -> dict | None:
    """Return the highest-priority follow-up for these gaps, or None."""
    gaps = set(analysis["gaps"])
    prof = analysis["profile"]
    for rule in RULES:
        if rule["code"] in gaps and prof in rule["profiles"]:
            text = rule["text"]
            if "{concept}" in text:
                missing = (analysis.get("concepts") or {}).get("missing") or []
                if not missing:
                    continue
                text = text.replace("{concept}", missing[0])
            text = text.replace("{org}", profile.get("org", "this organization"))
            return {"code": rule["code"], "text": text, "reason": rule["reason"]}
    return None
