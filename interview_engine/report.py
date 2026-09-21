"""End-of-session report, Markdown export and the compact record saved to the browser."""
from __future__ import annotations

from collections import defaultdict

from .analyzer import DIMENSIONS, STAR_LABELS
from .bank import competencies
from .coaching import lost_points, scaffold, strengths
from .session import analysis_for, question_at

LIMITATIONS = ("The scores measure structure and delivery signals (STAR components, numbers, ownership, hedging, length, pacing, keyword coverage). "
               "They cannot judge whether your story is impressive or whether a technical statement is factually correct. "
               "Written coaching from the AI coach, when it is on, can weigh content but can also be wrong. "
               "Treat all of it as a checklist for improvement, not a prediction of hiring outcomes.")

DIMENSION_LABELS = {"relevance": "Relevance", "structure": "Structure", "specificity": "Specificity", "concision": "Concision", "confidence": "Confidence & tone"}

# One concrete drill for the most common ways to lose points. Hand-written.
PRACTICE_TIPS = {
    "Situation": "Write a one-sentence opener for each of your 5 stories: when, where, who. Say it aloud until it takes under 10 seconds.",
    "Task": "For each story, finish the sentence “My responsibility was to…” so the goal is explicit.",
    "Action": "Rewrite each story’s middle as 3 numbered steps that each start with “I” plus a verb.",
    "Result": "End every story with: outcome + number + what you learned. If you have no number, estimate honestly (“about 10 hours saved”).",
    "Reflection": "For failure and conflict stories, add one closing sentence starting “Now I always…”.",
    "Numbers & measurable detail": "Go through your stories and add one number to each: people, hours, percent, dollars, grades, users.",
    "Named details (tools, places, courses)": "Name the tools, courses, teams and places in each story instead of “a project” or “a class”.",
    "Action verbs": "Replace “helped with / worked on” with a specific verb: built, organized, analyzed, negotiated, debugged.",
    "Concrete setting": "Start with a time marker (“Last semester in my…”) so the listener can picture where you were.",
    "Vague wording": "Circle “stuff / things / a lot” in your notes and replace each with the actual noun or number.",
    "Length": "Time yourself: 60 to 90 seconds for stories. Use a timer and cut background first, keep action and result.",
    "Sentence length": "Read your answer aloud. Wherever you run out of breath, put a period.",
    "Repetition": "Vary key words (“team” → “group”, “the project” → its name) or merge repeated points.",
    "Speaking pace": "Practice with a timer: aim for 110 to 170 words per minute and pause at each full stop.",
    "Ownership (I vs we)": "Keep “we” for context and switch to “I” for what YOU did. Ask “what was MY part?” for every sentence.",
    "Hedging": "Delete “I think / maybe / sort of” from your notes and say the sentence flat. It will feel blunt; that is confidence.",
    "Filler words": "Replace filler with a silent one-second pause. Record yourself once and count them.",
    "Ownership language": "Swap weak phrases (“was involved in”) for “I led / I built / I wrote” where it is true.",
    "Stays on the competency": "Use the question’s key word in your first sentence so the link is obvious.",
    "Answers the question asked": "Repeat the question’s key terms in your first sentence, then tell the story.",
    "Gives a real example": "Pick one real story before answering. General statements (“I always…”) are weaker than one specific event.",
    "Connects to the job description": "Underline three skills in the posting and prepare one honest example for each.",
    "Considers other people": "In situational answers, name who you would talk to first and why.",
    "Ordered steps": "Say “First… then… finally…” to make your plan easy to follow.",
    "Says what you would do": "Commit to actions: “I would first ask…, then I would…”.",
    "Follows through / verifies": "End situational answers with how you would check it worked and follow up.",
    "Explains the reasoning": "After each step add “because…” to show your thinking.",
    "Covers the key ideas": "Make a 1-page cheat sheet of core concepts for your field and explain each aloud in 3 sentences.",
    "Direct first sentence": "Practice one-sentence definitions for your top 10 concepts before adding detail.",
    "Explains why / trade-offs": "For every concept, prepare one trade-off (“the downside is…”).",
    "Gives an example": "For each concept, prepare a real example from your own project or class.",
    "Precise terminology": "Use the exact terms of your field rather than descriptions of them.",
    "Grounded in experience": "Connect concepts to a project: “I used this in my … project to …”.",
    "Gives actual reasons": "Use the frame “I’m drawn to X because Y”.",
    "Names the organization": "Say the organization’s name at least once so the reason sounds chosen, not generic.",
    "Connects to the role": "Tie your interest to a specific responsibility in the posting.",
    "Uses the company text you pasted": "Pick one detail from the organization’s own materials that genuinely resonates and cite it.",
    "Reason about the organization": "Find one organization-specific thing you honestly like and say why.",
    "Reason about the role": "Name one task in the role you look forward to, and why.",
    "Ties it to you": "Finish by connecting the reasons to your background or goals.",
    "Generic phrasing": "Delete “great company / fast-paced / innovative” and replace with one true, specific detail.",
    "Present: who you are now": "Open with one line: who you are now (year, school, focus).",
    "Past: relevant background": "Add two relevant experiences (project, job, class) in one sentence each.",
    "Future: why this role": "Close with where you are headed and why this role is the next step.",
    "Number of questions": "Prepare 4 questions; ask 2 to 3.",
    "Open-ended": "Start questions with How / What / Why to keep the conversation going.",
    "Refers to this org / role": "Mention the team, the role or something from their materials in each question.",
    "Forward-looking topics": "Ask about success in the role, learning and the team’s challenges.",
    "Too early for pay/perks": "Save salary, benefits and time-off questions for the recruiter or an offer.",
}


def _band(overall: int) -> tuple[str, str]:
    if overall >= 80:
        return "Strong", "Your answers show clear structure and confident delivery. Focus on polishing the few gaps below."
    if overall >= 65:
        return "Solid", "The basics are there. Fixing the top items below is the fastest route to a stronger interview."
    if overall >= 50:
        return "Developing", "There is a foundation to build on. Work on the top priorities below and re-run a session."
    return "Needs work", "Answers are missing key structure or specifics. Use the scaffolds below and practice the drills, then try again."


def build_report(state: dict) -> dict:
    rows, lost_agg = [], defaultdict(lambda: {"lost": 0.0, "count": 0, "dimension": "", "example": None})
    comp_scores = defaultdict(list)
    star_present = {k: [0, 0] for k in STAR_LABELS}
    metrics = defaultdict(list)
    matched_skills: set[str] = set()
    comps = competencies()
    skipped = 0
    for i, rec in enumerate(state["records"]):
        q = question_at(state, i)
        if rec["skipped"]:
            skipped += 1
            rows.append({"index": i + 1, "question": q["text"], "stage": q["stage"], "competency": comps.get(q["competency"], {}).get("label", q["competency"]), "skipped": True})
            continue
        a = analysis_for(state, i)
        fixes = lost_points(a)
        for f in fixes:
            agg = lost_agg[f["label"]]
            agg["lost"] += f["lost"]
            agg["count"] += 1
            agg["dimension"] = f["dimension"]
            if agg["example"] is None or f["lost"] > agg["example"]["lost"]:
                agg["example"] = {**f, "question": q["text"]}
        comp_label = comps.get(q["competency"], {}).get("label", q["competency"])
        comp_scores[comp_label].append(a["overall"])
        if a["star"]:
            for k, v in a["star"].items():
                star_present[k][1] += 1
                star_present[k][0] += v["status"] == "present"
        m = a["metrics"]
        for key in ("words", "numbers", "filler_per_100", "hedge_per_100", "action_verbs"):
            metrics[key].append(m[key])
        if m["personal_ratio"] is not None:
            metrics["personal_ratio"].append(m["personal_ratio"])
        if m["wpm"]:
            metrics["wpm"].append(m["wpm"])
        if a["jd"]:
            matched_skills.update(a["jd"]["matched"])
        fu = rec.get("followup")
        rows.append({
            "index": i + 1, "question": q["text"], "stage": q["stage"], "competency": comp_label, "skipped": False,
            "overall": a["overall"], "dims": {d: a["dimensions"][d]["score"] for d in DIMENSIONS}, "gaps": a["gaps"], "words": m["words"],
            "followup": {"text": fu["text"], "answered": bool(fu.get("answer")), "reason": fu["reason"]} if fu else None,
            "fixes": fixes[:3], "strengths": strengths(a, 2), "scaffold": scaffold(q, a),
        })
    answered = [r for r in rows if not r["skipped"]]
    n = len(answered)
    dims = {d: round(sum(r["dims"][d] for r in answered) / n) if n else 0 for d in DIMENSIONS}
    overall = round(sum(r["overall"] for r in answered) / n) if n else 0
    label, verdict = _band(overall) if n else ("No answers", "Answer at least one question to get a report.")

    priorities = []
    for lbl, agg in sorted(lost_agg.items(), key=lambda kv: -kv[1]["lost"])[:5]:
        priorities.append({"label": lbl, "dimension": agg["dimension"], "dimension_label": DIMENSION_LABELS.get(agg["dimension"], agg["dimension"]), "answers_affected": agg["count"],
                           "points_lost": round(agg["lost"]), "drill": PRACTICE_TIPS.get(lbl, agg["example"]["detail"]), "example": agg["example"]})

    def avg(key):
        v = metrics.get(key, [])
        return round(sum(v) / len(v), 1) if v else None

    delivery = {"avg_words": avg("words"), "avg_numbers": avg("numbers"), "filler_per_100": avg("filler_per_100"), "hedge_per_100": avg("hedge_per_100"),
                "personal_ratio": avg("personal_ratio"), "avg_wpm": avg("wpm"), "avg_action_verbs": avg("action_verbs")}
    prof = state["profile"]
    jd_skills = list(prof["jd_skills"])
    return {
        "org": prof["org"], "role": prof["role"], "family": prof["family_label"], "interview_type": prof["interview_type"], "date": state["created"], "session_id": state["id"],
        "planned": len(state["plan"]), "answered": n, "skipped": skipped, "unanswered": len(state["plan"]) - len(state["records"]),
        "overall": overall, "band": label, "verdict": verdict, "dims": dims, "dim_labels": DIMENSION_LABELS,
        "rows": rows, "priorities": priorities, "delivery": delivery,
        "star": {STAR_LABELS[k]: (round(100 * v[0] / v[1]) if v[1] else None) for k, v in star_present.items()},
        "competencies": sorted([{"label": k, "avg": round(sum(v) / len(v)), "n": len(v)} for k, v in comp_scores.items()], key=lambda r: r["avg"]),
        "jd": {"skills": jd_skills, "mentioned": [s for s in jd_skills if s in matched_skills], "never": [s for s in jd_skills if s not in matched_skills]} if jd_skills else None,
        "limitations": LIMITATIONS,
    }


def history_record(state: dict, report: dict | None = None) -> dict:
    """Compact record stored in the user's browser (localStorage) for the progress dashboard."""
    rep = report or build_report(state)
    return {
        "id": state["id"], "date": state["created"], "org": rep["org"], "role": rep["role"], "type": rep["interview_type"], "family": rep["family"],
        "overall": rep["overall"], "dims": rep["dims"], "answered": rep["answered"], "delivery": rep["delivery"],
        "priorities": [p["label"] for p in rep["priorities"][:3]],
        "questions": [{"q": r["question"], "competency": r["competency"], "overall": r["overall"]} for r in rep["rows"] if not r["skipped"]],
    }


def to_markdown(state: dict, report: dict | None = None) -> str:
    rep = report or build_report(state)
    L = [f"# Mock interview report: {rep['role']} at {rep['org']}", "",
         f"*Date: {rep['date']} · Type: {rep['interview_type']} · Role family: {rep['family']} · Answered {rep['answered']} of {rep['planned']} prompts*", "",
         f"## Overall: {rep['overall']}/100 ({rep['band']})", "", rep["verdict"], "",
         "| Dimension | Score |", "|---|---|"]
    L += [f"| {DIMENSION_LABELS[d]} | {rep['dims'][d]} |" for d in DIMENSIONS]
    if rep["priorities"]:
        L += ["", "## Top things to fix", ""]
        for p in rep["priorities"]:
            L.append(f"- **{p['label']}** ({p['dimension_label']}, affected {p['answers_affected']} answer(s)): {p['drill']}")
    if rep["jd"]:
        L += ["", "## Job description coverage", "",
              f"Mentioned: {', '.join(rep['jd']['mentioned']) or 'none'}", f"Never mentioned: {', '.join(rep['jd']['never']) or 'none'}"]
    L += ["", "## Question by question", ""]
    for r in rep["rows"]:
        if r["skipped"]:
            L += [f"### {r['index']}. {r['question']}", "*Skipped*", ""]
            continue
        L += [f"### {r['index']}. {r['question']}", f"Score **{r['overall']}** ({r['competency']}) · " + " · ".join(f"{DIMENSION_LABELS[d]} {r['dims'][d]}" for d in DIMENSIONS), ""]
        if r["followup"]:
            L.append(f"*Follow-up asked:* {r['followup']['text']}")
        for f in r["fixes"]:
            L.append(f"- Fix: **{f['label']}**: {f['detail']}")
        L.append("")
    L += ["---", f"*{rep['limitations']}*", ""]
    return "\n".join(L)
