"""Interview configuration, question planning and the interview state machine.

The whole session is one plain, JSON-serialisable dict (`state`) and every function here is a pure
function `state -> new state`. There is no server-side storage: the web layer round-trips the state
through the browser, which is what lets this run on stateless free hosts or, later, inside Pyodide.

Flow:  opener -> main questions (each may trigger one follow-up) -> "any questions for us?" -> done
Modes: asking -> [followup] -> feedback -> asking ... -> done
"""
from __future__ import annotations

import copy
import random
from datetime import date

from .analyzer import Context, analyze
from .bank import Question, competencies, fill, load_bank
from .followups import select_followup
from .personalize import extract_about_keywords, extract_duties, extract_skills, skill_questions
from .roles import families, infer_family, seniority_warning
from .text import word_count

STATE_VERSION = 1
INTERVIEW_TYPES = ("mixed", "behavioral", "situational", "technical")
SENIORITIES = ("internship", "entry")
DIFFICULTIES = ("mixed", "easy", "medium", "hard")
LIMITS = {"org": 80, "role": 100, "jd": 12000, "about": 6000, "answer": 4000, "n_min": 3, "n_max": 15}
DIFFICULTY_TARGET = {"easy": 1, "medium": 2, "hard": 3}
AI_EXCERPT = {"jd": 3000, "about": 2500}  # how much pasted text the AI coach gets to read
TIME_LIMITS = (0, 60, 90, 120, 180)  # seconds per answer; 0 = no limit (pressure mode when set)
ASK_US = {"id": "ask-us", "text": "That's most of what I wanted to cover. Do you have any questions for us?", "type": "closer", "profile": "ask_us",
          "competency": "communication", "families": ["general"], "difficulty": 1, "strong_answer": [
              "Asks two to four open-ended questions about the work, the team and how success is measured.",
              "Shows you read about the role or organization.",
              "Saves salary, benefits and time-off questions for later stages."], "concepts": []}


class StateError(ValueError):
    """The requested action is not valid in the session's current mode."""


# ---------------------------------------------------------------- configuration

def normalize_config(raw: dict) -> tuple[dict, dict[str, str]]:
    """Validate/clean the setup form. Returns (config, errors) where errors maps field -> message."""
    errors: dict[str, str] = {}
    cfg: dict = {}
    org = " ".join(str(raw.get("org", "")).split())
    role = " ".join(str(raw.get("role", "")).split())
    if not org:
        errors["org"] = "Enter the organization you are interviewing with."
    elif len(org) > LIMITS["org"]:
        errors["org"] = f"Keep the organization name under {LIMITS['org']} characters."
    if not role:
        errors["role"] = "Enter the position title."
    elif len(role) > LIMITS["role"]:
        errors["role"] = f"Keep the position title under {LIMITS['role']} characters."
    cfg["org"], cfg["role"] = org, role

    for key, allowed, default in (("interview_type", INTERVIEW_TYPES, "mixed"), ("seniority", SENIORITIES, "internship"), ("difficulty", DIFFICULTIES, "mixed")):
        val = str(raw.get(key) or default).lower()
        if val not in allowed:
            errors[key] = f"Choose one of: {', '.join(allowed)}."
        cfg[key] = val if val in allowed else default

    try:
        n = int(raw.get("n") or 6)
    except (TypeError, ValueError):
        n = 6
        errors["n"] = "Number of questions must be a whole number."
    if not LIMITS["n_min"] <= n <= LIMITS["n_max"]:
        errors["n"] = f"Choose between {LIMITS['n_min']} and {LIMITS['n_max']} questions."
        n = max(LIMITS["n_min"], min(LIMITS["n_max"], n))
    cfg["n"] = n

    warnings: list[str] = []
    for key, limit in (("jd_text", LIMITS["jd"]), ("about_text", LIMITS["about"])):
        val = str(raw.get(key) or "").strip()
        if len(val) > limit:
            val = val[:limit]
            warnings.append(f"{'Job description' if key == 'jd_text' else 'About-company text'} was longer than {limit:,} characters and was shortened.")
        cfg[key] = val
    if cfg["jd_text"] and word_count(cfg["jd_text"]) < 15:
        warnings.append("The job description is very short, so few skills could be extracted. Paste the full posting for better personalization.")

    family = str(raw.get("family") or "auto").lower()
    if family != "auto" and family not in families():
        errors["family"] = "Unknown role family."
        family = "auto"
    cfg["family"] = family

    d = str(raw.get("interview_date") or "").strip()
    cfg["interview_date"] = ""
    if d:
        try:
            cfg["interview_date"] = date.fromisoformat(d).isoformat()
        except ValueError:
            errors["interview_date"] = "Use a valid date."
    cfg["followups"] = str(raw.get("followups", "1")).lower() not in ("0", "false", "off", "no")
    try:
        tl = int(raw.get("time_limit") or 0)
    except (TypeError, ValueError):
        tl = 0
    if tl not in TIME_LIMITS:
        errors["time_limit"] = "Choose a time limit from the list."
        tl = 0
    cfg["time_limit"] = tl
    cfg["ai"] = str(raw.get("ai", "1")).lower() not in ("0", "false", "off", "no")
    cfg["seed"] = raw.get("seed")
    cfg["_warnings"] = warnings
    return cfg, errors


def build_profile(cfg: dict) -> dict:
    """Everything derived from the config: role family, JD skills, about-text keywords, warnings."""
    warnings = list(cfg.get("_warnings", []))
    if cfg["family"] == "auto":
        guess = infer_family(cfg["role"], cfg["jd_text"])
        family, source = guess.family, guess.source
    else:
        family, source = cfg["family"], "user"
    skills = extract_skills(cfg["jd_text"], family, limit=10)
    if family == "general" and cfg["family"] == "auto":
        warnings.append("Could not match the title to a role family, so the general question set is used. Pick a role family on the setup page for role-specific technical questions.")
    if (w := seniority_warning(cfg["role"])):
        warnings.append(w)
    return {
        "org": cfg["org"], "role": cfg["role"], "family": family, "family_label": families()[family]["label"], "family_source": source,
        "interview_type": cfg["interview_type"], "seniority": cfg["seniority"], "difficulty": cfg["difficulty"], "n": cfg["n"],
        "interview_date": cfg.get("interview_date", ""), "followups": cfg.get("followups", True), "time_limit": cfg.get("time_limit", 0),
        "skills": [{"name": s["name"], "kind": s["kind"], "count": s["count"], "in_requirements": s["in_requirements"]} for s in skills],
        "jd_skills": {s["name"]: s["aliases"] for s in skills},
        "about_keywords": extract_about_keywords(cfg["about_text"], cfg["org"]),
        "duties": extract_duties(cfg["jd_text"]),
        "has_jd": bool(cfg["jd_text"]), "has_about": bool(cfg["about_text"]),
        "ai": bool(cfg.get("ai", True)),
        "jd_excerpt": cfg["jd_text"][:AI_EXCERPT["jd"]] if cfg.get("ai", True) else "",
        "about_excerpt": cfg["about_text"][:AI_EXCERPT["about"]] if cfg.get("ai", True) else "",
        "warnings": warnings,
    }


# ---------------------------------------------------------------- planning

def _has_family_technical(family: str, bank) -> bool:
    return any(q.type == "technical" and family in q.families for q in bank)


def _weight(q: Question, profile: dict, counts: dict[str, int]) -> float:
    w = 1.0
    target = DIFFICULTY_TARGET.get(profile["difficulty"])
    if target:
        w *= {0: 4.0, 1: 2.0}.get(abs(q.difficulty - target), 1.0)
    elif profile["seniority"] == "internship" and q.difficulty == 3:
        w *= 0.5
    if profile["family"] in q.families and "general" not in q.families:
        w *= 2.0
    skills = profile["jd_skills"]
    if skills:
        wanted = {n.lower() for n in skills} | {a.lower() for al in skills.values() for a in al}
        if any(a.lower() in wanted for g in q.concepts for a in g):
            w *= 2.5
    w /= 1 + 2 * counts.get(q.competency, 0)
    return w


def _sample(pool: list[Question], k: int, rng: random.Random, profile: dict, counts: dict[str, int], chosen_ids: set[str]) -> list[Question]:
    out: list[Question] = []
    pool = [q for q in pool if q.id not in chosen_ids]
    while pool and len(out) < k:
        q = rng.choices(pool, weights=[_weight(x, profile, counts) for x in pool], k=1)[0]
        out.append(q)
        chosen_ids.add(q.id)
        counts[q.competency] = counts.get(q.competency, 0) + 1
        pool.remove(q)
    return out


def plan_questions(profile: dict, seed: int | None = None) -> list[dict]:
    """Choose the ordered list of questions. Each plan item is {"id": ...} or {"id": ..., "gen": {...}} for generated ones."""
    rng = random.Random(seed)
    bank = load_bank()
    fam, typ, n = profile["family"], profile["interview_type"], profile["n"]
    ok_family = lambda q: fam in q.families or "general" in q.families  # noqa: E731
    counts: dict[str, int] = {}
    chosen: set[str] = set()

    opener = rng.choice(["gen-open-01", "gen-open-01", "gen-open-02"])
    main: list[Question] = []
    forced: list[Question] = []
    if typ != "technical":
        forced = [next(q for q in bank if q.id == "gen-mot-01")]
        chosen.add("gen-mot-01")

    skill_qs: list[dict] = []
    if typ != "situational" and profile["skills"]:
        ranked = sorted([s for s in profile["skills"]], key=lambda s: (s["kind"] == "soft", -s["count"]))
        hits = [{"name": s["name"], "kind": s["kind"]} for s in ranked]
        skill_qs = skill_questions(hits, limit=min(2, max(1, n // 4)))

    remaining = n - len(forced) - len(skill_qs)
    behavioral = [q for q in bank if q.type in ("behavioral", "motivation") and ok_family(q) and q.id != "gen-mot-01"]
    situational = [q for q in bank if q.type == "situational" and ok_family(q)]
    technical = [q for q in bank if q.type == "technical" and fam in q.families]
    fam_beh = [q for q in behavioral if fam in q.families and "general" not in q.families]

    if typ == "behavioral":
        quotas = [(behavioral, remaining)]
    elif typ == "situational":
        quotas = [(situational, remaining)]
    elif typ == "technical":
        if _has_family_technical(fam, bank):
            quotas = [(technical, remaining), (fam_beh, remaining)]  # fam_beh only tops up if technical runs short
        else:
            quotas = [(behavioral, remaining)]
    else:  # mixed
        k_tech = round(0.3 * remaining) if technical else 0
        k_sit = round(0.2 * remaining)
        quotas = [(technical, k_tech), (situational, k_sit), (behavioral, remaining - k_tech - k_sit)]

    picked: list[Question] = []
    for pool, k in quotas:
        need = min(k, remaining - len(picked)) if typ in ("technical",) else k
        if need > 0:
            picked += _sample(pool, need, rng, profile, counts, chosen)
    if len(picked) < remaining:  # top up if a pool ran out
        picked += _sample(behavioral + situational, remaining - len(picked), rng, profile, counts, chosen)
    main = picked

    order = list(main)
    rng.shuffle(order)
    order.sort(key=lambda q: q.difficulty)  # warm up, then ramp
    plan: list[dict] = [{"id": opener}]
    plan += [{"id": q.id} for q in forced]
    gens = list(skill_qs)
    # interleave generated skill questions into the middle of the main block
    items = [{"id": q.id} for q in order]
    for g in gens:
        items.insert(min(len(items), max(1, len(items) // 2)), {"id": g["id"], "gen": g})
    plan += items
    plan.append({"id": ASK_US["id"]})
    return plan


def start_session(config: dict, seed: int | None = None, today: str | None = None) -> dict:
    """Create a new session state from a validated config."""
    profile = build_profile(config)
    if seed is None:
        seed = config.get("seed")
    if seed is None:
        seed = random.SystemRandom().randrange(1, 10**9)
    seed = int(seed)
    plan = plan_questions(profile, seed)
    return {
        "v": STATE_VERSION,
        "id": f"s{seed}",
        "created": today or date.today().isoformat(),
        "profile": profile,
        "plan": plan,
        "cursor": 0,
        "mode": "asking",
        "records": [],
        "rephrased": [],
    }


# ---------------------------------------------------------------- hydrating questions and contexts

def question_at(state: dict, index: int) -> dict:
    item = state["plan"][index]
    prof = state["profile"]
    if item["id"] == ASK_US["id"]:
        q = dict(ASK_US)
        q["stage"] = "ask_us"
    elif "gen" in item:
        q = dict(item["gen"])
        q["stage"] = "main"
    else:
        q = next(x for x in load_bank() if x.id == item["id"]).to_dict()
        q["stage"] = "opener" if q["type"] == "opener" else "main"
    q["text"] = fill(q["text"], prof["org"], prof["role"])
    q["_org"], q["_role"] = prof["org"], prof["role"]
    q["index"] = index
    return q


def build_context(q: dict, profile: dict, duration: float | None = None, spoken: bool = False) -> Context:
    comp = competencies().get(q["competency"], {})
    return Context(
        question_text=q["text"], profile=q["profile"], competency=q["competency"], competency_cues=[str(c) for c in comp.get("cues", [])],
        concepts=q.get("concepts", []), org=profile["org"], role=profile["role"], jd_skills=profile["jd_skills"],
        about_keywords=profile["about_keywords"], duration_seconds=duration, spoken=spoken,
    )


def full_text(record: dict) -> str:
    fu = record.get("followup")
    if fu and fu.get("answer"):
        return record["answer"] + "\n\n" + fu["answer"]
    return record["answer"]


def analysis_for(state: dict, index: int) -> dict | None:
    """Recompute the full (deterministic) analysis for an answered question."""
    if index >= len(state["records"]) or state["records"][index]["skipped"]:
        return None
    rec = state["records"][index]
    q = question_at(state, index)
    return analyze(full_text(rec), build_context(q, state["profile"], rec.get("duration"), rec.get("spoken", False)))


def _compact(analysis: dict) -> dict:
    return {"overall": analysis["overall"], "dims": {k: v["score"] for k, v in analysis["dimensions"].items()}, "gaps": analysis["gaps"], "words": analysis["metrics"]["words"]}


# ---------------------------------------------------------------- state machine

def _require(state: dict, *modes: str) -> None:
    if state.get("mode") not in modes:
        raise StateError(f"That action isn't available right now (session is in '{state.get('mode')}' mode).")


def _clean_answer(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if word_count(text) < 1:
        raise ValueError("Type or dictate an answer first. If you'd rather not answer, use Skip.")
    if len(text) > LIMITS["answer"]:
        raise ValueError(f"Answers are limited to {LIMITS['answer']:,} characters. Aim for the 1–2 minute version.")
    return text


def submit_answer(state: dict, text: str, duration_seconds: float | None = None, spoken: bool = False, followup_selector=None) -> dict:
    """Record an answer (or a follow-up answer). Returns a NEW state.

    followup_selector(analysis, question, profile, answer_text) -> {"code","text","reason",...} | None lets the caller
    decide the follow-up (the web layer plugs in the AI coach). Default: the built-in rules in followups.py.
    """
    _require(state, "asking", "followup")
    text = _clean_answer(text)
    st = copy.deepcopy(state)
    dur = float(duration_seconds) if duration_seconds and duration_seconds > 0 else None
    if st["mode"] == "asking":
        q = question_at(st, st["cursor"])
        rec = {"qid": q["id"], "answer": text, "followup": None, "skipped": False, "duration": dur, "spoken": bool(spoken and dur)}
        analysis = analyze(text, build_context(q, st["profile"], dur, spoken))
        selector = followup_selector or (lambda a, qq, pr, t: select_followup(a, qq, pr))
        fu = selector(analysis, q, st["profile"], text) if (st["profile"]["followups"] and q["stage"] != "ask_us") else None
        if fu:
            rec["followup"] = {**fu, "answer": None, "skipped": False}
            st["mode"] = "followup"
        else:
            rec["scores"] = _compact(analysis)
            st["mode"] = "feedback"
        st["records"].append(rec)
    else:  # followup
        rec = st["records"][-1]
        rec["followup"]["answer"] = text
        if dur:
            rec["duration"] = (rec.get("duration") or 0) + dur
            rec["spoken"] = rec.get("spoken", False) and bool(spoken)
        st["mode"] = "feedback"
        rec["scores"] = _compact(analysis_for_record(st, len(st["records"]) - 1, rec))
    return st


def analysis_for_record(state: dict, index: int, rec: dict) -> dict:
    q = question_at(state, index)
    return analyze(full_text(rec), build_context(q, state["profile"], rec.get("duration"), rec.get("spoken", False)))


def skip_followup(state: dict) -> dict:
    _require(state, "followup")
    st = copy.deepcopy(state)
    rec = st["records"][-1]
    rec["followup"]["skipped"] = True
    rec["scores"] = _compact(analysis_for_record(st, len(st["records"]) - 1, rec))
    st["mode"] = "feedback"
    return st


def skip_question(state: dict) -> dict:
    """Skip the current question. It is recorded as skipped and does not count toward scores."""
    _require(state, "asking")
    st = copy.deepcopy(state)
    q = question_at(st, st["cursor"])
    st["records"].append({"qid": q["id"], "answer": "", "followup": None, "skipped": True, "duration": None, "spoken": False})
    return _advance(st)


def next_question(state: dict) -> dict:
    _require(state, "feedback")
    return _advance(copy.deepcopy(state))


def _advance(st: dict) -> dict:
    st["cursor"] += 1
    st["mode"] = "done" if st["cursor"] >= len(st["plan"]) else "asking"
    return st


def finish_early(state: dict) -> dict:
    """End the session now. Unanswered questions are left out of the report."""
    st = copy.deepcopy(state)
    if st["mode"] == "followup":
        return finish_early(skip_followup(st))
    st["mode"] = "done"
    return st


def rephrase(state: dict) -> str:
    """A different way to think about the current question (hand-written per competency). Read-only."""
    q = question_at(state, state["cursor"])
    if q.get("rephrase"):
        return q["rephrase"]
    return competencies().get(q["competency"], {}).get("rephrase", "Take a moment to think of one specific example, then tell me what you did and what happened.")


def view(state: dict) -> dict:
    """What the UI needs to render the current step."""
    mode = state["mode"]
    total = len(state["plan"])
    idx = min(state["cursor"], total - 1)
    out = {"mode": mode, "index": idx, "total": total, "profile": state["profile"], "answered": sum(1 for r in state["records"] if not r["skipped"])}
    if mode in ("asking", "followup", "feedback"):
        out["question"] = question_at(state, idx)
    if mode == "followup":
        out["followup"] = state["records"][-1]["followup"]
        out["first_answer"] = state["records"][-1]["answer"]
    if mode == "feedback":
        out["record"] = state["records"][idx]
        out["analysis"] = analysis_for(state, idx)
        out["is_last"] = idx >= total - 1
    return out
