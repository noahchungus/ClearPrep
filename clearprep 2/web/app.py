"""Thin, stateless FastAPI layer over interview_engine.

The server keeps NO session data. The interview state is a JSON blob that round-trips through a hidden
form field, and history lives in the user's browser (localStorage). Every route is a pure function of
its request, which is why this can be hosted on a free tier with an ephemeral disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from interview_engine import session as sess
from interview_engine.analyzer import Context, analyze
from interview_engine.bank import load_bank
from interview_engine.coaching import competency_info, exemplar_for, full_scaffold, lost_points, scaffold, strengths
from interview_engine.explain import scoring_overview
from interview_engine.followups import RULES, select_followup
from interview_engine.personalize import extract_about_keywords, extract_duties, extract_keywords, extract_skills, questions_to_ask
from interview_engine.progress import build_progress
from interview_engine.report import DIMENSION_LABELS, LIMITATIONS, build_report, history_record, to_markdown
from interview_engine.roles import families, infer_family

from . import ai

BASE = Path(__file__).parent
MAX_STATE_BYTES = 600_000

app = FastAPI(title="ClearPrep", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


# ---------------------------------------------------------------- template helpers

def highlight(text: str, highlights: list[dict]) -> Markup:
    """Render the user's own answer with analyzer evidence spans wrapped in <mark>."""
    out, pos = [], 0
    for h in highlights:
        if h["start"] < pos:
            continue
        out.append(escape(text[pos : h["start"]]))
        out.append(Markup('<mark class="hl hl-{}" title="{}">{}</mark>').format(h["kind"], h["note"], text[h["start"] : h["end"]]))
        pos = h["end"]
    out.append(escape(text[pos:]))
    return Markup("").join(out)


def placeholders(text: str) -> Markup:
    """Escape text and highlight [bracketed placeholders] the coach left for the candidate to fill in."""
    import re

    parts = re.split(r"(\[[^\]\n]{1,80}\])", str(text or ""))
    return Markup("").join(Markup('<span class="ph">{}</span>').format(p) if i % 2 else escape(p) for i, p in enumerate(parts))


def band_class(score: int) -> str:
    return "good" if score >= 75 else "ok" if score >= 55 else "low"


def fmt_mmss(seconds) -> str:
    if not seconds:
        return ""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def static_v(name: str) -> int:
    """Version tag for a static file (its mtime), so browsers re-download it whenever it changes."""
    try:
        return int((BASE / "static" / name).stat().st_mtime)
    except OSError:
        return 0


templates.env.globals["static_v"] = static_v
templates.env.filters["highlight"] = highlight
templates.env.filters["placeholders"] = placeholders
templates.env.globals["ai_available"] = lambda: ai.get_coach().available()
templates.env.filters["mmss"] = fmt_mmss
templates.env.globals.update(band_class=band_class, DIM_LABELS=DIMENSION_LABELS, DIMS=list(DIMENSION_LABELS), LIMITATIONS=LIMITATIONS)


def render(request: Request, name: str, ctx: dict | None = None, status: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(request, name, ctx or {}, status_code=status)


def dump_state(state: dict) -> str:
    return json.dumps(state, separators=(",", ":"))


def load_state(raw: str) -> dict:
    if len(raw.encode("utf-8")) > MAX_STATE_BYTES:
        raise HTTPException(413, "Session data is too large.")
    try:
        state = json.loads(raw)
        assert isinstance(state, dict) and state.get("v") == sess.STATE_VERSION
        for key in ("profile", "plan", "cursor", "mode", "records"):
            assert key in state
        assert 0 <= int(state["cursor"]) <= len(state["plan"]) <= 40
        sess.question_at(state, min(state["cursor"], len(state["plan"]) - 1))  # hydrates; fails on unknown ids
        return state
    except (ValueError, AssertionError, KeyError, StopIteration, TypeError, IndexError):
        raise HTTPException(400, "That session could not be read. Start a new interview from the home page.") from None


def stage_context(state: dict, error: str | None = None, hint: str | None = None) -> dict:
    view = sess.view(state)
    ctx = {"state_json": dump_state(state), "view": view, "error": error, "hint": hint, "ai_on": bool(state["profile"].get("ai"))}
    if view["mode"] == "feedback":
        q, a = view["question"], view["analysis"]
        ctx.update(
            fixes=lost_points(a)[:4], strengths=strengths(a, 3), scaffold=scaffold(q, a), full_scaffold=full_scaffold(q) if q["profile"] == "star" else [], exemplar=exemplar_for(q),
            comp=competency_info(q["competency"]), a=a, strong_points=q.get("strong_answer", []), dims={d: v["score"] for d, v in a["dimensions"].items()},
        )
    return ctx


def load_eval_results() -> dict | None:
    """Measured accuracy from the evaluation harness (eval/results.json), if it has been generated."""
    path = BASE.parent / "eval" / "results.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_float(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def client_id(request: Request) -> str:
    return request.client.host if request.client else "local"


def ai_followup_selector(cid: str):
    """Follow-up chooser: the AI coach when it is on and reachable, otherwise the built-in rules."""

    def select(analysis, q, profile, text):
        coach = ai.get_coach()
        if profile.get("ai") and coach.available():
            try:
                return coach.followup(q, analysis, profile, text, cid)  # a dict, or None when the coach decides no follow-up is needed
            except ai.AIError:
                pass  # fall through to the rules; the user still gets a follow-up
        return select_followup(analysis, q, profile)

    return select


# ---------------------------------------------------------------- pages

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render(request, "home.html")


@app.get("/practice", response_class=HTMLResponse)
def practice(request: Request):
    return render(request, "setup.html", {"errors": {}, "form": {}, "families": families(), "types": sess.INTERVIEW_TYPES})


def _setup_form(fields: dict) -> dict:
    return {k: fields.get(k, "") for k in ("org", "role", "family", "interview_date", "jd_text", "about_text", "interview_type", "seniority", "n", "difficulty", "time_limit", "followups", "ai")}


@app.post("/interview/start", response_class=HTMLResponse)
def start(request: Request, org: Annotated[str, Form()] = "", role: Annotated[str, Form()] = "", family: Annotated[str, Form()] = "auto",
          interview_date: Annotated[str, Form()] = "", jd_text: Annotated[str, Form()] = "", about_text: Annotated[str, Form()] = "",
          interview_type: Annotated[str, Form()] = "mixed", seniority: Annotated[str, Form()] = "internship", n: Annotated[str, Form()] = "6",
          difficulty: Annotated[str, Form()] = "mixed", time_limit: Annotated[str, Form()] = "0", followups: Annotated[str, Form()] = "0",
          ai_coach: Annotated[str, Form()] = "0"):
    raw = dict(org=org, role=role, family=family, interview_date=interview_date, jd_text=jd_text, about_text=about_text, interview_type=interview_type,
               seniority=seniority, n=n, difficulty=difficulty, time_limit=time_limit, followups=followups, ai=ai_coach)
    cfg, errors = sess.normalize_config(raw)
    if errors:
        return render(request, "setup.html", {"errors": errors, "form": _setup_form(raw), "families": families(), "types": sess.INTERVIEW_TYPES}, status=422)
    state = sess.start_session(cfg)
    return render(request, "interview.html", stage_context(state))


def _stage(request: Request, state: dict, error: str | None = None, hint: str | None = None):
    return render(request, "_stage.html", stage_context(state, error, hint))


@app.post("/interview/answer", response_class=HTMLResponse)
def answer(request: Request, state: Annotated[str, Form()] = "", answer: Annotated[str, Form()] = "", duration: Annotated[str, Form()] = "", spoken: Annotated[str, Form()] = ""):
    st = load_state(state)
    try:
        new = sess.submit_answer(st, answer, parse_float(duration), spoken == "1", followup_selector=ai_followup_selector(client_id(request)))
    except ValueError as e:  # empty / too long / wrong mode
        return _stage(request, st, error=str(e))
    return _stage(request, new)


@app.post("/interview/skip", response_class=HTMLResponse)
def skip(request: Request, state: Annotated[str, Form()] = ""):
    st = load_state(state)
    try:
        return _stage(request, sess.skip_question(st))
    except sess.StateError as e:
        return _stage(request, st, error=str(e))


@app.post("/interview/skip-followup", response_class=HTMLResponse)
def skip_followup(request: Request, state: Annotated[str, Form()] = ""):
    st = load_state(state)
    try:
        return _stage(request, sess.skip_followup(st))
    except sess.StateError as e:
        return _stage(request, st, error=str(e))


@app.post("/interview/next", response_class=HTMLResponse)
def nxt(request: Request, state: Annotated[str, Form()] = ""):
    st = load_state(state)
    try:
        return _stage(request, sess.next_question(st))
    except sess.StateError as e:
        return _stage(request, st, error=str(e))


@app.post("/interview/rephrase", response_class=HTMLResponse)
def rephrase(request: Request, state: Annotated[str, Form()] = ""):
    st = load_state(state)
    return _stage(request, st, hint=sess.rephrase(st))


@app.post("/interview/finish", response_class=HTMLResponse)
def finish(request: Request, state: Annotated[str, Form()] = ""):
    return _stage(request, sess.finish_early(load_state(state)))


@app.post("/interview/coach", response_class=HTMLResponse)
def interview_coach(request: Request, state: Annotated[str, Form()] = ""):
    """Written AI coaching for the answer just scored. Loaded after the fast rule-based feedback is already on screen."""
    st = load_state(state)
    view = sess.view(st)
    prof = st["profile"]
    if view["mode"] != "feedback" or not prof.get("ai"):
        return HTMLResponse("")
    coach = ai.get_coach()
    if not coach.available():
        return render(request, "_coach.html", {"status": "unavailable", "message": "The AI coach isn't set up on this server yet."})
    try:
        data = coach.coach(view["question"], view["analysis"], prof, view["record"], client_id(request))
    except ai.AIError as e:
        return render(request, "_coach.html", {"status": "error", "message": e.public, "state_json": dump_state(st)})
    return render(request, "_coach.html", {"status": "ok", "c": data})


@app.post("/report/coach", response_class=HTMLResponse)
def report_coach(request: Request, state: Annotated[str, Form()] = ""):
    """The AI coach's end-of-session debrief and study plan."""
    st = load_state(state)
    if not st["profile"].get("ai"):
        return HTMLResponse("")
    rep = build_report(st)
    if not rep["answered"]:
        return HTMLResponse("")
    coach = ai.get_coach()
    if not coach.available():
        return render(request, "_coach_summary.html", {"status": "unavailable", "message": "The AI coach isn't set up on this server yet."})
    try:
        data = coach.summary(st, rep, client_id(request))
    except ai.AIError as e:
        return render(request, "_coach_summary.html", {"status": "error", "message": e.public, "state_json": dump_state(st)})
    return render(request, "_coach_summary.html", {"status": "ok", "c": data})


@app.post("/report", response_class=HTMLResponse)
def report(request: Request, state: Annotated[str, Form()] = ""):
    st = load_state(state)
    rep = build_report(st)
    rec = history_record(st, rep)
    return render(request, "report.html", {"r": rep, "state_json": dump_state(st), "record_json": json.dumps(rec, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"),
                                           "interview_date": st["profile"].get("interview_date", ""), "ai_on": bool(st["profile"].get("ai"))})


@app.post("/report.md")
def report_md(state: Annotated[str, Form()] = ""):
    st = load_state(state)
    rep = build_report(st)
    fname = "clearprep-report-" + "".join(c if c.isalnum() else "-" for c in rep["org"].lower()).strip("-") + ".md"
    return Response(to_markdown(st, rep), media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/progress", response_class=HTMLResponse)
def progress_page(request: Request):
    return render(request, "progress.html")


@app.post("/progress/render", response_class=HTMLResponse)
def progress_render(request: Request, history: Annotated[str, Form()] = "[]", interview_date: Annotated[str, Form()] = ""):
    try:
        hist = json.loads(history)
        if not isinstance(hist, list):
            raise ValueError
    except ValueError:
        hist = []
    return render(request, "_progress.html", {"p": build_progress(hist, interview_date)})


@app.get("/how-scoring-works", response_class=HTMLResponse)
def scoring_page(request: Request):
    return render(request, "scoring.html", {"s": scoring_overview(), "rules": RULES, "evalr": load_eval_results()})


@app.get("/prep", response_class=HTMLResponse)
def prep_page(request: Request):
    return render(request, "prep.html", {"p": None, "form": {}, "families": families(), "errors": {}})


@app.post("/prep", response_class=HTMLResponse)
def prep(request: Request, org: Annotated[str, Form()] = "", role: Annotated[str, Form()] = "", family: Annotated[str, Form()] = "auto",
         jd_text: Annotated[str, Form()] = "", about_text: Annotated[str, Form()] = ""):
    form = dict(org=org, role=role, family=family, jd_text=jd_text, about_text=about_text)
    org, role = " ".join(org.split()), " ".join(role.split())
    errors = {}
    if not org:
        errors["org"] = "Enter the organization."
    if not role:
        errors["role"] = "Enter the position."
    if errors:
        return render(request, "prep.html", {"p": None, "form": form, "families": families(), "errors": errors}, status=422)
    jd_text, about_text = jd_text[: sess.LIMITS["jd"]], about_text[: sess.LIMITS["about"]]
    fam = family if family in families() else infer_family(role, jd_text).family
    skills = extract_skills(jd_text, fam, limit=12)
    about_kw = extract_about_keywords(about_text, org)
    duties = extract_duties(jd_text)
    p = {"org": org, "role": role, "family": fam, "family_label": families()[fam]["label"], "skills": skills, "keywords": extract_keywords(jd_text, 10, exclude={s["name"].lower() for s in skills}),
         "about_keywords": about_kw, "duties": duties, "ask": questions_to_ask(org, role, skills, about_kw, duties), "has_jd": bool(jd_text.strip()), "has_about": bool(about_text.strip())}
    return render(request, "prep.html", {"p": p, "form": form, "families": families(), "errors": {}})


# ---------------------------------------------------------------- JSON API (used by tests, docs, and curious people)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response((BASE / "static" / "favicon.svg").read_bytes(), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def touch_icon():
    return Response(status_code=204)  # no touch icon; answer quietly instead of logging a 404 per page load


@app.get("/api/health")
def health():
    return {"status": "ok", "questions": len(load_bank())}


@app.post("/api/analyze")
async def api_analyze(request: Request):
    """Score one answer. Body: {"text": "...", "profile": "star", "competency": "teamwork", "question": "...", "org": "...", "role": "...", "jd_text": "..."}"""
    try:
        body = await request.json()
        text = str(body["text"])
    except (ValueError, KeyError, TypeError):
        return JSONResponse({"error": "Send JSON with a 'text' field."}, status_code=400)
    if len(text) > sess.LIMITS["answer"]:
        return JSONResponse({"error": f"'text' is limited to {sess.LIMITS['answer']} characters."}, status_code=413)
    from interview_engine.bank import competencies

    jd = str(body.get("jd_text", ""))[: sess.LIMITS["jd"]]
    skills = extract_skills(jd, None, 12)
    comp = str(body.get("competency", "")) or None
    ctx = Context(question_text=str(body.get("question", "")), profile=str(body.get("profile", "star")), competency=comp,
                  competency_cues=[str(c) for c in competencies().get(comp, {}).get("cues", [])], org=str(body.get("org", "")), role=str(body.get("role", "")),
                  jd_skills={s["name"]: s["aliases"] for s in skills}, about_keywords=extract_about_keywords(str(body.get("about_text", "")), str(body.get("org", ""))))
    try:
        return analyze(text, ctx)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    return templates.TemplateResponse(request, "error.html", {"code": exc.status_code, "message": exc.detail}, status_code=exc.status_code)
