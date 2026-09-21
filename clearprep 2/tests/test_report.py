from interview_engine import session as S
from interview_engine.progress import build_progress, prep_plan
from interview_engine.report import build_report, history_record, to_markdown
from tests.test_session import STORY, cfg, run_through


def finished_state(**over):
    return run_through(S.start_session(cfg(**over), seed=11))


def test_report_structure_and_numbers_are_consistent():
    st = finished_state()
    r = build_report(st)
    assert r["answered"] == len(st["plan"]) and r["skipped"] == 0
    assert 0 <= r["overall"] <= 100 and set(r["dims"]) == {"relevance", "structure", "specificity", "concision", "confidence"}
    assert r["priorities"] and all(p["drill"] for p in r["priorities"])
    assert r["jd"] and set(r["jd"]["mentioned"]) | set(r["jd"]["never"]) == set(r["jd"]["skills"])
    assert "structure and delivery" in r["limitations"]


def test_report_handles_skipped_and_early_finish_and_empty():
    st = S.start_session(cfg(), seed=11)
    st = S.skip_question(st)
    st = S.finish_early(st)
    r = build_report(st)
    assert r["answered"] == 0 and r["overall"] == 0 and r["skipped"] == 1 and "Answer at least one" in r["verdict"]
    md = to_markdown(st, r)
    assert "Skipped" in md


def test_markdown_export_contains_the_essentials():
    md = to_markdown(finished_state())
    for needle in ("# Mock interview report", "Acme Credit Union", "## Top things to fix", "| Relevance |", "cannot judge"):
        assert needle in md


def test_history_record_is_compact_and_has_no_answer_text():
    st = finished_state()
    rec = history_record(st)
    blob = str(rec)
    assert STORY[:40] not in blob and rec["overall"] == build_report(st)["overall"] and len(blob) < 5000


def test_progress_dashboard_from_history():
    recs = []
    for i, seed in enumerate((1, 2, 3)):
        st = run_through(S.start_session(cfg(), seed=seed), STORY)
        rec = history_record(st)
        rec["date"] = f"2026-09-1{i}"
        rec["overall"] = 50 + 10 * i
        recs.append(rec)
    p = build_progress(recs, interview_date="2026-09-25", today="2026-09-19")
    assert p["sessions"] == 3 and p["days_left"] == 6 and p["change"] > 0
    assert len(p["lines"]["overall"]) == 3 and p["plan"] and p["weakest"]


def test_progress_survives_empty_and_garbage_history():
    assert build_progress([])["sessions"] == 0
    p = build_progress([{"nope": 1}, "string", {"overall": "x", "dims": {}}], interview_date="garbage")
    assert p["sessions"] == 0 and p["days_left"] is None


def test_prep_plan_edges():
    assert prep_plan(None, [], 0) == []
    assert "passed" in prep_plan(-2, [], 0)[0]
    assert "Interview day" in prep_plan(0, [], 0)[0]
    assert len(prep_plan(30, ["Structure", "Relevance"], 0)) == 7
    assert len(prep_plan(3, [], 0)) == 3
