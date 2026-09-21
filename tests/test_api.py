import html
import json
import re

import pytest
from fastapi.testclient import TestClient

from web.app import MAX_STATE_BYTES, app
from tests.test_session import JD, STORY

client = TestClient(app)
FORM = {"org": "Acme Credit Union", "role": "Cybersecurity Analyst Intern", "n": "4", "jd_text": JD, "about_text": "We value member service.", "followups": "1"}


def get_state(text: str) -> str:
    return html.unescape(re.search(r'name="state" value="([^"]+)"', text).group(1))


def start(**over):
    r = client.post("/interview/start", data={**FORM, **over})
    assert r.status_code == 200, r.text[:300]
    return r, get_state(r.text)


@pytest.mark.parametrize("path", ["/", "/practice", "/prep", "/progress", "/how-scoring-works", "/api/health"])
def test_pages_render(path):
    assert client.get(path).status_code == 200


def test_static_assets_are_served_locally():
    for f in ("style.css", "app.js", "htmx.min.js"):
        assert client.get(f"/static/{f}").status_code == 200
    page = client.get("/").text
    assert "http://" not in page and "https://" not in page.replace("https://", "", 0) or "cdn" not in page  # no third-party assets


def test_setup_validation_errors_are_shown_and_values_kept():
    r = client.post("/interview/start", data={"org": "", "role": "Intern", "n": "2", "jd_text": "keep me"})
    assert r.status_code == 422
    assert "Enter the organization" in r.text and "Choose between 3 and 15" in r.text and "keep me" in r.text


def test_start_shows_personalization():
    r, _ = start()
    assert "Cybersecurity" in r.text and "SIEM" in r.text and 'id="stage"' in r.text and "Acme Credit Union" in r.text


def test_full_interview_flow_over_http():
    _, state = start(followups="0", n="3")
    seen = 0
    for _ in range(30):
        r = client.post("/interview/answer", data={"state": state, "answer": STORY, "duration": "45", "spoken": ""})
        assert r.status_code == 200 and "big-score" in r.text
        state = get_state(r.text)
        seen += 1
        r = client.post("/interview/next", data={"state": state})
        state = get_state(r.text)
        if "See my report" in r.text:
            break
    assert seen == 5  # opener + 3 + closing
    rep = client.post("/report", data={"state": state})
    assert rep.status_code == 200 and "Your top priorities" in rep.text and 'id="session-record"' in rep.text
    md = client.post("/report.md", data={"state": state})
    assert md.status_code == 200 and "attachment" in md.headers["content-disposition"] and md.text.startswith("# Mock interview report")


def test_followup_flow_over_http():
    _, state = start()
    r = client.post("/interview/answer", data={"state": state, "answer": "It went fine."})
    assert "Follow-up" in r.text and "Why this was asked" in r.text
    state = get_state(r.text)
    r = client.post("/interview/skip-followup", data={"state": state})
    assert "Next question" in r.text


def test_empty_answer_is_a_friendly_error_not_a_crash():
    _, state = start()
    r = client.post("/interview/answer", data={"state": state, "answer": "  "})
    assert r.status_code == 200 and "Type or dictate an answer" in r.text and "notice err" in r.text


def test_wrong_mode_action_is_a_friendly_error():
    _, state = start()
    r = client.post("/interview/next", data={"state": state})
    assert r.status_code == 200 and "available right now" in r.text


@pytest.mark.parametrize("bad", ["", "not json", "[]", '{"v": 99}', '{"v":1,"profile":{},"plan":[],"cursor":0,"mode":"asking","records":[]}'])
def test_garbage_state_is_rejected_cleanly(bad):
    r = client.post("/interview/answer", data={"state": bad, "answer": "hi"})
    assert r.status_code == 400 and "could not be read" in r.text


def test_tampered_question_id_and_oversized_state_are_rejected():
    _, state = start()
    st = json.loads(state)
    st["plan"][0]["id"] = "does-not-exist"
    assert client.post("/interview/answer", data={"state": json.dumps(st), "answer": "hi"}).status_code == 400
    assert client.post("/interview/answer", data={"state": "x" * (MAX_STATE_BYTES + 1), "answer": "hi"}).status_code == 413


def test_user_text_is_escaped_everywhere():
    evil = "<script>alert(1)</script>"
    r, state = start(org=evil, role="Software <b>Intern</b>")
    assert evil not in r.text and "&lt;script&gt;" in r.text
    r = client.post("/interview/answer", data={"state": state, "answer": f"I built {evil} last year as a student " * 5})
    assert evil not in r.text
    rep = client.post("/report", data={"state": get_state(r.text)})
    assert evil not in rep.text
    assert "</script><script>" not in rep.text.replace("\n", "")


def test_progress_render_with_and_without_history():
    empty = client.post("/progress/render", data={"history": "[]"})
    assert "No sessions yet" in empty.text
    hist = [{"id": f"s{i}", "date": f"2026-09-1{i}", "org": "A", "role": "B", "overall": 50 + i * 5, "dims": {d: 60 for d in ("relevance", "structure", "specificity", "concision", "confidence")}, "priorities": ["Result"], "answered": 5} for i in range(3)]
    full = client.post("/progress/render", data={"history": json.dumps(hist), "interview_date": "2030-01-01"})
    assert "<svg" in full.text and "practice sessions" in full.text
    assert "No sessions yet" in client.post("/progress/render", data={"history": "garbage"}).text


def test_prep_sheet_uses_only_pasted_text():
    r = client.post("/prep", data={"org": "Acme", "role": "Security Intern", "jd_text": JD, "about_text": "We value integrity."})
    assert r.status_code == 200 and "SIEM" in r.text and "integrity" in r.text and "What does success look like" in r.text
    assert client.post("/prep", data={"org": "", "role": ""}).status_code == 422


def test_json_api_analyze():
    r = client.post("/api/analyze", json={"text": STORY, "profile": "star", "competency": "teamwork", "question": "Tell me about teamwork", "jd_text": JD})
    assert r.status_code == 200
    body = r.json()
    assert body["overall"] > 50 and body["star"]["result"]["status"] == "present" and body["jd"]["matched"]
    assert client.post("/api/analyze", json={"nope": 1}).status_code == 400
    assert client.post("/api/analyze", content="not json").status_code == 400
    assert client.post("/api/analyze", json={"text": "hi", "profile": "bad"}).status_code == 400
    assert client.post("/api/analyze", json={"text": "x" * 5000}).status_code == 413


def test_unknown_route_404s():
    assert client.get("/nope").status_code == 404


def test_browser_icon_probes_do_not_404():
    assert client.get("/favicon.ico").headers["content-type"].startswith("image/svg+xml")
    assert client.get("/apple-touch-icon.png").status_code == 204
    assert client.get("/apple-touch-icon-precomposed.png").status_code == 204


def test_static_urls_are_versioned_so_edits_are_never_served_stale():
    page = client.get("/").text
    for name in ("style.css", "app.js", "htmx.min.js", "cs50-duck.png"):
        assert re.search(rf'/static/{re.escape(name)}\?v=\d+', page), name


def test_landing_page_has_hero_robot_description_and_calls_to_action():
    page = client.get("/").text
    assert 'class="hero-band"' in page and 'class="robot"' in page and "hello-bubble" not in page
    assert "A mock interview built around your job" in page and page.count("data-reveal") >= 8
    assert 'href="/practice"' in page and 'href="/prep"' in page
    assert 'id="setup-form"' not in page  # the form lives on /practice now


def test_practice_page_holds_the_setup_form_and_nav_points_to_it():
    page = client.get("/practice").text
    assert 'id="setup-form"' in page and 'action="/interview/start"' in page
    assert 'href="/practice"' in client.get("/prep").text


def test_nav_marks_the_current_page_on_every_top_level_page():
    for path in ("/prep", "/progress", "/how-scoring-works", "/practice"):
        assert 'aria-current="page"' in client.get(path).text


def test_site_is_named_clearprep_with_a_two_tone_stacked_logo():
    for path in ("/", "/practice", "/prep", "/progress", "/how-scoring-works"):
        text = client.get(path).text
        assert "Interview Trainer" not in text and "ClearPrep" in text
    page = client.get("/").text
    assert 'aria-label="ClearPrep"' in page and ">Clea</span>" in page and ">r</span>" in page and ">P</span>" in page and ">rep</span>" in page
    assert "<title>ClearPrep" in page


def test_home_reading_is_short():
    import re as _re

    page = client.get("/").text
    steps = _re.findall(r'<article class="step".*?<h3>(.*?)</h3>\s*<p>(.*?)</p>', page, _re.S)
    assert [t for t, _ in steps] == ["Enter the role", "Answer real questions", "Get coached", "Track your progress"]
    assert all(len(p.split()) <= 16 for _, p in steps)
    lead = _re.search(r'<p class="lead">(.*?)</p>', page, _re.S).group(1)
    assert len(lead.split()) <= 20


def test_scoring_page_no_longer_points_at_a_readme():
    text = client.get("/how-scoring-works").text.lower()
    assert "readme" not in text


def test_footer_credits_the_author_like_clearpath():
    for path in ("/", "/practice", "/how-scoring-works"):
        page = client.get(path).text
        assert "built by a community college student" in page and "Developed by Noah Chung" in page
