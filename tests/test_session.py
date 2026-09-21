import json

import pytest

from interview_engine import session as S

JD = "Requirements:\n- Splunk SIEM experience\n- Python scripting\n- strong communication and attention to detail and networking knowledge for the security team"
STORY = ("Last semester in my networking class, our team of four had to set up a small monitored network in six weeks and one teammate stopped responding. "
         "My job was to make sure we finished. First, I messaged him privately. Then I split his tasks into three pieces, took the logging piece myself, and reassigned the rest. "
         "As a result, we delivered on time and earned a 96 percent. I learned to check in early.")


def cfg(**over):
    raw = {"org": "Acme Credit Union", "role": "Cybersecurity Analyst Intern", "n": 5, "jd_text": JD, "about_text": "We value member service and integrity."}
    raw.update(over)
    c, errors = S.normalize_config(raw)
    assert not errors, errors
    return c


def run_through(state, answer=STORY):
    while state["mode"] != "done":
        if state["mode"] == "asking":
            state = S.submit_answer(state, answer)
        elif state["mode"] == "followup":
            state = S.submit_answer(state, "I personally wrote the checklist and ran the weekly check-in myself.")
        else:
            state = S.next_question(state)
    return state


def test_config_validation_collects_every_error():
    _, errors = S.normalize_config({"org": " ", "role": "", "n": "99", "interview_type": "nonsense", "interview_date": "not-a-date", "time_limit": "7"})
    assert set(errors) == {"org", "role", "n", "interview_type", "interview_date", "time_limit"}


def test_config_cleans_whitespace_and_truncates_long_pastes():
    c, errors = S.normalize_config({"org": "  Acme   Corp ", "role": "Intern", "jd_text": "x " * 20000})
    assert not errors and c["org"] == "Acme Corp" and len(c["jd_text"]) <= S.LIMITS["jd"] and c["_warnings"]


def test_plan_shape_and_personalization():
    st = S.start_session(cfg(), seed=1)
    ids = [p["id"] for p in st["plan"]]
    assert ids[0] in ("gen-open-01", "gen-open-02") and ids[-1] == "ask-us" and len(ids) == 5 + 2
    assert "gen-mot-01" in ids  # "Why {org}?" is always asked outside technical-only interviews
    assert st["profile"]["family"] == "cybersecurity"
    assert {"SIEM", "Python"} <= set(st["profile"]["jd_skills"])
    assert any(p.get("gen") for p in st["plan"]), "JD skills should yield a generated skill question"
    assert len(set(ids)) == len(ids)


def test_same_seed_same_plan_different_seed_different_plan():
    a, b, c = (S.start_session(cfg(), seed=s)["plan"] for s in (5, 5, 6))
    assert a == b and a != c


@pytest.mark.parametrize("kind", ["behavioral", "situational", "technical", "mixed"])
def test_every_interview_type_and_size_produces_a_valid_plan(kind):
    for n in (3, 8, 15):
        st = S.start_session(cfg(interview_type=kind, n=n), seed=n)
        assert len(st["plan"]) == n + 2 and len({p["id"] for p in st["plan"]}) == len(st["plan"])
        for i in range(len(st["plan"])):
            assert S.question_at(st, i)["text"]


def test_technical_interview_uses_family_technical_questions():
    st = S.start_session(cfg(interview_type="technical", n=6, jd_text=""), seed=3)
    types = [S.question_at(st, i)["type"] for i in range(1, 7)]
    assert types.count("technical") >= 5


def test_general_role_without_technical_bank_still_works():
    st = S.start_session(cfg(role="Barista", interview_type="technical", jd_text=""), seed=2)
    assert st["profile"]["family"] == "general" and len(st["plan"]) == 7 and st["profile"]["warnings"]


def test_org_and_role_are_filled_into_templates():
    st = S.start_session(cfg(), seed=1)
    texts = " ".join(S.question_at(st, i)["text"] for i in range(len(st["plan"])))
    assert "Acme Credit Union" in texts and "{org}" not in texts and "{role}" not in texts


def test_full_session_reaches_done_and_state_stays_json_and_small():
    st = run_through(S.start_session(cfg(), seed=4))
    assert st["mode"] == "done" and len(st["records"]) == len(st["plan"])
    assert len(json.dumps(st)) < 60_000
    json.loads(json.dumps(st))


def test_followup_is_triggered_by_gaps_and_folded_into_the_score():
    st = S.start_session(cfg(interview_type="behavioral", n=3), seed=9)
    while S.question_at(st, st["cursor"])["stage"] != "main":
        st = S.skip_question(st)
    st = S.submit_answer(st, "We had a group project once and we worked on a lot of stuff and it went well in the end for us all.")
    assert st["mode"] == "followup" and st["records"][-1]["followup"]["text"]
    st2 = S.submit_answer(st, STORY)
    assert st2["mode"] == "feedback" and st2["records"][-1]["scores"]["overall"] > 40
    assert S.view(st2)["analysis"]["overall"] == st2["records"][-1]["scores"]["overall"]


def test_followups_can_be_turned_off():
    st = S.start_session(cfg(followups="0"), seed=9)
    st = S.submit_answer(st, "It went fine.")
    assert st["mode"] == "feedback"


def test_skip_followup_and_skip_question():
    st = S.start_session(cfg(), seed=9)
    st = S.submit_answer(st, "It was fine.")
    assert st["mode"] == "followup"
    st = S.skip_followup(st)
    assert st["mode"] == "feedback" and st["records"][-1]["followup"]["skipped"]
    st = S.next_question(st)
    st = S.skip_question(st)
    assert st["records"][-1]["skipped"] and st["cursor"] == 2


def test_state_machine_rejects_invalid_transitions_and_bad_input():
    st = S.start_session(cfg(), seed=1)
    for bad in (S.next_question, S.skip_followup):
        with pytest.raises(S.StateError):
            bad(st)
    with pytest.raises(ValueError, match="answer"):
        S.submit_answer(st, "   ")
    with pytest.raises(ValueError, match="limited"):
        S.submit_answer(st, "word " * 2000)
    fb = S.submit_answer(S.start_session(cfg(followups="0"), seed=1), STORY)
    with pytest.raises(S.StateError):
        S.submit_answer(fb, STORY)
    with pytest.raises(S.StateError):
        S.skip_question(fb)


def test_transitions_do_not_mutate_the_input_state():
    st = S.start_session(cfg(), seed=1)
    snap = json.dumps(st)
    S.submit_answer(st, STORY)
    S.skip_question(st)
    assert json.dumps(st) == snap


def test_rephrase_returns_hand_written_hint_and_is_read_only():
    st = S.start_session(cfg(), seed=1)
    snap = json.dumps(st)
    assert len(S.rephrase(st)) > 20 and json.dumps(st) == snap


def test_finish_early_from_any_mode():
    st = S.start_session(cfg(), seed=1)
    st = S.submit_answer(st, "It was fine.")  # -> followup
    st = S.finish_early(st)
    assert st["mode"] == "done" and len(st["records"]) == 1


def test_ask_us_stage_has_no_followup_and_uses_its_own_profile():
    st = S.start_session(cfg(), seed=1)
    st["cursor"], st["mode"] = len(st["plan"]) - 1, "asking"
    q = S.question_at(st, st["cursor"])
    assert q["stage"] == "ask_us" and q["profile"] == "ask_us"
    st = S.submit_answer(st, "Hi")
    assert st["mode"] == "feedback"
