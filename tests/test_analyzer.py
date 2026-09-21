import json

import pytest

from interview_engine.analyzer import DIMENSIONS, IDEAL_WORDS, WEIGHTS, Context, analyze

GOOD = ("Last semester in my software engineering class, our team of four was building a web app and one teammate stopped responding two weeks before the deadline. "
        "My job as team lead was to get the whole app working before the demo. First, I messaged him privately to ask if something was wrong. "
        "Then I split his unfinished tasks into three smaller pieces, took the database piece myself, and reassigned the rest. "
        "As a result, we delivered on time and earned a 96% on the project. I learned to check in early.")
WEAK = "Um, I think we kind of had a group project and it was like really hard. We worked on a lot of stuff and I guess it went well in the end."
CTX = Context(question_text="Tell me about a time a teammate wasn't pulling their weight.", profile="star", competency="teamwork", competency_cues=["team", "teammate", "group"])


def test_good_beats_weak_on_every_headline_number():
    g, w = analyze(GOOD, CTX), analyze(WEAK, CTX)
    assert g["overall"] > w["overall"] + 25
    assert g["dimensions"]["structure"]["score"] > w["dimensions"]["structure"]["score"]
    assert g["dimensions"]["confidence"]["score"] > w["dimensions"]["confidence"]["score"]


def test_star_components_detected_in_good_answer():
    star = analyze(GOOD, CTX)["star"]
    assert {k: v["status"] for k, v in star.items()} == {"situation": "present", "task": "present", "action": "present", "result": "present"}
    assert star["result"]["evidence"], "evidence spans must be reported"


def test_missing_result_is_reported_as_a_gap():
    no_result = GOOD.split("As a result")[0]
    a = analyze(no_result, CTX)
    assert a["star"]["result"]["status"] != "present"
    assert "no_result" in a["gaps"]


def test_hypothetical_answers_do_not_get_story_credit():
    a = analyze("If I got stuck I would try to stay calm and think about it. I would probably ask for help if I needed to and try different things until it worked.", CTX)
    assert a["star"]["result"]["status"] != "present" and a["star"]["action"]["status"] != "present"


def test_numbers_are_found_but_years_are_not_results():
    a = analyze("We grew signups 40% in 3 weeks, and that was in 2024.", CTX)
    assert a["metrics"]["numbers"] == 2  # 40%, 3 weeks; not 2024


def test_fillers_hedges_and_weak_phrases_counted():
    a = analyze(WEAK, CTX)
    m = a["metrics"]
    assert m["fillers"] >= 2 and m["hedges"] >= 2
    conf = {c["label"]: c for c in a["dimensions"]["confidence"]["checks"]}
    assert conf["Hedging"]["earned"] < conf["Hedging"]["possible"]
    assert conf["Filler words"]["evidence"]


def test_all_we_answer_triggers_ownership_gap():
    text = "We had a project in class and we decided to split it up. We built the app and we tested it and we turned it in. Our team did great and we were proud of what we made."
    a = analyze(text, CTX)
    assert "all_we" in a["gaps"] and a["metrics"]["personal_ratio"] == 0


def test_length_gaps():
    assert "too_short" in analyze("It went fine.", CTX)["gaps"]
    assert "too_long" in analyze("word " * (int(IDEAL_WORDS["star"][1] * 1.4)), CTX)["gaps"]


def test_evidence_offsets_match_the_original_text():
    a = analyze(GOOD, CTX)
    for dim in a["dimensions"].values():
        for check in dim["checks"]:
            for ev in check["evidence"]:
                assert a["text"][ev["start"]:ev["end"]] == ev["text"]
    for h in a["highlights"]:
        assert 0 <= h["start"] < h["end"] <= len(a["text"])


def test_highlights_never_overlap():
    a = analyze(GOOD + " " + WEAK, CTX)
    spans = sorted((h["start"], h["end"]) for h in a["highlights"])
    assert all(b[0] >= a_[1] for a_, b in zip(spans, spans[1:]))


def test_scores_are_bounded_and_overall_is_the_weighted_mean():
    for text in (GOOD, WEAK, "a", "I " * 400):
        a = analyze(text, CTX)
        assert all(0 <= a["dimensions"][d]["score"] <= 100 for d in DIMENSIONS)
        assert a["overall"] == round(sum(a["dimensions"][d]["score"] * WEIGHTS[d] for d in DIMENSIONS))


def test_every_check_explains_itself():
    a = analyze(GOOD, CTX)
    for dim in a["dimensions"].values():
        assert dim["checks"]
        for c in dim["checks"]:
            assert c["label"] and c["detail"] and c["possible"] >= 0


def test_analysis_is_deterministic_and_json_serialisable():
    assert analyze(GOOD, CTX) == analyze(GOOD, CTX)
    json.dumps(analyze(GOOD, CTX))


def test_unknown_profile_rejected_and_empty_text_handled():
    with pytest.raises(ValueError):
        analyze("hello", Context(profile="poetry"))
    a = analyze("", CTX)
    assert a["overall"] >= 0 and "too_short" in a["gaps"]


def test_jd_coverage_reported_when_a_job_description_is_given():
    ctx = Context(profile="star", competency="teamwork", competency_cues=["team"], jd_skills={"Python": ["python3"], "Splunk": []})
    a = analyze("On my team I wrote Python scripts to parse logs last semester, and it saved us 3 hours a week.", ctx)
    assert a["jd"] == {"matched": ["Python"], "missing": ["Splunk"]}


def test_technical_concept_coverage_only_checks_mentions():
    ctx = Context(profile="technical", question_text="What is the difference between a list and a hash map?", competency="technical",
                  concepts=[["hash map", "dictionary"], ["lookup"], ["index", "indexed"], ["collision"]])
    good = analyze("A hash map maps keys to values, so lookup by key is constant time on average, while a list is indexed by position. Collisions can slow it down.", ctx)
    poor = analyze("It is a data structure that stores things.", ctx)
    assert len(good["concepts"]["matched"]) == 4 and poor["concepts"]["matched"] == []
    assert "missing_concepts" in poor["gaps"] and good["overall"] > poor["overall"]


def test_why_org_penalizes_generic_praise_and_rewards_using_pasted_text():
    ctx = Context(profile="why_org", org="Northwind", role="Security Intern", about_keywords=["privacy", "member", "service"])
    generic = analyze("Northwind is a great company and a fast-paced innovative place where I can gain experience and make a difference.", ctx)
    specific = analyze("I'm drawn to Northwind because you put member privacy first, and I studied network security last year, so a Security Intern role is where I can apply it.", ctx)
    assert "generic_why" in generic["gaps"] and "generic_why" not in specific["gaps"]
    assert specific["overall"] > generic["overall"]


def test_intro_profile_looks_for_present_past_future():
    ctx = Context(profile="intro", role="Data Analyst Intern")
    a = analyze("I'm a second-year student studying statistics. Last year I built a dashboard for my club with 40 users. I'm looking for a data analyst internship to apply what I've learned.", ctx)
    checks = {c["label"]: c for c in a["dimensions"]["structure"]["checks"]}
    assert all(c["earned"] == c["possible"] for c in checks.values())


def test_ask_us_flags_pay_questions_and_lazy_research():
    ctx = Context(profile="ask_us", org="Northwind", role="Intern")
    bad = analyze("What does your company do? How much does it pay and how many vacation days are there?", ctx)
    good = analyze("What does success look like in the first three months for an Intern? How does the team give new hires feedback?", ctx)
    labels = {c["label"] for c in bad["dimensions"]["relevance"]["checks"]}
    assert {"Too early for pay/perks", "Answerable from their website"} <= labels
    assert good["overall"] > bad["overall"]


def test_speaking_pace_only_scored_for_spoken_answers():
    typed = analyze(GOOD, Context(profile="star", duration_seconds=60, spoken=False))
    spoken = analyze(GOOD, Context(profile="star", duration_seconds=60, spoken=True))
    assert typed["metrics"]["wpm"] is None and spoken["metrics"]["wpm"] is not None
    assert "Speaking pace" in {c["label"] for c in spoken["dimensions"]["concision"]["checks"]}
