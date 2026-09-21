from interview_engine.personalize import extract_about_keywords, extract_duties, extract_keywords, extract_skills, questions_to_ask, skill_questions
from interview_engine.roles import infer_family, seniority_warning

JD = """Summer Intern - IT Security

Responsibilities:
- Monitor SIEM alerts in Splunk and triage phishing reports
- Document incidents and write clear reports

Requirements:
- Familiarity with Python scripting
- Knowledge of networking and TCP/IP
- Strong communication skills and attention to detail
"""


def test_title_to_family_rules():
    cases = {"Software Engineering Intern": "software", "Marketing Coordinator Intern": "marketing", "Cybersecurity Analyst Intern": "cybersecurity",
             "Junior Data Analyst": "data", "Financial Analyst Intern": "business", "Data Security Analyst": "cybersecurity", "Barista": "general"}
    for title, fam in cases.items():
        assert infer_family(title).family == fam, title


def test_family_falls_back_to_job_description_then_general():
    assert infer_family("Summer Intern", JD).family == "cybersecurity"
    assert infer_family("Summer Intern", JD).source == "job description"
    assert infer_family("Summer Intern").family == "general"


def test_seniority_warning_only_for_senior_titles():
    assert seniority_warning("Senior Software Engineer")
    assert seniority_warning("Software Engineering Intern") is None
    assert seniority_warning("Data Analyst") is None


def test_skill_extraction_ranks_and_flags_requirements():
    hits = {h["name"]: h for h in extract_skills(JD, "cybersecurity")}
    assert {"SIEM", "Python", "Networking", "Phishing", "Communication"} <= set(hits)
    assert hits["Python"]["in_requirements"] and not hits["SIEM"]["in_requirements"]
    assert extract_skills("") == [] and extract_skills("   ") == []


def test_skill_matching_is_whole_word():
    assert not [h for h in extract_skills("We use javascript and postgresql daily") if h["name"] == "Java"]
    assert {h["name"] for h in extract_skills("We use javascript and postgresql daily")} >= {"JavaScript", "SQL"}


def test_keywords_and_duties_come_only_from_the_pasted_text():
    kws = extract_keywords(JD, 8)
    assert kws and all(k in JD.lower() for k in kws)
    assert extract_duties(JD)[0].startswith("Monitor SIEM")
    about = "Acme is a member-owned credit union focused on member service, integrity and financial education."
    kw = extract_about_keywords(about, "Acme")
    assert "integrity" in kw and "acme" not in kw
    assert extract_about_keywords("", "Acme") == []


def test_skill_questions_use_the_skill_name():
    qs = skill_questions([{"name": "Splunk", "kind": "tool"}, {"name": "Communication", "kind": "soft"}], limit=2)
    assert len(qs) == 2 and "Splunk" in qs[0]["text"] and qs[1]["competency"] == "communication"
    assert all(q["generated"] and q["id"].startswith("skill-") for q in qs)


def test_questions_to_ask_never_invent_company_facts():
    plain = questions_to_ask("Acme", "Intern", [], [], [])
    assert plain and all("Acme" in q["text"] or "{" not in q["text"] for q in plain)
    assert not any("{org}" in q["text"] or "{role}" in q["text"] for q in plain)
    rich = questions_to_ask("Acme", "Intern", extract_skills(JD), ["integrity"], ["Monitor SIEM alerts in Splunk"])
    joined = " ".join(q["text"] for q in rich)
    assert "SIEM" in joined and "integrity" in joined and "Monitor SIEM" in joined
    # nothing about the company appears unless it was supplied
    assert "integrity" not in " ".join(q["text"] for q in plain)
