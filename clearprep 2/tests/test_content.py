"""The hand-written content files must stay consistent and usable."""
from collections import Counter

from interview_engine.analyzer import Context, analyze
from interview_engine.bank import competencies, load_bank, validate_bank
from interview_engine.loader import load_yaml
from interview_engine.roles import families
from interview_engine.personalize import _skill_table


def test_bank_is_valid():
    assert validate_bank() == []


def test_bank_size_and_coverage():
    bank = load_bank()
    assert len(bank) >= 120
    for fam in families():
        if fam == "general":
            continue
        assert sum(1 for q in bank if fam in q.families and q.type == "technical") >= 8, fam
        assert sum(1 for q in bank if fam in q.families) >= 12, fam
    used = {q.competency for q in bank}
    for comp in ("teamwork", "conflict", "failure", "initiative", "deadlines", "leadership", "adaptability", "motivation", "ethics"):
        assert comp in used
        assert sum(1 for q in bank if q.competency == comp and q.families == ["general"]) >= 3


def test_every_competency_has_coaching_content():
    for key, c in competencies().items():
        for field in ("label", "looks_for", "rephrase", "probe", "scaffold"):
            assert c.get(field), (key, field)
        assert set(c["scaffold"]) == {"situation", "task", "action", "result"}


def test_every_used_competency_has_exemplar_or_fallback():
    ex = load_yaml("exemplars.yaml")
    for comp in {q.competency for q in load_bank()} - {"technical"}:
        assert comp in ex, comp


def test_star_exemplars_score_well_under_the_analyzer():
    """The examples we show as models should be recognised as good answers by our own scorer."""
    ex = load_yaml("exemplars.yaml")
    comps = competencies()
    for key, entry in ex.items():
        if key not in comps or key == "technical":
            continue
        text = " ".join(str(entry["answer"]).split())
        res = analyze(text, Context(profile="star", competency=key, competency_cues=comps[key]["cues"]))
        assert res["overall"] >= 65, (key, res["overall"])
        assert res["star"]["action"]["status"] == "present", key
        assert res["star"]["result"]["status"] == "present", key


def test_exemplar_part_indexes_point_at_real_sentences():
    from interview_engine.text import sentences

    for key, entry in load_yaml("exemplars.yaml").items():
        n = len(sentences(" ".join(str(entry["answer"]).split())))
        for v in (entry.get("parts") or {}).values():
            for i in (v if isinstance(v, list) else [v]):
                assert 1 <= int(i) <= n, (key, i, n)


def test_skill_lexicon_is_well_formed():
    names = [s["name"] for s in _skill_table()]
    assert len(names) == len(set(names))
    for s in _skill_table():
        assert s["kind"] in ("tool", "practice", "soft")
        assert all(isinstance(a, str) for a in s["aliases"])


def test_no_duplicate_question_text():
    counts = Counter(q.text for q in load_bank())
    assert [t for t, c in counts.items() if c > 1] == []
