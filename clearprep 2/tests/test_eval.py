"""Regression guard for the STAR detector. Floors are set below the measured numbers in eval/results.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import evaluate, load  # noqa: E402


def test_labeled_sets_are_well_formed():
    for name in ("dev", "test"):
        data = load(name)
        assert len(data) >= 20 and len({d["id"] for d in data}) == len(data)
        assert {d["quality"] for d in data} == {"good", "mid", "weak"}
        for d in data:
            assert set(d["star"]) == {"situation", "task", "action", "result"}


def test_star_detector_regression_floors():
    dev, test = evaluate(load("dev")), evaluate(load("test"))
    assert dev["micro"]["precision"] >= 0.9 and dev["micro"]["recall"] >= 0.9
    assert test["micro"]["precision"] >= 0.9 and test["micro"]["recall"] >= 0.6


def test_scores_separate_good_from_weak_answers():
    for name in ("dev", "test"):
        s = evaluate(load(name))["separation"]
        assert s["good_mean"] > s["mid_mean"] > s["weak_mean"]
        assert s["pairwise_accuracy"] >= 0.95
