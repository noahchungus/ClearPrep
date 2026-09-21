"""Evaluate the STAR detector and the overall scorer against hand-labeled sample answers.

    python eval/run_eval.py             # dev set only (use this while tuning rules)
    python eval/run_eval.py --final     # dev + held-out test; writes eval/results.json (read by the app)

The test set is meant to be scored ONCE after the rules are frozen. See eval/LABELING.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interview_engine.analyzer import Context, analyze  # noqa: E402
from interview_engine.bank import competencies  # noqa: E402

PARTS = ("situation", "task", "action", "result")


def load(name: str) -> list[dict]:
    return json.loads((ROOT / "eval" / "data" / f"{name}.json").read_text(encoding="utf-8"))


def score_sample(sample: dict) -> dict:
    cues = [str(c) for c in competencies().get(sample["competency"], {}).get("cues", [])]
    ctx = Context(question_text=sample["question"], profile="star", competency=sample["competency"], competency_cues=cues)
    return analyze(sample["text"], ctx)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def evaluate(samples: list[dict], verbose: bool = False) -> dict:
    counts = {p: [0, 0, 0] for p in PARTS}  # tp, fp, fn
    overall = {"good": [], "mid": [], "weak": []}
    errors = []
    for s in samples:
        a = score_sample(s)
        overall[s["quality"]].append(a["overall"])
        for p in PARTS:
            pred = a["star"][p]["status"] == "present"
            gold = bool(s["star"][p])
            if pred and gold:
                counts[p][0] += 1
            elif pred and not gold:
                counts[p][1] += 1
                errors.append((s["id"], p, "false positive"))
            elif gold and not pred:
                counts[p][2] += 1
                errors.append((s["id"], p, "missed"))
    per_part = {p: prf(*c) for p, c in counts.items()}
    tp, fp, fn = (sum(c[i] for c in counts.values()) for i in range(3))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    pairs = list(itertools.product(overall["good"], overall["weak"]))
    pair_acc = sum(g > w for g, w in pairs) / len(pairs) if pairs else 0.0
    exact = sum(all((score_sample(s)["star"][p]["status"] == "present") == bool(s["star"][p]) for p in PARTS) for s in samples) / len(samples)
    out = {"n": len(samples), "per_part": per_part, "micro": prf(tp, fp, fn), "exact_match": exact,
           "separation": {"good_mean": mean(overall["good"]), "mid_mean": mean(overall["mid"]), "weak_mean": mean(overall["weak"]), "pairwise_accuracy": pair_acc}}
    if verbose:
        out["errors"] = errors
    return out


def show(name: str, r: dict) -> None:
    print(f"\n== {name} (n={r['n']}) ==")
    print(f"{'part':<10}{'precision':>10}{'recall':>8}{'f1':>6}   tp fp fn")
    for p, m in r["per_part"].items():
        print(f"{p:<10}{m['precision']:>10.2f}{m['recall']:>8.2f}{m['f1']:>6.2f}   {m['tp']:>2} {m['fp']:>2} {m['fn']:>2}")
    m = r["micro"]
    print(f"{'micro':<10}{m['precision']:>10.2f}{m['recall']:>8.2f}{m['f1']:>6.2f}   {m['tp']:>2} {m['fp']:>2} {m['fn']:>2}")
    s = r["separation"]
    print(f"all 4 parts exactly right: {r['exact_match']:.0%}")
    print(f"mean overall: good {s['good_mean']:.0f} | mid {s['mid_mean']:.0f} | weak {s['weak_mean']:.0f} | good>weak pairs {s['pairwise_accuracy']:.0%}")
    if "errors" in r:
        print("errors:", "; ".join(f"{i}:{p}:{k}" for i, p, k in r["errors"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="also score the held-out test set and write eval/results.json")
    args = ap.parse_args()
    dev = evaluate(load("dev"), verbose=True)
    show("dev", dev)
    if args.final:
        test = evaluate(load("test"), verbose=True)
        show("test (held out)", test)
        (ROOT / "eval" / "results.json").write_text(json.dumps({"dev": dev, "test": test}, indent=1), encoding="utf-8")
        print("\nwrote eval/results.json")
