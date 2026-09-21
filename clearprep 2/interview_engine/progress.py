"""Progress dashboard: turn browser-stored session history into trends and a countdown-based plan.

The browser sends only scores and labels (never answer text). Rendering the trend lines here keeps all
logic in Python and unit-testable.
"""
from __future__ import annotations

from datetime import date

from .analyzer import DIMENSIONS
from .report import DIMENSION_LABELS, PRACTICE_TIPS

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 640, 240, 36, 12, 12, 26


def _clean_history(history: list[dict]) -> list[dict]:
    out = []
    for h in history[-60:]:
        try:
            dims = {d: max(0, min(100, int(h["dims"][d]))) for d in DIMENSIONS}
            out.append({"id": str(h.get("id", ""))[:40], "date": str(h.get("date", ""))[:10], "org": str(h.get("org", ""))[:80], "role": str(h.get("role", ""))[:100],
                        "overall": max(0, min(100, int(h["overall"]))), "dims": dims, "priorities": [str(p)[:60] for p in h.get("priorities", [])][:3],
                        "answered": int(h.get("answered", 0)), "questions": [{"competency": str(q.get("competency", ""))[:40], "overall": int(q.get("overall", 0))} for q in h.get("questions", [])][:30]})
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda h: h["date"])
    return out


def _series(history: list[dict], key) -> list[tuple[float, float]]:
    n = len(history)
    pts = []
    for i, h in enumerate(history):
        x = PAD_L + ((W - PAD_L - PAD_R) * (i / (n - 1) if n > 1 else 0.5))
        y = PAD_T + (H - PAD_T - PAD_B) * (1 - key(h) / 100)
        pts.append((round(x, 1), round(y, 1)))
    return pts


def prep_plan(days_left: int | None, weakest: list[str], sessions: int) -> list[str]:
    """A simple countdown checklist. Hand-written templates, filled with the user's weakest areas."""
    w1 = weakest[0] if weakest else "your weakest area"
    w2 = weakest[1] if len(weakest) > 1 else w1
    if days_left is None:
        return []
    if days_left < 0:
        return ["Your interview date has passed. Add a new date to get a fresh plan."]
    steps = [
        ("Run a full mixed session to set a baseline.", 0),
        (f"Drill {w1}: re-do the questions where it lost the most points.", 1),
        ("Write out your 5 core stories (STAR) using the scaffolds, one sentence per part.", 2),
        (f"Run a session with the timer on and work on {w2}.", 3),
        ("Prepare your 3 questions to ask the interviewer and rehearse 'Tell me about yourself' aloud.", 4),
        ("Run a full pressure-mode session and review the report.", 5),
        ("Light review only: re-read your stories, sleep, and plan logistics.", 6),
    ]
    if days_left == 0:
        return ["Interview day: read your 5 stories once, rehearse your opener, and prepare 3 questions to ask. Do not cram new material."]
    picked = steps[: min(days_left, 7)] if days_left < 7 else steps
    if sessions >= 3:
        picked = [s for s in picked if not s[0].startswith("Run a full mixed session to set a baseline")]
    return [f"Day {i + 1}: {t}" for i, (t, _) in enumerate(picked)]


def build_progress(history: list[dict], interview_date: str = "", today: str | None = None) -> dict:
    hist = _clean_history(history)
    now = date.fromisoformat(today) if today else date.today()
    days_left = None
    if interview_date:
        try:
            days_left = (date.fromisoformat(interview_date) - now).days
        except ValueError:
            days_left = None
    n = len(hist)
    lines = {"overall": _series(hist, lambda h: h["overall"])}
    for d in DIMENSIONS:
        lines[d] = _series(hist, lambda h, d=d: h["dims"][d])
    latest = hist[-1] if hist else None
    change = None
    if n >= 2:
        k = min(3, n // 2)
        first = sum(h["overall"] for h in hist[:k]) / k
        last = sum(h["overall"] for h in hist[-k:]) / k
        change = round(last - first)
    weakest = []
    if latest:
        weakest = [DIMENSION_LABELS[d] for d, _ in sorted(latest["dims"].items(), key=lambda kv: kv[1])][:2]
    recurring: dict[str, int] = {}
    for h in hist[-5:]:
        for p in h["priorities"]:
            recurring[p] = recurring.get(p, 0) + 1
    recurring_rows = [{"label": k, "times": v, "drill": PRACTICE_TIPS.get(k, "")} for k, v in sorted(recurring.items(), key=lambda kv: -kv[1]) if v >= 2][:3]
    return {
        "sessions": n, "history": list(reversed(hist)), "latest": latest, "change": change, "days_left": days_left, "interview_date": interview_date,
        "lines": {k: [f"{x},{y}" for x, y in v] for k, v in lines.items()}, "dots": lines["overall"],
        "chart": {"w": W, "h": H, "pad_l": PAD_L, "pad_b": PAD_B, "pad_t": PAD_T, "pad_r": PAD_R, "grid": [{"v": v, "y": round(PAD_T + (H - PAD_T - PAD_B) * (1 - v / 100), 1)} for v in (0, 25, 50, 75, 100)]},
        "dim_labels": DIMENSION_LABELS, "weakest": weakest, "recurring": recurring_rows, "plan": prep_plan(days_left, weakest, n),
    }
