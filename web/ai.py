"""The AI coach: a thin, defensive wrapper around the Anthropic API.

Everything here is optional. If there is no API key, the coach is switched off, the SDK is missing, a call fails, or
the model returns something unusable, callers get an AIError with a message safe to show, and the app falls back to
the rule-based feedback. The rule-based scores are never replaced or changed by the AI.

Configuration (environment variables, or a `.env` file in the project root that is git-ignored):
    ANTHROPIC_API_KEY            required to turn the AI coach on
    INTERVIEW_TRAINER_MODEL      model id, default claude-opus-5 (e.g. claude-sonnet-5 costs less)
    INTERVIEW_TRAINER_EFFORT     low | medium | high, default medium (lower = faster and cheaper)
    AI_CALLS_PER_IP_PER_HOUR     default 60   (each answer uses up to 2 calls, the report 1)
    AI_CALLS_PER_HOUR_TOTAL      default 600  (global hourly cap; protects the bill on a public deployment)
    AI_CALLS_PER_DAY_TOTAL       default 4000 (global daily cap; the real backstop against a runaway bill)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from interview_engine import ai_prompts as P

log = logging.getLogger("interview_trainer.ai")
DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AIError(Exception):
    """A failure with a message that is safe to show the user."""

    def __init__(self, public: str, *, kind: str = "error"):
        super().__init__(public)
        self.public, self.kind = public, kind


def load_env_file(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win. No dependency needed."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


load_env_file(Path(__file__).resolve().parent.parent / ".env")


class RateLimiter:
    """Sliding-window limiter, per client and global. In-memory: best-effort protection, resets on restart."""

    DAY = 86400.0

    def __init__(self, per_client: int, total: int, window: float = 3600.0, daily_total: int | None = None):
        self.per_client, self.total, self.window, self.daily_total = per_client, total, window, daily_total
        self._hits: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()
        self._day: deque = deque()
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            for q, span in ((self._hits[client], self.window), (self._all, self.window), (self._day, self.DAY)):
                while q and q[0] <= now - span:
                    q.popleft()
            if len(self._hits[client]) >= self.per_client or len(self._all) >= self.total:
                return False
            if self.daily_total is not None and len(self._day) >= self.daily_total:
                return False
            self._hits[client].append(now)
            self._all.append(now)
            self._day.append(now)
            return True


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Coach:
    """Calls the model and returns validated dicts. `client` can be injected for tests."""

    def __init__(self, client=None, model: str | None = None, limiter: RateLimiter | None = None):
        self._injected = client
        self._client = client
        self.model = model or os.environ.get("INTERVIEW_TRAINER_MODEL", DEFAULT_MODEL)
        self.effort = os.environ.get("INTERVIEW_TRAINER_EFFORT", "medium")
        self.limiter = limiter or RateLimiter(_int_env("AI_CALLS_PER_IP_PER_HOUR", 60), _int_env("AI_CALLS_PER_HOUR_TOTAL", 600),
                                              daily_total=_int_env("AI_CALLS_PER_DAY_TOTAL", 4000))

    # ---- availability
    def available(self) -> bool:
        return self._injected is not None or bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise AIError("The AI coach isn't installed on this server.", kind="unavailable") from e
            self._client = anthropic.Anthropic(timeout=60.0, max_retries=1)
        return self._client

    # ---- the one place that talks to the API
    def call_json(self, req: dict, client_id: str = "local") -> dict:
        if not self.available():
            raise AIError("The AI coach isn't set up on this server yet.", kind="unavailable")
        if not self.limiter.allow(client_id):
            raise AIError("The AI coach is taking a short breather (usage limit reached). Try again in a little while.", kind="limited")
        client = self._get_client()
        common = dict(model=self.model, max_tokens=req["max_tokens"], system=req["system"], messages=[{"role": "user", "content": req["user"]}])
        fmt = {"format": {"type": "json_schema", "schema": req["schema"]}}
        # Attempts, most preferred first. Each later attempt only runs if the earlier one was a 400 (unsupported combination).
        attempts = [
            lambda: client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", output_config={"effort": self.effort, **fmt}, **common),
            lambda: client.messages.create(output_config={"effort": self.effort, **fmt}, **common),
            lambda: client.messages.create(output_config=fmt, **common),
        ]
        resp = None
        for i, attempt in enumerate(attempts):
            try:
                resp = attempt()
                break
            except Exception as e:  # noqa: BLE001 - mapped below
                if type(e).__name__ == "BadRequestError" and i < len(attempts) - 1:
                    log.warning("AI request rejected (%s); retrying with a simpler request: %s", i + 1, getattr(e, "message", e))
                    continue
                raise self._map_error(e) from e
        if getattr(resp, "stop_reason", None) == "refusal":
            raise AIError("Your coach couldn't respond to that answer.", kind="refused")
        try:
            text = next(b.text for b in resp.content if b.type == "text")
            return json.loads(text)
        except (StopIteration, ValueError, AttributeError) as e:
            log.warning("AI returned unparseable output (stop_reason=%s)", getattr(resp, "stop_reason", "?"))
            raise AIError("Your coach's reply came back garbled. Try again.", kind="bad_output") from e

    @staticmethod
    def _map_error(e: Exception) -> AIError:
        name = type(e).__name__
        log.warning("AI call failed: %s: %s", name, str(e)[:300])  # never log the key or the request body
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return AIError("The AI coach's API key was rejected. Check the server's AI key setting.", kind="auth")
        if name == "RateLimitError":
            return AIError("The AI coach is busy right now. Try again in a moment.", kind="busy")
        if name in ("APIConnectionError", "APITimeoutError"):
            return AIError("Couldn't reach the AI coach. Check the connection and try again.", kind="network")
        return AIError("The AI coach hit an error. Try again in a moment.", kind="error")

    # ---- the three jobs
    def coach(self, q: dict, analysis: dict, profile: dict, record: dict, client_id: str = "local") -> dict:
        return self._validated(P.build_coach_request(q, analysis, profile, record), P.normalize_coach, client_id)

    def followup(self, q: dict, analysis: dict, profile: dict, answer: str, client_id: str = "local") -> dict | None:
        return self._validated(P.build_followup_request(q, analysis, profile, answer), P.normalize_followup, client_id)

    def summary(self, state: dict, report: dict, client_id: str = "local") -> dict:
        return self._validated(P.build_summary_request(state, report), P.normalize_summary, client_id)

    def _validated(self, req: dict, normalize, client_id: str):
        data = self.call_json(req, client_id)
        try:
            return normalize(data)
        except ValueError as e:
            log.warning("AI output failed validation: %s", e)
            raise AIError("Your coach's reply wasn't usable. Try again.", kind="bad_output") from e


_coach = Coach()


def get_coach() -> Coach:
    return _coach


def set_coach(coach: Coach) -> None:
    """Swap the process-wide coach (used by tests)."""
    global _coach
    _coach = coach
