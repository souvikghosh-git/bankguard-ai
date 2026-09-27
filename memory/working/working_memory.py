"""
BankGuard AI — Working Memory (Valkey/Redis).

Short-lived, in-run state that persists across tool calls within one case:
  - Current investigation state (PLAN/ACT/OBSERVE/EVALUATE)
  - Tool call results cache
  - Evidence collected so far
  - Agent observations list
  - Intermediate hypotheses

TTL: 4 hours (a single investigation session)
Key pattern: wm:{run_id}:{slot}
"""

from __future__ import annotations

import json
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_TTL_SECONDS = 4 * 3600  # 4 hours


class WorkingMemory:
    """
    Per-run working memory backed by Valkey.

    Slots:
      state          → current loop state (string)
      plan           → investigation plan (list of steps)
      evidence       → collected evidence items (list)
      observations   → agent's natural-language observations (list)
      hypotheses     → candidate root causes with confidence (list)
      tool_results   → cache of tool outputs keyed by tool+hash (dict)
      context_cache  → last built context contract (dict)
    """

    def __init__(self, valkey: Any, run_id: str) -> None:
        self._v = valkey
        self._run_id = run_id
        self._prefix = f"wm:{run_id}"

    # ── Primitive get/set ─────────────────────────────────────────────────────

    async def set(self, slot: str, value: Any) -> None:
        key = f"{self._prefix}:{slot}"
        await self._v.setex(key, _TTL_SECONDS, json.dumps(value, default=str))

    async def get(self, slot: str, default: Any = None) -> Any:
        key = f"{self._prefix}:{slot}"
        raw = await self._v.get(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return raw

    async def delete(self, slot: str) -> None:
        await self._v.delete(f"{self._prefix}:{slot}")

    # ── Evidence management ───────────────────────────────────────────────────

    async def add_evidence(self, source: str, content: Any) -> None:
        """Append an evidence item to the evidence list."""
        evidence = await self.get("evidence", [])
        evidence.append({"source": source, "content": content})
        await self.set("evidence", evidence)

    async def get_evidence(self) -> list[dict]:
        return await self.get("evidence", [])

    # ── Observations (agent narration) ────────────────────────────────────────

    async def add_observation(self, text: str) -> None:
        observations = await self.get("observations", [])
        observations.append(text)
        # Keep last 20 observations to avoid bloat
        if len(observations) > 20:
            observations = observations[-20:]
        await self.set("observations", observations)

    async def get_observations(self) -> list[str]:
        return await self.get("observations", [])

    # ── State ─────────────────────────────────────────────────────────────────

    async def set_state(self, state: str) -> None:
        await self.set("state", state)
        log.debug("working_memory_state", run_id=self._run_id, state=state)

    async def get_state(self) -> str:
        return await self.get("state", "PLAN")

    # ── Tool result cache ─────────────────────────────────────────────────────

    async def cache_tool_result(self, cache_key: str, result: Any) -> None:
        cache = await self.get("tool_results", {})
        cache[cache_key] = result
        await self.set("tool_results", cache)

    async def get_cached_tool_result(self, cache_key: str) -> Any | None:
        cache = await self.get("tool_results", {})
        return cache.get(cache_key)

    # ── Hypotheses ────────────────────────────────────────────────────────────

    async def set_hypotheses(self, hypotheses: list[dict]) -> None:
        """
        hypotheses: [{"root_cause": str, "confidence": float, "evidence": [str]}]
        """
        await self.set("hypotheses", hypotheses)

    async def get_hypotheses(self) -> list[dict]:
        return await self.get("hypotheses", [])

    # ── Full state snapshot ───────────────────────────────────────────────────

    async def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "state": await self.get_state(),
            "evidence_count": len(await self.get_evidence()),
            "observation_count": len(await self.get_observations()),
            "hypotheses": await self.get_hypotheses(),
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def flush(self) -> None:
        """Delete all keys for this run."""
        pattern = f"{self._prefix}:*"
        keys = await self._v.keys(pattern)
        if keys:
            await self._v.delete(*keys)
        log.debug("working_memory_flushed", run_id=self._run_id)
