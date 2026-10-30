"""
Isolated Generator memory store backed by Redis / Memurai.

Key pattern:  memory:gen:<session_id>
Data format:  Redis List where each element is a JSON-serialised turn object.

The Generator's memory is strictly separated from the Evaluator's memory
(which will use memory:eval:<session_id>) to avoid any cross-contamination.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.cache.redis_client import get_redis_client
from config.settings import CACHE_TTL_MEMORY, MAX_MEMORY_TURNS

logger = logging.getLogger(__name__)


class GeneratorMemory:
    """Session-scoped conversation and refinement history for the Generator agent."""

    KEY_PREFIX = "memory:gen:"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._redis = get_redis_client()
        self._key = f"{self.KEY_PREFIX}{session_id}"

    # ──────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────

    def add_turn(
        self,
        query: str,
        answer: str,
        context_snippets: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        loop: int = 1,
        evaluator_feedback: Optional[str] = None,
    ) -> bool:
        """
        Append a turn to the Generator's session memory.

        Parameters
        ----------
        query : str
            The user's question.
        answer : str
            The Generator's produced answer.
        context_snippets : list[str] | None
            Summaries / first-N-chars of the retrieved chunks (not full text to save space).
        sources : list[str] | None
            Source identifiers (file names, URLs) used in this turn.
        loop : int
            Which refinement iteration this turn represents (1 = initial).
        evaluator_feedback : str | None
            Feedback from the Evaluator that prompted this refinement (None on loop 1).
        """
        turn: Dict[str, Any] = {
            "turn_index": self._current_length() + 1,
            "loop": loop,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "context_snippets": context_snippets or [],
            "sources": sources or [],
            "evaluator_feedback": evaluator_feedback,
        }
        try:
            self._redis.rpush(self._key, json.dumps(turn, ensure_ascii=False))
            # Reset the TTL on every write so active sessions stay alive
            self._redis.expire(self._key, CACHE_TTL_MEMORY)
            # Trim to MAX_MEMORY_TURNS (keep the most recent turns)
            self._redis.ltrim(self._key, -MAX_MEMORY_TURNS, -1)
            logger.info("GeneratorMemory [%s] turn %d saved (loop=%d)", self.session_id, turn["turn_index"], loop)
            return True
        except Exception as exc:
            logger.warning("GeneratorMemory [%s] add_turn failed: %s", self.session_id, exc)
            return False

    # ──────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history (list of turn dicts).

        Parameters
        ----------
        last_n : int | None
            If given, return only the most recent *n* turns.
        """
        try:
            if last_n is not None:
                raw_list = self._redis.lrange(self._key, -last_n, -1)
            else:
                raw_list = self._redis.lrange(self._key, 0, -1)
            return [json.loads(item) for item in raw_list]
        except Exception as exc:
            logger.warning("GeneratorMemory [%s] get_history failed: %s", self.session_id, exc)
            return []

    def get_last_turn(self) -> Optional[Dict[str, Any]]:
        """Return the most recent turn, or None if the session is empty."""
        history = self.get_history(last_n=1)
        return history[0] if history else None

    def format_for_prompt(self, last_n: int = 5) -> str:
        """
        Render recent history as a readable string suitable for injection
        into an LLM prompt (e.g. as system context).
        """
        turns = self.get_history(last_n=last_n)
        if not turns:
            return "(No prior conversation history for this session.)"

        lines: List[str] = []
        for t in turns:
            lines.append(f"--- Turn {t['turn_index']} (Loop {t['loop']}) ---")
            lines.append(f"Query: {t['query']}")
            lines.append(f"Answer: {t['answer'][:500]}")
            if t.get("evaluator_feedback"):
                lines.append(f"Evaluator Feedback: {t['evaluator_feedback'][:300]}")
            lines.append("")
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Admin
    # ──────────────────────────────────────────────

    def clear(self) -> bool:
        """Delete the entire session memory."""
        try:
            self._redis.delete(self._key)
            logger.info("GeneratorMemory [%s] cleared.", self.session_id)
            return True
        except Exception as exc:
            logger.warning("GeneratorMemory [%s] clear failed: %s", self.session_id, exc)
            return False

    def _current_length(self) -> int:
        try:
            return self._redis.llen(self._key)
        except Exception:
            return 0

