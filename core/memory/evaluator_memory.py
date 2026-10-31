"""Redis-backed Evaluator memory, isolated from GeneratorMemory by key prefix."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import CACHE_TTL_MEMORY, MAX_MEMORY_TURNS
from core.cache.redis_client import get_redis_client
from core.evaluator.schemas import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluatorMemory:
    KEY_PREFIX = "memory:eval:"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._redis = get_redis_client()
        self._key = f"{self.KEY_PREFIX}{session_id}"

    def add_evaluation(
        self,
        query: str,
        answer: str,
        retrieved_context: str,
        loop: int,
        evaluation: EvaluationResult,
    ) -> bool:
        """Store a verdict, not a duplicate copy of every retrieved chunk."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loop": loop,
            "query": query,
            "evaluated_answer": answer,
            "context_sha256": hashlib.sha256(
                retrieved_context.encode("utf-8")
            ).hexdigest(),
            "evaluation": evaluation.model_dump(),
        }
        try:
            self._redis.rpush(self._key, json.dumps(entry, ensure_ascii=False))
            self._redis.expire(self._key, CACHE_TTL_MEMORY)
            self._redis.ltrim(self._key, -MAX_MEMORY_TURNS, -1)
            return True
        except Exception as exc:
            logger.warning("EvaluatorMemory add_evaluation failed: %s", exc)
            return False

    def get_history(self, last_n: Optional[int] = None) -> list[dict[str, Any]]:
        try:
            if last_n is not None:
                if last_n <= 0:
                    return []
                raw = self._redis.lrange(self._key, -last_n, -1)
            else:
                raw = self._redis.lrange(self._key, 0, -1)
            return [json.loads(item) for item in raw]
        except Exception as exc:
            logger.warning("EvaluatorMemory get_history failed: %s", exc)
            return []

    def get_last_evaluation(self) -> Optional[dict[str, Any]]:
        history = self.get_history(last_n=1)
        return history[0] if history else None

    def clear(self) -> bool:
        try:
            self._redis.delete(self._key)
            return True
        except Exception as exc:
            logger.warning("EvaluatorMemory clear failed: %s", exc)
            return False
