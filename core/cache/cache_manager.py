"""
Cache manager for retrieval results and generator responses.

Key patterns:
  - cache:retrieval:<sha256>   → cached context chunks from ChromaDB
  - cache:gen:resp:<sha256>    → cached generator LLM response
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from core.cache.redis_client import get_redis_client
from config.settings import CACHE_TTL_RETRIEVAL, CACHE_TTL_RESPONSE

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    """Return a hex SHA-256 digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CacheManager:
    """Manages retrieval and generator response caches in Redis / Memurai."""

    # ── Key prefixes ──
    RETRIEVAL_PREFIX = "cache:retrieval:"
    RESPONSE_PREFIX = "cache:gen:resp:"

    def __init__(self):
        self._redis = get_redis_client()

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _safe_get(self, key: str) -> Optional[str]:
        """GET that returns None on any Redis failure."""
        try:
            return self._redis.get(key)
        except Exception as exc:
            logger.warning("Redis GET failed for key=%s: %s", key, exc)
            return None

    def _safe_setex(self, key: str, ttl: int, value: str) -> bool:
        """SETEX that silently returns False on failure."""
        try:
            self._redis.setex(key, ttl, value)
            return True
        except Exception as exc:
            logger.warning("Redis SETEX failed for key=%s: %s", key, exc)
            return False

    # ──────────────────────────────────────────────
    # Retrieval cache (context chunks from ChromaDB)
    # ──────────────────────────────────────────────

    def _retrieval_key(self, query: str, top_k: int, collection: str = "default") -> str:
        raw = f"{query.strip().lower()}|k={top_k}|col={collection}"
        return f"{self.RETRIEVAL_PREFIX}{_sha256(raw)}"

    def get_retrieval_cache(
        self, query: str, top_k: int, collection: str = "default"
    ) -> Optional[List[Dict[str, Any]]]:
        """Return cached retrieval results, or None on miss."""
        key = self._retrieval_key(query, top_k, collection)
        raw = self._safe_get(key)
        if raw is None:
            logger.debug("Retrieval cache MISS  key=%s", key)
            return None
        logger.info("Retrieval cache HIT   key=%s", key)
        return json.loads(raw)

    def set_retrieval_cache(
        self,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]],
        collection: str = "default",
    ) -> bool:
        """Store retrieval results in cache with TTL."""
        key = self._retrieval_key(query, top_k, collection)
        payload = json.dumps(results, ensure_ascii=False)
        ok = self._safe_setex(key, CACHE_TTL_RETRIEVAL, payload)
        if ok:
            logger.info("Retrieval cache SET   key=%s  (%d results)", key, len(results))
        return ok

    # ──────────────────────────────────────────────
    # Generator response cache
    # ──────────────────────────────────────────────

    def _response_key(self, query: str, context_hash: str) -> str:
        raw = f"{query.strip().lower()}|ctx={context_hash}"
        return f"{self.RESPONSE_PREFIX}{_sha256(raw)}"

    @staticmethod
    def hash_context(context_chunks: List[str]) -> str:
        """Produce a deterministic hash of the retrieved context for cache keying."""
        combined = "|".join(context_chunks)
        return _sha256(combined)

    def get_response_cache(
        self, query: str, context_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Return a cached generator response, or None on miss."""
        key = self._response_key(query, context_hash)
        raw = self._safe_get(key)
        if raw is None:
            logger.debug("Response cache MISS  key=%s", key)
            return None
        logger.info("Response cache HIT   key=%s", key)
        return json.loads(raw)

    def set_response_cache(
        self,
        query: str,
        context_hash: str,
        response: Dict[str, Any],
    ) -> bool:
        """Store a generator response in cache with TTL."""
        key = self._response_key(query, context_hash)
        payload = json.dumps(response, ensure_ascii=False)
        ok = self._safe_setex(key, CACHE_TTL_RESPONSE, payload)
        if ok:
            logger.info("Response cache SET   key=%s", key)
        return ok

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def flush_retrieval_cache(self) -> int:
        """Delete all retrieval cache keys. Returns count deleted."""
        return self._flush_by_prefix(self.RETRIEVAL_PREFIX)

    def flush_response_cache(self) -> int:
        """Delete all response cache keys. Returns count deleted."""
        return self._flush_by_prefix(self.RESPONSE_PREFIX)

    def _flush_by_prefix(self, prefix: str) -> int:
        """SCAN + DELETE for all keys matching prefix."""
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor=cursor, match=f"{prefix}*", count=200)
                if keys:
                    self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("flush_by_prefix(%s) error: %s", prefix, exc)
        return deleted

