"""
Cache manager for retrieval results and generator responses.

Features:
  - Exact Hash-Based Cache:
      * cache:retrieval:<sha256>   → cached context chunks from ChromaDB
      * cache:gen:resp:<sha256>    → cached generator LLM response
  - Native Semantic Cache (Memurai vectorset on Windows):
      * semantic_cache:prompts     → vector set of query embeddings (VSIM / VADD)
      * semantic_cache:answer:<id> → corresponding cached response
"""

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from core.cache.redis_client import get_redis_client
from config.settings import (
    CACHE_TTL_RETRIEVAL,
    CACHE_TTL_RESPONSE,
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_KEY,
    SEMANTIC_ANSWER_PREFIX,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    """Return a hex SHA-256 digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CacheManager:
    """Manages retrieval, exact response, and native semantic caching in Redis / Memurai."""

    # ── Key prefixes ──
    RETRIEVAL_PREFIX = "cache:retrieval:"
    RESPONSE_PREFIX = "cache:gen:resp:"
    SEMANTIC_CACHE_KEY = SEMANTIC_CACHE_KEY
    SEMANTIC_ANSWER_PREFIX = SEMANTIC_ANSWER_PREFIX

    def __init__(self):
        self._redis = get_redis_client()
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy-loaded HuggingFace embeddings model (same as test.ipynb)."""
        if self._embedding_model is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
        return self._embedding_model

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
    # Exact Generator response cache (hash-based)
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
        """Return a cached generator response by exact hash, or None on miss."""
        key = self._response_key(query, context_hash)
        raw = self._safe_get(key)
        if raw is None:
            logger.debug("Exact response cache MISS  key=%s", key)
            return None
        logger.info("Exact response cache HIT   key=%s", key)
        data = json.loads(raw)
        data["cache_type"] = "exact"
        return data

    def set_response_cache(
        self,
        query: str,
        context_hash: str,
        response: Dict[str, Any],
    ) -> bool:
        """Store a generator response in exact cache with TTL."""
        key = self._response_key(query, context_hash)
        payload = json.dumps(response, ensure_ascii=False)
        ok = self._safe_setex(key, CACHE_TTL_RESPONSE, payload)
        if ok:
            logger.info("Exact response cache SET   key=%s", key)
        return ok

    # ──────────────────────────────────────────────
    # Semantic Cache (Memurai vectorset VSIM / VADD)
    # ──────────────────────────────────────────────

    def get_semantic_response(
        self, query: str, threshold: float = SEMANTIC_CACHE_THRESHOLD
    ) -> Optional[Dict[str, Any]]:
        """
        Searches Memurai vectorset for a semantically equivalent query
        using native VSIM command.

        Returns the cached answer if similarity >= threshold, else None.
        """
        if not SEMANTIC_CACHE_ENABLED:
            return None

        try:
            if not self._redis.exists(self.SEMANTIC_CACHE_KEY):
                return None

            # 1. Convert question to vector embedding
            vector = self.embedding_model.embed_query(query)

            # 2. Search Memurai vectorset using VSIM
            results = self._redis.execute_command(
                "VSIM", self.SEMANTIC_CACHE_KEY,
                "VALUES", len(vector), *vector,
                "COUNT", 1,
                "WITHSCORES",
            )

            if not results:
                return None

            # Handle both dict and list return formats from Memurai
            if isinstance(results, dict):
                entry_id, similarity = next(iter(results.items()))
            elif isinstance(results, (list, tuple)) and len(results) >= 2:
                entry_id = results[0]
                similarity = results[1]
            else:
                return None

            similarity = float(similarity)
            logger.debug("Semantic cache VSIM top similarity: %.4f for query: %.60s", similarity, query)

            # 3. If similarity exceeds threshold, return cached answer
            if similarity >= threshold:
                raw_answer = self._safe_get(f"{self.SEMANTIC_ANSWER_PREFIX}{entry_id}")
                if raw_answer is not None:
                    data = json.loads(raw_answer)
                    data["cached"] = True
                    data["cache_type"] = "semantic"
                    data["similarity_score"] = round(similarity, 4)
                    logger.info("Semantic cache HIT (similarity=%.4f >= %.2f) entry=%s",
                                similarity, threshold, entry_id)
                    return data

        except Exception as exc:
            logger.warning("get_semantic_response failed: %s", exc)

        return None

    def set_semantic_response(
        self,
        query: str,
        response: Dict[str, Any],
    ) -> bool:
        """
        Stores query embedding in Memurai vectorset (VADD)
        and caches the corresponding answer payload.
        """
        if not SEMANTIC_CACHE_ENABLED:
            return False

        try:
            # 1. Embed query
            vector = self.embedding_model.embed_query(query)
            entry_id = uuid.uuid4().hex

            # 2. Store in Memurai vectorset
            self._redis.execute_command(
                "VADD", self.SEMANTIC_CACHE_KEY,
                "VALUES", len(vector), *vector,
                entry_id,
            )

            # 3. Store full answer payload with TTL
            payload = json.dumps(response, ensure_ascii=False)
            ok = self._safe_setex(f"{self.SEMANTIC_ANSWER_PREFIX}{entry_id}", CACHE_TTL_RESPONSE, payload)
            if ok:
                logger.info("Semantic cache SET entry=%s for query: %.60s", entry_id, query)
            return ok

        except Exception as exc:
            logger.warning("set_semantic_response failed: %s", exc)
            return False

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def flush_retrieval_cache(self) -> int:
        """Delete all retrieval cache keys. Returns count deleted."""
        return self._flush_by_prefix(self.RETRIEVAL_PREFIX)

    def flush_response_cache(self) -> int:
        """Delete all response cache keys. Returns count deleted."""
        return self._flush_by_prefix(self.RESPONSE_PREFIX)

    def flush_semantic_cache(self) -> int:
        """Delete Memurai vectorset and all semantic answer keys."""
        deleted = 0
        try:
            if self._redis.exists(self.SEMANTIC_CACHE_KEY):
                self._redis.delete(self.SEMANTIC_CACHE_KEY)
                deleted += 1
            deleted += self._flush_by_prefix(self.SEMANTIC_ANSWER_PREFIX)
            logger.info("Semantic cache flushed (%d keys removed).", deleted)
        except Exception as exc:
            logger.warning("flush_semantic_cache error: %s", exc)
        return deleted

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
