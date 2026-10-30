"""
Redis / Memurai connection manager with connection pooling and graceful fallback.
"""

import logging
from typing import Optional

import redis

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD

logger = logging.getLogger(__name__)

# Module-level singleton pool and client
_pool: Optional[redis.ConnectionPool] = None
_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """
    Returns a singleton Redis client backed by a connection pool.
    Creates the pool on first call and reuses it afterwards.
    """
    global _pool, _client

    if _client is not None:
        return _client

    _pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,       # All values returned as str (not bytes)
        max_connections=10,
        socket_connect_timeout=3,
        socket_timeout=3,
        retry_on_timeout=True,
    )
    _client = redis.Redis(connection_pool=_pool)

    # Validate connectivity at startup
    try:
        _client.ping()
        logger.info("Redis/Memurai connected at %s:%s (db=%s)", REDIS_HOST, REDIS_PORT, REDIS_DB)
    except redis.ConnectionError as exc:
        logger.warning("Redis/Memurai unreachable at %s:%s – cache will be bypassed. Error: %s",
                       REDIS_HOST, REDIS_PORT, exc)

    return _client


def is_redis_available() -> bool:
    """Quick health-check; returns False if Redis cannot be reached."""
    try:
        client = get_redis_client()
        return client.ping()
    except Exception:
        return False

