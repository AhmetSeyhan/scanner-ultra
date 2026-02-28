"""Scanner ULTRA — Content hash database (SHA-256 caching + blocklist)."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class HashDatabase:
    def __init__(self, redis_url: str | None = None, ttl: int = 86400) -> None:
        self.ttl = ttl
        self._redis = None
        self._cache: dict[str, dict[str, Any]] = {}

        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Hash DB connected to Redis")
            except Exception:
                logger.warning("Redis unavailable — in-memory hash DB")
                self._redis = None

    def compute_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    async def lookup(self, content_hash: str) -> dict[str, Any] | None:
        if self._redis:
            try:
                import json

                data = self._redis.get(f"scan:{content_hash}")
                if data:
                    return json.loads(data)
            except Exception:
                pass
        entry = self._cache.get(content_hash)
        if entry and time.time() - entry.get("_at", 0) < self.ttl:
            return entry.get("result")
        return None

    async def store(self, content_hash: str, result: dict[str, Any]) -> None:
        if self._redis:
            try:
                import json

                self._redis.setex(f"scan:{content_hash}", self.ttl, json.dumps(result, default=str))
            except Exception:
                pass
        self._cache[content_hash] = {"result": result, "_at": time.time()}

    async def is_known_fake(self, content_hash: str) -> bool:
        if self._redis:
            try:
                return bool(self._redis.sismember("blocklist:fakes", content_hash))
            except Exception:
                pass
        return False

    def clear_cache(self) -> None:
        self._cache.clear()
