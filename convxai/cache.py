"""Thread-safe LRU with TTL, byte/entry bounds and single-flight loading."""
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import json
import pickle
import threading
import time

@dataclass(frozen=True)
class CacheKey:
    dataset: str
    model_hash: str
    explainer: str
    scope: str
    instance_hash: str
    language: str = "neutral"
    schema: str = "2"
    prompt_hash: str = "source"
    source_hash: str = ""
    context_hash: str = ""

    @property
    def digest(self):
        return digest(self.__dict__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ByteLRU:
    def __init__(self, max_bytes=32 * 1024 * 1024, max_entries=512, ttl=3600, clock=time.monotonic):
        if min(max_bytes, max_entries, ttl) <= 0:
            raise ValueError("Cache limits must be positive")
        self.max_bytes, self.max_entries, self.ttl = max_bytes, max_entries, ttl
        self.clock = clock
        self._entries = OrderedDict()
        self._pending = {}
        self._lock = threading.RLock()
        self._epoch = 0
        self.bytes = self.hits = self.misses = self.coalesced = 0

    def _drop(self, key):
        item = self._entries.pop(key, None)
        if item: self.bytes -= len(item[1])

    def get_or_load(self, key, loader):
        """Size budget bounds serialised payload bytes, not total process RSS."""
        with self._lock:
            now = self.clock()
            for stale, (expiry, _) in list(self._entries.items()):
                if expiry <= now: self._drop(stale)
            item = self._entries.get(key)
            if item:
                self.hits += 1
                self._entries.move_to_end(key)
                # Only internally serialised values enter this cache.
                return pickle.loads(item[1]), True
            flight = self._pending.get(key)
            if flight:
                self.coalesced += 1
                owner = False
            else:
                flight = Future()
                self._pending[key] = flight
                self.misses += 1
                owner = True
                epoch = self._epoch
        if not owner:
            return pickle.loads(flight.result()), True
        try:
            value = loader()
            payload = pickle.dumps(value, protocol=5)
            with self._lock:
                if epoch == self._epoch and len(payload) <= self.max_bytes:
                    while self._entries and (self.bytes + len(payload) > self.max_bytes or len(self._entries) >= self.max_entries):
                        self._drop(next(iter(self._entries)))
                    self._entries[key] = (self.clock() + self.ttl, payload)
                    self.bytes += len(payload)
                flight.set_result(payload)
            return pickle.loads(payload), False
        except BaseException as exc:
            flight.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def invalidate(self, prefix=""):
        with self._lock:
            self._epoch += 1  # in-flight loads cannot repopulate after invalidation
            for key in list(self._entries):
                if key.startswith(prefix): self._drop(key)

    def stats(self):
        with self._lock:
            return {"payload_bytes": self.bytes, "entries": len(self._entries), "hits": self.hits,
                    "misses": self.misses, "coalesced": self.coalesced}
