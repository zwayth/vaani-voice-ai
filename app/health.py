"""Per-provider circuit breaker + latency stats. A failing provider is skipped for a cool-down window
so a degraded vendor never adds latency to live calls."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ProviderHealth:
    name: str
    kind: str
    configured: bool = True
    fails: int = 0
    successes: int = 0
    open_until: float = 0.0
    last_error: str = ""
    lat: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def available(self) -> bool:
        return self.configured and time.monotonic() >= self.open_until

    def ok(self, ms: float):
        self.successes += 1
        self.fails = 0
        self.lat.append(ms)

    def fail(self, err: str):
        self.fails += 1
        self.last_error = err[:200]
        # exponential cool-down: 5s, 10s, 20s ... capped at 120s
        self.open_until = time.monotonic() + min(120, 5 * 2 ** (self.fails - 1))

    def snapshot(self) -> dict:
        lat = sorted(self.lat)
        p50 = lat[len(lat) // 2] if lat else None
        p95 = lat[int(len(lat) * 0.95) - 1] if len(lat) >= 2 else p50
        state = "unconfigured" if not self.configured else ("healthy" if self.available else "cooling")
        return {"name": self.name, "kind": self.kind, "state": state, "p50": p50, "p95": p95,
                "ok": self.successes, "fails": self.fails, "last_error": self.last_error}


REGISTRY: dict[str, ProviderHealth] = {}


def health(kind: str, name: str, configured: bool = True) -> ProviderHealth:
    key = f"{kind}:{name}"
    h = REGISTRY.get(key)
    if h is None:
        h = REGISTRY[key] = ProviderHealth(name, kind, configured)
    h.configured = configured
    return h


def snapshot() -> list[dict]:
    return [h.snapshot() for h in REGISTRY.values()]
