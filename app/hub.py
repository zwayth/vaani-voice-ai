"""Live call registry + dashboard event fan-out + call log persistence."""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path

from .config import ROOT

LOG_DIR = ROOT / "data" / "calls"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Hub:
    def __init__(self):
        self.calls: dict = {}            # id -> CallSession
        self.monitors: set[asyncio.Queue] = set()
        self.history: deque = deque(maxlen=200)   # finished call summaries
        self.turn_latencies: deque = deque(maxlen=500)
        self.total_calls = 0
        self._load_history()

    def _load_history(self):
        files = sorted(LOG_DIR.glob("*.json"))[-200:]
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                self.history.append(d["summary"])
                self.total_calls += 1
                for t in d.get("turns", []):
                    if t.get("e2e_ms"):
                        self.turn_latencies.append(t["e2e_ms"])
            except Exception:  # noqa: BLE001
                pass

    def publish(self, ev: dict):
        ev.setdefault("ts", time.time())
        for q in list(self.monitors):
            if q.qsize() < 500:
                q.put_nowait(ev)

    def register(self, call):
        self.calls[call.id] = call
        self.total_calls += 1
        self.publish({"type": "call_started", "call": call.summary()})

    def unregister(self, call):
        self.calls.pop(call.id, None)
        s = call.summary()
        self.history.append(s)
        try:
            (LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{call.id}.json").write_text(
                json.dumps({"summary": s, "transcript": call.transcript, "turns": call.turns}, ensure_ascii=False,
                           indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        self.publish({"type": "call_ended", "call": s})

    def stats(self) -> dict:
        lat = sorted(self.turn_latencies)
        pct = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))]) if lat else None  # noqa: E731
        return {"active": len(self.calls), "total": self.total_calls, "p50": pct(0.5), "p90": pct(0.9),
                "turns": len(lat)}


HUB = Hub()
