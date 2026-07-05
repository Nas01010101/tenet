"""Mnemo — the memory agent: distillation + bi-temporal store, one interface.

    m = Mnemo()
    m.ingest("I moved to Toronto last week.")   # distills -> atomic keyed facts -> store
    m.recall("where does the user live?")        # bi-temporal, forgetting-aware retrieval

`ingest` is the write path (LLM distillation, one call per message). `recall` is the
read path (pure vector + decay, no LLM — frontier-correct for latency). The store
handles supersession/forgetting; the distiller supplies the keys that make it reliable.
"""
from __future__ import annotations

import time

from distill import distill
from memory import Memory, MemoryCore


class Mnemo:
    def __init__(self, db_path=None, *, now=time.time, distill_model: str | None = None):
        self.core = MemoryCore(db_path, now=now) if db_path else MemoryCore(now=now)
        self._now = now
        self._distill_model = distill_model

    def ingest(self, message: str, *, pinned: bool = False) -> list[int]:
        """Distill a raw message into atomic facts and store each (with supersession).
        Returns the stored memory ids. Empty if nothing durable was found."""
        kw = {"model": self._distill_model} if self._distill_model else {}
        facts = distill(message, **kw)
        ids = []
        for f in facts:
            ids.append(self.core.store(
                f.statement, key=f.key, salience=f.salience, pinned=pinned,
            ))
        return ids

    def store_fact(self, text: str, **kw) -> int:
        """Store a pre-formed fact directly (bypass distillation)."""
        return self.core.store(text, **kw)

    def recall(self, query: str, **kw) -> list[Memory]:
        return self.core.recall(query, **kw)

    def forget_sweep(self) -> int:
        return self.core.forget_sweep()

    def stats(self) -> dict:
        return self.core.stats()

    def close(self):
        self.core.close()
