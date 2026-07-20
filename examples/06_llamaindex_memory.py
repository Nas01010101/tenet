"""Tenet as LlamaIndex long-term memory — a `BaseMemoryBlock` in the modern
`Memory` block stack (llama-index-core >= 0.12).

    pip install -e ".[llamaindex,local]"
    python examples/06_llamaindex_memory.py     # no API key: block-level walkthrough

Why this beats the shipped FactExtractionMemoryBlock for facts that change:
that block appends extracted facts to a list, so "I live in Montreal" and a
later "I moved to Toronto" BOTH sit in the prompt. TenetMemoryBlock's puts go
through keyed supersession — the old value is retired to history (queryable via
`recall(as_of=...)`), and `aget` only ever injects what is currently believed.
Reads are LLM-free (embeddings + closed-form decay math).
"""
import asyncio
import os
import tempfile
from pathlib import Path

os.environ.setdefault("EMBED_PROVIDER", "local")   # keyless demo: local bge-small

from llama_index.core.base.llms.types import ChatMessage, MessageRole  # noqa: E402
from llama_index.core.memory import Memory  # noqa: E402

from tenet import Tenet  # noqa: E402
from tenet.integrations.llamaindex import TenetMemoryBlock  # noqa: E402


async def main() -> None:
    mem = Tenet(Path(tempfile.mkdtemp()) / "li_demo.db")

    # Pre-formed facts (zero-key path). With a DASHSCOPE_API_KEY, block.aput()
    # distills free-form chat into these automatically via Tenet.ingest().
    import time
    mem.store_fact("Alex's home city is Montreal.", key="alex::residence")
    mem.store_fact("Alex works as a data engineer.", key="alex::job")
    t0 = time.time()
    time.sleep(0.01)
    mem.store_fact("Alex's home city is Toronto.", key="alex::residence")  # supersedes

    block = TenetMemoryBlock(tenet=mem, k=5)

    # The block slots straight into LlamaIndex's Memory (what an AgentWorkflow /
    # chat engine consumes as `memory=`):
    li_memory = Memory.from_defaults(session_id="demo", memory_blocks=[block])
    _ = li_memory  # handed to your agent; below we call the block directly to show the mechanism

    out = await block.aget([ChatMessage(role=MessageRole.USER, content="Where does Alex live?")])
    print(out)
    print()
    print("superseded history is preserved, not deleted:")
    then = [m.text for m in mem.recall("Where does Alex live?", k=3, as_of=t0)]
    print("  as_of=t0 (before the move) ->", then)


if __name__ == "__main__":
    asyncio.run(main())
