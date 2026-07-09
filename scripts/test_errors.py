"""Deterministic tests for provider-error handling (config.chat's fail-loud path)
and the write-surface warnings it feeds — no network, the chat client is stubbed.

Regression coverage for the bug: chat() used to retry ALL exceptions then return
"" — permanent errors (bad key/quota/payment) were retried pointlessly and
swallowed to empty string, so distill() saw "" -> "{}" -> [] and ingest() silently
"learned 0 facts" on a genuine provider outage. Now permanent failures raise
config.ProviderError immediately (or after exhausting retries for anything else),
and every write surface (agent/mcp/api/cli) surfaces that instead of pretending
success.

Run: EMBED_PROVIDER=local python scripts/test_errors.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet import config  # noqa: E402
from tenet.agent import MemoryAgent  # noqa: E402
from tenet.config import ProviderError  # noqa: E402
from tenet.core import Tenet  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- fake OpenAI-SDK-shaped client: scripted sequence of Exception | str -----

class _FakeResp:
    def __init__(self, content):
        msg = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": msg})()
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)  # consumed left-to-right; leftovers prove no over-retry

    def create(self, **_kw):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResp(item)


class _FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(script)})()


def _stub(script):
    """Patch config.chat_client() to return a fake client that plays back `script`."""
    fake = _FakeClient(script)
    config.chat_client = lambda: fake
    return fake


def _run(fn):
    orig_client, orig_sleep = config.chat_client, time.sleep
    time.sleep = lambda *_a, **_kw: None  # no real backoff waits in tests
    try:
        fn()
    finally:
        config.chat_client = orig_client
        time.sleep = orig_sleep


# ---- tests -------------------------------------------------------------------

def test_permanent_error_raises_immediately():
    err = Exception("Error code: 403 - {'code': 'AllocationQuota.FreeTierOnly', "
                     "'message': 'quota exhausted'}")
    fake = _stub([err, err, err, err, err])  # would blow up every attempt if retried
    try:
        config.chat([{"role": "user", "content": "hi"}], qwen_default="test-model")
        check("403 raises ProviderError (no silent '')", False, "no exception raised")
    except ProviderError as e:
        check("403 raises ProviderError (no silent '')", True)
        check("reason preserved verbatim", "quota exhausted" in e.reason, e.reason)
        check("no retry consumed — permanent error fails on first attempt",
              len(fake.chat.completions.script) == 4,
              f"{len(fake.chat.completions.script)} left")


def test_401_and_402_also_permanent():
    for code, label in [("401", "auth"), ("402", "payment")]:
        err = Exception(f"Error code: {code} - insufficient balance, payment required")
        _stub([err, err, err, err, err])
        try:
            config.chat([{"role": "user", "content": "hi"}], qwen_default="test-model")
            check(f"{label} ({code}) raises ProviderError", False)
        except ProviderError:
            check(f"{label} ({code}) raises ProviderError", True)


def test_rate_limit_retries_then_succeeds():
    err = Exception("429 rate limit exceeded, please retry")
    _stub([err, "ok reply"])
    result = config.chat([{"role": "user", "content": "hi"}], qwen_default="test-model")
    check("transient 429 still retries then succeeds", result == "ok reply", repr(result))


def test_exhausted_retries_raise_not_empty():
    err = Exception("temporary upstream hiccup")  # not permanent, not 429 either
    _stub([err] * 5)
    try:
        config.chat([{"role": "user", "content": "hi"}], qwen_default="test-model")
        check("exhausted retries raise ProviderError (not '')", False)
    except ProviderError as e:
        check("exhausted retries raise ProviderError (not '')", True, e.reason)


def test_genuine_empty_content_still_returns_empty_string():
    """A real (non-error) empty completion is not a provider failure — chat()
    must not conflate 'the model said nothing' with 'the provider is down'."""
    _stub([""])
    result = config.chat([{"role": "user", "content": "hi"}], qwen_default="test-model")
    check("genuine empty response still returns ''", result == "", repr(result))


def test_distill_ingest_propagates_provider_error():
    err = Exception("Error code: 403 - AllocationQuota.FreeTierOnly: quota exhausted")
    _stub([err])
    db = Path(tempfile.mkdtemp()) / "ingest_err.db"
    m = Tenet(db)
    try:
        m.ingest("I live in Boston")
        check("Tenet.ingest() propagates ProviderError (library raises)", False)
    except ProviderError as e:
        check("Tenet.ingest() propagates ProviderError (library raises)", True)
        check("ingest() error reason preserved", "quota exhausted" in e.reason, e.reason)
    finally:
        m.close()


def test_agent_ingest_failure_produces_warning():
    """agent.respond(): reply generation succeeds, ingest fails — the reply must
    still be delivered, with a visible warning, not a silently-swallowed write."""
    db = Path(tempfile.mkdtemp()) / "agent_err.db"
    agent = MemoryAgent(db)
    orig_chat = config.chat
    config.chat = lambda *_a, **_kw: "a normal reply"

    def _boom(*_a, **_kw):
        raise ProviderError("qwen", "test-model", "quota exhausted")
    agent.m.ingest = _boom
    try:
        out = agent.respond("I live in Boston")
    finally:
        config.chat = orig_chat
        agent.m.close()

    check("respond() doesn't crash when ingest fails", True)
    check("out['learned'] is 0, not silently nonzero", out["learned"] == 0, out["learned"])
    check("out['warning'] names the failure",
          out.get("warning") == "memory write failed: quota exhausted — this turn was not memorized",
          out.get("warning"))
    check("reply still delivered", "a normal reply" in out["reply"], out["reply"])
    check("warning visible in the reply text too", "memory write failed" in out["reply"], out["reply"])


def main() -> int:
    for fn in [
        test_permanent_error_raises_immediately,
        test_401_and_402_also_permanent,
        test_rate_limit_retries_then_succeeds,
        test_exhausted_retries_raise_not_empty,
        test_genuine_empty_content_still_returns_empty_string,
        test_distill_ingest_propagates_provider_error,
        test_agent_ingest_failure_produces_warning,
    ]:
        print(f"[{fn.__name__}]")
        _run(fn)

    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  -", f)
        return 1
    print("\nERRORS ALL PASS ✅  (permanent failures fail loud, transient ones still retry)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
