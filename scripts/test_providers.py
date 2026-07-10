"""Deterministic tests for the ollama embedding provider (config.embed_texts /
EMBED_PROVIDER=ollama) — no network, the embeddings client is stubbed.

Covers the fully-local stack's embed half: OLLAMA_BASE_URL's OpenAI-compatible
/v1/embeddings endpoint, batching, unit-normalisation, and the same fail-loud
contract as the qwen path (a permanent error raises ProviderError instead of
silently degrading into a zero/garbage vector).

Run: python scripts/test_providers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tenet import config  # noqa: E402
from tenet.config import ProviderError  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- fake OpenAI-SDK-shaped embeddings client --------------------------------

class _FakeEmbedding:
    def __init__(self, vec):
        self.embedding = vec


class _FakeEmbeddingsResp:
    def __init__(self, vecs):
        self.data = [_FakeEmbedding(v) for v in vecs]


class _FakeEmbeddings:
    def __init__(self, script):
        self.script = list(script)  # consumed left-to-right per batch call
        self.calls = []

    def create(self, *, model, input):  # noqa: A002 - matches OpenAI SDK kwarg name
        self.calls.append((model, list(input)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeEmbeddingsResp(item)


class _FakeClient:
    def __init__(self, script):
        self.embeddings = _FakeEmbeddings(script)


def _stub(script):
    """Patch config.ollama_client() to return a fake client that plays back `script`."""
    fake = _FakeClient(script)
    config.ollama_client = lambda: fake
    return fake


def _run(fn):
    orig_client, orig_provider = config.ollama_client, config.EMBED_PROVIDER
    orig_sleep = __import__("time").sleep
    __import__("time").sleep = lambda *_a, **_kw: None  # no real backoff waits
    config.EMBED_PROVIDER = "ollama"
    try:
        fn()
    finally:
        config.ollama_client = orig_client
        config.EMBED_PROVIDER = orig_provider
        __import__("time").sleep = orig_sleep


# ---- tests --------------------------------------------------------------------

def test_ollama_embed_success_unit_normalised():
    fake = _stub([[[3.0, 4.0]]])  # one batch, one vector, norm=5
    vecs = config.embed_texts(["hello"])
    check("returns one vector", len(vecs) == 1, len(vecs))
    v = vecs[0]
    check("vector is unit-normalised", abs(float((v ** 2).sum()) - 1.0) < 1e-6, float((v ** 2).sum()))
    check("uses OLLAMA_EMBED_MODEL", fake.embeddings.calls[0][0] == config.OLLAMA_EMBED_MODEL,
          fake.embeddings.calls[0][0])


def test_ollama_embed_batches_in_groups_of_ten():
    texts = [f"t{i}" for i in range(23)]
    fake = _stub([[[1.0, 0.0]] * 10, [[1.0, 0.0]] * 10, [[1.0, 0.0]] * 3])
    vecs = config.embed_texts(texts)
    check("returns one vector per input text", len(vecs) == 23, len(vecs))
    check("batched into groups of <=10", [len(c[1]) for c in fake.embeddings.calls] == [10, 10, 3],
          [len(c[1]) for c in fake.embeddings.calls])


def test_ollama_embed_permanent_error_raises_immediately():
    err = Exception("Error code: 401 - invalid api key")
    fake = _stub([err, err, err, err, err])  # would blow up every attempt if retried
    try:
        config.embed_texts(["hi"])
        check("401 raises ProviderError (no silent zero-vector)", False, "no exception raised")
    except ProviderError as e:
        check("401 raises ProviderError (no silent zero-vector)", True)
        check("provider name is 'ollama'", e.provider == "ollama", e.provider)
        check("no retry consumed — permanent error fails on first attempt",
              len(fake.embeddings.script) == 4, f"{len(fake.embeddings.script)} left")


def test_ollama_embed_transient_error_retries_then_succeeds():
    err = Exception("connection reset, temporarily unavailable")
    _stub([err, [[0.0, 5.0]]])
    vecs = config.embed_texts(["hi"])
    check("transient error still retries then succeeds", len(vecs) == 1, len(vecs))


def test_ollama_embed_exhausted_retries_raise_not_silent():
    err = Exception("temporary upstream hiccup")  # not permanent, not rate-limit either
    _stub([err] * 5)
    try:
        config.embed_texts(["hi"])
        check("exhausted retries raise ProviderError (not a zero vector)", False)
    except ProviderError as e:
        check("exhausted retries raise ProviderError (not a zero vector)", True, e.reason)


def main() -> int:
    for fn in [
        test_ollama_embed_success_unit_normalised,
        test_ollama_embed_batches_in_groups_of_ten,
        test_ollama_embed_permanent_error_raises_immediately,
        test_ollama_embed_transient_error_retries_then_succeeds,
        test_ollama_embed_exhausted_retries_raise_not_silent,
    ]:
        print(f"[{fn.__name__}]")
        _run(fn)

    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  -", f)
        return 1
    print("\nPROVIDER TESTS ALL PASS ✅  (ollama embeddings: batched, normalised, fail-loud)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
