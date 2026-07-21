"""Spend guard: anonymous callers to the Qwen-costing endpoints (/chat, /ingest,
/recall, /memories) share a daily cap, so the public URL cannot be spammed to run
up the owner's Qwen bill. The X-Tenet-Token header (matching env TENET_LIVE_TOKEN)
bypasses the cap for the owner. This tests the guard logic directly, no LLM calls."""
import os

os.environ["TENET_LIVE_DAILY_CAP"] = "2"
os.environ.pop("TENET_LIVE_TOKEN", None)

from fastapi import HTTPException  # noqa: E402

import tenet.api as api  # noqa: E402


def test_cap_enforced():
    api._spent.update(day="", calls=0)
    api._spend_guard(None)  # call 1 — allowed
    api._spend_guard(None)  # call 2 — allowed
    try:
        api._spend_guard(None)  # call 3 — over the cap
    except HTTPException as e:
        assert e.status_code == 429, f"expected 429, got {e.status_code}"
    else:
        raise AssertionError("expected 429 once the daily cap is spent, none raised")


def test_bypass_token():
    os.environ["TENET_LIVE_TOKEN"] = "s3cret"
    try:
        api._spent.update(day="", calls=999)  # cap already blown for anonymous
        api._spend_guard("s3cret")  # matching token bypasses, must not raise
    finally:
        del os.environ["TENET_LIVE_TOKEN"]


def test_daily_reset():
    api._spent.update(day="1999-01-01", calls=999)  # stale day, counter maxed
    api._spend_guard(None)  # new day resets the counter before the cap check
    assert api._spent["calls"] == 1, api._spent


if __name__ == "__main__":
    test_cap_enforced()
    test_bypass_token()
    test_daily_reset()
    print("PASS scripts/test_ratelimit.py")
