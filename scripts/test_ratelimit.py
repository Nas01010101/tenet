"""Spend guard on the Qwen-costing endpoints (/chat, /ingest, /recall, /memories,
/forget).

TENET_LIVE_TOKEN, when set, AUTHENTICATES: a wrong or missing token is rejected,
not quietly downgraded to the shared budget. With no token configured the routes
are closed unless the operator sets TENET_LIVE_OPEN=1 to accept an open, billable
endpoint; the daily cap is only a courtesy brake behind that opt-in, since it is
per-process and resets on every cold start. Guard logic only, no LLM calls."""
import os

os.environ["TENET_LIVE_DAILY_CAP"] = "2"
os.environ.pop("TENET_LIVE_TOKEN", None)
os.environ["TENET_LIVE_OPEN"] = "1"  # the cap tests below live behind the opt-in

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


def test_configured_token_authenticates():
    """A wrong or absent token is a 401, not a fall-through to the shared budget."""
    os.environ["TENET_LIVE_TOKEN"] = "s3cret"
    try:
        for bad in (None, "wrong"):
            api._spent.update(day="", calls=0)  # budget wide open; must still reject
            try:
                api._spend_guard(bad)
            except HTTPException as e:
                assert e.status_code == 401, f"expected 401 for {bad!r}, got {e.status_code}"
            else:
                raise AssertionError(f"expected 401 for token {bad!r}, none raised")
    finally:
        del os.environ["TENET_LIVE_TOKEN"]


def test_closed_by_default_without_token_or_optin():
    """No token and no opt-in: billable routes must refuse to spend at all."""
    os.environ.pop("TENET_LIVE_OPEN", None)
    try:
        api._spent.update(day="", calls=0)
        try:
            api._spend_guard(None)
        except HTTPException as e:
            assert e.status_code == 401, f"expected 401, got {e.status_code}"
        else:
            raise AssertionError("expected 401 when closed by default, none raised")
    finally:
        os.environ["TENET_LIVE_OPEN"] = "1"


def test_session_id_cannot_escape_the_session_dir():
    """?session=../../data/tenet must not reach the filesystem."""
    for bad in ("../../data/tenet", "../etc/passwd", "a/b", "", "x" * 65):
        try:
            api._session_db(bad)
        except HTTPException as e:
            assert e.status_code == 400, f"expected 400 for {bad!r}, got {e.status_code}"
        else:
            raise AssertionError(f"expected 400 for session id {bad!r}, none raised")
    assert api._session_db("ok_sid-1").parent == api._SESSION_DIR.resolve()


def test_daily_reset():
    api._spent.update(day="1999-01-01", calls=999)  # stale day, counter maxed
    api._spend_guard(None)  # new day resets the counter before the cap check
    assert api._spent["calls"] == 1, api._spent


if __name__ == "__main__":
    test_cap_enforced()
    test_bypass_token()
    test_daily_reset()
    print("PASS scripts/test_ratelimit.py")
