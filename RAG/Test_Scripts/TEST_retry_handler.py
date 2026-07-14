"""
test_retry_handler.py
Simulates each error path using fake failing functions — no real Gemini
calls are made. Imports retry_handler as a library to keep the two separate.
"""

import types
import httpx
from unittest.mock import MagicMock
from google.genai import errors

from RAG.Pipeline import Retry_Handler as rh

def make_fake_api_error(code, status="UNKNOWN", message="fake error", retry_delay=None):
    """Builds a fake APIError without needing a real Gemini response."""
    e = errors.APIError.__new__(errors.APIError)
    e.code = code
    e.status = status
    e.message = message
    e.details = {}
    if retry_delay:
        e.details = {
            "error": {
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": f"{retry_delay}s",
                }]
            }
        }
    return e


# --- speed up tests: skip real sleeping ---
rh.time.sleep = lambda _: None


def test_retryable_then_success():
    print("\n[TEST] 429 -> succeeds on 2nd attempt")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise make_fake_api_error(429, "RESOURCE_EXHAUSTED", retry_delay=1.5)
        return types.SimpleNamespace(embeddings=[1, 2, 3])

    result = rh.call_gemini_with_handling(flaky)
    assert result.embeddings == [1, 2, 3]
    assert calls["n"] == 2
    print("PASS")


def test_max_attempts_exceeded():
    print("\n[TEST] Always 503 -> exhausts retries and raises")
    def always_fails():
        raise make_fake_api_error(503, "UNAVAILABLE")

    try:
        rh.call_gemini_with_handling(always_fails)
        assert False, "should have raised"
    except errors.APIError as e:
        assert e.code == 503
        print("PASS")


def test_non_retryable_404():
    print("\n[TEST] 404 -> fails fast, no retry")
    calls = {"n": 0}

    def not_found():
        calls["n"] += 1
        raise make_fake_api_error(404, "NOT_FOUND", "model missing")

    try:
        rh.call_gemini_with_handling(not_found, model="fake-model")
        assert False, "should have raised"
    except errors.APIError as e:
        assert e.code == 404
        assert calls["n"] == 1  # only tried once
        print("PASS")


def test_non_retryable_403():
    print("\n[TEST] 403 -> auth failure, fails fast")
    def forbidden():
        raise make_fake_api_error(403, "PERMISSION_DENIED")

    try:
        rh.call_gemini_with_handling(forbidden)
        assert False, "should have raised"
    except errors.APIError as e:
        assert e.code == 403
        print("PASS")


def test_non_retryable_400():
    print("\n[TEST] 400 -> invalid argument, fails fast")
    def bad_request():
        raise make_fake_api_error(400, "INVALID_ARGUMENT")

    try:
        rh.call_gemini_with_handling(bad_request)
        assert False, "should have raised"
    except errors.APIError as e:
        assert e.code == 400
        print("PASS")


def test_network_error_retries():
    print("\n[TEST] ConnectError -> retries then succeeds")
    calls = {"n": 0}

    def flaky_network():
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("connection refused")
        return types.SimpleNamespace(embeddings=[1, 2])

    result = rh.call_gemini_with_handling(flaky_network)
    assert result.embeddings == [1, 2]
    assert calls["n"] == 2
    print("PASS")


def test_network_error_exhausts():
    print("\n[TEST] Always ConnectError -> exhausts retries and raises")
    def always_disconnected():
        raise httpx.ConnectError("connection refused")

    try:
        rh.call_gemini_with_handling(always_disconnected)
        assert False, "should have raised"
    except httpx.ConnectError:
        print("PASS")


def test_empty_response_validation():
    print("\n[TEST] Empty embeddings -> ValueError, no retry")
    calls = {"n": 0}

    def blocked_content():
        calls["n"] += 1
        return types.SimpleNamespace(embeddings=[])

    try:
        rh.call_gemini_with_handling(blocked_content)
        assert False, "should have raised"
    except ValueError as e:
        assert calls["n"] == 1  # not retried
        print("PASS:", e)


def test_retry_delay_extraction():
    print("\n[TEST] retryDelay from error details is respected")
    e = make_fake_api_error(429, "RESOURCE_EXHAUSTED", retry_delay=42)
    delay = rh._extract_retry_delay(e)
    assert delay == 42.0
    print("PASS: extracted", delay)


if __name__ == "__main__":
    test_retryable_then_success()
    test_max_attempts_exceeded()
    test_non_retryable_404()
    test_non_retryable_403()
    test_non_retryable_400()
    test_network_error_retries()
    test_network_error_exhausts()
    test_empty_response_validation()
    test_retry_delay_extraction()
    print("\nAll tests passed.")