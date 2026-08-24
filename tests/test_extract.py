"""
Unit + integration tests for the AI GDS Extraction pipeline (v1.0).

These run on the Windows dev machine with a STUB model backend (no GPU, no
llama.cpp needed). The app's http_model_call is monkeypatched so the full
FastAPI stack can be exercised.

Run:  pytest -v
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
import requests

import gds_extraction_service as gds

# Matches DEFAULT_KEYS used when no .env is present.
VALID_KEY = "gds_key_0000"
ENV_EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
CASES = os.path.join(os.path.dirname(__file__), "cases")


class FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code, text="", payload=None):
        self.status_code = status_code
        self._text = text
        self._payload = payload or {}

    @property
    def text(self):
        return self._text

    def json(self):
        return self._payload


def _messages():
    return [{"role": "user", "content": "hi"}]


def _expected():
    with open(os.path.join(CASES, "amadeus_availability_expected.json")) as f:
        return json.load(f)


def _expected_text():
    with open(os.path.join(CASES, "amadeus_availability_input.txt")) as f:
        return f.read()


@pytest.fixture
def client(monkeypatch):
    calls = {"n": 0}

    def stub(messages, params):
        calls["n"] += 1
        return json.dumps(_expected())

    monkeypatch.setattr(gds, "http_model_call", stub)
    monkeypatch.setattr(gds, "DISABLE_THINKING", True)
    monkeypatch.setattr(gds, "API_KEY_DB", {VALID_KEY: ""})
    monkeypatch.setattr(gds, "_PARAMS_LEVEL", 0)

    app_client = TestClient(gds.app)
    app_client.calls = calls
    return app_client


def _auth():
    return {"x-api-key": VALID_KEY}


def _extract_json():
    return {"gds_text": _expected_text()}


# --------------------------------------------------------------------------
# Pure pipeline — build_prompt
# --------------------------------------------------------------------------
def test_build_prompt_has_system_and_delimited_user():
    msgs = gds.build_prompt("AN3SEPMNLNAN", 2025)
    assert msgs[0]["role"] == "system"
    assert "GDS" in msgs[0]["content"]
    assert "2025" in msgs[0]["content"]
    assert "__DEFAULT_YEAR__" not in "".join(m["content"] for m in msgs)
    assert msgs[1]["role"] == "user"
    assert "<<<GDS_DATA>>>" in msgs[1]["content"]
    assert "<<<END_GDS_DATA>>>" in msgs[1]["content"]
    assert "AN3SEPMNLNAN" in msgs[1]["content"]


def test_build_prompt_replaces_sentinel_only():
    msgs = gds.build_prompt("AN3SEP", 2026)
    assert "2026" in msgs[0]["content"]
    assert "__DEFAULT_YEAR__" not in "".join(m["content"] for m in msgs)


def test_build_prompt_encodes_key_rules():
    content = gds.build_prompt("x", 2025)[0]["content"]
    # Encoding assertions — the rules must be present so extraction is faithful.
    assert "APDI" in content
    assert "sacrosanct" in content
    assert "Philippine Airlines" in content
    assert "Business = C,D,I,J,Z" in content
    assert "B is Economy" in content
    assert "DCPR" in content
    assert "FJ:QF3873" in content


# --------------------------------------------------------------------------
# Params + degradation
# --------------------------------------------------------------------------
def test_build_params_defaults():
    params = gds.build_params()
    assert params["temperature"] == gds.MODEL_TEMP
    assert params["top_p"] == gds.MODEL_TOP_P
    assert params["top_k"] == gds.MODEL_TOP_K
    assert params["min_p"] == 0.0
    assert params["repetition_penalty"] == 1.0
    assert params["max_tokens"] == gds.MODEL_MAX_TOKENS
    assert params["reasoning_effort"] == 0
    assert params["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_params_reasoning_removed_when_enabled(monkeypatch):
    monkeypatch.setattr(gds, "DISABLE_THINKING", False)
    params = gds.build_params()
    assert "reasoning_effort" not in params
    assert "chat_template_kwargs" not in params


def test_build_params_degradation_levels():
    full = gds.build_params(0)
    l1 = gds._params_at_level(full, 1)
    l2 = gds._params_at_level(full, 2)
    assert "chat_template_kwargs" in full
    assert "chat_template_kwargs" not in l1
    assert "reasoning_effort" in l1
    assert "chat_template_kwargs" not in l2
    assert "reasoning_effort" not in l2


# --------------------------------------------------------------------------
# Context guard
# --------------------------------------------------------------------------
def _patch_budget(monkeypatch, ctx=8192, parallel=1, max_tokens=1024):
    monkeypatch.setattr(gds, "CONTEXT_SIZE", ctx)
    monkeypatch.setattr(gds, "MODEL_PARALLEL", parallel)
    monkeypatch.setattr(gds, "MODEL_MAX_TOKENS", max_tokens)
    monkeypatch.setattr(gds, "_SAFETY_MARGIN", 256)
    monkeypatch.setattr(gds, "CONTEXT_GUARD", "strict")


def test_slot_budget_divides_across_slots(monkeypatch):
    _patch_budget(monkeypatch, ctx=8192, parallel=1)
    assert gds.slot_budget() == 8192
    monkeypatch.setattr(gds, "MODEL_PARALLEL", 4)
    assert gds.slot_budget() == 2048  # 8192 // 4


def test_check_context_strict_raises_over_budget(monkeypatch):
    _patch_budget(monkeypatch, ctx=8192, parallel=1, max_tokens=1024)
    room = gds.usable_prompt_room()  # 8192 - 1024 - 256 = 6912
    big = "x" * (room * 3 + 1)  # estimate_tokens => room+1 > room
    with pytest.raises(gds.ContextGuardExceeded):
        gds.check_context([{"content": big}])


def test_check_context_warn_allows_over_budget(monkeypatch):
    _patch_budget(monkeypatch, ctx=8192, parallel=1, max_tokens=1024)
    monkeypatch.setattr(gds, "CONTEXT_GUARD", "warn")
    big = "x" * 100000
    gds.check_context([{"content": big}])  # no raise


def test_check_context_off_allows(monkeypatch):
    _patch_budget(monkeypatch, ctx=64, parallel=1, max_tokens=8)
    monkeypatch.setattr(gds, "CONTEXT_GUARD", "off")
    big = "x" * 100000
    gds.check_context([{"content": big}])  # no raise


def test_check_context_boundary_passes(monkeypatch):
    _patch_budget(monkeypatch, ctx=8192, parallel=1, max_tokens=1024)
    room = gds.usable_prompt_room()
    gds.check_context([{"content": "x" * (room * 3)}])  # exactly at budget passes


def test_estimate_tokens_is_conservative():
    assert gds.estimate_tokens("") == 0
    assert gds.estimate_tokens("abc") == 1
    assert gds.estimate_tokens("abcdef") == 2


# --------------------------------------------------------------------------
# Pure pipeline — extract_json (FAIL CLOSED on unparseable)
# --------------------------------------------------------------------------
def test_extract_json_basic():
    schedule = {"Record type": "none", "Passenger Name": ["none"], "PNR": "none", "segments": []}
    assert gds.extract_json(json.dumps(schedule)) == schedule


def test_extract_json_strips_thinking_blocks():
    raw = '<think>ok</think>{"Record type": "none", "PNR": "ABC123", "Passenger Name": ["none"], "segments": []}'
    assert gds.extract_json(raw)["PNR"] == "ABC123"
    raw2 = "<|begin_thinking|>hmm<|end_thinking|>{\"Record type\": \"none\", \"PNR\": \"X\", \"Passenger Name\": [\"none\"], \"segments\": []}"
    assert gds.extract_json(raw2)["PNR"] == "X"


def test_extract_json_strips_code_fences():
    raw = "```json\n{\"Record type\": \"none\", \"PNR\": \"Z\", \"Passenger Name\": [\"none\"], \"segments\": []}\n```"
    assert gds.extract_json(raw)["PNR"] == "Z"


def test_extract_json_strips_preamble_and_trailing():
    raw = 'Here is the schedule:\n{"Record type": "none", "PNR": "p", "Passenger Name": ["none"], "segments": []}\nHope this helps.'
    assert gds.extract_json(raw)["PNR"] == "p"


def test_extract_json_locates_first_open_last_close():
    raw = 'XYZ preamble {"Record type": "none", "PNR": "q", "Passenger Name": ["none"], "segments": []} end of message'
    assert gds.extract_json(raw)["PNR"] == "q"


def test_extract_json_fails_closed_on_garbage():
    with pytest.raises(gds.ModelUnavailable):
        gds.extract_json("no json here at all")


def test_extract_json_fails_closed_on_none():
    with pytest.raises(gds.ModelUnavailable):
        gds.extract_json(None)


def test_extract_json_fails_closed_on_non_object():
    with pytest.raises(gds.ModelUnavailable):
        gds.extract_json('["not", "an", "object"]')


# --------------------------------------------------------------------------
# Pure pipeline — _normalize_gds (defensive coercion; never invents data)
# --------------------------------------------------------------------------
def test_normalize_gds_empty_defaults():
    assert gds._normalize_gds({}) == {
        "Record type": "none",
        "Passenger Name": ["none"],
        "PNR": "none",
        "segments": [],
    }


def test_normalize_gds_coerces_types():
    raw = {
        "Record type": "none",
        "Passenger Name": ["none"],
        "PNR": "none",
        "segments": [
            {
                "segment_number": "3",
                "airline_code": "QF",
                "flight_number": 20,
                "originating_airport_code": "MNL",
                "departure_date_time": {"month": "9", "date": "3", "year": "2025"},
            }
        ],
    }
    seg = gds._normalize_gds(raw)["segments"][0]
    assert seg["segment_number"] == 3
    assert isinstance(seg["segment_number"], int)
    assert seg["flight_number"] == 20
    # missing optionals default to "none"
    assert seg["airline_name"] == "none"
    assert seg["originating_airport_name"] == "none"
    assert seg["destination_airport_code"] == "none"
    # date-time keys completed with defaults
    dt = seg["departure_date_time"]
    assert dt["month"] == 9 and dt["date"] == 3 and dt["year"] == 2025
    assert dt["month_name"] == "none"
    assert dt["day_of_week"] == "none"
    assert dt["time"] == "none"


def test_normalize_gds_drops_junk_keys():
    raw = {
        "Record type": "none",
        "Passenger Name": ["none"],
        "PNR": "none",
        "segments": [],
        "extra_unexpected_key": "should be dropped",
        "meta": 42,
    }
    assert "extra_unexpected_key" not in gds._normalize_gds(raw)
    assert "meta" not in gds._normalize_gds(raw)


def test_normalize_gds_passenger_name_scalar_and_empty():
    assert gds._normalize_gds({"Record type": "none", "Passenger Name": "SMITH/JOHN", "PNR": "none", "segments": []})[
        "Passenger Name"
    ] == ["SMITH/JOHN"]
    assert gds._normalize_gds({"Record type": "none", "Passenger Name": [], "PNR": "none", "segments": []})[
        "Passenger Name"
    ] == ["none"]


def test_normalize_gds_record_type_default():
    for bad in ("", "RESERVATION", 0, None):
        assert gds._normalize_gds({"Record type": bad, "Passenger Name": ["none"], "PNR": "none", "segments": []})[
            "Record type"
        ] == "none"


# --------------------------------------------------------------------------
# Pure pipeline — run_extract
# --------------------------------------------------------------------------
def test_run_extract_returns_schedule():
    out = gds.run_extract({"gds_text": "x"}, 2025, lambda m, p: json.dumps({"segments": []}))
    assert out["segments"] == []
    assert out["Record type"] == "none"


def test_run_extract_empty_arguments_passthrough():
    out = gds.run_extract({"gds_text": _expected_text()}, 2025, lambda m, p: json.dumps(_expected()))
    assert len(out["segments"]) == 10
    assert out["Record type"] == "none"
    assert out["segments"][0]["flight_number"] == 221
    assert out["segments"][0]["aircraft_type"] == "Airbus A321"


def test_run_extract_requires_nonempty():
    for payload in (
        {"gds_text": ""},
        {"gds_text": "   "},
        {"gds_text": 123},
        {"not_gds": "x"},
    ):
        with pytest.raises(ValueError):
            gds.run_extract(payload, 2025, lambda m, p: "")


def test_run_extract_guard_trips_before_model(monkeypatch):
    _patch_budget(monkeypatch, ctx=8192, parallel=1, max_tokens=1024)
    called = {"n": 0}

    def boom(m, p):
        called["n"] += 1
        return "{}"

    with pytest.raises(gds.ContextGuardExceeded):
        gds.run_extract({"gds_text": "x" * 30000}, 2025, boom)
    assert called["n"] == 0  # rejected before any network call


def test_run_extract_batch_isolates_failures():
    def stub(messages, params):
        text = messages[-1]["content"]
        if "boom" in text:
            raise gds.ModelUnavailable("unparseable output")
        return json.dumps({"Record type": "none", "PNR": "OK", "Passenger Name": ["none"], "segments": []})

    entries = [
        {"id": "r1", "gds_text": "AN3SEPMNLNAN"},
        {"id": "r2", "gds_text": "boom"},
        {"id": "r3", "gds_text": "AN3SEPMNLNAN"},
    ]
    results = gds.run_extract_batch(entries, 2025, stub)
    assert [r["status"] for r in results] == ["ok", "error", "ok"]
    assert results[0]["schedule"]["PNR"] == "OK"
    assert results[1]["id"] == "r2"
    assert "unparseable" in results[1]["error"]
    assert results[2]["schedule"]["PNR"] == "OK"


# --------------------------------------------------------------------------
# resolve_default_year
# --------------------------------------------------------------------------
def test_resolve_default_year_current(monkeypatch):
    monkeypatch.setattr(gds, "DEFAULT_YEAR_ENV", "")
    assert gds.resolve_default_year() == __import__("datetime").date.today().year


def test_resolve_default_year_pinned(monkeypatch):
    monkeypatch.setattr(gds, "DEFAULT_YEAR_ENV", "2025")
    assert gds.resolve_default_year() == 2025


def test_resolve_default_year_bad_falls_back(monkeypatch):
    monkeypatch.setattr(gds, "DEFAULT_YEAR_ENV", "not-a-year")
    assert gds.resolve_default_year() == __import__("datetime").date.today().year


# --------------------------------------------------------------------------
# HTTP layer (via TestClient + stub backend)
# --------------------------------------------------------------------------
def test_extract_http(client):
    r = client.post("/v1/extract", json=_extract_json(), headers=_auth())
    assert r.status_code == 200
    assert r.json()["segments"][0]["flight_number"] == 221
    assert r.headers["content-type"].startswith("application/json")


def test_extract_http_matches_golden(client):
    r = client.post("/v1/extract", json=_extract_json(), headers=_auth())
    assert r.status_code == 200
    assert r.json() == _expected()


def test_extract_http_missing_field_422(client):
    r = client.post("/v1/extract", json={"gds_text": ""}, headers=_auth())
    assert r.status_code == 422


def test_extract_http_over_budget_422(monkeypatch):
    monkeypatch.setattr(gds, "API_KEY_DB", {VALID_KEY: ""})
    _patch_budget(monkeypatch, ctx=8192, parallel=1, max_tokens=1024)
    c = TestClient(gds.app)
    r = c.post("/v1/extract", json={"gds_text": "x" * 30000}, headers=_auth())
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "slot" in detail and "token" in detail


def test_extract_batch_http(client):
    r = client.post(
        "/v1/extract_batch",
        json={"entries": [{"id": "r1", "gds_text": _expected_text()}]},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["status"] == "ok"
    assert r.json()["results"][0]["schedule"]["segments"][0]["flight_number"] == 221


def test_extract_batch_http_isolates(monkeypatch):
    # Override the global stub fixture: a bad entry must fail per-entry while the
    # good entries still succeed (overall status stays 200).
    monkeypatch.setattr(gds, "API_KEY_DB", {VALID_KEY: ""})
    monkeypatch.setattr(gds, "_PARAMS_LEVEL", 0)

    def fn(messages, params):
        if "garbage" in messages[-1]["content"]:
            raise gds.ModelUnavailable("unparseable output")
        return json.dumps(_expected())

    monkeypatch.setattr(gds, "http_model_call", fn)
    c = TestClient(gds.app)
    r = c.post(
        "/v1/extract_batch",
        json={
            "entries": [
                {"id": "r1", "gds_text": "AN3SEP"},
                {"id": "r2", "gds_text": "garbage-no-json"},
            ]
        },
        headers=_auth(),
    )
    assert r.status_code == 200
    statuses = {item["id"]: item["status"] for item in r.json()["results"]}
    assert statuses["r1"] == "ok"
    assert statuses["r2"] == "error"


def test_extract_batch_http_empty_entries_422(client):
    r = client.post("/v1/extract_batch", json={"entries": []}, headers=_auth())
    assert r.status_code == 422


def test_extract_batch_http_unavailable_all_fail(client):
    def boom(messages, params):
        raise gds.ModelUnavailable("server down")

    gds.http_model_call = boom
    r = client.post(
        "/v1/extract_batch",
        json={"entries": [{"id": "r1", "gds_text": "x"}, {"id": "r2", "gds_text": "y"}]},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert all(item["status"] == "error" for item in r.json()["results"])


def test_version_http_does_not_call_model(client):
    r = client.post("/v1/version", headers=_auth())
    assert r.status_code == 200
    assert r.json() == {"version": gds.VERSION}
    assert client.calls["n"] == 0


def test_missing_api_key_http_401(client):
    r = client.post("/v1/extract", json=_extract_json())
    assert r.status_code == 401


def test_bad_api_key_http_401(client):
    r = client.post("/v1/extract", json=_extract_json(), headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_extract_http_unavailable_503(monkeypatch):
    monkeypatch.setattr(gds, "API_KEY_DB", {VALID_KEY: ""})

    def boom(messages, params):
        raise gds.ModelUnavailable("refused")

    monkeypatch.setattr(gds, "http_model_call", boom)
    c = TestClient(gds.app)
    r = c.post("/v1/extract", json=_extract_json(), headers=_auth())
    assert r.status_code == 503


def test_extract_http_unparseable_503(monkeypatch):
    monkeypatch.setattr(gds, "API_KEY_DB", {VALID_KEY: ""})

    def garbage(messages, params):
        return "this is not json"

    monkeypatch.setattr(gds, "http_model_call", garbage)
    c = TestClient(gds.app)
    r = c.post("/v1/extract", json=_extract_json(), headers=_auth())
    assert r.status_code == 503
    assert "json" in r.json()["detail"].lower()


def test_analyze_http_cleaned_through_backend(monkeypatch):
    # Messy model output (thinking + fence + preamble) still yields clean JSON.
    monkeypatch.setattr(gds, "API_KEY_DB", {VALID_KEY: ""})

    def leaky(messages, params):
        return ("<think>think</think> Here is it:\n"
                "```json\n"
                '{"Record type": "none", "PNR": "clean", "Passenger Name": ["none"], "segments": []}\n'
                "```")

    monkeypatch.setattr(gds, "http_model_call", leaky)
    c = TestClient(gds.app)
    r = c.post("/v1/extract", json=_extract_json(), headers=_auth())
    assert r.status_code == 200
    assert r.json()["PNR"] == "clean"


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["version"] == gds.VERSION


def test_healthz_endpoint(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["version"] == gds.VERSION


def test_healthz_reports_context_check(monkeypatch):
    def fake_get(url, timeout=None):
        if url.endswith("/health"):
            return FakeResp(200, "", {"ctx_size": 65536})
        if url.endswith("/props"):
            return FakeResp(200, "", {"default_generation_settings": {"n_ctx": gds.slot_budget()}})
        return FakeResp(500, "", {})

    monkeypatch.setattr(gds.requests, "get", fake_get)
    with TestClient(gds.app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["model_server"] == "ready"
    assert body["context_budget"]["check"] == "match"
    assert body["context_budget"]["slot_tokens"] == gds.slot_budget()
    assert body["version"] == gds.VERSION


# --------------------------------------------------------------------------
# Model backend — http_model_call (degradation, resolution, error mapping)
# --------------------------------------------------------------------------
def test_http_call_connection_error_raises_model_unavailable(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(gds.requests, "post", boom)
    with pytest.raises(gds.ModelUnavailable):
        gds.http_model_call(_messages(), {"temperature": 0.0})


def test_http_call_timeout_raises_model_unavailable(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(gds.requests, "post", boom)
    with pytest.raises(gds.ModelUnavailable):
        gds.http_model_call(_messages(), {})


def test_http_call_non_200_raises_model_unavailable(monkeypatch):
    monkeypatch.setattr(gds.requests, "post", lambda *a, **k: FakeResp(500, "server boom"))
    with pytest.raises(gds.ModelUnavailable):
        gds.http_model_call(_messages(), {})


def test_http_call_200_returns_content(monkeypatch):
    monkeypatch.setattr(
        gds.requests, "post", lambda *a, **k: FakeResp(200, "", {"choices": [{"message": {"content": "GOOD"}}]})
    )
    assert gds.http_model_call(_messages(), {}) == "GOOD"


def test_http_call_empty_content_raises(monkeypatch):
    monkeypatch.setattr(
        gds.requests, "post", lambda *a, **k: FakeResp(200, "", {"choices": [{"message": {"content": ""}}]})
    )
    with pytest.raises(gds.ModelUnavailable):
        gds.http_model_call(_messages(), {})


def test_http_call_reasoning_content_fallback(monkeypatch):
    monkeypatch.setattr(
        gds.requests,
        "post",
        lambda *a, **k: FakeResp(
            200, "",
            {
                "choices": [
                    {"message": {"content": "", "reasoning_content": "```json\n{\"PNR\": \"x\"}\n```"}}
                ]
            },
        ),
    )
    raw = gds.http_model_call(_messages(), gds.build_params())
    assert "```json" in raw
    assert gds.extract_json(raw)["PNR"] == "x"


def test_http_call_context_length_raises_context_exceeded(monkeypatch):
    monkeypatch.setattr(
        gds.requests, "post", lambda *a, **k: FakeResp(400, "context length exceeded", {"error": "bad"})
    )
    with pytest.raises(gds.ContextExceeded):
        gds.http_model_call(_messages(), {})


def test_http_call_degrades_chat_template_kwargs(monkeypatch):
    monkeypatch.setattr(gds, "_PARAMS_LEVEL", 0)  # fresh starting level
    calls = {"n": 0}
    seen = []

    def fake(*a, **k):
        calls["n"] += 1
        seen.append(k.get("json", {}).get("chat_template_kwargs"))
        if calls["n"] == 1:
            return FakeResp(400, "chat_template_kwargs not supported", {"error": "bad"})
        return FakeResp(200, "", {"choices": [{"message": {"content": "cleaned"}}]})

    monkeypatch.setattr(gds.requests, "post", fake)
    out = gds.http_model_call(_messages(), gds.build_params())
    assert out == "cleaned"
    assert calls["n"] == 2
    assert seen[0] == {"enable_thinking": False}
    assert seen[1] is None  # dropped at level 1


def test_http_call_degrades_two_levels(monkeypatch):
    monkeypatch.setattr(gds, "_PARAMS_LEVEL", 0)
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(400, "chat_template_kwargs not supported", {})
        if calls["n"] == 2:
            return FakeResp(400, "reasoning_effort not supported", {})
        return FakeResp(200, "", {"choices": [{"message": {"content": "x"}}]})

    monkeypatch.setattr(gds.requests, "post", fake)
    out = gds.http_model_call(_messages(), gds.build_params())
    assert out == "x"
    assert calls["n"] == 3


def test_http_call_caches_working_level_after_degradation(monkeypatch):
    monkeypatch.setattr(gds, "DISABLE_THINKING", True)
    monkeypatch.setattr(gds, "_PARAMS_LEVEL", 0)

    def ok(*a, **k):
        return FakeResp(200, "", {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(gds.requests, "post", ok)
    gds.http_model_call(_messages(), gds.build_params())  # level 0 succeeds → cache 0
    assert gds._PARAMS_LEVEL == 0

    def reject_kw(*a, **k):
        body = k.get("json", {})
        if body.get("chat_template_kwargs") is not None:
            return FakeResp(400, "chat_template_kwargs not supported", {})
        return FakeResp(200, "", {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(gds.requests, "post", reject_kw)
    gds.http_model_call(_messages(), gds.build_params())  # 0 fails → 1 succeeds → cache 1
    assert gds._PARAMS_LEVEL == 1

    calls = {"n": 0}

    def count(*a, **k):
        calls["n"] += 1
        return FakeResp(200, "", {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(gds.requests, "post", count)
    gds.http_model_call(_messages(), gds.build_params())
    assert calls["n"] == 1  # starts at cached level 1, no retry


def test_http_call_sends_bearer_only_when_key_present(monkeypatch):
    sent = {}

    def capture(*a, **k):
        sent["headers"] = k.get("headers")
        return FakeResp(200, "", {"choices": [{"message": {"content": "x"}}]})

    monkeypatch.setattr(gds.requests, "post", capture)
    gds.http_model_call(_messages(), {})
    assert sent["headers"].get("Authorization") == f"Bearer {gds.LLAMA_SERVER_API_KEY}"


# --------------------------------------------------------------------------
# Config drift — .env.example must match code defaults (RC3 discipline)
# --------------------------------------------------------------------------
def test_defaults_sync_to_env_example():
    settings = {}
    with open(ENV_EXAMPLE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            settings[key.strip()] = val.strip().strip('"').strip("'")

    assert int(settings["CONTEXT_SIZE"]) == gds.CONTEXT_SIZE
    assert int(settings["MODEL_PARALLEL"]) == gds.MODEL_PARALLEL
    assert int(settings["API_PORT"]) == gds.API_PORT
    assert int(settings["MODEL_MAX_TOKENS"]) == gds.MODEL_MAX_TOKENS
    assert int(settings["REQUEST_TIMEOUT"]) == gds.REQUEST_TIMEOUT
    assert float(settings["MODEL_TEMP"]) == gds.MODEL_TEMP
    assert float(settings["MODEL_TOP_P"]) == gds.MODEL_TOP_P
    assert int(settings["MODEL_TOP_K"]) == gds.MODEL_TOP_K
    assert settings["LLAMA_SERVER_API_KEY"] == gds.LLAMA_SERVER_API_KEY
    assert settings["MODEL_NAME"] == gds.MODEL_NAME
