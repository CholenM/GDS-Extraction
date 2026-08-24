"""
AI GDS Extraction — FastAPI Service (v1.0)
======================================================================
Lean, JSON-in / JSON-out **GDS Extractor** ("make schedule") for a travel
agency. Parses raw GDS output lines (Amadeus availability displays and complete
reservations) into the structured flight-segment JSON defined in the project
spec, and runs fully offline on the NVIDIA DGX Spark by reusing Toby's
already-running llama.cpp (CUDA) model server. This service is a **gateway
only** — it never starts or stops the model server (see start.sh / stop.sh).

Architecture (mirrors the Proof-Reader / QA-Manager skeleton):

    Client  --POST /v1/extract-->  FastAPI gateway (:8084)
            {gds_text}
            {entries: [{id, gds_text}]}
                                           |
                                           v
                                      llama-server (:8006, shared server)
                                      Qwen3.8-27B, CUDA, text mode (--jinja)

The core logic (build_prompt / build_params / estimate_tokens / check_context /
extract_json / _normalize_gds / run_extract / run_extract_batch) are pure
functions taking an injectable ``model_call`` callable, so the entire business
logic is unit-tested on the Windows dev machine with a stub backend — no GPU
needed.

Usage:
    python gds_extraction_service.py      # reads config from .env
    uvicorn gds_extraction_service:app    # alternative via uvicorn CLI
"""

from __future__ import annotations

import os
import re
import json
import logging
from datetime import date
from typing import Callable, Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests

# ---------------------------------------------------------------------------
# Configuration — all values sourced from .env (see .env.example).
#
# NOTE: CONTEXT_SIZE and MODEL_PARALLEL mirror the SHARED server's launch flags
# (Proof-Reader's startserver.sh). They exist here only to compute the local
# per-slot context budget for the client-side guard and the /healthz cross-check.
# ---------------------------------------------------------------------------
load_dotenv()

CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "65536"))
MODEL_PARALLEL = int(os.getenv("MODEL_PARALLEL", "4"))
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:8006/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.8-27B")
# Internal bearer token the gateway presents to the shared llama-server. Sent
# ONLY if non-empty. THIS MUST MATCH the shared server's --api-key, or the
# server rejects the call with a 401 (the gateway then returns a 503 on every
# extract). Toby's shared server is started with --api-key sk-internal-proofreader.
LLAMA_SERVER_API_KEY = os.getenv("LLAMA_SERVER_API_KEY", "sk-internal-proofreader")
# Greedy decoding (temp=0) for run-to-run determinism on factual GDS parsing;
# validated on-DGX with byte-identical output across repeated runs.
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.0"))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", "0.5"))
MODEL_TOP_K = int(os.getenv("MODEL_TOP_K", "40"))
MODEL_MAX_TOKENS = max(64, int(os.getenv("MODEL_MAX_TOKENS", "8192")))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1").lower() == "1"
# Default YEAR used when the GDS line omits a year. Empty string => current
# year at request time.
DEFAULT_YEAR_ENV = os.getenv("DEFAULT_YEAR", "").strip()
# Client-side context guard:
#   strict (default) -> reject over-budget requests with a 422 + guidance
#   warn             -> allow the request but log a warning
#   off              -> skip the guard entirely (for testing only)
CONTEXT_GUARD = os.getenv("CONTEXT_GUARD", "strict").strip().lower()
API_KEY_AUTH_HEADER = os.getenv("API_KEY_AUTH_HEADER", "x-api-key").lower()
API_PORT = int(os.getenv("API_PORT", "8084"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# The documented service version.
VERSION = "1.0"

# Sentinel substituted into the system prompt per request (the JSON braces in the
# instructions must NOT go through str.format / an f-string).
_DEFAULT_YEAR_SENTINEL = "__DEFAULT_YEAR__"

# Tokens reserved beyond the output budget as headroom (safety margin).
_SAFETY_MARGIN = 256
# Degradation levels for chat-template / reasoning suppression (see roadmap).
# Level 0 = full params, level 1 drops chat_template_kwargs,
# level 2 drops reasoning_effort on top of level 1.
_PARAMS_LEVELS = (0, 1, 2)
# Last known-working degradation level, cached to avoid retrying on every call.
_PARAMS_LEVEL = 0

# Sampling parameters forwarded to the model request. Recognized keys must also
# be accepted by llama.cpp's OpenAI-compatible API.
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "max_tokens",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gds_extraction_service")

# ---------------------------------------------------------------------------
# Lean API-key database — loaded from .env, kept in memory.
# Format: key1:Label1,key2:Label2,...  (labels are cosmetic / for logging)
# ---------------------------------------------------------------------------
DEFAULT_KEYS = "gds_key_0000:GDS Extraction Local Testing"


def _parse_api_keys(raw: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 1)
        key = parts[0].strip()
        if key:
            keys[key] = parts[1].strip() if len(parts) > 1 else ""
    return keys


API_KEY_DB = _parse_api_keys(os.getenv("API_KEYS", DEFAULT_KEYS))
logger.info(f"Loaded {len(API_KEY_DB)} API key(s): {list(API_KEY_DB.values()) or '(none)'}")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ModelUnavailable(RuntimeError):
    """Raised when the model server cannot satisfy a request."""


class ContextExceeded(RuntimeError):
    """Raised when the model server rejects the request because it overflows
    the (fixed) context window. Maps to HTTP 503 (server-side)."""


class ContextGuardExceeded(ValueError):
    """Raised by the client-side guard before any network call when the prompt
    cannot fit in a slot. Maps to HTTP 422 (client-side, actionable)."""


# ===========================================================================
# PURE PIPELINE  (unit-tested on the dev machine with a stub model_call)
# ===========================================================================
# The system prompt encodes every parse rule from Toby's spec (AI_GDS_EXTRACTION.md).
# ``__DEFAULT_YEAR__`` is a sentinel replaced at call time.
GDS_SYSTEM = """You are a meticulous Global Distribution System (GDS) flight-data extractor.
You parse the provided GDS output and return ONLY a JSON object describing the
flight schedule. No commentary, no explanations, no markdown outside the JSON.

The DEFAULT YEAR is __DEFAULT_YEAR__. Use it for any date where the GDS line
omits a year entirely.

OUTPUT ONLY THIS JSON OBJECT (exact key names shown; do not add other keys):
{{
  "Record type": "reservation" | "none",
  "Passenger Name": [ "LASTNAME/FIRSTNAME(s)", ... ]  OR  [ "none" ],
  "PNR": "<6-char PNR>" | "none",
  "segments": [ {{ ...segment... }} ]
}}

Record type: set to "reservation" if the record contains one or more record
locators AND at least one passenger name; otherwise set to "none".
PNR: the FIRST PNR in the whole record (NOT a per-segment locator). "none" for
availability displays.

Passenger names (for reservations):
- Start AFTER the numerical passenger number, which marks the beginning of the
  passenger name.
- Capture everything until the first slash '/' (that is the boundary between
  last name and first names).
- Preserve the ORIGINAL format "LASTNAME/FIRSTNAME(s)" exactly, INCLUDING any
  prefix and ANY trailing characters that follow the locator digits.
- Extract EVERY passenger in the record (up to 9). If there are none, output
  [ "none" ].
- If a passenger's last name begins with "APDI", keep the ENTIRE last name
  including the "APDI" characters in the output, while listing all other names.

For an AVAILABILITY DISPLAY (no PNR, no passenger names):
- "Record type": "none", "Passenger Name": ["none"], "PNR": "none".
- For each segment set "segment_record_locator": "none",
  "service_class_letter": "none", "service_class": "none".
  (Availability counts such as "J5 C5 D5" are NOT a service class of service.)

For EACH flight segment, extract:
- segment_number: integer, as it appears (1-based).
- airline_code: 2-char airline code (sacrosanct — copy verbatim).
- airline_name: full airline name (e.g., PR -> Philippine Airlines, FJ -> Fiji
  Airways, QF -> Qantas).
- flight_number: integer.
- originating_airport_code: 3-letter code (sacrosanct — copy verbatim, NEVER
  change it).
- originating_airport_name: full airport name matching the code.
- originating_terminal: "none" if absent from the GDS line.
- destination_airport_code: 3-letter code (sacrosanct).
- destination_airport_name: full airport name matching the code.
- destination_terminal: "none" if absent.
- departure_date_time / arrival_date_time: objects with these exact keys:
  month (integer), month_name (string), date (integer), year (integer),
  day_of_week (string), time ("HH:MM").
- Resolve "+N" arrival day-offsets into the correct date, month, year AND
  day_of_week (handle month and year rollovers, e.g. Aug 31 -> Sep 1, or
  Dec 31 -> Jan 1 of the next year). Times are 24-hour "HH:MM".
- flight_duration: "HH:MM".
- aircraft_type: human-readable type (321 -> Airbus A321; 333 -> Airbus
  A330-300; 332 -> Airbus A330-200; 359 -> Airbus A350-900; 73H -> Boeing 737;
  7M8 -> Boeing 737 MAX 8). Do not state a more specific sub-model than the
  source style implies.
- service_class_letter: single letter.
- service_class: mapped class. Philippine Airlines: Business = C,D,I,J,Z;
  Premium = N,W; Economy = ALL OTHER codes (NOTE: B is Economy). Other
  airlines: report the letter with no mapping change.
- segment_record_locator: 6 characters following "DCPR" for Philippine
  Airlines; near the end of the segment data for Cebu Pacific. "none" for
  availability displays.
- For codeshare legs formatted like "FJ:QF3873", emit the QF marketing code
  and number (Qantas 3873).
- When translating airport codes, choose the airport that exactly matches the
  3-letter code; NEVER substitute a nearby or different airport.

Return ONLY the JSON object above. No preamble, no code fences, no commentary.
"""


def build_prompt(gds_text: str, default_year: str | int) -> list[dict]:
    """Assemble the OpenAI-style chat messages for a GDS extract request."""
    system = GDS_SYSTEM.replace(_DEFAULT_YEAR_SENTINEL, str(default_year))
    user = (
        "Parse the following GDS output. Output ONLY the required JSON object.\n\n"
        "GDS_DATA:\n"
        f"<<<GDS_DATA>>>\n{gds_text}\n<<<END_GDS_DATA>>>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --- Thinking-suppression degradation chain -------------------------------
def _base_params() -> dict:
    """Fixed sampling parameters (independent of degradation level)."""
    return {
        "temperature": MODEL_TEMP,
        "top_p": MODEL_TOP_P,
        "top_k": MODEL_TOP_K,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": MODEL_MAX_TOKENS,
    }


def build_params(level: int = 0) -> dict:
    """Full (level-0) sampling parameters with thinking suppression.

    Level 0 includes both ``reasoning_effort`` (if DISABLE_THINKING) and
    ``chat_template_kwargs: {enable_thinking: false}``. Higher levels are
    derived by :func:`_params_at_level` (dropping fields the server rejected).
    """
    params = _base_params()
    if not DISABLE_THINKING:
        return params
    params["reasoning_effort"] = 0
    if level == 0:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    return params


def _params_at_level(params: dict, level: int) -> dict:
    """Strip suppressed fields progressively as the degradation level rises."""
    out = dict(params)
    if level >= 1:
        out.pop("chat_template_kwargs", None)
    if level >= 2:
        out.pop("reasoning_effort", None)
    return out


# --- Client-side context guard --------------------------------------------
def estimate_tokens(text: str) -> int:
    """Conservative token heuristic (chars/3, ceiling). Dense GDS text runs
    low, so erring high avoids false rejections."""
    if not text:
        return 0
    return (len(str(text)) + 2) // 3


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def slot_budget() -> int:
    """Tokens available for a single request's prompt in one slot.

    llama.cpp divides its total ``--ctx-size`` across ``--parallel`` slots, so
    the per-slot budget is ``CONTEXT_SIZE // MODEL_PARALLEL``.
    """
    return max(1, CONTEXT_SIZE // MODEL_PARALLEL)


def usable_prompt_room() -> int:
    """Slot budget minus reserved output headroom minus safety margin."""
    return max(0, slot_budget() - MODEL_MAX_TOKENS - _SAFETY_MARGIN)


def check_context(messages: list[dict]) -> None:
    """Reject before any network call if the prompt cannot fit in a slot.

    Honors CONTEXT_GUARD: ``strict`` raises ``ContextGuardExceeded`` (422),
    ``warn`` logs a warning and allows, ``off`` skips entirely.
    """
    room = usable_prompt_room()
    est = _estimate_prompt_tokens(messages)
    if est <= room:
        return
    if CONTEXT_GUARD == "off":
        return
    if CONTEXT_GUARD == "warn":
        logger.warning(
            "Estimated prompt size %d exceeds slot budget %d; allowing request anyway.",
            est, room,
        )
        return
    raise ContextGuardExceeded(
            f"Estimated prompt size {est} tokens exceeds the gateway slot budget "
            f"{room} tokens (CONTEXT_SIZE={CONTEXT_SIZE}, MODEL_PARALLEL={MODEL_PARALLEL}, "
            f"max_tokens={MODEL_MAX_TOKENS}, safety margin {_SAFETY_MARGIN}). "
            f"Per-token slot budget = CONTEXT_SIZE // MODEL_PARALLEL = {slot_budget()} tokens. "
            f"To proceed on the DGX: re-provision the shared llama-server with a larger "
            f"--ctx-size (e.g. 131072) and lower --parallel (e.g. 2), then set "
            f"CONTEXT_SIZE={CONTEXT_SIZE} and MODEL_PARALLEL={MODEL_PARALLEL} in .env to match "
            f"and restart the gateway. Or submit a shorter GDS input."
        )


# ---------------------------------------------------------------------------
# JSON output extraction / sanitization
# ---------------------------------------------------------------------------
# Markers that indicate leaked reasoning/thinking content.
_THINKING_BLOCKS = [
    (r"<\|begin_thinking\|>.*?<\|end_thinking\|>", ""),
    (r"<think>.*?</think>", ""),
    (r"<reasoning>.*?</reasoning", ""),
]
_THINKING_MARKERS = re.compile(
    r"<\|begin_thinking\|>|<\|end_thinking\|>|<think>|</think>|<reasoning>|</reasoning>"
)


def _to_int(value: Any) -> int:
    """Coerce to int where possible; never raise (returns 0 otherwise)."""
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return 0


def _seg_str(value: Any, default: str = "none") -> str:
    """Return a non-empty string, else the default."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _normalize_date_time(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    return {
        "month": _to_int(value.get("month")),
        "month_name": _seg_str(value.get("month_name")),
        "date": _to_int(value.get("date")),
        "year": _to_int(value.get("year")),
        "day_of_week": _seg_str(value.get("day_of_week")),
        "time": _seg_str(value.get("time")),
    }


def _normalize_segment(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    return {
        "segment_number": _to_int(value.get("segment_number")),
        "segment_record_locator": _seg_str(value.get("segment_record_locator")),
        "airline_code": _seg_str(value.get("airline_code")),
        "airline_name": _seg_str(value.get("airline_name")),
        "flight_number": _to_int(value.get("flight_number")),
        "originating_airport_code": _seg_str(value.get("originating_airport_code")),
        "originating_airport_name": _seg_str(value.get("originating_airport_name")),
        "originating_terminal": _seg_str(value.get("originating_terminal")),
        "destination_airport_code": _seg_str(value.get("destination_airport_code")),
        "destination_airport_name": _seg_str(value.get("destination_airport_name")),
        "destination_terminal": _seg_str(value.get("destination_terminal")),
        "departure_date_time": _normalize_date_time(value.get("departure_date_time")),
        "arrival_date_time": _normalize_date_time(value.get("arrival_date_time")),
        "flight_duration": _seg_str(value.get("flight_duration")),
        "aircraft_type": _seg_str(value.get("aircraft_type")),
        "service_class_letter": _seg_str(value.get("service_class_letter")),
        "service_class": _seg_str(value.get("service_class")),
    }


def _normalize_gds(data: Any) -> dict:
    """Coerce parsed JSON into the exact schedule schema.

    - Drops any keys not in the schema (spelling-exact keys only).
    - Coerces numeric fields to int; missing optionals default to "none".
    - Guarantees ``segments`` is a list and ``Passenger Name`` is a list.
    - NEVER invents data: missing/unknown values become documented defaults.
    """
    if not isinstance(data, dict):
        raise ModelUnavailable("model output is not a JSON object")

    out: dict[str, Any] = {}

    rt = data.get("Record type")
    out["Record type"] = rt if rt in ("reservation", "none") else "none"

    pn = data.get("Passenger Name")
    if pn is None:
        pn = ["none"]
    if isinstance(pn, str):
        pn = [pn] if pn.strip() else ["none"]
    elif isinstance(pn, list):
        pn = [str(x) for x in pn]
        if not any(str(x).strip() for x in pn):
            pn = ["none"]
    else:
        pn = ["none"]
    out["Passenger Name"] = pn

    pnr = data.get("PNR")
    out["PNR"] = _seg_str(pnr)

    segs = data.get("segments")
    if not isinstance(segs, list):
        segs = []
    out["segments"] = [_normalize_segment(s) for s in segs]

    return out


def extract_json(raw: Optional[str]) -> dict:
    """Parse the model's raw text into the schedule schema.

    - Strips thinking/reasoning blocks (defense-in-depth against thinking mode).
    - Strips code fences (```` ```json ... ``` ````) and surrounding whitespace.
    - Locates the JSON span (first `{` to last `}`) and json.loads it.
    - Normalizes to the exact schema.
    - FAILS CLOSED (raises ModelUnavailable) if nothing parseable is found —
      a fabricated or malformed schedule is dangerous downstream travel data.
    """
    if raw is None:
        raise ModelUnavailable("model returned no output")

    text = str(raw)

    # Remove thinking/reasoning blocks (DOTALL so it spans newlines).
    for pattern, repl in _THINKING_BLOCKS:
        text = re.sub(pattern, repl, text, flags=re.DOTALL)
    text = _THINKING_MARKERS.sub("", text)

    # Strip JSON code fences: ```json ... ``` or ``` ... ```.
    text = re.sub(r"```(?:[A-Za-z]+)\s*", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return _normalize_gds(data)
        except json.JSONDecodeError:
            raise ModelUnavailable("model output contained a JSON span but failed to parse")
        except ModelUnavailable:
            raise

    raise ModelUnavailable("could not find a JSON object in model output")


# ===========================================================================
# EXTRACT ORCHESTRATION  (pure; the only part that touches the model)
# ===========================================================================
def resolve_default_year() -> int:
    """Year used when the GDS line omits one. Empty DEFAULT_YEAR => current
    year at request time; otherwise the pinned env value."""
    if DEFAULT_YEAR_ENV:
        try:
            return int(DEFAULT_YEAR_ENV)
        except ValueError:
            logger.warning("DEFAULT_YEAR=%r is not an integer; falling back to current year",
                           DEFAULT_YEAR_ENV)
    return date.today().year


def run_extract(request: dict, default_year: int,
                model_call: Callable[[list, dict], str]) -> dict:
    """Run a single extract request against ``model_call`` and return a
    normalized schedule object.

    ``model_call(messages, params) -> str`` is injected so this is pure/testable.
    """
    gds_text = request.get("gds_text")
    if gds_text is None:
        raise ValueError("Missing 'gds_text' field.")
    if not isinstance(gds_text, str):
        raise ValueError("'gds_text' must be a string.")
    if not gds_text.strip():
        raise ValueError("'gds_text' must be a non-empty string.")

    messages = build_prompt(gds_text, default_year)

    # Reject locally (before any network call) if the prompt cannot fit in a slot.
    check_context(messages)

    params = build_params(_PARAMS_LEVEL)
    raw = model_call(messages, params)
    schedule = extract_json(raw)
    logger.info(
        f"extract complete | record_type={schedule.get('Record type')} "
        f"segments={len(schedule.get('segments', []))} | out_len={len(schedule)}"
    )
    return schedule


def run_extract_batch(entries: list, default_year: int,
                      model_call: Callable[[list, dict], str]) -> list:
    """Run many extract entries SEQUENTIALLY, isolating per-entry failures.

    Each entry gets its own model call so one bad input cannot poison the
    others or overflow the shared slot budget. Overall HTTP status stays 200;
    each item reports ``status: "ok"`` or ``status: "error"``.
    """
    results: list[dict] = []
    for entry in entries:
        eid = entry.get("id") if isinstance(entry, dict) else None
        payload = {"gds_text": entry.get("gds_text")} if isinstance(entry, dict) else {"gds_text": None}
        try:
            schedule = run_extract(payload, default_year, model_call)
            results.append({"id": eid, "status": "ok", "schedule": schedule})
        except Exception as exc:  # noqa: BLE001 - record per-entry error, never crash the batch
            logger.error("extract entry %s failed: %s", eid, exc)
            results.append({"id": eid, "status": "error", "error": str(exc)})
    return results


# ===========================================================================
# MODEL BACKEND  (HTTP to the local llama.cpp server — used on the DGX)
# ===========================================================================
def _resolve_content(message: dict) -> str:
    """Return the model's text from the first non-empty field.

    Qwen3.8-27B is an adaptive thinking model: for hard inputs it may leave
    ``content`` empty and put the answer in ``reasoning_content`` (or a legacy
    name). We fall back across those fields, logging a warning when a reasoning
    field actually carried the text (thinking leaked despite suppression).
    """
    content = message.get("content") or ""
    if content:
        return content
    for field in ("reasoning_content", "thinking", "reasoning", "reason"):
        value = message.get(field)
        if value:
            logger.warning(f"Model answered in '{field}' (thinking leaked); extracting.")
            return str(value)
    raise ModelUnavailable("Model returned an empty response.")


def _post(body: dict, headers: dict) -> "requests.Response":
    try:
        return requests.post(MODEL_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        # Connection refused / timeout / DNS etc. → model is unavailable (503).
        raise ModelUnavailable(f"Model server unreachable: {exc}") from exc


def http_model_call(messages: list[dict], params: dict = None) -> str:
    """POST a chat completion to the llama.cpp OpenAI-compatible endpoint.

    Implements the thinking-suppression degradation chain: try full params
    (level 0), then drop chat_template_kwargs (level 1), then drop
    reasoning_effort too (level 2). The first working level is cached so later
    calls skip straight to it. Context-length rejections map to
    ``ContextExceeded`` (503); the resolved text is returned as-is for
    ``extract_json()`` to clean.
    """
    global _PARAMS_LEVEL

    if params is None:
        params = {}
    headers = {"Authorization": f"Bearer {LLAMA_SERVER_API_KEY}"} if LLAMA_SERVER_API_KEY else {}

    level = _PARAMS_LEVEL
    while level <= 2:
        attempt = _params_at_level(params, level)
        body = {"model": MODEL_NAME, "messages": messages}
        for key in _SAMPLING_KEYS:
            if key in attempt:
                body[key] = attempt[key]
        if "reasoning_effort" in attempt:
            body["reasoning_effort"] = attempt["reasoning_effort"]
        if "chat_template_kwargs" in attempt:
            body["chat_template_kwargs"] = attempt["chat_template_kwargs"]

        resp = _post(body, headers)
        if resp.status_code == 200:
            _PARAMS_LEVEL = level
            return _resolve_content(resp.json()["choices"][0]["message"])

        detail = resp.text[:300]
        lowered = detail.lower()

        # Context-length overflow → actionable 503.
        if "context length" in lowered or ("context" in lowered and any(
            w in lowered for w in ("exceed", "size", "length", "overflow", "too long"))):
            raise ContextExceeded(
                "Request exceeds the model's context window (server ctx is fixed at launch). "
                "Restart llama-server with a larger --ctx-size and set CONTEXT_SIZE in .env to match."
            )

        # Thinking-suppression degradation: the build rejected a suppression field.
        if "chat_template_kwargs" in lowered or "reasoning_effort" in lowered:
            level += 1
            continue

        # Any other non-200 → give up cleanly.
        raise ModelUnavailable(f"Model server responded {resp.status_code}: {detail}")

    raise ModelUnavailable(
        "Model server could not satisfy the request after param degradation."
    )


# ===========================================================================
# FastAPI Application
# ===========================================================================
app = FastAPI(
    title="AI GDS Extraction API",
    description=(
        "Lean JSON-in / JSON-out GDS Extractor ('make schedule') for a travel "
        "agency. Runs Qwen3.8-27B locally on the shared NVIDIA DGX Spark model server."
    ),
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lean API-key authentication -------------------------------------------
def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias=API_KEY_AUTH_HEADER, description="API key for access control"),
):
    # Optional header so a *missing* key also yields 401 (not 422).
    if not x_api_key or x_api_key not in API_KEY_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return x_api_key


# --- Request models ----------------------------------------------------------
class ExtractRequest(BaseModel):
    gds_text: str = Field(..., min_length=1, description="One or more lines of GDS output to parse.")


class BatchEntry(BaseModel):
    id: Optional[str] = Field(default=None, description="Client-supplied id echoed back in the result item.")
    gds_text: str = Field(..., min_length=1, description="One or more lines of GDS output to parse.")


class BatchRequest(BaseModel):
    entries: list[BatchEntry] = Field(
        ..., min_length=1, description="One or more GDS entries to extract (processed sequentially)."
    )


# --- Endpoints --------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "service": "AI GDS Extraction",
        "version": VERSION,
        "endpoints": ["/v1/extract", "/v1/extract_batch", "/v1/version", "/healthz"],
        "docs": "/docs",
    }


@app.get("/healthz", tags=["System"])
async def health_check():
    """Liveness probe — verifies model-server connectivity and reports the
    computed per-slot context budget, cross-checked against the server's
    per-slot n_ctx via /props when available."""
    model_status = "unreachable"
    server_ctx_size = None
    try:
        health_url = MODEL_URL.replace("/v1/chat/completions", "/health")
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            model_status = "ready"
            try:
                payload = r.json()
                server_ctx_size = payload.get("ctx_size") or payload.get("n_ctx")
            except Exception:  # noqa: BLE001 - health payload is best-effort
                server_ctx_size = None
        else:
            model_status = f"responding (status {r.status_code})"
    except Exception as exc:  # noqa: BLE001 - report any connection issue
        model_status = f"unreachable ({exc})"

    # Cross-check the gateway's assumed slot budget against the server's.
    slot = slot_budget()
    ctx_check = "unknown"
    try:
        props_url = MODEL_URL.replace("/v1/chat/completions", "/props")
        rp = requests.get(props_url, timeout=5)
        if rp.status_code == 200:
            server_slot = (
                rp.json().get("default_generation_settings", {}).get("n_ctx")
            )
            if server_slot is not None:
                ctx_check = "match" if server_slot == slot else "mismatch"
    except Exception:  # noqa: BLE001 - /props is best-effort (depends on build)
        pass

    return {
        "status": "healthy" if model_status == "ready" else "degraded",
        "model_server": model_status,
        "model_name": MODEL_NAME,
        "context_budget": {
            "slot_tokens": slot,
            "server_ctx_size": server_ctx_size,
            "check": ctx_check,
        },
        "default_year_mode": "current-year" if not DEFAULT_YEAR_ENV else f"pinned={DEFAULT_YEAR_ENV}",
        "version": VERSION,
    }


@app.post("/v1/extract", response_class=JSONResponse, tags=["Extract"])
async def extract(
    _api_key: str = Depends(verify_api_key),
    body: ExtractRequest = Body(...),
):
    """Parse a single GDS input into the schedule JSON."""
    default_year = resolve_default_year()
    try:
        schedule = run_extract(body.model_dump(), default_year, http_model_call)
    except ValueError as exc:
        # Includes ContextGuardExceeded (422 with actionable sizing guidance).
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except (ModelUnavailable, ContextExceeded) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - avoid leaking internals
        logger.error(f"Unexpected extract error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Processing failed.")

    return JSONResponse(content=schedule)


@app.post("/v1/extract_batch", response_class=JSONResponse, tags=["Extract"])
async def extract_batch(
    _api_key: str = Depends(verify_api_key),
    body: BatchRequest = Body(...),
):
    """Parse many GDS inputs sequentially. Per-entry failures are isolated and
    reported per item; overall status stays 200."""
    default_year = resolve_default_year()
    try:
        results = run_extract_batch(
            [entry.model_dump() for entry in body.entries],
            default_year,
            http_model_call,
        )
    except Exception as exc:  # noqa: BLE001 - avoid leaking internals
        logger.error(f"Unexpected batch error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Processing failed.")

    return JSONResponse(content={"results": results})


@app.post("/v1/version", response_class=JSONResponse, tags=["System"])
async def version():
    """Service version — never touches the model."""
    return JSONResponse(content={"version": VERSION})


# ===========================================================================
# Standalone Runner
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting AI GDS Extraction API on {API_HOST}:{API_PORT}")
    logger.info(f"Model server: {MODEL_URL} ({MODEL_NAME})")
    logger.info(f"DISABLE_THINKING={DISABLE_THINKING} | temp={MODEL_TEMP} | top_p={MODEL_TOP_P}")
    logger.info(
        f"CONTEXT_SIZE={CONTEXT_SIZE} MODEL_PARALLEL={MODEL_PARALLEL} | slot budget="
        f"{slot_budget()} tokens | prompt room={usable_prompt_room()} | guard={CONTEXT_GUARD}"
    )
    logger.info(f"Default-year mode: {'current-year' if not DEFAULT_YEAR_ENV else 'pinned='+DEFAULT_YEAR_ENV}")
    logger.info(f"Swagger UI: http://{API_HOST}:{API_PORT}/docs")

    uvicorn.run("gds_extraction_service:app", host=API_HOST, port=API_PORT, log_level="info")
