"""
AI GDS Extraction — FastAPI Service (v1.1 — vLLM Performance Fix)
======================================================================
Lean, JSON-in / JSON-out **GDS Extractor** ("make schedule") for a travel
agency. Parses raw GDS output lines (Amadeus availability displays and complete
reservations) into the structured flight-segment JSON defined in the project
spec, and runs fully offline on the NVIDIA DGX Spark via the vLLM server
(E:\\DGXSpark_Setup\\vllm-qwen, :8011, Qwen3.6-35B-A3B-NVFP4, NVFP4/FP8).
This service is a **gateway only** — it never starts or stops the model server
(see start.sh / stop.sh).

Architecture (vLLM-native, async):

    Client  --POST /v1/extract-->  FastAPI gateway (:8084, async httpx)
            {gds_text}
            {entries: [{id, gds_text}]}
                                           |
                                           v
                                      vLLM :8011
                                      Qwen3.6-35B-A3B-NVFP4 (MoE 35B/3B active)
                                      MAX_MODEL_LEN 32768 · prefix caching

The core logic (build_prompt / build_params / estimate_tokens / check_context /
extract_json / _normalize_gds / run_extract / run_extract_batch) are pure
functions taking an injectable ``model_call`` callable, so the entire business
logic is unit-tested on the Windows dev machine with a stub backend — no GPU
needed.  DECODING IS UNCHANGED (greedy temp=0, top_p 0.5, top_k 40) for
deterministic extraction.

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
# vLLM-native (2026-08-27): CONTEXT_SIZE now means MAX_MODEL_LEN (32768) on the
# vLLM server (E:\DGXSpark_Setup\vllm-qwen, :8011). No slot division — vLLM uses
# continuous batching. Legacy :8006 path kept for fallback when MODEL_URL
# contains 8006.  DECODING PARAMS ARE FROZEN (greedy) — do not change.
# ---------------------------------------------------------------------------
load_dotenv()

CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "32768"))
MODEL_PARALLEL = int(os.getenv("MODEL_PARALLEL", "1"))
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:8011/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.6-35B-A3B-NVFP4")
# Internal bearer token the gateway presents to the vLLM server. Sent ONLY if
# non-empty. vLLM default is empty (no auth) unless API_KEY is set in
# E:\DGXSpark_Setup\vllm-qwen\.env. Legacy 8006 used sk-internal-proofreader.
LLAMA_SERVER_API_KEY = os.getenv("LLAMA_SERVER_API_KEY", "")
# Greedy decoding (temp=0) for run-to-run determinism on factual GDS parsing;
# validated on-DGX with byte-identical output across repeated runs.
# *** DO NOT CHANGE — established decoding contract ***
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.0"))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", "0.5"))
MODEL_TOP_K = int(os.getenv("MODEL_TOP_K", "40"))
MODEL_MAX_TOKENS = max(64, int(os.getenv("MODEL_MAX_TOKENS", "3072")))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "1").lower() == "1"
# Optional guided decoding (xgrammar on vLLM). 0=off, 1=sends guided_json schema.
ENABLE_GUIDED_JSON = os.getenv("ENABLE_GUIDED_JSON", "0").lower() in ("1", "true", "yes")
# Batch concurrency (Phase D4). 1=sequential, >1 = asyncio.gather with semaphore.
BATCH_CONCURRENCY = max(1, int(os.getenv("BATCH_CONCURRENCY", "1")))
# Whether to use streaming for TTFT instrumentation (keeps non-streaming default for stability).
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "0").lower() in ("1", "true", "yes")
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
VERSION = "1.1"

# Sentinel substituted into the system prompt per request (the JSON braces in the
# instructions must NOT go through str.format / an f-string).
_DEFAULT_YEAR_SENTINEL = "__DEFAULT_YEAR__"

# Tokens reserved beyond the output budget as headroom (safety margin).
_SAFETY_MARGIN = 256
# vLLM-native degradation: only chat_template_kwargs matters; reasoning_effort
# is llama-only and never sent on vLLM path. Level 0 = with enable_thinking,
# level 1 = without. Legacy 8006 path still supports 3 levels for compat.
_PARAMS_LEVELS = (0, 1, 2)
# Last known-working degradation level, cached to avoid retrying on every call.
_PARAMS_LEVEL = 0

# Sampling parameters forwarded to the model request. DECODING FROZEN.
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "max_tokens",
)

# Guided JSON schema for §1.5 (used only when ENABLE_GUIDED_JSON=1).
_GDS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "Record type": {"type": "string", "enum": ["reservation", "none"]},
        "Passenger Name": {"type": "array", "items": {"type": "string"}},
        "PNR": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_number": {"type": "integer"},
                    "segment_record_locator": {"type": "string"},
                    "airline_code": {"type": "string"},
                    "airline_name": {"type": "string"},
                    "flight_number": {"type": "integer"},
                    "originating_airport_code": {"type": "string"},
                    "originating_airport_name": {"type": "string"},
                    "originating_terminal": {"type": "string"},
                    "destination_airport_code": {"type": "string"},
                    "destination_airport_name": {"type": "string"},
                    "destination_terminal": {"type": "string"},
                    "departure_date_time": {
                        "type": "object",
                        "properties": {
                            "month": {"type": "integer"}, "month_name": {"type": "string"},
                            "date": {"type": "integer"}, "year": {"type": "integer"},
                            "day_of_week": {"type": "string"}, "time": {"type": "string"},
                        },
                        "required": ["month", "month_name", "date", "year", "day_of_week", "time"],
                    },
                    "arrival_date_time": {
                        "type": "object",
                        "properties": {
                            "month": {"type": "integer"}, "month_name": {"type": "string"},
                            "date": {"type": "integer"}, "year": {"type": "integer"},
                            "day_of_week": {"type": "string"}, "time": {"type": "string"},
                        },
                        "required": ["month", "month_name", "date", "year", "day_of_week", "time"],
                    },
                    "flight_duration": {"type": "string"},
                    "aircraft_type": {"type": "string"},
                    "service_class_letter": {"type": "string"},
                    "service_class": {"type": "string"},
                },
                "required": ["segment_number", "airline_code", "flight_number"],
            },
        },
    },
    "required": ["Record type", "Passenger Name", "PNR", "segments"],
}

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
# v1.1 compression: ~30% shorter (<750 toks) but verbatim-faithful — all 11
# rule blocks kept, decoding unchanged.
GDS_SYSTEM = """You are a meticulous Global Distribution System (GDS) flight-data extractor.
Parse the GDS output and return ONLY a JSON object. No commentary, no explanations, no markdown, no <think> blocks.

The DEFAULT YEAR is __DEFAULT_YEAR__. Use it when the GDS line omits a year.

OUTPUT ONLY THIS JSON OBJECT (exact keys, no extras):
{{
  "Record type": "reservation" | "none",
  "Passenger Name": [ "LASTNAME/FIRSTNAME(s)", ... ] OR [ "none" ],
  "PNR": "<6-char PNR>" | "none",
  "segments": [ {{ ...segment... }} ]
}}

Rules:
- Record type: "reservation" if record locators AND passenger names present; else "none".
- PNR: FIRST PNR in the whole record (NOT per-segment locator); "none" for availability displays.
- Passenger names (reservations): start AFTER the numerical passenger number; capture until first '/' (last vs first name boundary); preserve ORIGINAL "LASTNAME/FIRSTNAME(s)" exactly including prefix and trailing chars after locator digits; extract EVERY passenger (up to 9); if none → ["none"]; if last name begins with "APDI", keep ENTIRE last name including "APDI".
- Availability display (no PNR/names): "Record type" "none", "Passenger Name" ["none"], "PNR" "none"; per segment "segment_record_locator" "none", "service_class_letter" "none", "service_class" "none" (availability counts like "J5 C5 D5" are NOT class of service).
- Per segment:
  - segment_number: integer (1-based as shown)
  - airline_code: 2-char code (sacrosanct — copy verbatim)
  - airline_name: full name (PR -> Philippine Airlines, FJ -> Fiji Airways, QF -> Qantas)
  - flight_number: integer
  - originating_airport_code: 3-letter (sacrosanct — copy verbatim, NEVER change it)
  - originating_airport_name: full airport name matching the code
  - originating_terminal: "none" if absent
  - destination_airport_code: 3-letter (sacrosanct)
  - destination_airport_name: full airport name matching the code
  - destination_terminal: "none" if absent
  - departure_date_time / arrival_date_time: {{month (int), month_name (string), date (int), year (int), day_of_week (string), time ("HH:MM")}}. Resolve "+N" arrival offsets into correct date/month/year AND day_of_week (handle Aug 31->Sep 1 and Dec 31->Jan 1 rollovers). Times 24-hour "HH:MM".
  - flight_duration: "HH:MM"
  - aircraft_type: human-readable (321 -> Airbus A321; 333 -> Airbus A330-300; 332 -> Airbus A330-200; 359 -> Airbus A350-900; 73H -> Boeing 737; 7M8 -> Boeing 737 MAX 8); do not add sub-model detail beyond source
  - service_class_letter: single letter
  - service_class: Philippine Airlines Business = C,D,I,J,Z; Premium = N,W; Economy = ALL OTHER codes (NOTE: B is Economy). Other airlines: report letter unchanged
  - segment_record_locator: 6 chars after "DCPR" (Philippine Airlines); near end for Cebu Pacific; "none" for availability
  - Codeshare "FJ:QF3873" → emit QF marketing code and number (Qantas 3873)
  - Airport translation: choose airport exactly matching 3-letter code; NEVER substitute nearby airport.

Return ONLY the JSON. No preamble, no fences, no reasoning.
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


# --- Adaptive max_tokens resolver (Phase C3) -------------------------------
# Heuristic segment count: counts airline flight tokens or line breaks.
_SEG_RE = re.compile(r"\b(?:PR|FJ|QF|AA|BA|VS|CX|SQ|EK|JL|NH|KE|CI|BR|MH|TG|VN|GA|5J|PR)\s*\d{1,4}\b")

def estimate_segments(gds_text: str) -> int:
    if not gds_text:
        return 1
    hits = len(_SEG_RE.findall(gds_text))
    # Fallback: newline-based estimate (availability displays have ~2 lines/segment)
    if hits == 0:
        hits = max(1, len([l for l in gds_text.splitlines() if l.strip()]) // 2)
    return max(1, min(20, hits))

def resolve_max_tokens(gds_text: str, base: int | None = None) -> int:
    """Adaptive cap: 1200 + 280*segs + 0.6*input_toks, clamped 1024-4096.

    Toby 2-seg → ~1800, golden 10-seg → ~4000. Prevents 8192 runaway.
    `base` is kept for compat but ignored when gds_text is provided — adaptive wins.
    When gds_text is empty and base is given, base is returned.
    """
    if gds_text is None or gds_text == "":
        if base is not None:
            return max(64, int(base))
        gds_text = ""
    segs = estimate_segments(gds_text)
    inp = estimate_tokens(gds_text)
    cap = int(1200 + 280 * segs + 0.6 * inp)
    adaptive = max(1024, min(4096, cap))
    # If base is provided, use adaptive (more accurate) — base is just fallback for empty
    return adaptive


# --- Thinking-suppression / sampling chain (vLLM-native, decoding FROZEN) --
def _base_params() -> dict:
    """Fixed sampling parameters (independent of degradation level). Decoding frozen."""
    return {
        "temperature": MODEL_TEMP,
        "top_p": MODEL_TOP_P,
        "top_k": MODEL_TOP_K,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": MODEL_MAX_TOKENS,
    }


def _is_legacy_llama_url() -> bool:
    return ":8006" in MODEL_URL

def build_params(level: int = 0, gds_text: str | None = None) -> dict:
    """Sampling parameters with vLLM-native thinking suppression.

    vLLM path (default, :8011): only ``chat_template_kwargs: {enable_thinking:false}`` when DISABLE_THINKING=1.
    Never sends ``reasoning_effort`` (not a vLLM param — would waste a retry).
    Legacy :8006 path: retains 3-level reasoning_effort + chat_template_kwargs chain for compat.
    Adaptive max_tokens is applied when gds_text is provided.
    """
    params = _base_params()
    if gds_text is not None:
        params["max_tokens"] = resolve_max_tokens(gds_text, params["max_tokens"])
    if not DISABLE_THINKING:
        return params
    # Legacy fallback keeps reasoning_effort for llama.cpp
    if _is_legacy_llama_url():
        params["reasoning_effort"] = 0
        if level == 0:
            params["chat_template_kwargs"] = {"enable_thinking": False}
        return params
    # vLLM path: only chat_template_kwargs, single-level degradation
    if level == 0:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    return params


def _params_at_level(params: dict, level: int) -> dict:
    """Strip suppressed fields progressively as degradation level rises."""
    out = dict(params)
    if _is_legacy_llama_url():
        if level >= 1:
            out.pop("chat_template_kwargs", None)
        if level >= 2:
            out.pop("reasoning_effort", None)
    else:
        # vLLM: only level 0 has chat_template_kwargs
        if level >= 1:
            out.pop("chat_template_kwargs", None)
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
    """Legacy slot budget (llama.cpp :8006). For vLLM, use context_budget()."""
    return max(1, CONTEXT_SIZE // MODEL_PARALLEL)

def context_budget() -> int:
    """vLLM-native budget: MAX_MODEL_LEN (CONTEXT_SIZE) — no slot division."""
    return max(1, CONTEXT_SIZE)

def usable_prompt_room() -> int:
    """Usable room — vLLM path uses MAX_MODEL_LEN, legacy uses slot_budget."""
    if _is_legacy_llama_url():
        return max(0, slot_budget() - MODEL_MAX_TOKENS - _SAFETY_MARGIN)
    return max(0, context_budget() - MODEL_MAX_TOKENS - _SAFETY_MARGIN)

def _usable_for_max_tokens(max_tokens: int) -> int:
    if _is_legacy_llama_url():
        return max(0, slot_budget() - max_tokens - _SAFETY_MARGIN)
    return max(0, context_budget() - max_tokens - _SAFETY_MARGIN)


def check_context(messages: list[dict], max_tokens: int | None = None) -> None:
    """Reject before any network call if prompt cannot fit.

    Honors CONTEXT_GUARD: strict → 422, warn → log, off → skip.
    Uses vLLM MAX_MODEL_LEN math by default, legacy slot math only for :8006.
    """
    mt = max_tokens if max_tokens is not None else MODEL_MAX_TOKENS
    room = _usable_for_max_tokens(mt)
    est = _estimate_prompt_tokens(messages)
    if est <= room:
        return
    if CONTEXT_GUARD == "off":
        return
    if CONTEXT_GUARD == "warn":
        logger.warning(
            "Estimated prompt size %d exceeds budget %d; allowing anyway.",
            est, room,
        )
        return
    if _is_legacy_llama_url():
        raise ContextGuardExceeded(
            f"Estimated prompt size {est} tokens exceeds the gateway slot budget "
            f"{room} tokens (CONTEXT_SIZE={CONTEXT_SIZE}, MODEL_PARALLEL={MODEL_PARALLEL}, "
            f"max_tokens={mt}, safety margin {_SAFETY_MARGIN}). "
            f"Per-slot budget = CONTEXT_SIZE // MODEL_PARALLEL = {slot_budget()} tokens. "
            f"To proceed on the DGX: re-provision the llama-server with larger --ctx-size and set "
            f"CONTEXT_SIZE/MODEL_PARALLEL in .env to match and restart gateway. Or submit shorter GDS."
        )
    raise ContextGuardExceeded(
        f"Estimated prompt size {est} tokens exceeds vLLM context budget "
        f"{room} tokens (MAX_MODEL_LEN={CONTEXT_SIZE}, max_tokens={mt}, safety {_SAFETY_MARGIN}). "
        f"Context = CONTEXT_SIZE={CONTEXT_SIZE} (vLLM MAX_MODEL_LEN), prompt_room = MAX_MODEL_LEN - max_tokens - {_SAFETY_MARGIN}. "
        f"To proceed: raise MAX_MODEL_LEN via E:\\DGXSpark_Setup\\vllm-qwen startserver.sh (--max-model-len) and set CONTEXT_SIZE to match, "
        f"or lower MODEL_MAX_TOKENS / submit shorter GDS. Adaptive cap for this request was {mt}."
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
    Uses adaptive max_tokens and per-request metrics logging.
    """
    import time as _time
    gds_text = request.get("gds_text")
    if gds_text is None:
        raise ValueError("Missing 'gds_text' field.")
    if not isinstance(gds_text, str):
        raise ValueError("'gds_text' must be a string.")
    if not gds_text.strip():
        raise ValueError("'gds_text' must be a non-empty string.")

    messages = build_prompt(gds_text, default_year)
    # Adaptive cap per request (preserves global MODEL_MAX_TOKENS as baseline)
    adaptive = resolve_max_tokens(gds_text, MODEL_MAX_TOKENS)
    # Guard with adaptive cap
    check_context(messages, max_tokens=adaptive)

    params = build_params(_PARAMS_LEVEL, gds_text=gds_text)
    # Ensure max_tokens in params reflects adaptive (build_params already did, but enforce)
    params["max_tokens"] = adaptive
    t0 = _time.perf_counter()
    raw = model_call(messages, params)
    elapsed = _time.perf_counter() - t0
    # Instrumentation: estimate usage when model_call is stub (no usage dict)
    est_prompt = _estimate_prompt_tokens(messages)
    # Try to get real token counts if raw came from http path with usage attached (not stub)
    # For pure pipeline we just log estimates
    schedule = extract_json(raw)
    segs = len(schedule.get('segments', []))
    # Detailed per-request metrics line (consumed by Grafana / log tail)
    logger.info(
        f"extract complete | record_type={schedule.get('Record type')} segments={segs} "
        f"gds_len={len(gds_text)} est_prompt_toks={est_prompt} est_segs={estimate_segments(gds_text)} "
        f"max_tokens={adaptive} elapsed={elapsed:.2f}s raw_len={len(raw) if raw else 0}"
    )
    # Warn if thinking leaked (raw still contains markers after extract_json stripping)
    if raw and _THINKING_MARKERS.search(str(raw)):
        logger.warning(f"thinking markers found in raw output (leaked) raw_len={len(raw)}")
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
# MODEL BACKEND  (HTTP to vLLM :8011 — async httpx + streaming instrumentation)
# decoding frozen — only thinking flag & max_tokens are dynamic
# ===========================================================================
def _resolve_content(message: dict) -> str:
    """Return model text from first non-empty field, handling reasoning split.

    Qwen3.6-35B-A3B is a reasoning MoE: with thinking ON it puts trace in
    ``reasoning_content`` and final answer in ``content``. With
    ``chat_template_kwargs:{enable_thinking:false}`` the trace should be empty.
    We warn when reasoning leaked despite suppression (counts toward gen time).
    """
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if reasoning and not content:
        # Thinking leaked and content empty — model answered only in reasoning
        logger.warning(f"Model answered in 'reasoning_content' only (thinking leaked, len={len(reasoning)}); extracting.")
        return str(reasoning)
    if reasoning and content:
        # Normal reasoning model: log leak length but return content (final answer)
        # Only warn if reasoning is substantial (>50 chars) to avoid noise
        if len(str(reasoning)) > 50:
            logger.warning(f"thinking trace leaked despite suppression (reasoning_len={len(str(reasoning))}, content_len={len(content)})")
    if content:
        return content
    for field in ("thinking", "reason", "reasoning"):
        value = message.get(field)
        if value:
            logger.warning(f"Model answered in '{field}' (thinking leaked); extracting.")
            return str(value)
    raise ModelUnavailable("Model returned an empty response.")


def _post(body: dict, headers: dict) -> "requests.Response":
    """Sync fallback for tests / healthz that still use requests."""
    try:
        return requests.post(MODEL_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise ModelUnavailable(f"Model server unreachable: {exc}") from exc

async def _post_async(body: dict, headers: dict) -> dict:
    """Async httpx POST — returns parsed JSON dict. Used by http_model_call_async."""
    try:
        import httpx
        import time as _time
        t0 = _time.perf_counter()
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(MODEL_URL, json=body, headers=headers)
            elapsed = _time.perf_counter() - t0
            if r.status_code == 200:
                data = r.json()
                # Instrumentation: log TTFT-ish via elapsed vs usage if available
                usage = data.get("usage", {})
                if usage:
                    logger.info(f"vLLM usage | prompt_toks={usage.get('prompt_tokens')} completion_toks={usage.get('completion_tokens')} elapsed={elapsed:.2f}s model={MODEL_NAME}")
                    # Detect truncation (hit max_tokens cap) — signals adaptive cap too low
                    if usage.get("completion_tokens") and body.get("max_tokens") and usage.get("completion_tokens") >= body.get("max_tokens"):
                        logger.warning(f"generation hit max_tokens cap ({body.get('max_tokens')}); possible truncation — consider raising cap")
                return data
            # Non-200: surface text for error mapping
            detail = r.text[:500]
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")
    except Exception as exc:
        if isinstance(exc, ModelUnavailable):
            raise
        if "Model server unreachable" in str(exc):
            raise
        # httpx timeout / connect errors → 503
        if "timeout" in str(exc).lower() or "connect" in str(exc).lower():
            raise ModelUnavailable(f"Model server unreachable (async): {exc}") from exc
        raise

def http_model_call(messages: list[dict], params: dict = None) -> str:
    """Sync entry point used by run_extract and tests. Delegates to async path via asyncio.

    Implements vLLM-native degradation: only chat_template_kwargs matters.
    Legacy :8006 still goes through 3-level chain. Keeps sync for test injection.
    On DGX the FastAPI routes call http_model_call_async directly (true async).
    """
    global _PARAMS_LEVEL
    if params is None:
        params = {}
    headers = {"Authorization": f"Bearer {LLAMA_SERVER_API_KEY}"} if LLAMA_SERVER_API_KEY else {}

    level = _PARAMS_LEVEL
    max_level = 2 if _is_legacy_llama_url() else 1
    while level <= max_level:
        attempt = _params_at_level(params, level)
        body: dict[str, Any] = {"model": MODEL_NAME, "messages": messages}
        for key in _SAMPLING_KEYS:
            if key in attempt:
                body[key] = attempt[key]
        if "reasoning_effort" in attempt:
            body["reasoning_effort"] = attempt["reasoning_effort"]
        if "chat_template_kwargs" in attempt:
            body["chat_template_kwargs"] = attempt["chat_template_kwargs"]
        # Guided JSON (xgrammar) when enabled — vLLM honors extra_body.guided_json
        if ENABLE_GUIDED_JSON and not _is_legacy_llama_url():
            body["extra_body"] = {"guided_json": _GDS_JSON_SCHEMA}
            # Also try response_format for compat
            body["response_format"] = {"type": "json_object"}

        try:
            # Prefer async path if we're already in an event loop; otherwise sync
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                # Inside async context — run blocking httpx via to_thread to avoid nesting
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    data = pool.submit(lambda: _post(body, headers)).result()
                    # _post returns requests.Response; normalize to dict
                    if isinstance(data, dict):
                        j = data
                    else:
                        # requests.Response
                        if data.status_code != 200:
                            detail = data.text[:500]
                            lowered = detail.lower()
                            if "context length" in lowered or ("context" in lowered and any(w in lowered for w in ("exceed","size","length","overflow","too long"))):
                                raise ContextExceeded("Request exceeds model's context window (MAX_MODEL_LEN).")
                            if "chat_template_kwargs" in lowered or "reasoning_effort" in lowered:
                                level += 1
                                continue
                            raise ModelUnavailable(f"Model server responded {data.status_code}: {detail}")
                        j = data.json()
            else:
                resp = _post(body, headers)
                if resp.status_code != 200:
                    detail = resp.text[:500]
                    lowered = detail.lower()
                    if "context length" in lowered or ("context" in lowered and any(w in lowered for w in ("exceed","size","length","overflow","too long"))):
                        raise ContextExceeded("Request exceeds model's context window (MAX_MODEL_LEN).")
                    if "chat_template_kwargs" in lowered or "reasoning_effort" in lowered:
                        level += 1
                        continue
                    raise ModelUnavailable(f"Model server responded {resp.status_code}: {detail}")
                j = resp.json()
            # Success path
            _PARAMS_LEVEL = level
            msg = j["choices"][0]["message"]
            # Log usage if present (vLLM returns usage when stream_options include_usage)
            usage = j.get("usage")
            if usage:
                logger.info(f"vLLM usage sync | prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
                if usage.get("completion_tokens") and body.get("max_tokens") and usage.get("completion_tokens") >= body.get("max_tokens"):
                    logger.warning(f"hit max_tokens cap {body.get('max_tokens')}")
            return _resolve_content(msg)
        except (ContextExceeded, ModelUnavailable):
            raise
        except RuntimeError as exc:
            detail = str(exc)
            lowered = detail.lower()
            if "context length" in lowered or ("context" in lowered and any(w in lowered for w in ("exceed","size","length","overflow","too long"))):
                raise ContextExceeded("Request exceeds model's context window.")
            if "chat_template_kwargs" in lowered or "reasoning_effort" in lowered:
                level += 1
                continue
            raise ModelUnavailable(f"Model server responded error: {detail[:500]}") from exc

    raise ModelUnavailable("Model server could not satisfy request after param degradation.")

async def http_model_call_async(messages: list[dict], params: dict = None) -> str:
    """True async path for FastAPI routes — uses httpx.AsyncClient and logs usage/TTFT."""
    global _PARAMS_LEVEL
    if params is None:
        params = {}
    headers = {"Authorization": f"Bearer {LLAMA_SERVER_API_KEY}"} if LLAMA_SERVER_API_KEY else {}
    level = _PARAMS_LEVEL
    max_level = 2 if _is_legacy_llama_url() else 1
    while level <= max_level:
        attempt = _params_at_level(params, level)
        body: dict[str, Any] = {"model": MODEL_NAME, "messages": messages}
        for key in _SAMPLING_KEYS:
            if key in attempt:
                body[key] = attempt[key]
        if "reasoning_effort" in attempt:
            body["reasoning_effort"] = attempt["reasoning_effort"]
        if "chat_template_kwargs" in attempt:
            body["chat_template_kwargs"] = attempt["chat_template_kwargs"]
        if ENABLE_GUIDED_JSON and not _is_legacy_llama_url():
            body["extra_body"] = {"guided_json": _GDS_JSON_SCHEMA}
            body["response_format"] = {"type": "json_object"}
        try:
            data = await _post_async(body, headers)
            _PARAMS_LEVEL = level
            msg = data["choices"][0]["message"]
            return _resolve_content(msg)
        except RuntimeError as exc:
            detail = str(exc)
            lowered = detail.lower()
            if "chat_template_kwargs" in lowered or "reasoning_effort" in lowered:
                level += 1
                continue
            if "context length" in lowered or ("context" in lowered and any(w in lowered for w in ("exceed","size","length","overflow","too long"))):
                raise ContextExceeded("Request exceeds model's context window.")
            raise ModelUnavailable(detail) from exc
    raise ModelUnavailable("Model server could not satisfy request after param degradation.")


# ===========================================================================
# FastAPI Application (vLLM :8011)
# ===========================================================================
app = FastAPI(
    title="AI GDS Extraction API",
    description=(
        "Lean JSON-in / JSON-out GDS Extractor ('make schedule') for a travel "
        "agency. Runs Qwen3.6-35B-A3B-NVFP4 via vLLM on DGX Spark (:8011). "
        "Greedy decoding (temp=0) for deterministic extraction."
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
    """Liveness probe — vLLM-native: checks /health + /v1/models + /metrics hint."""
    model_status = "unreachable"
    server_ctx_size = None
    vllm_model_id = None
    try:
        health_url = MODEL_URL.replace("/v1/chat/completions", "/health")
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            model_status = "ready"
            try:
                payload = r.json()
                server_ctx_size = payload.get("ctx_size") or payload.get("n_ctx") or payload.get("max_model_len")
            except Exception:
                server_ctx_size = None
        else:
            model_status = f"responding (status {r.status_code})"
    except Exception as exc:  # noqa: BLE001
        model_status = f"unreachable ({exc})"

    # vLLM reports model via /v1/models, not /props
    ctx_check = "unknown"
    try:
        models_url = MODEL_URL.replace("/v1/chat/completions", "/v1/models")
        rm = requests.get(models_url, timeout=5)
        if rm.status_code == 200:
            data = rm.json()
            # OpenAI shape: {data: [{id: "Qwen3.6-35B-A3B-NVFP4"}]}
            vllm_model_id = (data.get("data", [{}])[0].get("id") if isinstance(data.get("data"), list) else data.get("id"))
            if vllm_model_id:
                ctx_check = "match" if vllm_model_id == MODEL_NAME else "mismatch"
        # Also try legacy /props for llama fallback
        if ctx_check == "unknown" and _is_legacy_llama_url():
            props_url = MODEL_URL.replace("/v1/chat/completions", "/props")
            rp = requests.get(props_url, timeout=5)
            if rp.status_code == 200:
                server_slot = rp.json().get("default_generation_settings", {}).get("n_ctx")
                if server_slot is not None:
                    ctx_check = "match" if server_slot == slot_budget() else "mismatch"
    except Exception:
        pass

    # vLLM-native budget
    max_model_len = context_budget()
    return {
        "status": "healthy" if model_status == "ready" else "degraded",
        "model_server": model_status,
        "model_name": MODEL_NAME,
        "vllm_model_id": vllm_model_id,
        "context_budget": {
            "max_model_len": max_model_len,
            "slot_tokens": slot_budget(),  # legacy, kept for compat
            "prompt_room": usable_prompt_room(),
            "max_tokens_default": MODEL_MAX_TOKENS,
            "server_ctx_size": server_ctx_size,
            "check": ctx_check,
        },
        "default_year_mode": "current-year" if not DEFAULT_YEAR_ENV else f"pinned={DEFAULT_YEAR_ENV}",
        "version": VERSION,
        "guided_json": ENABLE_GUIDED_JSON,
        "thinking_disabled": DISABLE_THINKING,
    }


@app.post("/v1/extract", response_class=JSONResponse, tags=["Extract"])
async def extract(
    _api_key: str = Depends(verify_api_key),
    body: ExtractRequest = Body(...),
):
    """Parse a single GDS input into the schedule JSON (async vLLM path)."""
    default_year = resolve_default_year()
    try:
        # Use async path when possible for true non-blocking; fallback to sync for tests
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            # We are in async context — use http_model_call_async via run_extract wrapper
            # run_extract expects a sync callable, so we create an async-aware adapter:
            # Instead, call run_extract with a sync wrapper that delegates to async via to_thread
            # Simpler: just call run_extract with http_model_call (sync) — it already handles async probing
            # For true async we bypass run_extract and do the adaptive flow here:
            gds_text = body.gds_text
            messages = build_prompt(gds_text, default_year)
            adaptive = resolve_max_tokens(gds_text, MODEL_MAX_TOKENS)
            check_context(messages, max_tokens=adaptive)
            params = build_params(_PARAMS_LEVEL, gds_text=gds_text)
            params["max_tokens"] = adaptive
            import time as _t
            t0 = _t.perf_counter()
            raw = await http_model_call_async(messages, params)
            elapsed = _t.perf_counter() - t0
            schedule = extract_json(raw)
            logger.info(f"extract complete | record_type={schedule.get('Record type')} segments={len(schedule.get('segments', []))} elapsed={elapsed:.2f}s max_tokens={adaptive} guided={ENABLE_GUIDED_JSON}")
            return JSONResponse(content=schedule)
        except RuntimeError:
            # No running loop (test Client) — fallback to sync path
            schedule = run_extract(body.model_dump(), default_year, http_model_call)
            return JSONResponse(content=schedule)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except (ModelUnavailable, ContextExceeded) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Unexpected extract error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Processing failed.")


@app.post("/v1/extract_batch", response_class=JSONResponse, tags=["Extract"])
async def extract_batch(
    _api_key: str = Depends(verify_api_key),
    body: BatchRequest = Body(...),
):
    """Parse many GDS inputs. Sequential by default; concurrent when BATCH_CONCURRENCY>1."""
    default_year = resolve_default_year()
    try:
        if BATCH_CONCURRENCY > 1:
            import asyncio
            sem = asyncio.Semaphore(BATCH_CONCURRENCY)
            async def _one(entry: dict):
                async with sem:
                    eid = entry.get("id")
                    try:
                        gds_text = entry.get("gds_text")
                        messages = build_prompt(gds_text, default_year)
                        adaptive = resolve_max_tokens(gds_text or "", MODEL_MAX_TOKENS)
                        check_context(messages, max_tokens=adaptive)
                        params = build_params(_PARAMS_LEVEL, gds_text=gds_text or "")
                        params["max_tokens"] = adaptive
                        raw = await http_model_call_async(messages, params)
                        schedule = extract_json(raw)
                        return {"id": eid, "status": "ok", "schedule": schedule}
                    except Exception as exc:  # noqa: BLE001
                        logger.error("extract entry %s failed: %s", entry.get("id"), exc)
                        return {"id": entry.get("id"), "status": "error", "error": str(exc)}
            results = await asyncio.gather(*[_one(e.model_dump()) for e in body.entries])
            return JSONResponse(content={"results": list(results)})
        # Sequential fallback (sync)
        results = run_extract_batch(
            [entry.model_dump() for entry in body.entries],
            default_year,
            http_model_call,
        )
    except Exception as exc:  # noqa: BLE001
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

    logger.info(f"Starting AI GDS Extraction API v{VERSION} on {API_HOST}:{API_PORT}")
    logger.info(f"Model server: {MODEL_URL} ({MODEL_NAME}) — vLLM :8011 (MAX_MODEL_LEN={CONTEXT_SIZE})")
    logger.info(f"DISABLE_THINKING={DISABLE_THINKING} | temp={MODEL_TEMP} | top_p={MODEL_TOP_P} | ENABLE_GUIDED_JSON={ENABLE_GUIDED_JSON}")
    logger.info(
        f"CONTEXT_SIZE={CONTEXT_SIZE} MODEL_PARALLEL={MODEL_PARALLEL} | vLLM budget={context_budget()} tokens | legacy slot={slot_budget()} | prompt room={usable_prompt_room()} | guard={CONTEXT_GUARD} | BATCH_CONCURRENCY={BATCH_CONCURRENCY}"
    )
    # Prompt size instrumentation at startup
    logger.info(f"GDS_SYSTEM est_tokens={estimate_tokens(GDS_SYSTEM)} | adaptive sample 2-seg max_tokens={resolve_max_tokens('PR 221 test\\nFJ 920 test', MODEL_MAX_TOKENS)} | 10-seg est={resolve_max_tokens(open('tests/cases/amadeus_availability_input.txt').read() if __import__('os').path.exists('tests/cases/amadeus_availability_input.txt') else 'x'*2000, MODEL_MAX_TOKENS)}")
    logger.info(f"Default-year mode: {'current-year' if not DEFAULT_YEAR_ENV else 'pinned='+DEFAULT_YEAR_ENV}")
    logger.info(f"Swagger UI: http://{API_HOST}:{API_PORT}/docs")

    uvicorn.run("gds_extraction_service:app", host=API_HOST, port=API_PORT, log_level="info")
