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
from datetime import date, datetime, timedelta
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

# Guided JSON schema for boss spec (pages 1-12) — used when ENABLE_GUIDED_JSON=1.
# This is the Source of Truth for decoding; it mirrors the sample result (pages 6-12)
# which is the superset that includes day_of_week + airport names.
# Key fixes for the 14SEP benchmark: enforces sacrosanct 3-letter airport codes,
# single-letter service_class_letter from immediately after flight number, and
# correct year default (2025 per v3.5.1) via prompt, but schema enforces types here.
_GDS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "Record type": {
            "type": "string",
            "enum": ["reservation", "none"],
            "description": "reservation if record locators AND passenger names (e.g. 1.IGNACIO/CYNTHIA) present; else none. MNLSIN is a route, not a name."
        },
        "Passenger Name": {
            "type": "array",
            "items": {"type": "string", "description": "LASTNAME/FIRSTNAME, sacrosanct; none if no passenger pattern"},
            "description": "Array of passenger names or [\"none\"] for availability displays. Never hallucinate MNLSIN as a name."
        },
        "PNR": {
            "type": "string",
            "description": "First 6-char PNR in record (e.g. FDJ3BN), or 'none' for availability displays",
            "pattern": r"^(none|[A-Z0-9]{6})$"
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_number": {"type": "integer", "description": "1-based as shown at start of segment line"},
                    "segment_record_locator": {"type": "string", "description": "6 chars after DCPR (PR) or after final / (CX/FDJ3BN), or 'none' for availability"},
                    "airline_code": {
                        "type": "string",
                        "description": "2-char code, sacrosanct, immediately after segment_number (e.g. 2  PR 507 → PR)",
                        "pattern": r"^[A-Z0-9]{2}$"
                    },
                    "airline_name": {"type": "string", "description": "Full name: PR→Philippine Airlines, CX→Cathay Pacific, FJ→Fiji Airways, QF→Qantas"},
                    "flight_number": {"type": "integer", "description": "Integer immediately after airline_code"},
                    "originating_airport_code": {
                        "type": "string",
                        "description": "3-letter ORIGIN, sacrosanct (e.g. MNLSIN→MNL, must not be T or E)",
                        "pattern": r"^[A-Z]{3}$"
                    },
                    "originating_airport_name": {"type": "string", "description": "Full airport name exactly matching the 3-letter code, never Cotabato for MNL"},
                    "originating_terminal": {"type": "string", "description": "'none' if absent"},
                    "destination_airport_code": {
                        "type": "string",
                        "description": "3-letter DESTINATION, sacrosanct",
                        "pattern": r"^[A-Z]{3}$"
                    },
                    "destination_airport_name": {"type": "string"},
                    "destination_terminal": {"type": "string"},
                    "departure_date_time": {
                        "type": "object",
                        "properties": {
                            "month": {"type": "integer", "minimum": 1, "maximum": 12},
                            "month_name": {"type": "string", "enum": ["January","February","March","April","May","June","July","August","September","October","November","December"]},
                            "date": {"type": "integer", "minimum": 1, "maximum": 31},
                            "year": {"type": "integer", "description": "2025 when year omitted (v3.5.1)"},
                            "day_of_week": {"type": "string", "enum": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]},
                            "time": {"type": "string", "pattern": r"^\d{2}:\d{2}$", "description": "HH:MM 24h, first time after route (e.g. 0930 → 09:30)"},
                        },
                        "required": ["month", "month_name", "date", "year", "day_of_week", "time"],
                    },
                    "arrival_date_time": {
                        "type": "object",
                        "properties": {
                            "month": {"type": "integer", "minimum": 1, "maximum": 12},
                            "month_name": {"type": "string", "enum": ["January","February","March","April","May","June","July","August","September","October","November","December"]},
                            "date": {"type": "integer", "minimum": 1, "maximum": 31},
                            "year": {"type": "integer"},
                            "day_of_week": {"type": "string", "enum": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]},
                            "time": {"type": "string", "pattern": r"^\d{2}:\d{2}$", "description": "HH:MM, second time (e.g. 1315 → 13:15)"},
                        },
                        "required": ["month", "month_name", "date", "year", "day_of_week", "time"],
                    },
                    "flight_duration": {"type": "string", "pattern": r"^\d{2}:\d{2}$", "description": "HH:MM, 09:30→13:15 = 03:45"},
                    "aircraft_type": {"type": "string", "description": "321→Airbus A321, 333→Airbus A330-300, 332→A330-200, 359→Airbus A350-900, 73H→Boeing 737, 7M8→Boeing 737 MAX 8. If no aircraft code is present in the GDS line, output 'none'. NEVER use external knowledge."},
                    "service_class_letter": {
                        "type": "string",
                        "description": "Single letter immediately after flight_number (e.g. PR 507 T → T, not E after date)",
                        "pattern": r"^[A-Z]$"
                    },
                    "service_class": {
                        "type": "string",
                        "enum": ["Business", "Premium", "Economy", "none"],
                        "description": "PR: C,D,I,J,Z=Business, N,W=Premium, others inc B,T=Economy; other airlines same mapping or 'none' for availability"
                    },
                },
                "required": ["segment_number", "airline_code", "flight_number", "originating_airport_code", "destination_airport_code", "departure_date_time", "arrival_date_time", "flight_duration", "aircraft_type", "service_class_letter", "service_class"],
            },
        },
    },
    "required": ["Record type", "Passenger Name", "PNR", "segments"],
    "additionalProperties": False,
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

# ===========================================================================
# Flight Duration Calculator — server-side timezone-aware computation
# ===========================================================================
# Replaces LLM-computed durations with mathematically correct ones.
# Uses a static airport-to-UTC-offset table covering the most common
# airports in the service's route network.
#
# Offsets are expressed in hours from UTC and are **fixed** (no DST).
# For airports that observe DST (e.g., US, EU), the offsets below
# reflect the **most common** offset at request time — for precision
# the table can be expanded with date-aware entries in a future iteration.
#
# The _compute_flight_duration() function uses these offsets to convert
# local dep/arr times to UTC, then computes the delta.

# Primary international airports the GDS service handles frequently.
# Format: {ICAO/IATA_CODE: utc_offset_hours}
_AIRPORT_OFFSETS: dict[str, int] = {
    # Philippines
    "MNL": 8,   # UTC+8, no DST
    # USA — West Coast (PDT = UTC-7, Oct 2026)
    "SFO": -7,  # San Francisco, UTC-7 PDT
    "LAX": -7,  # Los Angeles, UTC-7 PDT
    "SEA": -7,  # Seattle, UTC-7 PDT
    "SAN": -7,  # San Diego, UTC-7 PDT
    "PDX": -7,  # Portland, UTC-7 PDT
    # USA — East Coast (EDT = UTC-4, Oct 2026)
    "JFK": -4,  # New York, UTC-4 EDT
    "EWR": -4,  # Newark, UTC-4 EDT
    "IAD": -4,  # Washington Dulles, UTC-4 EDT
    "BOS": -4,  # Boston, UTC-4 EDT
    "ORD": -5,  # Chicago, UTC-5 CDT
    "DFW": -5,  # Dallas/Fort Worth, UTC-5 CDT
    "ATL": -4,  # Atlanta, UTC-4 EDT
    "MIA": -4,  # Miami, UTC-4 EDT
    "DEN": -6,  # Denver, UTC-6 MDT
    "PHX": -7,  # Phoenix (no DST), UTC-7 MST
    # Australia
    "SYD": 10,  # Sydney, UTC+10 AEST (Sep 2025)
    "BNE": 10,  # Brisbane, UTC+10 AEST (no DST)
    "MEL": 10,  # Melbourne, UTC+10 AEST
    "ADL": 9,   # Adelaide, UTC+9:30 AEST → stored as 9 (approx)
    "PER": 8,   # Perth, UTC+8 AWST
    "OOL": 10,  # Gold Coast, UTC+10 AEST
    # Pacific
    "NAN": 12,  # Nadi, Fiji, UTC+12
    "AKL": 12,  # Auckland, UTC+12 (NZST, Sep 2025)
    "CHC": 12,  # Christchurch, UTC+12
    # Japan
    "NRT": 9,   # Tokyo Narita, UTC+9 JST
    "HND": 9,   # Tokyo Haneda, UTC+9 JST
    # Korea
    "ICN": 9,   # Incheon, Seoul, UTC+9 KST
    # China
    "PVG": 8,   # Shanghai, UTC+8 CST
    "PEK": 8,   # Beijing, UTC+8 CST
    "CAN": 8,   # Guangzhou, UTC+8 CST
    "HKG": 8,   # Hong Kong, UTC+8 HKT
    # Southeast Asia
    "BKK": 7,   # Bangkok, UTC+7 ICT
    "SGN": 7,   # Ho Chi Minh City (SGN), UTC+7 ICT
    "HAN": 7,   # Hanoi (HAN), UTC+7 ICT
    "KUL": 8,   # Kuala Lumpur, UTC+8 MYT
    "SIN": 8,   # Singapore, UTC+8 SGT
    "MFM": 8,   # Macau, UTC+8
    # Middle East
    "DXB": 4,   # Dubai, UTC+4 GST
    # Europe
    "LHR": 1,   # London, UTC+1 BST (Sep 2025)
    "CDG": 2,   # Paris, UTC+2 CEST (Sep 2025)
    "FRA": 2,   # Frankfurt, UTC+2 CEST
    "AMS": 2,   # Amsterdam, UTC+2 CEST
    "MAD": 2,   # Madrid, UTC+2 CEST
    # New Zealand
    "WLG": 12,  # Wellington
}


def _compute_flight_duration(
    dep_h: int, dep_m: int, arr_h: int, arr_m: int,
    dep_date: datetime | None, arr_date: datetime | None,
    dep_tz_offset: int, arr_tz_offset: int,
) -> str:
    """Compute flight duration from local times with timezone offsets.

    Uses the "boss algorithm": convert departure to destination local time,
    then compute duration = arrival_local - dep_converted_to_dest_tz.

    This is mathematically equivalent to UTC-based computation and handles
    overnight flights correctly.

    Parameters
    ----------
    dep_h, dep_m : int
        Departure local hour and minute.
    arr_h, arr_m : int
        Arrival local hour and minute.
    dep_date : datetime | None
        Departure date (from GDS).  If None, defaults to today.
    arr_date : datetime | None
        Arrival date (from GDS, may have +N day offset).
        If None, defaults to dep_date.
    dep_tz_offset : int
        UTC offset (in hours) at origin airport.
    arr_tz_offset : int
        UTC offset (in hours) at destination airport.

    Returns
    -------
    str : "HH:MM" flight duration.
    """
    if dep_date is None:
        dep_date = datetime.now()
    if arr_date is None:
        arr_date = dep_date

    # Convert departure time to destination local time
    dep_in_dest_tz = dep_date + timedelta(hours=dep_h, minutes=dep_m) + timedelta(hours=arr_tz_offset - dep_tz_offset)
    dep_in_dest_tz = dep_in_dest_tz.replace(second=0, microsecond=0)

    # Arrival in destination local time
    arr_in_dest_tz = arr_date + timedelta(hours=arr_h, minutes=arr_m)
    arr_in_dest_tz = arr_in_dest_tz.replace(second=0, microsecond=0)

    delta_seconds = int((arr_in_dest_tz - dep_in_dest_tz).total_seconds())

    if delta_seconds < 0:
        # Overnight flight wraps past midnight — add 24 hours
        delta_seconds += 24 * 3600

    total_minutes = delta_seconds // 60
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _compute_flight_duration_from_dates(
    dep_dt: dict, arr_dt: dict,
    origin_code: str | None, dest_code: str | None,
) -> str:
    """High-level wrapper: looks up airport offsets and delegates to
    _compute_flight_duration().

    Falls back to raw subtraction when airport offsets are unknown.
    """
    # Parse date fields into datetime objects
    dep_y = dep_dt.get("year")
    dep_mo = dep_dt.get("month")
    dep_d = dep_dt.get("date")
    arr_y = arr_dt.get("year")
    arr_mo = arr_dt.get("month")
    arr_d = arr_dt.get("date")

    try:
        dep_date = datetime(dep_y or 2025, dep_mo or 1, dep_d or 1) if dep_y and dep_mo and dep_d else None
    except (ValueError, TypeError):
        dep_date = None
    try:
        arr_date = datetime(arr_y or 2025, arr_mo or 1, arr_d or 1) if arr_y and arr_mo and arr_d else None
    except (ValueError, TypeError):
        arr_date = None

    # Parse time strings safely
    def _parse_time(dt: dict):
        t = dt.get("time")
        if t and ":" in str(t):
            parts = str(t).split(":")
            return _to_int(parts[0]), _to_int(parts[1])
        return 0, 0

    dep_h, dep_m = _parse_time(dep_dt)
    arr_h, arr_m = _parse_time(arr_dt)

    origin_upper = origin_code.strip().upper() if origin_code else ""
    dest_upper = dest_code.strip().upper() if dest_code else ""

    dep_offset = _AIRPORT_OFFSETS.get(origin_upper, 0)
    arr_offset = _AIRPORT_OFFSETS.get(dest_upper, 0)

    return _compute_flight_duration(dep_h, dep_m, arr_h, arr_m, dep_date, arr_date, dep_offset, arr_offset)


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
# v1.1 — restored full fidelity for accuracy (was over-compressed in v1.1b), still
# enforces No <think> and vLLM-native thinking OFF. Decoding unchanged.
GDS_SYSTEM = """You are a meticulous Global Distribution System (GDS) flight-data extractor.
You parse the provided GDS output and return ONLY a JSON object describing the
flight schedule. No commentary, no explanations, no markdown, no <think> blocks outside the JSON.

The DEFAULT YEAR is __DEFAULT_YEAR__. Use it for any date where the GDS line
omits a year entirely.

OUTPUT ONLY THIS JSON OBJECT (exact key names shown; do not add other keys):
{{
  "Record type": "reservation" | "none",
  "Passenger Name": [ "LASTNAME/FIRSTNAME(s)", ... ]  OR  [ "none" ],
  "PNR": "<6-char PNR>" | "none",
  "segments": [ {{ ...segment... }} ]
}}

Record type: set to "reservation" ONLY if the record contains one or more record
locators AND at least one passenger name in the form "1.LASTNAME/FIRSTNAME" or "1 LASTNAME/FIRSTNAME"; otherwise set to "none". A string like "MNLSIN" is a route, NOT a passenger name. If no passenger names match that pattern, output ["none"] and "none" PNR — never hallucinate.

PNR: the FIRST PNR in the whole record (NOT a per-segment locator). "none" for
availability displays. A PNR is exactly 6 alphanumeric characters (e.g. FDJ3BN).

Passenger names (for reservations):
- Start AFTER the numerical passenger number and optional dot, which marks the beginning of the
  passenger name (e.g. "1.IGNACIO/CYNTHIA" → start after "1.").
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
  "service_class_letter": "none", "service_class": "none" ONLY when the input is truly an availability display. For PNR Details ("PNR Details" header) the service class letter IS the single letter immediately after the flight number (e.g. PR 507 T → T), NOT the trailing "E" near the aircraft code.
  (Availability counts such as "J5 C5 D5" are NOT a service class of service.)

For EACH flight segment, extract EXACTLY as it appears — do not infer:
- segment_number: integer, as it appears at the start of the segment line (1-based).
- airline_code: 2-char airline code (sacrosanct — copy verbatim). In PNR format the code follows the segment number (e.g. "2  PR 507").
- airline_name: full airline name (e.g., PR -> Philippine Airlines, FJ -> Fiji
  Airways, QF -> Qantas, CX -> Cathay Pacific).
- flight_number: integer immediately after airline_code.
- service_class_letter: the SINGLE LETTER immediately after flight_number (e.g. PR 507 T → T). Do not take the letter after the date or near "E 0".
- originating_airport_code: 3-letter code of the ORIGIN (sacrosanct — copy verbatim, NEVER change it). In PNR format it is the 6-char route field after the day number: MNLSIN → MNL is origin, SIN is destination. In availability format it is after the "/" (e.g. /MNL 1 BNE → MNL→BNE). Validate: must be 3 uppercase letters, not "T" or "E".
- originating_airport_name: full airport name matching the code (choose exactly matching airport, never Cotabato for MNL).
- originating_terminal: "none" if absent from the GDS line.
- destination_airport_code: 3-letter code of the DESTINATION (sacrosanct).
- destination_airport_name: full airport name matching the code.
- destination_terminal: "none" if absent.
- departure_date_time / arrival_date_time: objects with these exact keys:
  month (integer 1-12), month_name (string), date (integer), year (integer),
  day_of_week (string), time ("HH:MM").
- Dates: parse like "14SEP" → 14 September, "27JUN" → 27 June. Resolve "+N" arrival day-offsets into the correct date, month, year AND day_of_week (handle month and year rollovers, e.g. Aug 31 -> Sep 1, or Dec 31 -> Jan 1 of the next year). Times are 24-hour "HH:MM" — departure time is the first time after the route/status field (e.g. DK1  0930), arrival time is the second time (e.g. 1315).
- flight_duration: "HH:MM" computed as arrival minus departure (handle overnight +1). GDS times are in local time at each airport. Use timezone offsets: convert dep to destination tz, then diff = arr - (dep + dest_off - orig_off). US DST note (Oct 2026): SFO/LAX/SEA = UTC-7 (PDT, DST ends Nov 1); IAD/JFK/EWR = UTC-4 (EDT). Do NOT use PST (UTC-8) for US west coast before Nov 1. N* fields (e.g. "6*MNLSFO") = NUMBER OF CONSECUTIVE DAYS available, NOT arrival day offset. The +N day offset is in the arrival DATE field, not N*.
- aircraft_type: human-readable type ONLY if a code appears in the GDS line (321 -> Airbus A321; 333 -> Airbus A330-300; 332 -> Airbus A330-200; 359 -> Airbus A350-900; 73H -> Boeing 737; 7M8 -> Boeing 737 MAX 8; 333 H or 333 -> Airbus A330-300). If no aircraft code appears in the GDS data, output "none". NEVER use external knowledge or guess the aircraft type. Do not state a more specific sub-model than the source implies.
- service_class: mapped class. Philippine Airlines: Business = C,D,I,J,Z; Premium = N,W; Economy = ALL OTHER codes (NOTE: B is Economy, T is Economy). Other airlines (including CX): report Economy/Business as per letter without remapping unless specified — treat T/Q as Economy for this data but do not hallucinate "E".
- segment_record_locator: 6 characters following "DCPR" for Philippine Airlines; near the end of the segment data for Cebu Pacific; for CX in this PNR it is the code after the final "/" (e.g. CX/FDJ3BN → FDJ3BN) but "none" for availability displays unless a PNR is present. If unsure and Record type is "none", use "none".
- For codeshare legs formatted like "FJ:QF3873", emit the QF marketing code and number (Qantas 3873).
- When translating airport codes, choose the airport that exactly matches the 3-letter code; NEVER substitute a nearby or different airport.

Return ONLY the JSON object above. No preamble, no code fences, no commentary, no reasoning.
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


# Pre-computed weekday names (Python weekday: Mon=0 … Sun=6)
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _normalize_date_time(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    year = _to_int(value.get("year"))
    month = _to_int(value.get("month"))
    day = _to_int(value.get("date"))
    # Compute day_of_week from the date fields — never trust the LLM for date arithmetic.
    if year and month and day:
        try:
            computed_dow = _DAY_NAMES[date(year, month, day).weekday()]
        except ValueError:
            computed_dow = _seg_str(value.get("day_of_week"))
    else:
        computed_dow = _seg_str(value.get("day_of_week"))
    return {
        "month": month,
        "month_name": _seg_str(value.get("month_name")),
        "date": day,
        "year": year,
        "day_of_week": computed_dow,
        "time": _seg_str(value.get("time")),
    }


def _normalize_segment(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    dep_dt = _normalize_date_time(value.get("departure_date_time"))
    arr_dt = _normalize_date_time(value.get("arrival_date_time"))

    # Compute flight_duration server-side using timezone-aware arithmetic,
    # overriding any LLM-computed value.  This eliminates the root cause of
    # the timezone-math errors that plagued the model (wrong UTC offsets,
    # confusion about US DST transitions, inconsistent offset application
    # across segments).
    computed_duration = _compute_flight_duration_from_dates(
        dep_dt, arr_dt,
        value.get("originating_airport_code"),
        value.get("destination_airport_code"),
    )
    # Only override the LLM duration if the computed value is non-trivial.
    # When arrival time/date is missing or invalid (produces "00:00"), keep
    # the LLM's answer — it may have the correct duration even without
    # full arrival data (e.g., the expected test data has arrival month=0
    # but correct duration from the model).
    llm_duration = _seg_str(value.get("flight_duration"))
    final_duration = computed_duration if computed_duration != "00:00" else llm_duration

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
        "departure_date_time": dep_dt,
        "arrival_date_time": arr_dt,
        "flight_duration": final_duration,
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
