# AI GDS Extraction — NVIDIA DGX Spark

Lean, JSON-in / JSON-out **GDS Extractor** — the local-model migration of Toby's API-based
"GDS Extractor v3.6" (`make schedule`). It parses raw **GDS output lines** (Amadeus availability
displays and complete reservations) into the structured flight-segment JSON defined in
`implementation.md`, running fully offline on the NVIDIA DGX Spark by connecting to Toby's
already-running **llama.cpp (CUDA)** model server (`Qwen3.8-27B`). This service is a **gateway
only** — it never starts or stops the model server.

> Built on the proven `start.sh / stop.sh` + FastAPI skeleton from Proof‑Reader and QA‑Manager,
> adapted to the same pre‑existing shared model server (no `setup.sh`, no model download).

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Deploy Order](#deploy-order)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Output Schema](#output-schema)
- [Configuration Reference](#configuration-reference)
- [Context Sizing (important)](#context-sizing-important)
- [Local Development & Testing](#local-development--testing)
- [Troubleshooting](#troubleshooting)
- [Files](#files)

---

## What It Does

- **`POST /v1/extract`** — parse a single GDS input into the schedule JSON.
- **`POST /v1/extract_batch`** — parse many GDS inputs sequentially, with per-entry failure isolation.
- **`POST /v1/version`** — instant parity command (never touches the model).

The prompt encodes Toby's full spec verbatim:

- **Record type** — `reservation` (record locators AND passenger names present) or `none`.
- **PNR** — the *first* PNR in the whole record, not a per-segment locator.
- **Passenger names** — extract every name (up to 9), preserving the original `LASTNAME/FIRSTNAME(s)`
  format including prefixes/trailing characters; an `APDI`-prefixed last name is kept intact.
- **Per segment** — airline code/name, flight number, origin/destination codes (sacrosanct — only
  the *name* is translated), terminals (`none` when absent), departure/arrival date-times (with
  `+N` day-offset and month/year rollovers resolved), `flight_duration`, aircraft type, and service
  class (Philippine Airlines: Business = C,D,I,J,Z; Premium = N,W; everything else incl. **B** is
  Economy; other carriers reported as-is).
- **Availability displays** (no PNR/names) → `Record type: "none"`, `Passenger Name: ["none"]`, and
  per-segment `segment_record_locator`/`service_class_letter`/`service_class` = `none`.
- **Default year** — uses the configured `DEFAULT_YEAR` (default: current year) when the GDS line omits one.

**Output:** a single schedule object (see [Output Schema](#output-schema)).

---

## Architecture

```
   Client (curl / n8n / UI)
               │  POST /v1/extract      {gds_text}
               │  POST /v1/extract_batch{entries:[{id,gds_text}]}
               │  x-api-key: <key>
               ▼
       ┌──────────────────────────────────────────────────────────────────┐
       │              DGX Spark (GB200 / Blackwell, unified memory)        │
       │                                                                     │
       │  Proof-Reader's startserver.sh  (owns the shared model server)     │
       │     └── llama-server :8006  CUDA, text mode, --jinja,               │
       │                    4 parallel slots · CONTEXT_SIZE 65536 · 16384/slot
       │                                                                     │
       │  start.sh   (THIS project's gateway — connects, never launches)    │
       │     └── FastAPI :8084  gds_extraction_service.py                   │
       │              ├── client-side context guard (422, before network)    │
       │              ├── extract_json / _normalize_gds (schema coercion)    │
       │              └── http_model_call → /v1/chat/completions (auth,       │
       │                                 thinking-suppression degradation)    │
       └──────────────────────────────────────────────────────────────────┘
                │
                ▼
       llama-server :8006  (Toby's Qwen3.8-27B, CUDA)
          • start.sh / stop.sh NEVER start or kill it
          • its --ctx-size / --parallel are fixed at its OWN launch
```

- **Gateway‑only access:** clients talk to the FastAPI gateway on **:8084** and never reach the model
  server directly.
- **The model is pre‑existing.** `start.sh` only *probes* `/health`; it does not launch or manage
  `llama-server`. `stop.sh` stops **only** the gateway and never touches the model.
- **Server ownership.** The shared server lives in the **Proof‑Reader** repo
  (`E:\Projects\Proof-Reader\startserver.sh`). It is launched **once** and shared across AI solutions.
  This project is a thin client; it reads `CONTEXT_SIZE`/`MODEL_PARALLEL` to compute its own slot
  budget for the guard and the `/healthz` cross-check.
- **JSON in / JSON out.** The model server can't be relaunched with llama.cpp's JSON‑schema flag, so
  the gateway **prompts the model to emit JSON** and **parses it client‑side** (`extract_json` +
  `_normalize_gds`): it strips thinking blocks / code fences / preamble, locates the JSON span,
  coerces it to the exact schema, and — unlike the QA Manager, where raw text degrades gracefully —
  **fails closed** (never fabricates a schedule) with a `503` when nothing parseable is found.

---

## Deploy Order

The model server is **shared and long-lived**. It must be running **before** the GDS-Extraction gateway.

```bash
# (1) From the Proof-Reader project — start the shared model server ONCE.
cd E:\Projects\Proof-Reader
./startserver.sh          # llama-server (text, --jinja) on :8006

# (2) From the GDS-Extraction project — start the gateway and connect to :8006.
cd E:\Projects\GDS-Extraction
./start.sh                # FastAPI gateway on :8084
```

> **Pre-flight is strict.** `start.sh` waits up to 120s for the shared model server (`:8006`) to be
> healthy. If it isn't reachable, it **stops and tells you to start it first** — run
> `./startserver.sh` (from the Proof‑Reader project), then `./start.sh`. The gateway never starts
> half-open.

> **No cross-repo changes needed.** QA‑Manager's v2 already re-provisioned the shared server
> (`--jinja`, `MODEL_PARALLEL=4`, `CONTEXT_SIZE=65536`). This project only *verifies* those flags and
> mirrors them in its `.env`. See [Context Sizing](#context-sizing-important).

---

## Requirements

- **NVIDIA DGX Spark** (GB200 Grace Blackwell, unified memory, Linux, CUDA drivers).
- The model server (`llama-server :8006`, `Qwen3.8-27B`, `--jinja`) **already running** — see
  "Context Sizing" below.
- `python3`, and a Python venv (auto‑created by `start.sh` on first run).
- `curl` (for testing).

---

## Quick Start

> All steps run **on the DGX Spark** (Linux). The shell scripts are developed on the Windows machine
> but executed on the DGX.

### 1. Launch

```bash
cd E:\Projects\GDS-Extraction
./start.sh
```

`start.sh` bootstraps a `.venv` + installs dependencies **once** (if absent), copies `.env.example`
→ `.env` **once** (if absent) and prints a note to review, then runs a **strict pre-flight**: it
waits up to 120s for the shared model server (`:8006`) to be healthy. If it's reachable, it starts the
FastAPI gateway on **:8084**:

```
======== GDS Extraction is LIVE ========
  API:    http://0.0.0.0:8084
  Docs:   http://0.0.0.0:8084/docs
  Health: http://0.0.0.0:8084/healthz
  Shared model server on :8006 (managed by Proof-Reader startserver.sh)
  Press Ctrl+C to stop the gateway only — the model server is left running.
========================================
```

### 2. Configure (optional)

Review/edit `.env` — API keys, model URL, ports, context sizing, and the default year:

```bash
nano .env
```

### 3. Test

```bash
curl -s -X POST http://localhost:8084/v1/extract \
  -H "x-api-key: gds_key_0000" -H "Content-Type: application/json" \
  -d '{"gds_text":"AN3SEPMNLNAN ** AMADEUS AVAILABILITY - AN **\n..."}'
```

### 4. Stop

```bash
./stop.sh
```

Stops **only the gateway** (PID‑file based, with a gateway‑only name fallback). The shared model
server is left untouched.

---

## API Reference

Base URL: `http://<dgx-spark>:8084`

### `POST /v1/extract`

Parse a single GDS input into the schedule JSON. **Requires the `x-api-key` header.**

**Request**

```json
{ "gds_text": "AN3SEPMNLNAN ** AMADEUS AVAILABILITY - AN **\nNANNADI.FJ 32 TU 03SEP…" }
```

The GDS text is placed into a delimited section in the prompt to reduce injection bleed.

**Success (`200`)** — one schedule object; see [Output Schema](#output-schema).

**`POST /v1/version`** — instant, never touches the model.

```json
{ "version": "1.0" }
```

**`POST /v1/extract_batch`** — parse many entries sequentially. Each entry is processed with its own
model call so one bad input cannot poison the others or overflow the shared slot budget. Overall
status is `200`; each item reports `status: "ok"` or `status: "error"`.

```json
// request
{ "entries": [
  { "id": "req-001", "gds_text": "AN3SEP…" },
  { "id": "req-002", "gds_text": "AN3SEP…" }
]}
// response
{ "results": [
  { "id": "req-001", "status": "ok",  "schedule": { … } },
  { "id": "req-002", "status": "error", "error": "could not find a JSON object in model output" }
]}
```

**Errors**

| Status | Meaning |
|--------|---------|
| `401` | Missing or invalid `x-api-key`. |
| `422` | Malformed body / missing or empty required field. |
| `422` | **Over-budget prompt** — the client-side context guard rejected it *before any network call*. The body contains exact sizing guidance (per-slot budget, `CONTEXT_SIZE`, `MODEL_PARALLEL`) and how to re-provision the server. |
| `503` | Model server unreachable / timed out / returned empty. |
| `503` | **Unparseable model output.** Extraction **fails closed** (never fabricates a schedule) — try again, or raise the shared server's context window. |
| `503` | Request exceeds the model's context window (server-side overflow). The model server's ctx is fixed at launch — restart `llama-server` with a larger `--ctx-size` and set `CONTEXT_SIZE` to match. |
| `500` | Unexpected pipeline error. |

**Examples**

```bash
# Single entry (Toby's Amadeus availability sample is in tests/cases/amadeus_availability_input.txt)
curl -s -X POST http://localhost:8084/v1/extract \
  -H "x-api-key: gds_key_0000" -H "Content-Type: application/json" \
  -d '{"gds_text":"AN3SEPMNLNAN ** AMADEUS AVAILABILITY - AN **\nNANNADI.FJ 32 TU 03SEP…"}'

# Batch (processed sequentially; bad entries are isolated per-item)
curl -s -X POST http://localhost:8084/v1/extract_batch \
  -H "x-api-key: gds_key_0000" -H "Content-Type: application/json" \
  -d '{"entries":[{"id":"r1","gds_text":"AN3SEP…"},{"id":"r2","gds_text":"AN3SEP…"}]}'

# Version (instant, no model call)
curl -s -X POST http://localhost:8084/v1/version
```

Interactive API docs: open **http://<dgx-spark>:8084/docs**.

### `GET /healthz`

Liveness probe; reports model‑server readiness, the computed per‑slot context budget, a
cross-check of that budget against the running server's per‑slot `n_ctx`, and the current default-year
mode.

```bash
curl -s http://localhost:8084/healthz
# {"status":"healthy","model_server":"ready","model_name":"Qwen3.8-27B",
#  "context_budget":{"slot_tokens":16384,"server_ctx_size":65536,"check":"match"},
#  "default_year_mode":"current-year","version":"1.0"}
```

- `context_budget.slot_tokens` = `CONTEXT_SIZE // MODEL_PARALLEL` (the gateway's assumed budget).
- `check` = `match` / `mismatch` / `unknown`: whether the gateway's per‑slot budget matches the
  server's actual per‑slot `n_ctx`. A `mismatch` means the `.env` sizing doesn't match the running
  server — re-provision and restart.

---

## Output Schema

```json
{
  "Record type": "reservation",
  "Passenger Name": ["SMITH/JOHN MR", "APDI/CORP TRAVEL"],
  "PNR": "ABC123",
  "segments": [
    {
      "segment_number": 1,
      "segment_record_locator": "XYZ789",
      "airline_code": "PR",
      "airline_name": "Philippine Airlines",
      "flight_number": 221,
      "originating_airport_code": "MNL",
      "originating_airport_name": "Ninoy Aquino International Airport",
      "originating_terminal": "1",
      "destination_airport_code": "BNE",
      "destination_airport_name": "Brisbane Airport",
      "destination_terminal": "International",
      "departure_date_time": {
        "month": 9, "month_name": "September", "date": 3,
        "year": 2025, "day_of_week": "Wednesday", "time": "23:45"
      },
      "arrival_date_time": {
        "month": 9, "month_name": "September", "date": 4,
        "year": 2025, "day_of_week": "Thursday", "time": "09:30"
      },
      "flight_duration": "07:45",
      "aircraft_type": "Airbus A321",
      "service_class_letter": "M",
      "service_class": "Economy"
    }
  ]
}
```

- Airport **codes are sacrosanct** — copied verbatim; only the name is translated.
- `+N` arrival-day offsets are resolved into the correct date, month, year, and day-of-week (month
  and year rollovers included). Times are 24-hour `HH:MM`.
- For availability displays: `Record type: "none"`, `Passenger Name: ["none"]`, `PNR: "none"`, and
  per-segment `segment_record_locator` / `service_class_letter` / `service_class` = `none`.

---

## Configuration Reference

All keys live in `.env` (see `.env.example`). The defaults here **are** the code defaults.

| Key | Default | Purpose |
|-----|---------|---------|
| `MODEL_URL` | `http://127.0.0.1:8006/v1/chat/completions` | Endpoint of the **running** model server. |
| `MODEL_NAME` | `Qwen3.8-27B` | Reported to the server (matches the shared launch). |
| `LLAMA_SERVER_API_KEY` | `sk-internal-proofreader` | Bearer token the gateway presents to the **shared** server (sent only if non‑empty). **Must match** the shared server's `--api-key`. |
| `CONTEXT_SIZE` | `65536` | Total ctx of the shared server. **Must match** the server's launch flags. See [Context Sizing](#context-sizing-important). |
| `MODEL_PARALLEL` | `4` | Shared-server slots; **must match** the server's launch. Per-slot budget = `CONTEXT_SIZE // MODEL_PARALLEL` = 16,384. |
| `MODEL_MAX_TOKENS` | `8192` | Max output tokens. GDS manifests are large (the golden sample alone emits ~10 segments ≈ 4k+ tokens); 8192 gives truncation headroom. Raising it reduces the local prompt-room budget. |
| `MODEL_TEMP` / `MODEL_TOP_P` / `MODEL_TOP_K` | `0.0` / `0.5` / `40` | Sampling. **Temp 0 (greedy)** gives run-to-run deterministic extraction — validated on-DGX. |
| `DISABLE_THINKING` | `1` | Request `reasoning_effort=0`; send `chat_template_kwargs:{enable_thinking:false}` (best-effort). `extract_json()` also strips leaked thinking defensively. |
| `CONTEXT_GUARD` | `strict` | Client-side context guard: `strict` (422 + guidance), `warn` (allow + log), or `off` (testing). |
| `REQUEST_TIMEOUT` | `300` | Timeout to the model server, in seconds. **Batch requests multiply this per entry.** |
| `DEFAULT_YEAR` | `""` (empty → current year at request time) | Year used when the GDS line omits one. Pin a year without a code change, e.g. `DEFAULT_YEAR="2025"`. |
| `API_PORT` / `API_HOST` | `8084` / `0.0.0.0` | Gateway bind. **Not 8082** (Proof‑Reader) or **8083** (QA‑Manager). |
| `API_KEY_AUTH_HEADER` | `x-api-key` | Header name for client auth. |
| `API_KEYS` | `gds_key_0000:GDS Extraction Local Testing` | Comma‑separated valid client keys. Replace with your own secrets. |

**Port map**

| Port | Owner | Managed by |
|------|-------|-----------|
| `8006` | Shared `llama-server` (`Qwen3.8-27B`, CUDA) | `Proof-Reader/startserver.sh` |
| `8082` | Proof‑Reader gateway | `Proof-Reader/start.sh` |
| `8083` | QA‑Manager gateway | `QA-Manager/start.sh` |
| `8084` | **GDS‑Extraction gateway** | `start.sh` (this project) |

**Rotating API keys:** edit `.env` and restart the gateway (`./stop.sh && ./start.sh`). Keys load at
startup. This only restarts the FastAPI gateway — the shared model server keeps running.

---

## Context Sizing (important)

This is the shared root cause **RC1**: llama.cpp splits its total `--ctx-size` **across** `--parallel`
slots, so the per-slot budget is `CONTEXT_SIZE // MODEL_PARALLEL`. GDS manifests are large, so
prompt headroom matters.

**Prompt-room budget** = `slot_tokens − MODEL_MAX_TOKENS − 256` safety margin. With the default
balanced profile (`16,384` per-slot) and `MODEL_MAX_TOKENS=8192`, that leaves ≈ **7,936 tokens** of
prompt headroom (~23.8k chars of GDS text) — ample for availability displays; the guard catches
outliers.

**Sizing matrix.** Set `CONTEXT_SIZE` and `MODEL_PARALLEL` in this repo's `.env` to **match** the
shared server's launch flags (Proof‑Reader's `startserver.sh`):

| Profile | `CONTEXT_SIZE` | `MODEL_PARALLEL` | Per-slot budget | Prompt room (with max_tokens 8192) |
|---------|---------------|------------------|-----------------|------------------------------------|
| Conservative | `32768` | `4` | 8,192 | 0 *(slot ≈ max_tokens — too little headroom for GDS | raise `MODEL_MAX_TOKENS` or the server ctx)* |
| **Balanced (default)** | `65536` | `4` | **16,384** | **7,936** |
| Maximum | `131072` | `2` | 65,536 *(verify GPU memory first)* | 57,088 |

The current shared server is provisioned at the **balanced** profile (`65536 / 4`).

**If a request overflows it**, the client-side guard rejects it with a `422` *before* any network
call and tells you exactly what to do. To allow larger inputs, re-provision the shared server:

```bash
# 1. From Proof-Reader: stop + re-provision the shared server with a larger window
#    (e.g. 131072 ctx, 2 parallel slots) — requires GPU-memory headroom.
cd E:\Projects\Proof-Reader
./stopserver.sh && ./startserver.sh     # startserver.sh sources .env; edit it first

# 2. Set the SAME values in .env to match the re-provisioned server.
#       CONTEXT_SIZE="131072"
#       MODEL_PARALLEL="2"

# 3. Restart the gateway so it picks up the new sizing.
cd E:\Projects\GDS-Extraction
./stop.sh && ./start.sh

# 4. Verify /healthz reports context_budget.check == "match".
curl -s http://localhost:8084/healthz
```

---

## Local Development & Testing

The core logic (`build_prompt`, `build_params`, `estimate_tokens`, `check_context`, `extract_json`,
`_normalize_gds`, `run_extract`, `run_extract_batch`) is **pure** and takes an injectable model call
— so the entire business logic is unit‑tested **on the Windows dev machine with a stub backend, no GPU
needed.**

```bash
# From the repo root
pip install -r requirements.txt   # includes pytest + httpx
pytest -v
```

The suite (`tests/test_extract.py`) covers the pure pipeline, thinking-suppression parameter
degradation, the client-side context guard (strict 422 / warn / off), JSON extraction (thinking‑block
/ fence / preamble stripping + **fail-closed** behavior), schema coercion (`_normalize_gds`),
single/batch orchestration with per-entry isolation, the HTTP layer (auth `401`, validation `422`,
over-budget `422`, unavailable `503`, unparseable `503`), the content-resolution chain
(`reasoning_content`), the `/healthz` budget cross-check, a golden-case plumbing check against
Toby's documented Amadeus sample, and a config-drift test pinning `.env.example` to the code defaults.
**All 62 cases pass locally.**

The Amadeus availability sample (input + expected 10-segment output) lives in
`tests/cases/` for on‑DGX smoke‑testing. See `implementation.md` §2 Phase E for the remaining on-DGX
verifications (determinism ×3, the reservation case, and a concurrency check that the sibling gateways
still work alongside :8084).

---

## Troubleshooting

**`422` over-budget prompt.**
The client-side context guard rejected the request because the estimated prompt exceeds the per-slot
budget, *before* any network call. The response body contains exact sizing guidance. Shorten the GDS
input, or raise the shared server's per-slot budget and restart (see [Context Sizing](#context-sizing-important)).

**`/healthz` reports `mismatch` under `context_budget.check`.**
The gateway's per-slot budget (`CONTEXT_SIZE // MODEL_PARALLEL`) doesn't match the running server's
per-slot `n_ctx`. The server was launched with different flags than `.env` states — re-provision it
(matching `.env`) and restart the gateway. See [Context Sizing](#context-sizing-important).

**`/healthz` reports `degraded` / `unreachable`.**
The gateway couldn't reach `llama-server :8006`. Wait for the model to be up. If it keeps failing,
confirm it's running (`nvidia-smi`, `curl http://127.0.0.1:8006/health`) and that `MODEL_URL` in
`.env` points at the right port. `start.sh`'s startup pre-flight fails fast (up to 120s) if the
shared server isn't healthy — run `./startserver.sh` (Proof‑Reader project) first, then `./start.sh`.

**`503` — "could not find a JSON object" / unparseable.**
Extraction **fails closed** (never fabricates a schedule). This can mean the model returned prose
instead of JSON, or emitted malformed JSON. Retry; if it persists, confirm `DISABLE_THINKING=1` and
that the server was started with `--jinja`, and that the GDS input isn't truncated. See "thinking" below.

**`503` — "Model server unavailable."**
`llama-server` isn't reachable / timed out / returned empty. Confirm it's running and reachable, and
that `LLAMA_SERVER_API_KEY` (`.env`) matches the shared server's `--api-key`; a key mismatch surfaces
as a 503.

**The model emits reasoning/"thinking" text.**
Confirm `DISABLE_THINKING=1` in `.env` and that the shared server was started with **`--jinja`**
(Proof‑Reader's `startserver.sh`). The response resolver also falls back to `reasoning_content`;
`extract_json()` strips any leaked `<think>` / `<|begin_thinking|>` blocks defensively. Check the
gateway log for a warning about extraction.

**The model never resolves `+N` arrival dates / day-of-week correctly.**
This is prompt-driven (the gateway does no date math — the spec chose a pure-prompt approach). If a
specific rule is consistently wrong, tighten the relevant rule in the `GDS_SYSTEM` prompt in
`gds_extraction_service.py`, then re-run the golden smoke test.

**A batch entry is broken but others succeed.**
Expected. Each entry is isolated (`status: "error"`, with the reason in `error`); good entries are
never lost. See the batch endpoint in [API Reference](#api-reference).

**API key rejected.**
Confirm the key in `API_KEYS` (`.env`) matches the `x-api-key` you send.

---

## Files

| File | Purpose |
|------|---------|
| `gds_extraction_service.py` | Single‑file FastAPI app: `.env` config, pure pipeline functions (prompt, params, guard, extract_json, `_normalize_gds`, single + batch extract), HTTP model backend (thinking-suppression degradation + content resolution), routes, lean API‑key auth. |
| `start.sh` | One‑entry launch: bootstrap `.venv`/`.env`, pre-flight probe the shared model server, start the gateway only. |
| `stop.sh` | Graceful shutdown of the **gateway only** (never the shared model server). |
| `.env.example` | Config template. Copy to `.env` and edit. **Must match the code defaults** (enforced by test). |
| `requirements.txt` | Runtime deps (`fastapi`, `uvicorn`, `requests`, `python-dotenv`) + dev/test (`pytest`, `httpx`). |
| `tests/test_extract.py` | pytest suite (stub backend) — runs on the dev machine, no GPU. |
| `tests/cases/amadeus_availability_input.txt` | Toby's Amadeus availability sample input. |
| `tests/cases/amadeus_availability_expected.json` | Expected 10-segment output — drives the golden-case plumbing test. |
| `implementation.md` | Design + roadmap (architecture decisions, contract, risks). |
