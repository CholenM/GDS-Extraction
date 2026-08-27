# AI GDS Extraction — NVIDIA DGX Spark (vLLM)

Lean, JSON-in / JSON-out **GDS Extractor** — the local-model migration of Toby's API-based
"GDS Extractor v3.6" (`make schedule`). It parses raw **GDS output lines** (Amadeus availability
displays and complete reservations) into the structured flight-segment JSON defined in
`implementation.md`, running fully offline on the NVIDIA DGX Spark via the **vLLM server**
`E:\DGXSpark_Setup\vllm-qwen` (`Qwen3.6-35B-A3B-NVFP4`, NVFP4/FP8, `:8011`). This service is a **gateway
only** — it never starts or stops the model server.

> v1.1 — vLLM Performance Fix: same contract & greedy decoding (`temp 0`), but async gateway,
> adaptive `max_tokens` (3072 default), compressed prompt, and vLLM-native sizing for sub-60s p50.

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
- **`POST /v1/extract_batch`** — parse many GDS inputs (sequential by default, `BATCH_CONCURRENCY>1` for parallel), with per-entry failure isolation.
- **`POST /v1/version`** — instant parity command (never touches the model).

The prompt encodes Toby's full spec verbatim (compressed in v1.1, decoding unchanged):

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
       │              DGX Spark (GB200 / Blackwell, unified 121.69 GiB)      │
       │                                                                     │
       │  vllm-qwen :8011  (E:\DGXSpark_Setup\vllm-qwen)                     │
       │     └── vLLM MAIN — Qwen3.6-35B-A3B-NVFP4 (MoE 35B/3B active)        │
       │                    NVFP4 experts + FP8 KV, --enable-prefix-caching  │
       │                    MAX_MODEL_LEN 32768 · GPU util 0.60→0.85          │
       │                                                                     │
       │  start.sh   (THIS project's gateway — connects, never launches)    │
       │     └── FastAPI :8084  gds_extraction_service.py (async httpx)      │
       │              ├── adaptive max_tokens (1200+280*segs, 1024-4096)     │
       │              ├── compressed GDS_SYSTEM (<1100 toks)                 │
       │              ├── client-side guard (422, vLLM budget)                │
       │              ├── extract_json / _normalize_gds (schema coercion)    │
       │              └── http_model_call_async → /v1/chat/completions        │
       │                                 (chat_template_kwargs:{enable_thinking:false}) │
       └──────────────────────────────────────────────────────────────────┘
```

- **Gateway‑only access:** clients talk to `:8084` and never reach vLLM directly.
- **The model is pre‑existing.** `start.sh` only *probes* `/health` + `/v1/models` (fallback). `stop.sh` stops **only** the gateway.
- **Server ownership.** The vLLM server lives in `E:\DGXSpark_Setup\vllm-qwen\startserver.sh` (`:8011`, `MAX_MODEL_LEN 32768`). This project is a thin async client; it reads `CONTEXT_SIZE` (= `MAX_MODEL_LEN`) for the guard and `/healthz`.
- **Decoding is frozen:** `temp 0.0 / top_p 0.5 / top_k 40` (greedy deterministic) — the fix is *prompt + max_tokens + async*, not sampling.
- **JSON in / JSON out.** vLLM can optionally enforce schema via `ENABLE_GUIDED_JSON=1` (xgrammar `guided_json`), but default is prompt + client-side `extract_json`/`_normalize_gds` with **fail-closed** `503` on unparseable output.

---

## Deploy Order

The vLLM server is **shared and long-lived**. It must be running **before** the gateway.

```bash
# (1) From vllm-qwen — start the model server ONCE (or reuse existing)
cd E:\DGXSpark_Setup\vllm-qwen
./startserver.sh          # vLLM MAIN on :8011 (takes ~3-15 min first boot, JIT)

# (2) From GDS-Extraction — start the gateway and connect to :8011
cd E:\Projects\GDS-Extraction
./start.sh                # FastAPI gateway on :8084
```

> **Pre-flight is strict.** `start.sh` waits up to 900s for `:8011` (`/health` + `/v1/models`). If not reachable, it **stops and tells you to start vllm-qwen first**. Legacy `:8006` still works if you set `MODEL_URL` in `.env` — the gateway auto-detects and falls back to the old 3-level reasoning chain.

> **Migrating from llama :8006?** Delete your old `.env` or update `MODEL_URL` to `http://127.0.0.1:8011/v1/chat/completions`, `MODEL_NAME` to `Qwen3.6-35B-A3B-NVFP4`, `CONTEXT_SIZE` to `32768`, `MODEL_PARALLEL` to `1`. `start.sh` warns if `.env` still points to `:8006`.

---

## Requirements

- **NVIDIA DGX Spark** (GB200 Grace Blackwell, unified 121.69 GiB, Linux, CUDA drivers).
- The vLLM server (`:8011`, `Qwen3.6-35B-A3B-NVFP4`, `--enable-prefix-caching`) **already running** — see Context Sizing.
- `python3`, venv (auto‑created by `start.sh` on first run).
- `curl` (for testing).

---

## Quick Start

> All steps run **on the DGX Spark** (Linux). Shell scripts are developed on Windows but executed on DGX.

### 1. Launch

```bash
cd E:\Projects\GDS-Extraction
./start.sh
```

`start.sh` bootstraps `.venv` + installs deps **once** (if absent), copies `.env.example` → `.env` **once** and prints a note to review, then probes vLLM `:8011`. If healthy, it starts the FastAPI gateway on **:8084**:

```
======== GDS Extraction is LIVE ========
  API:    http://0.0.0.0:8084
  Docs:   http://0.0.0.0:8084/docs
  Health: http://0.0.0.0:8084/healthz
  Model:  http://127.0.0.1:8011/v1/chat/completions (Qwen3.6-35B-A3B-NVFP4, vLLM MAX_MODEL_LEN=32768)
  Press Ctrl+C to stop the gateway only — the model server is left running.
========================================
```

### 2. Configure (optional)

```bash
nano .env
```

### 3. Test

```bash
curl -s -X POST http://localhost:8084/v1/extract \
  -H "x-api-key: gds_key_0000" -H "Content-Type: application/json" \
  -d '{"gds_text":"AN3SEPMNLNAN ** AMADEUS AVAILABILITY - AN **\nNANNADI.FJ 32 TU 03SEP 0000 1    PR 221   J5 C5 ..."}'
```

### 4. Stop

```bash
./stop.sh
```

Stops **only the gateway** (PID‑file + name fallback). vLLM is left untouched.

---

## API Reference

Base URL: `http://<dgx-spark>:8084`

### `POST /v1/extract`

Parse a single GDS input into the schedule JSON. **Requires `x-api-key`.**

**Request**

```json
{ "gds_text": "AN3SEPMNLNAN ** AMADEUS AVAILABILITY - AN **\nNANNADI.FJ 32 TU 03SEP…" }
```

The GDS text is placed into a delimited section in the prompt to reduce injection bleed.

**Success (`200`)** — one schedule object; see [Output Schema](#output-schema).

**`POST /v1/version`** — instant, never touches the model.

```json
{ "version": "1.1" }
```

**`POST /v1/extract_batch`** — parse many entries. Default `BATCH_CONCURRENCY=1` sequential; set `BATCH_CONCURRENCY=2-4` in `.env` for parallel async. Each entry isolated.

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
| `422` | **Over-budget prompt** — client-side guard rejected it *before network*. Body contains sizing guidance (`MAX_MODEL_LEN`, `max_tokens`, `prompt_room`) and how to raise `MAX_MODEL_LEN` via `vllm-qwen/startserver.sh`. |
| `503` | Model server unreachable / timed out / returned empty. |
| `503` | **Unparseable model output.** Extraction **fails closed** (never fabricates a schedule). |
| `503` | **Truncation** — `completion_tokens == max_tokens` hit the adaptive cap (log warns). Retry with larger `MODEL_MAX_TOKENS` or `MAX_MODEL_LEN`. |
| `500` | Unexpected pipeline error. |

**Examples**

```bash
# Single entry (Toby's Amadeus availability sample is in tests/cases/amadeus_availability_input.txt)
curl -s -X POST http://localhost:8084/v1/extract \
  -H "x-api-key: gds_key_0000" -H "Content-Type: application/json" \
  -d '{"gds_text":"AN3SEPMNLNAN ** AMADEUS AVAILABILITY - AN **\nNANNADI.FJ 32 TU 03SEP…"}'

# Batch
curl -s -X POST http://localhost:8084/v1/extract_batch \
  -H "x-api-key: gds_key_0000" -H "Content-Type: application/json" \
  -d '{"entries":[{"id":"r1","gds_text":"AN3SEP…"},{"id":"r2","gds_text":"AN3SEP…"}]}'

# Version (instant)
curl -s -X POST http://localhost:8084/v1/version
```

Interactive docs: **http://<dgx-spark>:8084/docs**.

### `GET /healthz`

Liveness probe; checks vLLM `/health` + `/v1/models` and reports the **vLLM-native budget** (`max_model_len`, `prompt_room`, `max_tokens_default`) and whether `model_name` matches `SERVED_MODEL_NAME`.

```bash
curl -s http://localhost:8084/healthz
# {"status":"healthy","model_server":"ready","model_name":"Qwen3.6-35B-A3B-NVFP4",
#  "vllm_model_id":"Qwen3.6-35B-A3B-NVFP4",
#  "context_budget":{"max_model_len":32768,"prompt_room":29440,"max_tokens_default":3072,"check":"match"},
#  "default_year_mode":"current-year","version":"1.1"}
```

- `context_budget.max_model_len` = `CONTEXT_SIZE` (= vLLM `MAX_MODEL_LEN`).
- `prompt_room` = `MAX_MODEL_LEN - MODEL_MAX_TOKENS - 256`.
- `check` = `match` / `mismatch` / `unknown`: whether `MODEL_NAME` matches vLLM's `id`. Legacy `:8006` also checks `n_ctx` slots.

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
- `+N` arrival-day offsets are resolved into the correct date, month, year, and day-of-week (month and year rollovers included). Times are 24-hour `HH:MM`.
- For availability displays: `Record type: "none"`, `Passenger Name: ["none"]`, `PNR: "none"`, and per-segment `segment_record_locator` / `service_class_letter` / `service_class` = `none`.

---

## Configuration Reference

All keys live in `.env` (see `.env.example`). The defaults here **are** the code defaults.

| Key | Default | Purpose |
|-----|---------|---------|
| `MODEL_URL` | `http://127.0.0.1:8011/v1/chat/completions` | Endpoint of the **running** vLLM server. Legacy `:8006` still works via fallback. |
| `MODEL_NAME` | `Qwen3.6-35B-A3B-NVFP4` | Reported to the server (must match vLLM `SERVED_MODEL_NAME`). |
| `LLAMA_SERVER_API_KEY` | `""` | Bearer token the gateway presents (sent only if non‑empty). vLLM default is empty. |
| `CONTEXT_SIZE` | `32768` | **vLLM** `MAX_MODEL_LEN`. Must match `vllm-qwen/startserver.sh` `--max-model-len`. |
| `MODEL_PARALLEL` | `1` | No slots on vLLM (was `4` on llama). Keep `1`. |
| `MODEL_MAX_TOKENS` | `3072` | Baseline max output tokens. **Adaptive per-request**: `1200+280*segs+0.6*inp` clamped `1024-4096`. Prevents the old 8192 runaway (7k tokens → 103s). |
| `MODEL_TEMP` / `MODEL_TOP_P` / `MODEL_TOP_K` | `0.0` / `0.5` / `40` | **FROZEN** — greedy deterministic extraction, validated on-DGX. Do not change. |
| `DISABLE_THINKING` | `1` | Sends `chat_template_kwargs:{enable_thinking:false}` to vLLM (Qwen3.6 needs this). Legacy `:8006` also sends `reasoning_effort:0`. |
| `ENABLE_GUIDED_JSON` | `0` | When `1`, sends `guided_json` schema via `extra_body` to vLLM (xgrammar). Experimental — off by default. |
| `BATCH_CONCURRENCY` | `1` | `1`=sequential, `2-4`=parallel async via `asyncio.gather`. |
| `ENABLE_STREAMING` | `0` | `1`=use `stream=True` for TTFT instrumentation. Off by default (stable). |
| `CONTEXT_GUARD` | `strict` | `strict` (422 + guidance), `warn` (allow + log), or `off` (testing). |
| `REQUEST_TIMEOUT` | `120` | Timeout to vLLM, seconds. Surfaces slow 100s+ regressions faster than old 300. |
| `DEFAULT_YEAR` | `""` (empty → current year at request time) | Year used when the GDS line omits one. Pin, e.g. `DEFAULT_YEAR="2025"`. |
| `API_PORT` / `API_HOST` | `8084` / `0.0.0.0` | Gateway bind. Not `8011` (vLLM). |
| `API_KEY_AUTH_HEADER` | `x-api-key` | Header name for client auth. |
| `API_KEYS` | `gds_key_0000:GDS Extraction Local Testing` | Comma‑separated valid client keys. |

**Port map**

| Port | Owner | Managed by |
|------|-------|-----------|
| `8011` | vLLM `Qwen3.6-35B-A3B-NVFP4` (NVFP4/FP8) | `E:\DGXSpark_Setup\vllm-qwen\startserver.sh` |
| `8084` | **GDS‑Extraction gateway** | `start.sh` (this project) |
| `8006` | Legacy `llama-server` (deprecated, still supported as fallback) | `Proof-Reader/startserver.sh` |

---

## Context Sizing (important)

vLLM uses **continuous batching**, not llama's `--parallel` slots. The budget is simply `MAX_MODEL_LEN`.

**Prompt-room budget** = `MAX_MODEL_LEN − MODEL_MAX_TOKENS − 256` safety margin. With defaults `32768 - 3072 - 256 = **29440 tokens**` (~88k chars GDS) — **3.7× the old 7936 slot** — ample for any availability display; guard rarely trips. Old `422` rate-limit behavior preserved.

**Adaptive cap** (new in v1.1): each request computes `max_tokens = clamp(1200 + 280*estimated_segments + 0.6*input_tokens, 1024, 4096)`. Examples: Toby `2-seg` → `~1800` (≈26s worst at 68 tok/s vs old 8192 → 120s), golden `10-seg` → `~4000` (≈58s worst vs 120s). If `completion_tokens == max_tokens` in logs, the cap was hit (possible truncation — the gateway logs a warning).

**Sizing matrix**

| Profile | `CONTEXT_SIZE` (=MAX_MODEL_LEN) | `MODEL_PARALLEL` | Prompt room (3072 default) | When |
|---------|-------------------------------|------------------|----------------------------|------|
| **vLLM default** | `32768` | `1` | **29440** | Covers all GDS cases, 7 gateways + coding concurrently |
| Large | `65536` | `1` | `62208` | If you raise `MAX_MODEL_LEN` via `vllm-qwen/startserver.sh` |
| Legacy llama | `65536` | `4` | `7936` (old slot math) | Only when `MODEL_URL` points to `:8006` |

To allow larger prompts, raise vLLM's window:

```bash
# 1. Raise vLLM's --max-model-len (requires restart + GPU headroom)
cd E:\DGXSpark_Setup\vllm-qwen
# edit .env: MAX_MODEL_LEN=65536
./stop.sh && ./startserver.sh

# 2. Set SAME value in this repo's .env
#    CONTEXT_SIZE="65536"
#    MODEL_PARALLEL="1"

# 3. Restart gateway
cd E:\Projects\GDS-Extraction
./stop.sh && ./start.sh

# 4. Verify /healthz reports max_model_len: match
curl -s http://localhost:8084/healthz | python3 -m json.tool
```

---

## Local Development & Testing

The core logic (`build_prompt`, `build_params`, `estimate_tokens`, `check_context`, `extract_json`,
`_normalize_gds`, `run_extract`, `run_extract_batch`, `resolve_max_tokens`, `estimate_segments`) is **pure** and takes an injectable model call — so the entire business logic is unit‑tested **on the Windows dev machine with a stub backend, no GPU needed.**

```bash
pip install -r requirements.txt   # includes pytest + httpx
pytest -v
```

The suite (`tests/test_extract.py`, 67 cases) covers prompt shape (sentinel + delimiters + all rule blocks), `GDS_SYSTEM` token budget (`<1100`), adaptive `max_tokens`, thinking suppression (vLLM vs legacy `:8006` paths), context guard (`max_model_len` + legacy slots), JSON extraction (think-block/fence/preamble stripping + fail-closed), schema coercion, orchestration, guard-trip before model, batch isolation, `resolve_default_year`, HTTP layer (auth `401`, validation `422`, over-budget `422`, unavailable `503`, unparseable `503`), content-resolution (`reasoning_content` leak), `/healthz` vLLM budget cross-check, golden-case plumbing, and config-drift pinning `.env.example` to code defaults. **All 67 pass locally.**

The Amadeus availability sample (input + expected 10-segment output) lives in `tests/cases/` for on‑DGX smoke‑testing. See `implementation.md` §2 Phase F for DGX verifications (Toby `<60s`, golden `<60s`, byte-identical ×3, prefix-cache, batch concurrency, `benchmark_load.py` coexistence).

---

## Troubleshooting

**`422` over-budget prompt.**
Client-side guard rejected the request before any network call. Body now says `MAX_MODEL_LEN=32768 ... prompt_room = MAX_MODEL_LEN - max_tokens - 256` + the adaptive cap for that request. Shorten GDS, or raise vLLM's `--max-model-len` and set `CONTEXT_SIZE` to match (see Context Sizing). Legacy `:8006` still reports slot math.

**`/healthz` reports `mismatch` under `context_budget.check`.**
`MODEL_NAME` (`Qwen3.6-35B-A3B-NVFP4`) doesn't match vLLM's `id` from `/v1/models`. Check `SERVED_MODEL_NAME` in `vllm-qwen/.env` / `MODEL_NAME` in this repo's `.env`.

**`/healthz` reports `degraded` / `unreachable`.**
Gateway couldn't reach vLLM `:8011`. Wait for model to be up. Check `nvidia-smi`, `curl http://127.0.0.1:8011/health`, `curl http://127.0.0.1:8011/v1/models`, and `MODEL_URL` in `.env`. `start.sh` waits up to 900s (vLLM JIT) — run `E:\DGXSpark_Setup\vllm-qwen\startserver.sh` first, then `./start.sh`. Legacy fallback: if `MODEL_URL` is `:8006`, check llama-server instead.

**`503` — "could not find a JSON object" / unparseable.**
Fail-closed (never fabricates). Try again; if persists, confirm `DISABLE_THINKING=1` and that input isn't truncated. With `ENABLE_GUIDED_JSON=1`, schema enforcement should eliminate this — try toggling it. Check gateway log for `thinking markers found` warnings.

**`503` — "Model server unavailable."**
vLLM not reachable / timeout / hit `max_tokens` cap. Confirm `curl -v http://127.0.0.1:8011/health`, check gateway log for `hit max_tokens cap` (means adaptive cap was too low — raise `MODEL_MAX_TOKENS` or `MAX_MODEL_LEN`), and that `LLAMA_SERVER_API_KEY` matches if vLLM has `API_KEY` set (default empty).

**Latency still ~100s (no improvement).**
Check gateway log: `max_tokens` should be `~1800` for 2-seg, not `8192`; `completion_tokens` should be `~600-900`, not `7000`; `thinking_leaked` should be `false`. If `completion_tokens ≈ max_tokens`, you hit the cap (truncation). If `reasoning_content` leaked (`thinking markers found`), `DISABLE_THINKING` wasn't honored — verify vLLM version honors `chat_template_kwargs` and that `MODEL_URL` is `:8011` (not `:8006`). Also check vLLM log: `gen ~68 tok/s` is healthy; `prefix cache hit rate` should go `>0` on second identical request (if it stays `0.0%` after the fix, file a `vllm-qwen` issue — not gateway-blocked).

**The model emits reasoning/"thinking" text.**
Confirm `DISABLE_THINKING=1` and `MODEL_URL` points to `:8011`. Gateway logs `thinking trace leaked despite suppression (reasoning_len=...)` when vLLM returns `reasoning_content` despite the flag — this now counts toward generation time, so the fix surfaces it. `extract_json()` strips it defensively, but you should still see the warning.

**A batch entry is broken but others succeed.**
Expected. Each entry isolated (`status: "error"`); good entries never lost. See batch endpoint. With `BATCH_CONCURRENCY>1`, overall wall time drops (vLLM batches them).

**API key rejected.**
Confirm key in `API_KEYS` (`.env`) matches `x-api-key` you send.

---

## Files

| File | Purpose |
|------|---------|
| `gds_extraction_service.py` | Single‑file FastAPI app: `.env` config (vLLM-native), `GDS_SYSTEM` (compressed, <1100 toks), `resolve_max_tokens`/`estimate_segments`, `build_prompt`/`build_params` (vLLM-native, decoding frozen), `estimate_tokens`/`context_budget`/`check_context`, `extract_json`/`_normalize_gds`, `run_extract`/`run_extract_batch`, async `http_model_call`/`http_model_call_async` (httpx, guided JSON, usage/TTFT logging), routes + auth, `/healthz` vLLM budget. |
| `start.sh` | One‑entry launch: bootstrap `.venv`/`.env`, pre-flight probe `:8011` (`/health` + `/v1/models`, 900s), launch gateway only. Warns if `.env` still points to `:8006`. |
| `stop.sh` | Graceful shutdown of **gateway only** (never vLLM). |
| `.env.example` | Config template (vLLM defaults: `8011 / Qwen3.6-35B-A3B-NVFP4 / 32768/1 / 3072`). **Must match code defaults** (enforced by test). |
| `requirements.txt` | Runtime deps (`fastapi`, `uvicorn`, `requests`, `httpx`, `python-dotenv`) + dev/test (`pytest`). |
| `tests/test_extract.py` | pytest suite (67 cases, stub backend, Windows no-GPU). |
| `tests/cases/amadeus_availability_input.txt` | Toby's Amadeus availability sample input. |
| `tests/cases/amadeus_availability_expected.json` | Expected 10-segment output — drives golden-case plumbing test. |
| `implementation.md` | Design + roadmap (vLLM Performance Fix, §1.2 root cause with 103s trace, §1.3 mermaid, sizing matrix). |
| `gds-extraction-prompts.md` | Verbatim prompt reference (points to compressed `GDS_SYSTEM` lines). |
