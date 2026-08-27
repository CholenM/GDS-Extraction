# GDS Extraction — Model & Prompts Reference (v1.1 — vLLM Performance Fix)

---

## 1. System Prompt (verbatim, compressed)

This is `GDS_SYSTEM` from `gds_extraction_service.py` (now compressed <1100 toks, v1.1). Sent character-for-character with `__DEFAULT_YEAR__` replaced. Decoding frozen (`temp 0`).

```text
You are a meticulous Global Distribution System (GDS) flight-data extractor.
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
```

> Compression note (v1.1): all 11 rule blocks from v1.0 kept verbatim-faithful — only wording tightened to save ~300 tokens for prefix-cache efficiency. Tests enforce required substrings: `APDI`, `sacrosanct`, `Philippine Airlines`, `Business = C,D,I,J,Z`, `B is Economy`, `DCPR`, `FJ:QF3873`, `+N`.

---

## 2. User Prompt (verbatim)

From `build_prompt()`:

```text
Parse the following GDS output. Output ONLY the required JSON object.

GDS_DATA:
<<<GDS_DATA>>>
<gds_text>
<<<END_GDS_DATA>>>
```

`<gds_text>` is the raw GDS output, inserted as-is between the delimiters.

---

## 3. Reproducing a Call (vLLM :8011)

```json
{
  "model": "Qwen3.6-35B-A3B-NVFP4",
  "messages": [
    { "role": "system", "content": "<Section 1, with __DEFAULT_YEAR__ substituted>" },
    { "role": "user",   "content": "<Section 2, with gds_text inserted>" }
  ],
  "temperature": 0.0,
  "top_p": 0.5,
  "top_k": 40,
  "min_p": 0.0,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "max_tokens": 1800,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

`max_tokens` is **adaptive** in the gateway: `1200 + 280*estimated_segments + 0.6*input_tokens` clamped `1024-4096` (`~1800` for Toby 2-seg, `~4000` for golden 10-seg). `reasoning_effort` is **not** sent on vLLM (was llama-only). When `ENABLE_GUIDED_JSON=1`, adds `extra_body: {guided_json: <§1.5 schema>}` + `response_format: {type:"json_object"}`.

Post-processing (replicate if comparing outputs):

1. Strip any leaked thinking/reasoning text (`<think>` / `reasoning_content`).
2. Locate JSON span and parse (`extract_json()`).
3. Schema-coerce/normalize (`_normalize_gds`).

A ready-made input/expected pair lives in `tests/cases/amadeus_availability_input.txt` / `amadeus_availability_expected.json`.

---

## 4. Testing Tips & Caveats

- **Keep temp at 0.0** — decoding is frozen for determinism; don't change sampling to chase speed.
- Prompt is compressed but rules are locked — new GDS formats need *rule* tweaks, not sampling changes.
- Airport/airline codes are `sacrosanct` — mutated codes are regressions.
- Guard now uses vLLM `MAX_MODEL_LEN` (32768) → `prompt_room = 32768 - max_tokens - 256` (~29440 default), not llama slots. Legacy `:8006` slot math only applies when `MODEL_URL` contains `8006`.
- If you still see 100s latency: check gateway log `completion_tokens` (≈600-900 for 2-seg, not 7000) and `thinking_leaked` flag, plus vLLM log `gen tok/s` and `prefix hit rate`.

---

## 5. Model & Serving Setup

| Item | Value |
|---|---|
| Model | **Qwen3.6-35B-A3B-NVFP4** (MoE 35B/3B active, reasoning) |
| Server | vLLM MAIN, `E:\DGXSpark_Setup\vllm-qwen`, `:8011`, `--enable-prefix-caching`, `--reasoning-parser qwen3` |
| Endpoint | `POST /v1/chat/completions` (via `MODEL_URL`) |
| Context | `MAX_MODEL_LEN 32768` (no slots; `MODEL_PARALLEL=1`) · `prompt_room 29440` at 3072 default |
| Prompt tokens | System ~1014 (compressed from ~1300) · Adaptive max_tokens 1024-4096 |

### Sampling Parameters (FROZEN)

| Parameter | Value | Why |
|---|---|---|
| `temperature` | **0.0** | Greedy → deterministic, byte-identical repeats (validated) |
| `top_p` | 0.5 | Tight nucleus (irrelevant at temp 0, kept) |
| `top_k` | 40 | Same |
| `max_tokens` | **adaptive 1024-4096** (baseline 3072) | Was 8192 (caused 7k-token runaway → 103s); adaptive prevents runaway while covering 10-seg (≈4000) |
| `min_p` | 0.0 | Neutral |
| `repetition_penalty` | 1.0 | Neutral |
| `presence_penalty` | 0.0 | Neutral |

### Thinking Suppression (vLLM-native)

Qwen3.6 is a reasoning MoE. Gateway sends only:

```json
"chat_template_kwargs": { "enable_thinking": false }
```

No `reasoning_effort` on vLLM (was llama-only; sending it wastes a retry). Server's `--reasoning-parser qwen3` splits any leaked trace into `reasoning_content` — gateway logs `thinking trace leaked` and strips it defensively before JSON parse. Legacy `:8006` path still uses the 3-level `reasoning_effort` + `chat_template_kwargs` chain for compat.

---
