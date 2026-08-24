# GDS Extraction — Model & Prompts Reference

---


## 1. System Prompt (verbatim)

This is `GDS_SYSTEM` from `gds_extraction_service.py` (~lines 169–244), sent character-for-character.
Replace `__DEFAULT_YEAR__` with the desired default year (service default: current year at request time).

```text
You are a meticulous Global Distribution System (GDS) flight-data extractor.
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
```

> **Quirk worth knowing:** the doubled braces (`{{`, `}}`) are sent *literally* — the constant
> is processed with a plain string replace (for `__DEFAULT_YEAR__`), never `.format()`.
> Harmless in practice (output JSON comes back correctly), but you'll notice it if you
> diff the rendered prompt.

---

## 2. User Prompt (verbatim)

From `build_prompt()` (~line 250):

```text
Parse the following GDS output. Output ONLY the required JSON object.

GDS_DATA:
<<<GDS_DATA>>>
<gds_text>
<<<END_GDS_DATA>>>
```

`<gds_text>` is the raw GDS output, inserted as-is between the delimiters.

---

## 3. Reproducing a Call

```json
{
  "model": "Qwen3.8-27B",
  "messages": [
    { "role": "system", "content": "<Section 3, with __DEFAULT_YEAR__ substituted>" },
    { "role": "user",   "content": "<Section 4, with gds_text inserted>" }
  ],
  "temperature": 0.0,
  "top_p": 0.5,
  "top_k": 40,
  "min_p": 0.0,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "max_tokens": 8192,
  "reasoning_effort": 0,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

Post-processing applied by the service (replicate if comparing outputs):

1. Strip any leaked thinking/reasoning text.
2. Locate the JSON span in the reply and parse it (`extract_json()`).
3. Schema-coerce/normalize fields (`_normalize_gds`).

A ready-made input/expected pair lives in `tests/cases/amadeus_availability_input.txt` /
`amadeus_availability_expected.json` — good smoke-test material.

---

## 4. Testing Tips & Caveats

- **Keep temp at 0.0** for comparisons — anything higher reintroduces run-to-run variance.
- The prompt is tuned tightly to our schema and current sample data. New GDS formats may
  need *rule* tweaks in the system prompt rather than sampling changes.
- Airport/airline codes are instructed to stay verbatim; only names are translated. If you
  see mutated codes, that's a regression worth flagging.
- If a request overflows a slot budget (16k/slot), the client-side guard rejects it before
  hitting the network — trim the input or raise context.

---

## 5. Model & Serving Setup

| Item | Value |
|---|---|
| Model | **Qwen3.8-27B** (adaptive "thinking" model) |
| Server | llama.cpp `llama-server`, CUDA, on the shared NVIDIA DGX Spark |
| Endpoint | OpenAI-compatible `POST /v1/chat/completions` (configured via `MODEL_URL`) |
| Context | 65,536 total ÷ 4 parallel slots = **16,384 tokens/slot** |

### Sampling Parameters

| Parameter | Value | Why |
|---|---|---|
| `temperature` | **0.0** | Greedy decoding → deterministic extraction. Repeated runs on the same manifest are byte-identical (validated). |
| `top_p` | 0.5 | Tight nucleus; irrelevant at temp 0 but kept consistent. |
| `top_k` | 40 | Same. |
| `max_tokens` | 8192 | Headroom for large manifests (~10 segments ≈ 4k+ output tokens) so JSON never truncates mid-object. |
| `min_p` | 0.0 | Neutral. |
| `repetition_penalty` | 1.0 | Neutral — don't distort repeated field names/JSON. |
| `presence_penalty` | 0.0 | Neutral. |

### Thinking Suppression

Qwen3.8-27B is a thinking model; for extraction we want raw answers, not reasoning.
Two suppression fields are sent (best-effort, with an automatic degradation chain):

```json
"reasoning_effort": 0,
"chat_template_kwargs": { "enable_thinking": false }
```

If the server build rejects either field, the caller retries without it (then without both)
and caches whichever level worked. Any leaked thinking text is also stripped from the
output defensively before JSON parsing.

---
