# GDS Office Benchmark — CX FDJ3BN Ignacio (MNL-HKG-JFK-HKG-MNL) — RAMP [1, 5, 10, 15, 25] = 56 requests

- **Run at:** 2026-08-27T13:44:37.225219+00:00
- **Gateway:** `http://192.168.1.65:8084/v1/extract` (api_key: `gds_***`)
- **Payload:** CX FDJ3BN Ignacio — 2 pax, 4 segs (MNL->HKG->JFK->HKG->MNL)
- **Mode:** ramp — steps [1, 5, 10, 15, 25] with interval 8.0s, stagger 0.4s
- **Timeout:** 600.0s
- **Decoding (frozen):** temp 0.0 / top_p 0.5 / top_k 40 (greedy deterministic)
- **Gate correctness ground truth:** PNR=FDJ3BN, names=['IGNACIO/CYNTHIA', 'IGNACIO/MANUEL'], segs=4

## Summary

- **Wall time:** 332.1s  ·  **Throughput:** 0.17 req/s
- **Results:** 56 ok · 0 mismatch (HTTP 200 but wrong decode) · 0 error/timeout · 56 total
- **Correctness rate:** 100.0% (56/56 of parseable responses)
- **Latency (all):** p50 66613.4ms · p95 97672.1ms · p99 98189.4ms · min 20058.5ms · max 98239.6ms
- **Latency (ok only):** p50 66613.4ms · p95 97672.1ms · mean 75271.8ms

### Per-step breakdown (ramp)

| step | concurrency | ok | mismatch | error | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 0 | 20058.5 | 20058.5 | 20058.5 |
| 2 | 5 | 5 | 0 | 0 | 41075.7 | 41970.0 | 42080.3 |
| 3 | 10 | 10 | 0 | 0 | 64742.7 | 65849.7 | 66486.4 |
| 4 | 15 | 15 | 0 | 0 | 64913.0 | 66970.3 | 66996.1 |
| 5 | 25 | 25 | 0 | 0 | 95022.3 | 98132.7 | 98239.6 |

## Per-request latency (sorted)

| id | step | wave | status | latency (ms) | http | PNR | segs | correct |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1 | 1 | ok | 20058.5 | 200 | FDJ3BN | 4 | True |
| 2004 | 2 | 2 | ok | 40321.9 | 200 | FDJ3BN | 4 | True |
| 2003 | 2 | 2 | ok | 40603.5 | 200 | FDJ3BN | 4 | True |
| 2001 | 2 | 2 | ok | 41075.7 | 200 | FDJ3BN | 4 | True |
| 2002 | 2 | 2 | ok | 41528.7 | 200 | FDJ3BN | 4 | True |
| 2000 | 2 | 2 | ok | 42080.3 | 200 | FDJ3BN | 4 | True |
| 4014 | 4 | 4 | ok | 62482.0 | 200 | FDJ3BN | 4 | True |
| 4012 | 4 | 4 | ok | 63062.9 | 200 | FDJ3BN | 4 | True |
| 4011 | 4 | 4 | ok | 63394.4 | 200 | FDJ3BN | 4 | True |
| 4013 | 4 | 4 | ok | 63478.0 | 200 | FDJ3BN | 4 | True |
| 3008 | 3 | 3 | ok | 63502.8 | 200 | FDJ3BN | 4 | True |
| 4010 | 4 | 4 | ok | 63658.0 | 200 | FDJ3BN | 4 | True |
| 3007 | 3 | 3 | ok | 63805.2 | 200 | FDJ3BN | 4 | True |
| 3009 | 3 | 3 | ok | 63806.9 | 200 | FDJ3BN | 4 | True |
| 4009 | 4 | 4 | ok | 63864.0 | 200 | FDJ3BN | 4 | True |
| 4008 | 4 | 4 | ok | 64125.4 | 200 | FDJ3BN | 4 | True |
| 3005 | 3 | 3 | ok | 64387.8 | 200 | FDJ3BN | 4 | True |
| 3004 | 3 | 3 | ok | 64655.9 | 200 | FDJ3BN | 4 | True |
| 3006 | 3 | 3 | ok | 64829.4 | 200 | FDJ3BN | 4 | True |
| 3003 | 3 | 3 | ok | 64849.2 | 200 | FDJ3BN | 4 | True |
| 4004 | 4 | 4 | ok | 64913.0 | 200 | FDJ3BN | 4 | True |
| 3002 | 3 | 3 | ok | 65055.3 | 200 | FDJ3BN | 4 | True |
| 3001 | 3 | 3 | ok | 65071.5 | 200 | FDJ3BN | 4 | True |
| 4001 | 4 | 4 | ok | 65262.6 | 200 | FDJ3BN | 4 | True |
| 4007 | 4 | 4 | ok | 65591.7 | 200 | FDJ3BN | 4 | True |
| 4006 | 4 | 4 | ok | 65912.4 | 200 | FDJ3BN | 4 | True |
| 4005 | 4 | 4 | ok | 66192.7 | 200 | FDJ3BN | 4 | True |
| 3000 | 3 | 3 | ok | 66486.4 | 200 | FDJ3BN | 4 | True |
| 4003 | 4 | 4 | ok | 66740.4 | 200 | FDJ3BN | 4 | True |
| 4002 | 4 | 4 | ok | 66959.3 | 200 | FDJ3BN | 4 | True |
| 4000 | 4 | 4 | ok | 66996.1 | 200 | FDJ3BN | 4 | True |
| 5023 | 5 | 5 | ok | 91466.5 | 200 | FDJ3BN | 4 | True |
| 5024 | 5 | 5 | ok | 91961.3 | 200 | FDJ3BN | 4 | True |
| 5021 | 5 | 5 | ok | 92104.0 | 200 | FDJ3BN | 4 | True |
| 5022 | 5 | 5 | ok | 92688.0 | 200 | FDJ3BN | 4 | True |
| 5019 | 5 | 5 | ok | 92725.0 | 200 | FDJ3BN | 4 | True |
| 5018 | 5 | 5 | ok | 93037.5 | 200 | FDJ3BN | 4 | True |
| 5017 | 5 | 5 | ok | 93320.6 | 200 | FDJ3BN | 4 | True |
| 5020 | 5 | 5 | ok | 93394.6 | 200 | FDJ3BN | 4 | True |
| 5016 | 5 | 5 | ok | 93560.6 | 200 | FDJ3BN | 4 | True |
| 5015 | 5 | 5 | ok | 93829.4 | 200 | FDJ3BN | 4 | True |
| 5013 | 5 | 5 | ok | 94325.0 | 200 | FDJ3BN | 4 | True |
| 5012 | 5 | 5 | ok | 94572.3 | 200 | FDJ3BN | 4 | True |
| 5010 | 5 | 5 | ok | 95022.3 | 200 | FDJ3BN | 4 | True |
| 5009 | 5 | 5 | ok | 95247.4 | 200 | FDJ3BN | 4 | True |
| 5014 | 5 | 5 | ok | 95394.4 | 200 | FDJ3BN | 4 | True |
| 5008 | 5 | 5 | ok | 95406.1 | 200 | FDJ3BN | 4 | True |
| 5006 | 5 | 5 | ok | 95787.9 | 200 | FDJ3BN | 4 | True |
| 5005 | 5 | 5 | ok | 95888.9 | 200 | FDJ3BN | 4 | True |
| 5001 | 5 | 5 | ok | 96113.2 | 200 | FDJ3BN | 4 | True |
| 5002 | 5 | 5 | ok | 96253.0 | 200 | FDJ3BN | 4 | True |
| 5011 | 5 | 5 | ok | 96374.6 | 200 | FDJ3BN | 4 | True |
| 5007 | 5 | 5 | ok | 97539.4 | 200 | FDJ3BN | 4 | True |
| 5000 | 5 | 5 | ok | 98070.2 | 200 | FDJ3BN | 4 | True |
| 5004 | 5 | 5 | ok | 98148.3 | 200 | FDJ3BN | 4 | True |
| 5003 | 5 | 5 | ok | 98239.6 | 200 | FDJ3BN | 4 | True |

## Health snapshots

```json
{
  "gateway_health": {
    "status": 200,
    "body": {
      "status": "healthy",
      "model_server": "ready",
      "model_name": "Qwen3.6-35B-A3B-NVFP4",
      "vllm_model_id": null,
      "context_budget": {
        "max_model_len": 32768,
        "slot_tokens": 32768,
        "prompt_room": 29440,
        "max_tokens_default": 3072,
        "server_ctx_size": null,
        "check": "unknown"
      },
      "default_year_mode": "pinned=2025",
      "version": "1.1",
      "guided_json": false,
      "thinking_disabled": true
    }
  },
  "vllm_health": {
    "url": "http://192.168.1.65:8011/health",
    "status": 200
  }
}
```

After:
```json
{
  "gateway_health": {
    "status": 200,
    "body": {
      "status": "healthy",
      "model_server": "ready",
      "model_name": "Qwen3.6-35B-A3B-NVFP4",
      "vllm_model_id": null,
      "context_budget": {
        "max_model_len": 32768,
        "slot_tokens": 32768,
        "prompt_room": 29440,
        "max_tokens_default": 3072,
        "server_ctx_size": null,
        "check": "unknown"
      },
      "default_year_mode": "pinned=2025",
      "version": "1.1",
      "guided_json": false,
      "thinking_disabled": true
    }
  },
  "vllm_health": {
    "url": "http://192.168.1.65:8011/health",
    "status": 200
  }
}
```

## Notes

- **Ramp mode** simulates office-realistic load: 1 user arrives, then 5, then 10… — find the knee where p95 degrades. `burst` is the peak shock test.
- `ramp-delay` staggers arrivals inside each step (e.g. 0.4s -> 25 users trickle over ~10s).
- `ramp-interval` is the pause between steps to let vLLM drain.
- `mismatch` = HTTP 200 but validator failed (decoder bug). `error`/`timeout` = non-200 or timeout.
- Correctness ground truth is from boss spec Pages 6-12; adapt `EXPECTED` if your PNR changes.
