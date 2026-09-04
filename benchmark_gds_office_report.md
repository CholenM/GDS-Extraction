# GDS Office Benchmark — CX FDJ3BN Ignacio (MNL-HKG-JFK-HKG-MNL) — RAMP [1, 5, 10, 15, 25] = 56 requests

- **Run at:** 2026-08-28T17:06:13.833492+00:00
- **Gateway:** `http://192.168.1.65:8084/v1/extract` (api_key: `gds_***`)
- **Payload:** CX FDJ3BN Ignacio — 2 pax, 4 segs (MNL->HKG->JFK->HKG->MNL)
- **Mode:** ramp — steps [1, 5, 10, 15, 25] with interval 8.0s, stagger 0.4s
- **Timeout:** 600.0s
- **Decoding (frozen):** temp 0.0 / top_p 0.5 / top_k 40 (greedy deterministic)
- **Gate correctness ground truth:** PNR=FDJ3BN, names=['IGNACIO/CYNTHIA', 'IGNACIO/MANUEL'], segs=4

## Summary

- **Wall time:** 343.03s  ·  **Throughput:** 0.16 req/s
- **Results:** 56 ok · 0 mismatch (HTTP 200 but wrong decode) · 0 error/timeout · 56 total
- **Correctness rate:** 100.0% (56/56 of parseable responses)
- **Latency (all):** p50 67783.8ms · p95 95663.3ms · p99 95723.2ms · min 21425.1ms · max 95746.9ms
- **Latency (ok only):** p50 67783.8ms · p95 95663.3ms · mean 76474.8ms

### Per-step breakdown (ramp)

| step | concurrency | ok | mismatch | error | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 0 | 21425.1 | 21425.1 | 21425.1 |
| 2 | 5 | 5 | 0 | 0 | 42404.4 | 42511.8 | 42522.2 |
| 3 | 10 | 10 | 0 | 0 | 67067.9 | 67320.8 | 67344.5 |
| 4 | 15 | 15 | 0 | 0 | 67371.3 | 67870.2 | 67926.9 |
| 5 | 25 | 25 | 0 | 0 | 95182.7 | 95699.6 | 95746.9 |

## Per-request latency (sorted)

| id | step | wave | status | latency (ms) | http | PNR | segs | correct |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1 | 1 | ok | 21425.1 | 200 | FDJ3BN | 4 | True |
| 2000 | 2 | 2 | ok | 42214.7 | 200 | FDJ3BN | 4 | True |
| 2004 | 2 | 2 | ok | 42219.8 | 200 | FDJ3BN | 4 | True |
| 2003 | 2 | 2 | ok | 42404.4 | 200 | FDJ3BN | 4 | True |
| 2001 | 2 | 2 | ok | 42470.4 | 200 | FDJ3BN | 4 | True |
| 2002 | 2 | 2 | ok | 42522.2 | 200 | FDJ3BN | 4 | True |
| 4000 | 4 | 4 | ok | 65617.0 | 200 | FDJ3BN | 4 | True |
| 3000 | 3 | 3 | ok | 66062.4 | 200 | FDJ3BN | 4 | True |
| 4001 | 4 | 4 | ok | 66532.7 | 200 | FDJ3BN | 4 | True |
| 3009 | 3 | 3 | ok | 66550.8 | 200 | FDJ3BN | 4 | True |
| 4014 | 4 | 4 | ok | 66608.7 | 200 | FDJ3BN | 4 | True |
| 3008 | 3 | 3 | ok | 66801.6 | 200 | FDJ3BN | 4 | True |
| 3001 | 3 | 3 | ok | 66865.2 | 200 | FDJ3BN | 4 | True |
| 4013 | 4 | 4 | ok | 66870.6 | 200 | FDJ3BN | 4 | True |
| 3007 | 3 | 3 | ok | 67015.2 | 200 | FDJ3BN | 4 | True |
| 4002 | 4 | 4 | ok | 67058.9 | 200 | FDJ3BN | 4 | True |
| 3002 | 3 | 3 | ok | 67120.6 | 200 | FDJ3BN | 4 | True |
| 4012 | 4 | 4 | ok | 67123.5 | 200 | FDJ3BN | 4 | True |
| 3006 | 3 | 3 | ok | 67174.4 | 200 | FDJ3BN | 4 | True |
| 3003 | 3 | 3 | ok | 67253.4 | 200 | FDJ3BN | 4 | True |
| 3005 | 3 | 3 | ok | 67291.9 | 200 | FDJ3BN | 4 | True |
| 4011 | 4 | 4 | ok | 67333.0 | 200 | FDJ3BN | 4 | True |
| 3004 | 3 | 3 | ok | 67344.5 | 200 | FDJ3BN | 4 | True |
| 4003 | 4 | 4 | ok | 67371.3 | 200 | FDJ3BN | 4 | True |
| 4010 | 4 | 4 | ok | 67532.0 | 200 | FDJ3BN | 4 | True |
| 4004 | 4 | 4 | ok | 67636.4 | 200 | FDJ3BN | 4 | True |
| 4009 | 4 | 4 | ok | 67723.1 | 200 | FDJ3BN | 4 | True |
| 4005 | 4 | 4 | ok | 67737.8 | 200 | FDJ3BN | 4 | True |
| 4008 | 4 | 4 | ok | 67829.8 | 200 | FDJ3BN | 4 | True |
| 4007 | 4 | 4 | ok | 67845.9 | 200 | FDJ3BN | 4 | True |
| 4006 | 4 | 4 | ok | 67926.9 | 200 | FDJ3BN | 4 | True |
| 5000 | 5 | 5 | ok | 92057.5 | 200 | FDJ3BN | 4 | True |
| 5001 | 5 | 5 | ok | 93044.5 | 200 | FDJ3BN | 4 | True |
| 5024 | 5 | 5 | ok | 93729.6 | 200 | FDJ3BN | 4 | True |
| 5002 | 5 | 5 | ok | 93737.7 | 200 | FDJ3BN | 4 | True |
| 5023 | 5 | 5 | ok | 94054.8 | 200 | FDJ3BN | 4 | True |
| 5003 | 5 | 5 | ok | 94262.1 | 200 | FDJ3BN | 4 | True |
| 5022 | 5 | 5 | ok | 94321.7 | 200 | FDJ3BN | 4 | True |
| 5021 | 5 | 5 | ok | 94597.6 | 200 | FDJ3BN | 4 | True |
| 5004 | 5 | 5 | ok | 94620.0 | 200 | FDJ3BN | 4 | True |
| 5020 | 5 | 5 | ok | 94801.6 | 200 | FDJ3BN | 4 | True |
| 5005 | 5 | 5 | ok | 94903.4 | 200 | FDJ3BN | 4 | True |
| 5019 | 5 | 5 | ok | 94994.1 | 200 | FDJ3BN | 4 | True |
| 5006 | 5 | 5 | ok | 95182.7 | 200 | FDJ3BN | 4 | True |
| 5018 | 5 | 5 | ok | 95188.4 | 200 | FDJ3BN | 4 | True |
| 5017 | 5 | 5 | ok | 95386.7 | 200 | FDJ3BN | 4 | True |
| 5007 | 5 | 5 | ok | 95438.4 | 200 | FDJ3BN | 4 | True |
| 5015 | 5 | 5 | ok | 95543.8 | 200 | FDJ3BN | 4 | True |
| 5016 | 5 | 5 | ok | 95561.3 | 200 | FDJ3BN | 4 | True |
| 5014 | 5 | 5 | ok | 95613.3 | 200 | FDJ3BN | 4 | True |
| 5012 | 5 | 5 | ok | 95622.5 | 200 | FDJ3BN | 4 | True |
| 5010 | 5 | 5 | ok | 95654.3 | 200 | FDJ3BN | 4 | True |
| 5013 | 5 | 5 | ok | 95656.9 | 200 | FDJ3BN | 4 | True |
| 5011 | 5 | 5 | ok | 95682.5 | 200 | FDJ3BN | 4 | True |
| 5009 | 5 | 5 | ok | 95703.9 | 200 | FDJ3BN | 4 | True |
| 5008 | 5 | 5 | ok | 95746.9 | 200 | FDJ3BN | 4 | True |

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
      "default_year_mode": "current-year",
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
      "default_year_mode": "current-year",
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
