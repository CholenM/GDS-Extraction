#!/usr/bin/env python3
"""
GDS Office Load Benchmark — burst vs ramp (office-realistic)
======================================================================
Simulates office traffic on the same CX reservation (IGNACIO PNR FDJ3BN).

Two modes:
  burst — 25 clients hit /v1/extract at the same instant (blast). Good for
          peak contention.  (old default)
  ramp  — office-realistic: users trickle in and concurrency *increases* over
          time, e.g. 1 -> 5 -> 10 -> 15 -> 25, with a pause between steps and
          optional stagger inside each step. Good for finding the knee where
          latency/throughput degrades.

This is the load sibling to benchmark.py / benchmark_load.py in
E:\\DGXSpark_Setup\\vllm-qwen\\ — but GDS-specific and single-payload focused.

Variables we SAVE per request (the contract you asked to plan):
---------------------------------------------------------------
Config:
  - payload (the GDS string) — fixed
  - target_url, api_key, mode, concurrency/steps, ramp_delay, timeout, waves

Per-request (one row per user arrival):
  - id, thread, wave, step, concurrency_at_this_step, queued_ms
  - t_start / t_end (ISO), latency_ms, wall_s
  - http_status, status (ok/mismatch/error/timeout), error (truncated)
  - response bytes, json_ok
  - usage.prompt_tokens / completion_tokens (if gateway returns usage)
  - validation: record_type_ok, pnr_ok, passenger_ok, segment_count_ok, overall_correct
  - decoded fields: record_type, PNR, passenger_names, segments[] (airline, flight_nbr, origin, dest, svc_letter)

Aggregate (written to report):
  - total / ok / mismatch / error, correctness_rate
  - latency p50/p95/p99/min/max/mean/std (all + ok-only + per-step)
  - throughput req/s, avg_segments/s
  - wall_time_total_s + per-step wall
  - gateway/model health snapshots before/after

Usage (on DGX Spark, gateway + vLLM must be up):
  # burst (old): 25 at once
  python benchmark_gds_office.py --mode burst --concurrency 25

  # ramp (office-realistic): gradually increase 1->5->10->15->25
  python benchmark_gds_office.py --mode ramp --ramp-steps 1,5,10,15,25 --ramp-interval 8 --ramp-delay 0.4

  # gentle ramp: 25 users trickle in over ~30s (one every 1.2s)
  python benchmark_gds_office.py --mode ramp --ramp-steps 25 --ramp-delay 1.2

Outputs (next to this script):
  benchmark_gds_office_results.json  — raw per-request rows
  benchmark_gds_office_report.md     — aggregate + per-step + percentiles + correctness
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import textwrap

try:
    import requests
except ImportError:
    print("requests is required: pip install requests", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# The office payload — the CX FDJ3BN reservation you gave
# Note: user string had '*' before city pairs (6*MNLHKG etc.). We keep it verbatim
# as the GDS would — the extractor must handle it.
# ---------------------------------------------------------------------------
GDS_IGNACIO = (
    "--- TST TSM RLR MSC RLP SFP ---"
    "RP/MNLPH21GN/MNLPH21GN            EE/SU  15JUN26/0804Z   FDJ3BNMNLPH21GN/0928LD/6MAY26  "
    "1.IGNACIO/CYNTHIA   2.IGNACIO/MANUEL  "
    "3  CX 974 Q 27JUN 6*MNLHKG HK2  0530 0750  27JUN  E  CX/FDJ3BN  "
    "4  CX 830 Q 27JUN 6*HKGJFK HK2  0905 1310  27JUN  E  CX/FDJ3BN  "
    "5  CX 831 Q 10JUL 5*JFKHKG HK2  1455 1905  11JUL  E  CX/FDJ3BN  "
    "6  CX 939 Q 11JUL 6*HKGMNL HK2  2155 0015  12JUL  E  CX/FDJ3BN"
)

# Expected (ground truth from boss doc Pages 6-12 semantics, applied to this PNR)
# We validate these, not the full 20-field blob, to keep office test focused.
EXPECTED = {
    "Record type": "reservation",
    "PNR": "FDJ3BN",
    "Passenger Name": ["IGNACIO/CYNTHIA", "IGNACIO/MANUEL"],
    "segments_len": 4,
    # Per segment minimal checks: (seg_num, airline, flight_nbr, origin, dest, svc_letter)
    "segments": [
        (3, "CX", 974, "MNL", "HKG", "Q"),
        (4, "CX", 830, "HKG", "JFK", "Q"),
        (5, "CX", 831, "JFK", "HKG", "Q"),
        (6, "CX", 939, "HKG", "MNL", "Q"),
    ],
}


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    # linear interpolation percentile (simple)
    k = (len(sorted_vals) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def validate_response(j: dict) -> dict:
    """Return correctness booleans for the Ignacio PNR."""
    out = {
        "record_type_ok": j.get("Record type") == EXPECTED["Record type"],
        "pnr_ok": j.get("PNR") == EXPECTED["PNR"],
        "passenger_ok": j.get("Passenger Name") == EXPECTED["Passenger Name"],
        "segment_count_ok": len(j.get("segments", [])) == EXPECTED["segments_len"],
        "segments_ok": False,
        "overall_correct": False,
        "detail": "",
    }
    segs = j.get("segments", [])
    if len(segs) == len(EXPECTED["segments"]):
        ok = True
        for (exp_num, exp_air, exp_fn, exp_org, exp_dst, exp_svc), got in zip(EXPECTED["segments"], segs):
            # server may renumber 1..4 vs 3..6 — accept either if code/flight/route match
            if got.get("airline_code") != exp_air or got.get("flight_number") != exp_fn:
                ok = False
                break
            if got.get("originating_airport_code") != exp_org or got.get("destination_airport_code") != exp_dst:
                ok = False
                break
            if got.get("service_class_letter") != exp_svc:
                ok = False
                break
        out["segments_ok"] = ok
    out["overall_correct"] = all([out["record_type_ok"], out["pnr_ok"], out["passenger_ok"], out["segment_count_ok"], out["segments_ok"]])
    if not out["overall_correct"]:
        out["detail"] = f"expected PNR={EXPECTED['PNR']} segs={EXPECTED['segments_len']} got PNR={j.get('PNR')} segs={len(segs)} names={j.get('Passenger Name')}"
    return out


def do_one(req_id: int, url: str, api_key: str, payload: str, timeout: float, wave: int, step: int = 1, concurrency: int = 1, stagger_delay: float = 0.0) -> dict:
    # stagger inside the step to simulate users arriving over time, not all at t0
    if stagger_delay > 0:
        # Use req_id low bits to spread: each of the N requests in this step waits (idx * delay)
        # Caller ensures this by submitting with small sleeps; here we also honor it if set.
        time.sleep((req_id % 100) * stagger_delay * 0.1)  # tiny jitter, real stagger is in submit loop
    rec = {
        "id": req_id,
        "wave": wave,
        "step": step,
        "concurrency": concurrency,
        "thread": None,
        "t_start": None,
        "t_end": None,
        "latency_ms": None,
        "http_status": None,
        "status": "error",
        "error": None,
        "response_bytes": None,
        "json_ok": False,
        "record_type": None,
        "pnr": None,
        "passenger_names": None,
        "segments_len": None,
        "correctness": None,
        "usage": None,
    }
    import threading
    rec["thread"] = threading.current_thread().name
    rec["t_start"] = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    try:
        r = requests.post(
            url,
            json={"gds_text": payload},
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )
        rec["http_status"] = r.status_code
        rec["response_bytes"] = len(r.content)
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["t_end"] = datetime.now(timezone.utc).isoformat()
        if r.status_code == 200:
            try:
                j = r.json()
                rec["json_ok"] = True
                rec["record_type"] = j.get("Record type")
                rec["pnr"] = j.get("PNR")
                rec["passenger_names"] = j.get("Passenger Name")
                rec["segments_len"] = len(j.get("segments", []))
                # usage not exposed by gateway today; capture if present
                rec["usage"] = j.get("usage") or r.headers.get("X-Usage")
                corr = validate_response(j)
                rec["correctness"] = corr
                if corr["overall_correct"]:
                    rec["status"] = "ok"
                else:
                    rec["status"] = "mismatch"
                    rec["error"] = corr["detail"]
            except Exception as exc:
                rec["error"] = f"json parse failed: {exc}"
                rec["status"] = "error"
        else:
            rec["error"] = r.text[:800]
            rec["status"] = "error"
    except requests.exceptions.Timeout:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["t_end"] = datetime.now(timezone.utc).isoformat()
        rec["error"] = f"timeout after {timeout}s"
        rec["status"] = "timeout"
    except Exception as exc:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["t_end"] = datetime.now(timezone.utc).isoformat()
        rec["error"] = str(exc)[:800]
        rec["status"] = "error"
    return rec


def health_snapshot(base_url: str):
    """Probe gateway /healthz and vLLM /health for report header."""
    out = {}
    try:
        h = requests.get(base_url.replace("/v1/extract", "/healthz"), timeout=5)
        out["gateway_health"] = {"status": h.status_code, "body": h.json() if h.headers.get("content-type","").startswith("application/json") else h.text[:500]}
    except Exception as e:
        out["gateway_health"] = {"error": str(e)}
    # vLLM health via gateway's MODEL_URL is not directly reachable from workstation; try common vLLM URL
    for vllm_url in [os.getenv("VLLM_URL", "http://127.0.0.1:8011/health"), "http://192.168.1.65:8011/health"]:
        try:
            hv = requests.get(vllm_url, timeout=5)
            out["vllm_health"] = {"url": vllm_url, "status": hv.status_code}
            break
        except Exception:
            continue
    return out


def write_report(rows: list, meta: dict, out_path: Path):
    ok = [r for r in rows if r["status"] == "ok"]
    mismatch = [r for r in rows if r["status"] == "mismatch"]
    err = [r for r in rows if r["status"] not in ("ok", "mismatch")]
    lats = sorted([r["latency_ms"] for r in rows if isinstance(r["latency_ms"], (int, float))])
    lats_ok = sorted([r["latency_ms"] for r in ok if isinstance(r["latency_ms"], (int, float))])

    def p(vals, pct):
        v = percentile(vals, pct)
        return f"{v:.1f}" if v is not None else "-"

    lines = []
    mode_str = meta.get("mode", "burst")
    if mode_str == "ramp":
        lines.append(f"# GDS Office Benchmark — {meta['payload_name']} — RAMP {meta['ramp_steps']} = {meta['total']} requests")
    else:
        lines.append(f"# GDS Office Benchmark — {meta['payload_name']} — {meta['concurrency']}×{meta['waves']} = {meta['total']} requests")
    lines.append("")
    lines.append(f"- **Run at:** {meta['started_at']}")
    lines.append(f"- **Gateway:** `{meta['url']}` (api_key: `{meta['api_key_masked']}`)")
    lines.append(f"- **Payload:** CX FDJ3BN Ignacio — 2 pax, 4 segs (MNL->HKG->JFK->HKG->MNL)")
    if mode_str == "ramp":
        lines.append(f"- **Mode:** ramp — steps {meta['ramp_steps']} with interval {meta['ramp_interval']}s, stagger {meta['ramp_delay']}s")
    else:
        lines.append(f"- **Mode:** burst — {meta['concurrency']} threads × {meta['waves']} waves")
    lines.append(f"- **Timeout:** {meta['timeout']}s")
    lines.append(f"- **Decoding (frozen):** temp 0.0 / top_p 0.5 / top_k 40 (greedy deterministic)")
    lines.append(f"- **Gate correctness ground truth:** PNR=FDJ3BN, names={EXPECTED['Passenger Name']}, segs={EXPECTED['segments_len']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Wall time:** {meta['wall_s']}s  ·  **Throughput:** {meta['req_per_s']} req/s")
    lines.append(f"- **Results:** {len(ok)} ok · {len(mismatch)} mismatch (HTTP 200 but wrong decode) · {len(err)} error/timeout · {len(rows)} total")
    lines.append(f"- **Correctness rate:** {meta['correctness_rate']}% ({len(ok)}/{len([r for r in rows if r['status'] in ('ok','mismatch')])} of parseable responses)")
    lines.append(f"- **Latency (all):** p50 {p(lats,50)}ms · p95 {p(lats,95)}ms · p99 {p(lats,99)}ms · min {min(lats) if lats else '-'}ms · max {max(lats) if lats else '-'}ms")
    lines.append(f"- **Latency (ok only):** p50 {p(lats_ok,50)}ms · p95 {p(lats_ok,95)}ms · mean {round(statistics.mean(lats_ok),1) if lats_ok else '-'}ms")
    lines.append("")
    # Per-step breakdown for ramp
    if mode_str == "ramp" and any("step" in r for r in rows):
        lines.append("### Per-step breakdown (ramp)")
        lines.append("")
        lines.append("| step | concurrency | ok | mismatch | error | p50 (ms) | p95 (ms) | max (ms) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        steps = sorted(set(r.get("step", 1) for r in rows))
        for s in steps:
            s_rows = [r for r in rows if r.get("step") == s]
            s_ok = [r for r in s_rows if r["status"] == "ok"]
            s_lats = sorted([r["latency_ms"] for r in s_ok if r["latency_ms"]])
            conc = s_rows[0].get("concurrency", "-") if s_rows else "-"
            lines.append(f"| {s} | {conc} | {len(s_ok)} | {sum(1 for r in s_rows if r['status']=='mismatch')} | {sum(1 for r in s_rows if r['status'] not in ('ok','mismatch'))} | {p(s_lats,50)} | {p(s_lats,95)} | {max(s_lats) if s_lats else '-'} |")
        lines.append("")
    if mismatch:
        lines.append("### Mismatches (HTTP 200 but validator failed) — first 5")
        lines.append("")
        for r in mismatch[:5]:
            lines.append(f"- `id={r['id']}` step={r.get('step','-')} latency={r['latency_ms']}ms PNR={r['pnr']} names={r['passenger_names']} segs={r['segments_len']} — {r['correctness']['detail'] if r['correctness'] else ''}")
        lines.append("")
    if err:
        lines.append("### Errors/timeouts — first 5")
        lines.append("")
        for r in err[:5]:
            lines.append(f"- `id={r['id']}` step={r.get('step','-')} status={r['status']} http={r['http_status']} latency={r['latency_ms']}ms — {(r['error'] or '')[:300]}")
        lines.append("")
    lines.append("## Per-request latency (sorted)")
    lines.append("")
    lines.append("| id | step | wave | status | latency (ms) | http | PNR | segs | correct |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["latency_ms"] is None, x["latency_ms"])):
        corr = r["correctness"]["overall_correct"] if r["correctness"] else "-"
        lines.append(f"| {r['id']} | {r.get('step','-')} | {r['wave']} | {r['status']} | {r['latency_ms']} | {r['http_status']} | {r['pnr']} | {r['segments_len']} | {corr} |")
    lines.append("")
    lines.append("## Health snapshots")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta.get("health_before", {}), indent=2))
    lines.append("```")
    lines.append("")
    if meta.get("health_after"):
        lines.append("After:")
        lines.append("```json")
        lines.append(json.dumps(meta["health_after"], indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    if mode_str == "ramp":
        lines.append("- **Ramp mode** simulates office-realistic load: 1 user arrives, then 5, then 10… — find the knee where p95 degrades. `burst` is the peak shock test.")
        lines.append("- `ramp-delay` staggers arrivals inside each step (e.g. 0.4s -> 25 users trickle over ~10s).")
        lines.append("- `ramp-interval` is the pause between steps to let vLLM drain.")
    else:
        lines.append("- **Burst mode** — 25 users hit at the same instant (peak). See `--mode ramp` for gradual office-realistic load.")
    lines.append("- `mismatch` = HTTP 200 but validator failed (decoder bug). `error`/`timeout` = non-200 or timeout.")
    lines.append("- Correctness ground truth is from boss spec Pages 6-12; adapt `EXPECTED` if your PNR changes.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent(__doc__))
    ap.add_argument("--url", default=os.getenv("BENCH_URL", "http://192.168.1.65:8084/v1/extract"), help="Gateway POST url (default $BENCH_URL or 127.0.0.1:8084)")
    ap.add_argument("--api-key", default=os.getenv("BENCH_API_KEY", "gds_key_0000"), help="x-api-key header")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("BENCH_CONCURRENCY", "25")), help="Concurrent threads for burst mode (office users)")
    ap.add_argument("--waves", type=int, default=int(os.getenv("BENCH_WAVES", "1")), help="Number of bursts in burst mode")
    ap.add_argument("--mode", choices=["burst", "ramp"], default=os.getenv("BENCH_MODE", "burst"), help="burst = all at once (peak), ramp = gradual increasing (office-realistic)")
    ap.add_argument("--ramp-steps", type=str, default=os.getenv("RAMP_STEPS", "1,5,10,15,25"), help="Comma-separated concurrencies for ramp mode, e.g. 1,5,10,15,25")
    ap.add_argument("--ramp-interval", type=float, default=float(os.getenv("RAMP_INTERVAL", "8")), help="Seconds to pause between ramp steps")
    ap.add_argument("--ramp-delay", type=float, default=float(os.getenv("RAMP_DELAY", "0.4")), help="Seconds to stagger arrivals inside each ramp step (0 = all at once)")
    ap.add_argument("--timeout", type=float, default=float(os.getenv("BENCH_TIMEOUT", "600")), help="Per-request timeout seconds")
    ap.add_argument("--payload-file", type=str, default=None, help="Optional file containing GDS text (overrides built-in Ignacio payload)")
    ap.add_argument("--out-dir", type=str, default=".", help="Directory to write results")
    args = ap.parse_args()

    payload = Path(args.payload_file).read_text(encoding="utf-8").strip() if args.payload_file else GDS_IGNACIO
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve mode
    if args.mode == "ramp":
        steps = [int(x.strip()) for x in args.ramp_steps.split(",") if x.strip()]
        total = sum(steps)
        print(f"[office-bench] RAMP mode — steps {steps} (total {total} requests) -> {args.url}")
        print(f"[office-bench] interval={args.ramp_interval}s between steps, stagger={args.ramp_delay}s inside each step")
    else:
        total = args.concurrency * args.waves
        print(f"[office-bench] BURST mode — {args.concurrency}×{args.waves} = {total} requests -> {args.url}")

    print(f"[office-bench] payload: {payload[:120]}...")
    print(f"[office-bench] timeout={args.timeout}s  api_key={args.api_key[:4]}***")

    health_before = health_snapshot(args.url)
    print(f"[office-bench] gateway health before: {health_before.get('gateway_health',{}).get('status') or health_before}")

    started_iso = datetime.now(timezone.utc).isoformat()
    wall0 = time.perf_counter()
    rows = []

    if args.mode == "ramp":
        for step_idx, conc in enumerate(steps, start=1):
            print(f"[office-bench] ramp step {step_idx}/{len(steps)} — {conc} concurrent (stagger {args.ramp_delay}s) ...", flush=True)
            with ThreadPoolExecutor(max_workers=conc, thread_name_prefix=f"ramp{step_idx}") as pool:
                futs = []
                for i in range(conc):
                    # stagger submissions to simulate trickle-in
                    if args.ramp_delay > 0 and i > 0:
                        time.sleep(args.ramp_delay)
                    fid = step_idx * 1000 + i
                    futs.append(pool.submit(do_one, fid, args.url, args.api_key, payload, args.timeout, step_idx, step_idx, conc, 0.0))
                for f in as_completed(futs):
                    rows.append(f.result())
            if step_idx < len(steps):
                print(f"[office-bench]   ... step {step_idx} done, sleeping {args.ramp_interval}s before next step", flush=True)
                time.sleep(args.ramp_interval)
    else:
        for wave in range(1, args.waves + 1):
            print(f"[office-bench] burst wave {wave}/{args.waves} — firing {args.concurrency} concurrent ...", flush=True)
            with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix=f"wave{wave}") as pool:
                futs = [pool.submit(do_one, wave * 1000 + i, args.url, args.api_key, payload, args.timeout, wave, wave, args.concurrency) for i in range(args.concurrency)]
                for f in as_completed(futs):
                    rows.append(f.result())
            if wave < args.waves:
                time.sleep(1)

    wall_s = round(time.perf_counter() - wall0, 2)
    health_after = health_snapshot(args.url)

    ok = sum(1 for r in rows if r["status"] == "ok")
    mismatch = sum(1 for r in rows if r["status"] == "mismatch")
    err = len(rows) - ok - mismatch
    parseable = ok + mismatch
    correctness_rate = round(100 * ok / parseable, 1) if parseable else 0.0
    req_per_s = round(len(rows) / wall_s, 2) if wall_s else 0

    if args.mode == "ramp":
        meta = {
            "payload_name": "CX FDJ3BN Ignacio (MNL-HKG-JFK-HKG-MNL)",
            "url": args.url,
            "api_key_masked": args.api_key[:4] + "***" if len(args.api_key) > 4 else "***",
            "payload_len": len(payload),
            "mode": "ramp",
            "ramp_steps": steps,
            "ramp_interval": args.ramp_interval,
            "ramp_delay": args.ramp_delay,
            "concurrency": f"ramp {steps}",
            "waves": f"ramp {len(steps)} steps",
            "total": total,
            "timeout": args.timeout,
            "started_at": started_iso,
            "wall_s": wall_s,
            "req_per_s": req_per_s,
            "correctness_rate": correctness_rate,
            "health_before": health_before,
            "health_after": health_after,
        }
    else:
        meta = {
            "payload_name": "CX FDJ3BN Ignacio (MNL-HKG-JFK-HKG-MNL)",
            "url": args.url,
            "api_key_masked": args.api_key[:4] + "***" if len(args.api_key) > 4 else "***",
            "payload_len": len(payload),
            "mode": "burst",
            "concurrency": args.concurrency,
            "waves": args.waves,
            "total": total,
            "timeout": args.timeout,
            "started_at": started_iso,
            "wall_s": wall_s,
            "req_per_s": req_per_s,
            "correctness_rate": correctness_rate,
            "health_before": health_before,
            "health_after": health_after,
        }

    # Write raw JSON
    results_path = out_dir / "benchmark_gds_office_results.json"
    results_path.write_text(json.dumps({"meta": meta, "expected": EXPECTED, "results": rows}, indent=2), encoding="utf-8")
    # Write markdown report
    report_path = out_dir / "benchmark_gds_office_report.md"
    write_report(rows, meta, report_path)

    print(f"\n[office-bench] done in {wall_s}s — {ok} ok, {mismatch} mismatch, {err} error/timeout of {len(rows)}")
    print(f"[office-bench] correctness {correctness_rate}% — latency p50 {percentile(sorted([r['latency_ms'] for r in rows if r['latency_ms']]), 50):.1f}ms p95 {percentile(sorted([r['latency_ms'] for r in rows if r['latency_ms']]), 95):.1f}ms")
    print(f"[office-bench] raw:  {results_path}")
    print(f"[office-bench] report: {report_path}")
    # Exit code: fail if correctness <100% or any error
    if err > 0 or mismatch > 0:
        print(f"[office-bench] WARN: {mismatch} mismatches / {err} errors — see report for detail", file=sys.stderr)


if __name__ == "__main__":
    main()
