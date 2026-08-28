"""
Focused arithmetic analysis of the 4-segment October 2026 UA flights.

Questions answered:
  Q1: Are there arithmetic errors in the prior analysis?
  Q2: Is the model's 06:52 for segment 5 actually correct?
  Q3: Could GDS arrival "11:22" be a misread of "10:22"?
  Q4: Which segment is wrong — and can any single offset fix all 4?
"""
from datetime import datetime, timedelta


def to_min(h, m):
    return h * 60 + m


def mins_to_str(total_min):
    return f"{total_min // 60:02d}:{abs(total_min % 60):02d}"


print("=" * 80)
print("FACT CHECK: US DST 2026")
print("=" * 80)
print()
print("US DST 2026: Ends Sunday, November 1, 2026 at 2:00 AM local time.")
print("Before Nov 1, 2026: all US zones are in Daylight Saving Time (DST).")
print("  SFO (Pacific): PDT = UTC-7  (NOT PST=UTC-8)")
print("  IAD (Eastern): EDT = UTC-4  (NOT EST=UTC-5)")
print()
print("Source: US federal law (Energy Policy Act of 2005), codified.")
print()

print("=" * 80)
print("Q1: RE-VERIFY ALL MATH FROM SCRATCH (NO PRIOR ASSUMPTIONS)")
print("=" * 80)
print()

# GDS offsets (verified against IATA/timezone databases for Oct 2026)
offsets = {
    "MNL": 8,   # UTC+8, no DST ever
    "SFO": -7,  # PDT (UTC-7) — DST active until Nov 1
    "IAD": -4,  # EDT (UTC-4) — DST active until Nov 1
}
SFO_PST = -8  # Hypothetical (wrong for Oct)

# Segments: (id, route, dep_loc, dep_tz, dep_hm, arr_loc, arr_tz, arr_hm, dep_day, arr_day)
segments = [
    (3, "MNL->SFO", "MNL", offsets["MNL"],  (8, 40), "SFO", offsets["SFO"], (6, 35), 3, 3),
    (4, "SFO->IAD", "SFO", offsets["SFO"],  (10, 50), "IAD", offsets["IAD"], (19, 4), 3, 3),
    (5, "IAD->SFO", "IAD", offsets["IAD"],  (8, 30), "SFO", offsets["SFO"], (11, 22), 19, 19),
    (6, "SFO->MNL", "SFO", offsets["SFO"],  (13, 50), "MNL", offsets["MNL"], (19, 0), 19, 20),
]

computed = {}
for sid, route, dep_loc, doff, dep_hm, arr_loc, aoff, arr_hm, dep_day, arr_day in segments:
    dep_dt = datetime(2026, 10, dep_day)
    arr_dt = datetime(2026, 10, arr_day)

    # Step 1: Convert both to UTC
    dep_utc_min = to_min(*dep_hm) - doff * 60
    arr_utc_min = to_min(*arr_hm) - aoff * 60

    # Step 2: Compute UTC duration (with day wrap)
    delta = arr_utc_min - dep_utc_min
    if delta < 0:
        delta += 24 * 60

    # Boss method: convert dep to dest local, then subtract
    dep_in_dest = to_min(*dep_hm) + (aoff - doff) * 60
    arr_in_dest = to_min(*arr_hm)
    delta_b = arr_in_dest - dep_in_dest
    if delta_b < 0:
        delta_b += 24 * 60

    computed[sid] = (delta, delta_b)
    assert delta == delta_b, f"Seg {sid}: UTC and Boss differ! {delta} vs {delta_b}"
    print(f"  Segment {sid}: {route}")
    print(f"    dep {dep_loc} {dep_hm[0]:02d}:{dep_hm[1]:02d} (UTC{doff:+d}) -> UTC {dep_utc_min//60:02d}:{dep_utc_min%60:02d}")
    print(f"    arr {arr_loc} {arr_hm[0]:02d}:{arr_hm[1]:02d} (UTC{aoff:+d}) -> UTC {arr_utc_min//60:02d}:{arr_utc_min%60:02d}")
    print(f"    Duration = {mins_to_str(delta)}")
    print(f"    Boss method = {mins_to_str(delta_b)} (MATCH: {delta == delta_b})")
    print()

print("RESULT: Q1 — NO ARITHMETIC ERRORS FOUND.")
print("  Both methods (UTC and Boss) give identical results for every segment.")
print("  The math in the prior analysis was correct.")
print()

print("=" * 80)
print("Q2: IS MODEL'S 06:52 FOR SEGMENT 5 ACTUALLY CORRECT?")
print("=" * 80)
print()

print("  Segment 5: IAD->SFO, dep=08:30, arr=11:22, 19OCT")
print("  Computed (PDT): 05:52")
print("  Model answer:   06:52")
print()

# What offset would give 06:52 = 412 minutes?
# dep UTC = 12:30 = 750 min
# For 06:52 = 412 min: arr UTC must be 750 + 412 = 1162 min = 19:22
# arr UTC = 11:22 + SFO_offset => SFO_offset = 19:22 - 11:22 = 8 hours = UTC-8
print("  For model's 06:52 to be correct:")
print("    arr UTC would need to be 19:22")
print("    arr local = 11:22, so SFO offset = 19:22 - 11:22 = 8 hours -> SFO=UTC-8 (PST)")
print("  But SFO is UTC-7 (PDT) in October 2026.")
print()
print("  VERDICT: 06:52 is WRONG. 05:52 is correct for PDT.")
print("  The model used PST for SFO on this segment.")
print()

print("=" * 80)
print("Q3: COULD '11:22' BE A MISREAD OF '10:22'?")
print("=" * 80)
print()

print("  If arr = 10:22 SFO with PDT (UTC-7):")
print("    dep UTC = 12:30, arr UTC = 10:22 + 7 = 17:22")
print("    Duration = 17:22 - 12:30 = 04:52")
print("    This does NOT match any 'correct' (05:52 or 06:52).")
print()

print("  If arr = 10:22 SFO with PST (UTC-8):")
print("    dep UTC = 12:30, arr UTC = 10:22 + 8 = 18:22")
print("    Duration = 18:22 - 12:30 = 05:52")
print("    This matches the 'correct' 05:52 BUT only because we used the WRONG")
print("    PST offset to compensate for the wrong arrival time — double error.")
print()

print("  VERDICT: Changing 11:22 to 10:22 does NOT help. The issue is the SFO")
print("  offset, not the arrival time digits. The arrival time 11:22 is likely")
print("  correct as read from GDS.")
print()

print("=" * 80)
print("Q4: WHICH SEGMENT IS WRONG? THE INCONSISTENCY TABLE")
print("=" * 80)
print()

# Compute what duration each segment WOULD have under PST
pst_durations = {}
for sid, route, dep_loc, doff, dep_hm, arr_loc, aoff, arr_hm, dep_day, arr_day in segments:
    dep_dt = datetime(2026, 10, dep_day)
    arr_dt = datetime(2026, 10, arr_day)

    # PST version: SFO = -8
    pst_doff = SFO_PST if dep_loc == "SFO" else doff
    pst_aoff = SFO_PST if arr_loc == "SFO" else aoff

    dep_utc_min = to_min(*dep_hm) - pst_doff * 60
    arr_utc_min = to_min(*arr_hm) - pst_aoff * 60
    delta = arr_utc_min - dep_utc_min
    if delta < 0:
        delta += 24 * 60
    pst_durations[sid] = delta

# The "stated correct" durations (from the test annotations)
stated_correct = {3: 13*60+55, 4: 5*60+14, 5: 5*60+52, 6: 14*60+10}

print("  Comparison table (all durations in minutes):")
print()
print(f"  {'Seg':>3} | {'PDT (correct)':>12} | {'PST (wrong)':>12} | {'Stated':>10} | {'Match?'}")
print(f"  {'':->5}-+-{'':->13}-+-{'':->13}-+-{'':->11}-+-{'':->10}")
for sid in [3, 4, 5, 6]:
    dt_str = mins_to_str(computed[sid][0])
    ps_str = mins_to_str(pst_durations[sid])
    sc_str = mins_to_str(stated_correct[sid])
    match_pdt = "PDT" if computed[sid][0] == stated_correct[sid] else ""
    match_pst = "PST" if pst_durations[sid] == stated_correct[sid] else ""
    match_str = f"{match_pdt}/{match_pst}".rstrip("/")
    if match_pdt and not match_pst:
        match_str = "PDT"
    elif not match_pdt and match_pst:
        match_str = "PST"
    elif not match_pdt and not match_pst:
        match_str = "NEITHER"
    print(f"  {sid:>3} | {dt_str:>12} | {ps_str:>12} | {sc_str:>10} | {match_str:>10}")

print()
print("  COUNT OF MATCHES:")
print(f"    PDT (UTC-7) matches: {sum(1 for s in [3,4,5,6] if computed[s][0] == stated_correct[s])} of 4 segments")
print(f"    PST (UTC-8) matches: {sum(1 for s in [3,4,5,6] if pst_durations[s] == stated_correct[s])} of 4 segments")
print()

# Now try: what if the stated_correct values are just WRONG for some?
# Check: if we assume PDT is correct, which stated_correct is wrong?
print("  Assuming PDT (UTC-7) is the correct SFO offset:")
pdt_matches = {s: computed[s][0] == stated_correct[s] for s in [3, 4, 5, 6]}
for sid in [3, 4, 5, 6]:
    status = "MATCH" if pdt_matches[sid] else "MISMATCH"
    diff = computed[sid][0] - stated_correct[sid]
    print(f"    Seg {sid}: PDT={mins_to_str(computed[sid][0])}, stated={mins_to_str(stated_correct[sid])} => {status} (diff={diff//60:+d}h{diff%60:+d}m)")

print()

# Reverse: if we assume PST is correct, which stated_correct is wrong?
print("  Assuming PST (UTC-8) is the correct SFO offset:")
pst_matches = {s: pst_durations[s] == stated_correct[s] for s in [3, 4, 5, 6]}
for sid in [3, 4, 5, 6]:
    status = "MATCH" if pst_matches[sid] else "MISMATCH"
    diff = pst_durations[sid] - stated_correct[sid]
    print(f"    Seg {sid}: PST={mins_to_str(pst_durations[sid])}, stated={mins_to_str(stated_correct[sid])} => {status} (diff={diff//60:+d}h{diff%60:+d}m)")

print()
print("=" * 80)
print("Q5: COMPLETE VERDICT")
print("=" * 80)
print()
print("  1. SFO in October 2026 is unequivocally PDT (UTC-7).")
print("     US DST ends Nov 1, 2026. No exception for SFO.")
print()
print("  2. The 4 'stated correct' durations are INCONSISTENT with any single")
print("     SFO offset. 3 out of 4 match PDT; 1 matches PST.")
print()
print("  3. The OUTLIER is Segment 3: its stated correct of 13:55 uses")
print("     PST (UTC-8) for SFO, but the correct offset is PDT (UTC-7).")
print()
print("  4. The correct answer for Segment 3 is: 12:55")
print("     The 13:55 value was computed with the WRONG SFO offset.")
print()
print("  5. Segments 4, 5, 6 are all internally consistent with PDT.")
print("     Their stated correct values (05:14, 05:52, 14:10) are correct.")
print()
print("  6. The MODEL's answers (15:55, 04:14, 06:52, 17:10) are all WRONG.")
print("     They show no consistent pattern — the model is independently")
print("     confusing itself on each segment, using whatever offsets it")
print("     hallucinates each time.")
print()
print("=" * 80)
print("  ROOT CAUSE: The model does not reliably compute timezone offsets.")
print("  The fix is server-side computation (which has been implemented).")
print("=" * 80)
