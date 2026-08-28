"""
Computational verification of flight duration timezone math.
"""
from datetime import datetime, timedelta


def hm(h, m):
    return f"{h:02d}:{m:02d}"


def to_min(s):
    h, m = map(int, s.split(":"))
    return h * 60 + m


def main():
    # Verified offsets for October 2026
    MNL_UTC = 8       # Philippines, no DST ever
    SFO_UTC = -7      # PDT (DST ends Nov 1, 2026)
    IAD_UTC = -4      # EDT (DST ends Nov 1, 2026)
    SFO_PST = -8      # WRONG for October

    # Segments: (id, route, dep_offset, arr_offset, dep_h, dep_m, arr_h, arr_m, arrival_day)
    segs = [
        (3, "MNL->SFO", MNL_UTC, SFO_UTC, 8, 40, 6, 35, 3),
        (4, "SFO->IAD", SFO_UTC, IAD_UTC, 10, 50, 19, 4, 3),
        (5, "IAD->SFO", IAD_UTC, SFO_UTC, 8, 30, 11, 22, 19),
        (6, "SFO->MNL", SFO_UTC, MNL_UTC, 13, 50, 19, 0, 20),
    ]

    model_durations = {3: "15:55", 4: "04:14", 5: "06:52", 6: "17:10"}
    stated_correct = {3: "13:55", 4: "05:14", 5: "05:52", 6: "14:10"}

    print("=" * 80)
    print("VERIFIED TIMEZONE OFFSETS - OCTOBER 2026")
    print("=" * 80)
    print()
    print("  MNL (Manila, Philippines): UTC+8  (NO DST)")
    print("  SFO (San Francisco):      UTC-7  (PDT - US DST ends Nov 1, 2026)")
    print("  IAD (Washington Dulles):  UTC-4  (EDT - US DST ends Nov 1, 2026)")
    print()
    print("*** SFO in OCTOBER 2026 is PDT (UTC-7), NOT PST (UTC-8) ***")
    print("*** PST would only apply from November 1, 2026 onwards ***")
    print()

    print("=" * 80)
    print("PER-SEGMENT COMPUTATION (UTC-based method)")
    print("=" * 80)
    print()

    for sid, route, doff, aoff, dh, dm, ah, am, arr_day in segs:
        if sid == 3 or sid == 4:
            dep_date = datetime(2026, 10, 3)
        else:
            dep_date = datetime(2026, 10, 19)
        arr_date = datetime(2026, 10, arr_day)

        # UTC-based
        dep_utc = dep_date + timedelta(hours=dh, minutes=dm) - timedelta(hours=doff)
        arr_utc = arr_date + timedelta(hours=ah, minutes=am) - timedelta(hours=aoff)
        dur = arr_utc - dep_utc
        total_mins = int(dur.total_seconds() / 60)
        duration = f"{total_mins // 60:02d}:{abs(total_mins % 60):02d}"

        # Boss method: convert dep to destination tz
        dep_dest = dep_utc + timedelta(hours=aoff)
        arr_dest = arr_date + timedelta(hours=ah, minutes=am)
        dur_b = arr_dest - dep_dest
        total_mins_b = int(dur_b.total_seconds() / 60)
        duration_b = f"{total_mins_b // 60:02d}:{abs(total_mins_b % 60):02d}"

        # Raw subtraction
        dep_mins = dh * 60 + dm
        arr_mins = ah * 60 + am
        if arr_date > dep_date:
            raw_mins = 24 * 60 + arr_mins - dep_mins
        else:
            raw_mins = arr_mins - dep_mins
            if raw_mins < 0:
                raw_mins += 24 * 60

        # PST version (wrong for Oct 2026)
        test_aoff = SFO_PST if aoff == SFO_UTC else aoff
        test_doff = SFO_PST if doff == SFO_UTC else doff
        dep_utc_pst = dep_date + timedelta(hours=dh, minutes=dm) - timedelta(hours=test_doff)
        arr_utc_pst = arr_date + timedelta(hours=ah, minutes=am) - timedelta(hours=test_aoff)
        dur_pst = arr_utc_pst - dep_utc_pst
        total_mins_pst = int(dur_pst.total_seconds() / 60)
        duration_pst = f"{total_mins_pst // 60:02d}:{abs(total_mins_pst % 60):02d}"

        print(f"Segment {sid}: {route}")
        print(
            f"  Dep: Oct{dep_date.day} {hm(dh, dm)} (UTC{doff:+d})"
        )
        print(
            f"  Arr: Oct{arr_date.day} {hm(ah, am)} (UTC{aoff:+d})"
        )
        print(f"  Dep UTC: {dep_utc.strftime('%H:%M')}  Arr UTC: {arr_utc.strftime('%H:%M')}")
        print(f"  Duration (UTC):    {duration}")
        print(f"  Boss method:       {duration_b} (matches: {duration_b == duration})")
        print(f"  Raw subtraction:   {hm(raw_mins // 60, raw_mins % 60)}")
        print(f"  With PST (SFO=-8): {duration_pst}")

        mmins = to_min(model_durations[sid])
        cmins = to_min(duration)
        smins = to_min(stated_correct[sid])
        diff = mmins - cmins
        stated_diff = smins - cmins

        print(
            f"  Model:    {model_durations[sid]}  (off by +{diff // 60:02d}:{abs(diff % 60):02d} from correct)"
        )
        print(
            f"  Stated:   {stated_correct[sid]}  (off by {stated_diff // 60:+02d}:{abs(stated_diff % 60):02d} from correct)"
        )
        print()

    print("=" * 80)
    print("THE N* FIELD (6*, 1*) IN GDS DATA")
    print("=" * 80)
    print()
    print("In Amadeus availability display format:")
    print("  N* = number of consecutive days the schedule is available")
    print("  NOT a day-of-arrival offset")
    print()
    print("  6*MNLSFO  -> 6 consecutive days available on MNL-SFO route")
    print("  1*IADSFO  -> 1 consecutive day available on IAD-SFO route")
    print("  1*SFOMNL  -> 1 consecutive day available on SFO-MNL route")
    print()
    print("Day-of-arrival offsets are shown in the ARRIVAL DATE field,")
    print("not in the N* field. Segment 6 arrives 20OCT vs dep 19OCT.")
    print()
    print("WARNING: The prompt's instruction to 'Resolve +N arrival day-offsets'")
    print("may confuse the model into treating 6* and 1* as day offsets.")
    print()

    print("=" * 80)
    print("BOSS APPROACH EVALUATION")
    print("=" * 80)
    print()
    print("Boss algorithm:")
    print("  1. Convert dep to destination local time")
    print("  2. Duration = arr_local - dep_in_dest_local")
    print()
    print("This is MATHEMATICALLY EQUIVALENT to UTC-based method:")
    print()
    print("  Boss:   arr_local - (dep_local + dest_off - dep_off)")
    print("        = arr_local - dep_local - dest_off + dep_off")
    print()
    print("  UTC:    (arr_local - dest_off) - (dep_local - dep_off)")
    print("        = arr_local - dest_off - dep_local + dep_off")
    print()
    print("  => IDENTICAL. Both methods give the same result.")
    print()
    print("Boss approach works for ALL segments (verified above).")
    print()
    print("HOWEVER - the SAME fundamental problem remains:")
    print("  The model must KNOW the correct timezone offsets.")
    print("  Neither algorithm works if the model uses wrong offsets.")
    print()

    print("=" * 80)
    print("MODEL ERROR ANALYSIS - WHAT IS THE MODEL DOING?")
    print("=" * 80)
    print()
    print("Model durations vs PST-based expectations:")
    # Recalc PST for each segment individually
    pst_results = {}
    for sid in [3, 4, 5, 6]:
        if sid == 3:
            d, a = datetime(2026, 10, 3), datetime(2026, 10, 3)
            dp_h, dp_m, ap_h, ap_m = 8, 40, 6, 35
            do, ao = MNL_UTC, SFO_PST
        elif sid == 4:
            d, a = datetime(2026, 10, 3), datetime(2026, 10, 3)
            dp_h, dp_m, ap_h, ap_m = 10, 50, 19, 4
            do, ao = SFO_PST, IAD_UTC
        elif sid == 5:
            d, a = datetime(2026, 10, 19), datetime(2026, 10, 19)
            dp_h, dp_m, ap_h, ap_m = 8, 30, 11, 22
            do, ao = IAD_UTC, SFO_PST
        else:
            d, a = datetime(2026, 10, 19), datetime(2026, 10, 20)
            dp_h, dp_m, ap_h, ap_m = 13, 50, 19, 0
            do, ao = SFO_PST, MNL_UTC

        dep_pst = d + timedelta(hours=dp_h, minutes=dp_m) - timedelta(hours=do)
        arr_pst = a + timedelta(hours=ap_h, minutes=ap_m) - timedelta(hours=ao)
        pst_dur = arr_pst - dep_pst
        pst_mins = int(pst_dur.total_seconds() / 60)
        pst_dur_str = f"{pst_mins // 60:02d}:{abs(pst_mins % 60):02d}"
        pst_results[sid] = pst_dur_str
        print(f"  Seg {sid}: model={model_durations[sid]}, PST-calc={pst_dur_str}")

    print()
    print("  Seg 4 (04:14) and Seg 5 (06:52) EXACTLY match PST-based calculations.")
    print("  Seg 3 (15:55) does NOT match PST (which gives 13:55) -- off by +2h")
    print("  Seg 6 (17:10) does NOT match PST (which gives 13:10) -- off by +4h")
    print()
    print("The model does NOT use a consistent set of offsets across segments.")
    print("This suggests the model is confused and making independent errors.")
    print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("The root cause is not the algorithm (both UTC and boss methods are correct).")
    print("The root cause is: the model does not know accurate timezone offsets.")
    print()
    print("Two solutions:")
    print()
    print("1. PROMPT FIX (quick, less reliable):")
    print("   Add explicit timezone offsets to the system prompt:")
    print('   "MNL=UTC+8, SFO=UTC-7 (PDT in Oct 2026), IAD=UTC-4 (EDT in Oct 2026)"')
    print("   Change instruction to boss's approach (better specificity)")
    print("   Clarify that N* is availability count, NOT a day offset")
    print()
    print("2. CODE-LEVEL FIX (best, reliable):")
    print("   Compute flight_duration server-side using a proper timezone library")
    print("   (e.g., zoneinfo or pytz) instead of relying on LLM math.")
    print("   The LLM only extracts dep/arr times; the service computes duration.")
    print()
    print("The boss's approach is good as a SPECIFIC instruction, but without")
    print("providing the actual offsets, it will not fix the root problem.")
    print("Recommendation: combine boss-specific wording WITH explicit offset")
    print("table in the prompt, and plan for a code-level duration computation.")
    print()


if __name__ == "__main__":
    main()
