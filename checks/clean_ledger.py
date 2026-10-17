"""Enforced check: every row removed in cleaning is attributed to a named rule, with its denominator.

Prints "CLEAN rule=<name> removed=<n> of <N>" for every rule in the ledger, then verifies against the Parquet
files themselves (not the ledger's own totals):
  - the first rule's N equals the raw row count; each next N equals the previous N minus its removals;
  - the last remaining count equals the cleaned row count (no unmapped rows);
  - the cleaned data has no null key/value and no duplicate (station, slot).
Exit status 1 (refuse) on any mismatch.
Usage: python checks/clean_ledger.py [--raw PATH] [--clean PATH] [--ledger PATH]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import CLEAN_PARQUET, OUT_ROOT, RAW_PARQUET, get_spark  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_PARQUET)
    ap.add_argument("--clean", default=CLEAN_PARQUET)
    ap.add_argument("--ledger", default=os.path.join(OUT_ROOT, "ledger", "clean_ledger.json"))
    a = ap.parse_args()
    led = json.load(open(a.ledger))
    spark = get_spark("p2-clean-ledger", "clean_ledger")
    n_raw = spark.read.parquet(a.raw).count()
    clean = spark.read.parquet(a.clean)
    n_clean = clean.count()
    problems = []
    remaining = n_raw
    for r in led["rules"]:
        if "of" not in r or "removed" not in r:
            problems.append(f"rule {r.get('rule')} has no count or denominator")
            continue
        extra = f" adjusted={r['adjusted']}" if "adjusted" in r else ""
        print(f"CLEAN rule={r['rule']} removed={r['removed']} of {r['of']}{extra}")
        if r["of"] != remaining:
            problems.append(f"rule {r['rule']}: denominator {r['of']} != rows remaining {remaining}")
        remaining = r["of"] - r["removed"]
    unmapped = remaining - n_clean
    print(f"CLEAN raw_rows={n_raw} clean_rows={n_clean} removed_by_rules={n_raw - remaining} unmapped_rows={unmapped}")
    if unmapped != 0:
        problems.append(f"{unmapped} rows differ between the ledger and the cleaned data (unmapped)")
    nulls = clean.filter(F.col("station_id").isNull() | F.col("slot").isNull() | F.col("rain_mm").isNull()).count()
    dups = clean.groupBy("station_id", "slot").count().filter("count > 1").count()
    print(f"CLEAN null_keys={nulls} duplicate_station_slots={dups}")
    if nulls:
        problems.append(f"{nulls} cleaned rows with null key or value")
    if dups:
        problems.append(f"{dups} duplicate (station, slot) in cleaned data")
    spark.stop()
    for p in problems:
        print("CLEAN PROBLEM:", p)
    if problems:
        print("CLEAN LEDGER: REFUSED")
        sys.exit(1)
    print("CLEAN LEDGER: PASS")


if __name__ == "__main__":
    main()
