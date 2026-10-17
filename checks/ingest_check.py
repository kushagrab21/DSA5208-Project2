"""Enforced check: all eight annual datasets 2017-2024 are ingested, and nothing was lost on the way to Parquet.

Prints per year: Parquet rows, CSV data lines (local runs), stations, first/last timestamp; then
  MISSING YEARS: n
Exit status 1 (refuse) if any year is missing, empty, or its row count differs from the CSV's data lines.
Usage: python checks/ingest_check.py [--parquet PATH] [--years 2017-2024] [--skip-csv]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import OUT_ROOT, RAW_CSV, RAW_PARQUET, YEARS, get_spark, is_remote  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def csv_data_lines(path):
    out = subprocess.check_output(["wc", "-l", path], text=True)
    return int(out.split()[0]) - 1   # minus header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=RAW_PARQUET)
    ap.add_argument("--years", default=f"{YEARS[0]}-{YEARS[-1]}")
    ap.add_argument("--skip-csv", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    expected = list(range(lo, hi + 1))
    spark = get_spark("p2-ingest-check", "ingest_check")
    df = spark.read.parquet(a.parquet)
    stats = {r["year"]: r for r in df.groupBy("year").agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("station_id").alias("stations"),
        F.min("timestamp").alias("first"), F.max("timestamp").alias("last"),
        F.sum(F.col("_corrupt_record").isNotNull().cast("int")).alias("corrupt")).collect()}
    missing, problems, table = [], [], []
    for y in expected:
        r = stats.get(y)
        if r is None or r["rows"] == 0:
            missing.append(y)
            print(f"INGEST year={y} MISSING")
            continue
        csv_n = None
        csv_path = RAW_CSV.format(year=y)
        if not a.skip_csv and not is_remote(csv_path) and os.path.exists(csv_path):
            csv_n = csv_data_lines(csv_path)
            if csv_n != r["rows"]:
                problems.append(f"year {y}: parquet rows {r['rows']} != csv data lines {csv_n}")
        table.append({"year": y, "rows": r["rows"], "csv_lines": csv_n, "stations": r["stations"],
                      "first": r["first"], "last": r["last"], "corrupt": r["corrupt"]})
        print(f"INGEST year={y} rows={r['rows']} csv_lines={csv_n} stations={r['stations']} "
              f"first={r['first']} last={r['last']} corrupt_records={r['corrupt']}")
    extra = sorted(set(stats) - set(expected))
    if extra:
        problems.append(f"unexpected years present: {extra}")
    print(f"MISSING YEARS: {len(missing)}" + (f" {missing}" if missing else ""))
    for p in problems:
        print("INGEST PROBLEM:", p)
    if a.parquet == RAW_PARQUET and not missing and not problems:
        os.makedirs(os.path.join(OUT_ROOT, "ingest"), exist_ok=True)
        json.dump(table, open(os.path.join(OUT_ROOT, "ingest", "ingest_table.json"), "w"), indent=1)
    spark.stop()
    if missing or problems:
        print("INGEST CHECK: REFUSED")
        sys.exit(1)
    print("INGEST CHECK: PASS")


if __name__ == "__main__":
    main()
