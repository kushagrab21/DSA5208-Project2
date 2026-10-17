"""How many Task 1 totals sit exactly on a threshold, where summation order decides the count.

Task 1 counts hours with a total >= 20 mm or >= 40 mm and complete days with a (scaled) total >= 50 mm. Readings are
multiples of 0.2 mm, so many totals equal a threshold exactly in decimal, but a floating-point sum can land a hair
below or above it depending on the order of addition (19.9999999 vs 20.0000001). Spark sums in partition order, so
two clusters can disagree on these counts by at most the number of totals within 1e-6 mm of the threshold.
checks/cloud_parity.py uses this file to accept such differences only when they are within that bound.

Writes outputs/threshold_boundary.json: {"hours_ge_20": {"year": {...}, "month": {...}}, "hours_ge_40": {...},
"heavy_days": {...}} with counts of totals within EPS of the threshold. Usage: python checks/threshold_boundary.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]
from pyspark.sql import functions as F  # noqa: E402

from common import CLEAN_PARQUET, OUT_ROOT, SLOTS_PER_DAY, get_spark  # noqa: E402
from task1 import HEAVY_DAY_MM, HEAVY_HOUR_MM, VERY_HEAVY_HOUR_MM  # noqa: E402

EPS = 1e-6


def near(col, t):
    return (F.abs(F.col(col) - t) < EPS).cast("long")


def by(df, key, aggs):
    return {str(r[key]): {k: int(r[k]) for k in aggs} for r in df.groupBy(key).agg(
        *[e.alias(k) for k, e in aggs.items()]).collect()}


def main():
    spark = get_spark("p2-threshold-boundary", "threshold_boundary")
    c = (spark.read.parquet(CLEAN_PARQUET).withColumn("day", F.to_date("ts"))
         .withColumn("hour_start", F.hour(F.col("ts") - F.expr("INTERVAL 5 MINUTES")))
         .withColumn("day_start", F.to_date(F.col("ts") - F.expr("INTERVAL 5 MINUTES"))))
    hr = (c.groupBy("station_id", "year", "day_start", "hour_start")
          .agg(F.count(F.lit(1)).alias("n"), F.sum("rain_mm").alias("mm")).filter("n = 12")
          .withColumn("month", F.month("day_start")).cache())
    dd = (c.groupBy("station_id", "year", "day").agg(F.count(F.lit(1)).alias("n"), F.sum("rain_mm").alias("mm"))
          .filter(F.col("n") >= 0.9 * SLOTS_PER_DAY).withColumn("s", F.col("mm") * SLOTS_PER_DAY / F.col("n")))
    h20, h40, d50 = near("mm", HEAVY_HOUR_MM), near("mm", VERY_HEAVY_HOUR_MM), near("s", HEAVY_DAY_MM)
    hy = by(hr, "year", {"h20": F.sum(h20), "h40": F.sum(h40)})
    hm = by(hr, "month", {"h20": F.sum(h20)})
    dy = by(dd, "year", {"d50": F.sum(d50)})
    out = {"eps": EPS,
           "hours_ge_20": {"year": {k: v["h20"] for k, v in hy.items()}, "month": {k: v["h20"] for k, v in hm.items()}},
           "hours_ge_40": {"year": {k: v["h40"] for k, v in hy.items()}},
           "heavy_days": {"year": {k: v["d50"] for k, v in dy.items()}}}
    path = os.path.join(OUT_ROOT, "threshold_boundary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for col, d in out.items():
        if col != "eps":
            for key, vals in d.items():
                print(f"BOUNDARY {col} by {key}: " + " ".join(f"{k}={vals[k]}" for k in sorted(vals, key=int)))
    print(f"BOUNDARY written {path}")
    spark.stop()


if __name__ == "__main__":
    main()
