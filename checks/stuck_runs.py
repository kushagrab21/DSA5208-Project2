"""Stuck-gauge check (print only): runs of consecutive slots at one station with the SAME positive value.

A gauge that is stuck would repeat one positive value for hours. Prints the number of runs of >= 6 and >= 12
slots per year and the longest runs, and writes outputs/ledger/stuck_runs.json. No rows are removed by it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import CLEAN_PARQUET, OUT_ROOT, get_spark  # noqa: E402
from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main():
    spark = get_spark("p2-stuck-runs", "stuck_runs")
    c = spark.read.parquet(CLEAN_PARQUET).select("station_id", "slot", "rain_mm", "year")
    w = Window.partitionBy("station_id").orderBy("slot")
    x = (c.withColumn("prev", F.lag("rain_mm").over(w)).withColumn("pslot", F.lag("slot").over(w))
          .withColumn("brk", ((F.col("prev") != F.col("rain_mm")) | (F.col("pslot") != F.col("slot") - 1)
                              | F.col("prev").isNull()).cast("int"))
          .withColumn("grp", F.sum("brk").over(w.rowsBetween(Window.unboundedPreceding, 0))))
    runs = (x.groupBy("station_id", "grp").agg(F.count(F.lit(1)).alias("len"), F.first("rain_mm").alias("v"),
                                               F.min("year").alias("year"), F.min("slot").alias("start"))
            .filter("v > 0").cache())
    by_year = (runs.groupBy("year").agg(F.sum((F.col("len") >= 6).cast("long")).alias("runs_ge6"),
                                        F.sum((F.col("len") >= 12).cast("long")).alias("runs_ge12"),
                                        F.max("len").alias("max_len")).orderBy("year").toPandas())
    top = runs.orderBy(F.desc("len")).limit(5).toPandas()
    out = {"max_len": int(top.len.iloc[0]), "max_len_value": float(top.v.iloc[0]),
           "max_len_station": top.station_id.iloc[0], "max_len_year": int(top.year.iloc[0]),
           "runs_ge12": int(by_year.runs_ge12.sum()), "runs_ge6": int(by_year.runs_ge6.sum()),
           "by_year": by_year.to_dict(orient="records")}
    for r in by_year.itertuples():
        print(f"STUCK year={r.year} runs>=6={r.runs_ge6} runs>=12={r.runs_ge12} longest={r.max_len}")
    print(f"STUCK longest run: {out['max_len']} slots of {out['max_len_value']:g} mm at {out['max_len_station']} "
          f"({out['max_len_year']})")
    os.makedirs(os.path.join(OUT_ROOT, "ledger"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT_ROOT, "ledger", "stuck_runs.json"), "w"), indent=1, default=int)
    spark.stop()


if __name__ == "__main__":
    main()
