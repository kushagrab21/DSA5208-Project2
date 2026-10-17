"""Task 2, step 1: per (station, 5-minute slot) base table with past-rain features and the 30-minute future total.

Prediction time t = the end of a reported slot. Everything named r* uses readings with slot end <= t; f30 is the
station's total over the six slots ending t+5 ... t+30 min, i.e. rain in (t, t+30 min]. A row's label is defined only
when all six future slots were reported and lie in the same calendar year as t (so no label crosses the
train/validation/test year boundary).

  r5, r15, r30, r60, r120, r24h   station rain in the last 5 ... 1440 minutes (sum of reported slots)
  n120                            reported slots among the last 24 (completeness of the recent past)
  mins_since_wet                  minutes since the last wet slot, capped at 360
  f30, f_n, label_ok              future 30-minute total, reported future slots, label defined

Also writes the island-wide state per slot (fraction of reporting stations wet now, max and mean r30), and the
distribution of f30 in the TRAINING years only, used to choose the threshold grid (outputs/task2/f30_train_dist.csv).
Usage: python src/task2_features.py
"""
import os

import common  # noqa: F401
from common import CLEAN_PARQUET, DATA_ROOT, FEATURES_PARQUET, PARQUET_ROOT, SLOT_SECONDS, get_spark, write_csv_local
from pyspark.sql import Window
from pyspark.sql import functions as F

ISLAND_PARQUET = PARQUET_ROOT + "/island_state"
TRAIN_YEARS = list(range(2017, 2023))   # must match config/eval_plan.json (checked by src/task2_sample.py)
CANDIDATE_THRESHOLDS = [0.0, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]


def main():
    spark = get_spark("p2-task2-features", "task2_features")
    c = spark.read.parquet(CLEAN_PARQUET).select("station_id", "slot", "rain_mm", "year")
    w = Window.partitionBy("station_id").orderBy("slot")

    def past(k):   # k slots ending at t (inclusive)
        return F.sum("rain_mm").over(w.rangeBetween(-(k - 1), 0))

    b = (c.withColumn("r15", past(3)).withColumn("r30", past(6)).withColumn("r60", past(12))
          .withColumn("r120", past(24)).withColumn("r24h", past(288))
          .withColumn("n120", F.count("rain_mm").over(w.rangeBetween(-23, 0)))
          .withColumn("last_wet", F.max(F.when(F.col("rain_mm") > 0, F.col("slot"))).over(w.rangeBetween(-71, 0)))
          .withColumn("f30", F.sum("rain_mm").over(w.rangeBetween(1, 6)))
          .withColumn("f_n", F.count("rain_mm").over(w.rangeBetween(1, 6))))
    b = (b.withColumnRenamed("rain_mm", "r5")
          .withColumn("mins_since_wet", F.coalesce((F.col("slot") - F.col("last_wet")) * 5, F.lit(360)).cast("int"))
          .withColumn("label_ok", (F.col("f_n") == 6) &
                      (F.year(F.timestamp_seconds((F.col("slot") + 6) * SLOT_SECONDS)) == F.col("year")))
          .drop("last_wet"))
    (b.repartition("year").sortWithinPartitions("station_id", "slot")
      .write.mode("overwrite").partitionBy("year").parquet(FEATURES_PARQUET))
    print("TASK2 features_base written", flush=True)

    b = spark.read.parquet(FEATURES_PARQUET)
    isl = (b.groupBy("slot").agg(
        F.count(F.lit(1)).alias("isl_n"),
        F.avg((F.col("r5") > 0).cast("double")).alias("isl_wet_now"),
        F.avg((F.col("r30") > 0).cast("double")).alias("isl_wet30"),
        F.max("r30").alias("isl_max_r30"), F.avg("r30").alias("isl_mean_r30")))
    isl.write.mode("overwrite").parquet(ISLAND_PARQUET)
    print("TASK2 island_state written", flush=True)

    # threshold grid evidence, training years only
    tr = b.filter(F.col("year").isin(TRAIN_YEARS) & F.col("label_ok"))
    aggs = [F.count(F.lit(1)).alias("n")]
    for th in CANDIDATE_THRESHOLDS:
        aggs.append(F.sum((F.col("f30") > th).cast("long")).alias(f"gt_{th:g}"))
    aggs += [F.expr("percentile_approx(case when f30 > 0 then f30 end, array(0.25,0.5,0.75,0.9,0.95,0.99,0.999))")
             .alias("q_wet")]
    row = tr.agg(*aggs).first().asDict()
    import pandas as pd
    rows = [{"threshold_mm": th, "rows": row["n"], "positives": row[f"gt_{th:g}"],
             "base_rate": row[f"gt_{th:g}"] / row["n"]} for th in CANDIDATE_THRESHOLDS]
    write_csv_local(pd.DataFrame(rows), "f30_train_dist", "task2")
    qs = row["q_wet"]
    write_csv_local(pd.DataFrame({"quantile": [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999], "f30_mm": qs}),
                    "f30_train_wet_quantiles", "task2")
    for r in rows:
        print(f"TASK2 f30>{r['threshold_mm']:g}: {r['positives']} of {r['rows']} ({100 * r['base_rate']:.3f}%)")
    print("TASK2 wet f30 quantiles (25,50,75,90,95,99,99.9%):", qs)
    # label coverage per year
    for r in b.groupBy("year").agg(F.count(F.lit(1)).alias("n"),
                                   F.sum(F.col("label_ok").cast("long")).alias("ok")).orderBy("year").collect():
        print(f"TASK2 year={r['year']} prediction_times={r['n']} label_defined={r['ok']}")
    spark.stop()


if __name__ == "__main__":
    main()
