"""Task 2, step 2: build the training and validation tables and the two baselines, from the FROZEN plan.

Reads config/eval_plan.json only through checks/eval_plan.load_frozen_plan() (refuses if not frozen or changed).
Builds (station, t) samples for the training and validation years; the test years are not read here.

  - stratified, deterministic sampling (hash of station, slot and seed), weight = 1 / sampling rate
  - neighbour features: the 6 nearest stations' readings at the same slot t
  - island features: share of reporting stations wet at t, max and mean of their last-30-minute totals
  - thresholds: one per training row (uniform over the grid); every threshold for each validation row
  - baselines from the full (unsampled) training years: climatology (station, month, hour, theta) and persistence

Usage: python src/task2_sample.py
"""
import math
import os
import sys

import common  # noqa: F401
from common import DATA_ROOT, PARQUET_ROOT, FEATURES_PARQUET, ROOT, STATIONS_PARQUET, get_spark, write_csv_local
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(ROOT, "checks"))
import eval_plan as ep  # noqa: E402

ISLAND_PARQUET = PARQUET_ROOT + "/island_state"
T2 = PARQUET_ROOT + "/task2"
K_NEIGHBOURS = 6
CLIM_PSEUDO = 20.0


def u01(seed, *cols):
    """Deterministic uniform(0,1) per row from a hash (independent of partitioning, unlike rand())."""
    return (F.pmod(F.xxhash64(F.lit(seed), *cols), F.lit(1_000_003)) / F.lit(1_000_003.0))


def neighbour_pairs(spark):
    st = spark.read.parquet(STATIONS_PARQUET).select("station_id", "lat", "lon").toPandas()
    rows = []
    for a in st.itertuples():
        d = []
        for b in st.itertuples():
            if a.station_id == b.station_id:
                continue
            p1, p2 = math.radians(a.lat), math.radians(b.lat)
            dl = math.radians(b.lon - a.lon)
            h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
            d.append((6371.0 * 2 * math.asin(math.sqrt(h)), b.station_id))
        for km, nb in sorted(d)[:K_NEIGHBOURS]:
            rows.append((a.station_id, nb, float(km)))
    return spark.createDataFrame(rows, ["station_id", "nb_id", "nb_km"])


def add_time_station(df, st):
    ts = F.timestamp_seconds(F.col("slot") * 300)
    m = F.month(ts)
    tod = (F.hour(ts) * 60 + F.minute(ts)) / 1440.0
    doy = (F.dayofyear(ts) - 1) / 365.25
    return (df.withColumn("hour", F.hour(ts)).withColumn("month", m)
              .withColumn("season", F.when(m.isin(12, 1, 2, 3), 0).when(m.isin(4, 5), 1)
                          .when(m.isin(6, 7, 8, 9), 2).otherwise(3))
              .withColumn("hour_sin", F.sin(2 * math.pi * tod)).withColumn("hour_cos", F.cos(2 * math.pi * tod))
              .withColumn("doy_sin", F.sin(2 * math.pi * doy)).withColumn("doy_cos", F.cos(2 * math.pi * doy))
              .join(st, "station_id", "left"))


def station_table(spark, b_train):
    """Station coordinates and training-years wet rate (a function of the station only)."""
    wet = (b_train.groupBy("station_id")
           .agg(F.avg((F.col("r5") > 0).cast("double")).alias("station_train_wet_rate")))
    overall = b_train.agg(F.avg((F.col("r5") > 0).cast("double"))).first()[0]
    return (spark.read.parquet(STATIONS_PARQUET).select("station_id", F.col("lat").alias("station_lat"),
                                                        F.col("lon").alias("station_lon"))
            .join(wet, "station_id", "left").fillna({"station_train_wet_rate": overall}))


def labelled_base(spark, years):
    return (spark.read.parquet(FEATURES_PARQUET).filter(F.col("year").isin(years)).filter("label_ok")
            .withColumn("active", (F.col("r120") > 0) | (F.col("f30") > 0)))


def neighbour_source(spark, years):
    """Neighbours' readings at t: every reported slot (not only rows with a defined label)."""
    return spark.read.parquet(FEATURES_PARQUET).select(
        F.col("station_id").alias("nb_id"), "slot", F.col("r5").alias("nb_r5"), F.col("r30").alias("nb_r30"),
        "year").filter(F.col("year").isin(years)).drop("year")


def build_split(spark, plan, b, split, years, st, pairs, nb_src, isl, out_name=None):
    """Sample one split exactly as the frozen plan says and write it to {T2}/<split>_sample. Returns the path."""
    grid = plan["threshold_grid_mm"]
    sp = plan["sampling"][split]
    rate = F.when(F.col("active"), F.lit(sp["active_rate"])).otherwise(F.lit(sp["inactive_rate"]))
    s = (b.filter(F.col("year").isin(years)).withColumn("rate", rate)
          .filter(u01(sp["seed"], F.col("station_id"), F.col("slot")) < F.col("rate"))
          .withColumn("weight", 1.0 / F.col("rate")).drop("rate"))
    s = s.cache()
    nb = (s.select("station_id", "slot").join(pairs, "station_id").join(nb_src, ["nb_id", "slot"])
           .groupBy("station_id", "slot").agg(
               F.max("nb_r30").alias("nb_max_r30"), F.avg("nb_r30").alias("nb_mean_r30"),
               F.max("nb_r5").alias("nb_max_r5"), F.avg((F.col("nb_r5") > 0).cast("double")).alias("nb_wet_now_frac"),
               F.count(F.lit(1)).alias("nb_n")))
    s = (s.join(nb, ["station_id", "slot"], "left").join(isl, "slot", "left")
          .fillna({"nb_max_r30": 0.0, "nb_mean_r30": 0.0, "nb_max_r5": 0.0, "nb_wet_now_frac": 0.0, "nb_n": 0}))
    s = add_time_station(s, st)
    arr = F.array(*[F.lit(float(t)) for t in grid])
    if split == "train":
        idx = (F.pmod(F.xxhash64(F.lit(sp["seed"] + 1), "station_id", "slot"), F.lit(len(grid)))).cast("int")
        s = s.withColumn("theta", arr[idx])
    else:
        s = s.withColumn("theta", F.explode(arr))
    s = s.withColumn("y", (F.col("f30") > F.col("theta")).cast("double"))
    name = out_name or split
    out = f"{T2}/{name}_sample"
    s.write.mode("overwrite").parquet(out)
    s = spark.read.parquet(out)
    r = s.agg(F.count(F.lit(1)).alias("rows"), F.sum("y").alias("pos"), F.sum("weight").alias("w"),
              F.sum(F.col("weight") * F.col("y")).alias("wpos")).first()
    print(f"TASK2 sample split={name} years={years} rows={r['rows']} positives={int(r['pos'])} "
          f"weighted_rows={r['w']:.0f} weighted_base_rate={r['wpos'] / r['w']:.4f}", flush=True)
    per = (s.groupBy("theta").agg(F.count(F.lit(1)).alias("rows"), F.sum("y").alias("positives"),
                                  (F.sum(F.col("weight") * F.col("y")) / F.sum("weight")).alias("weighted_base_rate"))
           .orderBy("theta").toPandas())
    write_csv_local(per, f"{name}_sample_by_theta", "task2")
    return out


def main():
    plan = ep.load_frozen_plan()
    spark = get_spark("p2-task2-sample", "task2_sample")
    grid = plan["threshold_grid_mm"]
    train_y, val_y = plan["split"]["train_years"], plan["split"]["validation_years"]
    b = labelled_base(spark, train_y + val_y)
    ep.guard_no_test_rows(b, plan)
    isl = spark.read.parquet(ISLAND_PARQUET)
    st = station_table(spark, b.filter(F.col("year").isin(train_y)))
    st.write.mode("overwrite").parquet(f"{T2}/station_features")
    pairs = neighbour_pairs(spark)
    write_csv_local(pairs.toPandas(), "neighbour_pairs", "task2")
    nb_src = neighbour_source(spark, train_y + val_y)
    for split, years in (("train", train_y), ("validation", val_y)):
        build_split(spark, plan, b, split, years, st, pairs, nb_src, isl)

    # ------------------------------------------------------------- baselines from the full training years
    tr = b.filter(F.col("year").isin(train_y))
    ts = F.timestamp_seconds(F.col("slot") * 300)
    tr = tr.withColumn("month", F.month(ts)).withColumn("hour", F.hour(ts))
    ag = [F.count(F.lit(1)).alias("n")] + [F.sum((F.col("f30") > t).cast("double")).alias(f"k{i}")
                                            for i, t in enumerate(grid)]
    stack = "stack({}, {}) as (theta, k)".format(len(grid), ", ".join(f"{float(t)}D, k{i}" for i, t in enumerate(grid)))
    full = tr.groupBy("station_id", "month", "hour").agg(*ag).selectExpr("station_id", "month", "hour", "n", stack)
    pooled = (tr.groupBy("month", "hour").agg(*ag).selectExpr("month", "hour", "n as n_pool", stack)
                .withColumn("p_pool", F.col("k") / F.col("n_pool")).drop("k"))
    clim = (full.join(pooled, ["month", "hour", "theta"])
                .withColumn("p_clim", (F.col("k") + CLIM_PSEUDO * F.col("p_pool")) / (F.col("n") + CLIM_PSEUDO))
                .select("station_id", "month", "hour", "theta", "p_clim"))
    clim.write.mode("overwrite").parquet(f"{T2}/baseline_climatology")
    pooled.select("month", "hour", "theta", F.col("p_pool").alias("p_clim_pool")).write.mode("overwrite").parquet(
        f"{T2}/baseline_climatology_pooled")
    pers_rows = []
    for t in grid:
        r = tr.groupBy((F.col("r30") > t).alias("past_gt")).agg(
            F.count(F.lit(1)).alias("n"), F.sum((F.col("f30") > t).cast("double")).alias("k")).collect()
        for x in r:
            pers_rows.append((float(t), bool(x["past_gt"]), int(x["n"]), float(x["k"]), float(x["k"] / x["n"])))
    pers = spark.createDataFrame(pers_rows, ["theta", "past_gt", "n", "k", "p_persist"])
    pers.write.mode("overwrite").parquet(f"{T2}/baseline_persistence")
    write_csv_local(pers.toPandas(), "baseline_persistence", "task2")
    print(f"TASK2 baselines written: climatology rows={spark.read.parquet(f'{T2}/baseline_climatology').count()} "
          f"persistence rows={len(pers_rows)}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
