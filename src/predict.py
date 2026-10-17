"""predict(station, prediction time, threshold) -> P(station's rain in (t, t+30 min] > threshold), Models A and B.

Model A (strict) uses only the station, the time and the threshold.
Model B (nowcast) has the same interface; internally it looks up the station's own readings and its six nearest
neighbours' readings with slot end <= t, plus the island-wide state at t, from the cleaned data. If the station
has no reading at t, B's inputs are unavailable and the answer falls back to Model A (reported in `used`).

Usage:
  python src/predict.py --station S24 --time 2023-11-14T15:30 --threshold 0.5 1 5
  python src/predict.py --station S24 --time 2023-11-14T15:30 --threshold 2 --model A
Times are SGT; t is moved back to the latest 5-minute slot end at or before it.
"""
import argparse
import datetime as _dt
import math
import os
import sys

import common  # noqa: F401
from common import DATA_ROOT, PARQUET_ROOT, FEATURES_PARQUET, ROOT, STATIONS_PARQUET, get_spark
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(ROOT, "checks"))
import eval_plan as ep  # noqa: E402

T2 = PARQUET_ROOT + "/task2"
ISLAND_PARQUET = PARQUET_ROOT + "/island_state"
MODELS = os.environ.get("P2_MODELS_ROOT", os.path.join(ROOT, "models"))
SGT = _dt.timezone(_dt.timedelta(hours=8))


def station_features(spark, plan):
    """Station coordinates and training-years wet rate (same definition as src/task2_sample.py); cached."""
    p = f"{T2}/station_features"
    try:
        return spark.read.parquet(p)
    except Exception:
        pass
    tr = spark.read.parquet(FEATURES_PARQUET).filter(F.col("year").isin(plan["split"]["train_years"]) & F.col("label_ok"))
    wet = tr.groupBy("station_id").agg(F.avg((F.col("r5") > 0).cast("double")).alias("station_train_wet_rate"))
    overall = tr.agg(F.avg((F.col("r5") > 0).cast("double"))).first()[0]
    st = (spark.read.parquet(STATIONS_PARQUET).select("station_id", F.col("lat").alias("station_lat"),
                                                      F.col("lon").alias("station_lon"))
          .join(wet, "station_id", "left").fillna({"station_train_wet_rate": overall}))
    st.write.mode("overwrite").parquet(p)
    return spark.read.parquet(p)


def build_rows(spark, plan, station, t, thetas):
    slot = int(t.timestamp() // 300)
    st = station_features(spark, plan).filter(F.col("station_id") == station)
    if st.count() == 0:
        raise SystemExit(f"unknown station {station}")
    ts = _dt.datetime.fromtimestamp(slot * 300, SGT)
    tod = (ts.hour * 60 + ts.minute) / 1440.0
    doy = (ts.timetuple().tm_yday - 1) / 365.25
    m = ts.month
    season = 0 if m in (12, 1, 2, 3) else 1 if m in (4, 5) else 2 if m in (6, 7, 8, 9) else 3
    base = spark.createDataFrame([(station, slot, ts.hour, m, season, math.sin(2 * math.pi * tod),
                                   math.cos(2 * math.pi * tod), math.sin(2 * math.pi * doy),
                                   math.cos(2 * math.pi * doy))],
                                 ["station_id", "slot", "hour", "month", "season", "hour_sin", "hour_cos",
                                  "doy_sin", "doy_cos"]).join(st, "station_id")
    fb = spark.read.parquet(FEATURES_PARQUET).filter(F.col("year") == ts.year)
    own = fb.filter((F.col("station_id") == station) & (F.col("slot") == slot)).select(
        "station_id", "slot", "r5", "r15", "r30", "r60", "r120", "r24h", "n120", "mins_since_wet")
    has_past = own.count() > 0
    pairs = spark.read.csv(os.path.join(ROOT, "outputs", "task2", "neighbour_pairs.csv"), header=True,
                           inferSchema=True).filter(F.col("station_id") == station)
    nb = (pairs.join(fb.select(F.col("station_id").alias("nb_id"), "slot", F.col("r5").alias("nb_r5"),
                               F.col("r30").alias("nb_r30")).filter(F.col("slot") == slot), "nb_id")
          .groupBy("station_id").agg(F.max("nb_r30").alias("nb_max_r30"), F.avg("nb_r30").alias("nb_mean_r30"),
                                     F.max("nb_r5").alias("nb_max_r5"),
                                     F.avg((F.col("nb_r5") > 0).cast("double")).alias("nb_wet_now_frac"),
                                     F.count(F.lit(1)).alias("nb_n")))
    isl = spark.read.parquet(ISLAND_PARQUET).filter(F.col("slot") == slot)
    rows = (base.join(own, ["station_id", "slot"], "left").join(nb, "station_id", "left").join(isl, "slot", "left")
                .fillna({"nb_max_r30": 0.0, "nb_mean_r30": 0.0, "nb_max_r5": 0.0, "nb_wet_now_frac": 0.0, "nb_n": 0}))
    th = spark.createDataFrame([(float(x),) for x in thetas], ["theta"])
    return rows.crossJoin(th), has_past, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--time", required=True, help="SGT, e.g. 2023-11-14T15:30")
    ap.add_argument("--threshold", type=float, nargs="+", required=True)
    ap.add_argument("--model", choices=["A", "B", "both"], default="both")
    a = ap.parse_args()
    plan = ep.load_frozen_plan()
    t = _dt.datetime.fromisoformat(a.time)
    t = t.replace(tzinfo=SGT) if t.tzinfo is None else t
    spark = get_spark("p2-predict", "predict")
    rows, has_past, ts = build_rows(spark, plan, a.station, t, a.threshold)
    for fam in (["A", "B"] if a.model == "both" else [a.model]):
        used = fam if (fam == "A" or has_past) else "A (no reading at t; B falls back)"
        model = PipelineModel.load(os.path.join(MODELS, f"model_{used[0]}"))
        out = model.transform(rows).withColumn("p", vector_to_array("probability")[1]).select("theta", "p", "r30")
        for r in out.orderBy("theta").collect():
            print(f"PREDICT model={fam} used={used} station={a.station} t={ts.isoformat()} theta={r['theta']:g} mm "
                  f"P(rain in next 30 min > theta)={r['p']:.4f}" + (f" (last 30 min: {r['r30']:.1f} mm)"
                                                                   if r['r30'] is not None and fam == 'B' else ""))
    spark.stop()


if __name__ == "__main__":
    main()
