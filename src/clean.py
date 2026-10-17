"""Stage 2: clean the raw Parquet into one row per (station, 5-minute slot) and write the cleaning ledger.

Rules are applied in order; each removes rows from what the previous rule left, so
  rows_in - sum(removed) = rows_out
holds exactly, and checks/clean_ledger.py verifies it against the Parquet files themselves.

  R1 corrupt      line could not be parsed into the schema (_corrupt_record set)
  R2 missing      station_id, timestamp or reading_value is null
  R3 type_unit    reading_type is not "... 5 Minute Total ..." or unit is not "mm"
  R4 bad_ts       timestamp does not parse as ISO 8601 with offset
  R5 year_mismatch  SGT year of the timestamp differs from the file's year
  R6 negative     reading_value < 0
  R7 implausible  reading_value > 50 mm in 5 minutes (600 mm/h; the Singapore 1-hour record is far below)
  A1 snap         (adjustment, not removal) timestamps off the 5-minute grid are moved to the nearest slot
  R8 duplicate    more than one reading for the same (station, slot) after snapping; keep the latest update

Output columns: station_id, slot (epoch seconds / 300, slot END), ts, rain_mm, year.
Also writes the station table (modal coordinates per station) and outputs/ledger/clean_ledger.json.
"""
import json
import os

import common  # noqa: F401
from common import CLEAN_PARQUET, OUT_ROOT, RAW_PARQUET, SLOT_SECONDS, STATIONS_PARQUET, get_spark
from pyspark.sql import Window
from pyspark.sql import functions as F

IMPLAUSIBLE_MM = 50.0
TS_FMT = "yyyy-MM-dd'T'HH:mm:ssXXX"


def main():
    spark = get_spark("p2-clean", "clean")
    raw = spark.read.parquet(RAW_PARQUET)
    ledger = []
    cur = raw.cache()
    n0 = cur.count()

    def apply(name, keep_cond, why):
        nonlocal cur
        before = cur.count()
        nxt = cur.filter(keep_cond)
        after = nxt.count()
        ledger.append({"rule": name, "removed": before - after, "of": before, "why": why})
        print(f"CLEAN rule={name} removed={before - after} of {before}", flush=True)
        cur = nxt

    apply("corrupt", F.col("_corrupt_record").isNull(), "line not parseable with the explicit schema")
    apply("missing", F.col("station_id").isNotNull() & F.col("timestamp").isNotNull() & F.col("reading_value").isNotNull(),
          "station, timestamp or value missing")
    apply("type_unit", F.col("reading_type").contains("5 Minute Total") & (F.col("reading_unit") == "mm"),
          "not a 5-minute rainfall total in mm")
    cur = cur.withColumn("ts_raw", F.try_to_timestamp(F.col("timestamp"), F.lit(TS_FMT)))
    apply("bad_ts", F.col("ts_raw").isNotNull(), "timestamp not ISO 8601 with offset")
    apply("year_mismatch", F.year("ts_raw") == F.col("year"), "timestamp outside the file's calendar year (SGT)")
    apply("negative", F.col("reading_value") >= 0, "negative rainfall")
    apply("implausible", F.col("reading_value") <= IMPLAUSIBLE_MM, f"more than {IMPLAUSIBLE_MM:g} mm in 5 minutes")

    # A1: snap to the 5-minute grid (e.g. 08:04:59 -> 08:05:00); counted, nothing removed
    epoch = F.unix_timestamp("ts_raw")
    cur = cur.withColumn("slot", F.floor((epoch + SLOT_SECONDS // 2) / SLOT_SECONDS).cast("long"))
    cur = cur.withColumn("snapped", (epoch % SLOT_SECONDS) != 0)
    stats = cur.agg(F.count(F.lit(1)).alias("n"), F.sum(F.col("snapped").cast("long")).alias("snapped"),
                    F.max(F.abs(epoch - F.col("slot") * SLOT_SECONDS)).alias("max_shift")).first()
    ledger.append({"rule": "snap", "adjusted": stats["snapped"], "of": stats["n"], "removed": 0,
                   "max_shift_seconds": stats["max_shift"], "why": "timestamp moved to the nearest 5-minute slot end"})
    print(f"CLEAN rule=snap adjusted={stats['snapped']} of {stats['n']} (removed=0, max shift {stats['max_shift']} s)",
          flush=True)
    # snapping must not move a reading across a year boundary
    cur = cur.withColumn("ts", F.timestamp_seconds(F.col("slot") * SLOT_SECONDS))
    apply("snap_year_cross", F.year("ts") == F.col("year"), "snapping moved the slot into another year")

    # R8: duplicates on (station, slot); keep the most recent update, then the larger value
    w = Window.partitionBy("station_id", "slot").orderBy(F.col("update_timestamp").desc(), F.col("reading_value").desc())
    cur = (cur.withColumn("rn", F.row_number().over(w))
              .withColumn("nvals", F.size(F.collect_set("reading_value").over(Window.partitionBy("station_id", "slot")))))
    cur = cur.persist()
    conflicts = cur.filter((F.col("rn") == 1) & (F.col("nvals") > 1)).count()
    apply("duplicate", F.col("rn") == 1, "second or later reading for the same station and slot")
    ledger[-1]["slots_with_conflicting_values"] = conflicts
    print(f"CLEAN note=duplicate slots_with_conflicting_values={conflicts}", flush=True)

    clean = cur.select("station_id", "slot", "ts", F.col("reading_value").alias("rain_mm"), "year",
                       "station_name", "location_longitude", "location_latitude")
    clean.select("station_id", "slot", "ts", "rain_mm", "year").repartition("year").sortWithinPartitions(
        "station_id", "slot").write.mode("overwrite").partitionBy("year").parquet(CLEAN_PARQUET)
    n_out = spark.read.parquet(CLEAN_PARQUET).count()

    # station table: modal coordinates, latest name, years active
    coords = (clean.groupBy("station_id", "location_latitude", "location_longitude").count()
              .withColumn("r", F.row_number().over(Window.partitionBy("station_id").orderBy(F.desc("count"))))
              .filter("r = 1").select("station_id", F.col("location_latitude").alias("lat"),
                                      F.col("location_longitude").alias("lon")))
    ncoord = clean.groupBy("station_id").agg(
        F.countDistinct("location_latitude", "location_longitude").alias("n_coords"),
        F.max_by("station_name", "slot").alias("station_name"),
        F.min("year").alias("first_year"), F.max("year").alias("last_year"),
        F.countDistinct("year").alias("n_years"), F.count(F.lit(1)).alias("rows"))
    st = coords.join(ncoord, "station_id")
    st.coalesce(1).write.mode("overwrite").parquet(STATIONS_PARQUET)
    stp = spark.read.parquet(STATIONS_PARQUET).toPandas().sort_values("station_id")
    d = os.path.join(OUT_ROOT, "ledger")
    os.makedirs(d, exist_ok=True)
    stp.to_csv(os.path.join(d, "stations.csv"), index=False)
    out = {"rows_in": n0, "rows_out": n_out, "rules": ledger, "implausible_mm": IMPLAUSIBLE_MM,
           "stations": int(len(stp)), "stations_with_moved_coords": int((stp.n_coords > 1).sum())}
    json.dump(out, open(os.path.join(d, "clean_ledger.json"), "w"), indent=1)
    print(f"CLEAN rows_in={n0} rows_out={n_out} stations={len(stp)} "
          f"stations_with_more_than_one_coordinate={out['stations_with_moved_coords']}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
