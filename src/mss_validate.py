"""External validation of the data.gov.sg 5-minute dataset against official MSS daily station records (weather.gov.sg).

MSS files are used ONLY to validate; nothing in Task 1 or Task 2 is replaced by MSS values.
Day convention: our daily total for date D sums the 5-minute readings whose interval starts on D (slot ends from
D 00:05 to D+1 00:00), i.e. local 00:00-24:00, as MSS reports daily totals.

  1. Agreement on complete days: days where our station reported all 288 slots and MSS has a daily total.
  2. What gaps cost: per station-month, official rain on days we have no reading at all (missing days) and on days
     we have only part of (partial days), against the official month total.
  3. Monthly rates for the matched stations: our coverage-normalised rate vs the official rate, by year-month, so that
     every Task 1 month affected by a gap can be named.
Writes outputs/mss/*.csv and prints MSS lines. Usage: python src/mss_validate.py
"""
import json
import os

import pandas as pd

import common  # noqa: F401
from common import CLEAN_PARQUET, OUT_ROOT, ROOT, SLOTS_PER_DAY, get_spark, write_csv_local
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

MSS_RAW = os.path.join(ROOT, "data", "mss", "raw")
SUB = "mss"
COLS = ["station_name", "y", "m", "d", "rain", "max30", "max60", "max120", "tmean", "tmax", "tmin", "wmean", "wmax"]


def main():
    # ANSI off for this job only: corr() over a station with a single complete day must give NULL, not an error
    spark = get_spark("p2-mss-validate", "mss_validate", {"spark.sql.ansi.enabled": "false"})
    schema = StructType([StructField(c, StringType()) for c in COLS])
    mss = (spark.read.schema(schema).option("header", True).csv(os.path.join(MSS_RAW, "DAILYDATA_*.csv"))
           .withColumn("file", F.input_file_name())
           .withColumn("station_id", F.regexp_extract("file", r"DAILYDATA_(S\d+)_\d{6}\.csv", 1))
           .withColumn("date", F.make_date(F.trim("y").cast("int"), F.trim("m").cast("int"), F.trim("d").cast("int")))
           .withColumn("mss_mm", F.expr("try_cast(trim(rain) AS DOUBLE)"))
           .select("station_id", "station_name", "date", "mss_mm")
           .filter(F.col("date").isNotNull() & (F.year("date").between(2017, 2024)))
           .dropDuplicates(["station_id", "date"]).cache())
    n_files = len([f for f in os.listdir(MSS_RAW) if f.endswith(".csv")])
    stations = [r[0] for r in mss.select("station_id").distinct().collect()]
    print(f"MSS files={n_files} stations={len(stations)} station_days={mss.count()} "
          f"with_daily_total={mss.filter(F.col('mss_mm').isNotNull()).count()}", flush=True)

    c = spark.read.parquet(CLEAN_PARQUET).filter(F.col("station_id").isin(stations))
    ours = (c.withColumn("date", F.to_date(F.col("ts") - F.expr("INTERVAL 5 MINUTES")))
            .groupBy("station_id", "date").agg(F.count(F.lit(1)).alias("n"), F.sum("rain_mm").alias("our_mm")))
    j = (mss.join(ours, ["station_id", "date"], "left").fillna({"n": 0})
         .withColumn("year", F.year("date")).withColumn("month", F.month("date")).cache())

    # 1. agreement on complete days
    full = j.filter((F.col("n") == SLOTS_PER_DAY) & F.col("mss_mm").isNotNull())
    full = full.withColumn("diff", F.col("our_mm") - F.col("mss_mm"))
    a = full.agg(
        F.count(F.lit(1)).alias("days"), F.countDistinct("station_id").alias("stations"),
        F.sum("mss_mm").alias("mss_total"), F.sum("our_mm").alias("our_total"),
        F.avg("diff").alias("mean_diff"), F.avg(F.abs("diff")).alias("mae"),
        F.corr("our_mm", "mss_mm").alias("corr"),
        F.avg((F.abs("diff") <= 0.2 + 1e-9).cast("double")).alias("within_0p2"),
        F.avg((F.abs("diff") <= F.greatest(F.lit(1.0), 0.05 * F.col("mss_mm"))).cast("double")).alias("within_1mm_or_5pct"),
        F.sum(((F.col("mss_mm") > 0) | (F.col("our_mm") > 0)).cast("long")).alias("wet_days"),
        F.sum(((F.col("mss_mm") >= 1) & (F.col("our_mm") >= 1)).cast("long")).alias("both_wet1"),
        F.sum(((F.col("mss_mm") >= 1) != (F.col("our_mm") >= 1)).cast("long")).alias("wet1_disagree"),
        F.max(F.abs("diff")).alias("max_abs_diff")).first().asDict()
    a["ratio_total"] = a["our_total"] / a["mss_total"]
    print("MSS AGREEMENT complete days: days={days} stations={stations} corr={corr:.4f} mean_diff={mean_diff:+.3f} mm "
          "MAE={mae:.3f} mm within_0.2mm={within_0p2:.3f} within_max(1mm,5%)={within_1mm_or_5pct:.3f} "
          "total_ratio={ratio_total:.4f} wet(>=1mm)_disagree={wet1_disagree} max_abs_diff={max_abs_diff:.1f}".format(**a))
    write_csv_local(full.select("station_id", "date", "n", "our_mm", "mss_mm", "diff").toPandas(),
                    "complete_day_pairs", SUB)
    per_st = (full.groupBy("station_id").agg(F.count(F.lit(1)).alias("days"), F.corr("our_mm", "mss_mm").alias("corr"),
                                             F.avg(F.abs("diff")).alias("mae"), F.sum("our_mm").alias("our_total"),
                                             F.sum("mss_mm").alias("mss_total"))
              .withColumn("ratio", F.col("our_total") / F.col("mss_total")).orderBy("ratio").toPandas())
    write_csv_local(per_st, "agreement_by_station", SUB)
    per_y = (full.groupBy("year").agg(F.count(F.lit(1)).alias("days"), F.corr("our_mm", "mss_mm").alias("corr"),
                                      F.avg(F.abs("diff")).alias("mae"),
                                      (F.sum("our_mm") / F.sum("mss_mm")).alias("ratio")).orderBy("year").toPandas())
    write_csv_local(per_y, "agreement_by_year", SUB)
    for r in per_y.itertuples():
        print(f"MSS AGREEMENT year={r.year} days={r.days} corr={r.corr:.4f} MAE={r.mae:.3f} ratio={r.ratio:.4f}")
    big = full.filter(F.abs("diff") > 20).count()
    print(f"MSS AGREEMENT complete days with |difference| > 20 mm: {big} of {a['days']}")
    worst = full.orderBy(F.desc(F.abs("diff"))).limit(8).toPandas()
    write_csv_local(worst, "largest_complete_day_differences", SUB)
    for r in worst.itertuples():
        print(f"    {r.station_id} {r.date}: ours {r.our_mm:.1f} mm vs official {r.mss_mm:.1f} mm (all 288 slots reported)")

    # 2. what gaps cost. Every MSS station-day is put in one class:
    #    complete / partial (our station reported 288 / 1-287 slots that day);
    #    network_outage (no station at all in our dataset reported that day);
    #    station_gap (our station has readings in that month but none that day, while the network reported);
    #    station_absent_month (our station has no reading in that whole month: network membership, not a gap).
    allc = spark.read.parquet(CLEAN_PARQUET).withColumn("date", F.to_date(F.col("ts") - F.expr("INTERVAL 5 MINUTES")))
    net_days = allc.select("date").distinct().withColumn("net_up", F.lit(True))
    active = (allc.filter(F.col("station_id").isin(stations))
              .select("station_id", F.year("date").alias("year"), F.month("date").alias("month")).distinct()
              .withColumn("active_month", F.lit(True)))
    k = (j.filter(F.col("mss_mm").isNotNull()).join(net_days, "date", "left")
         .join(active, ["station_id", "year", "month"], "left")
         .withColumn("cls", F.when(F.col("n") == SLOTS_PER_DAY, "complete").when(F.col("n") > 0, "partial")
                     .when(F.col("net_up").isNull(), "network_outage")
                     .when(F.col("active_month").isNotNull(), "station_gap").otherwise("station_absent_month")).cache())
    cls = k.groupBy("cls").agg(F.count(F.lit(1)).alias("station_days"), F.sum("mss_mm").alias("official_mm")).toPandas()
    tot = cls.official_mm.sum()
    cls["share_of_official_rain"] = cls.official_mm / tot
    write_csv_local(cls, "official_rain_by_day_class", SUB)
    for r in cls.sort_values("official_mm", ascending=False).itertuples():
        print(f"MSS DAYCLASS {r.cls:<21} station_days={r.station_days:>6} official_rain={r.official_mm:>9.0f} mm "
              f"({100 * r.share_of_official_rain:.1f}%)")
    sm = (k.filter(F.col("active_month").isNotNull()).groupBy("station_id", "year", "month").agg(
        F.sum("mss_mm").alias("mss_month_mm"), F.sum(F.coalesce("our_mm", F.lit(0.0))).alias("our_month_mm"),
        F.sum(F.when(F.col("cls") == "network_outage", F.col("mss_mm")).otherwise(0)).alias("mss_on_network_outage"),
        F.sum(F.when(F.col("cls") == "station_gap", F.col("mss_mm")).otherwise(0)).alias("mss_on_station_gap"),
        F.sum(F.when(F.col("cls") == "partial", F.col("mss_mm")).otherwise(0)).alias("mss_on_partial_days"),
        F.sum(F.col("cls").isin("network_outage", "station_gap").cast("int")).alias("missing_days"),
        F.sum((F.col("cls") == "partial").cast("int")).alias("partial_days"))
        .withColumn("mss_on_missing_days", F.col("mss_on_network_outage") + F.col("mss_on_station_gap"))
        .withColumn("missed_share", F.when(F.col("mss_month_mm") > 0, F.col("mss_on_missing_days") / F.col("mss_month_mm")))
        .orderBy(F.desc("mss_on_missing_days")).toPandas())
    write_csv_local(sm, "gap_cost_station_month", SUB)
    t2 = sm.mss_month_mm.sum()
    print(f"MSS GAPS active station-months={len(sm)} official_rain={t2:.0f} mm; on days we lack: "
          f"network outages {sm.mss_on_network_outage.sum():.0f} mm ({100 * sm.mss_on_network_outage.sum() / t2:.1f}%), "
          f"single-station gaps {sm.mss_on_station_gap.sum():.0f} mm ({100 * sm.mss_on_station_gap.sum() / t2:.1f}%); "
          f"on partial days {sm.mss_on_partial_days.sum():.0f} mm ({100 * sm.mss_on_partial_days.sum() / t2:.1f}%)")
    print("MSS GAPS worst active station-months by official rain on days we lack:")
    for r in sm.head(15).itertuples():
        print(f"    {r.station_id} {r.year}-{r.month:02d}: official {r.mss_month_mm:.1f} mm, ours {r.our_month_mm:.1f} mm; "
              f"{r.missing_days} days missing hold {r.mss_on_missing_days:.1f} mm ({100 * r.missed_share:.0f}%; "
              f"network outage {r.mss_on_network_outage:.1f})")
    ym = (sm.groupby(["year", "month"]).agg(stations=("station_id", "nunique"), mss_mm=("mss_month_mm", "sum"),
                                             missed_mm=("mss_on_missing_days", "sum"),
                                             outage_mm=("mss_on_network_outage", "sum"),
                                             partial_mm=("mss_on_partial_days", "sum")).reset_index())
    ym["missed_share"] = ym.missed_mm / ym.mss_mm
    write_csv_local(ym, "gap_cost_year_month", SUB)
    # spans of consecutive network-outage days, with the last reading before and the first after (from our data)
    od = sorted(r[0] for r in allc.select("date").distinct().collect())
    spans, prev = [], None
    for d in od:
        if prev is not None and (d - prev).days > 1:
            spans.append((prev, d))
        prev = d
    ts_bounds = []
    for last_day, next_day in spans:
        lo = allc.filter(F.col("date") == last_day).agg(F.max("ts")).first()[0]
        hi = allc.filter(F.col("date") == next_day).agg(F.min("ts")).first()[0]
        ts_bounds.append({"last_reading": str(lo), "first_after": str(hi), "days_without_any_station": (next_day - last_day).days - 1})
    write_csv_local(pd.DataFrame(ts_bounds), "network_outage_spans", SUB)
    outage_days = (k.filter(F.col("cls") == "network_outage").groupBy("date")
                   .agg(F.avg("mss_mm").alias("official_mean_mm"), F.max("mss_mm").alias("official_max_mm"),
                        F.count(F.lit(1)).alias("stations")).orderBy(F.desc("official_mean_mm")).toPandas())
    write_csv_local(outage_days, "network_outage_days", SUB)
    print(f"MSS OUTAGES network-wide outage days with an official record: {len(outage_days)}; wettest:")
    for r in outage_days.head(6).itertuples():
        print(f"    {r.date}: official mean {r.official_mean_mm:.1f} mm, max {r.official_max_mm:.1f} mm over {r.stations} stations")

    # 3. monthly rates on the SAME station-months (stations active in our data that month): ours vs official
    act = active.select("station_id", "year", "month")
    have = k.select("station_id", "year", "month").distinct()          # station-months with an MSS file
    our_m = (c.withColumn("year", F.year("ts")).withColumn("month", F.month("ts"))
             .join(have, ["station_id", "year", "month"]).groupBy("year", "month")
             .agg(F.sum("rain_mm").alias("rain_sum"), F.count(F.lit(1)).alias("slots"))
             .withColumn("our_rate", F.col("rain_sum") / F.col("slots") * SLOTS_PER_DAY).toPandas())
    mss_m = (k.join(act, ["station_id", "year", "month"]).groupBy("year", "month")
             .agg(F.avg("mss_mm").alias("mss_rate"), F.count(F.lit(1)).alias("mss_station_days")).toPandas())
    mr = our_m.merge(mss_m, on=["year", "month"]).merge(ym[["year", "month", "missed_share", "outage_mm"]],
                                                         on=["year", "month"])
    mr["ratio"] = mr.our_rate / mr.mss_rate
    mr = mr.sort_values(["year", "month"])
    write_csv_local(mr, "monthly_rate_ours_vs_mss", SUB)
    print(f"MSS MONTHLY ours/official rate, same station-months: median ratio {mr.ratio.median():.3f}; "
          f"outside [0.85, 1.15]:")
    for r in mr[(mr.ratio < 0.85) | (mr.ratio > 1.15)].sort_values("ratio").itertuples():
        print(f"    {r.year}-{r.month:02d}: ours {r.our_rate:.2f} vs official {r.mss_rate:.2f} mm/day "
              f"(ratio {r.ratio:.2f}; official rain on days we lack {100 * r.missed_share:.0f}%)")
    json.dump({k: (float(v) if v is not None else None) for k, v in a.items()},
              open(os.path.join(OUT_ROOT, SUB, "agreement_summary.json"), "w"), indent=1)
    spark.stop()


if __name__ == "__main__":
    main()
