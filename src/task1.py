"""Task 1: weather patterns within a year and across 2017-2024 (Spark DataFrame API).

Every comparison between years or months is made per reported slot (or per complete station-day), never as a raw
sum over a station set that changes from year to year (54 to 80 stations; 2018 has 3 days of August data).
Rates are mean 5-minute rain x 288 = mm per station-day. Annual values are month-balanced: the sum over the 12
months of (monthly rate x days in month), so a missing week in a wet month cannot lower the year.

Writes small aggregated tables to outputs/task1/*.csv; src/plot_task1.py draws the figures from them.
"""
import common  # noqa: F401
from common import CLEAN_PARQUET, SLOTS_PER_DAY, STATIONS_PARQUET, get_spark, write_csv_local
from pyspark.sql import Window
from pyspark.sql import functions as F

SUB = "task1"
HEAVY_HOUR_MM = 20.0     # station-hour total counted as heavy (MSS "heavy rain" scale starts near this)
VERY_HEAVY_HOUR_MM = 40.0
WET_DAY_MM = 1.0
HEAVY_DAY_MM = 50.0
INTENSE_5MIN_MM = 2.5    # 30 mm/h equivalent


def season_col(m):
    return (F.when(m.isin(12, 1, 2, 3), "NE monsoon (Dec-Mar)")
             .when(m.isin(4, 5), "Inter-monsoon I (Apr-May)")
             .when(m.isin(6, 7, 8, 9), "SW monsoon (Jun-Sep)")
             .otherwise("Inter-monsoon II (Oct-Nov)"))


def save(df, name):
    pdf = df.toPandas()
    write_csv_local(pdf, name, SUB)
    print(f"TASK1 table={name} rows={len(pdf)}", flush=True)
    return pdf


def main():
    spark = get_spark("p2-task1", "task1")
    c = (spark.read.parquet(CLEAN_PARQUET)
         .withColumn("month", F.month("ts")).withColumn("hour", F.hour("ts"))
         .withColumn("day", F.to_date("ts")).withColumn("doy", F.dayofyear("ts"))
         .withColumn("wet", (F.col("rain_mm") > 0).cast("int"))
         .withColumn("intense", (F.col("rain_mm") >= INTENSE_5MIN_MM).cast("int")))
    c = c.withColumn("season", season_col(F.col("month")))
    # the reading at hh:00 covers hh-1:55..hh:00; assign each slot to the hour in which its 5 minutes START
    c = c.withColumn("hour_start", F.hour(F.col("ts") - F.expr("INTERVAL 5 MINUTES")))
    c = c.withColumn("day_start", F.to_date(F.col("ts") - F.expr("INTERVAL 5 MINUTES"))).cache()

    agg = [F.count(F.lit(1)).alias("slots"), F.sum("wet").alias("wet_slots"), F.sum("rain_mm").alias("rain_sum"),
           F.sum(F.when(F.col("intense") == 1, F.col("rain_mm")).otherwise(0)).alias("intense_rain_sum"),
           F.sum("intense").alias("intense_slots"), F.countDistinct("station_id").alias("stations"),
           F.countDistinct("day").alias("days_with_data")]

    # within-year and across-year: month x year
    save(c.groupBy("year", "month").agg(*agg).orderBy("year", "month"), "monthly_by_year")
    # robustness: a fixed panel of stations that report >= 50% of slots in EVERY year (network changes removed)
    sy = c.groupBy("station_id", "year").agg(F.count(F.lit(1)).alias("n"))
    full_year = 365 * SLOTS_PER_DAY
    panel = (sy.filter(F.col("n") >= 0.5 * full_year).groupBy("station_id").agg(F.countDistinct("year").alias("ny"))
               .filter(F.col("ny") == 8).select("station_id"))
    print(f"TASK1 panel stations (>=50% of slots in all 8 years): {panel.count()}", flush=True)
    save(c.join(F.broadcast(panel), "station_id").groupBy("year", "month").agg(*agg).orderBy("year", "month"),
         "monthly_by_year_panel")
    # positive readings that are multiples of 0.2 mm, by station and year (instrument/processing change)
    save(c.filter("rain_mm > 0").groupBy("station_id", "year").agg(
        F.count(F.lit(1)).alias("wet_slots"),
        F.sum((F.abs(F.col("rain_mm") * 5 - F.round(F.col("rain_mm") * 5)) < 1e-6).cast("long")).alias("mult_0p2")),
        "resolution_by_station_year")
    # diurnal cycle by season (all years) and by year (all seasons)
    save(c.groupBy("season", "hour_start").agg(*agg[:5]).orderBy("season", "hour_start"), "diurnal_by_season")
    save(c.groupBy("year", "hour_start").agg(*agg[:5]).orderBy("year", "hour_start"), "diurnal_by_year")
    # season x year
    save(c.groupBy("year", "season").agg(*agg).orderBy("year", "season"), "season_by_year")
    # network mean per calendar day (for the day-of-year curve)
    save(c.groupBy("year", "day", "doy").agg(*agg[:3]).orderBy("day"), "daily_network")
    # data resolution: share of positive readings that are multiples of 0.2 mm (tipping bucket)
    pos = c.filter("rain_mm > 0")
    save(pos.groupBy("year").agg(
        F.count(F.lit(1)).alias("wet_slots"),
        F.sum((F.abs(F.col("rain_mm") * 5 - F.round(F.col("rain_mm") * 5)) < 1e-6).cast("long")).alias("mult_0p2"),
        F.min("rain_mm").alias("min_positive"),
        F.expr("percentile_approx(rain_mm, 0.5)").alias("p50_wet"),
        F.expr("percentile_approx(rain_mm, 0.99)").alias("p99_wet"),
        F.expr("percentile_approx(rain_mm, 0.999)").alias("p999_wet"),
        F.max("rain_mm").alias("max_5min")).orderBy("year"), "resolution_by_year")
    # intensity histogram of wet 5-minute readings by year (rain volume by intensity class)
    bins = [0.0, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 1e9]
    expr = None
    for lo, hi in zip(bins[:-1], bins[1:]):
        cond = (F.col("rain_mm") > lo) & (F.col("rain_mm") <= hi)
        lab = F.lit(f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}")
        expr = F.when(cond, lab) if expr is None else expr.when(cond, lab)
    save(pos.withColumn("cls", expr).groupBy("year", "cls").agg(
        F.count(F.lit(1)).alias("slots"), F.sum("rain_mm").alias("rain_sum")).orderBy("year", "cls"),
        "intensity_classes_by_year")

    # hourly station totals (complete hours only: all 12 slots reported)
    hr = (c.groupBy("station_id", "year", "day_start", "hour_start")
          .agg(F.count(F.lit(1)).alias("n"), F.sum("rain_mm").alias("mm"))
          .filter("n = 12").cache())
    save(hr.groupBy("year").agg(
        F.count(F.lit(1)).alias("complete_station_hours"),
        F.sum((F.col("mm") >= HEAVY_HOUR_MM).cast("long")).alias("hours_ge_20"),
        F.sum((F.col("mm") >= VERY_HEAVY_HOUR_MM).cast("long")).alias("hours_ge_40"),
        F.max("mm").alias("max_hour_mm"),
        F.expr("percentile_approx(case when mm > 0 then mm end, 0.99)").alias("p99_wet_hour")).orderBy("year"),
        "hourly_extremes_by_year")
    save(hr.groupBy("station_id", "year").agg(F.max("mm").alias("annual_max_hour_mm"),
                                               F.count(F.lit(1)).alias("complete_hours")),
         "station_annual_max_hour")
    save(hr.withColumn("month", F.month("day_start")).groupBy("month").agg(
        F.count(F.lit(1)).alias("complete_station_hours"),
        F.sum((F.col("mm") >= HEAVY_HOUR_MM).cast("long")).alias("hours_ge_20")).orderBy("month"),
        "hourly_extremes_by_month")

    # daily station totals on complete days (>= 90% of slots); wet days, heavy days, wet spells
    dd = (c.groupBy("station_id", "year", "day")
          .agg(F.count(F.lit(1)).alias("n"), F.sum("rain_mm").alias("mm"))
          .filter(F.col("n") >= 0.9 * SLOTS_PER_DAY)
          .withColumn("mm_scaled", F.col("mm") * SLOTS_PER_DAY / F.col("n")).cache())
    save(dd.groupBy("year").agg(
        F.count(F.lit(1)).alias("full_station_days"),
        F.sum((F.col("mm_scaled") >= WET_DAY_MM).cast("long")).alias("wet_days"),
        F.sum((F.col("mm_scaled") >= HEAVY_DAY_MM).cast("long")).alias("heavy_days"),
        F.max("mm_scaled").alias("max_day_mm"),
        F.expr("percentile_approx(mm_scaled, 0.99)").alias("p99_day")).orderBy("year"), "daily_extremes_by_year")
    # wet spells: runs of consecutive complete wet days at one station (a gap or dry day ends the run)
    w = Window.partitionBy("station_id").orderBy("day")
    sp = (dd.withColumn("is_wet", F.col("mm_scaled") >= WET_DAY_MM)
          .withColumn("prev_day", F.lag("day").over(w)).withColumn("prev_wet", F.lag("is_wet").over(w))
          .withColumn("brk", (~F.col("is_wet")) | F.col("prev_wet").isNull() | (~F.col("prev_wet"))
                      | (F.datediff("day", "prev_day") != 1))
          .withColumn("grp", F.sum(F.col("brk").cast("int")).over(w.rowsBetween(Window.unboundedPreceding, 0)))
          .filter("is_wet").groupBy("station_id", "grp")
          .agg(F.count(F.lit(1)).alias("len"), F.min("year").alias("year")))
    save(sp.groupBy("year").agg(F.count(F.lit(1)).alias("spells"), F.avg("len").alias("mean_len"),
                                F.max("len").alias("max_len"),
                                F.sum((F.col("len") >= 5).cast("long")).alias("spells_ge5")).orderBy("year"),
         "wet_spells_by_year")

    # spatial: per station x season, with modal coordinates
    st = spark.read.parquet(STATIONS_PARQUET)
    save(c.groupBy("station_id", "season").agg(*agg[:3]).join(st, "station_id"), "station_season")
    save(c.groupBy("station_id", "year").agg(*agg[:3]).join(st, "station_id"), "station_year")
    spark.stop()


if __name__ == "__main__":
    main()
