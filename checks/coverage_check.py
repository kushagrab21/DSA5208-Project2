"""Coverage per year (print only; not proven to bind). Carries E5: years are compared per reported station-day.

  stations         stations with at least one cleaned reading that year
  station_days     (station, day) pairs with at least one reading
  full_days        station-days with >= 90% of the 288 slots
  slot_share       reported slots / (stations x slots in the year)
Writes outputs/coverage/coverage_by_year.csv for the report and Task 1.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import CLEAN_PARQUET, SLOTS_PER_DAY, get_spark, write_csv_local  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main():
    spark = get_spark("p2-coverage", "coverage_check")
    c = spark.read.parquet(CLEAN_PARQUET).withColumn("day", F.to_date("ts"))
    sd = c.groupBy("year", "station_id", "day").agg(F.count(F.lit(1)).alias("n"))
    per = (sd.groupBy("year").agg(
        F.countDistinct("station_id").alias("stations"),
        F.count(F.lit(1)).alias("station_days"),
        F.sum((F.col("n") >= 0.9 * SLOTS_PER_DAY).cast("long")).alias("full_days"),
        F.sum("n").alias("slots"),
        F.countDistinct("day").alias("days_with_data"))
        .orderBy("year").toPandas())
    per["days_in_year"] = [366 if y % 4 == 0 else 365 for y in per.year]
    per["slot_share"] = per.slots / (per.stations * per.days_in_year * SLOTS_PER_DAY)
    per["full_day_share"] = per.full_days / per.station_days
    for r in per.itertuples():
        print(f"COVERAGE year={r.year} stations={r.stations} station_days={r.station_days} "
              f"full_days={r.full_days} slots={r.slots} slot_share={r.slot_share:.3f} days_with_data={r.days_with_data}")
    write_csv_local(per, "coverage_by_year", "coverage")
    spark.stop()


if __name__ == "__main__":
    main()
