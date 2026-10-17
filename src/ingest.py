"""Stage 1: read the eight annual CSVs with an explicit schema and write them, unchanged, to Parquet partitioned by year.

Nothing is filtered here; malformed lines are kept with their raw text in `_corrupt_record` so that the
cleaning stage (src/clean.py) can count them. Usage: python src/ingest.py [year ...]
"""
import sys

import common  # noqa: F401  (sets sys.path)
from common import RAW_CSV, RAW_PARQUET, YEARS, get_spark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

RAW_SCHEMA = StructType([
    StructField("date", StringType()),
    StructField("timestamp", StringType()),
    StructField("update_timestamp", StringType()),
    StructField("station_id", StringType()),
    StructField("station_name", StringType()),
    StructField("station_device_id", StringType()),
    StructField("location_longitude", DoubleType()),
    StructField("location_latitude", DoubleType()),
    StructField("reading_update_timestamp", StringType()),
    StructField("reading_value", DoubleType()),
    StructField("reading_type", StringType()),
    StructField("reading_unit", StringType()),
    StructField("_corrupt_record", StringType()),
])


def main(years):
    spark = get_spark("p2-ingest", "ingest")
    for y in years:
        df = (spark.read.schema(RAW_SCHEMA)
              .option("header", True)
              .option("mode", "PERMISSIVE")
              .option("columnNameOfCorruptRecord", "_corrupt_record")
              .csv(RAW_CSV.format(year=y))
              .withColumn("year", F.lit(y)))
        (df.write.mode("overwrite").option("partitionOverwriteMode", "dynamic")
           .partitionBy("year").parquet(RAW_PARQUET))
        print(f"INGEST wrote year={y}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or YEARS)
