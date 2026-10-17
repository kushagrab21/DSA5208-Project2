"""Shared Spark session, paths and constants for the DSA5208 Project 2 pipeline.

Every job starts Spark through `get_spark()`, which also writes runs/<run_id>/env.json
(checks/env_capture.py) so the report's environment section comes from the run itself.
Paths default to the local build folder; set P2_DATA_ROOT (e.g. gs://bucket/p2) to run on Dataproc.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_ROOT = os.environ.get("P2_DATA_ROOT", os.path.join(ROOT, "data"))
OUT_ROOT = os.environ.get("P2_OUT_ROOT", os.path.join(ROOT, "outputs"))
RUNS_ROOT = os.environ.get("P2_RUNS_ROOT", os.path.join(ROOT, "runs"))

RAW_CSV = DATA_ROOT + "/raw/rainfall_{year}.csv"
# Derived parquet; on Dataproc each run gets its own prefix (P2_PARQUET_ROOT) so no stale partitions can be read back.
PARQUET_ROOT = os.environ.get("P2_PARQUET_ROOT", DATA_ROOT + "/parquet")
RAW_PARQUET = PARQUET_ROOT + "/raw"
CLEAN_PARQUET = PARQUET_ROOT + "/clean"
FEATURES_PARQUET = PARQUET_ROOT + "/features_base"
STATIONS_PARQUET = PARQUET_ROOT + "/stations"

YEARS = list(range(2017, 2025))
SLOT_SECONDS = 300          # one reading = rain in the 5 minutes ending at the timestamp
SLOTS_PER_DAY = 288
TZ = "Asia/Singapore"       # all timestamps in the files are +08:00


def is_remote(path):
    return "://" in path


def get_spark(app_name, job, extra_conf=None):
    """Start (or reuse) a SparkSession and capture its environment into runs/<run_id>/env.json."""
    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName(app_name)
    if not os.environ.get("P2_ON_CLUSTER"):
        b = (b.master(os.environ.get("P2_MASTER", "local[10]"))
             .config("spark.driver.memory", os.environ.get("P2_DRIVER_MEM", "10g"))
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.local.dir", os.path.join(ROOT, ".spark_tmp")))
    b = b.config("spark.sql.session.timeZone", TZ).config("spark.ui.showConsoleProgress", "false")
    for k, v in (extra_conf or {}).items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    from checks.env_capture import capture
    capture(spark, job)
    return spark


def write_csv_local(pdf, name, subdir):
    """Write a small pandas table (already aggregated by Spark) under outputs/<subdir>/."""
    d = os.path.join(OUT_ROOT, subdir)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".csv")
    pdf.to_csv(p, index=False)
    return p
