"""First Dataproc job: confirm the driver's Python has what the pipeline needs before the long stages run.
Prints versions; exits 1 if pandas or pyarrow (needed by DataFrame.toPandas for the small result tables) is missing."""
import platform
import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("p2-envcheck").getOrCreate()
print("ENVCHECK python", platform.python_version(), "spark", spark.version)
missing = []
for mod in ("pandas", "pyarrow", "numpy"):
    try:
        m = __import__(mod)
        print("ENVCHECK", mod, m.__version__)
    except ImportError:
        missing.append(mod)
        print("ENVCHECK", mod, "MISSING")
print("ENVCHECK rows", spark.range(1000).count())
spark.stop()
sys.exit(1 if missing else 0)
