# Source this before running anything locally: `source env.sh` (from any directory)
P2_BUILD="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
export P2_BUILD
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$P2_BUILD/.venv/bin:$PATH"
export PYSPARK_PYTHON="$P2_BUILD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$P2_BUILD/.venv/bin/python"
export SPARK_LOCAL_IP=127.0.0.1
