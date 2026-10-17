"""Driver-side wrapper for Dataproc: unpack the project code shipped with --py-files, point the pipeline at GCS,
run one stage script, then copy the small local outputs (tables, env.json) to GCS.

  gcloud dataproc jobs submit pyspark cloud/run_on_cluster.py --py-files=p2code.zip ... -- \
      --data-root gs://BUCKET/p2/data --out-root gs://BUCKET/p2/outputs --stage src/task1.py [stage args]

No Python UDFs are used anywhere, so executors need no project code; only the driver unpacks it.
"""
import argparse
import atexit
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile


def sync(src, dst):
    """Mirror a directory tree into another (local <-> gs://), merging into existing folders.
    `hadoop fs -copyFromLocal dir/. gs://...` nested outputs/outputs/... one level deeper on every stage (second Dataproc
    run), so the cluster's gcloud rsync is used; per-file hadoop copies are the fallback."""
    if shutil.which("gcloud"):
        rc = subprocess.call(["gcloud", "storage", "rsync", "--recursive", src, dst])
        if rc == 0:
            return
    for root, _, files in os.walk(src) if not src.startswith("gs://") else []:
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else f"{dst}/{rel}"
        subprocess.call(["hadoop", "fs", "-mkdir", "-p", target])
        for f in files:
            subprocess.call(["hadoop", "fs", "-copyFromLocal", "-f", os.path.join(root, f), f"{target}/{f}"])
    if src.startswith("gs://"):
        subprocess.call(["hadoop", "fs", "-copyToLocal", src + "/*", dst + "/"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--run-id", default=None)
    a, rest = ap.parse_known_args()
    zp = next((p for p in sys.path if p.endswith("p2code.zip")), None) or "p2code.zip"
    work = tempfile.mkdtemp(prefix="p2_")
    zipfile.ZipFile(zp).extractall(work)
    local_out = os.path.join(work, "outputs")
    # Each stage runs in a fresh directory, but some read small outputs an earlier stage wrote (e.g. clean_ledger
    # reads outputs/ledger/clean_ledger.json written by clean). Bring the earlier stages' outputs down from GCS first.
    os.makedirs(local_out, exist_ok=True)
    remote_out = f"{a.out_root.rstrip('/')}/outputs"
    if subprocess.call(["hadoop", "fs", "-test", "-d", remote_out]) == 0:
        sync(remote_out, local_out)
    os.environ.update({"P2_ON_CLUSTER": "1", "P2_DATA_ROOT": a.data_root, "P2_OUT_ROOT": local_out,
                       "P2_PARQUET_ROOT": a.out_root.rstrip("/") + "/parquet",
                       "P2_RUNS_ROOT": os.path.join(work, "runs"),
                       "P2_MODELS_ROOT": a.out_root.rstrip("/") + "/models"})
    if a.run_id:
        os.environ["P2_RUN_ID"] = a.run_id
    sys.path[:0] = [os.path.join(work, "src"), os.path.join(work, "checks"), work]
    sys.argv = [os.path.join(work, a.stage)] + rest
    rc = 0
    try:
        runpy.run_path(os.path.join(work, a.stage), run_name="__main__")
    except SystemExit as e:
        rc = e.code or 0
    finally:
        # env_capture records wall_seconds in an exit hook; run it before the copy to GCS (the first full Dataproc
        # run copied env.json first, so its wall times were taken from the jobs API instead).
        atexit._run_exitfuncs()
        for sub in ("outputs", "runs"):
            src = os.path.join(work, sub)
            if os.path.isdir(src):
                sync(src, f"{a.out_root.rstrip('/')}/{sub}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
