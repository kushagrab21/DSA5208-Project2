"""Capture the computing environment of a Spark run into runs/<run_id>/env.json.

Called by src/common.get_spark() for every job. Prints one line:
  ENV run=<id> job=<job> spark=<v> master=<m> executors=<n> cores=<c> memory=<m> host=<cpu>/<ram>/<os>
The report's computing-environment section is generated from these files (src/report_macros.py), never typed.
Run standalone (`python checks/env_capture.py`) to list captured runs.
"""
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _host_info():
    info = {"os": platform.platform(), "machine": platform.machine(), "hostname_hash": None,
            "cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(), "ram_bytes": None}
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["ram_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
            info["os"] = "macOS " + subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_bytes"] = int(line.split()[1]) * 1024
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu"] = line.split(":", 1)[1].strip()
                        break
            try:
                with open("/etc/os-release") as f:
                    kv = dict(l.strip().split("=", 1) for l in f if "=" in l)
                info["os"] = kv.get("PRETTY_NAME", info["os"]).strip('"')
            except OSError:
                pass
    except Exception as e:  # environment capture must never kill a job, but it records why it is partial
        info["capture_error"] = repr(e)
    return info


def capture(spark, job):
    sc = spark.sparkContext
    conf = dict(sc.getConf().getAll())
    jvm = sc._jvm
    run_id = os.environ.get("P2_RUN_ID") or (_dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + job)
    # executors as seen by the driver (local mode: the driver is the only "executor")
    try:
        ems = sc._jsc.sc().getExecutorMemoryStatus()
        n_exec_entries = ems.size()
    except Exception:
        n_exec_entries = None
    env = {
        "run_id": run_id,
        "job": job,
        "captured_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "spark_version": spark.version,
        "java_version": jvm.System.getProperty("java.version"),
        "java_vendor": jvm.System.getProperty("java.vendor"),
        "scala_version": jvm.scala.util.Properties.versionNumberString(),
        "python_version": platform.python_version(),
        "master": sc.master,
        "app_id": sc.applicationId,
        "default_parallelism": sc.defaultParallelism,
        "executor_entries_incl_driver": n_exec_entries,
        "executor_instances": conf.get("spark.executor.instances"),
        "executor_cores": conf.get("spark.executor.cores"),
        "executor_memory": conf.get("spark.executor.memory"),
        "driver_memory": conf.get("spark.driver.memory"),
        "dynamic_allocation": conf.get("spark.dynamicAllocation.enabled"),
        "shuffle_partitions": conf.get("spark.sql.shuffle.partitions"),
        "session_timezone": conf.get("spark.sql.session.timeZone"),
        "dataproc_image": os.environ.get("DATAPROC_IMAGE_VERSION"),
        "dataproc_cluster": os.environ.get("P2_CLUSTER_NAME"),
        "host": _host_info(),
    }
    d = os.environ.get("P2_RUNS_ROOT", os.path.join(ROOT, "runs"))
    if "://" in d:
        d = "/tmp/p2_runs"   # on Dataproc the env files are copied to GCS by cloud/dataproc_run.sh
    os.makedirs(os.path.join(d, run_id), exist_ok=True)
    path = os.path.join(d, run_id, "env.json")
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
    t0 = _dt.datetime.now().astimezone()

    def _finish():   # record wall time when the job's Python process ends
        try:
            e = json.load(open(path))
            e["finished_at"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
            e["wall_seconds"] = round((_dt.datetime.now().astimezone() - t0).total_seconds(), 1)
            json.dump(e, open(path, "w"), indent=2)
        except Exception:
            pass
    import atexit
    atexit.register(_finish)
    h = env["host"]
    ram = f"{(h.get('ram_bytes') or 0) / 2**30:.0f}GiB"
    print(f"ENV run={run_id} job={job} spark={env['spark_version']} java={env['java_version']} "
          f"python={env['python_version']} master={env['master']} executors={env['executor_instances'] or n_exec_entries} "
          f"cores={env['executor_cores'] or env['default_parallelism']} memory={env['executor_memory'] or env['driver_memory']} "
          f"host={h.get('cpu')}/{h.get('logical_cpus')}cpu/{ram}/{h.get('os')}", flush=True)
    return env


if __name__ == "__main__":
    d = os.path.join(ROOT, "runs")
    for r in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        p = os.path.join(d, r, "env.json")
        if os.path.exists(p):
            e = json.load(open(p))
            print(f"ENV run={e['run_id']} job={e['job']} spark={e['spark_version']} master={e['master']} "
                  f"host={e['host'].get('cpu')}/{e['host'].get('logical_cpus')}cpu/{e['host'].get('os')}")
