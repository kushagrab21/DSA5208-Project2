"""Qualify the enforced checks: each must REFUSE a deliberately broken input before we rely on it.

  Q1 ingest_check    a raw Parquet set with one year (2017) missing           -> must exit 1, "MISSING YEARS: 1"
  Q2 clean_ledger    a ledger that accounts for one row fewer than was removed -> must exit 1, "unmapped_rows=1"
  Q3 eval_plan       a plan with a random split                                -> freeze must refuse
  Q4 eval_plan       the frozen plan edited after freezing                     -> verify must refuse
  Q5 eval_plan       a second evaluation of the test years                     -> claim_test_run must refuse
  Q6 eval_plan       a fitting table that contains a 2024 row                  -> guard_no_test_rows must refuse
  Q8 cloud_parity        a cloud output set identical to the local one passes; the same set with one count changed
                        (coverage slots of 2019, +1) must be refused
  Q7 requirements_check  a compliant minimal .tex passes; the same .tex with one hand-typed number, and with one
                        requirement unmapped, and with an undefined \\val key, must each be refused

Fixtures are built under build/tests/fixtures/ (symlinks to the real Parquet partitions, small JSON edits); the real
plan, lock and data are never modified. Writes the evidence to outputs/qualification/qualification.log.
Usage: python checks/qualify.py [Q1 Q2 ...]
"""
import copy
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "checks"))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIX = os.path.join(ROOT, "tests", "fixtures")
LOG = os.path.join(ROOT, "outputs", "qualification", "qualification.log")
PY = sys.executable
results = []


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def run(cmd):
    env = dict(os.environ, P2_RUN_ID="qualify-" + _dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" +
               os.path.basename(cmd[1]).replace(".py", ""))   # qualification runs are not pipeline runs
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    lines = [l for l in (p.stdout + p.stderr).splitlines()
             if l.startswith(("INGEST", "MISSING", "CLEAN", "EVAL", "ENV"))]
    return p.returncode, lines


def record(qid, what, refused, evidence):
    results.append((qid, refused))
    log(f"QUALIFY {qid} {what}: {'REFUSED as required -> qualified' if refused else 'NOT REFUSED -> NOT qualified'}")
    for l in evidence:
        log("    " + l)


def q1():
    d = os.path.join(FIX, "raw_missing_2017")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    src = os.path.join(ROOT, "data", "parquet", "raw")
    for p in os.listdir(src):
        if p.startswith("year=") and p != "year=2017":
            os.symlink(os.path.join(src, p), os.path.join(d, p))
    rc, lines = run([PY, "checks/ingest_check.py", "--parquet", d, "--skip-csv"])
    record("Q1", "ingest_check on raw Parquet without 2017", rc != 0 and "MISSING YEARS: 1 [2017]" in lines, lines)


def q2():
    led = json.load(open(os.path.join(ROOT, "outputs", "ledger", "clean_ledger.json")))
    bad = copy.deepcopy(led)
    for r in bad["rules"]:
        if r["rule"] == "duplicate":
            r["removed"] -= 1      # one removed row no longer attributed to any rule
    os.makedirs(FIX, exist_ok=True)
    p = os.path.join(FIX, "ledger_one_unmapped.json")
    json.dump(bad, open(p, "w"), indent=1)
    rc, lines = run([PY, "checks/clean_ledger.py", "--ledger", p])
    record("Q2", "clean_ledger with one unmapped row", rc != 0 and any("unmapped_rows=1" in l for l in lines), lines)


def _plan_fixture(name, mutate):
    plan = json.load(open(os.path.join(ROOT, "config", "eval_plan.json")))
    mutate(plan)
    d = os.path.join(FIX, "plans")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".json")
    json.dump(plan, open(p, "w"), indent=1)
    return p


def q3():
    def rnd(plan):
        plan["split"] = {"method": "randomSplit", "weights": [0.7, 0.3], "seed": 42,
                         "train_years": list(range(2017, 2025)), "validation_years": list(range(2017, 2025)),
                         "test_years": list(range(2017, 2025))}
    p = _plan_fixture("random_split", rnd)
    lock = os.path.join(FIX, "plans", "random_split.lock")
    if os.path.exists(lock):
        os.remove(lock)
    rc, lines = run([PY, "checks/eval_plan.py", "freeze", "--plan", p, "--lock", lock])
    record("Q3", "eval_plan freeze of a random split", rc != 0 and not os.path.exists(lock), lines)


def q4():
    import eval_plan as ep
    p = _plan_fixture("frozen_then_edited", lambda plan: None)
    lock = os.path.join(FIX, "plans", "frozen_then_edited.lock")
    if os.path.exists(lock):
        os.remove(lock)
    rc0, l0 = run([PY, "checks/eval_plan.py", "freeze", "--plan", p, "--lock", lock])
    plan = json.load(open(p))
    plan["threshold_grid_mm"] = plan["threshold_grid_mm"][:-1]   # quietly drop the hardest threshold
    json.dump(plan, open(p, "w"), indent=1)
    rc, lines = run([PY, "checks/eval_plan.py", "verify", "--plan", p, "--lock", lock])
    raised = False
    try:
        ep.load_frozen_plan(p, lock)
    except ep.PlanError as e:
        raised = True
        lines.append(f"load_frozen_plan raised PlanError: {e}")
    record("Q4", "eval_plan verify after editing a frozen plan", rc0 == 0 and rc != 0 and raised, l0 + lines)


def q5():
    import eval_plan as ep
    p = _plan_fixture("test_twice", lambda plan: None)
    lock = os.path.join(FIX, "plans", "test_twice.lock")
    tr = os.path.join(FIX, "plans", "test_twice.test_run.json")
    for f in (lock, tr):
        if os.path.exists(f):
            os.remove(f)
    run([PY, "checks/eval_plan.py", "freeze", "--plan", p, "--lock", lock])
    lines = []
    ep.claim_test_run(p, lock, tr)
    lines.append("first claim_test_run: accepted (fixture plan)")
    raised = False
    try:
        ep.claim_test_run(p, lock, tr)
    except ep.PlanError as e:
        raised = True
        lines.append(f"second claim_test_run raised PlanError: {e}")
    record("Q5", "a second evaluation of the test years", raised, lines)


def q6():
    import eval_plan as ep
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[2]").appName("qualify-q6")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    plan = json.load(open(os.path.join(ROOT, "config", "eval_plan.json")))
    df = spark.createDataFrame([("S1", 2021, 0.0), ("S1", 2022, 1.0), ("S2", 2024, 0.0)], ["station_id", "year", "y"])
    lines, raised = [], False
    try:
        ep.guard_no_test_rows(df, plan)
    except ep.PlanError as e:
        raised = True
        lines.append(f"guard_no_test_rows raised PlanError: {e}")
    ok_df = df.filter("year < 2024")
    ep.guard_no_test_rows(ok_df, plan)
    lines.append("same table without the 2024 row: accepted")
    spark.stop()
    record("Q6", "a fitting table containing a 2024 row", raised, lines)


GOOD_TEX = r"""\documentclass{article}
\newcommand{\req}[1]{}
\input{generated/macros.tex}
\author{Kushagra Bhatnagar (A0351591A) \and Pratyush Sinha (A0352482A)}
\begin{document}
\section{Data}\req{E1}\req{E2}\req{E3} We read \NRawRows{} rows (Figure~\ref{fig:1}).
\includegraphics[width=0.9\linewidth]{figures/fig01_coverage.png}
\section{Task 1}\req{E4}\req{E5}\req{E6} The wettest month is \WettestMonth.
\section{Task 2}\req{E7}\req{E8}\req{E9} AUC-PR is \BAucPr{} on \texttt{task2\_train.py} output.
\section{Environment}\req{E10}\req{E13}\req{E14}
\section{Declaration of AI use}\req{E11} We used Claude Code.
\section{Limitations}\req{E12}
\end{document}
"""


def q7():
    d = os.path.join(FIX, "tex")
    os.makedirs(d, exist_ok=True)
    good = os.path.join(d, "good.tex")
    open(good, "w").write(GOOD_TEX)
    num = os.path.join(d, "hand_typed_number.tex")
    open(num, "w").write(GOOD_TEX.replace("The wettest month is \\WettestMonth.",
                                          "The wettest month is \\WettestMonth{} with 10.5 mm per day."))
    unm = os.path.join(d, "unmapped_row.tex")
    open(unm, "w").write(GOOD_TEX.replace("\\req{E5}", ""))
    undef = os.path.join(d, "undefined_value.tex")
    open(undef, "w").write(GOOD_TEX.replace("The wettest month is \\WettestMonth.",
                                            "The wettest month is \\val{no_such_key}."))
    rc3, l3 = run_req(undef)
    rc0, l0 = run_req(good)
    rc1, l1 = run_req(num)
    rc2, l2 = run_req(unm)
    ok = (rc0 == 0 and rc1 != 0 and rc2 != 0 and rc3 != 0 and any("hand-typed numbers in body: 1" in l for l in l1)
          and any("undefined: 1" in l for l in l3))
    record("Q7", "requirements_check: good .tex passes; hand-typed number and unmapped E5 refused", ok,
           ["[good.tex] " + l for l in l0 if "REQUIREMENTS" in l or "hand-typed" in l]
           + ["[hand_typed_number.tex] " + l for l in l1 if "REQ" in l and ("hand" in l or "line" in l or "CHECK" in l)]
           + ["[unmapped_row.tex] " + l for l in l2 if "E5" in l or "CHECK" in l]
           + ["[undefined_value.tex] " + l for l in l3 if "undefined" in l or "CHECK" in l])


def q8():
    base = os.path.join(FIX, "cloud_parity")
    shutil.rmtree(base, ignore_errors=True)
    lines, rcs = [], []
    for name, bad in (("same", False), ("one_count_off", True)):
        d = os.path.join(base, name)
        shutil.copytree(os.path.join(ROOT, "outputs"), os.path.join(d, "outputs"),
                        ignore=shutil.ignore_patterns("cloud_parity.json", "qualification"))
        if bad:
            import pandas as pd
            p = os.path.join(d, "outputs", "coverage", "coverage_by_year.csv")
            t = pd.read_csv(p)
            t.loc[t.year == 2019, "slots"] += 1
            t.to_csv(p, index=False)
        q = subprocess.run([PY, "checks/cloud_parity.py", "--cloud", d, "--out", os.path.join(d, "parity.json")],
                           capture_output=True, text=True, cwd=ROOT)
        rcs.append(q.returncode)
        lines += [f"[{name}] " + l for l in q.stdout.splitlines() if l.startswith(("PARITY exact", "PARITY:", "    "))]
    record("Q8", "cloud_parity: identical outputs pass; one count off refused", rcs[0] == 0 and rcs[1] != 0, lines)


def run_req(path):
    p = subprocess.run([PY, "checks/requirements_check.py", path], capture_output=True, text=True, cwd=ROOT)
    return p.returncode, [l for l in p.stdout.splitlines()]


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    want = sys.argv[1:] or ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    log(f"=== qualification run {_dt.datetime.now().astimezone().isoformat(timespec='seconds')} {want}")
    for q in want:
        globals()[q.lower()]()
    bad = [q for q, ok in results if not ok]
    log("QUALIFY SUMMARY: " + ("all refused as required" if not bad else f"NOT qualified: {bad}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
