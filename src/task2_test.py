"""Task 2, step 4: the single evaluation on the test year (2024), under the frozen plan.

  python src/task2_test.py --dry-run   exercise the whole path on the VALIDATION year without claiming the test:
                                       rebuild the validation sample under a new name (must equal the original row
                                       for row) and re-score the chosen models and baselines (must reproduce
                                       validation_metrics.csv). Nothing from the test year is read.
  python src/task2_test.py             claim the test run (checks/eval_plan.claim_test_run: refuses if the plan
                                       changed or the test was already run), build the 2024 sample with the plan's
                                       test seed, score the CHOSEN Model A, Model B and both baselines once.

No refit, no re-tuning: the models are loaded from models/model_A and models/model_B as saved after validation.
Writes outputs/task2/test_metrics.csv, test_reliability.csv, test_monotonicity.csv, test_run.json.
"""
import argparse
import json
import os
import sys

import common  # noqa: F401
from common import DATA_ROOT, PARQUET_ROOT, OUT_ROOT, ROOT, get_spark, write_csv_local
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(ROOT, "checks"))
import eval_plan as ep  # noqa: E402
from task2_sample import (ISLAND_PARQUET, build_split, labelled_base, neighbour_pairs,  # noqa: E402
                          neighbour_source, station_table)
from task2_train import MODELS, fmt, metrics, monotonicity  # noqa: E402

T2 = PARQUET_ROOT + "/task2"
KEEP = ["station_id", "slot", "theta", "y", "weight"]


def score_all(spark, df, sel):
    clim = spark.read.parquet(f"{T2}/baseline_climatology")
    pool = spark.read.parquet(f"{T2}/baseline_climatology_pooled")
    pers = spark.read.parquet(f"{T2}/baseline_persistence")
    out = {
        "baseline_climatology": (df.join(clim, ["station_id", "month", "hour", "theta"], "left")
                                 .join(pool, ["month", "hour", "theta"], "left")
                                 .withColumn("p", F.coalesce("p_clim", "p_clim_pool")).select(*KEEP, "p")),
        "baseline_persistence": (df.withColumn("past_gt", F.col("r30") > F.col("theta"))
                                 .join(pers.select("theta", "past_gt", F.col("p_persist").alias("p")),
                                       ["theta", "past_gt"], "left").select(*KEEP, "p")),
    }
    for fam in ("A", "B"):
        m = PipelineModel.load(os.path.join(MODELS, f"model_{fam}"))
        out[sel[fam]["chosen"]] = (m.transform(df).withColumn("p", vector_to_array("probability")[1])
                                   .select(*KEEP, "p"))
    return out


def evaluate(spark, df, sel, tag):
    rows, rels, mono = [], [], []
    for label, sc in score_all(spark, df, sel).items():
        sc = sc.cache()
        r, rel = metrics(sc, label, spark)
        rows += r
        rels.append(rel)
        mono.append(monotonicity(sc, label))
        print(f"TASK2 {tag} {label}: " + fmt(r[0]), flush=True)
        sc.unpersist()
    return rows, rels, mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    sel = json.load(open(os.path.join(OUT_ROOT, "task2", "selection.json")))
    if a.dry_run:
        plan = ep.load_frozen_plan()
        spark = get_spark("p2-task2-test-dryrun", "task2_test_dryrun")
        tr_y, va_y = plan["split"]["train_years"], plan["split"]["validation_years"]
        st = station_table(spark, labelled_base(spark, tr_y))
        b = labelled_base(spark, va_y)
        ep.guard_no_test_rows(b, plan)
        path = build_split(spark, plan, b, "validation", va_y, st, neighbour_pairs(spark),
                           neighbour_source(spark, va_y), spark.read.parquet(ISLAND_PARQUET), out_name="dryrun_validation")
        new, old = spark.read.parquet(path), spark.read.parquet(f"{T2}/validation_sample")
        cols = sorted(old.columns)
        diff = new.select(cols).exceptAll(old.select(cols)).count() + old.select(cols).exceptAll(new.select(cols)).count()
        print(f"TASK2 DRYRUN rebuilt validation sample rows={new.count()} original={old.count()} differing_rows={diff}")
        rows, _, _ = evaluate(spark, new, sel, "DRYRUN")
        ref = pd.read_csv(os.path.join(OUT_ROOT, "task2", "validation_metrics.csv"))
        bad = 0
        for r in rows:
            if r["theta"] != "all":
                continue
            q = ref[(ref.model == r["model"]) & (ref.theta == "all")].iloc[0]
            d = max(abs(q.auc_pr - r["auc_pr"]), abs(q.log_loss - r["log_loss"]), abs(q.brier - r["brier"]))
            bad += d > 1e-4   # same rows; only floating-point summation order and evaluator binning differ
            print(f"TASK2 DRYRUN {r['model']} max |difference| vs validation_metrics.csv = {d:.2e}")
        print("TASK2 DRYRUN: " + ("PASS (test path reproduces validation)" if diff == 0 and bad == 0 else "FAIL"))
        spark.stop()
        sys.exit(0 if diff == 0 and bad == 0 else 1)

    plan = ep.claim_test_run()            # refuses if the plan changed or the test was already run
    spark = get_spark("p2-task2-test", "task2_test")
    tr_y, te_y = plan["split"]["train_years"], plan["split"]["test_years"]
    st = station_table(spark, labelled_base(spark, tr_y))
    b = labelled_base(spark, te_y)
    path = build_split(spark, plan, b, "test", te_y, st, neighbour_pairs(spark), neighbour_source(spark, te_y),
                       spark.read.parquet(ISLAND_PARQUET))
    rows, rels, mono = evaluate(spark, spark.read.parquet(path), sel, "TEST")
    write_csv_local(pd.DataFrame(rows), "test_metrics", "task2")
    write_csv_local(pd.concat(rels), "test_reliability", "task2")
    write_csv_local(pd.DataFrame(mono), "test_monotonicity", "task2")
    rec = json.load(open(ep.TEST_RUN))
    rec.update({"models": {f: sel[f]["chosen"] for f in ("A", "B")}, "test_years": te_y,
                "finished": True})
    json.dump(rec, open(os.path.join(OUT_ROOT, "task2", "test_run.json"), "w"), indent=1)
    print(f"TEST RUN done once under plan sha256={rec['sha256']} started_at={rec['started_at']} "
          f"models A={sel['A']['chosen']} B={sel['B']['chosen']}")
    spark.stop()


if __name__ == "__main__":
    main()
