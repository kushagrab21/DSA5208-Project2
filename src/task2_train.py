"""Task 2, step 3: fit Models A (strict) and B (nowcast) with Spark MLlib on 2017-2022, tune on 2023 only.

Everything is read through the frozen plan (checks/eval_plan.py). The test years are never read here.
Each candidate is a Pipeline (SQLTransformer -> StringIndexer -> OneHotEncoder -> VectorAssembler -> classifier),
so the saved PipelineModel takes the raw columns (station_id, slot, theta and, for B, the looked-up past readings).

Writes:
  models/model_A, models/model_B               the chosen fitted PipelineModels (PipelineModel.save)
  outputs/task2/validation_metrics.csv         pooled and per-threshold metrics, every candidate and baseline
  outputs/task2/validation_reliability.csv     reliability table (10 probability bins)
  outputs/task2/validation_monotonicity.csv    share of (station, t) where P rises with the threshold
  outputs/task2/feature_importance_*.csv       GBT feature importances
  outputs/task2/selection.json                 which candidate was chosen for A and B, and why
Usage: python src/task2_train.py [--only A_lr,A_gbt,...]
"""
import argparse
import json
import os
import sys
import time

import common  # noqa: F401
from common import DATA_ROOT, PARQUET_ROOT, OUT_ROOT, ROOT, get_spark, write_csv_local
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(ROOT, "checks"))
import eval_plan as ep  # noqa: E402

T2 = PARQUET_ROOT + "/task2"
MODELS = os.environ.get("P2_MODELS_ROOT", os.path.join(ROOT, "models"))
REL_EDGES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0000001]
EPS = 1e-6


def derive_sql(grid):
    """Derived inputs, computed inside the pipeline so the saved model needs only raw columns."""
    case = " ".join(f"WHEN theta = {float(t)} THEN {i}" for i, t in enumerate(grid))
    return ("SELECT *, LN(1 + theta) AS log1p_theta, "
            f"CAST(CASE {case} ELSE {len(grid)} END AS DOUBLE) AS theta_idx, "
            "CAST(hour AS DOUBLE) AS hour_d, CAST(month - 1 AS DOUBLE) AS month_idx, CAST(season AS DOUBLE) AS season_d, "
            "LN(1 + r5) AS log1p_r5, LN(1 + r15) AS log1p_r15, LN(1 + r30) AS log1p_r30, LN(1 + r60) AS log1p_r60, "
            "LN(1 + r120) AS log1p_r120, LN(1 + r24h) AS log1p_r24h, CAST(r5 > 0 AS DOUBLE) AS wet_now, "
            "r30 - theta AS r30_minus_theta, CAST(n120 AS DOUBLE) AS n120_d, CAST(nb_n AS DOUBLE) AS nb_n_d "
            "FROM __THIS__")


def build_pipeline(name, cand, plan):
    grid = plan["threshold_grid_mm"]
    sql = SQLTransformer(statement=derive_sql(grid))
    b_num = ["log1p_r5", "log1p_r15", "log1p_r30", "log1p_r60", "log1p_r120", "log1p_r24h", "n120_d",
             "mins_since_wet", "wet_now", "r30_minus_theta", "nb_max_r30", "nb_mean_r30", "nb_max_r5",
             "nb_wet_now_frac", "nb_n_d", "isl_wet_now", "isl_wet30", "isl_max_r30", "isl_mean_r30"]
    cyc = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    is_b = name.startswith("B")
    hp = cand["params"]
    if cand["estimator"] == "LogisticRegression":
        si = StringIndexer(inputCol="station_id", outputCol="station_index", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=["station_index", "hour_d", "month_idx", "season_d", "theta_idx"],
                            outputCols=["station_oh", "hour_oh", "month_oh", "season_oh", "theta_oh"],
                            handleInvalid="keep", dropLast=True)
        cols = ["station_oh", "hour_oh", "month_oh", "season_oh", "theta_oh", "station_train_wet_rate",
                "log1p_theta"] + cyc + (b_num if is_b else [])
        va = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
        clf = LogisticRegression(featuresCol="features", labelCol="y", weightCol="weight", maxIter=100,
                                 regParam=hp["regParam"], elasticNetParam=hp["elasticNetParam"], standardization=True)
        stages = [sql, si, ohe, va, clf]
        feat_names = cols
    else:
        cols = ["station_lat", "station_lon", "station_train_wet_rate"] + cyc + \
               ["month_idx", "season_d", "theta", "log1p_theta"] + (b_num if is_b else [])
        va = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
        clf = GBTClassifier(featuresCol="features", labelCol="y", weightCol="weight", maxDepth=hp["maxDepth"],
                            maxIter=hp["maxIter"], stepSize=hp["stepSize"], maxBins=64, subsamplingRate=0.8,
                            seed=5208, cacheNodeIds=True, checkpointInterval=-1)
        stages = [sql, va, clf]
        feat_names = cols
    return Pipeline(stages=stages), feat_names


def candidates(plan):
    out = []
    for fam in ("A", "B"):
        for m in plan["models"][fam]:
            keys = list(m["grid"].keys())
            combos = [{}]
            for k in keys:
                combos = [dict(c, **{k: v}) for c in combos for v in m["grid"][k]]
            for c in combos:
                tag = "_".join(f"{k}{v:g}" for k, v in c.items())
                out.append({"family": fam, "name": f"{m['name']}__{tag}", "base": m["name"],
                            "estimator": m["estimator"], "params": c})
    return out


def metrics(scored, label, spark):
    """scored: theta, y, weight, p. Returns pooled + per-threshold metric rows and a reliability table."""
    pc = F.greatest(F.least(F.col("p"), F.lit(1 - EPS)), F.lit(EPS))
    rows = []
    roc = BinaryClassificationEvaluator(rawPredictionCol="p", labelCol="y", weightCol="weight",
                                        metricName="areaUnderROC")
    pr = BinaryClassificationEvaluator(rawPredictionCol="p", labelCol="y", weightCol="weight",
                                       metricName="areaUnderPR")
    thetas = [r[0] for r in scored.select("theta").distinct().orderBy("theta").collect()]
    for th in [None] + thetas:
        d = scored if th is None else scored.filter(F.col("theta") == th)
        a = d.agg(F.sum("weight").alias("w"), F.sum(F.col("weight") * F.col("y")).alias("wy"),
                  F.count(F.lit(1)).alias("n"), F.sum("y").alias("npos"),
                  (F.sum(F.col("weight") * (F.col("p") - F.col("y")) ** 2) / F.sum("weight")).alias("brier"),
                  (-F.sum(F.col("weight") * (F.col("y") * F.log(pc) + (1 - F.col("y")) * F.log(1 - pc)))
                   / F.sum("weight")).alias("log_loss"),
                  (F.sum(F.col("weight") * F.col("p")) / F.sum("weight")).alias("mean_p")).first()
        base = a["wy"] / a["w"]
        brier_ref = base * (1 - base)
        rows.append({"model": label, "theta": "all" if th is None else th, "rows": a["n"], "positives": int(a["npos"]),
                     "base_rate": base, "mean_p": a["mean_p"], "auc_roc": roc.evaluate(d), "auc_pr": pr.evaluate(d),
                     "brier": a["brier"], "brier_skill": 1 - a["brier"] / brier_ref if brier_ref > 0 else None,
                     "log_loss": a["log_loss"]})
    bins = F.lit(0)
    for i, e in enumerate(REL_EDGES[1:-1], start=1):
        bins = F.when(F.col("p") >= e, F.lit(i)).otherwise(bins)
    rel = (scored.withColumn("bin", bins).groupBy("bin").agg(
        F.sum("weight").alias("w"), (F.sum(F.col("weight") * F.col("p")) / F.sum("weight")).alias("mean_p"),
        (F.sum(F.col("weight") * F.col("y")) / F.sum("weight")).alias("obs_rate"),
        F.count(F.lit(1)).alias("rows")).orderBy("bin").toPandas())
    rel["model"] = label
    rel["bin_lo"] = [REL_EDGES[int(b)] for b in rel["bin"]]
    rel["bin_hi"] = [min(REL_EDGES[int(b) + 1], 1.0) for b in rel["bin"]]
    return rows, rel


def monotonicity(scored, label):
    w = Window.partitionBy("station_id", "slot").orderBy("theta")
    d = scored.withColumn("p_prev", F.lag("p").over(w)).filter(F.col("p_prev").isNotNull())
    a = d.agg(F.count(F.lit(1)).alias("pairs"),
              F.sum((F.col("p") > F.col("p_prev") + 1e-9).cast("long")).alias("rises"),
              F.max(F.col("p") - F.col("p_prev")).alias("max_rise")).first()
    return {"model": label, "adjacent_threshold_pairs": a["pairs"], "pairs_where_p_rises": a["rises"],
            "share_rises": a["rises"] / a["pairs"], "max_rise": a["max_rise"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    plan = ep.load_frozen_plan()
    spark = get_spark("p2-task2-train", "task2_train")
    train = spark.read.parquet(f"{T2}/train_sample")
    val = spark.read.parquet(f"{T2}/validation_sample")
    ep.guard_no_test_rows(train, plan)
    ep.guard_no_test_rows(val, plan)
    assert set(r[0] for r in val.select("year").distinct().collect()) <= set(plan["split"]["validation_years"])
    assert set(r[0] for r in train.select("year").distinct().collect()) <= set(plan["split"]["train_years"])
    train = train.repartition(40).cache()
    val = val.repartition(40).cache()
    print(f"TASK2 train rows={train.count()} validation rows={val.count()}", flush=True)
    keep = ["station_id", "slot", "theta", "y", "weight"]

    all_rows, all_rel, mono, fitted, timing = [], [], [], {}, []
    # ---------------------------------------------------------------- baselines on 2023
    clim = spark.read.parquet(f"{T2}/baseline_climatology")
    pool = spark.read.parquet(f"{T2}/baseline_climatology_pooled")
    pers = spark.read.parquet(f"{T2}/baseline_persistence")
    if not args.only:
        bc = (val.join(clim, ["station_id", "month", "hour", "theta"], "left")
                 .join(pool, ["month", "hour", "theta"], "left")
                 .withColumn("p", F.coalesce("p_clim", "p_clim_pool")).select(*keep, "p").cache())
        bp = (val.withColumn("past_gt", F.col("r30") > F.col("theta"))
                 .join(pers.select("theta", "past_gt", F.col("p_persist").alias("p")), ["theta", "past_gt"], "left")
                 .select(*keep, "p").cache())
        for label, sc in (("baseline_climatology", bc), ("baseline_persistence", bp)):
            r, rel = metrics(sc, label, spark)
            all_rows += r
            all_rel.append(rel)
            mono.append(monotonicity(sc, label))
            print(f"TASK2 VAL {label}: " + fmt(r[0]), flush=True)

    # ---------------------------------------------------------------- candidates
    only = set(x for x in args.only.split(",") if x)
    for cand in candidates(plan):
        if only and cand["base"] not in only and cand["name"] not in only:
            continue
        pipe, feat_names = build_pipeline(cand["name"], cand, plan)
        t0 = time.time()
        model = pipe.fit(train)
        fit_s = time.time() - t0
        sc = (model.transform(val).withColumn("p", vector_to_array("probability")[1]).select(*keep, "p").cache())
        r, rel = metrics(sc, cand["name"], spark)
        all_rows += r
        all_rel.append(rel)
        mono.append(monotonicity(sc, cand["name"]))
        timing.append({"model": cand["name"], "fit_seconds": fit_s, "train_rows": train.count()})
        print(f"TASK2 VAL {cand['name']} (fit {fit_s:.0f} s): " + fmt(r[0]), flush=True)
        fitted[cand["name"]] = (cand, model, r[0], feat_names)
        sc.unpersist()

    import pandas as pd
    out = os.path.join(OUT_ROOT, "task2")
    tag = "" if not only else "_partial"
    dfm = pd.DataFrame(all_rows)
    write_csv_local(dfm, "validation_metrics" + tag, "task2")
    write_csv_local(pd.concat(all_rel), "validation_reliability" + tag, "task2")
    write_csv_local(pd.DataFrame(mono), "validation_monotonicity" + tag, "task2")
    write_csv_local(pd.DataFrame(timing), "fit_timing" + tag, "task2")

    # ---------------------------------------------------------------- selection on 2023, per family
    sel = {"plan_sha256": ep.sha256_file(ep.PLAN), "rule": plan["selection"]}
    for fam in ("A", "B"):
        cands = [(k, v) for k, v in fitted.items() if v[0]["family"] == fam]
        if not cands:
            continue
        best = sorted(cands, key=lambda kv: (kv[1][2]["log_loss"], -kv[1][2]["auc_pr"]))[0]
        name, (cand, model, pooled, feat_names) = best
        path = os.path.join(MODELS, f"model_{fam}")
        model.write().overwrite().save(path)
        sel[fam] = {"chosen": name, "estimator": cand["estimator"], "params": cand["params"],
                    "validation_log_loss": pooled["log_loss"], "validation_auc_pr": pooled["auc_pr"],
                    "validation_auc_roc": pooled["auc_roc"], "path": os.path.relpath(path, ROOT),
                    "candidates": {k: {"log_loss": v[2]["log_loss"], "auc_pr": v[2]["auc_pr"]} for k, v in cands}}
        clf = model.stages[-1]
        if hasattr(clf, "featureImportances"):
            imp = clf.featureImportances.toArray()
            write_csv_local(pd.DataFrame({"feature": feat_names, "importance": imp})
                            .sort_values("importance", ascending=False), f"feature_importance_{fam}" + tag, "task2")
        print(f"TASK2 SELECT model_{fam} = {name} (validation log loss {pooled['log_loss']:.5f}, "
              f"AUC-PR {pooled['auc_pr']:.4f}) saved to {os.path.relpath(path, ROOT)}", flush=True)
    json.dump(sel, open(os.path.join(out, "selection" + tag + ".json"), "w"), indent=1)
    spark.stop()


def fmt(r):
    return (f"AUC-ROC={r['auc_roc']:.4f} AUC-PR={r['auc_pr']:.4f} Brier={r['brier']:.5f} "
            f"BSS={r['brier_skill']:.3f} logloss={r['log_loss']:.5f} base={r['base_rate']:.4f} mean_p={r['mean_p']:.4f}")


if __name__ == "__main__":
    main()
