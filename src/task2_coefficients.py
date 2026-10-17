"""Standardised coefficients of the chosen logistic-regression Model B (and A), in place of feature importances.

Standardised coefficient = coefficient x (weighted standard deviation of that feature in the training sample), i.e.
the change in log-odds for a one-standard-deviation change of the feature. Feature names come from the
VectorAssembler's ML attribute metadata (one-hot slots are named e.g. station_oh_S24, theta_oh_3).
Writes outputs/task2/coefficients_<fam>.csv. Usage: python src/task2_coefficients.py
"""
import os

import common  # noqa: F401
from common import DATA_ROOT, PARQUET_ROOT, ROOT, get_spark, write_csv_local
from pyspark.ml import PipelineModel
from pyspark.ml.stat import Summarizer
from pyspark.sql import functions as F

T2 = PARQUET_ROOT + "/task2"
MODELS = os.environ.get("P2_MODELS_ROOT", os.path.join(ROOT, "models"))


def main():
    import pandas as pd
    spark = get_spark("p2-task2-coef", "task2_coefficients")
    train = spark.read.parquet(f"{T2}/train_sample")
    for fam in ("A", "B"):
        pm = PipelineModel.load(os.path.join(MODELS, f"model_{fam}"))
        lr = pm.stages[-1]
        if not hasattr(lr, "coefficients"):
            print(f"TASK2 coef model_{fam}: not a linear model, skipped")
            continue
        feats = PipelineModel(stages=pm.stages[:-1]).transform(train)
        meta = feats.schema["features"].metadata["ml_attr"]["attrs"]
        names = {}
        for kind in meta.values():
            for a in kind:
                names[a["idx"]] = a["name"]
        std = (feats.select(Summarizer.metrics("std").summary(F.col("features"), F.col("weight")).alias("s"))
               .first()["s"]["std"].toArray())
        coef = lr.coefficients.toArray()
        df = pd.DataFrame({"feature": [names.get(i, f"f{i}") for i in range(len(coef))], "coefficient": coef,
                           "std": std, "std_coefficient": coef * std})
        df["group"] = df.feature.str.replace(r"_oh_.*$", "", regex=True)
        df = df.reindex(df.std_coefficient.abs().sort_values(ascending=False).index)
        write_csv_local(df, f"coefficients_{fam}", "task2")
        print(f"TASK2 coef model_{fam}: {len(df)} coefficients, intercept {lr.intercept:.3f}; top 8 by |std coef|:")
        for r in df.head(8).itertuples():
            print(f"    {r.feature:<28} coef={r.coefficient:+.4f} std={r.std:.4f} std_coef={r.std_coefficient:+.3f}")
    spark.stop()


if __name__ == "__main__":
    main()
