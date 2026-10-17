"""Parity of the Dataproc run with the local run (print, and refuse on an exact-match failure).

Exact: the counts that do not depend on floating-point order must be identical
  - ingest: rows and stations per year            (outputs/ingest/ingest_table.json; the cloud skips CSV line counts)
  - cleaning ledger: every rule's removed / of     (outputs/ledger/clean_ledger.json)
  - coverage: stations, station-days, slots, days  (outputs/coverage/coverage_by_year.csv)
  - Task 1: every count column of every table      (outputs/task1/*.csv)
  - Task 2 samples: rows and positives per theta   (outputs/task2/{train,validation}_sample_by_theta.csv)
Close: floating-point sums (Task 1 rain sums) to a relative 1e-6, and the Task 2 validation metrics of every candidate
(expected close, not identical: summation order and parallelism differ).
Two classes are reported rather than held to those rules, because they depend on the order Spark combines partitions:
  - approximate quantiles (columns p50_*, p99_*, ...: Spark's percentile_approx, rank accuracy 1/10000): largest
    relative difference reported;
  - threshold counts on summed rain (hours >= 20/40 mm, days >= 50 mm): a difference is accepted only if, in every row,
    it is no larger than the number of totals within 1e-6 mm of the threshold (outputs/threshold_boundary.json, from
    checks/threshold_boundary.py); otherwise it is an exact mismatch. Writes
outputs/cloud_parity.json for the report. Usage: python checks/cloud_parity.py [--cloud DIR] [--out FILE]
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "outputs")
FLOAT_RTOL = 1e-6
def is_count(col, s):
    return pd.api.types.is_integer_dtype(s) or col in ("year", "month", "hour_start", "doy")


BOUNDARY = os.path.join(LOCAL, "threshold_boundary.json")


def is_approx_quantile(col):
    return col[:1] == "p" and col[1:2].isdigit()


def boundary_ok(name, col, lo, a, b, bound, boundary_rows):
    """Threshold counts may differ by at most the number of totals sitting on the threshold, row by row."""
    if col not in bound:
        return False
    key = next((k for k in ("year", "month") if k in lo.columns and k in bound[col]), None)
    if key is None:
        return False
    lim = lo[key].astype(str).map(bound[col][key]).fillna(0)
    diff = (a - b).abs()
    ok = bool((diff <= lim).all())
    boundary_rows.append({"column": f"{name}.{col}", "rows_differing": int((diff > 0).sum()),
                          "max_diff": int(diff.max()), "min_on_threshold": int(lim[diff > 0].min()) if (diff > 0).any() else 0,
                          "within_bound": ok})
    return ok


def compare_tables(name, lo, cl, exact_fail, close_rows, approx_rows=None, boundary_rows=None, bound=None):
    if lo.shape != cl.shape or list(lo.columns) != list(cl.columns):
        exact_fail.append(f"{name}: shape/columns differ {lo.shape} vs {cl.shape}")
        return
    keys = [c for c in lo.columns if not pd.api.types.is_numeric_dtype(lo[c])]
    if keys:
        lo, cl = lo.sort_values(keys + [c for c in lo.columns if c in ("year", "month")]).reset_index(drop=True), \
                 cl.sort_values(keys + [c for c in cl.columns if c in ("year", "month")]).reset_index(drop=True)
    for c in lo.columns:
        a, b = lo[c], cl[c]
        if not pd.api.types.is_numeric_dtype(a):
            if not a.astype(str).equals(b.astype(str)):
                exact_fail.append(f"{name}.{c}: labels differ")
            continue
        if is_count(c, a) and is_count(c, b):
            n = int((a != b).sum())
            if n and not boundary_ok(name, c, lo, a, b, bound or {}, boundary_rows if boundary_rows is not None else []):
                exact_fail.append(f"{name}.{c}: {n} counts differ")
        else:
            den = a.abs().where(a.abs() > 0, 1.0)
            rel = float(((a - b).abs() / den).max()) if len(a) else 0.0
            if is_approx_quantile(c) and approx_rows is not None:
                approx_rows.append((f"{name}.{c}", rel))
            else:
                close_rows.append((f"{name}.{c}", rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=os.environ.get("P2_CLOUD_RUN", os.path.join(ROOT, "cloud_run")))
    ap.add_argument("--out", default=os.environ.get("P2_CLOUD_PARITY", os.path.join(LOCAL, "cloud_parity.json")))
    a = ap.parse_args()
    C = os.path.join(a.cloud, "outputs")
    if not os.path.isdir(C):
        print(f"PARITY cloud outputs not found under {a.cloud}: not yet run")
        sys.exit(0)
    exact_fail, close_rows, checked, approx_rows, boundary_rows = [], [], [], [], []
    bound = json.load(open(BOUNDARY)) if os.path.exists(BOUNDARY) else {}

    li = {r["year"]: r for r in json.load(open(os.path.join(LOCAL, "ingest", "ingest_table.json")))}
    p = os.path.join(C, "ingest", "ingest_table.json")
    if os.path.exists(p):
        ci = {r["year"]: r for r in json.load(open(p))}
        for y in li:
            for k in ("rows", "stations", "first", "last"):
                if ci.get(y, {}).get(k) != li[y][k]:
                    exact_fail.append(f"ingest {y}.{k}: local {li[y][k]} cloud {ci.get(y, {}).get(k)}")
        checked.append("ingest rows/stations/span per year")
    ll = json.load(open(os.path.join(LOCAL, "ledger", "clean_ledger.json")))
    p = os.path.join(C, "ledger", "clean_ledger.json")
    if os.path.exists(p):
        cl = json.load(open(p))
        for x, y in zip(ll["rules"], cl["rules"]):
            for k in ("rule", "removed", "of", "adjusted"):
                if x.get(k) != y.get(k):
                    exact_fail.append(f"ledger {x['rule']}.{k}: local {x.get(k)} cloud {y.get(k)}")
        for k in ("rows_in", "rows_out", "stations"):
            if ll[k] != cl[k]:
                exact_fail.append(f"ledger {k}: local {ll[k]} cloud {cl[k]}")
        checked.append("cleaning ledger (every rule)")
    tables = [os.path.join("coverage", "coverage_by_year.csv")] + \
             [os.path.relpath(f, LOCAL) for f in sorted(glob.glob(os.path.join(LOCAL, "task1", "*.csv")))] + \
             [os.path.join("task2", f"{s}_sample_by_theta.csv") for s in ("train", "validation")] + \
             [os.path.join("task2", "f30_train_dist.csv")]
    for t in tables:
        lp, cp = os.path.join(LOCAL, t), os.path.join(C, t)
        if t.endswith("annual_balanced.csv") or not os.path.exists(cp):
            continue          # annual_balanced is written by the plotting step, which runs locally only
        compare_tables(t, pd.read_csv(lp), pd.read_csv(cp), exact_fail, close_rows, approx_rows, boundary_rows, bound)
        checked.append(t)
    worst = max(close_rows, key=lambda r: r[1]) if close_rows else ("--", 0.0)
    float_fail = [r for r in close_rows if r[1] > FLOAT_RTOL]

    # Task 2 validation metrics (close, not identical)
    mrows = []
    lv, cvp = os.path.join(LOCAL, "task2", "validation_metrics.csv"), os.path.join(C, "task2", "validation_metrics.csv")
    if os.path.exists(cvp):
        lv, cv = pd.read_csv(lv), pd.read_csv(cvp)
        lv, cv = lv[lv.theta == "all"].set_index("model"), cv[cv.theta == "all"].set_index("model")
        for m in lv.index:
            if m in cv.index:
                mrows.append({"model": m, **{f"{k}_local": float(lv.loc[m, k]) for k in ("auc_pr", "log_loss", "brier")},
                              **{f"{k}_cloud": float(cv.loc[m, k]) for k in ("auc_pr", "log_loss", "brier")}})
        sel_l = json.load(open(os.path.join(LOCAL, "task2", "selection.json")))
        sp = os.path.join(C, "task2", "selection.json")
        sel_c = json.load(open(sp)) if os.path.exists(sp) else {}
        same_choice = all(sel_c.get(f, {}).get("chosen") == sel_l[f]["chosen"] for f in ("A", "B")) if sel_c else None
    else:
        same_choice = None
    for r in mrows:
        print(f"PARITY task2 {r['model']}: AUC-PR local {r['auc_pr_local']:.4f} cloud {r['auc_pr_cloud']:.4f}; "
              f"log loss local {r['log_loss_local']:.5f} cloud {r['log_loss_cloud']:.5f}")
    print(f"PARITY checked: {'; '.join(checked)}")
    print(f"PARITY exact mismatches: {len(exact_fail)}")
    for e in exact_fail[:20]:
        print("    " + e)
    print(f"PARITY float columns compared: {len(close_rows)}; largest relative difference {worst[1]:.2e} ({worst[0]}); "
          f"above {FLOAT_RTOL:g}: {len(float_fail)}")
    aworst = max(approx_rows, key=lambda r: r[1]) if approx_rows else ("--", 0.0)
    print(f"PARITY approximate quantiles compared: {len(approx_rows)}; largest relative difference {aworst[1]:.2e} ({aworst[0]})")
    for r in boundary_rows:
        print(f"PARITY threshold count {r['column']}: {r['rows_differing']} rows differ, by at most {r['max_diff']}; "
              f"those rows have at least {r['min_on_threshold']} totals on the threshold; within bound: {r['within_bound']}")
    print(f"PARITY task2 same models chosen on 2023: {same_choice}")
    out = {"checked": checked, "exact_mismatches": exact_fail, "float_columns": len(close_rows),
           "float_max_rel": worst[1], "float_max_col": worst[0], "float_over_tol": len(float_fail),
           "task2": mrows, "same_choice": same_choice,
           "approx_quantiles": len(approx_rows), "approx_max_rel": aworst[1], "approx_max_col": aworst[0],
           "threshold_counts": boundary_rows}
    json.dump(out, open(a.out, "w"), indent=1)
    ok = not exact_fail and not float_fail
    print("PARITY: " + ("PASS" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
