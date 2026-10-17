"""Enforced check: the Task 2 evaluation plan is frozen by SHA-256 before any model sees 2024.

The plan (config/eval_plan.json) is the single source the Task 2 scripts read for the split, threshold grid,
sampling, features, candidate models, metrics and baselines. Commands:

  python checks/eval_plan.py freeze [--plan P]   validate the plan, then write config/eval_plan.lock (sha256, time).
                                                  Refuses an invalid plan (e.g. a random split, overlapping years,
                                                  test year not after training) and refuses to overwrite a lock
                                                  whose hash differs.
  python checks/eval_plan.py verify [--plan P]   recompute the hash and compare it to the lock. Exit 1 if changed.
  python checks/eval_plan.py status              print the plan line and whether the test has been run.

Library use (src/task2_*.py):
  load_frozen_plan()            -> plan dict, after verify; raises if the plan is not frozen or has changed
  guard_no_test_rows(df, plan)  raises if a DataFrame used for fitting/tuning contains a test-year row
  claim_test_run(plan)          raises unless the plan verifies and the test has never been run; then records the
                                single permitted test run in config/test_run.json (2024 is evaluated once).
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "config", "eval_plan.json")
LOCK = os.path.join(ROOT, "config", "eval_plan.lock")
TEST_RUN = os.path.join(ROOT, "config", "test_run.json")


class PlanError(RuntimeError):
    pass


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def validate(plan):
    errs = []
    sp = plan.get("split", {})
    if sp.get("method") != "by_year":
        errs.append(f"split method must be 'by_year' (time-ordered), got {sp.get('method')!r}; a random split puts "
                    "neighbouring 5-minute rows of one storm in both training and evaluation")
    tr, va, te = (set(sp.get(k, [])) for k in ("train_years", "validation_years", "test_years"))
    if not tr or not va or not te:
        errs.append("train, validation and test years must all be non-empty")
    if tr & va or tr & te or va & te:
        errs.append(f"split years overlap: train&val={sorted(tr & va)} train&test={sorted(tr & te)} "
                    f"val&test={sorted(va & te)}")
    if tr and va and te and not (max(tr) < min(va) and max(va) < min(te)):
        errs.append("years must be ordered: all train < all validation < all test")
    for k in ("random_split", "randomSplit", "fraction_split"):
        if k in sp:
            errs.append(f"split contains a random-split field {k!r}")
    th = plan.get("threshold_grid_mm", [])
    if not th or th != sorted(th) or len(set(th)) != len(th):
        errs.append("threshold grid must be non-empty, ascending and without repeats")
    if not plan.get("label", {}).get("horizon_minutes") == 30:
        errs.append("label horizon must be 30 minutes")
    fa = set(plan.get("features", {}).get("A", []))
    allowed_a = set(plan.get("features", {}).get("A_allowed_inputs", []))
    if not fa or not fa <= allowed_a:
        errs.append(f"Model A may use only station/time/threshold features; extra: {sorted(fa - allowed_a)}")
    for k in ("metrics", "baselines", "selection"):
        if not plan.get(k):
            errs.append(f"plan has no {k!r}")
    return errs


def freeze(plan_path=PLAN, lock_path=LOCK):
    plan = json.load(open(plan_path))
    errs = validate(plan)
    if errs:
        for e in errs:
            print("EVAL PLAN PROBLEM:", e)
        print("EVAL PLAN: REFUSED (not frozen)")
        return 1
    h = sha256_file(plan_path)
    if os.path.exists(lock_path):
        old = json.load(open(lock_path))
        if old["sha256"] != h:
            print(f"EVAL PLAN PROBLEM: a different plan is already frozen (sha256={old['sha256']}); "
                  f"this plan has sha256={h}")
            print("EVAL PLAN: REFUSED (lock exists)")
            return 1
        print(f"EVAL PLAN frozen sha256={h} (already, at {old['frozen_at']})")
        return 0
    lock = {"sha256": h, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "plan_file": os.path.relpath(plan_path, ROOT), "test_years": plan["split"]["test_years"]}
    json.dump(lock, open(lock_path, "w"), indent=1)
    print(f"EVAL PLAN frozen sha256={h} at {lock['frozen_at']}")
    return 0


def verify(plan_path=PLAN, lock_path=LOCK, quiet=False):
    if not os.path.exists(lock_path):
        if not quiet:
            print("EVAL PLAN PROBLEM: no lock; the plan has not been frozen")
            print("EVAL PLAN: REFUSED")
        return 1
    lock = json.load(open(lock_path))
    h = sha256_file(plan_path)
    errs = validate(json.load(open(plan_path)))
    if h != lock["sha256"] or errs:
        if not quiet:
            print(f"EVAL PLAN PROBLEM: plan sha256={h} differs from frozen sha256={lock['sha256']}"
                  if h != lock["sha256"] else "EVAL PLAN PROBLEM: " + "; ".join(errs))
            print("EVAL PLAN: REFUSED")
        return 1
    if not quiet:
        print(f"EVAL PLAN frozen sha256={h} verified (frozen_at={lock['frozen_at']})")
    return 0


def load_frozen_plan(plan_path=PLAN, lock_path=LOCK):
    if verify(plan_path, lock_path) != 0:
        raise PlanError("evaluation plan is not frozen or has changed; refusing to continue")
    return json.load(open(plan_path))


def guard_no_test_rows(df, plan, year_col="year"):
    from pyspark.sql import functions as F
    test = plan["split"]["test_years"]
    n = df.filter(F.col(year_col).isin(test)).limit(1).count()
    if n:
        raise PlanError(f"a table used for fitting or tuning contains rows from test years {test}; refusing")
    print(f"EVAL PLAN guard: no rows from test years {test} in the fitting/tuning table")


def claim_test_run(plan_path=PLAN, lock_path=LOCK, test_run_path=TEST_RUN):
    plan = load_frozen_plan(plan_path, lock_path)
    if os.path.exists(test_run_path):
        prev = json.load(open(test_run_path))
        raise PlanError(f"the test years were already evaluated at {prev['started_at']} under plan "
                        f"{prev['sha256']}; the test set is evaluated once")
    rec = {"sha256": sha256_file(plan_path),
           "started_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    json.dump(rec, open(test_run_path, "w"), indent=1)
    print(f"EVAL PLAN test run claimed under sha256={rec['sha256']}")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "verify", "status"])
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--lock", default=LOCK)
    a = ap.parse_args()
    if a.cmd == "freeze":
        sys.exit(freeze(a.plan, a.lock))
    if a.cmd == "verify":
        sys.exit(verify(a.plan, a.lock))
    rc = verify(a.plan, a.lock)
    print("EVAL PLAN test run:", json.load(open(TEST_RUN)) if os.path.exists(TEST_RUN) else "not run")
    sys.exit(rc)


if __name__ == "__main__":
    main()
