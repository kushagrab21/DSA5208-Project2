"""Task 2 figures from the Spark-computed metric tables in outputs/task2/.

  fig11_threshold_grid        share of f30 > theta in the training years (justifies the grid)
  fig12_per_threshold         AUC-PR and Brier skill per threshold, validation 2023 and test 2024:
                              chosen A, chosen B, climatology, persistence
  fig13_reliability           reliability diagrams, validation 2023 and test 2024 (pooled over thresholds)
  fig14_coefficients_B        standardised coefficients of the chosen Model B (logistic regression)
  fig15_gbt_calibration       validation 2023: mean prediction vs observed rate per threshold, B-LR vs B-GBT
"""
import json
import os

import numpy as np
import pandas as pd

import plotstyle as P

plt = P.plt
ROOT = P.ROOT
T = os.path.join(ROOT, "outputs", "task2")
C, INK2, MUTED, AXIS = P.C, P.INK2, P.MUTED, P.AXIS
NICE = {"theta": "threshold θ", "log1p_theta": "log(1+θ)", "mins_since_wet": "minutes since last wet slot",
        "isl_max_r30": "island max, last 30 min", "isl_wet30": "island share wet, last 30 min",
        "isl_wet_now": "island share wet now", "isl_mean_r30": "island mean, last 30 min",
        "log1p_r5": "log(1+rain last 5 min)", "log1p_r15": "log(1+rain last 15 min)",
        "log1p_r30": "log(1+rain last 30 min)", "log1p_r60": "log(1+rain last 60 min)",
        "log1p_r120": "log(1+rain last 2 h)", "log1p_r24h": "log(1+rain last 24 h)", "wet_now": "wet now",
        "r30_minus_theta": "rain last 30 min − θ", "nb_max_r30": "neighbours max, last 30 min",
        "nb_mean_r30": "neighbours mean, last 30 min", "nb_max_r5": "neighbours max, last 5 min",
        "nb_wet_now_frac": "neighbours share wet now", "nb_n_d": "neighbours reporting", "n120_d": "slots reported, last 2 h",
        "hour_sin": "hour (sin)", "hour_cos": "hour (cos)", "doy_sin": "day of year (sin)", "doy_cos": "day of year (cos)",
        "station_train_wet_rate": "station wet rate (train)"}


GRID = json.load(open(os.path.join(ROOT, "config", "eval_plan.json")))["threshold_grid_mm"]


def nice(f):
    if f.startswith("theta_oh_"):
        return f"θ = {GRID[int(f.split('_')[-1])]:g} mm (class)"
    if "_oh_" in f:
        g, v = f.split("_oh_", 1)
        return {"theta": "θ class", "hour": "hour", "month": "month", "season": "season", "station": "station"}.get(g, g) + " = " + v
    return NICE.get(f, f)


def main():
    plan = json.load(open(os.path.join(ROOT, "config", "eval_plan.json")))
    grid = plan["threshold_grid_mm"]
    sel = json.load(open(os.path.join(T, "selection.json")))
    a_name, b_name = sel["A"]["chosen"], sel["B"]["chosen"]
    show = [(b_name, "Model B (nowcast, LR)", C[0], "o"), (a_name, "Model A (strict, LR)", C[1], "s"),
            ("baseline_persistence", "persistence", C[2], "^"), ("baseline_climatology", "climatology", C[6], "D")]

    d = pd.read_csv(os.path.join(T, "f30_train_dist.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.bar(range(len(d)), 100 * d.base_rate, color=[C[0] if t in grid else AXIS for t in d.threshold_mm], width=0.65)
    for i, r in d.iterrows():
        ax.text(i, 100 * r.base_rate * 1.15, f"{int(r.positives):,}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(d)), [f"{t:g}" for t in d.threshold_mm])
    ax.set_xlabel("threshold θ (mm in 30 minutes)")
    ax.set_title("Prediction times with rain in the next 30 min > θ (%), training years 2017-2022\n"
                 "blue: thresholds in the grid; numbers: positive cases")
    ax.grid(axis="x", visible=False)
    P.save(fig, "fig11_threshold_grid")

    sets = [("validation_metrics.csv", "validation 2023"), ("test_metrics.csv", "test 2024")]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for j, (fn, lab_set) in enumerate(sets):
        m = pd.read_csv(os.path.join(T, fn))
        m = m[m.theta != "all"].copy()
        m["theta"] = m.theta.astype(float)
        for name, lab, col, mk in show:
            g = m[m.model == name].sort_values("theta")
            ax[0, j].plot(range(len(g)), g.auc_pr, color=col, marker=mk, ms=4, label=lab)
            ax[1, j].plot(range(len(g)), g.brier_skill, color=col, marker=mk, ms=4)
        base = m[m.model == show[0][0]].sort_values("theta")
        ax[0, j].plot(range(len(base)), base.base_rate, color=MUTED, ls="--", lw=1, label="no skill (base rate)")
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"AUC-PR, {lab_set}")
        ax[1, j].axhline(0, color=AXIS, lw=1)
        ax[1, j].set_title(f"Brier skill score, {lab_set}")
        ax[1, j].set_xticks(range(len(grid)), [f"{t:g}" for t in grid])
        ax[1, j].set_xlabel("threshold θ (mm)")
    ax[0, 0].legend(fontsize=6.5, loc="lower left")
    P.save(fig, "fig12_per_threshold")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for j, (fn, lab_set) in enumerate([("validation_reliability.csv", "validation 2023"),
                                       ("test_reliability.csv", "test 2024")]):
        r = pd.read_csv(os.path.join(T, fn))
        ax[j].plot([0, 1], [0, 1], color=AXIS, lw=1)
        for name, lab, col, mk in show:
            g = r[(r.model == name) & (r.rows >= 50)].sort_values("bin")
            ax[j].plot(g.mean_p, g.obs_rate, color=col, marker=mk, ms=4, label=lab)
        ax[j].set_xlabel("mean predicted probability (bin)")
        ax[j].set_title(f"Reliability, {lab_set}")
        ax[j].set_aspect("equal")
        ax[j].set_xlim(0, 1)
        ax[j].set_ylim(0, 1)
    ax[0].set_ylabel("observed frequency")
    ax[0].legend(fontsize=6.5, loc="upper left")
    P.save(fig, "fig13_reliability")

    cf = pd.read_csv(os.path.join(T, "coefficients_B.csv"))
    top = cf.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh([nice(f) for f in top.feature], top.std_coefficient,
            color=[C[0] if v > 0 else C[1] for v in top.std_coefficient], height=0.6)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("standardised coefficient (change in log-odds per 1 s.d.)")
    ax.set_title("Model B (logistic regression): 18 largest standardised coefficients\n"
                 "blue raises the probability, orange lowers it")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    P.save(fig, "fig14_coefficients_B")

    m = pd.read_csv(os.path.join(T, "validation_metrics.csv"))
    m = m[m.theta != "all"].copy()
    m["theta"] = m.theta.astype(float)
    gbt = sorted(sel["B"]["candidates"], key=lambda k: sel["B"]["candidates"][k]["log_loss"])
    gbt = [k for k in gbt if "gbt" in k][0]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    g0 = m[m.model == b_name].sort_values("theta")
    g1 = m[m.model == gbt].sort_values("theta")
    x = np.arange(len(g0))
    ax.plot(x, g0.base_rate, color=INK2, ls="--", lw=1.2, marker="x", ms=5, label="observed rate")
    ax.plot(x, g0.mean_p, color=C[0], marker="o", ms=4, label="B, logistic regression (chosen)")
    ax.plot(x, g1.mean_p, color=C[7], marker="v", ms=4, label="B, gradient-boosted trees")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{t:g}" for t in g0.theta])
    ax.set_xlabel("threshold θ (mm)")
    ax.set_title("Mean predicted probability vs observed rate, validation 2023")
    ax.legend(fontsize=7)
    P.save(fig, "fig15_gbt_calibration")


if __name__ == "__main__":
    main()
