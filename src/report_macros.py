"""Generate every number and table the report uses, from the pipeline's own outputs (no Spark needed).

Writes report/generated/values.tex (\\setval{key}{value} for every \\val{key} in the report) and
report/generated/tab_*.tex (table bodies). Sources: outputs/**, config/eval_plan.*, config/test_run.json,
runs/*/env.json (local) and cloud_run/runs/*/env.json (Dataproc, if the cloud run has been fetched).
Usage: python src/report_macros.py
"""
import datetime as _dt
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(ROOT, "outputs")
G = os.environ.get("P2_REPORT_GEN", os.path.join(ROOT, "report", "generated"))   # override only for tests
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
MON3 = [m[:3] for m in MONTHS]
V = {}


def put(key, val):
    V[key] = str(val)


def n0(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def f(x, d=1):
    return f"{x:,.{d}f}".replace(",", "{,}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\,\\%"


def signed(x, d=0):
    return f"{x:+.{d}f}".replace("-", "$-$").replace("+", "$+$")


def rj(*p):
    return json.load(open(os.path.join(O, *p)))


def rc(*p):
    return pd.read_csv(os.path.join(O, *p))


def table(name, header, rows, spec):
    os.makedirs(G, exist_ok=True)
    with open(os.path.join(G, f"tab_{name}.tex"), "w") as fh:
        fh.write(f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(str(x) for x in r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def short(model):
    return {"baseline_climatology": "Climatology baseline", "baseline_persistence": "Persistence baseline"}.get(
        model, model)


# ------------------------------------------------------------------ constants from the code and plan
plan = json.load(open(os.path.join(ROOT, "config", "eval_plan.json")))
lock = json.load(open(os.path.join(ROOT, "config", "eval_plan.lock")))
put("slot_min", 5)
put("slots_day", 288)
put("horizon", plan["label"]["horizon_minutes"])
put("n_years", 8)
put("grid", ", ".join(f"{t:g}" for t in plan["threshold_grid_mm"]))
put("grid_n", len(plan["threshold_grid_mm"]))
put("grid_max", f"{max(plan['threshold_grid_mm']):g}")
put("plan_sha", lock["sha256"][:12])
put("plan_frozen", lock["frozen_at"].replace("T", " ")[:16])
for split in ("train", "validation", "test"):
    sp = plan["sampling"][split]
    put(f"rate_active.{split}", f"{sp['active_rate']:g}")
    put(f"rate_inactive.{split}", f"{sp['inactive_rate']:g}")
put("k_nb", 6)
put("clim_pseudo", 20)
put("implausible_mm", 50)
put("full_day_pct", 90)
import sys  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src"))
import task1 as _t1  # noqa: E402  (constants only; no Spark session is started)
put("heavy_hour_mm", f"{_t1.HEAVY_HOUR_MM:g}")
put("heavy_day_mm", f"{_t1.HEAVY_DAY_MM:g}")
put("intense_mmh", f"{_t1.INTENSE_5MIN_MM * 12:g}")
put("per_hours", "10{,}000")
put("per_days", "1{,}000")
put("gauge_step", "0.2")
put("zero", "0")
put("rel_bins", 10)
put("rel_min_rows", 50)
put("smooth_days", 31)
put("index_base", 100)
put("n_coef_shown", 18)
put("gap_ratio_pct", 15)
put("gap_missed_pct", 10)
put("win_list", "5, 15, 30, 60 and 120 minutes and 24 hours")
put("win_slots", 24)
put("active_hours", 2)
put("share_win_lo", "12:00")
put("share_win_hi", "20:00")
put("cand_th_hi1", 30)
put("cand_th_hi2", 40)
put("example_th", 5)
st = json.load(open(os.path.join(O, "ledger", "stuck_runs.json")))
put("stuck_len", st["max_len"])
put("stuck_val", f"{st['max_len_value']:g}")
put("stuck_station", st["max_len_station"])
put("stuck_year", st["max_len_year"])
put("low_cov_days", 15)

# ------------------------------------------------------------------ ingest
ing = rj("ingest", "ingest_table.json")
rows = []
for r in ing:
    rows.append([r["year"], n0(r["rows"]), n0(r["csv_lines"]), r["stations"], r["first"][:19].replace("T", " "),
                 r["last"][:19].replace("T", " ")])
    put(f"ing_rows.{r['year']}", n0(r["rows"]))
    put(f"ing_stations.{r['year']}", r["stations"])
table("ingest", ["Year", "Parquet rows", "CSV data lines", "Stations", "First reading (SGT)", "Last reading (SGT)"],
      rows, "rrrrll")
put("ing_total", n0(sum(r["rows"] for r in ing)))
put("ing_st_min", min(r["stations"] for r in ing))
put("ing_st_max", max(r["stations"] for r in ing))
put("ing_missing_years", 0)
_first17 = [r for r in ing if r["year"] == 2017][0]["first"]
_f = pd.Timestamp(_first17[:19]).round("5min")
put("first2017", _f.strftime("%H:%M"))
put("first2017_raw", _first17[11:19])
put("first2017_day", f"{int(_first17[8:10])} {MONTHS[int(_first17[5:7]) - 1]}")

# ------------------------------------------------------------------ cleaning
led = rj("ledger", "clean_ledger.json")
why = {r["rule"]: r["why"] for r in led["rules"]}
rows = []
for r in led["rules"]:
    if r["rule"] == "snap":
        rows.append([r["rule"].replace("_", "\\_"), "0 (" + n0(r["adjusted"]) + " adjusted)", n0(r["of"]), why[r["rule"]]])
    else:
        rows.append([r["rule"].replace("_", "\\_"), n0(r["removed"]), n0(r["of"]), why[r["rule"]]])
table("ledger", ["Rule", "Removed", "Of", "What it removes"], rows, "lrrp{7.3cm}")
put("clean_in", n0(led["rows_in"]))
put("clean_out", n0(led["rows_out"]))
put("clean_removed", n0(led["rows_in"] - led["rows_out"]))
snap = [r for r in led["rules"] if r["rule"] == "snap"][0]
put("snap_n", n0(snap["adjusted"]))
put("snap_shift", snap["max_shift_seconds"])
dup = [r for r in led["rules"] if r["rule"] == "duplicate"][0]
put("dup_n", n0(dup["removed"]))
put("dup_conflict", dup.get("slots_with_conflicting_values", 0))
put("n_stations", led["stations"])
put("n_stations_moved", led["stations_with_moved_coords"])

cov = rc("coverage", "coverage_by_year.csv")
rows = []
for r in cov.itertuples():
    rows.append([r.year, r.stations, n0(r.station_days), pct(r.full_day_share, 1), pct(r.slot_share, 1), r.days_with_data])
    put(f"cov_share.{r.year}", pct(r.slot_share, 0))
    put(f"cov_days.{r.year}", r.days_with_data)
table("coverage", ["Year", "Stations", "Station-days", "Days $\\geq$ 90\\,\\% slots", "Slots reported", "Days with data"],
      rows, "rrrrrr")
put("cov_share_min", pct(cov.slot_share.min(), 0))
put("cov_share_max", pct(cov.slot_share.max(), 0))
mb = rc("task1", "monthly_by_year.csv")
dd = mb.pivot(index="year", columns="month", values="days_with_data")
put("days.2018.aug", int(dd.loc[2018, 8]))
put("days.2018.feb", int(dd.loc[2018, 2]))
put("days.2023.jan", int(dd.loc[2023, 1]))
put("days.2021.jan", int(dd.loc[2021, 1]))
put("days.2019.jul", int(dd.loc[2019, 7]))
res = rc("task1", "resolution_by_year.csv")
for r in res.itertuples():
    put(f"mult02.{r.year}", pct(r.mult_0p2 / r.wet_slots, 0))
    put(f"max5.{r.year}", f(r.max_5min))
put("min_pos.2020", f"{res[res.year == 2020].min_positive.iloc[0]:g}")

# ------------------------------------------------------------------ Task 1
S = rj("task1", "summary.json")
for k in ("wettest_month_rate", "driest_month_rate"):
    put(k, f(S[k]))
put("wettest_month", MONTHS[MON3.index(S["wettest_month"])])
put("driest_month", MONTHS[MON3.index(S["driest_month"])])
for m, v in S["monthly_rate"].items():
    put(f"mrate.{m}", f(v))
for y, v in S["annual_balanced_mm"].items():
    put(f"ann.{y}", n0(v))
ann = rc("task1", "annual_balanced.csv")
put("ann_mean", n0(ann.balanced_mm.mean()))
for r in ann.itertuples():
    put(f"panel.{r.year}", n0(r.panel_balanced_mm))
    put(f"freq.{r.year}", pct(r.wet_freq, 2))
    put(f"int.{r.year}", f(r.intensity_mmh))
    put(f"freqidx.{r.year}", f"{100 * r.wet_freq / ann.wet_freq.mean():.0f}")
    put(f"intidx.{r.year}", f"{100 * r.intensity_mmh / ann.intensity_mmh.mean():.0f}")
put("panel_n", int(ann.panel_stations.min()))
put("panel_maxdev", pct((ann.panel_balanced_mm / ann.balanced_mm - 1).abs().max(), 1))
put("naive_ratio_21_17", f"{ann.set_index('year').naive_network_mm[2021] / ann.set_index('year').naive_network_mm[2017]:.2f}")
put("bal_ratio_21_17", f"{ann.set_index('year').balanced_mm[2021] / ann.set_index('year').balanced_mm[2017]:.2f}")
put("corr_freq", f"{S['corr_annual_freq']:.2f}")
put("corr_int", f"{S['corr_annual_intensity']:.2f}")
seas = {"NE monsoon (Dec-Mar)": "ne", "Inter-monsoon I (Apr-May)": "imi", "SW monsoon (Jun-Sep)": "sw",
        "Inter-monsoon II (Oct-Nov)": "imii"}
for s, k in seas.items():
    put(f"peak_hour.{k}", f"{S['diurnal_peak_hour'][s]:02d}:00--{S['diurnal_peak_hour'][s] + 1:02d}:00")
    put(f"peak_rate.{k}", f(S["diurnal_peak_rate_mmh"][s], 2))
    put(f"min_rate.{k}", f(S["diurnal_min_rate_mmh"][s], 2))
    put(f"season_rate.{k}", f(S["season_rate"][s]))
    put(f"corr_lon.{k}", signed(S["corr_lon_rate"][s], 2))
    for y, v in S["season_anom"][s].items():
        put(f"sanom.{k}.{y}", signed(v))
put("share_12_20", pct(S["share_rain_12_20"], 0))
for m in MON3:
    put(f"wetpct.{m}", f(S["month_wet_pct"][m]))
    put(f"intmon.{m}", f(S["month_intensity_mmh"][m]))
    put(f"intshare.{m}", pct(S["month_intense_share"][m], 0))
    put(f"heavyh_mon.{m}", f(S["heavy_hours_per_10k_month"][m]))
put("intense_share_all", pct(S["intense_share_all"], 0))
put("wet_pct_all", f(S["wet_pct_all"]))
for y in S["heavy_hours_per_10k"]:
    put(f"heavyh.{y}", f(S["heavy_hours_per_10k"][y]))
    put(f"heavyd.{y}", f(S["heavy_days_per_1k"][y]))
    put(f"maxhour.{y}", f(S["max_hour_mm"][y]))
    put(f"medmaxhour.{y}", f(S["median_annual_max_hour"][y]))
    put(f"wetday.{y}", pct(S["wet_day_share"][y], 0))
    put(f"spellmax.{y}", S["wet_spell_max_len"][y])
put("heavy_hours_total", n0(S["heavy_hours_total"]))
put("station_hours_total", n0(S["complete_station_hours_total"]))
put("corr_lon_all", signed(S["corr_lon_rate_all"], 2))
put("n_st_1yr", S["n_stations_ge1yr"])
put("st_rate_min", f(S["station_rate_min"]))
put("st_rate_max", f(S["station_rate_max"]))
put("st_rate_min_id", S["station_rate_min_id"])
put("st_rate_max_id", S["station_rate_max_id"])
mrate = mb.copy()
mrate["rate"] = mrate.rain_sum / mrate.slots * 288
clim = mrate.groupby("month").apply(lambda g: g.rain_sum.sum() / g.slots.sum() * 288, include_groups=False)
for (y, m), r in mrate.set_index(["year", "month"]).iterrows():
    put(f"manom.{y}.{MON3[m - 1]}", signed(100 * (r.rate / clim[m] - 1)))
    put(f"mr.{y}.{MON3[m - 1]}", f(r.rate))
sp = rc("task1", "wet_spells_by_year.csv")
put("spell_mean_all", f((sp.mean_len * sp.spells).sum() / sp.spells.sum(), 2))

# ------------------------------------------------------------------ MSS validation
if os.path.exists(os.path.join(O, "mss", "agreement_summary.json")):
    A = rj("mss", "agreement_summary.json")
    put("mss_days", n0(A["days"]))
    put("mss_stations", int(A["stations"]))
    put("mss_corr", f"{A['corr']:.3f}")
    put("mss_mae", f(A["mae"], 3))
    put("mss_meandiff", signed(A["mean_diff"], 3))
    put("mss_within02", pct(A["within_0p2"], 1))
    put("mss_within1", pct(A["within_1mm_or_5pct"], 1))
    put("mss_ratio", f"{A['ratio_total']:.3f}")
    put("mss_wet_disagree", n0(A["wet1_disagree"]))
    put("mss_maxdiff", f(A["max_abs_diff"]))
    pairs = rc("mss", "complete_day_pairs.csv")
    put("mss_big20", n0((pairs["diff"].abs() > 20).sum()))
    put("mss_big5", n0((pairs["diff"].abs() > 5).sum()))
    hd = _t1.HEAVY_DAY_MM
    put("mss_heavy_ours", n0((pairs.our_mm >= hd).sum()))
    put("mss_heavy_off", n0((pairs.mss_mm >= hd).sum()))
    put("mss_false_heavy", n0(((pairs.our_mm >= hd) & (pairs.mss_mm < 5)).sum()))
    put("mss_missed_heavy", n0(((pairs.mss_mm >= hd) & (pairs.our_mm < 5)).sum()))
    put("mss_small", 5)
    pairs["year"] = pairs.date.astype(str).str[:4]
    for y, g in pairs.groupby("year"):
        put(f"mss_heavyrate_ours.{y}", f(1000 * (g.our_mm >= hd).mean()))
        put(f"mss_heavyrate_off.{y}", f(1000 * (g.mss_mm >= hd).mean()))
    sp_ = pairs[pairs["diff"] > 0].sort_values("diff", ascending=False).iloc[0]
    zr_ = pairs[pairs["diff"] < 0].sort_values("diff").iloc[0]
    for k_, r_ in (("spur", sp_), ("zero", zr_)):
        put(f"mss_{k_}.st", r_.station_id)
        put(f"mss_{k_}.date", str(r_.date))
        put(f"mss_{k_}.ours", f(r_.our_mm))
        put(f"mss_{k_}.off", f(r_.mss_mm))
    lg = rc("mss", "largest_complete_day_differences.csv")
    for i, r in enumerate(lg.head(3).itertuples()):
        put(f"mss_lg{i}.st", r.station_id)
        put(f"mss_lg{i}.date", str(r.date))
        put(f"mss_lg{i}.ours", f(r.our_mm))
        put(f"mss_lg{i}.off", f(r.mss_mm))
    dc = rc("mss", "official_rain_by_day_class.csv").set_index("cls")
    for k in dc.index:
        put(f"mss_cls.{k}", pct(dc.loc[k, "share_of_official_rain"], 1))
        put(f"mss_clsdays.{k}", n0(dc.loc[k, "station_days"]))
    gs = rc("mss", "gap_cost_station_month.csv")
    t = gs.mss_month_mm.sum()
    put("mss_gap_outage", pct(gs.mss_on_network_outage.sum() / t, 1))
    put("mss_gap_station", pct(gs.mss_on_station_gap.sum() / t, 1))
    put("mss_gap_partial", pct(gs.mss_on_partial_days.sum() / t, 1))
    put("mss_active_sm", n0(len(gs)))
    rows = []
    for r in gs.head(10).itertuples():
        rows.append([r.station_id, f"{MON3[r.month - 1]} {r.year}", f(r.mss_month_mm), f(r.our_month_mm),
                     r.missing_days, f(r.mss_on_missing_days), pct(r.missed_share, 0), f(r.mss_on_network_outage)])
    table("gaps", ["Station", "Month", "Official (mm)", "Ours (mm)", "Days missing", "Official on them (mm)",
                   "Share", "of which network outage"], rows, "llrrrrrr")
    s24 = gs[(gs.station_id == "S24") & (gs.year == 2021) & (gs.month == 1)]
    if len(s24):
        r = s24.iloc[0]
        put("s24_jan21_off", f(r.mss_month_mm))
        put("s24_jan21_ours", f(r.our_month_mm))
        put("s24_jan21_missed", f(r.mss_on_missing_days))
    raw24 = pd.read_csv(os.path.join(ROOT, "data", "mss", "raw", "DAILYDATA_S24_202101.csv"), encoding="utf-8-sig")
    raw24.columns = [c.strip() for c in raw24.columns]
    rr = raw24.set_index("Day")["Daily Rainfall Total (mm)"].astype(str).str.strip().astype(float)
    put("s24_jan1", f(rr[1]))
    put("s24_jan2", f(rr[2]))
    sp_ = rc("mss", "network_outage_spans.csv")
    sp_["days"] = sp_.days_without_any_station
    big = sp_.sort_values("days", ascending=False)
    put("outage_spans", len(sp_))
    put("outage_span_days_total", int(sp_.days.sum()))
    j21s = sp_[sp_.last_reading.str.startswith("2021-01")]
    if len(j21s):
        lo, hi = pd.Timestamp(j21s.last_reading.iloc[0]), pd.Timestamp(j21s.first_after.iloc[0])
        put("jan21_last", lo.strftime("%H:%M"))
        put("jan21_last_day", f"{lo.day} {MONTHS[lo.month - 1]}")
        put("jan21_first_day", f"{hi.day} {MONTHS[hi.month - 1]}")
        put("jan21_first", hi.strftime("%H:%M"))
        put("jan21_gap_days", f"{lo.day}--{(hi - pd.Timedelta(days=1)).day} {MONTHS[lo.month - 1]}")
    put("s24_d1", "1 January")
    put("s24_d2", "2 January")
    od = rc("mss", "network_outage_days.csv")
    put("mss_outage_days", len(od))
    put("outage1.date", str(od.date.iloc[0]))
    put("outage1.mean", f(od.official_mean_mm.iloc[0]))
    put("outage1.max", f(od.official_max_mm.iloc[0]))
    put("outage1.n", int(od.stations.iloc[0]))
    mr = rc("mss", "monthly_rate_ours_vs_mss.csv")
    put("mss_month_median_ratio", f"{mr.ratio.median():.2f}")
    for r in mr.itertuples():
        mk = f"{r.year}.{MON3[r.month - 1]}"
        put(f"mssratio.{mk}", f"{r.ratio:.2f}")
        put(f"mssours.{mk}", f(r.our_rate, 2))
        put(f"mssoff.{mk}", f(r.mss_rate, 2))
        put(f"mssmissed.{mk}", pct(r.missed_share, 0))
        put(f"mssoffanom.{mk}", signed(100 * (r.mss_rate / clim[r.month] - 1)))
    bad = mr[((mr.ratio < 0.85) | (mr.ratio > 1.15)) & (mr.missed_share > 0.1)].sort_values("ratio")
    odd = mr[((mr.ratio < 0.85) | (mr.ratio > 1.15)) & (mr.missed_share <= 0.1)]
    put("mss_odd_n", len(odd))
    put("mss_odd_list", ", ".join(f"{MON3[r.month - 1]} {r.year} ({r.ratio:.2f})" for r in odd.itertuples()))
    rows = []
    for r in bad.itertuples():
        rows.append([f"{MON3[r.month - 1]} {r.year}", f(r.our_rate, 2), f(r.mss_rate, 2), f"{r.ratio:.2f}",
                     pct(r.missed_share, 0), V[f"manom.{r.year}.{MON3[r.month - 1]}"]])  # the Task 1 network anomaly, as in the text and figure
    put("mss_bad_months", len(bad))
    table("gapmonths", ["Month", "Ours (mm/day)", "Official (mm/day)", "Ratio", "Official rain on days we lack",
                        "Our anomaly (\\%)"], rows, "lrrrrr")
    put("mss_ratio_range_lo", f"{mr.ratio.quantile(0.1):.2f}")
    put("mss_ratio_range_hi", f"{mr.ratio.quantile(0.9):.2f}")
    # how much would January 2021's official rate change the anomaly
    j21 = mr[(mr.year == 2021) & (mr.month == 1)]
    if len(j21):
        put("jan21_off_anom", signed(100 * (j21.mss_rate.iloc[0] / clim[1] - 1)))
    put("mss_n_files", len(glob.glob(os.path.join(ROOT, "data", "mss", "raw", "DAILYDATA_*.csv"))))
    put("mss_n_probe", len(json.load(open(os.path.join(ROOT, "data", "mss", "stations_found.json")))))

# ------------------------------------------------------------------ Task 2
fd = rc("task2", "f30_train_dist.csv").set_index("threshold_mm")
for t in plan["threshold_grid_mm"] + [30, 40]:
    put(f"f30pos.{t:g}", n0(fd.loc[t, "positives"]))
    put(f"f30rate.{t:g}", pct(fd.loc[t, "base_rate"], 2))
put("f30_rows", n0(fd.rows.iloc[0]))
q = rc("task2", "f30_train_wet_quantiles.csv").set_index("quantile").f30_mm
put("q50", f(q[0.5]))
put("q90", f(q[0.9]))
put("q99", f(q[0.99]))
for split in ("train", "validation", "test"):
    p = os.path.join(O, "task2", f"{split}_sample_by_theta.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        put(f"n_{split}", n0(d.rows.sum()))
        put(f"pos_{split}", n0(d.positives.sum()))
sel = rj("task2", "selection.json")
NAMES = {sel["A"]["chosen"]: "A: logistic regression", sel["B"]["chosen"]: "B: logistic regression"}


def label(m):
    if m in NAMES:
        return NAMES[m] + " (chosen)"
    if m.startswith("baseline"):
        return short(m)
    fam, rest = m.split("_", 1)
    est = "logistic regression" if rest.startswith("lr") else "gradient-boosted trees"
    hp = rest.split("__")[1].replace("regParam", "reg ").replace("_elasticNetParam0", "").replace(
        "maxDepth", "depth ").replace("_maxIter", ", trees ").replace("_stepSize0.1", "")
    return f"{fam}: {est} ({hp})"


vm = rc("task2", "validation_metrics.csv")
mono = rc("task2", "validation_monotonicity.csv").set_index("model")
ft = rc("task2", "fit_timing.csv").set_index("model")
pooled = vm[vm.theta == "all"]
order = ["baseline_climatology", "baseline_persistence"] + [m for m in pooled.model if m.startswith("A_")] + \
        [m for m in pooled.model if m.startswith("B_")]
rows = []
for m in order:
    r = pooled[pooled.model == m].iloc[0]
    fit = f"{ft.loc[m, 'fit_seconds']:.0f}" if m in ft.index else "--"
    rows.append([label(m), f"{r.auc_roc:.3f}", f"{r.auc_pr:.3f}", f"{r.brier:.5f}", f"{r.brier_skill:.3f}",
                 f"{r.log_loss:.4f}", f"{r.mean_p:.4f}", pct(mono.loc[m, "share_rises"], 2), fit])
table("val", ["Model (validation 2023)", "AUC-ROC", "AUC-PR", "Brier", "BSS", "Log loss", "Mean $p$",
              "$p$ rises with $\\theta$", "Fit (s)"], rows, "lrrrrrrrr")
put("val_base", f"{pooled.base_rate.iloc[0]:.4f}")


def metric_keys(df, prefix):
    for r in df.itertuples():
        m = r.model
        key = ("clim" if m == "baseline_climatology" else "pers" if m == "baseline_persistence" else
               "A" if m == sel["A"]["chosen"] else "B" if m == sel["B"]["chosen"] else
               "Bgbt" if m.startswith("B_gbt__maxDepth7") else "Bgbtfive" if m.startswith("B_gbt") else
               "Agbt" if m.startswith("A_gbt") else None)
        if key is None:
            continue
        th = "all" if r.theta == "all" else f"{float(r.theta):g}"
        put(f"{prefix}.{key}.{th}.aucroc", f"{r.auc_roc:.3f}")
        put(f"{prefix}.{key}.{th}.aucpr", f"{r.auc_pr:.3f}")
        put(f"{prefix}.{key}.{th}.brier", f"{r.brier:.5f}")
        put(f"{prefix}.{key}.{th}.bss", f"{r.brier_skill:.3f}")
        put(f"{prefix}.{key}.{th}.ll", f"{r.log_loss:.4f}")
        put(f"{prefix}.{key}.{th}.meanp", f"{r.mean_p:.4f}")
        put(f"{prefix}.{key}.{th}.base", f"{r.base_rate:.4f}")


metric_keys(vm, "v")
put("mono.Bgbt", pct(mono.loc[[m for m in mono.index if m.startswith("B_gbt__maxDepth7")][0], "share_rises"], 1))
put("mono.Bgbt.max", f"{mono.loc[[m for m in mono.index if m.startswith('B_gbt__maxDepth7')][0], 'max_rise']:.2f}")
if os.path.exists(os.path.join(O, "task2", "test_metrics.csv")):
    tm = rc("task2", "test_metrics.csv")
    metric_keys(tm, "t")
    tmono = rc("task2", "test_monotonicity.csv").set_index("model")
    rows = []
    for th in ["all"] + [f"{t:g}" for t in plan["threshold_grid_mm"]]:
        sub = tm[tm.theta.astype(str).isin([th, th + ".0"]) if th != "all" else tm.theta == "all"]
        if not len(sub):
            continue
        base = sub.base_rate.iloc[0]
        cells = [("all" if th == "all" else th), f"{base:.4f}"]
        for m in ["baseline_climatology", "baseline_persistence", sel["A"]["chosen"], sel["B"]["chosen"]]:
            r = sub[sub.model == m].iloc[0]
            cells += [f"{r.auc_pr:.3f}", f"{r.brier_skill:.3f}"]
        rows.append(cells)
    table("test", ["$\\theta$ (mm)", "Base rate", "Clim.\\ AUC-PR", "BSS", "Pers.\\ AUC-PR", "BSS", "A AUC-PR", "BSS",
                   "B AUC-PR", "BSS"], rows, "lrrrrrrrrr")
    rows = []
    for m in ["baseline_climatology", "baseline_persistence", sel["A"]["chosen"], sel["B"]["chosen"]]:
        r = tm[(tm.model == m) & (tm.theta == "all")].iloc[0]
        rows.append([label(m).replace(" (chosen)", ""), f"{r.auc_roc:.3f}", f"{r.auc_pr:.3f}", f"{r.brier:.5f}",
                     f"{r.brier_skill:.3f}", f"{r.log_loss:.4f}", f"{r.mean_p:.4f}", pct(tmono.loc[m, "share_rises"], 2)])
    table("testpooled", ["Model (test 2024, once)", "AUC-ROC", "AUC-PR", "Brier", "BSS", "Log loss", "Mean $p$",
                         "$p$ rises with $\\theta$"], rows, "lrrrrrrr")
    tr = json.load(open(os.path.join(ROOT, "config", "test_run.json")))
    put("test_started", tr["started_at"].replace("T", " ")[:16])
    rel = rc("task2", "test_reliability.csv")
    rb = rel[rel.model == sel["B"]["chosen"]]
    put("t_rel_B_maxgap", f"{(rb[rb.rows >= 50].mean_p - rb[rb.rows >= 50].obs_rate).abs().max():.3f}")
cf = rc("task2", "coefficients_B.csv")
num = cf[~cf.feature.str.contains("_oh_")]
put("coefB.top1", num.iloc[0].feature.replace("_", "\\_"))
for feat in ("mins_since_wet", "isl_max_r30", "isl_wet30", "isl_wet_now", "log1p_r5", "nb_max_r5", "log1p_r30",
             "nb_wet_now_frac", "log1p_theta"):
    rr = cf[cf.feature == feat]
    if len(rr):
        put(f"coefB.{feat}", signed(rr.std_coefficient.iloc[0], 2))
put("n_coef_B", len(cf))
put("n_coef_A", len(rc("task2", "coefficients_A.csv")))

# ------------------------------------------------------------------ computing environment
JOBS = ["ingest", "ingest_check", "clean", "clean_ledger", "coverage_check", "stuck_runs", "task1", "task2_features",
        "task2_sample", "task2_train", "task2_test", "mss_validate"]
NICEJOB = {"ingest": "ingest (CSV $\\to$ Parquet)", "ingest_check": "ingest check", "clean": "cleaning + ledger",
           "clean_ledger": "ledger check", "coverage_check": "coverage", "stuck_runs": "stuck-gauge check", "task1": "Task 1 aggregation",
           "task2_features": "Task 2 features", "task2_sample": "Task 2 samples + baselines",
           "task2_train": "Task 2 fit + validation", "task2_test": "Task 2 test (once)", "mss_validate": "MSS validation"}


# runs made by checks/qualify.py on broken fixtures before it tagged its runs (see the qualification log)
QUALIFY_RUNS = {"20261017T143429-ingest_check", "20261017T143444-clean_ledger"}


def env_runs(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "env.json"))):
        e = json.load(open(p))
        if e["run_id"].startswith("qualify") or e["run_id"] in QUALIFY_RUNS:
            continue
        if e.get("job") in JOBS:
            prev = out.get(e["job"])
            # latest run of each job; a run with a recorded wall time wins over one without
            if prev is None or e.get("wall_seconds") is not None or prev.get("wall_seconds") is None:
                out[e["job"]] = e
    return out


def env_block(prefix, runs, title):
    if not runs:
        put(f"{prefix}.status", "not yet run")
        return False
    e0 = runs.get("task2_train") or list(runs.values())[-1]
    h = e0["host"]
    put(f"{prefix}.status", "run")
    put(f"{prefix}.spark", e0["spark_version"])
    put(f"{prefix}.java", e0["java_version"])
    put(f"{prefix}.python", e0["python_version"])
    put(f"{prefix}.master", e0["master"].replace("[", "{[}").replace("]", "{]}"))
    put(f"{prefix}.cpu", h.get("cpu"))
    put(f"{prefix}.ncpu", h.get("logical_cpus"))
    put(f"{prefix}.ram", f"{(h.get('ram_bytes') or 0) / 2 ** 30:.0f}")
    put(f"{prefix}.os", h.get("os"))
    put(f"{prefix}.driver_mem", e0.get("driver_memory") or "default")
    put(f"{prefix}.exec_inst", e0.get("executor_instances") or "--")
    put(f"{prefix}.exec_cores", e0.get("executor_cores") or "--")
    put(f"{prefix}.exec_mem", e0.get("executor_memory") or "--")
    put(f"{prefix}.parallelism", e0.get("default_parallelism"))
    put(f"{prefix}.shuffle", e0.get("shuffle_partitions") or "default")
    put(f"{prefix}.image", e0.get("dataproc_image") or "--")
    da = e0.get("dynamic_allocation")
    put(f"{prefix}.dynalloc", "off" if da == "false" else "on" if da == "true" else "at its default")
    rows, tot = [], 0.0
    for j in JOBS:
        if j in runs:
            e = runs[j]
            ws = e.get("wall_seconds")
            rows.append([NICEJOB[j], e["run_id"].replace("_", "\\_"),
                         f"{ws:.0f}" if ws is not None else "not recorded$^{\\dagger}$"])
            tot += ws or 0.0
    table(prefix.replace(".", "_") + "_runs", ["Stage", "Run id (\\texttt{runs/<id>/env.json})", "Wall time (s)"],
          rows, "llr")
    put(f"{prefix}.total_min", f"{tot / 60:.0f}")
    put(f"{prefix}.n_runs", len(rows))
    return True


LOCAL_RUNS = env_runs(os.path.join(ROOT, "runs"))
env_block("env", LOCAL_RUNS, "local")
# The Dataproc run, if its outputs were fetched into cloud_run/ (P2_CLOUD_RUN overrides the directory, for tests).
CLOUD = os.environ.get("P2_CLOUD_RUN", os.path.join(ROOT, "cloud_run"))
PARITY = os.environ.get("P2_CLOUD_PARITY", os.path.join(O, "cloud_parity.json"))
CLOUD_RUNS = env_runs(os.path.join(CLOUD, "runs"))
# On the cluster, env.json is copied to GCS before env_capture's exit hook writes wall_seconds, so the cloud wall
# times come from the Dataproc jobs API instead (cloud_run/job_times.json: driver start to driver exit).
JT = os.path.join(CLOUD, "job_times.json")
cloud_wall_from_api = False
if os.path.exists(JT):
    for j, t in json.load(open(JT))["jobs"].items():
        if j in CLOUD_RUNS and CLOUD_RUNS[j].get("wall_seconds") is None:
            CLOUD_RUNS[j]["wall_seconds"] = t["seconds"]
            cloud_wall_from_api = True
    put("cloud.job_overhead", f"{json.load(open(JT))['jobs'].get('envcheck', {}).get('seconds', 0):.0f}")
cj = os.path.join(CLOUD, "cluster.json")
cloud_ok = bool(CLOUD_RUNS) and os.path.exists(cj)
for stale in ("tab_cloud_runs.tex", "tab_cloud_compare.tex"):
    if os.path.exists(os.path.join(G, stale)):
        os.remove(os.path.join(G, stale))
if cloud_ok:
    env_block("cloud", CLOUD_RUNS, "Dataproc")
    c = json.load(open(cj))
    cfg = c.get("config", {})
    mc, wc = cfg.get("masterConfig", {}), cfg.get("workerConfig", {})
    put("cloud.cluster", c.get("clusterName", "--"))
    put("cloud.master_type", mc.get("machineTypeUri", "--").split("/")[-1])
    put("cloud.masters", mc.get("numInstances", 1))
    put("cloud.workers", wc.get("numInstances", "--"))
    put("cloud.worker_type", wc.get("machineTypeUri", "--").split("/")[-1])
    put("cloud.disk", f"{wc.get('diskConfig', {}).get('bootDiskSizeGb', '--')}\\,GB {wc.get('diskConfig', {}).get('bootDiskType', '')}")
    put("cloud.image_cfg", cfg.get("softwareConfig", {}).get("imageVersion", "--"))
    put("cloud.zone", cfg.get("gceClusterConfig", {}).get("zoneUri", "--").split("/")[-1])
    rows = []
    for j in JOBS:
        if j in CLOUD_RUNS and j in LOCAL_RUNS:
            lw, cw = LOCAL_RUNS[j].get("wall_seconds"), CLOUD_RUNS[j].get("wall_seconds")
            rows.append([NICEJOB[j], f"{lw:.0f}" if lw is not None else "not recorded",
                         f"{cw:.0f}" if cw is not None else "--",
                         f"{cw / lw:.1f}" if (lw and cw) else "--"])
    table("cloud_compare", ["Stage", "Local (s)", "Dataproc (s)", "Dataproc / local"], rows, "lrrr")
    par = json.load(open(PARITY)) if os.path.exists(PARITY) else None
    if par:
        put("par.exact", len(par["exact_mismatches"]))
        put("par.nchecked", len(par["checked"]))
        put("par.floatcols", par["float_columns"])
        put("par.floatmax", f"{par['float_max_rel']:.1e}".replace("e-0", "e-").replace("e", "\\times10^{") + "}" if par["float_max_rel"] > 0 else "0")
        bl = [r for r in par["task2"] if r["model"] == sel["B"]["chosen"]]
        if bl:
            put("par.B.aucpr.l", f"{bl[0]['auc_pr_local']:.4f}")
            put("par.B.aucpr.c", f"{bl[0]['auc_pr_cloud']:.4f}")
            put("par.B.ll.l", f"{bl[0]['log_loss_local']:.5f}")
            put("par.B.ll.c", f"{bl[0]['log_loss_cloud']:.5f}")
        dm = max((abs(r["auc_pr_local"] - r["auc_pr_cloud"]) for r in par["task2"]), default=None)
        put("par.t2maxdiff", f"{dm:.4f}" if dm is not None else "--")
        thr = par.get("threshold_counts", [])
        put("par.thr_cols", len(thr))
        put("par.thr_maxdiff", max((r["max_diff"] for r in thr), default=0))
        put("par.thr_ok", "all" if all(r["within_bound"] for r in thr) else "not all")
        put("par.approx_n", par.get("approx_quantiles", 0))
        put("par.approx_max", f"{100 * par.get('approx_max_rel', 0):.1f}\\,\\%")
        put("par.same_choice", {True: "the same", False: "different", None: "(not available)"}[par["same_choice"]])
if not cloud_ok:
    put("cloud.status", "not yet run")
put("cloud_ready", "yes" if cloud_ok else "no")
with open(os.path.join(G, "cloud_limitation.tex"), "w") as fh:
    if cloud_ok:
        fh.write("\\item \\emph{Cloud security.} The Dataproc nodes had public IP addresses, which the cluster needed to "
                 "download the data from data.gov.sg itself, and Google warned that the project's default network firewall "
                 "allowed ingress from any address. The clusters existed only for this run and were deleted afterwards, "
                 "with a 30-minute idle limit as a backstop.\n")
with open(os.path.join(G, "cloud_section.tex"), "w") as fh:
    if cloud_ok:
        V_ = V
        fh.write(
            "We also ran the pipeline once on Google Cloud Dataproc, from ingestion to the Task 2 fit and validation. "
            "The 2024 test and every number reported elsewhere come from the local runs. "
            f"Cluster \\code{{{V_['cloud.cluster']}}}, zone \\code{{{V_['cloud.zone']}}}, image "
            f"\\code{{{V_['cloud.image_cfg']}}}: {V_['cloud.masters']} {V_['cloud.master_type']} master and "
            f"{V_['cloud.workers']} {V_['cloud.worker_type']} workers, each with a boot disk of {V_['cloud.disk']}; "
            f"Spark {V_['cloud.spark']} on Java {V_['cloud.java']} and Python {V_['cloud.python']}, master "
            f"\\code{{{V_['cloud.master']}}}; {V_['cloud.exec_inst']} executors with {V_['cloud.exec_cores']} cores and "
            f"{V_['cloud.exec_mem']} each, driver memory {V_['cloud.driver_mem']}, {V_['cloud.shuffle']} shuffle "
            f"partitions, dynamic allocation {V_['cloud.dynalloc']}.\n\n"
            "These three machines took the project's whole vCPU quota, so the cluster could not be larger. We sized the "
            f"executors as Lecture 6 does. Each machine has {V_['cloud.ncpu']} vCPUs and {V_['cloud.ram']}\\,GiB of "
            "memory. Lecture 6 recommends three to five cores per executor and keeping some capacity on each worker for "
            "the cluster manager and the operating system, so we ran one executor per worker with "
            f"{V_['cloud.exec_cores']} cores and left the remaining vCPU free. Executor memory, plus an overhead that "
            "Lecture 6 puts at about a tenth of it (more for PySpark), has to fit within what YARN offers on a worker, "
            f"which is less than the machine's {V_['cloud.ram']}\\,GiB; {V_['cloud.exec_mem']} fits with room to spare "
            "for the Python worker processes. We turned dynamic allocation off so that every job ran on exactly these "
            "executors. The raw CSVs were downloaded from data.gov.sg straight into Cloud Storage by a job on the "
            "cluster (\\code{cloud/fetch\\_raw\\_on\\_cluster.py}) rather than uploaded from our local machine.")
        cj1 = os.path.join(CLOUD, "cluster_part1.json")
        if os.path.exists(cj1) and json.load(open(cj1)).get("clusterUuid") != c.get("clusterUuid"):
            fh.write(" The run used two clusters of this configuration. The stages up to the cleaning-ledger check ran "
                     "on the first (\\code{cloud\\_run/cluster\\_part1.json}). A status call on our side hung, the "
                     "cluster sat idle until its idle limit deleted it, and we resumed the remaining stages on a second "
                     "cluster, which read the first cluster's outputs from the same Cloud Storage prefix.")
        fh.write("\n\n"
                 f"Table~\\ref{{tab:cloudcmp}} compares the wall time of each stage; the cloud stages took about "
                 f"{V_['cloud.total_min']} minutes in total.")
        if "cloud.job_overhead" in V_:
            fh.write(f" Every Dataproc job pays a fixed start-up cost (the trivial environment-check job took "
                     f"{V_['cloud.job_overhead']}\\,s), and on the short stages that cost dominates. Our local machine also has "
                     f"{V_['env.ncpu']} cores against the cluster's {int(V_['cloud.exec_inst']) * int(V_['cloud.exec_cores'])} "
                     "executor cores, and reads local disk rather than Cloud Storage. Cleaning, the longest data stage, "
                     "is where the cluster catches up.")
        fh.write("\n\n")
        if "par.exact" in V_:
            fh.write(
                f"To check that the cluster computed the same results, \\code{{checks/cloud\\_parity.py}} compares its "
                f"outputs with the local ones ({V_['par.nchecked']} tables). Counts that do not depend on the order of "
                f"summation (rows, stations, every cleaning rule, coverage, slot and event counts, sample sizes) "
                + ("match exactly" if V_['par.exact'] == "0" else f"differ in {V_['par.exact']} places") + f"; across {V_['par.floatcols']} floating-point columns the largest relative difference is "
                f"${V_['par.floatmax']}$. "
                + (f"The {V_['par.thr_cols']} threshold counts on summed rain (hours with at least 20 or 40\\,mm, days with at "
                   f"least 50\\,mm) differ by at most {V_['par.thr_maxdiff']} per row. Most readings are multiples of "
                   "0.2\\,mm, so many totals equal a threshold exactly, and a floating-point sum lands just above or below "
                   "it depending on the order in which Spark adds partitions. \\code{checks/threshold\\_boundary.py} counts "
                   f"the totals within $10^{{-6}}$\\,mm of each threshold, and {V_['par.thr_ok']} differences are within "
                   "that count. "
                   if V_['par.thr_cols'] else "")
                + f"The {V_['par.approx_n']} approximate quantiles (Spark's \\code{{percentile\\_approx}}) differ by up to "
                f"{V_['par.approx_max']}. On 2023 the Task 2 fit on the cluster chose {V_['par.same_choice']} models as "
                f"the local fit; Model B's validation AUC-PR is {V_.get('par.B.aucpr.c', '--')} on Dataproc and "
                f"{V_.get('par.B.aucpr.l', '--')} locally, and the largest AUC-PR difference over all candidates is "
                f"{V_['par.t2maxdiff']}.\n\n")
        fh.write("\\begin{table}[H]\\centering\\small\n\\input{generated/tab_cloud_compare.tex}\n"
                 "\\caption{Wall time per stage in seconds, on the local machine and on Dataproc, and their ratio; a ratio "
                 "below one means Dataproc was faster. \\src{src/report\\_macros.py from "
                 "runs/*/env.json and " + ("cloud\\_run/job\\_times.json (Dataproc jobs API; driver start to exit, including the "
                 "small Cloud Storage copies)" if cloud_wall_from_api else "cloud\\_run/runs/*/env.json") + "}}\\label{tab:cloudcmp}\n\\end{table}\n")
    else:
        fh.write("The Dataproc run has not been made yet. The script \\code{cloud/dataproc\\_run.sh} runs the same "
                 "stages on a cluster (Lecture 6 image \\code{3.0-debian13}); when its outputs are fetched into "
                 "\\code{cloud\\_run/}, \\code{src/report\\_macros.py} fills this subsection from "
                 "\\code{cloud\\_run/runs/*/env.json} and \\code{cloud\\_run/cluster.json}.\n")
put("generated_at", _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"))

# ------------------------------------------------------------------ write
os.makedirs(G, exist_ok=True)
with open(os.path.join(G, "values.tex"), "w") as fh:
    fh.write("% generated by src/report_macros.py; do not edit\n")
    for k in sorted(V):
        fh.write(f"\\setval{{{k}}}{{{V[k]}}}\n")
print(f"REPORT values={len(V)} tables={len(glob.glob(os.path.join(G, 'tab_*.tex')))} cloud_env={'yes' if cloud_ok else 'not yet run'}")
