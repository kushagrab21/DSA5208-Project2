"""Draw the Task 1 figures (PNG) from the Spark-aggregated tables in outputs/task1/, and write the Task 1
summary numbers to outputs/task1/summary.json (the report macros are generated from it).

No aggregation of raw data happens here: every input row was produced by src/task1.py.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from plotstyle import *  # noqa: F401,F403,E402
from plotstyle import (AXIS, C, COV, DAYS_IN_MONTH, DIV, FIG, GRID, INK, INK2, LOW_COVERAGE_DAYS, MONTHS,  # noqa: E402
                       MUTED, ROOT, SEASON_SHORT, SEASONS, SEQ, T, rate, rd, save)

S = {}  # summary numbers

# ---------------------------------------------------------------- coverage
cov = pd.read_csv(COV)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].bar(cov.year, cov.stations, color=C[0], width=0.6)
for x, y in zip(cov.year, cov.stations):
    ax[0].text(x, y + 1, str(y), ha="center", va="bottom", color=INK2, fontsize=8)
ax[0].set_title("Stations reporting")
ax[0].set_ylim(0, cov.stations.max() * 1.18)
ax[1].bar(cov.year, 100 * cov.slot_share, color=C[0], width=0.6)
for x, y in zip(cov.year, 100 * cov.slot_share):
    ax[1].text(x, y + 1, f"{y:.0f}", ha="center", va="bottom", color=INK2, fontsize=8)
ax[1].set_title("5-minute slots reported (%)")
ax[1].set_ylim(0, 110)
for a in ax:
    a.set_xticks(cov.year)
    a.grid(axis="x", visible=False)
save(fig, "fig01_coverage")

# ---------------------------------------------------------------- monthly cycle and year x month
m = rd("monthly_by_year")
m["rate"] = rate(m)
m["wet_freq"] = m.wet_slots / m.slots
m["intensity_mmh"] = m.rain_sum / m.wet_slots * 12
clim = m.groupby("month").apply(lambda g: g.rain_sum.sum() / g.slots.sum() * 288, include_groups=False)
years = sorted(m.year.unique())
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (y, g) in enumerate(m.groupby("year")):
    ax.plot(g.month, g.rate, color=AXIS, lw=0.9, zorder=1)
ax.plot(clim.index, clim.values, color=C[0], lw=2.4, marker="o", ms=4, zorder=3, label="2017-2024 (pooled)")
ax.plot([], [], color=AXIS, lw=0.9, label="single years")
ax.set_xticks(range(1, 13), MONTHS)
ax.set_ylabel("mm per station-day")
ax.set_title("Monthly mean rainfall per station-day")
ax.legend(loc="upper left")
save(fig, "fig02_monthly_cycle")
S["wettest_month"] = MONTHS[int(clim.idxmax()) - 1]
S["wettest_month_rate"] = float(clim.max())
S["driest_month"] = MONTHS[int(clim.idxmin()) - 1]
S["driest_month_rate"] = float(clim.min())
S["monthly_rate"] = {MONTHS[k - 1]: float(v) for k, v in clim.items()}

piv = m.pivot(index="year", columns="month", values="rate")
days = m.pivot(index="year", columns="month", values="days_with_data")
anom = piv / clim.values[None, :] * 100 - 100
fig, ax = plt.subplots(figsize=(7.2, 3.2))
lim = np.nanmax(np.abs(anom.values))
# months the MSS check finds distorted by gaps (same criterion as the report's gap-month table)
GAP_CELLS = set()
_mp = os.path.join(ROOT, "outputs", "mss", "monthly_rate_ours_vs_mss.csv")
if os.path.exists(_mp):
    _m = pd.read_csv(_mp)
    for _r in _m[((_m.ratio < 0.85) | (_m.ratio > 1.15)) & (_m.missed_share > 0.1)].itertuples():
        GAP_CELLS.add((list(anom.index).index(_r.year), _r.month - 1))
im = ax.imshow(anom.values, cmap=DIV, norm=TwoSlopeNorm(0, -100, max(100, lim)), aspect="auto")
for i in range(anom.shape[0]):
    for j in range(anom.shape[1]):
        v = anom.values[i, j]
        low = days.values[i, j] < LOW_COVERAGE_DAYS
        gap = (i, j) in GAP_CELLS
        ax.text(j, i, f"{v:+.0f}" + ("*" if low else "") + ("†" if gap else ""), ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 70 else INK)
ax.set_xticks(range(12), MONTHS)
ax.set_yticks(range(len(anom.index)), anom.index)
ax.grid(False)
ax.set_title("Monthly rainfall anomaly (% of the 2017-2024 month mean)")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("%", color=INK2)
cb.outline.set_visible(False)
ax.text(0, -0.16, f"* fewer than {LOW_COVERAGE_DAYS} days with data in that month;  † rate distorted by gaps "
        "(official MSS check)", transform=ax.transAxes,
        color=MUTED, fontsize=7)
save(fig, "fig03_year_month_anomaly")

# ---------------------------------------------------------------- annual, month-balanced; frequency x intensity
ann = []
for y, g in m.groupby("year"):
    g = g.set_index("month").reindex(range(1, 13))
    bal = float(np.nansum(g.rate.values * DAYS_IN_MONTH))           # mm per station per year, month-balanced
    naive = float(g.rain_sum.sum())                                  # raw network total (changing station set)
    ann.append({"year": y, "balanced_mm": bal, "naive_network_mm": naive,
                "wet_freq": g.wet_slots.sum() / g.slots.sum(),
                "intensity_mmh": g.rain_sum.sum() / g.wet_slots.sum() * 12})
ann = pd.DataFrame(ann)
# fixed panel of stations present (>= 50% of slots) in all eight years
mp = rd("monthly_by_year_panel")
mp["rate"] = rate(mp)
pan = []
for y, g in mp.groupby("year"):
    g = g.set_index("month").reindex(range(1, 13))
    pan.append({"year": y, "panel_balanced_mm": float(np.nansum(g.rate.values * DAYS_IN_MONTH)),
                "panel_wet_freq": g.wet_slots.sum() / g.slots.sum(),
                "panel_intensity_mmh": g.rain_sum.sum() / g.wet_slots.sum() * 12,
                "panel_stations": int(g.stations.max())})
ann = ann.merge(pd.DataFrame(pan), on="year")
ann.to_csv(os.path.join(T, "annual_balanced.csv"), index=False)
mean_bal = ann.balanced_mm.mean()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
cols = [C[0] if v >= mean_bal else C[1] for v in ann.balanced_mm]
ax[0].bar(ann.year, ann.balanced_mm, color=C[0], width=0.6)
ax[0].axhline(mean_bal, color=INK2, lw=1, ls="--")
lowcov = set(days.index[(days < LOW_COVERAGE_DAYS).any(axis=1)])
for x, y in zip(ann.year, ann.balanced_mm):
    ax[0].text(x, y - 200, f"{y:.0f}" + ("*" if x in lowcov else ""), ha="center", va="top", fontsize=6.5,
               color="white", rotation=90)
if lowcov:
    ax[0].text(0, -0.2, "* a month with fewer than %d days of data" % LOW_COVERAGE_DAYS, transform=ax[0].transAxes,
               fontsize=6.5, color=MUTED)
ax[0].set_title(f"Annual rainfall per station (mm)\ndashed line: 2017-2024 mean, {mean_bal:.0f} mm")
ax[0].plot(ann.year, ann.panel_balanced_mm, ls="none", marker="D", ms=5, color=C[1], mec="white", mew=0.8,
           label=f"fixed panel ({ann.panel_stations.min()} stations)")
ax[0].legend(loc="upper left", fontsize=6.5)
ax[0].set_ylim(0, max(ann.balanced_mm.max(), ann.panel_balanced_mm.max()) * 1.18)
ax[0].set_xticks(ann.year, [str(y)[2:] for y in ann.year])
ax[0].grid(axis="x", visible=False)
fi = ann.wet_freq / ann.wet_freq.mean() * 100
ii = ann.intensity_mmh / ann.intensity_mmh.mean() * 100
ax[1].axhline(100, color=AXIS, lw=1)
ax[1].plot(ann.year, fi, color=C[0], marker="o", ms=4, label="wet-slot frequency")
ax[1].plot(ann.year, ii, color=C[1], marker="s", ms=4, label="intensity when wet")
ax[1].plot(ann.year, ann.panel_wet_freq / ann.wet_freq.mean() * 100, color=C[0], ls=":", lw=1.6,
           label="frequency, fixed panel")
ax[1].plot(ann.year, ann.panel_intensity_mmh / ann.intensity_mmh.mean() * 100, color=C[1], ls=":", lw=1.6,
           label="intensity, fixed panel")
ax[1].set_title("Frequency and intensity\n(index, 2017-2024 mean = 100)")
ax[1].set_xticks(ann.year, [str(y)[2:] for y in ann.year])
ax[1].legend(loc="lower left", fontsize=7)
save(fig, "fig04_annual_balanced")
S["annual_balanced_mm"] = {int(r.year): float(r.balanced_mm) for r in ann.itertuples()}
S["annual_wet_freq"] = {int(r.year): float(r.wet_freq) for r in ann.itertuples()}
S["annual_intensity_mmh"] = {int(r.year): float(r.intensity_mmh) for r in ann.itertuples()}
S["annual_naive_network_mm"] = {int(r.year): float(r.naive_network_mm) for r in ann.itertuples()}
S["panel_balanced_mm"] = {int(r.year): float(r.panel_balanced_mm) for r in ann.itertuples()}
S["panel_wet_freq"] = {int(r.year): float(r.panel_wet_freq) for r in ann.itertuples()}
S["panel_stations"] = int(ann.panel_stations.min())
S["corr_annual_freq"] = float(np.corrcoef(ann.balanced_mm, ann.wet_freq)[0, 1])
S["corr_annual_intensity"] = float(np.corrcoef(ann.balanced_mm, ann.intensity_mmh)[0, 1])

# ---------------------------------------------------------------- diurnal cycle by season
d = rd("diurnal_by_season")
d["rate_mmh"] = d.rain_sum / d.slots * 12
d["wet_pct"] = d.wet_slots / d.slots * 100
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
for i, s in enumerate(SEASONS):
    g = d[d.season == s].sort_values("hour_start")
    ax[0].plot(g.hour_start + 0.5, g.rate_mmh, color=C[i], label=SEASON_SHORT[s].replace("\n", " "))
    ax[1].plot(g.hour_start + 0.5, g.wet_pct, color=C[i])
    pk = g.loc[g.rate_mmh.idxmax()]
    S.setdefault("diurnal_peak_hour", {})[s] = int(pk.hour_start)
    S.setdefault("diurnal_peak_rate_mmh", {})[s] = float(pk.rate_mmh)
    S.setdefault("diurnal_min_rate_mmh", {})[s] = float(g.rate_mmh.min())
ax[0].set_title("Mean rain rate (mm/h)")
ax[1].set_title("Share of 5-min slots with rain (%)")
for a in ax:
    a.set_xticks(range(0, 25, 3))
    a.set_xlim(0, 24)
    a.set_xlabel("hour of day (SGT)")
ax[0].legend(fontsize=7, loc="upper left")
save(fig, "fig05_diurnal_by_season")
tot = d.groupby("hour_start")[["rain_sum", "slots"]].sum()
S["share_rain_12_20"] = float(tot.loc[12:19].rain_sum.sum() / tot.rain_sum.sum())

# ---------------------------------------------------------------- frequency vs intensity by month
mm_ = m.groupby("month")[["rain_sum", "slots", "wet_slots", "intense_rain_sum"]].sum()
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax[0].bar(mm_.index, mm_.wet_slots / mm_.slots * 100, color=C[0], width=0.65)
ax[0].set_title("Wet slots (%)")
ax[1].bar(mm_.index, mm_.rain_sum / mm_.wet_slots * 12, color=C[0], width=0.65)
ax[1].set_title("Intensity when wet (mm/h)")
ax[2].bar(mm_.index, mm_.intense_rain_sum / mm_.rain_sum * 100, color=C[0], width=0.65)
ax[2].set_title("Rain from >= 30 mm/h (%)")
for a in ax:
    a.set_xticks(range(1, 13), [x[0] for x in MONTHS])
    a.grid(axis="x", visible=False)
save(fig, "fig06_frequency_intensity_month")
S["month_wet_pct"] = {MONTHS[k - 1]: float(v) for k, v in (mm_.wet_slots / mm_.slots * 100).items()}
S["month_intensity_mmh"] = {MONTHS[k - 1]: float(v) for k, v in (mm_.rain_sum / mm_.wet_slots * 12).items()}
S["month_intense_share"] = {MONTHS[k - 1]: float(v) for k, v in (mm_.intense_rain_sum / mm_.rain_sum).items()}
S["intense_share_all"] = float(mm_.intense_rain_sum.sum() / mm_.rain_sum.sum())
S["wet_pct_all"] = float(mm_.wet_slots.sum() / mm_.slots.sum() * 100)

# ---------------------------------------------------------------- extremes across years
he = rd("hourly_extremes_by_year")
de = rd("daily_extremes_by_year")
am = rd("station_annual_max_hour")
am = am[am.complete_hours >= 0.5 * 8760]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
ax[0].bar(he.year, he.hours_ge_20 / he.complete_station_hours * 1e4, color=C[0], width=0.6)
ax[0].set_title("Hours >= 20 mm\nper 10,000 station-hours")
ax[1].bar(de.year, de.heavy_days / de.full_station_days * 1000, color=C[0], width=0.6)
ax[1].set_title("Days >= 50 mm\nper 1,000 station-days")
bx = [am[am.year == y].annual_max_hour_mm.values for y in he.year]
bp = ax[2].boxplot(bx, positions=he.year, widths=0.55, patch_artist=True, showfliers=True,
                   medianprops={"color": INK, "lw": 1.5}, boxprops={"facecolor": "#cde2fb", "edgecolor": C[0]},
                   whiskerprops={"color": C[0]}, capprops={"color": C[0]},
                   flierprops={"marker": "o", "ms": 3, "mfc": C[0], "mec": "white", "mew": 0.5})
ax[2].set_title("Annual max hourly rain\nper station (mm)")
for a in ax:
    a.set_xticks(he.year, [str(y)[2:] for y in he.year])
    a.grid(axis="x", visible=False)
save(fig, "fig07_extremes_by_year")
S["heavy_hours_per_10k"] = {int(r.year): float(r.hours_ge_20 / r.complete_station_hours * 1e4) for r in he.itertuples()}
S["heavy_days_per_1k"] = {int(r.year): float(r.heavy_days / r.full_station_days * 1e3) for r in de.itertuples()}
S["max_hour_mm"] = {int(r.year): float(r.max_hour_mm) for r in he.itertuples()}
S["median_annual_max_hour"] = {int(y): float(np.median(v)) for y, v in zip(he.year, bx)}
S["wet_day_share"] = {int(r.year): float(r.wet_days / r.full_station_days) for r in de.itertuples()}
S["heavy_hours_total"] = int(he.hours_ge_20.sum())
S["complete_station_hours_total"] = int(he.complete_station_hours.sum())
hm = rd("hourly_extremes_by_month")
S["heavy_hours_per_10k_month"] = {MONTHS[int(r.month) - 1]: float(r.hours_ge_20 / r.complete_station_hours * 1e4)
                                  for r in hm.itertuples()}

# ---------------------------------------------------------------- seasonal anomalies by year
sy = rd("season_by_year")
sy["rate"] = rate(sy)
sclim = sy.groupby("season").apply(lambda g: g.rain_sum.sum() / g.slots.sum() * 288, include_groups=False)
sy["anom"] = sy.apply(lambda r: r.rate / sclim[r.season] * 100 - 100, axis=1)
sp_ = sy.pivot(index="year", columns="season", values="anom")[SEASONS]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
im = ax.imshow(sp_.values.T, cmap=DIV, norm=TwoSlopeNorm(0, -80, 80), aspect="auto")
for i in range(sp_.shape[0]):
    for j in range(sp_.shape[1]):
        v = sp_.values[i, j]
        ax.text(i, j, f"{v:+.0f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 55 else INK)
ax.set_yticks(range(4), [SEASON_SHORT[s] for s in SEASONS], fontsize=8)
ax.set_xticks(range(len(sp_.index)), sp_.index)
ax.grid(False)
ax.set_title("Seasonal rainfall anomaly (% of the 2017-2024 season mean)")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.outline.set_visible(False)
save(fig, "fig08_season_anomaly")
S["season_rate"] = {s: float(sclim[s]) for s in SEASONS}
S["season_anom"] = {s: {int(y): float(sp_.loc[y, s]) for y in sp_.index} for s in SEASONS}

# ---------------------------------------------------------------- spatial pattern
ss = rd("station_season")
ss = ss[ss.slots >= 0.25 * ss.groupby("season").slots.transform("max")]
ss["rate"] = rate(ss)
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=True)
ax = ax.ravel()
vmin, vmax = ss.rate.quantile(0.02), ss.rate.quantile(0.98)
for i, s in enumerate(SEASONS):
    g = ss[ss.season == s]
    sc = ax[i].scatter(g.lon, g.lat, c=g.rate, cmap=SEQ, vmin=vmin, vmax=vmax, s=34, edgecolors="white", linewidths=0.6)
    ax[i].set_title(SEASON_SHORT[s].replace("\n", " "), fontsize=9)
    ax[i].set_ylim(1.20, 1.50)
    ax[i].set_aspect("equal")
    ax[i].tick_params(labelsize=7)
    r = np.corrcoef(g.lon, g.rate)[0, 1]
    S.setdefault("corr_lon_rate", {})[s] = float(r)
    S.setdefault("n_stations_map", {})[s] = int(len(g))
    ax[i].text(0.98, 0.04, f"r(rain, longitude) = {r:+.2f}", transform=ax[i].transAxes, fontsize=7, color=INK2,
               ha="right")
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.03)
cb.set_label("mm per station-day", color=INK2, fontsize=8)
for a in ax[2:]:
    a.set_xlabel("longitude (E)", fontsize=8)
for a in ax[::2]:
    a.set_ylabel("latitude (N)", fontsize=8)
cb.outline.set_visible(False)
fig.suptitle("Station mean rainfall by season", x=0.07, ha="left", fontsize=10, fontweight="bold", y=0.98)
save(fig, "fig09_station_map_season")

# all-season station means: east-west gradient
sy2 = rd("station_year")
st_all = sy2.groupby(["station_id", "lat", "lon"])[["rain_sum", "slots"]].sum().reset_index()
st_all = st_all[st_all.slots >= 105120]      # at least one year of 5-minute slots
st_all["rate"] = rate(st_all)
S["n_stations_ge1yr"] = int(len(st_all))
S["corr_lon_rate_all"] = float(np.corrcoef(st_all.lon, st_all.rate)[0, 1])
S["station_rate_min"] = float(st_all.rate.min())
S["station_rate_max"] = float(st_all.rate.max())
S["station_rate_max_id"] = str(st_all.loc[st_all.rate.idxmax(), "station_id"])
S["station_rate_min_id"] = str(st_all.loc[st_all.rate.idxmin(), "station_id"])

# ---------------------------------------------------------------- day-of-year curve
dn = rd("daily_network")
dn["rate"] = rate(dn)
dn = dn[dn.doy <= 365]
fig, ax = plt.subplots(figsize=(7.2, 2.8))
pooled = dn.groupby("doy")[["rain_sum", "slots"]].sum()
ext = pd.concat([pooled.iloc[-15:], pooled, pooled.iloc[:15]]).rolling(31, center=True).sum().iloc[15:-15]
sm = ext.rain_sum / ext.slots * 288
for y, g in dn.groupby("year"):
    g = g.set_index("doy")[["rain_sum", "slots"]].reindex(range(1, 366))
    r_ = g.rolling(31, center=True, min_periods=20).sum()
    ok = g.slots.notna().rolling(31, center=True).sum() >= 20
    ax.plot(r_.index, (r_.rain_sum / r_.slots * 288).where(ok), color=AXIS, lw=0.8)
ax.plot(sm.index, sm.values, color=C[0], lw=2.4, label="2017-2024 pooled, 31-day mean")
ax.plot([], [], color=AXIS, lw=0.8, label="single years, 31-day mean")
starts = np.cumsum([0] + list(DAYS_IN_MONTH[:-1].round()))
ax.set_xticks(starts + 1, MONTHS)
ax.set_xlim(1, 365)
ax.set_ylabel("mm per station-day")
ax.set_title("Rainfall through the year (network mean, 31-day running mean)")
ax.legend(loc="upper center", fontsize=7, ncol=2)
save(fig, "fig10_day_of_year")

# ---------------------------------------------------------------- data resolution (observation)
res = rd("resolution_by_year")
S["share_mult_0p2"] = {int(r.year): float(r.mult_0p2 / r.wet_slots) for r in res.itertuples()}
S["max_5min"] = {int(r.year): float(r.max_5min) for r in res.itertuples()}
ws = rd("wet_spells_by_year")
S["wet_spell_mean_len"] = {int(r.year): float(r.mean_len) for r in ws.itertuples()}
S["wet_spell_max_len"] = {int(r.year): int(r.max_len) for r in ws.itertuples()}

json.dump(S, open(os.path.join(T, "summary.json"), "w"), indent=1)
print("TASK1 summary written:", os.path.relpath(os.path.join(T, "summary.json"), ROOT))
