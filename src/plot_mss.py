"""Figure 16: agreement with the official MSS daily records and the cost of gaps (from outputs/mss/, made by
src/mss_validate.py)."""
import os

import numpy as np
import pandas as pd

import plotstyle as P

plt = P.plt
T = os.path.join(P.ROOT, "outputs", "mss")


def main():
    d = pd.read_csv(os.path.join(T, "complete_day_pairs.csv"))
    ym = pd.read_csv(os.path.join(T, "gap_cost_year_month.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.35]})
    lim = max(d.mss_mm.max(), d.our_mm.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color=P.AXIS, lw=1, zorder=1)
    ax[0].scatter(d.mss_mm, d.our_mm, s=5, color=P.C[0], alpha=0.35, linewidths=0, zorder=2)
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("official MSS daily total (mm)")
    ax[0].set_ylabel("our daily total (mm)")
    ax[0].set_title(f"(a) Complete days, n = {len(d):,}")
    piv = ym.pivot(index="year", columns="month", values="missed_share") * 100
    im = ax[1].imshow(piv.values, cmap=P.LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#ec835a", "#b8302f"]),
                      vmin=0, vmax=max(40, np.nanmax(piv.values)), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v) and v >= 5:
                ax[1].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                           color="white" if v > 45 else P.INK)
    ax[1].set_xticks(range(12), [m[0] for m in P.MONTHS])
    ax[1].set_yticks(range(len(piv.index)), piv.index)
    ax[1].grid(False)
    ax[1].set_title("(b) Official rain on days we lack (%)")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    P.save(fig, "fig16_mss_validation")


if __name__ == "__main__":
    main()
