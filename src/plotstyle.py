"""Shared figure style for the report (palette from the dataviz reference instance)."""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "outputs", "task1")
COV = os.path.join(ROOT, "outputs", "coverage", "coverage_by_year.csv")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# palette (reference instance of the dataviz skill): categorical slots, ink, grid; blue sequential; blue-red diverging
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# diverging for rainfall anomalies: drier = red, wetter = blue, grey midpoint
DIV = LinearSegmentedColormap.from_list("div", ["#b8302f", "#ef9a8f", "#f0efec", "#86b6ef", "#1c5cab"])
SEASONS = ["NE monsoon (Dec-Mar)", "Inter-monsoon I (Apr-May)", "SW monsoon (Jun-Sep)", "Inter-monsoon II (Oct-Nov)"]
SEASON_SHORT = {"NE monsoon (Dec-Mar)": "NE monsoon\nDec-Mar", "Inter-monsoon I (Apr-May)": "Inter-monsoon I\nApr-May",
                "SW monsoon (Jun-Sep)": "SW monsoon\nJun-Sep", "Inter-monsoon II (Oct-Nov)": "Inter-monsoon II\nOct-Nov"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
LOW_COVERAGE_DAYS = 15   # month-cells with fewer days of data are marked

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 2.0, "figure.facecolor": "white"})


def rd(name):
    return pd.read_csv(os.path.join(T, name + ".csv"))


def rate(df):
    """mm per station-day from slot counts: mean 5-minute rain x 288."""
    return df.rain_sum / df.slots * 288


def save(fig, name):
    p = os.path.join(FIG, name + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("FIGURE", os.path.relpath(p, ROOT))
    return p


