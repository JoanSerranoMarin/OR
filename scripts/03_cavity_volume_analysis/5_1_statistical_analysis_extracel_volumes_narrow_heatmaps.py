#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2: extracellular cavity volumes and activation-dependent changes
across GPCR classes, combining three ROIs into a single 3×3 panel figure.

This version uses a compressed right column for the heatmaps and a single
shared colorbar, giving more space to the bar/scatter panels.

Expects three Excel files named 'gpcr_log2_active_inactive.xlsx' in:

- volumen_5_portal_EC
- volumen_2_ec_vestibule
- volumen_3_cleaft_volume

Outputs:
- A 3×3 figure (A–I) with:
  * Left column: absolute volumes (Inactive vs Active) + paired t-test lines.
  * Middle column: log2(Active/Inactive) per class.
  * Right column: between-class contrasts in log2(Active/Inactive).
- An Excel file with:
  * state_active_vs_inactive_BH: paired tests Active vs Inactive per class & ROI.
  * log2FC_between_classes_posthoc: post-hoc class comparisons per ROI.
  * summary_by_class: mean log2(Active/Inactive) per class & ROI.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import stats

# ---------------------------------------------------------------------
# Canonical column set and class order
# ---------------------------------------------------------------------
REQ_COLS = {"GPCR_name", "Inactive", "Active",
            "log2(active/inactive)", "Class"}

CLASS_ORDER_CANON = [
    "Class A",
    "Class B1",
    "Class B2",
    "Class C",
    "Class F",
    "Class O1",
    "Class O2",
    "Class T2",
]

# ---------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------
def class_display_name(name: str) -> str:
    """Display 'Class A', 'Class B', ... (strip parentheses content)."""
    s = str(name).strip()
    s = re.sub(r"\s*\([^)]*\)\s*", "", s).strip()
    return s

def class_display_list(names):
    return [class_display_name(n) for n in names]

# ---------------------------------------------------------------------
# Flexible column mapping
# ---------------------------------------------------------------------
SYNONYMS = {
    "GPCR_name": [
        "gpcr_name", "receptor", "_receptor", "receptor_name", "gpcr", "name"
    ],
    "Inactive": [
        "inactive", "inactive_volume", "free_volume_inactive",
        "inactive_free_volume", "inactive_vol", "inactive (å³)"
    ],
    "Active": [
        "active", "active_volume", "free_volume_active",
        "active_free_volume", "active_vol", "active (å³)"
    ],
    "Class": [
        "class", "gpcr_class", "clase", "class_name", "classe"
    ],
    "log2(active/inactive)": [
        "log2(active/inactive)", "log2_ratio", "log2(active/inact)",
        "log2_active_inactive", "log2activeinactive", "log2(a/i)", "log2 ai"
    ],
}

def _normalize_colname(c):
    """Normalize column name for comparison."""
    return re.sub(r"\s+", " ", str(c).strip()).lower()

def detect_column_mapping(df_columns):
    """Return dict {canonical_name: column_in_df}."""
    cols_norm = {_normalize_colname(c): c for c in df_columns}
    mapping = {}
    for canonical, aliases in SYNONYMS.items():
        for alias in aliases:
            if alias in cols_norm:
                mapping[canonical] = cols_norm[alias]
                break
    return mapping

# ---------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------
def load_table(xlsx_path: Path, sheet=None) -> pd.DataFrame:
    """
    Read Excel and return DataFrame with canonical columns:
    ['GPCR_name','Inactive','Active','log2(active/inactive)','Class'].
    """
    xf = pd.ExcelFile(xlsx_path)
    if sheet is not None:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    else:
        preferred = "Sheet2"
        sheet_to_read = preferred if preferred in xf.sheet_names else xf.sheet_names[0]
        df = pd.read_excel(xlsx_path, sheet_name=sheet_to_read)

    mapping = detect_column_mapping(df.columns)

    if not ({"Inactive", "Active"}.issubset(mapping.keys()) or
            "log2(active/inactive)" in mapping):
        raise ValueError(
            "Could not find 'Active'/'Inactive' nor precomputed log2 ratio.\n"
            f"Columns: {list(df.columns)}"
        )
    if "Class" not in mapping:
        raise ValueError("Could not find 'Class' column.")
    if "GPCR_name" not in mapping:
        df["_temp_gpcr"] = [f"receptor_{i+1}" for i in range(len(df))]
        mapping["GPCR_name"] = "_temp_gpcr"

    rename_map = {mapping[k]: k for k in mapping}
    d = df.rename(columns=rename_map).copy()

    if "Inactive" in d:
        d["Inactive"] = pd.to_numeric(d["Inactive"], errors="coerce")
    if "Active" in d:
        d["Active"] = pd.to_numeric(d["Active"], errors="coerce")

    if "log2(active/inactive)" not in d and {"Active", "Inactive"}.issubset(d.columns):
        d.loc[~np.isfinite(d["Inactive"]) | (d["Inactive"] <= 0), "Inactive"] = np.nan
        d.loc[~np.isfinite(d["Active"])   | (d["Active"]   <= 0), "Active"]   = np.nan
        d["log2(active/inactive)"] = np.log2(d["Active"] / d["Inactive"])

    missing = REQ_COLS - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns after mapping: {missing}")

    used = {k: mapping.get(k, k)
            for k in ["GPCR_name","Inactive","Active",
                      "Class","log2(active/inactive)"]}
    print(f"[INFO] Column mapping -> canonical: {used}")

    return d[["GPCR_name","Inactive","Active",
              "log2(active/inactive)","Class"]].copy()

def remove_classless_and_outliers(df: pd.DataFrame, z=2.0) -> pd.DataFrame:
    """
    Remove 'CLASSLESS' entries and 2σ outliers in log2(active/inactive) per class.
    """
    d = df.copy()
    d = d[d["Class"].astype(str).str.strip().str.upper() != "CLASSLESS"]
    d = d[np.isfinite(d["log2(active/inactive)"])]

    def filt(g):
        x = g["log2(active/inactive)"].values
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1) if len(x) > 1 else 0.0
        if not np.isfinite(sd) or sd <= 0:
            return g
        lo, hi = mu - z * sd, mu + z * sd
        return g[(g["log2(active/inactive)"] >= lo) &
                 (g["log2(active/inactive)"] <= hi)]

    return d.groupby("Class", group_keys=False, sort=False).apply(filt).reset_index(drop=True)

# ---------------------------------------------------------------------
# ANOVA + post-hoc
# ---------------------------------------------------------------------
def one_way_anova(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class"):
    groups = [g[ycol].dropna().values for _, g in df.groupby(gcol, sort=False)]
    labels = [str(n) for n, _ in df.groupby(gcol, sort=False)]
    valid = [(lab, arr) for lab, arr in zip(labels, groups) if len(arr) >= 2]
    if len(valid) < 2:
        raise ValueError("Need ≥2 classes with at least 2 observations after filtering.")
    labels, groups = zip(*valid)
    F, p = stats.f_oneway(*groups)
    SSE = sum(((arr - np.mean(arr))**2).sum() for arr in groups)
    df_within = sum(len(arr) for arr in groups) - len(groups)
    MSE = SSE / df_within if df_within > 0 else np.nan
    return F, p, list(labels), list(groups), df_within, MSE

def tukey_or_holm(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class",
                  df_within=None, MSE=None, alpha=0.05):
    """
    Try TukeyHSD via statsmodels; otherwise use Holm-corrected
    pairwise t-tests using ANOVA MSE.
    """
    means = df.groupby(gcol, sort=False)[ycol].mean().to_dict()
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        res = pairwise_tukeyhsd(endog=df[ycol].values,
                                groups=df[gcol].values, alpha=alpha)
        out = pd.DataFrame(res._results_table.data[1:],
                           columns=res._results_table.data[0])
        out = out.rename(columns={"p-adj":"p_adj",
                                  "reject":"reject",
                                  "meandiff":"meandiff"})
        out["p_adj"] = out["p_adj"].astype(float)
        out["reject"] = out["reject"].astype(bool)
        method = "TukeyHSD"
    except Exception:
        from itertools import combinations
        from scipy.stats import t as t_dist
        if df_within is None or MSE is None or not np.isfinite(MSE):
            raise RuntimeError("No Tukey available and missing MSE/df for Holm.")
        rows = []
        groups_dict = {k: v.dropna().values
                       for k, v in df.groupby(gcol, sort=False)[ycol]}
        names = list(groups_dict.keys())
        for g1, g2 in combinations(names, 2):
            x1, x2 = groups_dict[g1], groups_dict[g2]
            if len(x1) < 2 or len(x2) < 2:
                continue
            n1, n2 = len(x1), len(x2)
            m1, m2 = x1.mean(), x2.mean()
            se = np.sqrt(MSE * (1/n1 + 1/n2)) if MSE > 0 else np.nan
            tval = (m1 - m2) / se if (se and se > 0) else np.nan
            p_raw = 2 * (1 - t_dist.cdf(abs(tval), df=df_within)) if np.isfinite(tval) else np.nan
            rows.append({"group1":g1, "group2":g2,
                         "meandiff":m1-m2, "p_raw":p_raw})
        out = pd.DataFrame(rows).sort_values("p_raw")
        m = len(out)
        out["p_adj"] = [min(1.0, (m - i + 1) * p)
                        for i, p in enumerate(out["p_raw"], start=1)]
        out["reject"] = out["p_adj"] < alpha
        method = "Holm"

    out["mean1"] = out["group1"].map(means)
    out["mean2"] = out["group2"].map(means)
    out["higher"] = np.where(out["mean1"] > out["mean2"],
                             out["group1"], out["group2"])
    out = out[["group1","group2","mean1","mean2",
               "meandiff","p_adj","reject","higher"]]
    return out, method

def summarize_per_class(df: pd.DataFrame,
                        ycol="log2(active/inactive)", gcol="Class"):
    """Per-class mean and n, respecting categorical order."""
    s = df.groupby(gcol, sort=False).agg(
        n=(ycol,"size"), mean=(ycol,"mean")
    ).reset_index()
    if pd.api.types.is_categorical_dtype(df[gcol]):
        cats = list(df[gcol].cat.categories)
    else:
        cats = CLASS_ORDER_CANON
    s = s.set_index(gcol).reindex(
        [c for c in cats if c in set(s[gcol])]
    ).reset_index()
    return s

def p_to_stars(p):
    if pd.isna(p): return "ns"
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else "ns"))

# ---------------------------------------------------------------------
# Benjamini–Hochberg and sig lines (for active vs inactive)
# ---------------------------------------------------------------------
def benjamini_hochberg(pvals):
    """BH FDR correction, returning adjusted p-values in original order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    ranks = np.arange(1, n + 1, dtype=float)

    bh = ranked_p * n / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)

    p_adj = np.empty_like(bh)
    p_adj[order] = bh
    return p_adj

def add_sig_line(ax, x1, x2, y, dy, p_adj, fontsize=10):
    """
    Draw a horizontal significance line only if p_adj < 0.05.
    """
    if np.isnan(p_adj) or p_adj >= 0.05:
        return

    if p_adj < 0.001:
        text = "***"
    elif p_adj < 0.01:
        text = "**"
    else:
        text = "*"

    ax.plot([x1, x2], [y, y], lw=1, c="black")
    ax.text((x1 + x2) / 2.0, y + dy, text,
            ha="center", va="bottom", fontsize=fontsize)

# ---------------------------------------------------------------------
# Paired tests per class (for Excel + plotting)
# ---------------------------------------------------------------------
def paired_tests_per_class(df: pd.DataFrame, classes_order, roi_label: str):
    """
    Paired t-tests (Active vs Inactive) within each class for a given ROI.

    Returns a DataFrame with:
    ROI, Class, n, mean_active, mean_inactive, t_stat, p_raw, p_adj_BH
    """
    rows = []
    for cls in classes_order:
        sub = df[df["Class"] == cls][["Active", "Inactive"]].dropna()
        if len(sub) <= 1:
            continue
        t_stat, p_raw = stats.ttest_rel(sub["Active"], sub["Inactive"])
        rows.append(
            {
                "ROI": roi_label,
                "Class": cls,
                "n": int(len(sub)),
                "mean_active": float(sub["Active"].mean()),
                "mean_inactive": float(sub["Inactive"].mean()),
                "t_stat": float(t_stat),
                "p_raw": float(p_raw),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["ROI","Class","n","mean_active","mean_inactive",
                     "t_stat","p_raw","p_adj_BH"]
        )

    df_tests = pd.DataFrame(rows)
    df_tests["p_adj_BH"] = benjamini_hochberg(df_tests["p_raw"].values)
    return df_tests

# ---------------------------------------------------------------------
# Compact, publication-oriented style
# ---------------------------------------------------------------------
# The default layout is intended for a two-column manuscript figure, where
# the final printed width is usually ~170–185 mm.  To keep the figure readable,
# class names are shortened on the x/y axes and the full class names are kept
# in the exported Excel tables.

SHORT_CLASS_LABELS = {
    "Class A": "A",
    "Class B1": "B1",
    "Class B2": "B2",
    "Class C": "C",
    "Class F": "F",
    "Class O1": "O1",
    "Class O2": "O2",
    "Class T2": "T2",
}

# Bars (absolute volumes) – light; points – darker
COL_INACT_BAR   = "#dcdcdc"
COL_ACT_BAR     = "#f6d3a0"
COL_INACT_POINT = "#555555"
COL_ACT_POINT   = "#a45a15"

# For log2 panels, use the active-state brown/orange palette
COL_LOG2_BAR    = COL_ACT_BAR
COL_LOG2_POINT  = COL_ACT_POINT

EDGE      = "#222222"
ALPHA_BAR = 0.85
ALPHA_DOT = 0.45
POINT_SIZE = 4.2

# Font sizes tuned for a readable full-width two-column figure.
# This version intentionally prioritizes legibility over maximum information
# density: the figure is meant to complement the text, not to carry every
# quantitative detail on its own.
FONT_SIZE = 8.0
LABEL_SIZE = 8.8
TICK_SIZE = 7.4
PANEL_SIZE = 11.5
STAR_SIZE = 7.8
FC_SIZE = 6.8
HEATMAP_TICK_SIZE = 6.4

PASTEL_MIX = 0.55


def compact_class_label(name: str) -> str:
    """Return a compact class label for plotting, e.g. 'Class O2' -> 'O2'."""
    clean = class_display_name(name)
    return SHORT_CLASS_LABELS.get(clean, clean.replace("Class ", ""))


def compact_class_labels(names):
    return [compact_class_label(n) for n in names]


def apply_compact_style():
    """Apply matplotlib defaults optimized for compact multi-panel figures."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    })


def make_pastel_cmap(hex_color=COL_LOG2_BAR, mix=PASTEL_MIX):
    base = np.array(to_rgb(hex_color))
    pastel = 1 - (1 - base) * mix
    return LinearSegmentedColormap.from_list(
        "pastel_orange", [(1, 1, 1), tuple(pastel)], N=256
    )


COLMAP_MATRIX = make_pastel_cmap()


def clean_axis(ax):
    """Minimal axis styling for bar/scatter panels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=1.5)


def add_panel_label(ax, label: str):
    """Place a compact panel label inside the top-left corner of an axis."""
    # Keep panel labels outside the plotting area so they remain readable
    # and do not compete with the data.
    ax.text(
        -0.13, 1.08, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=PANEL_SIZE,
        fontweight="bold",
        clip_on=False,
    )

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_active_inactive_absolute_ax(ax, df: pd.DataFrame, classes_order,
                                     state_tests_df: pd.DataFrame):
    """
    Left column (A, D, G): absolute volumes, inactive vs active.

    The active-vs-inactive significance line is drawn close to the local
    maximum of each class, rather than at the very top of the panel. This
    reduces empty space and improves readability in compact figures.
    """
    classes = [c for c in classes_order if (df["Class"] == c).any()]
    labels_display = compact_class_labels(classes)
    x = np.arange(len(classes))
    width = 0.34

    mean_inact = (df.groupby("Class", sort=False)["Inactive"]
                    .mean()
                    .reindex(classes))
    mean_act   = (df.groupby("Class", sort=False)["Active"]
                    .mean()
                    .reindex(classes))

    ax.bar(x - width/2, mean_inact.values, width=width,
           color=COL_INACT_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=0.35, label="inactive")
    ax.bar(x + width/2, mean_act.values, width=width,
           color=COL_ACT_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=0.35, label="active")

    rng = np.random.default_rng(123)
    jitter = 0.045
    for i, cls in enumerate(classes):
        vals_in = df.loc[df["Class"] == cls, "Inactive"].dropna().values
        vals_ac = df.loc[df["Class"] == cls, "Active"].dropna().values
        xs_in = (x[i] - width/2) + rng.normal(0, jitter, size=len(vals_in))
        xs_ac = (x[i] + width/2) + rng.normal(0, jitter, size=len(vals_ac))
        ax.scatter(xs_in, vals_in, s=POINT_SIZE, alpha=ALPHA_DOT,
                   color=COL_INACT_POINT, edgecolors="none", rasterized=True)
        ax.scatter(xs_ac, vals_ac, s=POINT_SIZE, alpha=ALPHA_DOT,
                   color=COL_ACT_POINT, edgecolors="none", rasterized=True)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=0, ha="center")
    ax.set_ylabel("Volume (Å³)")
    clean_axis(ax)

    if state_tests_df is None or state_tests_df.empty:
        return

    p_map = {row["Class"]: row["p_adj_BH"]
             for _, row in state_tests_df.iterrows()}

    # Local significance placement. First determine the natural data range.
    all_vals = pd.concat([df["Inactive"], df["Active"]], ignore_index=True).dropna().values
    if len(all_vals) == 0:
        return
    data_min = min(0.0, float(np.nanmin(all_vals)))
    data_max = float(np.nanmax(all_vals))
    data_range = max(1e-9, data_max - data_min)
    line_offset = 0.045 * data_range
    text_offset = 0.012 * data_range
    max_annot_y = data_max

    for i, cls in enumerate(classes):
        p_adj = p_map.get(cls, np.nan)
        if np.isnan(p_adj) or p_adj >= 0.05:
            continue

        vals_cls = df.loc[df["Class"] == cls, ["Inactive", "Active"]].values.ravel()
        vals_cls = vals_cls[np.isfinite(vals_cls)]
        if len(vals_cls) == 0:
            continue
        local_max = max(float(np.nanmax(vals_cls)), float(mean_inact.loc[cls]), float(mean_act.loc[cls]))
        y_line = local_max + line_offset
        max_annot_y = max(max_annot_y, y_line + 2.5 * text_offset)

        x_center = x[i]
        x1 = x_center - width/2
        x2 = x_center + width/2
        add_sig_line(ax, x1, x2, y_line, text_offset, p_adj, fontsize=STAR_SIZE)

    ax.set_ylim(bottom=max(0, data_min - 0.03 * data_range),
                top=max_annot_y + 0.04 * data_range)


def plot_means_transparent_ax(ax, summary: pd.DataFrame, df: pd.DataFrame):
    """
    Middle column (B, E, H): mean log2(Active/Inactive) per class + points.
    Fold-change labels are deliberately small to avoid clutter.
    """
    classes = summary["Class"].tolist()
    labels_display = compact_class_labels(classes)
    x = np.arange(len(classes))
    means = summary["mean"].values

    ax.bar(x, means,
           color=COL_LOG2_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=0.35)

    rng = np.random.default_rng(42)
    jitter = 0.055
    for i, cls in enumerate(classes):
        vals = df.loc[df["Class"] == cls, "log2(active/inactive)"].values
        xs = x[i] + rng.normal(0, jitter, size=len(vals))
        ax.scatter(xs, vals, s=POINT_SIZE, alpha=ALPHA_DOT,
                   color=COL_LOG2_POINT, edgecolors="none", rasterized=True)

    vals_all = df["log2(active/inactive)"].dropna().values
    if vals_all.size:
        y_min = min(float(np.nanmin(vals_all)), float(np.nanmin(means)), 0.0)
        y_max = max(float(np.nanmax(vals_all)), float(np.nanmax(means)), 0.0)
    else:
        y_min, y_max = min(float(np.nanmin(means)), 0.0), max(float(np.nanmax(means)), 0.0)
    y_rng = max(1e-6, y_max - y_min)

    for i, mu in enumerate(means):
        fold = 2.0 ** mu
        # Put labels slightly above positive bars and slightly below negative bars.
        if mu >= 0:
            y_txt = mu + 0.035 * y_rng
            va = "bottom"
        else:
            y_txt = mu - 0.035 * y_rng
            va = "top"
        ax.text(x[i], y_txt, f"×{fold:.2f}",
                ha="center", va=va, fontsize=FC_SIZE,
                color=EDGE, clip_on=False)

    ax.axhline(0, color="#777777", lw=0.45, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=0, ha="center")
    ax.set_ylabel(r"log$_2$(active/inactive)")
    ax.set_ylim(y_min - 0.10 * y_rng, y_max + 0.14 * y_rng)
    clean_axis(ax)


def plot_diff_matrix_lower_abs_ax(ax, pairs_df: pd.DataFrame,
                                  summary: pd.DataFrame, alpha=0.05,
                                  show_xlabels=True, show_ylabels=False):
    """
    Right column (C, F, I): lower-triangular |Δmean| matrix with stars.

    For compactness, the heatmap uses short class labels and omits the
    per-class means from the tick labels; these values remain available in
    the exported Excel summary sheet.
    """
    classes = summary["Class"].tolist()
    means = summary["mean"].tolist()
    K = len(classes)
    if K == 0:
        return None

    p_map = {}
    for _, r in pairs_df.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        p = float(r["p_adj"])
        p_map[(a, b)] = p
        p_map[(b, a)] = p

    D = np.full((K, K), np.nan, float)
    P = np.full((K, K), np.nan, float)
    for i in range(K):
        for j in range(i):
            mi, mj = means[i], means[j]
            D[i, j] = abs(mi - mj)
            P[i, j] = p_map.get((classes[i], classes[j]), np.nan)

    vmax = np.nanmax(D) if np.isfinite(D).any() else 1.0
    cmap = COLMAP_MATRIX
    cmap.set_bad(color=(1, 1, 1, 0))

    mD = np.ma.masked_invalid(D)
    im = ax.imshow(mD, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=1)
    ax.set_facecolor((1, 1, 1, 0))

    for i in range(K):
        for j in range(i):
            p = P[i, j]
            if np.isfinite(p) and p < alpha:
                ax.text(j, i, p_to_stars(p),
                        ha="center", va="center",
                        fontsize=STAR_SIZE, color="#111")

    tick_labels = compact_class_labels(classes)
    ax.set_xticks(np.arange(K))
    if show_xlabels:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=HEATMAP_TICK_SIZE)
    else:
        ax.set_xticklabels([])

    ax.set_yticks(np.arange(K))
    if show_ylabels:
        ax.set_yticklabels(tick_labels, fontsize=HEATMAP_TICK_SIZE)
    else:
        ax.set_yticklabels([])

    return im

# ---------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------
def process_dataset(xlsx_path: Path, alpha: float, roi_label: str):
    """Load, clean, run ANOVA + post-hoc + paired tests for one ROI."""
    print(f"[INFO] Processing {roi_label}: {xlsx_path}")
    df_all = load_table(xlsx_path)

    classes_order = [c for c in CLASS_ORDER_CANON if (df_all["Class"] == c).any()]
    if not classes_order:
        classes_order = [
            c for c in pd.unique(df_all["Class"])
            if str(c).strip() and str(c).strip().upper() != "CLASSLESS"
        ]

    df = remove_classless_and_outliers(df_all, z=2.0)
    df["Class"] = pd.Categorical(df["Class"],
                                 categories=classes_order, ordered=True)

    F, p, labels, arrays, df_within, MSE = one_way_anova(df)
    pairs_df, method = tukey_or_holm(df, df_within=df_within,
                                     MSE=MSE, alpha=alpha)
    summary = summarize_per_class(df)

    # annotate with ROI and method
    pairs_df = pairs_df.copy()
    pairs_df["ROI"] = roi_label
    pairs_df["posthoc_method"] = method

    summary = summary.copy()
    summary["ROI"] = roi_label

    # paired tests Active vs Inactive per class
    state_df = paired_tests_per_class(df, classes_order, roi_label)

    print(f"  ANOVA F={F:.3f}, p={p:.3g} | post-hoc: {method}")
    return df, summary, pairs_df, classes_order, state_df

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate 3×3 multi-ROI figure for extracellular cavity volumes and export stats to Excel."
    )
    ap.add_argument(
        "--portal-xlsx",
        default="volumen_5_portal_EC/gpcr_log2_active_inactive.xlsx",
        help="Path to portal ROI Excel.",
    )
    ap.add_argument(
        "--ecv-xlsx",
        default="volumen_2_ec_vestibule/gpcr_log2_active_inactive.xlsx",
        help="Path to EC vestibule ROI Excel.",
    )
    ap.add_argument(
        "--cleft-xlsx",
        default="volumen_3_cleft_volume/gpcr_log2_active_inactive.xlsx",
        help="Path to TM5–TM7 cleft ROI Excel.",
    )
    ap.add_argument(
        "--outfig",
        default="figure2_extracellular_cavities_narrow_heatmaps.png",
        help="Output raster image file, usually PNG or TIFF.",
    )
    ap.add_argument(
        "--outpdf",
        default="figure2_extracellular_cavities_narrow_heatmaps.pdf",
        help="Optional vector PDF output for manuscript layout. Use 'none' to skip.",
    )
    ap.add_argument(
        "--fig-width",
        type=float,
        default=7.6,
        help="Figure width in inches. 7.2–7.8 inches is typical for full two-column width.",
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=8.0,
        help="Figure height in inches.",
    )
    ap.add_argument(
        "--heatmap-width-ratio",
        type=float,
        default=0.50,
        help="Relative width of the heatmap column. Lower values give more room to the bar plots.",
    )
    ap.add_argument(
        "--excel-out",
        default="gpcr_extracellular_stats_BH.xlsx",
        help="Output Excel file with statistical comparisons.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold.",
    )
    ap.add_argument(
        "--pastel",
        type=float,
        default=PASTEL_MIX,
        help="Mixing factor with white for matrix colormap.",
    )
    args = ap.parse_args()

    apply_compact_style()
    global COLMAP_MATRIX
    COLMAP_MATRIX = make_pastel_cmap(mix=args.pastel)

    # Process each ROI
    portal_df, portal_summary, portal_pairs, portal_classes, portal_state = process_dataset(
        Path(args.portal_xlsx), args.alpha, roi_label="Portal"
    )
    ecv_df, ecv_summary, ecv_pairs, ecv_classes, ecv_state = process_dataset(
        Path(args.ecv_xlsx), args.alpha, roi_label="ECV"
    )
    cleft_df, cleft_summary, cleft_pairs, cleft_classes, cleft_state = process_dataset(
        Path(args.cleft_xlsx), args.alpha, roi_label="TM5–TM7 cleft"
    )

    # ==========================
    # Build compact 3×3 figure
    # ==========================
    fig, axes = plt.subplots(
        3, 3,
        figsize=(args.fig_width, args.fig_height),
        gridspec_kw={"width_ratios": [1.35, 1.15, args.heatmap_width_ratio]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.012, h_pad=0.02, wspace=0.035, hspace=0.08)

    # Row 1 – portal (A–C)
    axA, axB, axC = axes[0]
    plot_active_inactive_absolute_ax(axA, portal_df, portal_classes, portal_state)
    plot_means_transparent_ax(axB, portal_summary, portal_df)
    imC = plot_diff_matrix_lower_abs_ax(axC, portal_pairs, portal_summary,
                                        alpha=args.alpha,
                                        show_xlabels=False, show_ylabels=False)
    add_panel_label(axA, "A")
    add_panel_label(axB, "B")
    add_panel_label(axC, "C")


    # Row 2 – EC vestibule (D–F)
    axD, axE, axF = axes[1]
    plot_active_inactive_absolute_ax(axD, ecv_df, ecv_classes, ecv_state)
    plot_means_transparent_ax(axE, ecv_summary, ecv_df)
    imF = plot_diff_matrix_lower_abs_ax(axF, ecv_pairs, ecv_summary,
                                        alpha=args.alpha,
                                        show_xlabels=False, show_ylabels=False)
    add_panel_label(axD, "D")
    add_panel_label(axE, "E")
    add_panel_label(axF, "F")


    # Row 3 – TM5–TM7 cleft (G–I)
    axG, axH, axI = axes[2]
    plot_active_inactive_absolute_ax(axG, cleft_df, cleft_classes, cleft_state)
    plot_means_transparent_ax(axH, cleft_summary, cleft_df)
    imI = plot_diff_matrix_lower_abs_ax(axI, cleft_pairs, cleft_summary,
                                        alpha=args.alpha,
                                        show_xlabels=True, show_ylabels=False)
    add_panel_label(axG, "G")
    add_panel_label(axH, "H")
    add_panel_label(axI, "I")

    # One shared colorbar for all three heatmaps. This is much more compact
    # than one colorbar per row and gives more horizontal space to A/B, D/E, G/H.
    im_for_cb = imI if imI is not None else (imF if imF is not None else imC)
    if im_for_cb is not None:
        cb = fig.colorbar(
            im_for_cb, ax=[axC, axF, axI],
            fraction=0.030, pad=0.012, aspect=28
        )
        cb.set_label(r"|Δlog$_2$FC|", fontsize=LABEL_SIZE)
        cb.ax.tick_params(labelsize=TICK_SIZE, length=1.5, width=0.4)
        cb.outline.set_visible(False)

    # Shared legend only once, outside the axes.
    handles = [
        Patch(facecolor=COL_INACT_BAR, edgecolor=EDGE, linewidth=0.35,
              alpha=ALPHA_BAR, label="inactive"),
        Patch(facecolor=COL_ACT_BAR, edgecolor=EDGE, linewidth=0.35,
              alpha=ALPHA_BAR, label="active"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.50, 1.025),
               handlelength=1.2, columnspacing=1.2)

    fig.savefig(args.outfig, dpi=600, bbox_inches="tight")
    if str(args.outpdf).lower() not in {"", "none", "no", "false"}:
        fig.savefig(args.outpdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Combined figure written to {args.outfig}")
    if str(args.outpdf).lower() not in {"", "none", "no", "false"}:
        print(f"[OK] Vector PDF written to {args.outpdf}")

    # ==========================
    # Excel with statistics
    # ==========================
    state_all = pd.concat([portal_state, ecv_state, cleft_state],
                          ignore_index=True)
    class_all = pd.concat([portal_pairs, ecv_pairs, cleft_pairs],
                          ignore_index=True)
    summary_all = pd.concat([portal_summary, ecv_summary, cleft_summary],
                            ignore_index=True)

    with pd.ExcelWriter(args.excel_out) as writer:
        state_all.to_excel(
            writer, sheet_name="state_active_vs_inactive_BH", index=False
        )
        class_all.to_excel(
            writer, sheet_name="log2FC_between_classes_posthoc", index=False
        )
        summary_all.to_excel(
            writer, sheet_name="summary_by_class", index=False
        )

    print(f"[OK] Excel file written to {args.excel_out}")

if __name__ == "__main__":
    main()

