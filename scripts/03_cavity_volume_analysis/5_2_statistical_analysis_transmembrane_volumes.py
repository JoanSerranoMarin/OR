#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2 (two ROIs): extracellular cavity volumes and activation-dependent
changes across GPCR classes, combining two ROIs into a 2×3 panel figure.

Expects two Excel files named 'gpcr_log2_active_inactive.xlsx' in e.g.:

- volumen_5_portal_EC
- volumen_2_ec_vestibule

Outputs:
- A 2×3 figure (A–F) with:
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
# BH and sig lines
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
    """Draw a horizontal significance line only if p_adj < 0.05."""
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
# Style and colors (matched to ionic-locks script)
# ---------------------------------------------------------------------
COL_INACT_BAR   = "#dcdcdc"
COL_ACT_BAR     = "#f6d3a0"
COL_INACT_POINT = "#555555"
COL_ACT_POINT   = "#a45a15"

COL_LOG2_BAR    = COL_ACT_BAR
COL_LOG2_POINT  = COL_ACT_POINT

EDGE      = "#222222"
ALPHA_BAR = 0.85
ALPHA_DOT = 0.50

TEXT_SCALE = 2.0
TEXT_BASE_SIZE = 7
TEXT_SIZE = int(round(TEXT_BASE_SIZE * TEXT_SCALE))
TEXT_SIZE_FC = max(6, TEXT_SIZE - 2)  # smaller for ×fold labels

POINT_SIZE = 8  # smaller scatter points

PASTEL_MIX = 0.55

def apply_text_scaling(scale: float = 2.0):
    """Globally scale matplotlib default font size."""
    base = float(plt.rcParams.get("font.size", 10.0))
    plt.rcParams["font.size"] = base * scale

def make_pastel_cmap(hex_color=COL_LOG2_BAR, mix=PASTEL_MIX):
    base = np.array(to_rgb(hex_color))
    pastel = 1 - (1 - base) * mix
    return LinearSegmentedColormap.from_list(
        "pastel_orange", [(1,1,1), tuple(pastel)], N=256
    )

COLMAP_MATRIX = make_pastel_cmap()

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_active_inactive_absolute_ax(ax, df: pd.DataFrame, classes_order,
                                     state_tests_df: pd.DataFrame):
    """
    Left column: absolute volumes, inactive vs active, with BH-corrected
    paired t-test lines above each pair of bars.
    """
    classes = [c for c in classes_order if (df["Class"] == c).any()]
    labels_display = class_display_list(classes)
    x = np.arange(len(classes))
    width = 0.38

    mean_inact = (df.groupby("Class", sort=False)["Inactive"]
                    .mean()
                    .reindex(classes))
    mean_act   = (df.groupby("Class", sort=False)["Active"]
                    .mean()
                    .reindex(classes))

    ax.bar(x - width/2, mean_inact.values, width=width,
           color=COL_INACT_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=0.6)
    ax.bar(x + width/2, mean_act.values, width=width,
           color=COL_ACT_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=0.6)

    rng = np.random.default_rng(123)
    jitter = 0.08
    for i, cls in enumerate(classes):
        vals_in = df.loc[df["Class"]==cls, "Inactive"].dropna().values
        vals_ac = df.loc[df["Class"]==cls, "Active"].dropna().values
        xs_in = (x[i] - width/2) + rng.normal(0, jitter, size=len(vals_in))
        xs_ac = (x[i] + width/2) + rng.normal(0, jitter, size=len(vals_ac))
        ax.scatter(xs_in, vals_in, s=POINT_SIZE, alpha=ALPHA_DOT,
                   color=COL_INACT_POINT, edgecolors="none")
        ax.scatter(xs_ac, vals_ac, s=POINT_SIZE, alpha=ALPHA_DOT,
                   color=COL_ACT_POINT, edgecolors="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=45, ha="right")
    ax.set_ylabel("Volume (Å³)")

    if state_tests_df is None or state_tests_df.empty:
        return

    p_map = {row["Class"]: row["p_adj_BH"]
             for _, row in state_tests_df.iterrows()}

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    base_line = y_max - 0.08 * y_range
    dy = 0.02 * y_range

    for i, cls in enumerate(classes):
        p_adj = p_map.get(cls, np.nan)
        if np.isnan(p_adj) or p_adj >= 0.05:
            continue
        x_center = x[i]
        x1 = x_center - width/2
        x2 = x_center + width/2
        add_sig_line(ax, x1, x2, base_line, dy, p_adj,
                     fontsize=TEXT_SIZE_FC)

def plot_means_transparent_ax(ax, summary: pd.DataFrame, df: pd.DataFrame):
    """
    Middle column: mean log2(Active/Inactive) per class + points.
    Uses brown palette, fold-change labels slightly smaller.
    """
    classes = summary["Class"].tolist()
    labels_display = class_display_list(classes)
    x = np.arange(len(classes))
    means = summary["mean"].values

    ax.bar(x, means,
           color=COL_LOG2_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=0.6)

    rng = np.random.default_rng(42)
    jitter = 0.10
    for i, cls in enumerate(classes):
        vals = df.loc[df["Class"]==cls, "log2(active/inactive)"].values
        xs = x[i] + rng.normal(0, jitter, size=len(vals))
        ax.scatter(xs, vals, s=POINT_SIZE, alpha=ALPHA_DOT,
                   color=COL_LOG2_POINT, edgecolors="none")

    y_min = np.nanmin(np.concatenate([means, [0.0]]))
    y_max = np.nanmax(np.concatenate([means, [0.0]]))
    y_rng = max(1e-6, y_max - y_min)
    base_off = 0.10 * y_rng

    for i, mu in enumerate(means):
        fold = 2.0**mu
        y_txt = mu + base_off
        ax.text(
            x[i], y_txt, f"×{fold:.2f}",
            ha="center", va="bottom",
            fontsize=TEXT_SIZE_FC,
            color=EDGE, clip_on=False,
        )

    ax.axhline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=45, ha="right")
    ax.set_ylabel(r"Log$_2$(Active/Inactive)")

def plot_diff_matrix_lower_abs_ax(ax, pairs_df: pd.DataFrame,
                                  summary: pd.DataFrame, alpha=0.05):
    """
    Right column: lower-triangular |Δmean| matrix with stars.
    """
    classes = summary["Class"].tolist()
    means   = summary["mean"].tolist()
    K = len(classes)
    if K == 0:
        return None

    p_map = {}
    for _, r in pairs_df.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        p = float(r["p_adj"])
        p_map[(a,b)] = p
        p_map[(b,a)] = p

    D = np.full((K,K), np.nan, float)
    P = np.full((K,K), np.nan, float)
    for i in range(K):
        for j in range(i):
            mi, mj = means[i], means[j]
            D[i,j] = abs(mi - mj)
            P[i,j] = p_map.get((classes[i], classes[j]), np.nan)

    vmax = np.nanmax(D) if np.isfinite(D).any() else 1.0
    cmap = COLMAP_MATRIX
    cmap.set_bad(color=(1,1,1,0))

    mD = np.ma.masked_invalid(D)
    im = ax.imshow(mD, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor((1,1,1,0))

    for i in range(K):
        for j in range(i):
            p = P[i,j]
            if np.isfinite(p) and p < alpha:
                ax.text(j, i, p_to_stars(p),
                        ha="center", va="center",
                        fontsize=TEXT_SIZE+1, color="#111")

    xt = [class_display_name(c) for c in classes]
    yt = [f"{class_display_name(c)}\n(mean={m:.2f})"
          for c, m in zip(classes, means)]
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=TEXT_SIZE+1)
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels(yt, fontsize=TEXT_SIZE+1)
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

    pairs_df = pairs_df.copy()
    pairs_df["ROI"] = roi_label
    pairs_df["posthoc_method"] = method

    summary = summary.copy()
    summary["ROI"] = roi_label

    state_df = paired_tests_per_class(df, classes_order, roi_label)

    print(f"  ANOVA F={F:.3f}, p={p:.3g} | post-hoc: {method}")
    return df, summary, pairs_df, classes_order, state_df

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate 2×3 multi-ROI figure (two ROIs) for extracellular cavity volumes and export stats to Excel."
    )
    ap.add_argument(
        "--roi1-xlsx",
        default="volumen_5_portal_EC/gpcr_log2_active_inactive.xlsx",
        help="Path to ROI 1 Excel (e.g. Portal).",
    )
    ap.add_argument(
        "--roi1-label",
        default="Portal",
        help="Label for ROI 1 (for Excel and printing).",
    )
    ap.add_argument(
        "--roi2-xlsx",
        default="volumen_2_ec_vestibule/gpcr_log2_active_inactive.xlsx",
        help="Path to ROI 2 Excel (e.g. ECV).",
    )
    ap.add_argument(
        "--roi2-label",
        default="ECV",
        help="Label for ROI 2.",
    )
    ap.add_argument(
        "--outfig",
        default="figure2_twoROIs_extracellular_cavities.png",
        help="Output PNG file.",
    )
    ap.add_argument(
        "--excel-out",
        default="gpcr_extracellular_stats_twoROIs_BH.xlsx",
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

    apply_text_scaling(TEXT_SCALE)
    global COLMAP_MATRIX
    COLMAP_MATRIX = make_pastel_cmap(mix=args.pastel)

    # Process the two ROIs
    roi1_df, roi1_summary, roi1_pairs, roi1_classes, roi1_state = process_dataset(
        Path(args.roi1_xlsx), args.alpha, roi_label=args.roi1_label
    )
    roi2_df, roi2_summary, roi2_pairs, roi2_classes, roi2_state = process_dataset(
        Path(args.roi2_xlsx), args.alpha, roi_label=args.roi2_label
    )

    # ==========================
    # Build figure 2×3
    # ==========================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(wspace=0.4, hspace=0.5)

    # Row 1 – ROI 1 (A–C)
    axA, axB, axC = axes[0]
    plot_active_inactive_absolute_ax(axA, roi1_df, roi1_classes, roi1_state)
    plot_means_transparent_ax(axB, roi1_summary, roi1_df)
    imC = plot_diff_matrix_lower_abs_ax(axC, roi1_pairs, roi1_summary,
                                        alpha=args.alpha)

    axA.text(-0.1, 1.05, "A)", transform=axA.transAxes,
             fontsize=TEXT_SIZE+4, fontweight="bold")
    axB.text(-0.1, 1.05, "B)", transform=axB.transAxes,
             fontsize=TEXT_SIZE+4, fontweight="bold")
    axC.text(-0.1, 1.05, "C)", transform=axC.transAxes,
             fontsize=TEXT_SIZE+4, fontweight="bold")

    if imC is not None:
        cbC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
        cbC.set_label(r"|meanᵢ − meanⱼ| (Å³)")
        cbC.outline.set_visible(False)

    # Row 2 – ROI 2 (D–F)
    axD, axE, axF = axes[1]
    plot_active_inactive_absolute_ax(axD, roi2_df, roi2_classes, roi2_state)
    plot_means_transparent_ax(axE, roi2_summary, roi2_df)
    imF = plot_diff_matrix_lower_abs_ax(axF, roi2_pairs, roi2_summary,
                                        alpha=args.alpha)

    axD.text(-0.1, 1.05, "D)", transform=axD.transAxes,
             fontsize=TEXT_SIZE+4, fontweight="bold")
    axE.text(-0.1, 1.05, "E)", transform=axE.transAxes,
             fontsize=TEXT_SIZE+4, fontweight="bold")
    axF.text(-0.1, 1.05, "F)", transform=axF.transAxes,
             fontsize=TEXT_SIZE+4, fontweight="bold")

    if imF is not None:
        cbF = fig.colorbar(imF, ax=axF, fraction=0.046, pad=0.04)
        cbF.set_label(r"|meanᵢ − meanⱼ| (Å³)")
        cbF.outline.set_visible(False)

    plt.tight_layout()
    fig.savefig(args.outfig, dpi=300)
    plt.close(fig)
    print(f"[OK] Two-ROI figure written to {args.outfig}")

    # ==========================
    # Excel with statistics
    # ==========================
    state_all = pd.concat([roi1_state, roi2_state], ignore_index=True)
    class_all = pd.concat([roi1_pairs, roi2_pairs], ignore_index=True)
    summary_all = pd.concat([roi1_summary, roi2_summary], ignore_index=True)

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

