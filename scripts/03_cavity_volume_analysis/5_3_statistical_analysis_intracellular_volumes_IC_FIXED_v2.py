#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compact single-region GPCR volume figure for the intracellular cavity.

This version uses, by default:
    volumen_6_IC/gpcr_log2_active_inactive.xlsx

It replaces the right-column post-hoc heatmap with compact letter labels above
the log2FC panel. The resulting layout is 1×2:

    intracellular cavity volume      log2(active/inactive) + post-hoc letters

Compact letter display interpretation:
classes sharing at least one letter are not significantly different; classes
sharing no letters are significantly different according to the post-hoc test.

Expected input: one Excel file containing at least the canonical columns or
recognized synonyms for:
    GPCR_name, Inactive, Active, log2(active/inactive), Class

Outputs:
    - PNG figure
    - PDF vector figure
    - Excel file with paired tests, post-hoc comparisons, summary statistics,
      and compact post-hoc letters.
"""

import argparse
from pathlib import Path
import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats


# ---------------------------------------------------------------------
# Input columns and class order
# ---------------------------------------------------------------------
REQ_COLS = {"GPCR_name", "Inactive", "Active", "log2(active/inactive)", "Class"}

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


# ---------------------------------------------------------------------
# Plot style: designed for a full-width panel in a two-column paper
# ---------------------------------------------------------------------
COL_INACT_BAR = "#dcdcdc"
COL_ACT_BAR = "#f6d3a0"
COL_INACT_POINT = "#555555"
COL_ACT_POINT = "#a45a15"
COL_LOG2_BAR = COL_ACT_BAR
COL_LOG2_POINT = COL_ACT_POINT
EDGE = "#222222"

ALPHA_BAR = 0.90
ALPHA_DOT = 0.46
POINT_SIZE_ABS = 5.0
POINT_SIZE_LOG2 = 5.0
BAR_EDGE_LW = 0.55
AXIS_LW = 0.85
GRID_LW = 0.45

FONT_SIZE = 9
LABEL_SIZE = 10
TICK_SIZE = 8.5
PANEL_SIZE = 13
STAR_SIZE = 9
LETTER_SIZE = 9
FOLD_SIZE = 7.3


def set_style():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.linewidth": AXIS_LW,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------
def class_display_name(name: str) -> str:
    """Return short labels: 'Class A' -> 'A', 'Class B1' -> 'B1'."""
    s = str(name).strip()
    s = re.sub(r"\s*\([^)]*\)\s*", "", s).strip()
    s = re.sub(r"^Class\s+", "", s, flags=re.IGNORECASE).strip()
    return s


def class_display_list(names):
    return [class_display_name(n) for n in names]


def infer_roi_label_from_path(xlsx_path) -> str:
    """
    Infer a human-readable region name from the input Excel path.

    The script expects files such as:
        volumen_7_OPV/gpcr_log2_active_inactive.xlsx
        volumen_1_hydrophobic_slab/gpcr_log2_active_inactive.xlsx
        volumen_5_portal_EC/gpcr_log2_active_inactive.xlsx

    The inferred label is used in plot titles and in the output Excel tables.
    A manually supplied --roi*-label value still overrides this inference.
    """
    p = Path(xlsx_path)
    folder = p.parent.name
    full = "/".join(p.parts).lower()
    f = folder.lower()

    # Specific patterns first.
    if "portal_throat" in full or "throat" in full:
        return "Portal throat"
    if ("portal" in f or "portal" in full) and ("ec" in f or "extracellular" in full):
        return "EC portal"
    if "ec_vestibule" in full or "extracellular_vestibule" in full or "vestibule" in f:
        return "EC vestibule"
    if "cleaft" in full or "cleft" in full:
        # This is the region previously referred to in the figure as TM5–TM7 cleft.
        return "TM5–TM7 cleft"
    if "intracellular" in full or re.search(r"(^|[_\-/])ic([_\-/]|$)", full) or "cytoplasmic" in full:
        return "Intracellular free volume"
    if "hydrophobic_slab" in full or ("hydrophobic" in full and "slab" in full):
        return "Hydrophobic slab"
    if "opv" in full or "orthosteric" in full:
        return "Orthosteric pocket"
    if "tm1" in full and "tm7" in full:
        return "TM1–TM7 cleft"
    if "tm5" in full and "tm6" in full:
        return "TM5–TM6 portal"
    if "lateral" in full:
        return "Lateral/orthosteric region"

    # Fallback: remove leading numbering and make a readable title.
    name = re.sub(r"^volumen[_-]*\d+[_-]*", "", folder, flags=re.IGNORECASE)
    name = name.replace("gpcr_log2_active_inactive", "")
    name = re.sub(r"[_-]+", " ", name).strip()
    return name.title() if name else "Volume region"


def _normalize_colname(c):
    return re.sub(r"\s+", " ", str(c).strip()).lower()


def detect_column_mapping(df_columns):
    cols_norm = {_normalize_colname(c): c for c in df_columns}
    mapping = {}
    for canonical, aliases in SYNONYMS.items():
        for alias in aliases:
            if alias in cols_norm:
                mapping[canonical] = cols_norm[alias]
                break
    return mapping


# ---------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------
def load_table(xlsx_path: Path, sheet=None) -> pd.DataFrame:
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
            "Could not find 'Active'/'Inactive' columns nor precomputed log2 ratio.\n"
            f"Columns found: {list(df.columns)}"
        )
    if "Class" not in mapping:
        raise ValueError("Could not find a 'Class' column.")
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
        d.loc[~np.isfinite(d["Active"]) | (d["Active"] <= 0), "Active"] = np.nan
        d["log2(active/inactive)"] = np.log2(d["Active"] / d["Inactive"])

    missing = REQ_COLS - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns after mapping: {missing}")

    used = {k: mapping.get(k, k) for k in [
        "GPCR_name", "Inactive", "Active", "Class", "log2(active/inactive)"
    ]}
    print(f"[INFO] Column mapping -> canonical: {used}")

    return d[["GPCR_name", "Inactive", "Active", "log2(active/inactive)", "Class"]].copy()


def remove_classless_and_outliers(df: pd.DataFrame, z=2.0) -> pd.DataFrame:
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
# Statistics
# ---------------------------------------------------------------------
def one_way_anova(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class"):
    groups = [g[ycol].dropna().values for _, g in df.groupby(gcol, sort=False, observed=True)]
    labels = [str(n) for n, _ in df.groupby(gcol, sort=False, observed=True)]
    valid = [(lab, arr) for lab, arr in zip(labels, groups) if len(arr) >= 2]
    if len(valid) < 2:
        raise ValueError("Need at least two classes with at least two observations after filtering.")

    labels, groups = zip(*valid)
    F, p = stats.f_oneway(*groups)
    SSE = sum(((arr - np.mean(arr)) ** 2).sum() for arr in groups)
    df_within = sum(len(arr) for arr in groups) - len(groups)
    MSE = SSE / df_within if df_within > 0 else np.nan
    return F, p, list(labels), list(groups), df_within, MSE


def tukey_or_holm(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class",
                  df_within=None, MSE=None, alpha=0.05):
    means = df.groupby(gcol, sort=False, observed=True)[ycol].mean().to_dict()

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        res = pairwise_tukeyhsd(endog=df[ycol].values,
                                groups=df[gcol].astype(str).values,
                                alpha=alpha)
        out = pd.DataFrame(res._results_table.data[1:],
                           columns=res._results_table.data[0])
        out = out.rename(columns={"p-adj": "p_adj", "reject": "reject",
                                  "meandiff": "meandiff"})
        out["p_adj"] = out["p_adj"].astype(float)
        out["reject"] = out["reject"].astype(bool)
        method = "TukeyHSD"

    except Exception:
        from itertools import combinations
        from scipy.stats import t as t_dist

        if df_within is None or MSE is None or not np.isfinite(MSE):
            raise RuntimeError("No Tukey available and missing MSE/df for Holm fallback.")

        groups_dict = {
            str(k): v.dropna().values
            for k, v in df.groupby(gcol, sort=False, observed=True)[ycol]
        }
        rows = []
        for g1, g2 in combinations(list(groups_dict.keys()), 2):
            x1, x2 = groups_dict[g1], groups_dict[g2]
            if len(x1) < 2 or len(x2) < 2:
                continue
            n1, n2 = len(x1), len(x2)
            m1, m2 = x1.mean(), x2.mean()
            se = np.sqrt(MSE * (1 / n1 + 1 / n2)) if MSE > 0 else np.nan
            tval = (m1 - m2) / se if np.isfinite(se) and se > 0 else np.nan
            p_raw = 2 * (1 - t_dist.cdf(abs(tval), df=df_within)) if np.isfinite(tval) else np.nan
            rows.append({"group1": g1, "group2": g2,
                         "meandiff": m1 - m2, "p_raw": p_raw})

        out = pd.DataFrame(rows).sort_values("p_raw")
        m = len(out)
        out["p_adj"] = [min(1.0, (m - i + 1) * p)
                        for i, p in enumerate(out["p_raw"], start=1)]
        out["reject"] = out["p_adj"] < alpha
        method = "Holm"

    out["group1"] = out["group1"].astype(str)
    out["group2"] = out["group2"].astype(str)
    out["mean1"] = out["group1"].map(means)
    out["mean2"] = out["group2"].map(means)
    out["higher"] = np.where(out["mean1"] > out["mean2"], out["group1"], out["group2"])
    return out[["group1", "group2", "mean1", "mean2", "meandiff", "p_adj", "reject", "higher"]], method


def summarize_per_class(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class"):
    s = df.groupby(gcol, sort=False, observed=True).agg(
        n=(ycol, "size"), mean=(ycol, "mean")
    ).reset_index()

    if pd.api.types.is_categorical_dtype(df[gcol]):
        cats = list(df[gcol].cat.categories)
    else:
        cats = CLASS_ORDER_CANON

    present = set(s[gcol].astype(str))
    s = s.set_index(gcol).reindex([c for c in cats if str(c) in present]).reset_index()
    return s


def benjamini_hochberg(pvals):
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


def p_to_stars(p):
    if pd.isna(p):
        return "ns"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def paired_tests_per_class(df: pd.DataFrame, classes_order, roi_label: str):
    rows = []
    for cls in classes_order:
        sub = df[df["Class"] == cls][["Active", "Inactive"]].dropna()
        if len(sub) <= 1:
            continue
        t_stat, p_raw = stats.ttest_rel(sub["Active"], sub["Inactive"])
        rows.append({
            "ROI": roi_label,
            "Class": cls,
            "n": int(len(sub)),
            "mean_active": float(sub["Active"].mean()),
            "mean_inactive": float(sub["Inactive"].mean()),
            "t_stat": float(t_stat),
            "p_raw": float(p_raw),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ROI", "Class", "n", "mean_active", "mean_inactive",
            "t_stat", "p_raw", "p_adj_BH"
        ])

    out = pd.DataFrame(rows)
    out["p_adj_BH"] = benjamini_hochberg(out["p_raw"].values)
    return out


# ---------------------------------------------------------------------
# Compact letter display for post-hoc class comparisons
# ---------------------------------------------------------------------
def _letter_sequence():
    base = list(string.ascii_lowercase)
    for l in base:
        yield l
    for a in base:
        for b in base:
            yield a + b


def compact_letter_display(summary: pd.DataFrame, pairs_df: pd.DataFrame, alpha=0.05):
    """
    Greedy compact letter display.

    Classes sharing at least one letter are not significantly different.
    Classes sharing no letters are significantly different. The algorithm is
    conservative when a comparison is missing: missing pairs are treated as
    significant to avoid grouping classes without an available comparison.
    """
    classes = [str(c) for c in summary["Class"].tolist()]
    means = {str(r["Class"]): float(r["mean"]) for _, r in summary.iterrows()}

    sig = {(a, b): False for a in classes for b in classes}
    for _, r in pairs_df.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        p = float(r["p_adj"])
        is_sig = np.isfinite(p) and p < alpha
        sig[(a, b)] = is_sig
        sig[(b, a)] = is_sig

    def significantly_different(a, b):
        if a == b:
            return False
        if (a, b) not in sig:
            return True
        return sig[(a, b)]

    sorted_classes = sorted(classes, key=lambda c: means.get(c, np.nan), reverse=True)

    letters = []
    letter_names = []
    names_iter = _letter_sequence()

    for c in sorted_classes:
        placed = False
        for L in letters:
            if all(not significantly_different(c, existing) for existing in L):
                L.add(c)
                placed = True
        if not placed:
            letters.append({c})
            letter_names.append(next(names_iter))

    changed = True
    while changed:
        changed = False
        for c in sorted_classes:
            for L in letters:
                if c in L:
                    continue
                if all(not significantly_different(c, existing) for existing in L):
                    L.add(c)
                    changed = True

    class_to_letters = {c: "" for c in classes}
    for lname, L in zip(letter_names, letters):
        for c in L:
            class_to_letters[c] += lname

    out = summary[["Class", "n", "mean"]].copy()
    out["fold_change"] = np.power(2.0, out["mean"])
    out["posthoc_letters"] = out["Class"].astype(str).map(class_to_letters)
    return class_to_letters, out


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e7e7e7", lw=GRID_LW, zorder=0)
    ax.set_axisbelow(True)


def add_sig_line(ax, x1, x2, y, dy, p_adj, fontsize=STAR_SIZE):
    if np.isnan(p_adj) or p_adj >= 0.05:
        return
    ax.plot([x1, x2], [y, y], lw=0.9, c="black", clip_on=False)
    ax.text((x1 + x2) / 2.0, y + dy, p_to_stars(p_adj),
            ha="center", va="bottom", fontsize=fontsize, clip_on=False)


def plot_active_inactive_absolute_ax(ax, df: pd.DataFrame, classes_order,
                                     state_tests_df: pd.DataFrame,
                                     show_ylabel=True):
    classes = [c for c in classes_order if (df["Class"] == c).any()]
    labels_display = class_display_list(classes)
    x = np.arange(len(classes))
    width = 0.38

    mean_inact = df.groupby("Class", sort=False, observed=True)["Inactive"].mean().reindex(classes)
    mean_act = df.groupby("Class", sort=False, observed=True)["Active"].mean().reindex(classes)

    ax.bar(x - width / 2, mean_inact.values, width=width,
           color=COL_INACT_BAR, alpha=ALPHA_BAR, edgecolor=EDGE,
           linewidth=BAR_EDGE_LW, zorder=2)
    ax.bar(x + width / 2, mean_act.values, width=width,
           color=COL_ACT_BAR, alpha=ALPHA_BAR, edgecolor=EDGE,
           linewidth=BAR_EDGE_LW, zorder=2)

    rng = np.random.default_rng(123)
    jitter = 0.065
    for i, cls in enumerate(classes):
        vals_in = df.loc[df["Class"] == cls, "Inactive"].dropna().values
        vals_ac = df.loc[df["Class"] == cls, "Active"].dropna().values
        ax.scatter((x[i] - width / 2) + rng.normal(0, jitter, size=len(vals_in)),
                   vals_in, s=POINT_SIZE_ABS, alpha=ALPHA_DOT,
                   color=COL_INACT_POINT, edgecolors="none", rasterized=True, zorder=3)
        ax.scatter((x[i] + width / 2) + rng.normal(0, jitter, size=len(vals_ac)),
                   vals_ac, s=POINT_SIZE_ABS, alpha=ALPHA_DOT,
                   color=COL_ACT_POINT, edgecolors="none", rasterized=True, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_display)
    if show_ylabel:
        ax.set_ylabel("Volume (Å³)")
    clean_axis(ax)

    ymin, ymax = ax.get_ylim()
    data_max = np.nanmax(np.r_[df["Inactive"].values, df["Active"].values])
    ax.set_ylim(ymin, data_max * 1.18 if data_max > 0 else ymax)

    if state_tests_df is None or state_tests_df.empty:
        return

    p_map = {row["Class"]: row["p_adj_BH"] for _, row in state_tests_df.iterrows()}
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    dy = 0.012 * yr

    for i, cls in enumerate(classes):
        p_adj = p_map.get(cls, np.nan)
        if np.isnan(p_adj) or p_adj >= 0.05:
            continue

        local = df.loc[df["Class"] == cls, ["Inactive", "Active"]].to_numpy().ravel()
        local = local[np.isfinite(local)]
        local_max = np.nanmax(local) if len(local) else np.nan
        y = min(local_max + 0.055 * yr, ymax - 0.075 * yr)
        add_sig_line(ax, x[i] - width / 2, x[i] + width / 2,
                     y, dy, p_adj, fontsize=STAR_SIZE)


def plot_log2_fc_letters_ax(ax, summary: pd.DataFrame, df: pd.DataFrame,
                            class_to_letters: dict,
                            show_ylabel=True,
                            show_fold_labels=True):
    classes = summary["Class"].tolist()
    labels_display = class_display_list(classes)
    x = np.arange(len(classes))
    means = summary["mean"].values.astype(float)

    ax.bar(x, means, width=0.62, color=COL_LOG2_BAR, alpha=ALPHA_BAR,
           edgecolor=EDGE, linewidth=BAR_EDGE_LW, zorder=2)

    rng = np.random.default_rng(42)
    jitter = 0.08
    for i, cls in enumerate(classes):
        vals = df.loc[df["Class"] == cls, "log2(active/inactive)"].dropna().values
        ax.scatter(x[i] + rng.normal(0, jitter, size=len(vals)),
                   vals, s=POINT_SIZE_LOG2, alpha=ALPHA_DOT,
                   color=COL_LOG2_POINT, edgecolors="none", rasterized=True, zorder=3)

    finite_vals = df["log2(active/inactive)"].dropna().values
    y_min = min(np.nanmin(finite_vals), np.nanmin(means), 0.0)
    y_max = max(np.nanmax(finite_vals), np.nanmax(means), 0.0)
    yr = max(1e-6, y_max - y_min)

    letter_y = y_max + 0.18 * yr
    ax.set_ylim(y_min - 0.12 * yr, y_max + 0.34 * yr)

    for i, cls in enumerate(classes):
        letters = class_to_letters.get(str(cls), "")
        ax.text(x[i], letter_y, letters, ha="center", va="bottom",
                fontsize=LETTER_SIZE, fontweight="bold", clip_on=False)

    if show_fold_labels:
        off = 0.045 * yr
        for i, mu in enumerate(means):
            fold = 2.0 ** mu
            if mu >= 0:
                y_txt = mu + off
                va = "bottom"
            else:
                y_txt = mu - off
                va = "top"
            ax.text(x[i], y_txt, f"×{fold:.2f}", ha="center", va=va,
                    fontsize=FOLD_SIZE, color=EDGE, clip_on=False)

    ax.axhline(0, color="#888888", lw=0.75, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display)
    if show_ylabel:
        ax.set_ylabel(r"log$_2$(active/inactive)")
    clean_axis(ax)


def add_panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=PANEL_SIZE, fontweight="bold", ha="left", va="bottom")


# ---------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------
def process_dataset(xlsx_path: Path, alpha: float, roi_label: str):
    print(f"[INFO] Processing {roi_label}: {xlsx_path}")
    df_all = load_table(xlsx_path)

    classes_order = [c for c in CLASS_ORDER_CANON if (df_all["Class"] == c).any()]
    if not classes_order:
        classes_order = [
            c for c in pd.unique(df_all["Class"])
            if str(c).strip() and str(c).strip().upper() != "CLASSLESS"
        ]

    df = remove_classless_and_outliers(df_all, z=2.0)
    df["Class"] = pd.Categorical(df["Class"], categories=classes_order, ordered=True)

    F, p, labels, arrays, df_within, MSE = one_way_anova(df)
    pairs_df, method = tukey_or_holm(df, df_within=df_within, MSE=MSE, alpha=alpha)
    summary = summarize_per_class(df)

    pairs_df = pairs_df.copy()
    pairs_df["ROI"] = roi_label
    pairs_df["posthoc_method"] = method

    summary = summary.copy()
    summary["ROI"] = roi_label

    state_df = paired_tests_per_class(df, classes_order, roi_label)

    class_to_letters, letters_df = compact_letter_display(summary, pairs_df, alpha=alpha)
    letters_df["ROI"] = roi_label
    letters_df["posthoc_method"] = method

    print(f"  ANOVA F={F:.3f}, p={p:.3g} | post-hoc: {method}")
    return df, summary, pairs_df, classes_order, state_df, class_to_letters, letters_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate compact 1×2 intracellular-cavity volume figure from volumen_6_IC/gpcr_log2_active_inactive.xlsx."
    )
    ap.add_argument(
        "--xlsx",
        default="volumen_6_IC/gpcr_log2_active_inactive.xlsx",
        help="Path to the intracellular-cavity Excel file. The region name is inferred from the parent folder unless --roi-label is provided.",
    )
    ap.add_argument(
        "--roi-label",
        default=None,
        help="Optional manual label for the region. By default, it is inferred from the input folder name.",
    )
    ap.add_argument(
        "--outfig",
        default="figure2_oneROI_intracellular_cavities.png",
        help="Output PNG file.",
    )
    ap.add_argument(
        "--outpdf",
        default="figure2_oneROI_intracellular_cavities.pdf",
        help="Output PDF vector file.",
    )
    ap.add_argument(
        "--excel-out",
        default="gpcr_intracellular_stats_oneROI_BH.xlsx",
        help="Output Excel file with statistics and compact letters.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold.",
    )
    ap.add_argument(
        "--fig-width",
        type=float,
        default=7.25,
        help="Figure width in inches; 7.0–7.4 is typical for a full-width two-column figure.",
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=3.05,
        help="Figure height in inches.",
    )
    ap.add_argument(
        "--no-fold-labels",
        action="store_true",
        help="Do not print fold-change labels on the log2FC panel.",
    )
    args = ap.parse_args()

    set_style()

    roi_label = args.roi_label or infer_roi_label_from_path(args.xlsx)
    print(f"[INFO] Region label used for input: {roi_label}")

    df, summary, pairs, classes_order, state, class_to_letters, letters_df = process_dataset(
        Path(args.xlsx), args.alpha, roi_label=roi_label
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(args.fig_width, args.fig_height),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.27},
    )
    # Leave a dedicated top margin for the legend so it does not overlap
    # the panel letters or titles when the figure is saved with tight bounds.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.205, top=0.79)
    ax_abs, ax_log = axes

    plot_active_inactive_absolute_ax(ax_abs, df, classes_order, state, show_ylabel=True)
    plot_log2_fc_letters_ax(
        ax_log,
        summary,
        df,
        class_to_letters,
        show_ylabel=True,
        show_fold_labels=not args.no_fold_labels,
    )

    ax_abs.set_title(roi_label, fontsize=LABEL_SIZE, pad=5)
    ax_log.set_title(f"{roi_label}: activation effect", fontsize=LABEL_SIZE, pad=5)

    add_panel_label(ax_abs, "A")
    add_panel_label(ax_log, "B")

    handles = [
        Patch(facecolor=COL_INACT_BAR, edgecolor=EDGE, linewidth=BAR_EDGE_LW, label="inactive"),
        Patch(facecolor=COL_ACT_BAR, edgecolor=EDGE, linewidth=BAR_EDGE_LW, label="active"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.19),
        handlelength=1.2,
        columnspacing=1.3,
        borderaxespad=0.0,
    )

    fig.text(
        0.99,
        0.01,
        "Letters: classes sharing a letter are not significantly different.",
        ha="right",
        va="bottom",
        fontsize=7.1,
    )

    fig.savefig(args.outfig, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(args.outpdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"[OK] Figure written to {args.outfig}")
    print(f"[OK] Vector figure written to {args.outpdf}")

    with pd.ExcelWriter(args.excel_out) as writer:
        state.to_excel(writer, sheet_name="state_active_vs_inactive_BH", index=False)
        pairs.to_excel(writer, sheet_name="log2FC_between_classes_posthoc", index=False)
        summary.to_excel(writer, sheet_name="summary_by_class", index=False)
        letters_df.to_excel(writer, sheet_name="posthoc_letters", index=False)

    print(f"[OK] Excel file written to {args.excel_out}")


if __name__ == "__main__":
    main()
