#!/usr/bin/env python3
"""Pairwise comparison of GPCR-family log2 fold changes."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


FAMILY = "Receptor family"
LOG2FC = "log2(active/inactive)"


def sig_label(p: float) -> str:
    if not np.isfinite(p):
        return "NE"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def analyse(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    data = pd.read_excel(input_path)
    missing = {FAMILY, LOG2FC}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data[FAMILY] = data[FAMILY].astype("string").str.strip()
    data[LOG2FC] = pd.to_numeric(data[LOG2FC], errors="coerce")
    data[LOG2FC] = data[LOG2FC].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[FAMILY])

    groups = {
        str(name): group[LOG2FC].dropna().to_numpy(float)
        for name, group in data.groupby(FAMILY, sort=True)
    }
    summary_rows = []
    for name, values in groups.items():
        n = len(values)
        summary_rows.append(
            {
                FAMILY: name,
                "n_log2FC": n,
                "mean_log2FC": np.mean(values) if n else np.nan,
                "median_log2FC": np.median(values) if n else np.nan,
                "sd_log2FC": np.std(values, ddof=1) if n >= 2 else np.nan,
                "sem_log2FC": stats.sem(values) if n >= 2 else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(FAMILY).reset_index(drop=True)

    valid_groups = [values for values in groups.values() if len(values) >= 2]
    if len(valid_groups) >= 2:
        omnibus_f, omnibus_p = stats.f_oneway(*valid_groups)
    else:
        omnibus_f, omnibus_p = np.nan, np.nan

    pair_rows = []
    for family_1, family_2 in combinations(groups, 2):
        values_1, values_2 = groups[family_1], groups[family_2]
        testable = len(values_1) >= 2 and len(values_2) >= 2
        statistic, p_raw = np.nan, np.nan
        if testable:
            result = stats.ttest_ind(values_1, values_2, equal_var=False)
            statistic, p_raw = float(result.statistic), float(result.pvalue)
            if not np.isfinite(p_raw):
                testable = False
        mean_1 = np.mean(values_1) if len(values_1) else np.nan
        mean_2 = np.mean(values_2) if len(values_2) else np.nan
        pair_rows.append(
            {
                "family_1": family_1,
                "family_2": family_2,
                "n_1": len(values_1),
                "n_2": len(values_2),
                "mean_log2FC_1": mean_1,
                "mean_log2FC_2": mean_2,
                "mean_difference_1_minus_2": mean_1 - mean_2,
                "welch_t_statistic": statistic,
                "p_raw": p_raw,
                "testable": testable,
            }
        )

    pairs = pd.DataFrame(pair_rows)
    valid = pairs["testable"] & pairs["p_raw"].notna()
    pairs["p_adj_BH"] = np.nan
    pairs.loc[valid, "p_adj_BH"] = stats.false_discovery_control(
        pairs.loc[valid, "p_raw"].to_numpy(), method="bh"
    )
    pairs["significant_BH_0.05"] = pd.array(
        pairs["p_adj_BH"] < 0.05, dtype="boolean"
    )
    pairs.loc[~valid, "significant_BH_0.05"] = pd.NA
    pairs["significance"] = pairs["p_adj_BH"].map(sig_label)
    pairs = pairs.sort_values(
        ["p_adj_BH", "family_1", "family_2"], na_position="last"
    ).reset_index(drop=True)
    return summary, pairs, float(omnibus_f), float(omnibus_p)


def matrices(
    summary: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    ordered = summary.sort_values(
        "mean_log2FC", ascending=True, na_position="first"
    )[FAMILY].tolist()
    means = summary.set_index(FAMILY)["mean_log2FC"]
    difference = pd.DataFrame(np.nan, index=ordered, columns=ordered)
    p_adjusted = pd.DataFrame(np.nan, index=ordered, columns=ordered)
    for family_1 in ordered:
        for family_2 in ordered:
            difference.loc[family_1, family_2] = means[family_1] - means[family_2]
    np.fill_diagonal(p_adjusted.values, 1.0)
    for row in pairs.itertuples(index=False):
        p_adjusted.loc[row.family_1, row.family_2] = row.p_adj_BH
        p_adjusted.loc[row.family_2, row.family_1] = row.p_adj_BH
    return ordered, difference, p_adjusted


def save_excel(
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    omnibus_f: float,
    omnibus_p: float,
    output_path: Path,
) -> None:
    ordered, difference, p_adjusted = matrices(summary, pairs)
    methodology = pd.DataFrame(
        {
            "item": [
                "Response variable",
                "Grouping variable",
                "Omnibus test",
                "Omnibus F",
                "Omnibus p",
                "Pairwise test",
                "Multiple-testing correction",
                "Correction scope",
                "Significance threshold",
                "Minimum sample size",
                "NE",
            ],
            "value": [
                LOG2FC,
                FAMILY,
                "One-way ANOVA",
                omnibus_f,
                omnibus_p,
                "Two-sided independent Welch t-test",
                "Benjamini-Hochberg FDR",
                "All valid pairwise family comparisons",
                "Adjusted p < 0.05",
                "At least 2 finite log2FC values in each family",
                "Not evaluable",
            ],
        }
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="family_summary", index=False)
        pairs.to_excel(writer, sheet_name="pairwise_welch", index=False)
        difference.to_excel(writer, sheet_name="mean_difference_matrix")
        p_adjusted.to_excel(writer, sheet_name="p_adj_BH_matrix")
        methodology.to_excel(writer, sheet_name="methodology", index=False)
        for sheet_name in ("family_summary", "pairwise_welch"):
            sheet = writer.book[sheet_name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
        writer.book["family_summary"].column_dimensions["A"].width = 46
        pair_sheet = writer.book["pairwise_welch"]
        pair_sheet.column_dimensions["A"].width = 46
        pair_sheet.column_dimensions["B"].width = 46
        for column in "CDEFGHIJKLM":
            pair_sheet.column_dimensions[column].width = 20


def save_heatmap(summary: pd.DataFrame, pairs: pd.DataFrame, output_path: Path) -> None:
    ordered, difference, p_adjusted = matrices(summary, pairs)
    values = difference.to_numpy(float)
    finite = values[np.isfinite(values)]
    limit = max(float(np.max(np.abs(finite))), 0.1)

    # Show each comparison once in the lower triangle.
    upper_triangle = np.triu(np.ones_like(values, dtype=bool), k=0)
    plot_values = np.ma.array(values, mask=upper_triangle | ~np.isfinite(values))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    figure, axis = plt.subplots(figsize=(30, 28))
    image = axis.imshow(
        plot_values,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="equal",
        interpolation="nearest",
    )
    locations = np.arange(len(ordered))
    axis.set_xticks(locations, ordered, rotation=90, fontsize=5)
    axis.set_yticks(locations, ordered, fontsize=5)
    axis.tick_params(length=0)
    axis.set_xlabel("Family 2", fontsize=11)
    axis.set_ylabel("Family 1", fontsize=11)
    axis.set_title(
        "Pairwise differences in mean log2FC between GPCR receptor families",
        fontsize=16,
        pad=18,
    )

    p_values = p_adjusted.to_numpy(float)
    for row in range(len(ordered)):
        for column in range(row):
            p_value = p_values[row, column]
            if np.isfinite(p_value) and p_value < 0.05:
                axis.plot(column, row, marker="o", markersize=2.2, color="black")
            elif not np.isfinite(p_value):
                axis.plot(column, row, marker="x", markersize=1.8, color="#777777")

    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.set_label("Mean log2FC (Family 1 − Family 2)", fontsize=10)
    figure.text(
        0.5,
        0.01,
        "Black dot: significant after global Benjamini-Hochberg correction (p<0.05). "
        "No dot: not significant. Grey ×: not evaluable. Welch independent two-sample tests.",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, pairs, omnibus_f, omnibus_p = analyse(args.input)
    excel_path = args.output_dir / "family_paired_analysis.xlsx"
    heatmap_path = args.output_dir / "family_paired_heatmap.png"
    save_excel(summary, pairs, omnibus_f, omnibus_p, excel_path)
    save_heatmap(summary, pairs, heatmap_path)

    valid = pairs["p_adj_BH"].notna()
    print(f"Families: {len(summary)}")
    print(f"Families with n>=2: {(summary['n_log2FC'] >= 2).sum()}")
    print(f"Valid pairwise comparisons: {valid.sum()}")
    print(f"Significant pairwise comparisons after BH: {(pairs['p_adj_BH'] < 0.05).sum()}")
    print(f"One-way ANOVA: F={omnibus_f:.6g}, p={omnibus_p:.6g}")
    print(excel_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
