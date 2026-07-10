#!/usr/bin/env python3
"""Compare Active, Inactive and log2FC across odorant receptor families only."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


FAMILY = "Receptor family"
METRICS = {
    "Active": {"slug": "active", "title": "Active volume"},
    "Inactive": {"slug": "inactive", "title": "Inactive volume"},
    "log2(active/inactive)": {"slug": "log2fc", "title": "log2(Active/Inactive)"},
}


def family_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 10**9


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


def analyse_metric(
    data: pd.DataFrame, metric: str
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    groups = {
        str(name): group[metric].dropna().to_numpy(float)
        for name, group in data.groupby(FAMILY)
    }
    ordered = sorted(groups, key=family_number)
    summary_rows = []
    for family in ordered:
        values = groups[family]
        summary_rows.append(
            {
                FAMILY: family,
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "sd": np.std(values, ddof=1) if len(values) >= 2 else np.nan,
                "sem": stats.sem(values) if len(values) >= 2 else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    valid_groups = [groups[name] for name in ordered if len(groups[name]) >= 2]
    omnibus_f, omnibus_p = stats.f_oneway(*valid_groups)

    rows = []
    for family_1, family_2 in combinations(ordered, 2):
        values_1, values_2 = groups[family_1], groups[family_2]
        testable = len(values_1) >= 2 and len(values_2) >= 2
        statistic, p_raw = np.nan, np.nan
        if testable:
            test = stats.ttest_ind(values_1, values_2, equal_var=False)
            statistic, p_raw = float(test.statistic), float(test.pvalue)
            if not np.isfinite(p_raw):
                testable = False
        rows.append(
            {
                "family_1": family_1,
                "family_2": family_2,
                "n_1": len(values_1),
                "n_2": len(values_2),
                "mean_1": np.mean(values_1),
                "mean_2": np.mean(values_2),
                "mean_difference_1_minus_2": np.mean(values_1) - np.mean(values_2),
                "welch_t_statistic": statistic,
                "p_raw": p_raw,
                "testable": testable,
            }
        )
    pairs = pd.DataFrame(rows)
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
    pairs = pairs.sort_values(["p_adj_BH", "family_1", "family_2"]).reset_index(drop=True)
    return summary, pairs, float(omnibus_f), float(omnibus_p)


def matrices(
    summary: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    ordered = summary[FAMILY].tolist()
    means = summary.set_index(FAMILY)["mean"]
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


def save_excel(results: dict, output_path: Path) -> None:
    method_rows = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for metric, config in METRICS.items():
            summary, pairs, omnibus_f, omnibus_p = results[metric]
            _, difference, p_adjusted = matrices(summary, pairs)
            slug = config["slug"]
            summary.to_excel(writer, sheet_name=f"{slug}_summary", index=False)
            pairs.to_excel(writer, sheet_name=f"{slug}_pairwise", index=False)
            difference.to_excel(writer, sheet_name=f"{slug}_mean_diff")
            p_adjusted.to_excel(writer, sheet_name=f"{slug}_p_adj_BH")
            method_rows.extend(
                [
                    {"metric": metric, "item": "One-way ANOVA F", "value": omnibus_f},
                    {"metric": metric, "item": "One-way ANOVA p", "value": omnibus_p},
                    {
                        "metric": metric,
                        "item": "Significant pairwise comparisons",
                        "value": int((pairs["p_adj_BH"] < 0.05).sum()),
                    },
                ]
            )
            for sheet_name in (f"{slug}_summary", f"{slug}_pairwise"):
                sheet = writer.book[sheet_name]
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
            writer.book[f"{slug}_summary"].column_dimensions["A"].width = 24
            pair_sheet = writer.book[f"{slug}_pairwise"]
            pair_sheet.column_dimensions["A"].width = 24
            pair_sheet.column_dimensions["B"].width = 24

        method_rows.extend(
            [
                {
                    "metric": "All",
                    "item": "Filter",
                    "value": "Receptor family starts with 'Odorant family'",
                },
                {
                    "metric": "All",
                    "item": "Pairwise test",
                    "value": "Two-sided independent Welch t-test",
                },
                {
                    "metric": "All",
                    "item": "Correction",
                    "value": "Benjamini-Hochberg FDR separately for each metric",
                },
                {
                    "metric": "All",
                    "item": "Threshold",
                    "value": "Adjusted p < 0.05",
                },
            ]
        )
        pd.DataFrame(method_rows).to_excel(writer, sheet_name="methodology", index=False)


def save_heatmaps(results: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(ncols=3, figsize=(29, 10.5))
    for axis, (metric, config) in zip(axes, METRICS.items()):
        summary, pairs, _, _ = results[metric]
        ordered, difference, p_adjusted = matrices(summary, pairs)
        values = difference.to_numpy(float)
        limit = max(float(np.nanmax(np.abs(values))), 0.1)
        mask = np.triu(np.ones_like(values, dtype=bool), k=0)
        shown = np.ma.array(values, mask=mask)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        image = axis.imshow(
            shown,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            interpolation="nearest",
        )
        positions = np.arange(len(ordered))
        short_labels = [name.replace("Odorant family ", "OR family ") for name in ordered]
        axis.set_xticks(positions, short_labels, rotation=90, fontsize=7)
        axis.set_yticks(positions, short_labels, fontsize=7)
        axis.tick_params(length=0)
        axis.set_xlabel("Family 2")
        axis.set_ylabel("Family 1")
        axis.set_title(config["title"], fontsize=13)

        p_values = p_adjusted.to_numpy(float)
        for row in range(len(ordered)):
            for column in range(row):
                axis.text(
                    column,
                    row,
                    sig_label(p_values[row, column]),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="black",
                )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.025)
        colorbar.set_label("Mean difference (Family 1 − Family 2)", fontsize=8)

    figure.suptitle(
        "Pairwise comparisons among odorant receptor families",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Welch tests with separate Benjamini-Hochberg correction per metric. "
        "* adjusted p<0.05, ** <0.01, *** <0.001, ns = not significant.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.input)
    missing = {FAMILY, *METRICS}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[
        data[FAMILY].astype(str).str.startswith("Odorant family", na=False)
    ].copy()
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data[metric] = data[metric].replace([np.inf, -np.inf], np.nan)

    results = {metric: analyse_metric(data, metric) for metric in METRICS}
    excel_path = args.output_dir / "odorant_family_three_metric_analysis.xlsx"
    heatmap_path = args.output_dir / "odorant_family_significance_heatmaps.png"
    save_excel(results, excel_path)
    save_heatmaps(results, heatmap_path)
    for metric, (_, pairs, omnibus_f, omnibus_p) in results.items():
        print(
            f"{metric}: ANOVA F={omnibus_f:.6g}, p={omnibus_p:.6g}; "
            f"valid pairs={pairs['p_adj_BH'].notna().sum()}, "
            f"significant pairs={(pairs['p_adj_BH'] < 0.05).sum()}"
        )
    print(excel_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
