#!/usr/bin/env python3
"""Compare Active and Inactive volumes separately across GPCR families."""

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
STATES = ("Active", "Inactive")


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


def analyse_state(
    data: pd.DataFrame, state: str
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    groups = {
        str(name): group[state].dropna().to_numpy(float)
        for name, group in data.groupby(FAMILY, sort=True)
    }

    summary_rows = []
    for family, values in groups.items():
        n = len(values)
        summary_rows.append(
            {
                FAMILY: family,
                "n": n,
                f"mean_{state.lower()}": np.mean(values) if n else np.nan,
                f"median_{state.lower()}": np.median(values) if n else np.nan,
                f"sd_{state.lower()}": np.std(values, ddof=1) if n >= 2 else np.nan,
                f"sem_{state.lower()}": stats.sem(values) if n >= 2 else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(FAMILY).reset_index(drop=True)

    valid_groups = [values for values in groups.values() if len(values) >= 2]
    if len(valid_groups) >= 2:
        omnibus_f, omnibus_p = stats.f_oneway(*valid_groups)
    else:
        omnibus_f, omnibus_p = np.nan, np.nan

    rows = []
    for family_1, family_2 in combinations(groups, 2):
        values_1, values_2 = groups[family_1], groups[family_2]
        mean_1 = np.mean(values_1) if len(values_1) else np.nan
        mean_2 = np.mean(values_2) if len(values_2) else np.nan
        statistic, p_raw = np.nan, np.nan
        testable = len(values_1) >= 2 and len(values_2) >= 2
        if testable:
            result = stats.ttest_ind(values_1, values_2, equal_var=False)
            statistic, p_raw = float(result.statistic), float(result.pvalue)
            if not np.isfinite(p_raw):
                testable = False
        rows.append(
            {
                "family_1": family_1,
                "family_2": family_2,
                "n_1": len(values_1),
                "n_2": len(values_2),
                f"mean_{state.lower()}_1": mean_1,
                f"mean_{state.lower()}_2": mean_2,
                "mean_difference_1_minus_2": mean_1 - mean_2,
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
    pairs = pairs.sort_values(
        ["p_adj_BH", "family_1", "family_2"], na_position="last"
    ).reset_index(drop=True)
    return summary, pairs, float(omnibus_f), float(omnibus_p)


def matrices(
    summary: pd.DataFrame, pairs: pd.DataFrame, state: str
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    mean_column = f"mean_{state.lower()}"
    ordered = summary.sort_values(
        mean_column, ascending=True, na_position="first"
    )[FAMILY].tolist()
    means = summary.set_index(FAMILY)[mean_column]
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
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, float, float]],
    output_path: Path,
) -> None:
    methodology_rows = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for state in STATES:
            summary, pairs, omnibus_f, omnibus_p = results[state]
            _, difference, p_adjusted = matrices(summary, pairs, state)
            prefix = state.lower()
            summary.to_excel(writer, sheet_name=f"{prefix}_family_summary", index=False)
            pairs.to_excel(writer, sheet_name=f"{prefix}_pairwise_welch", index=False)
            difference.to_excel(writer, sheet_name=f"{prefix}_mean_diff")
            p_adjusted.to_excel(writer, sheet_name=f"{prefix}_p_adj_BH")
            methodology_rows.extend(
                [
                    {"state": state, "item": "Omnibus test", "value": "One-way ANOVA"},
                    {"state": state, "item": "Omnibus F", "value": omnibus_f},
                    {"state": state, "item": "Omnibus p", "value": omnibus_p},
                ]
            )

            summary_sheet = writer.book[f"{prefix}_family_summary"]
            summary_sheet.freeze_panes = "A2"
            summary_sheet.auto_filter.ref = summary_sheet.dimensions
            summary_sheet.column_dimensions["A"].width = 46
            pair_sheet = writer.book[f"{prefix}_pairwise_welch"]
            pair_sheet.freeze_panes = "A2"
            pair_sheet.auto_filter.ref = pair_sheet.dimensions
            pair_sheet.column_dimensions["A"].width = 46
            pair_sheet.column_dimensions["B"].width = 46
            for column in "CDEFGHIJKLM":
                pair_sheet.column_dimensions[column].width = 20

        methodology_rows.extend(
            [
                {"state": "Both", "item": "Grouping variable", "value": FAMILY},
                {
                    "state": "Both",
                    "item": "Pairwise test",
                    "value": "Two-sided independent Welch t-test",
                },
                {
                    "state": "Both",
                    "item": "Multiple-testing correction",
                    "value": "Benjamini-Hochberg FDR, separately within each state",
                },
                {
                    "state": "Both",
                    "item": "Significance threshold",
                    "value": "Adjusted p < 0.05",
                },
                {
                    "state": "Both",
                    "item": "Minimum sample size",
                    "value": "At least 2 finite values in each family",
                },
            ]
        )
        pd.DataFrame(methodology_rows).to_excel(
            writer, sheet_name="methodology", index=False
        )


def save_heatmap(
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    state: str,
    output_path: Path,
) -> None:
    ordered, difference, p_adjusted = matrices(summary, pairs, state)
    values = difference.to_numpy(float)
    finite = values[np.isfinite(values)]
    limit = max(float(np.max(np.abs(finite))), 0.1)
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
    axis.set_xlabel("Family 2")
    axis.set_ylabel("Family 1")
    axis.set_title(
        f"Pairwise differences in mean {state} volume between GPCR families",
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
    colorbar.set_label(f"Mean {state} volume (Family 1 − Family 2)")
    figure.text(
        0.5,
        0.01,
        "Black dot: significant after Benjamini-Hochberg correction (p<0.05). "
        "No dot: not significant. Grey ×: not evaluable. Welch independent tests.",
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

    data = pd.read_excel(args.input)
    missing = {FAMILY, *STATES}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data[FAMILY] = data[FAMILY].astype("string").str.strip()
    data = data.dropna(subset=[FAMILY])
    for state in STATES:
        data[state] = pd.to_numeric(data[state], errors="coerce")
        data[state] = data[state].replace([np.inf, -np.inf], np.nan)

    results = {state: analyse_state(data, state) for state in STATES}
    excel_path = args.output_dir / "family_state_pairwise_analysis.xlsx"
    save_excel(results, excel_path)
    for state in STATES:
        summary, pairs, omnibus_f, omnibus_p = results[state]
        heatmap_path = args.output_dir / f"family_{state.lower()}_pairwise_heatmap.png"
        save_heatmap(summary, pairs, state, heatmap_path)
        print(
            f"{state}: families={len(summary)}, valid_pairs={pairs['p_adj_BH'].notna().sum()}, "
            f"significant_pairs={(pairs['p_adj_BH'] < 0.05).sum()}, "
            f"ANOVA_F={omnibus_f:.6g}, ANOVA_p={omnibus_p:.6g}"
        )
    print(excel_path)


if __name__ == "__main__":
    main()
