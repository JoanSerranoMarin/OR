#!/usr/bin/env python3
"""Plot family means as bars with every individual GPCR value overlaid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY = "Receptor family"
METRICS = {
    "log2(active/inactive)": {
        "slug": "log2fc",
        "label": "log2(Active/Inactive)",
        "color": "#7b3294",
        "zero_line": True,
    },
    "Active": {
        "slug": "active",
        "label": "Active volume",
        "color": "#d95f02",
        "zero_line": False,
    },
    "Inactive": {
        "slug": "inactive",
        "label": "Inactive volume",
        "color": "#1b9e77",
        "zero_line": False,
    },
}

TARGETS = {
    "or2v1": {"color": "#d62728", "marker": "*", "label": "or2v1"},
    "o51e2": {"color": "#1565c0", "marker": "D", "label": "o51e2"},
}


def plot_metric(data: pd.DataFrame, metric: str, config: dict, output: Path) -> None:
    means = data.groupby(FAMILY, sort=True)[metric].mean().sort_values()
    families = means.index.tolist()
    y_positions = np.arange(len(families))
    height = max(14.0, 0.30 * len(families))
    rng = np.random.default_rng(20260629)

    figure, axis = plt.subplots(figsize=(13, height))
    axis.barh(
        y_positions,
        means.to_numpy(),
        height=0.66,
        color=config["color"],
        alpha=0.58,
        edgecolor="black",
        linewidth=0.35,
        label="Family mean",
        zorder=1,
    )

    for y, family in enumerate(families):
        values = data.loc[data[FAMILY] == family, metric].dropna().to_numpy(float)
        jitter = rng.uniform(-0.22, 0.22, size=len(values))
        axis.scatter(
            values,
            np.full(len(values), y) + jitter,
            s=14,
            facecolor="white",
            edgecolor="black",
            linewidth=0.55,
            alpha=0.82,
            zorder=3,
        )

    family_positions = {family: y for y, family in enumerate(families)}
    normalized_names = data["GPCR_name"].astype(str).str.casefold()
    for target_name, target_config in TARGETS.items():
        target_rows = data.loc[normalized_names == target_name]
        for _, target in target_rows.iterrows():
            value = target[metric]
            family = target[FAMILY]
            if not np.isfinite(value) or family not in family_positions:
                continue
            y = family_positions[family]
            axis.scatter(
                [value],
                [y],
                s=115,
                marker=target_config["marker"],
                facecolor=target_config["color"],
                edgecolor="black",
                linewidth=0.8,
                label=target_config["label"],
                zorder=6,
            )
            axis.annotate(
                target_config["label"],
                xy=(value, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=target_config["color"],
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5},
                zorder=7,
            )

    if config["zero_line"]:
        axis.axvline(0, color="#333333", linestyle="--", linewidth=1.0, zorder=2)
    display_families = [
        family.replace("<sub>", " ").replace("</sub>", "") for family in families
    ]
    axis.set_yticks(y_positions, display_families, fontsize=8)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel(config["label"], fontsize=11)
    axis.set_ylabel("Receptor family", fontsize=11)
    axis.set_title(
        f"{config['label']} by GPCR receptor family: mean and individual receptors",
        fontsize=14,
        pad=14,
    )
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.margins(y=0.005)
    axis.legend(loc="lower right", frameon=True, title="Highlighted GPCRs")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    figure.text(
        0.5,
        0.004,
        "Bars show family means; circles show individual GPCR values.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.input)
    missing = {"GPCR_name", FAMILY, *METRICS}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data[FAMILY] = data[FAMILY].astype("string").str.strip()
    data = data.dropna(subset=[FAMILY])

    for metric, config in METRICS.items():
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data[metric] = data[metric].replace([np.inf, -np.inf], np.nan)
        output = args.output_dir / f"family_{config['slug']}_mean_individual_points.png"
        plot_metric(data, metric, config, output)
        print(f"{metric}: {data[metric].notna().sum()} values, {data.groupby(FAMILY)[metric].mean().notna().sum()} families")
        print(output)


if __name__ == "__main__":
    main()
