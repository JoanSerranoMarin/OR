#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots 2x2 por dataset (only_trad / only_alt / both) y por lock (ionic / alt / dry)
usando SOLO la tabla de medias por receptor del principio de summary_stats.

- Filtra a Class A (Rhodopsin) únicamente (sin O1/O2).
- NO elimina outliers.
- Paneles:
    A) sidechain N–O (active vs inactive)
    C) Cα–Cα (active vs inactive)
    B) log2FC sidechain (active/inactive)
    D) log2FC Cα–Cα (active/inactive)
- Paired t-test active vs inactive dentro de la clase (BH dentro de esa métrica).
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ==========================
# Clases a plotear (SOLO Class A)
# ==========================
CLASSES_TO_PLOT = ["Class A (Rhodopsin)"]
CLASS_LABELS_SHORT = ["Class A"]

# Colores
BAR_STATE_COLORS = {"inactive": "#dcdcdc", "active": "#f6d3a0"}
POINT_STATE_COLORS = {"inactive": "#555555", "active": "#a45a15"}
BAR_BLUE = "#b3d3e8"
POINT_BLUE = "#1f77b4"


# ==========================
# Locks a plotear
# ==========================
LOCK_CONFIGS = [
    {
        "lock_key": "ionic",
        "metric_sc": "ionic_sidechain_NO",
        "metric_ca": "ionic_CA_CA",
        "ylabel_sc": r"traditional ionic lock R$^{3.50}$–E/D$^{6.30}$ [Å]",
        "ylabel_ca": r"C$\alpha$–C$\alpha$ R$^{3.50}$–E/D$^{6.30}$ [Å]",
        "tag": "traditional_ionic_lock",
    },
    {
        "lock_key": "alt",
        "metric_sc": "alt_sidechain_NO",
        "metric_ca": "alt_CA_CA",
        "ylabel_sc": r"alternative ionic lock D$^{3.49}$–K$^{6.30}$ [Å]",
        "ylabel_ca": r"C$\alpha$–C$\alpha$ D$^{3.49}$–K$^{6.30}$ [Å]",
        "tag": "alternative_ionic_lock",
    },
    {
        "lock_key": "dry",
        "metric_sc": "dry_sidechain_NO",
        "metric_ca": "dry_CA_CA",
        "ylabel_sc": r"DRY lock D/E$^{3.49}$–R/K$^{3.50}$ N–O [Å]",
        "ylabel_ca": r"C$\alpha$–C$\alpha$ D/E$^{3.49}$–R/K$^{3.50}$ [Å]",
        "tag": "dry_lock",
    },
]


# ----------------------------------------------------------------------
# Benjamini–Hochberg FDR
# ----------------------------------------------------------------------
def benjamini_hochberg(pvals: List[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    ranks = np.arange(1, n + 1)

    bh = ranked_p * n / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    p_adj = np.empty_like(bh)
    p_adj[order] = bh
    return p_adj


# ----------------------------------------------------------------------
# Línea de significancia (sin brackets)
# ----------------------------------------------------------------------
def add_sig_line(ax, x1, x2, y, dy, p_adj, fontsize=10):
    if np.isnan(p_adj) or p_adj >= 0.05:
        return

    if p_adj < 0.001:
        text = "***"
    elif p_adj < 0.01:
        text = "**"
    else:
        text = "*"

    ax.plot([x1, x2], [y, y], lw=1, c="black")
    ax.text((x1 + x2) / 2.0, y + dy, text, ha="center", va="bottom", fontsize=fontsize)


# ----------------------------------------------------------------------
# Leer SOLO la tabla "means" desde summary_stats
# ----------------------------------------------------------------------
def read_means_from_summary_stats(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="summary_stats", engine="openpyxl")

    # Normaliza nombres esperados
    rename_map = {}
    if "class" in df.columns:
        rename_map["class"] = "gpcr_class"
    if "state_norm" in df.columns:
        rename_map["state_norm"] = "state"
    if rename_map:
        df = df.rename(columns=rename_map)

    needed_core = ["receptor_gpcrdb", "gpcr_class", "state"]
    for c in needed_core:
        if c not in df.columns:
            raise ValueError(
                f"[{excel_path}] No encuentro columna '{c}' en summary_stats."
            )

    # Quédate SOLO con filas del bloque de means
    df = df[df["receptor_gpcrdb"].notna() & df["gpcr_class"].notna() & df["state"].notna()].copy()

    # Normaliza estado a inactive/active
    df["state"] = df["state"].astype(str).str.strip().str.lower()
    df = df[df["state"].isin(["inactive", "active"])].copy()

    # Fuerza numéricos (evita dtype object en scipy)
    for col in [
        "ionic_sidechain_NO", "ionic_CA_CA",
        "alt_sidechain_NO", "alt_CA_CA",
        "dry_sidechain_NO", "dry_CA_CA",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ----------------------------------------------------------------------
# Plot 2x2 para un lock (A,C,B,D)
# ----------------------------------------------------------------------
def plot_2x2_lock(
    df_means: pd.DataFrame,
    dataset_name: str,
    lock_cfg: Dict,
    outdir: str,
    make_excel_stats: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    metric_sc = lock_cfg["metric_sc"]
    metric_ca = lock_cfg["metric_ca"]

    if metric_sc not in df_means.columns and metric_ca not in df_means.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = df_means.copy()
    df = df[df["gpcr_class"].isin(CLASSES_TO_PLOT)].copy()

    # ¿hay datos?
    has_any = False
    for m in [metric_sc, metric_ca]:
        if m in df.columns:
            arr = df[m].to_numpy(dtype=float, na_value=np.nan)
            if np.isfinite(arr).any():
                has_any = True
    if not has_any:
        return pd.DataFrame(), pd.DataFrame()

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axA, axC = axes[0, 0], axes[0, 1]
    axB, axD = axes[1, 0], axes[1, 1]

    state_stats_rows = []
    class_stats_rows = []

    def _plot_top(ax, metric, ylabel, panel_letter, show_legend):
        tmp = df[["receptor_gpcrdb", "gpcr_class", "state", metric]].copy()
        tmp = tmp.dropna(subset=[metric])

        wide = tmp.pivot_table(
            index=["receptor_gpcrdb", "gpcr_class"],
            columns="state",
            values=metric,
            aggfunc="mean",
        )

        sns.barplot(
            data=tmp,
            x="gpcr_class",
            y=metric,
            hue="state",
            order=CLASSES_TO_PLOT,
            hue_order=["inactive", "active"],
            palette=[BAR_STATE_COLORS["inactive"], BAR_STATE_COLORS["active"]],
            errorbar=None,
            dodge=True,
            ax=ax,
            alpha=0.85,
            edgecolor=None,
        )

        sns.stripplot(
            data=tmp,
            x="gpcr_class",
            y=metric,
            hue="state",
            order=CLASSES_TO_PLOT,
            hue_order=["inactive", "active"],
            palette=[POINT_STATE_COLORS["inactive"], POINT_STATE_COLORS["active"]],
            dodge=True,
            alpha=0.50,
            linewidth=0,
            size=5,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        if show_legend:
            ax.legend(
                handles[:2],
                labels[:2],
                title="",
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=2,
            )
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(CLASS_LABELS_SHORT)))
        ax.set_xticklabels(CLASS_LABELS_SHORT)
        ax.set_ylim(0, 30)
        ax.text(-0.18, 1.05, f"{panel_letter})", transform=ax.transAxes, fontweight="bold", fontsize=14)

        # Paired t-test dentro de cada clase + BH (aquí solo 1 clase)
        top_tests = []
        for i, cls in enumerate(CLASSES_TO_PLOT):
            sub_w = wide[wide.index.get_level_values("gpcr_class") == cls]
            if sub_w.empty or "active" not in sub_w.columns or "inactive" not in sub_w.columns:
                continue

            a = pd.to_numeric(sub_w["active"], errors="coerce")
            b = pd.to_numeric(sub_w["inactive"], errors="coerce")
            mask = (~a.isna()) & (~b.isna())
            a = a[mask].to_numpy(dtype=float)
            b = b[mask].to_numpy(dtype=float)

            n_pairs = int(len(a))
            if n_pairs <= 1:
                continue

            t_stat, p_raw = stats.ttest_rel(a, b)

            top_tests.append(
                {
                    "metric": metric,
                    "class": cls,
                    "index": i,
                    "n": n_pairs,
                    "mean_active": float(np.mean(a)),
                    "mean_inactive": float(np.mean(b)),
                    "p_raw": float(p_raw),
                }
            )

            ax.text(i, 0.5, f"n={n_pairs}", ha="center", va="bottom", fontsize=11)

        if top_tests:
            raw_p = [t["p_raw"] for t in top_tests]
            adj_p = benjamini_hochberg(raw_p)

            xticks = ax.get_xticks()
            delta = 0.18

            for t, p_adj in zip(top_tests, adj_p):
                state_stats_rows.append(
                    {
                        "dataset": dataset_name,
                        "lock": lock_cfg["lock_key"],
                        "metric": t["metric"],
                        "class": t["class"],
                        "test": "paired t-test",
                        "n_pairs": t["n"],
                        "mean_active": t["mean_active"],
                        "mean_inactive": t["mean_inactive"],
                        "p_raw": t["p_raw"],
                        "p_adj_BH": float(p_adj),
                    }
                )

                i = t["index"]
                x_center = xticks[i]
                x1 = x_center - delta
                x2 = x_center + delta
                y_class = tmp.loc[tmp["gpcr_class"] == t["class"], metric].max()
                y_line = float(y_class) + 0.6
                add_sig_line(ax, x1, x2, y_line, dy=0.3, p_adj=p_adj, fontsize=10)

        return wide

    def _plot_bottom(ax, wide, metric, panel_letter):
        if "active" not in wide.columns or "inactive" not in wide.columns:
            ax.axis("off")
            return

        log2fc = np.log2(wide["active"] / wide["inactive"])
        log_df = log2fc.rename("log2FC").reset_index().dropna(subset=["log2FC"]).copy()

        sns.barplot(
            data=log_df,
            x="gpcr_class",
            y="log2FC",
            order=CLASSES_TO_PLOT,
            color=BAR_BLUE,
            errorbar=None,
            ax=ax,
            alpha=0.85,
            edgecolor=None,
        )

        sns.stripplot(
            data=log_df,
            x="gpcr_class",
            y="log2FC",
            order=CLASSES_TO_PLOT,
            color=POINT_BLUE,
            alpha=0.50,
            size=5,
            ax=ax,
        )

        ax.axhline(0, ls="--", c="black", lw=1)
        ax.set_xlabel("")
        ax.set_ylabel("log2(FC) active / inactive")
        ax.set_xticks(range(len(CLASS_LABELS_SHORT)))
        ax.set_xticklabels(CLASS_LABELS_SHORT)
        ax.set_ylim(-1, 3)
        ax.text(-0.18, 1.05, f"{panel_letter})", transform=ax.transAxes, fontweight="bold", fontsize=14)

    # ---- A y C ----
    wide_sc = None
    wide_ca = None

    if metric_sc in df.columns:
        wide_sc = _plot_top(axA, metric_sc, lock_cfg["ylabel_sc"], "A", show_legend=True)
    else:
        axA.axis("off")

    if metric_ca in df.columns:
        wide_ca = _plot_top(axC, metric_ca, lock_cfg["ylabel_ca"], "C", show_legend=False)
    else:
        axC.axis("off")

    # ---- B y D ----
    if wide_sc is not None:
        _plot_bottom(axB, wide_sc, metric_sc, "B")
    else:
        axB.axis("off")

    if wide_ca is not None:
        _plot_bottom(axD, wide_ca, metric_ca, "D")
    else:
        axD.axis("off")

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)

    fig_path = os.path.join(outdir, f"{dataset_name}__{lock_cfg['tag']}__2x2.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    state_df = pd.DataFrame(state_stats_rows)
    class_df = pd.DataFrame(class_stats_rows)

    if make_excel_stats:
        xlsx_path = os.path.join(outdir, f"{dataset_name}__{lock_cfg['tag']}__stats_BH.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            state_df.to_excel(writer, sheet_name="state_active_vs_inactive_BH", index=False)
            class_df.to_excel(writer, sheet_name="log2FC_between_classes_BH", index=False)

    return state_df, class_df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-trad", default="only_trad_final.xlsx", help="Excel only traditional lock")
    ap.add_argument("--only-alt", default="only_alt_final.xlsx", help="Excel only alternative lock")
    ap.add_argument("--both", default="both_final.xlsx", help="Excel both locks")
    ap.add_argument("--outdir", default="figs_locks_classA", help="Directorio de salida")
    args = ap.parse_args()

    datasets = [
        ("only_trad", args.only_trad),
        ("only_alt", args.only_alt),
        ("both", args.both),
    ]

    for dataset_name, xlsx in datasets:
        if not os.path.isfile(xlsx):
            print(f"[WARN] No existe: {xlsx} (skip)")
            continue

        df_means = read_means_from_summary_stats(xlsx)

        for lock_cfg in LOCK_CONFIGS:
            plot_2x2_lock(
                df_means=df_means,
                dataset_name=dataset_name,
                lock_cfg=lock_cfg,
                outdir=args.outdir,
                make_excel_stats=True,
            )

    print(f"[OK] Figuras/Excel stats en: {args.outdir}")


if __name__ == "__main__":
    main()

