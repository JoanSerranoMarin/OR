#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figura 6 paneles A–F para ionic locks/DRY en GPCR,
con outliers (>2 SD) eliminados, p-values ajustados por Benjamini–Hochberg
y export a Excel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================
# Parámetros de entrada
# ==========================

CSV_PATH = "gpcr_distances_annot_loose.csv"
FIGURE_PATH = "figure_ionic_locks_DRY_BH.png"
EXCEL_PATH = "gpcr_stats_results_BH.xlsx"

CLASSES_TO_PLOT = [
    "Class A (Rhodopsin)",
    "Class O1 (fish-like odorant)",
    "Class O2 (tetrapod specific odorant)",
]
CLASS_LABELS_SHORT = ["Class A", "Class O1", "Class O2"]

# Y-labels con superíndices usando mathtext
METRICS_INFO = [
    (
        "ionic_lock_R3.50_E_D6.30",
        r"traditional ionic lock R$^{3.50}$–E/D$^{6.30}$",
        "A",
        "D",
    ),
    (
        "alt_lock_min_dist",
        r"alternative ionic lock D$^{3.49}$–K$^{6.30}$",
        "B",
        "E",
    ),
    (
        "dry_min_dist",
        r"DRY distance D$^{3.49}$–R$^{3.50}$",
        "C",
        "F",
    ),
]

# Colores: barras claras, puntos oscuros
BAR_STATE_COLORS = {"inactive": "#dcdcdc", "active": "#f6d3a0"}  # claro
POINT_STATE_COLORS = {"inactive": "#555555", "active": "#a45a15"}  # más oscuro
BAR_BLUE = "#b3d3e8"
POINT_BLUE = "#1f77b4"


# ----------------------------------------------------------------------
# Outliers > 2 SD dentro de cada grupo (métrica × clase × estado)
# ----------------------------------------------------------------------
def remove_outliers_two_sd(df, metric):
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for (cls, state), group in df.groupby(["gpcr_class", "state"]):
        vals = group[metric].dropna()
        if len(vals) < 2:
            continue
        mean = vals.mean()
        std = vals.std(ddof=1)
        if std == 0 or np.isnan(std):
            continue
        group_mask = (group[metric] - mean).abs() <= 2 * std
        group_mask = group_mask.fillna(True)
        mask.loc[group.index] = group_mask

    return df[mask]


# ----------------------------------------------------------------------
# Benjamini–Hochberg FDR
# ----------------------------------------------------------------------
def benjamini_hochberg(pvals):
    """
    pvals: array-like (sin NaNs).
    Devuelve array de p ajustados BH en el mismo orden.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    ranks = np.arange(1, n + 1)

    bh = ranked_p * n / ranks
    # acumulado desde el final para asegurar monotonicidad
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    p_adj = np.empty_like(bh)
    p_adj[order] = bh
    return p_adj


# ----------------------------------------------------------------------
# Línea de significancia (sin brackets)
# ----------------------------------------------------------------------
def add_sig_line(ax, x1, x2, y, dy, p_adj, fontsize=10):
    """
    Dibuja línea horizontal de significancia sólo si p_adj < 0.05.
    x1, x2: posiciones en X
    y: altura de la línea
    dy: desplazamiento vertical del texto respecto a y
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
    ax.text((x1 + x2) / 2.0, y + dy, text, ha="center", va="bottom", fontsize=fontsize)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Cargar datos
    df = pd.read_csv(CSV_PATH)
    df["receptor_id"] = df["file"].str.extract(r"(.*)_all_")[0]
    df = df[df["gpcr_class"].isin(CLASSES_TO_PLOT)].copy()

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=False)

    state_stats = []   # activo vs inactivo dentro de cada clase
    class_stats = []   # comparaciones entre clases de log2FC

    for col, (metric, ylabel, letter_top, letter_bottom) in enumerate(METRICS_INFO):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        # -------------------------------------------------
        # Subconjunto para esta métrica + filtrado outliers
        # -------------------------------------------------
        tmp = df[["receptor_id", "gpcr_class", "state", metric]].copy()
        tmp = tmp.dropna(subset=[metric])
        tmp = remove_outliers_two_sd(tmp, metric)

        wide = tmp.pivot_table(
            index=["receptor_id", "gpcr_class"],
            columns="state",
            values=metric,
        )

        # ==========================
        # Panel superior A–C
        # ==========================
        sns.barplot(
            data=tmp,
            x="gpcr_class",
            y=metric,
            hue="state",
            order=CLASSES_TO_PLOT,
            hue_order=["inactive", "active"],
            palette=[BAR_STATE_COLORS["inactive"], BAR_STATE_COLORS["active"]],
            ci=None,
            dodge=True,
            ax=ax_top,
            alpha=0.85,  # algo transparente, pero no demasiado
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
            alpha=0.50,  # casi opacos
            linewidth=0,
            size=5,
            ax=ax_top,
        )

        # Leyenda bien visible sólo en panel A
        handles, labels = ax_top.get_legend_handles_labels()
        if col == 0:
            ax_top.legend(
                handles[:2],
                labels[:2],
                title="",
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=2,
            )
        else:
            ax_top.get_legend().remove()

        ax_top.set_xlabel("")
        ax_top.set_ylabel(ylabel)
        ax_top.set_xticklabels(CLASS_LABELS_SHORT)
        ax_top.set_ylim(0, 30)  # eje Y fijo 0–30
        ax_top.text(
            -0.2, 1.05, f"{letter_top})",
            transform=ax_top.transAxes, fontweight="bold", fontsize=14
        )

        # --- Paired t-test activo vs inactivo dentro de cada clase + BH
        xticks = ax_top.get_xticks()
        delta = 0.18
        top_tests = []

        for i, cls in enumerate(CLASSES_TO_PLOT):
            mask_cls = wide.index.get_level_values("gpcr_class") == cls
            sub_w = wide[mask_cls]

            if sub_w.empty or "active" not in sub_w.columns or "inactive" not in sub_w.columns:
                continue

            vals_act = sub_w["active"]
            vals_inact = sub_w["inactive"]
            mask = (~vals_act.isna()) & (~vals_inact.isna())
            vals_act = vals_act[mask]
            vals_inact = vals_inact[mask]

            if len(vals_act) <= 1:
                continue

            t_stat, p_raw = stats.ttest_rel(vals_act, vals_inact)

            top_tests.append(
                {
                    "metric": metric,
                    "class": cls,
                    "index": i,
                    "n": len(vals_act),
                    "mean_active": float(vals_act.mean()),
                    "mean_inactive": float(vals_inact.mean()),
                    "p_raw": float(p_raw),
                }
            )

        # Ajuste BH dentro de esta métrica (todas las clases)
        if top_tests:
            raw_p = [t["p_raw"] for t in top_tests]
            adj_p = benjamini_hochberg(raw_p)
            for t, p_adj in zip(top_tests, adj_p):
                # Guardar en Excel
                state_stats.append(
                    {
                        "metric": t["metric"],
                        "class": t["class"],
                        "test": "paired t-test",
                        "n": t["n"],
                        "mean_active": t["mean_active"],
                        "mean_inactive": t["mean_inactive"],
                        "p_raw": t["p_raw"],
                        "p_adj_BH": float(p_adj),
                    }
                )

                # Línea de significancia, usando p ajustado
                i = t["index"]
                x_center = xticks[i]
                x1 = x_center - delta
                x2 = x_center + delta
                # altura: max de las barras de esa clase + margen
                y_class = tmp.loc[tmp["gpcr_class"] == t["class"], metric].max()
                y_line = y_class + 0.6  # cerca de la barra
                add_sig_line(ax_top, x1, x2, y_line, dy=0.3, p_adj=p_adj, fontsize=10)

        # ==========================
        # Panel inferior D–F (log2FC)
        # ==========================
        if "active" in wide.columns and "inactive" in wide.columns:
            log2fc = np.log2(wide["active"] / wide["inactive"])
        else:
            log2fc = pd.Series(dtype=float)

        log_df_m = log2fc.rename("log2FC").reset_index().dropna(subset=["log2FC"])

        sns.barplot(
            data=log_df_m,
            x="gpcr_class",
            y="log2FC",
            order=CLASSES_TO_PLOT,
            color=BAR_BLUE,
            ci=None,
            ax=ax_bottom,
            alpha=0.85,
            edgecolor=None,
        )

        sns.stripplot(
            data=log_df_m,
            x="gpcr_class",
            y="log2FC",
            order=CLASSES_TO_PLOT,
            color=POINT_BLUE,
            alpha=0.50,
            size=5,
            ax=ax_bottom,
        )

        ax_bottom.axhline(0, ls="--", c="black", lw=1)
        ax_bottom.set_xlabel("")
        ax_bottom.set_ylabel("log2(FC) active / inactive")
        ax_bottom.set_xticklabels(CLASS_LABELS_SHORT)
        ax_bottom.set_ylim(-1, 3)  # eje Y fijo -1–3
        ax_bottom.text(
            -0.2, 1.05, f"{letter_bottom})",
            transform=ax_bottom.transAxes, fontweight="bold", fontsize=14
        )

        # Comparaciones entre clases (Welch t-test sobre log2FC) + BH
        bottom_tests = []
        for i1, i2 in [(0, 1), (0, 2), (1, 2)]:
            c1 = CLASSES_TO_PLOT[i1]
            c2 = CLASSES_TO_PLOT[i2]
            vals1 = log_df_m[log_df_m["gpcr_class"] == c1]["log2FC"]
            vals2 = log_df_m[log_df_m["gpcr_class"] == c2]["log2FC"]

            if len(vals1) <= 1 or len(vals2) <= 1:
                continue

            t_stat, p_raw = stats.ttest_ind(vals1, vals2, equal_var=False)

            bottom_tests.append(
                {
                    "metric": metric,
                    "comparison": f"{c1} vs {c2}",
                    "i1": i1,
                    "i2": i2,
                    "n1": len(vals1),
                    "n2": len(vals2),
                    "mean1": float(vals1.mean()),
                    "mean2": float(vals2.mean()),
                    "p_raw": float(p_raw),
                }
            )

        if bottom_tests:
            raw_p = [t["p_raw"] for t in bottom_tests]
            adj_p = benjamini_hochberg(raw_p)

            # un pelín más separadas pero dentro del rango -1..3
            baseline = 1.9   # un poco por debajo del techo
            step = 0.35      # espacio entre líneas apiladas
            offset = 0.0

            for t, p_adj in zip(bottom_tests, adj_p):
                # guardar
                class_stats.append(
                    {
                        "metric": t["metric"],
                        "comparison": t["comparison"],
                        "test": "Welch t-test",
                        "n1": t["n1"],
                        "n2": t["n2"],
                        "mean1": t["mean1"],
                        "mean2": t["mean2"],
                        "p_raw": t["p_raw"],
                        "p_adj_BH": float(p_adj),
                    }
                )

                if p_adj < 0.05:
                    y_line = baseline + offset
                    add_sig_line(
                        ax_bottom,
                        t["i1"],
                        t["i2"],
                        y=y_line,
                        dy=0.12,   # texto un poco más separado de la línea
                        p_adj=p_adj,
                        fontsize=9,
                    )
                    offset += step

    # Layout + guardado figura
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.show()

    # ==========================
    # Excel con estadísticas
    # ==========================
    state_df = pd.DataFrame(state_stats)
    class_df = pd.DataFrame(class_stats)

    with pd.ExcelWriter(EXCEL_PATH) as writer:
        state_df.to_excel(
            writer, sheet_name="state_active_vs_inactive_BH", index=False
        )
        class_df.to_excel(
            writer, sheet_name="log2FC_between_classes_BH", index=False
        )


if __name__ == "__main__":
    main()

