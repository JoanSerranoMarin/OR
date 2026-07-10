#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figura para la métrica DRY:

    - Distancia SC–SC (N–O mínima)  -> dry_min_dist
    - Distancia Cα–Cα               -> dry_CA_3.49_3.50
    - log2(FC) active / inactive    -> para cada una de las dos

Tests estadísticos (t-test pareado + Welch entre clases, con BH)
SOLO se aplican a la métrica SC–SC y su log2FC. Para la métrica Cα–Cα
y su log2FC NO se realizan tests ni se dibujan asteriscos.

Input:
    gpcr_distances_full_annot.csv   (mismas columnas que el script original)
    Necesario:
        - file
        - gpcr_class
        - state  (active / inactive)
        - dry_min_dist
        - dry_CA_3.49_3.50

Output:
    figure_dry_lock.png
    dry_lock_stats_BH.xlsx
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================
# Parámetros de entrada
# ==========================

CSV_PATH = "gpcr_distances_annot_dry.csv"

# Métricas DRY
DRY_SC_METRIC = "dry_min_dist"          # SC–SC (N–O mínima)
DRY_CA_METRIC = "dry_CA_3.49_3.50"      # Cα–Cα

FIG_DRY = "figure_dry_lock.png"
EXCEL_PATH = "dry_lock_stats_BH.xlsx"

# Clases a mostrar (nombres largos en el CSV)
CLASSES_TO_PLOT = [
    "Class A (Rhodopsin)",
    "Class O1 (fish-like odorant)",
    "Class O2 (tetrapod specific odorant)",
]
# Etiquetas cortas en el eje X
CLASS_LABELS_SHORT = ["Class A", "Class O1", "Class O2"]

# Colores (igual estilo que el script de ionic lock)
BAR_STATE_COLORS = {"inactive": "#dcdcdc", "active": "#f6d3a0"}    # barras
POINT_STATE_COLORS = {"inactive": "#555555", "active": "#a45a15"}  # puntos
BAR_BLUE = "#b3d3e8"
POINT_BLUE = "#1f77b4"


# ----------------------------------------------------------------------
# Benjamini–Hochberg FDR
# ----------------------------------------------------------------------
def benjamini_hochberg(pvals):
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
# Línea de significancia
# ----------------------------------------------------------------------
def add_sig_line(ax, x1, x2, y, dy, p_adj, fontsize=10):
    """Dibuja una línea de significancia si p_adj < 0.05."""
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
# Filtrado conjunto de outliers para un grupo de métricas
# ----------------------------------------------------------------------
def joint_filter_by_metrics(df, metric_list):
    """
    Devuelve un df filtrado donde:
    - se eliminan outliers (>2 SD) por métrica y por grupo (clase, estado),
      y se descartan filas que sean outlier en CUALQUIERA de las métricas
      de metric_list.
    - se eliminan filas con NaN en alguna métrica de metric_list.
    """
    df = df.copy()
    if df.empty:
        return df

    mask_all = pd.Series(True, index=df.index)

    for metric in metric_list:
        metric_mask = pd.Series(True, index=df.index)

        for (cls, state), group in df.groupby(["gpcr_class", "state"]):
            vals = group[metric].dropna()
            if len(vals) < 2:
                continue
            mean = vals.mean()
            std = vals.std(ddof=1)
            if std == 0 or np.isnan(std):
                continue

            cond = ((group[metric] - mean).abs() <= 2 * std) | group[metric].isna()
            metric_mask.loc[group.index] = cond

        mask_all &= metric_mask

    df = df[mask_all]
    df = df.dropna(subset=metric_list)
    return df


# ----------------------------------------------------------------------
# Paneles superior/inferior para una lista de métricas (p.ej. DRY SC y DRY CA)
# metrics_info: lista de tuplas
#   (metric, ylabel, letter_top, letter_bottom, group_id, do_stats)
# group_id se usa para compartir filtrado (SC–SC/Cα–Cα) dentro de cada pareja
# do_stats indica si se hacen tests estadísticos para esa métrica
# ----------------------------------------------------------------------
def make_panels_for_subset(
    df_sub,
    metrics_info,
    fig_path,
    subset_label,
    state_stats,
    class_stats,
    summary_state,
    summary_log2fc,
    fig_kind="rect",
):
    df_sub = df_sub.copy()

    # Nos quedamos sólo con las clases presentes en este subset
    classes_present = [
        cls for cls in CLASSES_TO_PLOT if cls in df_sub["gpcr_class"].unique()
    ]
    if not classes_present:
        print(f"[AVISO] Subconjunto '{subset_label}' vacío (sin clases presentes).")
        return

    class_to_short = dict(zip(CLASSES_TO_PLOT, CLASS_LABELS_SHORT))
    short_labels_present = [class_to_short[c] for c in classes_present]

    df_sub = df_sub[df_sub["gpcr_class"].isin(classes_present)]
    if df_sub.empty:
        print(f"[AVISO] Subconjunto '{subset_label}' vacío, no hay figura.")
        return

    # Construimos grupos de métricas para filtrado conjunto
    group_to_metrics = {}
    for metric, _, _, _, group_id, _ in metrics_info:
        group_to_metrics.setdefault(group_id, []).append(metric)

    # Filtrado conjunto por grupo
    group_to_df = {}
    for group_id, mlist in group_to_metrics.items():
        group_to_df[group_id] = joint_filter_by_metrics(df_sub, mlist)

    n_metrics = len(metrics_info)

    # Tamaño de figura
    if fig_kind == "square":
        fig_w, fig_h = 14, 8
    else:
        fig_w, fig_h = 4 * n_metrics + 2, 6

    fig, axes = plt.subplots(2, n_metrics, figsize=(fig_w, fig_h), sharey=False)

    # Normalizar shape cuando n_metrics == 1
    if n_metrics == 1:
        ax_top, ax_bottom = axes
        axes = np.empty((2, 1), dtype=object)
        axes[0, 0] = ax_top
        axes[1, 0] = ax_bottom

    sns.set(style="whitegrid")

    for col, (metric, ylabel, letter_top, letter_bottom, group_id, do_stats) in enumerate(
        metrics_info
    ):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        df_group = group_to_df[group_id]
        tmp = df_group[["receptor_id", "gpcr_class", "state", metric]].copy()
        tmp = tmp.dropna(subset=[metric])
        if tmp.empty:
            continue

        # --- Resumen por estado/clase (panel superior) ---
        for cls in classes_present:
            for st in ["inactive", "active"]:
                vals = tmp[(tmp["gpcr_class"] == cls) & (tmp["state"] == st)][metric]
                vals = vals.dropna()
                if len(vals) == 0:
                    continue
                summary_state.append(
                    {
                        "subset": subset_label,
                        "metric": metric,
                        "class": cls,
                        "state": st,
                        "n_structures": len(vals),
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                        "median": float(vals.median()),
                    }
                )

        wide = tmp.pivot_table(
            index=["receptor_id", "gpcr_class"], columns="state", values=metric
        )

        # --------------------
        # Panel superior: barras + puntos (active vs inactive)
        # --------------------
        sns.barplot(
            data=tmp,
            x="gpcr_class",
            y=metric,
            hue="state",
            order=classes_present,
            hue_order=["inactive", "active"],
            palette=[BAR_STATE_COLORS["inactive"], BAR_STATE_COLORS["active"]],
            ci=None,
            dodge=True,
            ax=ax_top,
            alpha=0.85,
            edgecolor=None,
        )

        sns.stripplot(
            data=tmp,
            x="gpcr_class",
            y=metric,
            hue="state",
            order=classes_present,
            hue_order=["inactive", "active"],
            palette=[POINT_STATE_COLORS["inactive"], POINT_STATE_COLORS["active"]],
            dodge=True,
            alpha=0.5,
            linewidth=0,
            size=5,
            ax=ax_top,
        )

        # Leyenda sólo en la primera columna
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

        # Grid horizontal explícito y sin grid vertical
        ax_top.yaxis.grid(True, linestyle="-", linewidth=0.7, alpha=0.6)
        ax_top.xaxis.grid(False)

        ax_top.set_xlabel("")
        ax_top.set_ylabel(ylabel, fontsize=9)
        ax_top.tick_params(axis="y", labelsize=9)
        ax_top.tick_params(axis="x", labelsize=9)
        ax_top.set_xticklabels(short_labels_present)
        ax_top.set_ylim(0, 30)  # ajusta si tus distancias van en otro rango
        ax_top.text(
            -0.2,
            1.05,
            f"{letter_top})",
            transform=ax_top.transAxes,
            fontweight="bold",
            fontsize=14,
        )

        xticks = ax_top.get_xticks()
        delta = 0.18
        top_tests = []
        class_n_pairs = {}

        # --- Paired t-test active vs inactive + n de pares (receptores) ---
        for i, cls in enumerate(classes_present):
            mask_cls = wide.index.get_level_values("gpcr_class") == cls
            sub_w = wide[mask_cls]

            if (
                sub_w.empty
                or "active" not in sub_w.columns
                or "inactive" not in sub_w.columns
            ):
                continue

            vals_act = sub_w["active"]
            vals_inact = sub_w["inactive"]
            mask = (~vals_act.isna()) & (~vals_inact.isna())
            vals_act = vals_act[mask]
            vals_inact = vals_inact[mask]
            n_pairs = len(vals_act)

            if n_pairs == 0:
                continue

            class_n_pairs[cls] = n_pairs

            # Si no queremos tests estadísticos para esta métrica, nos quedamos sólo con n
            if not do_stats:
                continue

            if n_pairs <= 1:
                continue

            t_stat, p_raw = stats.ttest_rel(vals_act, vals_inact)
            top_tests.append(
                {
                    "subset": subset_label,
                    "metric": metric,
                    "class": cls,
                    "index": i,
                    "n": n_pairs,
                    "mean_active": float(vals_act.mean()),
                    "mean_inactive": float(vals_inact.mean()),
                    "p_raw": float(p_raw),
                }
            )

        if do_stats and top_tests:
            raw_p = [t["p_raw"] for t in top_tests]
            adj_p = benjamini_hochberg(raw_p)

            for t, p_adj in zip(top_tests, adj_p):
                state_stats.append(
                    {
                        "subset": t["subset"],
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

                i = t["index"]
                x_center = xticks[i]
                x1 = x_center - delta
                x2 = x_center + delta
                y_class = tmp.loc[tmp["gpcr_class"] == t["class"], metric].max()
                y_line = y_class + 0.6
                add_sig_line(ax_top, x1, x2, y_line, dy=0.3, p_adj=p_adj, fontsize=10)

        # --- n de pares en la base (esto sí lo dejamos siempre) ---
        for i, cls in enumerate(classes_present):
            n_pairs = class_n_pairs.get(cls, 0)
            if n_pairs <= 0:
                continue
            x_center = xticks[i]
            y_n = 0.3
            ax_top.text(
                x_center,
                y_n,
                f"n={n_pairs}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # --------------------
        # Panel inferior: log2FC active/inactive
        # --------------------
        if "active" in wide.columns and "inactive" in wide.columns:
            log2fc = np.log2(wide["active"] / wide["inactive"])
        else:
            log2fc = pd.Series(dtype=float)

        log_df = log2fc.rename("log2FC").reset_index().dropna(subset=["log2FC"])

        # Resumen log2FC por clase (descriptivo)
        for cls in classes_present:
            vals = log_df[log_df["gpcr_class"] == cls]["log2FC"]
            if len(vals) == 0:
                continue
            summary_log2fc.append(
                {
                    "subset": subset_label,
                    "metric": metric,
                    "class": cls,
                    "n_receptors": len(vals),
                    "mean_log2FC": float(vals.mean()),
                    "std_log2FC": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "median_log2FC": float(vals.median()),
                }
            )

        sns.barplot(
            data=log_df,
            x="gpcr_class",
            y="log2FC",
            order=classes_present,
            color=BAR_BLUE,
            ci=None,
            ax=ax_bottom,
            alpha=0.85,
            edgecolor=None,
        )

        sns.stripplot(
            data=log_df,
            x="gpcr_class",
            y="log2FC",
            order=classes_present,
            color=POINT_BLUE,
            alpha=0.5,
            size=5,
            ax=ax_bottom,
        )

        ax_bottom.yaxis.grid(True, linestyle="-", linewidth=0.7, alpha=0.6)
        ax_bottom.xaxis.grid(False)

        ax_bottom.axhline(0, ls="--", c="black", lw=1)
        ax_bottom.set_xlabel("")
        ax_bottom.set_ylabel("log2(FC) active / inactive", fontsize=9)
        ax_bottom.tick_params(axis="y", labelsize=9)
        ax_bottom.tick_params(axis="x", labelsize=9)
        ax_bottom.set_xticklabels(short_labels_present)
        ax_bottom.set_ylim(-1, 3)
        ax_bottom.text(
            -0.2,
            1.05,
            f"{letter_bottom})",
            transform=ax_bottom.transAxes,
            fontweight="bold",
            fontsize=14,
        )

        # Welch t-test entre clases presentes (log2FC) + BH
        # SOLO si do_stats=True para esta métrica
        if do_stats:
            bottom_tests = []
            for i1, i2 in itertools.combinations(range(len(classes_present)), 2):
                c1 = classes_present[i1]
                c2 = classes_present[i2]
                vals1 = log_df[log_df["gpcr_class"] == c1]["log2FC"]
                vals2 = log_df[log_df["gpcr_class"] == c2]["log2FC"]
                if len(vals1) <= 1 or len(vals2) <= 1:
                    continue
                t_stat, p_raw = stats.ttest_ind(vals1, vals2, equal_var=False)
                bottom_tests.append(
                    {
                        "subset": subset_label,
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

                baseline = 1.9
                step = 0.35
                offset = 0.0

                for t, p_adj in zip(bottom_tests, adj_p):
                    class_stats.append(
                        {
                            "subset": t["subset"],
                            "metric": t["metric"],
                            "comparison": t["comparison"],
                            "test": "Welch t-test",
                            "n1": t["n1"],
                            "n2": t["n2"],
                            "mean1": float(t["mean1"]),
                            "mean2": float(t["mean2"]),
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
                            dy=0.12,
                            p_adj=p_adj,
                            fontsize=9,
                        )
                        offset += step

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = pd.read_csv(CSV_PATH)

    # Si no existe receptor_id, lo inferimos del nombre de fichero
    if "receptor_id" not in df.columns:
        df["receptor_id"] = df["file"].str.extract(r"(.*)_all_")[0]

    # Comprobación mínima
    for col in ["gpcr_class", "state", DRY_SC_METRIC, DRY_CA_METRIC]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en {CSV_PATH}")

    state_stats = []
    class_stats = []
    summary_state = []
    summary_log2fc = []

    # Métricas DRY: SC–SC y Cα–Cα
    # El último elemento del tuple indica si se hacen tests estadísticos (True/False)
    metrics_dry = [
        (
            DRY_SC_METRIC,
            r"DRY lock D/E$^{3.49}$–R/K$^{3.50}$ N–O [Å]",
            "A",
            "B",
            "dry",
            True,   # SÍ tests para SC–SC
        ),
        (
            DRY_CA_METRIC,
            r"Cα–Cα D/E$^{3.49}$–R/K$^{3.50}$ [Å]",
            "C",
            "D",
            "dry",
            False,  # NO tests para Cα–Cα ni su log2FC
        ),
    ]

    # Usamos TODO el df como subset (no distinguimos compatibilidad aquí)
    make_panels_for_subset(
        df,
        metrics_dry,
        FIG_DRY,
        "dry_all",
        state_stats,
        class_stats,
        summary_state,
        summary_log2fc,
        fig_kind="rect",
    )

    # Guardar tablas de resultados y resúmenes en Excel
    state_df = pd.DataFrame(state_stats)
    class_df = pd.DataFrame(class_stats)
    summary_state_df = pd.DataFrame(summary_state)
    summary_log2fc_df = pd.DataFrame(summary_log2fc)

    with pd.ExcelWriter(EXCEL_PATH) as writer:
        state_df.to_excel(
            writer, sheet_name="state_active_vs_inactive_BH", index=False
        )
        class_df.to_excel(
            writer, sheet_name="log2FC_between_classes_BH", index=False
        )
        summary_state_df.to_excel(
            writer, sheet_name="per_state_summary", index=False
        )
        summary_log2fc_df.to_excel(
            writer, sheet_name="per_class_log2FC_summary", index=False
        )


if __name__ == "__main__":
    main()

