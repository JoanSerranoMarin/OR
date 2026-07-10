#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figuras para ionic lock tradicional/alternativo por subgrupos de compatibilidad,
incluyendo distancias SC–SC (N–O mínima) y Cα–Cα.

En cada figura, para cada pareja (SC–SC, Cα–Cα) se usan únicamente los
receptores que:
    - Tienen ambas métricas definidas
    - NO son outliers (>2 SD) en ninguna de las dos métricas (por clase/estado)

Además, para los receptores compatibles con AMBOS locks:
    - Se clasifica cada receptor según cuál lock tiene la distancia SC–SC mínima
      en el estado inactivo (lock "dominante").

Input:
    gpcr_distances_full_annot.csv

Output:
    figure_lock_traditional_only.png
    figure_lock_alternative_only.png
    figure_lock_traditional_and_alternative.png
    figure_lock_both_dominant_traditional.png
    figure_lock_both_dominant_alternative.png
    gpcr_lock_subsets_stats_BH.xlsx
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

CSV_PATH = "gpcr_distances_annot_loose.csv"

# Columnas de distancias (SC–SC y Cα–Cα).
# OJO: se asume que las SC–SC son N–O mínimas como en los papers citados.
CANONICAL_METRIC = "ionic_lock_R3.50_E/D6.30"          # SC–SC (N–O mínima)
CANONICAL_CA_METRIC = "ionic_lock_CA_R3.50_E/D6.30"     # Cα–Cα
ALTERNATIVE_METRIC = "alt_lock_min_dist"                # SC–SC (N–O mínima)
ALTERNATIVE_CA_METRIC = "alt_lock_CA_3.49_6.30"         # Cα–Cα

FIG_CANONICAL_ONLY = "figure_lock_traditional_only.png"
FIG_ALT_ONLY = "figure_lock_alternative_only.png"
FIG_BOTH_ALL = "figure_lock_traditional_and_alternative.png"
FIG_BOTH_DOM_TRAD = "figure_lock_both_dominant_traditional.png"
FIG_BOTH_DOM_ALT = "figure_lock_both_dominant_alternative.png"

EXCEL_PATH = "gpcr_lock_subsets_stats_BH.xlsx"

# Clases a mostrar (nombres largos en el CSV)
CLASSES_TO_PLOT = [
    "Class A (Rhodopsin)",
    "Class O1 (fish-like odorant)",
    "Class O2 (tetrapod specific odorant)",
]
# Etiquetas cortas en el eje X
CLASS_LABELS_SHORT = ["Class A", "Class O1", "Class O2"]

# Colores
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
# Compatibilidad por receptor (secuencia)
# ----------------------------------------------------------------------
def annotate_compatibility(df):
    """
    Añade columna 'compatibility' a df a nivel de receptor_id:

    - canonical_only: sólo tiene distancias definidas para lock tradicional (SC–SC),
                      ninguna para lock alternativo.
    - alternative_only: sólo alternativo.
    - both: tiene ambas.
    - none: ninguna (no se usa aquí).
    """
    df = df.copy()
    df["receptor_id"] = df["file"].str.extract(r"(.*)_all_")[0]

    compat = df.groupby("receptor_id").agg(
        has_canonical=(CANONICAL_METRIC, lambda s: s.notna().any()),
        has_alternative=(ALTERNATIVE_METRIC, lambda s: s.notna().any()),
    )

    def classify(row):
        if row["has_canonical"] and not row["has_alternative"]:
            return "canonical_only"
        elif row["has_alternative"] and not row["has_canonical"]:
            return "alternative_only"
        elif row["has_canonical"] and row["has_alternative"]:
            return "both"
        else:
            return "none"

    compat["compatibility"] = compat.apply(classify, axis=1)
    df = df.merge(compat["compatibility"], left_on="receptor_id", right_index=True)
    return df


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

            cond = ((group[metric] - mean).abs() <= 4 * std) | group[metric].isna()
            metric_mask.loc[group.index] = cond

        mask_all &= metric_mask

    df = df[mask_all]
    df = df.dropna(subset=metric_list)
    return df


# ----------------------------------------------------------------------
# Paneles superior/inferior para un subset y lista de métricas
# metrics_info: lista de tuplas (metric, ylabel, letter_top, letter_bottom, group_id)
# group_id se usa para compartir filtrado (SC–SC/Cα–Cα) dentro de cada pareja
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
    for metric, _, _, _, group_id in metrics_info:
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

    for col, (metric, ylabel, letter_top, letter_bottom, group_id) in enumerate(
        metrics_info
    ):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        df_group = group_to_df[group_id]
        tmp = df_group[["receptor_id", "gpcr_class", "state", metric]].copy()
        tmp = tmp.dropna(subset=[metric])
        if tmp.empty:
            continue

        # --- Resumen por estado/clase (panel superior)
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
        ax_top.set_ylim(0, 30)
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

        # --- Paired t-test active vs inactive + n de pares (receptores)
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

        if top_tests:
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

        # --- n de pares en la base
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

        # Resumen log2FC por clase
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
                        "p_raw": float(t["p_raw"]),
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
    df = annotate_compatibility(df)

    state_stats = []
    class_stats = []
    summary_state = []
    summary_log2fc = []

    # Subsets básicos
    df_canonical_only = df[df["compatibility"] == "canonical_only"]
    df_alternative_only = df[df["compatibility"] == "alternative_only"]
    df_both = df[df["compatibility"] == "both"]

    # 1) SOLO tradicional: SC–SC y Cα–Cα lado a lado (grupo "canonical")
    metrics_trad = [
        (
            CANONICAL_METRIC,
            r"traditional ionic lock R$^{3.50}$–E/D$^{6.30}$ [Å]",
            "A",
            "B",
            "canonical",
        ),
        (
            CANONICAL_CA_METRIC,
            "Cα–Cα R$^{3.50}$–E/D$^{6.30}$ [Å]",
            "C",
            "D",
            "canonical",
        ),
    ]
    make_panels_for_subset(
        df_canonical_only,
        metrics_trad,
        FIG_CANONICAL_ONLY,
        "canonical_only",
        state_stats,
        class_stats,
        summary_state,
        summary_log2fc,
        fig_kind="rect",
    )

    # 2) SOLO alternativo: SC–SC y Cα–Cα lado a lado (grupo "alternative")
    metrics_alt = [
        (
            ALTERNATIVE_METRIC,
            r"alternative ionic lock D$^{3.49}$–K$^{6.30}$ [Å]",
            "A",
            "B",
            "alternative",
        ),
        (
            ALTERNATIVE_CA_METRIC,
            "Cα–Cα D$^{3.49}$–K$^{6.30}$ [Å]",
            "C",
            "D",
            "alternative",
        ),
    ]
    make_panels_for_subset(
        df_alternative_only,
        metrics_alt,
        FIG_ALT_ONLY,
        "alternative_only",
        state_stats,
        class_stats,
        summary_state,
        summary_log2fc,
        fig_kind="rect",
    )

    # 3) Compatibles con AMBOS: todos los receptores "both"
    metrics_both = [
        (
            CANONICAL_METRIC,
            r"traditional ionic lock R$^{3.50}$–E/D$^{6.30}$ [Å]",
            "A",
            "E",
            "canonical",
        ),
        (
            CANONICAL_CA_METRIC,
            "Cα–Cα R$^{3.50}$–E/D$^{6.30}$ [Å]",
            "B",
            "F",
            "canonical",
        ),
        (
            ALTERNATIVE_METRIC,
            r"alternative ionic lock D$^{3.49}$–K$^{6.30}$ [Å]",
            "C",
            "G",
            "alternative",
        ),
        (
            ALTERNATIVE_CA_METRIC,
            "Cα–Cα D$^{3.49}$–K$^{6.30}$ [Å]",
            "D",
            "H",
            "alternative",
        ),
    ]
    make_panels_for_subset(
        df_both,
        metrics_both,
        FIG_BOTH_ALL,
        "both_all",
        state_stats,
        class_stats,
        summary_state,
        summary_log2fc,
        fig_kind="square",
    )

    # -------- CLASIFICACIÓN DE LOCK DOMINANTE ENTRE RECEPTRORES "BOTH" --------
    # Usamos SOLO estado inactivo y la distancia mínima SC–SC para cada lock.

    df_both_inact = df_both[df_both["state"] == "inactive"].copy()

    # filtramos outliers de forma conjunta en las dos métricas SC–SC
    df_both_inact_clean = joint_filter_by_metrics(
        df_both_inact, [CANONICAL_METRIC, ALTERNATIVE_METRIC]
    )

    dom_trad_ids = []
    dom_alt_ids = []

    for rid, g in df_both_inact_clean.groupby("receptor_id"):
        can_vals = g[CANONICAL_METRIC].dropna()
        alt_vals = g[ALTERNATIVE_METRIC].dropna()
        if len(can_vals) == 0 or len(alt_vals) == 0:
            continue
        min_can = can_vals.min()
        min_alt = alt_vals.min()
        if min_can < min_alt:
            dom_trad_ids.append(rid)
        elif min_alt < min_can:
            dom_alt_ids.append(rid)
        # empates → no se clasifican

    df_both_dom_trad = df_both[df_both["receptor_id"].isin(dom_trad_ids)].copy()
    df_both_dom_alt = df_both[df_both["receptor_id"].isin(dom_alt_ids)].copy()

    # 4) Receptores BOTH cuyo lock dominante (distancia mínima inactiva) es TRADICIONAL
    make_panels_for_subset(
        df_both_dom_trad,
        metrics_both,
        FIG_BOTH_DOM_TRAD,
        "both_dominant_traditional",
        state_stats,
        class_stats,
        summary_state,
        summary_log2fc,
        fig_kind="square",
    )

    # 5) Receptores BOTH cuyo lock dominante es ALTERNATIVO
    make_panels_for_subset(
        df_both_dom_alt,
        metrics_both,
        FIG_BOTH_DOM_ALT,
        "both_dominant_alternative",
        state_stats,
        class_stats,
        summary_state,
        summary_log2fc,
        fig_kind="square",
    )

    # --------- TABLAS PARA COMPROBAR LOCKS DOMINANTES (EXCEL) ---------
    # Exportamos todas las estructuras (activo/inactivo) de cada grupo,
    # con las cuatro distancias relevantes.

    cols_for_export = [
        "receptor_id",
        "gpcr_class",
        "state",
        "file",
        CANONICAL_METRIC,
        CANONICAL_CA_METRIC,
        ALTERNATIVE_METRIC,
        ALTERNATIVE_CA_METRIC,
    ]

    trad_dom_export = df_both_dom_trad[cols_for_export].sort_values(
        ["gpcr_class", "receptor_id", "state"]
    )
    alt_dom_export = df_both_dom_alt[cols_for_export].sort_values(
        ["gpcr_class", "receptor_id", "state"]
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
        trad_dom_export.to_excel(
            writer, sheet_name="dominant_traditional_raw", index=False
        )
        alt_dom_export.to_excel(
            writer, sheet_name="dominant_alternative_raw", index=False
        )


if __name__ == "__main__":
    main()

