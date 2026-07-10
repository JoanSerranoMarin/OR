#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BAR_STATE_COLORS = {"inactive": "#dcdcdc", "active": "#f6d3a0"}
POINT_STATE_COLORS = {"inactive": "#555555", "active": "#a45a15"}
BAR_BLUE = "#b3d3e8"
POINT_BLUE = "#1f77b4"

# =========================
# SOLO O1 / O2 (SIN CLASS A)
# =========================
CLASSES_TO_KEEP = [
    "Class O1 (fish-like odorant)",
    "Class O2 (tetrapod specific odorant)",
]
CLASS_LABELS_SHORT = ["Class O1", "Class O2"]

# =========================
# Y LABEL CORRECTA (sin cursivas, ambos pueden ser E/D)
# residuos: 2.50 y 3.39
# =========================
Y_LABEL = r"acidic diad $\mathrm{E/D}^{2.50}$-$\mathrm{E/D}^{3.39}$ [$\AA$]"


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    bh = ranked * n / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    out = np.empty_like(bh)
    out[order] = bh
    return out


def add_sig_line(ax, x1, x2, y, dy, p_adj, fontsize=10):
    """
    Dibuja SOLO una línea horizontal + estrellas (sin brackets),
    igual para Panel A y Panel B.
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


def extract_receptor_id(file_value: str) -> str:
    s = str(file_value)
    m = re.match(r"^(.*?)_all_", s)
    if m:
        return m.group(1)
    return s.replace(".pdb", "")


def normalize_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["state"] = df["state"].astype(str).str.strip().str.lower()
    df.loc[df["state"].str.startswith("act"), "state"] = "active"
    df.loc[df["state"].str.startswith("inact"), "state"] = "inactive"
    return df


def remove_outliers_two_sd(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Elimina outliers > 2 SD dentro de cada grupo (gpcr_class × state) para 'metric'.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for (cls, state), grp in df.groupby(["gpcr_class", "state"], dropna=False):
        vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        if std == 0 or np.isnan(std):
            continue

        grp_vals = pd.to_numeric(grp[metric], errors="coerce")
        grp_mask = (grp_vals - mean).abs() <= 2.0 * std
        grp_mask = grp_mask.fillna(True)
        mask.loc[grp.index] = grp_mask

    return df[mask]


def plot_like_original(
    df: pd.DataFrame,
    metric_col: str,
    tag: str,
    out_fig: str,
    out_excel: str,
    ylim_top=None,
    ylim_bottom=None,
):
    """
    Figura 1x2:
      A) métrica por clase y estado (sin agregación)
      B) log2FC por receptor (pivot receptor_id para ratio)
    """
    df = df.copy()
    df = df[df["gpcr_class"].isin(CLASSES_TO_KEEP)].copy()

    # Outliers >2 SD (por clase × estado) ANTES de todo
    df = remove_outliers_two_sd(df, metric_col)

    if df.empty:
        raise ValueError(f"[{tag}] Dataset vacío tras filtrar outliers y clases.")

    sns.set(style="whitegrid")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # ---------------- Panel A ----------------
    sns.barplot(
        data=df,
        x="gpcr_class",
        y=metric_col,
        hue="state",
        order=CLASSES_TO_KEEP,
        hue_order=["inactive", "active"],
        palette=[BAR_STATE_COLORS["inactive"], BAR_STATE_COLORS["active"]],
        errorbar=None,
        dodge=True,
        ax=axA,
        alpha=0.85,
        edgecolor=None,
    )
    sns.stripplot(
        data=df,
        x="gpcr_class",
        y=metric_col,
        hue="state",
        order=CLASSES_TO_KEEP,
        hue_order=["inactive", "active"],
        palette=[POINT_STATE_COLORS["inactive"], POINT_STATE_COLORS["active"]],
        dodge=True,
        alpha=0.50,
        linewidth=0,
        size=5,
        ax=axA,
    )

    handles, labels = axA.get_legend_handles_labels()
    axA.legend(
        handles[:2],
        labels[:2],
        title="",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
    )

    axA.set_xlabel("")
    axA.set_ylabel(Y_LABEL)
    axA.set_xticks(range(len(CLASSES_TO_KEEP)))
    axA.set_xticklabels(CLASS_LABELS_SHORT)
    if ylim_top is not None:
        axA.set_ylim(0, float(ylim_top))
    axA.text(-0.15, 1.05, "A)", transform=axA.transAxes, fontweight="bold", fontsize=14)

    # Pivot para pares y log2FC
    wide = df.pivot_table(
        index=["receptor_id", "gpcr_class"],
        columns="state",
        values=metric_col,
        aggfunc="mean",
    )

    # n pares por clase
    for i, cls in enumerate(CLASSES_TO_KEEP):
        if cls not in wide.index.get_level_values("gpcr_class"):
            n_pairs = 0
        else:
            sub = wide.loc[wide.index.get_level_values("gpcr_class") == cls]
            if ("active" in sub.columns) and ("inactive" in sub.columns):
                n_pairs = int((~sub["active"].isna() & ~sub["inactive"].isna()).sum())
            else:
                n_pairs = 0
        y0, y1 = axA.get_ylim()
        axA.text(i, y0 + 0.02 * (y1 - y0), f"n={n_pairs}", ha="center", va="bottom", fontsize=10)

    # Paired t-test por clase + BH (activo vs inactivo dentro de cada clase)
    top_tests = []
    for i, cls in enumerate(CLASSES_TO_KEEP):
        if cls not in wide.index.get_level_values("gpcr_class"):
            continue
        sub = wide.loc[wide.index.get_level_values("gpcr_class") == cls]
        if sub.empty or ("active" not in sub.columns) or ("inactive" not in sub.columns):
            continue

        a = sub["active"]
        b = sub["inactive"]
        m = (~a.isna()) & (~b.isna())
        a = a[m].astype(float)
        b = b[m].astype(float)

        if len(a) < 2:
            continue

        _, p_raw = stats.ttest_rel(a.to_numpy(float), b.to_numpy(float))
        top_tests.append({"class": cls, "i": i, "n_pairs": int(len(a)), "p_raw": float(p_raw)})

    state_stats_rows = []
    if top_tests:
        adj_top = benjamini_hochberg([t["p_raw"] for t in top_tests])
        xticks = axA.get_xticks()
        delta = 0.18
        for t, p_adj in zip(top_tests, adj_top):
            state_stats_rows.append(
                {
                    "tag": tag,
                    "metric": metric_col,
                    "class": t["class"],
                    "test": "paired t-test",
                    "n_pairs": t["n_pairs"],
                    "p_raw": t["p_raw"],
                    "p_adj_BH": float(p_adj),
                }
            )
            i = t["i"]
            x_center = xticks[i]
            x1 = x_center - delta
            x2 = x_center + delta
            y_max = df.loc[df["gpcr_class"] == t["class"], metric_col].max()
            y0, y1 = axA.get_ylim()
            y_line = y_max + 0.04 * (y1 - y0)
            add_sig_line(axA, x1, x2, y=y_line, dy=0.01 * (y1 - y0), p_adj=p_adj, fontsize=10)

    # ---------------- Panel B (log2FC) ----------------
    if ("active" in wide.columns) and ("inactive" in wide.columns):
        log2fc = np.log2(wide["active"] / wide["inactive"])
    else:
        log2fc = pd.Series(dtype=float)

    log_df = log2fc.rename("log2FC").reset_index().dropna(subset=["log2FC"])

    sns.barplot(
        data=log_df,
        x="gpcr_class",
        y="log2FC",
        order=CLASSES_TO_KEEP,
        color=BAR_BLUE,
        errorbar=None,
        ax=axB,
        alpha=0.85,
        edgecolor=None,
    )
    sns.stripplot(
        data=log_df,
        x="gpcr_class",
        y="log2FC",
        order=CLASSES_TO_KEEP,
        color=POINT_BLUE,
        alpha=0.50,
        size=5,
        ax=axB,
    )

    axB.axhline(0, ls="--", c="black", lw=1)
    axB.set_xlabel("")
    axB.set_ylabel("log2(FC) active / inactive")
    axB.set_xticks(range(len(CLASSES_TO_KEEP)))
    axB.set_xticklabels(CLASS_LABELS_SHORT)
    if ylim_bottom is not None:
        axB.set_ylim(float(ylim_bottom[0]), float(ylim_bottom[1]))
    axB.text(-0.15, 1.05, "B)", transform=axB.transAxes, fontweight="bold", fontsize=14)

    # Welch entre clases sobre log2FC + BH
    class_stats_rows = []
    bottom_tests = []
    present_classes = [c for c in CLASSES_TO_KEEP if c in set(log_df["gpcr_class"].unique())]

    for i1 in range(len(present_classes)):
        for i2 in range(i1 + 1, len(present_classes)):
            c1, c2 = present_classes[i1], present_classes[i2]
            v1 = log_df.loc[log_df["gpcr_class"] == c1, "log2FC"].to_numpy(float)
            v2 = log_df.loc[log_df["gpcr_class"] == c2, "log2FC"].to_numpy(float)
            if len(v1) < 2 or len(v2) < 2:
                continue
            _, p_raw = stats.ttest_ind(v1, v2, equal_var=False)
            bottom_tests.append({"c1": c1, "c2": c2, "comparison": f"{c1} vs {c2}", "p_raw": float(p_raw)})

    drew_any = False
    if bottom_tests:
        pvals_bottom_adj = list(benjamini_hochberg([t["p_raw"] for t in bottom_tests]))

        for t, p_adj in zip(bottom_tests, pvals_bottom_adj):
            class_stats_rows.append(
                {
                    "tag": tag,
                    "metric": metric_col,
                    "comparison": t["comparison"],
                    "test": "Welch t-test",
                    "p_raw": t["p_raw"],
                    "p_adj_BH": float(p_adj),
                }
            )

        # ====== SIGNIFICANCIA EN PANEL B: SOLO LÍNEA (sin brackets) ======
        # Mapeo clase->posición x según order=CLASSES_TO_KEEP
        xmap = {cls: i for i, cls in enumerate(CLASSES_TO_KEEP)}

        y_max_data = float(log_df["log2FC"].max()) if not log_df.empty else 0.0
        y_min_data = float(log_df["log2FC"].min()) if not log_df.empty else 0.0
        span = (y_max_data - y_min_data) if (y_max_data > y_min_data) else 1.0

        # coloca la línea por encima del máximo
        y_line = y_max_data + 0.12 * span
        dy_text = 0.02 * span

        for t, p_adj in zip(bottom_tests, pvals_bottom_adj):
            if (t["c1"] in xmap) and (t["c2"] in xmap):
                x1, x2 = xmap[t["c1"]], xmap[t["c2"]]
                add_sig_line(axB, x1, x2, y=y_line, dy=dy_text, p_adj=float(p_adj), fontsize=12)
                if (not np.isnan(p_adj)) and (p_adj < 0.05):
                    drew_any = True

        # si hace falta, expande ylim para que se vea la línea (solo si no fijaste ylim_bottom)
        if drew_any and (ylim_bottom is None):
            y0, y1 = axB.get_ylim()
            needed_top = y_line + 0.08 * span
            if y1 < needed_top:
                axB.set_ylim(y0, needed_top)

    plt.suptitle(tag, y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.show()

    with pd.ExcelWriter(out_excel) as writer:
        pd.DataFrame(state_stats_rows).to_excel(writer, sheet_name="active_vs_inactive_BH", index=False)
        pd.DataFrame(class_stats_rows).to_excel(writer, sheet_name="log2FC_between_classes_BH", index=False)

    print(f"[OK] ({tag}) Figura: {out_fig}")
    print(f"[OK] ({tag}) Stats : {out_excel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="gpcr_339_250_metrics_class.xlsx")
    ap.add_argument("--sheet", default="0", help="Hoja (índice numérico o nombre). Por defecto 0.")
    ap.add_argument("--out-prefix", default="gpcr_339_250")
    ap.add_argument("--ylim-top", type=float, default=None)
    ap.add_argument(
        "--ylim-bottom",
        nargs=2,
        type=float,
        default=None,
        help="Ej: --ylim-bottom -1 3",
    )
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet)

    need = {"gpcr_class", "state", "file"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    # ---- ELIMINAR TODO LO QUE NO SEA O1/O2 ANTES DE NADA ----
    df = df[df["gpcr_class"].isin(CLASSES_TO_KEEP)].copy()

    metric_col = df.columns[-1]  # última columna (distancia)

    if len(df.columns) < 12:
        raise ValueError(f"El Excel tiene {len(df.columns)} columnas, necesito al menos 12.")
    col9_name = df.columns[8]
    col12_name = df.columns[11]
    print(f"[INFO] Columna 9  -> '{col9_name}'")
    print(f"[INFO] Columna 12 -> '{col12_name}'")
    print(f"[INFO] Métrica    -> '{metric_col}'")

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("OK")].copy()

    df = normalize_state(df)
    df["receptor_id"] = df["file"].apply(extract_receptor_id)
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    df[col9_name] = pd.to_numeric(df[col9_name], errors="coerce").fillna(0)
    df[col12_name] = pd.to_numeric(df[col12_name], errors="coerce").fillna(0)

    df_all = df.dropna(subset=["gpcr_class", "state", "receptor_id", metric_col]).copy()

    # Figura 1: ALL
    plot_like_original(
        df_all,
        metric_col=metric_col,
        tag="ALL (Class O1 / O2 only) | outliers >2SD removed",
        out_fig=f"{args.out_prefix}_ALL_O1_O2_outliers2SD.png",
        out_excel=f"{args.out_prefix}_ALL_O1_O2_outliers2SD_stats_BH.xlsx",
        ylim_top=args.ylim_top,
        ylim_bottom=args.ylim_bottom,
    )

    # Figura 2: col9==1 y col12==1
    df_both = df_all[(df_all[col9_name] == 1) & (df_all[col12_name] == 1)].copy()
    print(f"[INFO] Filtrado col9==1 & col12==1: {len(df_both)} filas")

    if df_both.empty:
        print("[WARN] Subset col9==1 & col12==1 vacío -> no genero la segunda figura.")
        return

    plot_like_original(
        df_both,
        metric_col=metric_col,
        tag="BOTH col9==1 & col12==1 (O1/O2 only) | outliers >2SD removed",
        out_fig=f"{args.out_prefix}_BOTH_9_12_EQ1_O1_O2_outliers2SD.png",
        out_excel=f"{args.out_prefix}_BOTH_9_12_EQ1_O1_O2_outliers2SD_stats_BH.xlsx",
        ylim_top=args.ylim_top,
        ylim_bottom=args.ylim_bottom,
    )


if __name__ == "__main__":
    main()

