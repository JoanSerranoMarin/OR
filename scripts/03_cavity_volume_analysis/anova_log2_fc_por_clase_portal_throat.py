#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# Conjunto canónico de columnas de trabajo (interno del script)
REQ_COLS = {"GPCR_name","Inactive","Active","log2(active/inactive)","Class"}

# Orden canónico deseado para TODAS las figuras
CLASS_ORDER_CANON = ["Class A","Class B1","Class B2","Class C","Class F","Class O1","Class O2","Class T2"]

# -------------------- utilidades de nombres --------------------
def class_display_name(name: str) -> str:
    """
    Mostrar 'Class A', 'Class B', ... (eliminar el texto entre paréntesis).
    """
    s = str(name).strip()
    s = re.sub(r"\s*\([^)]*\)\s*", "", s).strip()
    return s

def class_display_list(names):
    return [class_display_name(n) for n in names]

# -------------------- detección y mapeo de columnas --------------------
# Mapeo flexible de nombres alternativos -> nombres canónicos
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

def _normalize_colname(c):
    """Normaliza nombres de columna para comparar (minúsculas, trim, compactar espacios)."""
    return re.sub(r"\s+", " ", str(c).strip()).lower()

def detect_column_mapping(df_columns):
    """
    Detecta el mapeo de columnas del Excel a los nombres canónicos REQ_COLS.
    Devuelve dict {canonico: columna_en_df} con las que encuentre.
    """
    cols_norm = {_normalize_colname(c): c for c in df_columns}
    mapping = {}
    for canonical, aliases in SYNONYMS.items():
        for alias in aliases:
            if alias in cols_norm:
                mapping[canonical] = cols_norm[alias]
                break
    return mapping

# -------------------- carga y limpieza de datos --------------------
def load_table(xlsx_path: Path, sheet=None) -> pd.DataFrame:
    """
    Lee Excel y devuelve DataFrame con columnas canónicas:
    ['GPCR_name','Inactive','Active','log2(active/inactive)','Class']
    - Detecta nombres alternativos de columnas (como en tu Excel).
    - Calculará log2(active/inactive) si no viene en el archivo.
    - Convierte 'Active' y 'Inactive' a numérico (coerce).
    """
    xf = pd.ExcelFile(xlsx_path)
    if sheet is not None:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    else:
        preferred = "Sheet2"
        sheet_to_read = preferred if preferred in xf.sheet_names else xf.sheet_names[0]
        df = pd.read_excel(xlsx_path, sheet_name=sheet_to_read)

    mapping = detect_column_mapping(df.columns)

    if not ({"Inactive","Active"}.issubset(mapping.keys()) or "log2(active/inactive)" in mapping):
        raise ValueError(
            "No encuentro columnas suficientes para 'Active' e 'Inactive' ni la razón log2.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )
    if "Class" not in mapping:
        raise ValueError("No encuentro la columna de 'Class' (clase GPCR).")
    if "GPCR_name" not in mapping:
        df["_temp_gpcr"] = [f"receptor_{i+1}" for i in range(len(df))]
        mapping["GPCR_name"] = "_temp_gpcr"

    rename_map = {mapping[k]: k for k in mapping}
    d = df.rename(columns=rename_map).copy()

    if "Inactive" in d:
        d["Inactive"] = pd.to_numeric(d["Inactive"], errors="coerce")
    if "Active" in d:
        d["Active"]  = pd.to_numeric(d["Active"], errors="coerce")

    if "log2(active/inactive)" not in d and {"Active","Inactive"}.issubset(d.columns):
        d.loc[~np.isfinite(d["Inactive"]) | (d["Inactive"] <= 0), "Inactive"] = np.nan
        d.loc[~np.isfinite(d["Active"])   | (d["Active"]   <= 0), "Active"]   = np.nan
        d["log2(active/inactive)"] = np.log2(d["Active"] / d["Inactive"])

    missing = REQ_COLS - set(d.columns)
    if missing:
        raise ValueError(f"Faltan columnas tras el mapeo/cálculo: {missing}")

    used = {k: mapping.get(k, k) for k in ["GPCR_name","Inactive","Active","Class","log2(active/inactive)"]}
    print(f"[INFO] Mapeo de columnas -> canónicas: {used}")

    return d[["GPCR_name","Inactive","Active","log2(active/inactive)","Class"]].copy()

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
        lo, hi = mu - z*sd, mu + z*sd
        return g[(g["log2(active/inactive)"] >= lo) & (g["log2(active/inactive)"] <= hi)]
    return d.groupby("Class", group_keys=False, sort=False).apply(filt).reset_index(drop=True)

# -------------------- ANOVA + post-hoc --------------------
def one_way_anova(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class"):
    from scipy import stats
    groups = [g[ycol].dropna().values for _, g in df.groupby(gcol, sort=False)]
    labels = [str(n) for n, _ in df.groupby(gcol, sort=False)]
    valid = [(lab, arr) for lab, arr in zip(labels, groups) if len(arr) >= 2]
    if len(valid) < 2:
        raise ValueError("Need ≥2 classes with at least 2 observations after filtering.")
    labels, groups = zip(*valid)
    F, p = stats.f_oneway(*groups)
    SSE = sum(((arr - np.mean(arr))**2).sum() for arr in groups)
    df_within = sum(len(arr) for arr in groups) - len(groups)
    MSE = SSE / df_within if df_within > 0 else np.nan
    return F, p, list(labels), list(groups), df_within, MSE

def tukey_or_holm(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class",
                  df_within=None, MSE=None, alpha=0.05):
    means = df.groupby(gcol, sort=False)[ycol].mean().to_dict()
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        res = pairwise_tukeyhsd(endog=df[ycol].values, groups=df[gcol].values, alpha=alpha)
        out = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
        out = out.rename(columns={"p-adj":"p_adj", "reject":"reject", "meandiff":"meandiff"})
        out["p_adj"] = out["p_adj"].astype(float)
        out["reject"] = out["reject"].astype(bool)
        method = "TukeyHSD"
    except Exception:
        from itertools import combinations
        from scipy.stats import t as t_dist
        if df_within is None or MSE is None or not np.isfinite(MSE):
            raise RuntimeError("No Tukey available and missing MSE/df for Holm correction.")
        rows = []
        groups_dict = {k: v.dropna().values for k, v in df.groupby(gcol, sort=False)[ycol]}
        names = list(groups_dict.keys())
        for g1, g2 in combinations(names, 2):
            x1, x2 = groups_dict[g1], groups_dict[g2]
            if len(x1) < 2 or len(x2) < 2:
                continue
            n1, n2 = len(x1), len(x2)
            m1, m2 = x1.mean(), x2.mean()
            se = np.sqrt(MSE * (1/n1 + 1/n2)) if MSE > 0 else np.nan
            tval = (m1 - m2) / se if (se and se>0) else np.nan
            p_raw = 2*(1 - t_dist.cdf(abs(tval), df=df_within)) if np.isfinite(tval) else np.nan
            rows.append({"group1":g1, "group2":g2, "meandiff": m1-m2, "p_raw": p_raw})
        out = pd.DataFrame(rows).sort_values("p_raw")
        m = len(out)
        out["p_adj"] = [min(1.0, (m - i + 1) * p) for i, p in enumerate(out["p_raw"], start=1)]
        out["reject"] = out["p_adj"] < alpha
        method = "Holm"
    out["mean1"] = out["group1"].map(means)
    out["mean2"] = out["group2"].map(means)
    out["higher"] = np.where(out["mean1"] > out["mean2"], out["group1"], out["group2"])
    out = out[["group1","group2","mean1","mean2","meandiff","p_adj","reject","higher"]]
    return out, method

def summarize_per_class(df: pd.DataFrame, ycol="log2(active/inactive)", gcol="Class"):
    """
    Resumen por clase respetando SIEMPRE el orden categórico del campo 'Class'
    (si es Categorical y ordered=True). Si no, usa el orden canónico.
    """
    s = df.groupby(gcol, sort=False).agg(n=(ycol,"size"), mean=(ycol,"mean")).reset_index()
    # Reordenar por categorías
    if pd.api.types.is_categorical_dtype(df[gcol]):
        cats = list(df[gcol].cat.categories)
    else:
        cats = CLASS_ORDER_CANON
    s = s.set_index(gcol).reindex([c for c in cats if c in set(s[gcol])]).reset_index()
    return s

def p_to_stars(p):
    if pd.isna(p): return "ns"
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else "ns"))

# -------------------- estilo y colores --------------------
COL_BAR   = "#F28E2B"     # naranja (activo y barras del 1er gráfico)
COL_INACT = "#9CA3AF"     # gris para inactivo
EDGE      = "#222222"
ALPHA_BAR = 0.35
ALPHA_DOT = 0.35

# Escala global de textos (2.0 = el doble)
TEXT_SCALE = 2.0
TEXT_BASE_SIZE = 7
TEXT_SIZE = int(round(TEXT_BASE_SIZE * TEXT_SCALE))

PASTEL_MIX = 0.55  # menor -> más pastel / mayor -> más saturado

def apply_text_scaling(scale: float = 2.0):
    """
    Escala globalmente el tamaño de fuente por defecto de matplotlib (ticks, labels, etc.).
    Mantiene los tamaños relativos de 'large', 'medium', etc. al subir 'font.size'.
    """
    base = float(plt.rcParams.get("font.size", 10.0))
    plt.rcParams["font.size"] = base * scale

def make_pastel_cmap(hex_color=COL_BAR, mix=PASTEL_MIX):
    base = np.array(to_rgb(hex_color))
    pastel = 1 - (1 - base)*mix  # mezcla con blanco
    return LinearSegmentedColormap.from_list(
        "pastel_orange", [(1,1,1), tuple(pastel)], N=256
    )

COLMAP_MATRIX = make_pastel_cmap()  # naranja pastel para la matriz

# -------------------- gráficos --------------------
def _square_figsize(n_classes: int, min_side: float = 8.0, scale: float = 1.0) -> tuple:
    side = max(min_side, scale * n_classes)
    return (side, side)

def plot_means_transparent(summary: pd.DataFrame, df: pd.DataFrame, out_png: Path):
    """
    Figura 1: barras de la media de log2(Active/Inactive) por clase + puntos.
    Anotación: SOLO el factor multiplicativo ×(2**mean), p.ej. ×1.20.
    """
    classes = summary["Class"].tolist()
    labels_display = class_display_list(classes)
    x = np.arange(len(classes))
    means = summary["mean"].values

    plt.figure(figsize=_square_figsize(len(classes), min_side=8.0, scale=1.0))
    plt.bar(x, means, color=COL_BAR, alpha=ALPHA_BAR, edgecolor=EDGE, linewidth=0.6)

    rng = np.random.default_rng(42); jitter = 0.10
    for i, cls in enumerate(classes):
        vals = df.loc[df["Class"]==cls, "log2(active/inactive)"].values
        xs = x[i] + rng.normal(0, jitter, size=len(vals))
        plt.scatter(xs, vals, s=18, alpha=ALPHA_DOT, color=COL_BAR, edgecolors="none")

    y_min = np.nanmin(np.concatenate([means, [0.0]]))
    y_max = np.nanmax(np.concatenate([means, [0.0]]))
    y_rng = max(1e-6, y_max - y_min)
    base_off = 0.10*y_rng

    for i, mu in enumerate(means):
        fold = 2.0**mu
        y_txt = mu + base_off
        plt.text(x[i], y_txt, f"×{fold:.2f}", ha="center", va="bottom",
                 fontsize=TEXT_SIZE, color=EDGE, clip_on=False)

    plt.axhline(0, color="#999999", lw=0.8, zorder=0)
    plt.xticks(x, labels_display, rotation=45, ha="right")
    plt.ylabel(r"Log$_2$(Active/Inactive)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_diff_matrix_lower_abs(pairs_df: pd.DataFrame, summary: pd.DataFrame, out_png: Path,
                               alpha=0.05):
    """
    Figura 2: Triángulo inferior (i>j) de |Δμ|, con asteriscos por significación.
    Eje Y: 'Class' + (mean=..). Eje X: solo 'Class' (sin mean).
    """
    classes = summary["Class"].tolist()
    means   = summary["mean"].tolist()
    K = len(classes)
    if K == 0:
        return

    # mapa de p-valores para pares
    p_map = {}
    for _, r in pairs_df.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        p = float(r["p_adj"])
        p_map[(a,b)] = p
        p_map[(b,a)] = p

    # construir triángulo inferior para |Δμ|
    D = np.full((K,K), np.nan, float)
    P = np.full((K,K), np.nan, float)
    for i in range(K):
        for j in range(i):  # solo inferior
            mi, mj = means[i], means[j]
            D[i, j] = abs(mi - mj)
            P[i, j] = p_map.get((classes[i], classes[j]), np.nan)

    vmax = np.nanmax(D) if np.isfinite(D).any() else 1.0

    cmap = COLMAP_MATRIX
    cmap.set_bad(color=(1,1,1,0))

    fig, ax = plt.subplots(figsize=(max(8, 0.8*K), max(8, 0.8*K)))
    mD = np.ma.masked_invalid(D)
    im = ax.imshow(mD, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor((1,1,1,0))

    for i in range(K):
        for j in range(i):
            p = P[i, j]
            if np.isfinite(p) and p < alpha:
                ax.text(j, i, p_to_stars(p), ha="center", va="center",
                        fontsize=TEXT_SIZE+1, color="#111")

    # etiquetas de ejes
    xt = [class_display_name(c) for c in classes]  # SOLO clase en X
    yt = [f"{class_display_name(c)}\n(mean={m:.2f})" for c, m in zip(classes, means)]
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=TEXT_SIZE+1)
    ax.set_yticks(np.arange(K)); ax.set_yticklabels(yt, fontsize=TEXT_SIZE+1)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"|meanᵢ − meanⱼ| (Å³)")
    cb.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_active_inactive_absolute(df: pd.DataFrame, out_png: Path, classes_order):
    """
    Figura 3: para cada clase, dos barras (Inactive/Active) + puntos jitter.
    Orden rígido: 'classes_order' (filtrado a las presentes).
    """
    classes = [c for c in classes_order if (df["Class"] == c).any()]
    labels_display = class_display_list(classes)
    x = np.arange(len(classes))
    width = 0.38

    mean_inact = (df.groupby("Class", sort=False)["Inactive"]
                    .mean()
                    .reindex(classes))
    mean_act   = (df.groupby("Class", sort=False)["Active"]
                    .mean()
                    .reindex(classes))

    plt.figure(figsize=_square_figsize(len(classes), min_side=8.0, scale=1.0))
    plt.bar(x - width/2, mean_inact.values, width=width, color=COL_INACT, alpha=ALPHA_BAR,
            edgecolor=EDGE, linewidth=0.6)
    plt.bar(x + width/2, mean_act.values,   width=width, color=COL_BAR,   alpha=ALPHA_BAR,
            edgecolor=EDGE, linewidth=0.6)

    rng = np.random.default_rng(123)
    jitter = 0.08
    for i, cls in enumerate(classes):
        vals_in = df.loc[df["Class"]==cls, "Inactive"].dropna().values
        vals_ac = df.loc[df["Class"]==cls, "Active"].dropna().values
        xs_in = (x[i] - width/2) + rng.normal(0, jitter, size=len(vals_in))
        xs_ac = (x[i] + width/2) + rng.normal(0, jitter, size=len(vals_ac))
        plt.scatter(xs_in, vals_in, s=18, alpha=ALPHA_DOT, color=COL_INACT, edgecolors="none")
        plt.scatter(xs_ac, vals_ac, s=18, alpha=ALPHA_DOT, color=COL_BAR,   edgecolors="none")

    plt.xticks(x, labels_display, rotation=45, ha="right")
    plt.ylabel("Volume (Å³)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="ANOVA + post-hoc; figuras cuadradas, sin títulos ni leyendas; orden rígido de clases.")
    ap.add_argument("--in", dest="xlsx", required=True, help="Ruta del Excel de entrada")
    ap.add_argument("--sheet", dest="sheet", default=None,
                    help="Nombre o índice de hoja (opcional). Si no se da, intenta 'Sheet2' y si no existe usa la primera.")
    ap.add_argument("--outdir", default="anova_log2_outputs", help="Carpeta de salida")
    ap.add_argument("--alpha", type=float, default=0.05, help="Nivel de significación")
    ap.add_argument("--pastel", type=float, default=PASTEL_MIX, help="Mezcla con blanco para la matriz (0=blanco, 1=color base)")
    args = ap.parse_args()

    # Duplicar el tamaño del texto globalmente
    apply_text_scaling(TEXT_SCALE)

    xlsx = Path(args.xlsx)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # cargar tabla
    df_all = load_table(xlsx, sheet=args.sheet)

    # Orden rígido solicitado, filtrado a clases presentes en los datos originales
    classes_order = [c for c in CLASS_ORDER_CANON if (df_all["Class"] == c).any()]
    if not classes_order:
        # Fallback por si el Excel usa otros nombres
        classes_order = [c for c in pd.unique(df_all["Class"])
                         if str(c).strip() and str(c).strip().upper() != "CLASSLESS"]

    # limpiar
    df = remove_classless_and_outliers(df_all, z=2.0)

    # fijar dtype categórico con el orden rígido
    df["Class"] = pd.Categorical(df["Class"], categories=classes_order, ordered=True)

    # ajustar pastel desde CLI
    global COLMAP_MATRIX
    COLMAP_MATRIX = make_pastel_cmap(mix=args.pastel)

    # ANOVA + post-hoc + summary (respetando orden categórico)
    F, p, labels, arrays, df_within, MSE = one_way_anova(df)
    pairs_df, method = tukey_or_holm(df, df_within=df_within, MSE=MSE, alpha=args.alpha)
    summary = summarize_per_class(df)

    # figuras (orden rígido)
    plot_means_transparent(summary, df, outdir/"means_by_class.png")
    plot_diff_matrix_lower_abs(pairs_df, summary, outdir/"posthoc_matrix_lower.png", alpha=args.alpha)
    plot_active_inactive_absolute(df, outdir/"active_inactive_by_class.png", classes_order)

    print(f"[OK] ANOVA F={F:.3f}, p={p:.3g} | post-hoc: {method}")
    print(f" Figures: {outdir/'means_by_class.png'}, {outdir/'posthoc_matrix_lower.png'}, {outdir/'active_inactive_by_class.png'}")
    print(f" Tables:  {outdir/'anova.tsv'}, {outdir/f'posthoc_{method}.tsv'}, {outdir/'summary_by_class.tsv'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import sys
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

