#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receptor-level log2FC (active/inactive) by metric and GPCR class.
- Optional outlier filtering on raw data (by class and state) using z*SD.
- ANOVA + Tukey HSD on per-receptor log2FC.
- Excel matrices:
    * TukeyP_<metric>: Tukey HSD p-values (FWER) within the metric.
    * BHqWithin_<metric>: Benjamini–Hochberg q-values (FDR) applied to that metric’s Tukey p-values.
    * ANOVA_<metric>: global ANOVA p-value.
- Figures: bars (mean) + semi-transparent points + significance brackets.
  The significance source for the plot is selectable: Tukey or BHqWithin (flag --sig-source).
"""

import argparse, re, math, itertools, zlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, t as _t_dist
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ======== aesthetics ========
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300,
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 0.8,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE = "#0EA5E9"       # points / bars
EDGE = "#111827"       # lines / outlines
GRID = "#E5E7EB"       # grid

# ======== utils ========
def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

def _bw_superscripts(text: str) -> str:
    s = str(text).replace("_", " ")
    s = re.sub(r'\b([A-Za-z]+)\s*([1-7]\.\d{2})\b', r'\1 $^{\2}$', s)
    s = re.sub(r'(?:(?<=\s)|^)([1-7]\.\d{2})(?=\s|$)', r'BW $^{\1}$', s)
    return s

def prettify_metric(s: str) -> str:
    s = str(s).replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'\btm(\d)\b', r'TM\1', s, flags=re.IGNORECASE)
    return _bw_superscripts(s)

def trim_class(s: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", str(s).strip())

def receptor_from_file(fn: str) -> str:
    base = Path(str(fn)).name
    return base.split("_", 1)[0].lower()

def infer_state(text: str) -> str:
    t = str(text).lower()
    if t.endswith("_active.pdb") or "active" in t: return "active"
    if t.endswith("_inactive.pdb") or "inactive" in t: return "inactive"
    return ""

def bh_qvalues(pvals):
    p = np.array(list(pvals), float)
    q = np.full_like(p, np.nan)
    msk = np.isfinite(p)
    if msk.sum()==0: return q
    ord_ = np.argsort(p[msk])
    ranks = np.empty_like(ord_, dtype=float); ranks[ord_] = np.arange(1, msk.sum()+1, dtype=float)
    m = float(msk.sum())
    q_raw = p[msk]*m/ranks
    q_mon = np.minimum.accumulate(q_raw[::-1])[::-1]
    q[msk] = np.minimum(q_mon, 1.0)
    return q

def tcrit_95(n):
    try: return float(_t_dist.ppf(0.975, df=max(1, n-1)))
    except Exception: return 1.96

def jitter(n, width=0.10, seed=123):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)

def savefig_all(fig, root: Path):
    fig.savefig(root.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(root.with_suffix(".svg"), bbox_inches="tight")

def add_bracket(ax, x0, x1, y, text, h=0.02):
    ax.plot([x0, x0, x1, x1], [y, y+h, y+h, y], lw=0.9, c=EDGE)
    ax.text((x0+x1)/2, y+h, text, ha="center", va="bottom", fontsize=12)

def excel_sheet_name(metric: str, tag: str) -> str:
    base = sanitize_filename(metric)
    candidate = re.sub(r"[\[\]:\*\?/\\]", "_", f"{tag}_{base}")
    if len(candidate) <= 31: return candidate
    h = format(zlib.crc32(base.encode("utf-8")), "08x")[:6]
    keep = 31 - (len(tag) + 1 + 1 + len(h))
    keep = max(6, keep)
    safe = f"{tag}_{base[:keep]}_{h}"
    safe = re.sub(r"[\[\]:\*\?/\\]", "_", safe)
    return safe[:31]

# ======== outliers by class/state (raw) ========
def mark_outliers_by_state(pivot_df, z):
    out_a, out_i = set(), set()
    if z is None or z <= 0: return out_a, out_i
    a = pivot_df["active"].dropna()
    if len(a)>=3:
        mu, sd = float(a.mean()), float(a.std(ddof=1))
        if sd>0: out_a.update(a[(np.abs(a-mu) > z*sd)].index.tolist())
    i = pivot_df["inactive"].dropna()
    if len(i)>=3:
        mu, sd = float(i.mean()), float(i.std(ddof=1))
        if sd>0: out_i.update(i[(np.abs(i-mu) > z*sd)].index.tolist())
    return out_a, out_i

# ======== ANOVA + Tukey ========
def anova_and_tukey(log2_by_class):
    groups = [(c, np.asarray(v,float)) for c,v in log2_by_class.items() if np.isfinite(v).sum()>=2]
    if len(groups)<2:
        return np.nan, pd.DataFrame(columns=["group1","group2","meandiff","p_adj","reject"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, p_anova = f_oneway(*[v for _,v in groups])
    p_anova = float(p_anova)
    data = np.concatenate([v for _,v in groups], axis=0)
    labels = np.concatenate([[c]*len(v) for c,v in groups], axis=0)
    tk = pairwise_tukeyhsd(endog=data, groups=labels, alpha=0.05)
    rows=[]
    for r in tk.summary().data[1:]:
        g1,g2,meandiff,p_adj,lower,upper,reject = r
        rows.append({"group1":g1,"group2":g2,"meandiff":float(meandiff),
                     "p_adj":float(p_adj),"reject":bool(reject)})
    return p_anova, pd.DataFrame(rows)

# ======== main ========
def main():
    ap = argparse.ArgumentParser(description="Receptor-level log2FC; ANOVA+Tukey; Excel+figures.")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-dir", default="gpcr_log2fc")
    ap.add_argument("--force-metrics", default="")
    ap.add_argument("--classes-to-plot",
        default="Class A (Rhodopsin),Class O1 (fish-like odorant),Class O2 (tetrapod specific odorant)")
    ap.add_argument("--outlier-z", type=float, default=0.0,
                    help="Filter outliers in raw values by class/state: |x-μ|>z·SD (z=0: no filter).")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--sig-source", choices=["bh","tukey"], default="tukey",
                    help="Significance source for the plot: 'tukey' (Tukey HSD p) or 'bh' (BH-FDR q of Tukey p within metric).")
    ap.add_argument("--debug-pairs", action="store_true",
                    help="Save/print the exact pairs and values used for the plot brackets.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figs_dir = out/"figs_log2fc"; figs_dir.mkdir(exist_ok=True)
    excel_path = out/"significance_matrices.xlsx"

    df = pd.read_csv(args.in_csv)

    if "receptor" not in df.columns:
        df["receptor"] = df["file"].astype(str).apply(receptor_from_file)
    if "state" not in df.columns or df["state"].isna().any():
        df["state"] = df["file"].astype(str).apply(infer_state)
    df["state"] = df["state"].astype(str).str.strip().str.lower().replace(
        {"activo":"active","activa":"active","inactivo":"inactive","inactiva":"inactive"}
    )
    if "gpcr_class" not in df.columns:
        raise ValueError("Input must contain 'gpcr_class' column.")

    # numeric columns
    blocked = {"file","chain_used","status","skip_reason","state","gpcr_class","receptor",
               "W6.48_used_BW","W6.48_used_resname",
               "alt_lock_used_3.49","alt_lock_used_6.30","alt_lock_used_bw_3.49","alt_lock_used_bw_6.30",
               "network_used_3.39","network_used_6.40","network_used_bw_3.39","network_used_bw_6.40",
               "dry_used_3.49","dry_used_3.50","dry_used_bw_3.49","dry_used_bw_3.50"}
    forced = [x.strip() for x in args.force_metrics.split(",") if x.strip()]
    num_cols=[]
    for col in forced if forced else [c for c in df.columns if c not in blocked]:
        if col not in df.columns: continue
        s = pd.to_numeric(df[col], errors="coerce")
        if forced or (np.isfinite(s).mean()>=0.8):
            df[col]=s; num_cols.append(col)
    if not num_cols:
        raise ValueError("No numeric metrics detected (use --force-metrics).")

    classes = sorted(df["gpcr_class"].unique())
    plot_raw = [s.strip() for s in args.classes_to_plot.split(",") if s.strip()]
    plot_trim = [trim_class(s) for s in plot_raw]
    plot_classes_full = [c for c in classes if trim_class(c) in plot_trim]

    xlw = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    pd.DataFrame({
        "note":[
            "log2FC is computed per receptor: log2((active+eps)/(inactive+eps)).",
            f"If active<=0 or inactive<=0 → set to NaN (eps={args.eps:g}).",
            f"Outlier filtering per class & state on raw values (|x-μ|>z*SD): z={args.outlier_z}",
            f"Plot significance source: {args.sig_source} (tukey=HSD p; bh=BH-FDR of Tukey p within metric)."
        ]
    }).to_excel(xlw, sheet_name="Legend", index=False)

    base = df[["gpcr_class","receptor","state"] + num_cols].copy()

    global_tukey_rows = []
    summary_rows = []

    for metric in num_cols:
        piv = (base.pivot_table(index=["gpcr_class","receptor"], columns="state",
                                values=metric, aggfunc="mean", observed=True)
                    .reset_index())

        # raw outliers by class/state
        keep = np.ones(len(piv), dtype=bool)
        if args.outlier_z>0:
            for cls, sub in piv.groupby("gpcr_class", observed=True):
                sub_ix = sub.set_index("receptor")
                oa, oi = mark_outliers_by_state(sub_ix[["active","inactive"]], args.outlier_z)
                if oa or oi:
                    bad = sub["receptor"].isin(set(oa)|set(oi))
                    keep[sub.index] = ~bad
        piv = piv.loc[keep].copy()

        # per-receptor log2FC
        eps = float(args.eps)
        def safe_log2(a, i):
            if not np.isfinite(a) or not np.isfinite(i): return np.nan
            if a<=0 or i<=0: return np.nan
            return float(np.log2((a+eps)/(i+eps)))
        piv["log2FC"] = [safe_log2(a,i) for a,i in zip(piv.get("active"), piv.get("inactive"))]

        # by class
        classes_here = sorted(piv["gpcr_class"].dropna().unique())
        class2vals = {c: piv.loc[piv["gpcr_class"]==c, "log2FC"].dropna().values for c in classes_here}

        # ANOVA + Tukey
        p_anova, tukey_df = anova_and_tukey(class2vals)

        # Tukey pairs and p-values
        pairs_all = list(itertools.combinations(classes_here, 2))
        tmap = {(r["group1"], r["group2"]): r["p_adj"] for _, r in tukey_df.iterrows()}
        def tukey_p(c1,c2): return tmap.get((c1,c2), tmap.get((c2,c1), np.nan))
        pair_p = [tukey_p(c1,c2) for (c1,c2) in pairs_all]
        q_within = bh_qvalues(pair_p)

        # full matrices (all classes present for the metric)
        Mq_full = pd.DataFrame(np.nan, index=classes_here, columns=classes_here, dtype=float)
        Mp_full = pd.DataFrame(np.nan, index=classes_here, columns=classes_here, dtype=float)
        for (c1,c2), p, q in zip(pairs_all, pair_p, q_within):
            Mp_full.loc[c1,c2] = Mp_full.loc[c2,c1] = p
            Mq_full.loc[c1,c2] = Mq_full.loc[c2,c1] = q
        np.fill_diagonal(Mq_full.values, np.nan); np.fill_diagonal(Mp_full.values, np.nan)

        # versions with trimmed labels (for Excel)
        Mq_trim = Mq_full.rename(index=trim_class, columns=trim_class)
        Mp_trim = Mp_full.rename(index=trim_class, columns=trim_class)

        # Excel sheets per metric
        Mp_trim.to_excel(xlw, sheet_name=excel_sheet_name(metric, "TukeyP"))
        Mq_trim.to_excel(xlw, sheet_name=excel_sheet_name(metric, "BHqWithin"))
        pd.DataFrame([{"metric":metric, "ANOVA_p":p_anova}]).to_excel(
            xlw, sheet_name=excel_sheet_name(metric, "ANOVA"), index=False)

        # lookup for figures
        LOOK = Mp_trim if args.sig_source=="tukey" else Mq_trim
        sig_lookup = {(ri, cj): float(LOOK.loc[ri, cj])
                      for ri in LOOK.index for cj in LOOK.columns}

        # summary means/CI by class (for potential table)
        for cls in classes_here:
            vals = class2vals.get(cls, np.array([], float))
            vals = vals[np.isfinite(vals)]; n=len(vals)
            if n==0:
                summary_rows.append({"metric":metric,"class":trim_class(cls),
                                     "mean":np.nan,"ci95_lo":np.nan,"ci95_hi":np.nan,"n":0})
            else:
                m=float(np.mean(vals))
                if n>=2:
                    se=float(np.std(vals, ddof=1)/np.sqrt(n)); tc=tcrit_95(n)
                    lo,hi=m-tc*se, m+tc*se
                else:
                    lo=hi=np.nan
                summary_rows.append({"metric":metric,"class":trim_class(cls),
                                     "mean":m,"ci95_lo":lo,"ci95_hi":hi,"n":n})

        # optional global BH (across all metrics later)
        for (c1,c2), p in zip(pairs_all, pair_p):
            global_tukey_rows.append({"metric":metric,"class1":c1,"class2":c2,"p_adj":p})

        # ===== figure (requested class subset) =====
        if plot_classes_full:
            use_full = [c for c in plot_classes_full if c in classes_here]
            if len(use_full) >= 2:
                # --- square figure (no title, no error bars) ---
                fig, ax = plt.subplots(figsize=(5,5))
                xs = np.arange(len(use_full))
                ybar=[]; ylo=[]; yhi=[]
                for idx, cls in enumerate(use_full):
                    vals = class2vals.get(cls, np.array([], float))
                    vals = vals[np.isfinite(vals)]; n=len(vals)
                    if n>0:
                        ax.scatter(xs[idx]+jitter(n,0.10,seed=idx+21), vals,
                                   s=20, color=BLUE, alpha=0.15, edgecolor="none", zorder=3)
                        m=float(np.mean(vals)); ybar.append(m)
                        if n>=2:
                            se=float(np.std(vals, ddof=1)/np.sqrt(n)); tc=tcrit_95(n)
                            ylo.append(m-tc*se); yhi.append(m+tc*se)
                        else:
                            ylo.append(np.nan); yhi.append(np.nan)
                    else:
                        ybar.append(np.nan); ylo.append(np.nan); yhi.append(np.nan)
                ybar=np.array(ybar,float); ylo=np.array(ylo,float); yhi=np.array(yhi,float)

                # mean bars (NO error bars)
                ax.bar(xs, ybar, color=BLUE, alpha=0.45, zorder=2)

                xt = [trim_class(c) for c in use_full]

                # pairs used in the plot (for optional debugging)
                used_pairs=[]
                for i in range(len(xt)):
                    for j in range(i+1, len(xt)):
                        val = sig_lookup.get((xt[i], xt[j]), np.nan)
                        used_pairs.append((xt[i], xt[j], val))
                if args.debug_pairs:
                    print(f"\n[DEBUG] {metric} — significance source: {args.sig_source}")
                    for a,b,v in used_pairs:
                        print(f"    {a} vs {b}: {v:.6g}")
                    pd.DataFrame(used_pairs, columns=["class1","class2","value"]).to_csv(
                        out/f"pairs_used_for_plot__{sanitize_filename(metric)}__{args.sig_source}.csv", index=False)

                # brackets (only if value < 0.05)
                sig=[]
                for i in range(len(xt)):
                    for j in range(i+1, len(xt)):
                        v = sig_lookup.get((xt[i], xt[j]), np.nan)
                        if np.isfinite(v) and v < 0.05:
                            sig.append((i,j,v))
                if sig:
                    ymax_candidates = [np.nanmax(ybar)] if np.any(np.isfinite(ybar)) else [0.0]
                    if np.any(np.isfinite(yhi)): ymax_candidates.append(np.nanmax(yhi))
                    ymax = np.nanmax(ymax_candidates)
                    span = (np.nanmax(ybar)-np.nanmin(ybar)) if np.any(np.isfinite(ybar)) else 1.0
                    step = 0.10*(abs(span) if span!=0 else 1.0)
                    cur  = ymax + 0.12*(abs(span) if span!=0 else 1.0)
                    def stars(v): return "***" if v<1e-3 else ("**" if v<1e-2 else ("*" if v<5e-2 else ""))
                    for (i,j,v) in sorted(sig, key=lambda t:(t[1]-t[0], t[0])):
                        add_bracket(ax, xs[i], xs[j], cur, stars(v), h=0.04)
                        cur += step

                ax.axhline(0, lw=0.9, ls="--", color=EDGE, alpha=0.6)
                ax.set_xticks(xs); ax.set_xticklabels(xt)
                ax.set_ylabel("log2(FC) active / inactive")
                # No title (as requested)
                ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
                savefig_all(fig, figs_dir / sanitize_filename(metric))
                plt.close(fig)

    # global BH across all metrics (optional, informative)
    if global_tukey_rows:
        gt = pd.DataFrame(global_tukey_rows)
        gt["q_global"] = bh_qvalues(gt["p_adj"].values)
        for metric, sub in gt.groupby("metric"):
            cl = sorted(set(sub["class1"]).union(set(sub["class2"])))
            Mq = pd.DataFrame(np.nan, index=cl, columns=cl, dtype=float)
            for _, r in sub.iterrows():
                Mq.loc[r["class1"], r["class2"]] = r["q_global"]
                Mq.loc[r["class2"], r["class1"]] = r["q_global"]
            np.fill_diagonal(Mq.values, np.nan)
            Mq.rename(index=trim_class, columns=trim_class).to_excel(
                xlw, sheet_name=excel_sheet_name(metric, "BHqGlobal"))

    xlw.close()
    print(f"[OK] Excel  → {excel_path}")
    print(f"[OK] Figures → {figs_dir}")

if __name__ == "__main__":
    main()

