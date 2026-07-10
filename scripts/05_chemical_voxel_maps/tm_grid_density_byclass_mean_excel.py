#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tm_grid_density_byclass_mean_excel.py

Objetivo:
- Promedio por voxel de cada característica química (hydrophobic/positive/negative/polar/special),
  separado por CLASE de receptor (según familias.txt), promediando a través de los PDBs de esa clase.
- Alineación PCA global (PC1=mayor varianza) -> grid 30,15,15 en (PC1,PC2,PC3).
- Excel con hoja 'meta' + una hoja por clase con mean_density_* (y sem_density_*).
- Un PDB por clase con el receptor de referencia (cadena R) y voxels (cadena V) con B=mean_density.
- PDB/PML de debug: voxels crudos del grid.
- Excel de **comparaciones por pares** (--xlsx-pairs): una hoja por combinación A vs B
  con p/q/efecto (log2OR) por voxel y química, y además:
    * top4_BW_A: los 4 residuos BW más frecuentes EN ESE VOXEL en la clase A (con cuentas)
    * top4_BW_B: idem para la clase B
  **NOVEDAD**: los nombres de pestaña y el campo 'pair' en meta_pairs se generan
  **eliminando cualquier texto entre paréntesis** del nombre de las clases.

Requisitos: numpy, pandas, openpyxl (o xlsxwriter)
"""

import os, re, sys, json, argparse, itertools, math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------- Config química ----------------------------------
HYDROPHOBIC = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","TYR","PRO"}
POSITIVE    = {"LYS","ARG","HIS"}
NEGATIVE    = {"ASP","GLU"}
POLAR       = {"SER","THR","ASN","GLN","CYS","TYR","HIS"}
SPECIAL     = {"GLY","PRO"}

CHEM_ORDER = ["hydrophobic","positive","negative","polar","special"]
CHEM_LABEL = {
    "hydrophobic": "Hidrofóbicos",
    "positive":    "Positivos",
    "negative":    "Negativos",
    "polar":       "Polares (sin carga)",
    "special":     "Especiales (Gly/Pro)",
}
CHEM_TO_RESN = {"hydrophobic":"VXH","positive":"VXP","negative":"VXN","polar":"VXL","special":"VXS"}
CHEM_TO_ELEM = {"hydrophobic":"C","positive":"N","negative":"O","polar":"S","special":"P"}

def classify_residue(resname: str) -> str:
    r = resname.upper()
    if r in SPECIAL:      return "special"
    if r in HYDROPHOBIC:  return "hydrophobic"
    if r in POSITIVE:     return "positive"
    if r in NEGATIVE:     return "negative"
    if r in POLAR:        return "polar"
    return "polar"

# ---------------------------- familias.txt ------------------------------------
def load_familias_map(fpath: Path) -> Dict[str, str]:
    mp = {}
    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for ln in fh:
            ln = ln.strip()
            if not ln: continue
            cols = ln.split("\t")
            if header is None and (("Class" in cols[-1]) or ("GPCRs" in cols[0])):
                header = cols; continue
            if len(cols) < 2: continue
            key = cols[0].strip().upper()
            val = cols[-1].strip()
            if key: mp[key] = val
    return mp

# ---------------------------- util prefijo ------------------------------------
def extract_prefix_from_filename(p: Path) -> str:
    m = re.match(r"^([A-Za-z0-9]+)", p.stem)
    return (m.group(1) if m else p.stem).upper()

def map_prefix_to_class(prefix: str, mp: Dict[str,str]) -> str|None:
    if prefix in mp: return mp[prefix]
    c1 = [k for k in mp if k.startswith(prefix)]
    if c1: c1.sort(key=len); return mp[c1[0]]
    c2 = [k for k in mp if prefix.startswith(k)]
    if c2: c2.sort(key=len, reverse=True); return mp[c2[0]]
    return None

# ---------------------------- Parseo PDB / BW ---------------------------------
BW_RGXS = [re.compile(r"\b([1-7])[x\.](\d{2})\b"), re.compile(r"\b([1-7])x(\d{2})\b")]
def parse_bw_from_token10(tok: str):
    if not tok: return None, None
    for rgx in BW_RGXS:
        m = rgx.search(tok)
        if m: return int(m.group(1)), f"{m.group(1)}x{m.group(2)}"
    return None, None

def read_tm_ca_points_by_chem(pdb_path: Path) -> Dict[str, np.ndarray]:
    """
    Devuelve {chem: (N,3)} con coords crudas (no-PCA) de CA en TMs (1..7) según BW en token10.
    """
    buckets = {c: [] for c in CHEM_ORDER}
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"): continue
            parts = line.split()
            tok10 = parts[9] if len(parts) >= 10 else ""
            tmh, _ = parse_bw_from_token10(tok10)
            if tmh is None or not (1 <= tmh <= 7): continue
            if line[12:16].strip() != "CA": continue
            resname = line[17:20].strip().upper()
            chain   = line[21].strip()
            resseq  = line[22:26].strip()
            icode   = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen: continue
            seen.add(key)
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                try: x = float(parts[6]); y = float(parts[7]); z = float(parts[8])
                except Exception: continue
            buckets[classify_residue(resname)].append((x,y,z))
    return {c: (np.asarray(v,float) if v else np.zeros((0,3))) for c,v in buckets.items()}

def read_tm_ca_points_with_bw(pdb_path: Path):
    """
    Devuelve lista de (x,y,z,bw_str) para CA en TMs(1..7) con notación BW (p.ej. '3x32'),
    deduplicando por (chain, resseq, icode). Coordenadas crudas (no PCA).
    """
    out = []
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            tok10 = parts[9] if len(parts) >= 10 else ""
            tmh, bw = parse_bw_from_token10(tok10)
            if tmh is None or not (1 <= tmh <= 7) or not bw:
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode  = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                try:
                    x = float(parts[6]); y = float(parts[7]); z = float(parts[8])
                except Exception:
                    continue
            out.append((x,y,z,bw))
    return out

# ---------------------------- PCA ---------------------------------------------
def pca_fit(points: np.ndarray):
    X = points - points.mean(axis=0, keepdims=True)
    _,_,Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt.T
    mu = points.mean(axis=0)
    return mu, R

def apply_affine(X: np.ndarray, mu: np.ndarray, R: np.ndarray):
    return (X - mu) @ R

# ---------------------------- Grid --------------------------------------------
def make_grid_edges(mins: np.ndarray, maxs: np.ndarray):
    bins = [30,15,15]  # PC1, PC2, PC3
    edges = [np.linspace(mins[i], maxs[i], bins[i]+1) for i in range(3)]
    return edges, bins

def histogramdd(points: np.ndarray, edges):
    if points.size == 0:
        return np.zeros((len(edges[0])-1, len(edges[1])-1, len(edges[2])-1), float)
    H, _ = np.histogramdd(points, bins=edges)
    return H

def sanitize_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()

# ---------------------------- PDB helpers --------------------------------------
def pdb_line(record, serial, name, resn, chain, resseq, x,y,z, occ, b, element):
    name4 = f"{name:>4}" if len(name.strip())==4 else f" {name:<3}"
    return (f"{record:<6}{serial:>5} {name4} {resn:>3} {chain:1}"
            f"{resseq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{occ:>6.2f}{b:>6.2f}          {element:>2}\n")

def transform_pdb_and_write(in_pdb: Path, out_handle, mu: np.ndarray, R: np.ndarray, start_serial=1, chain_override=None):
    serial = start_serial
    with open(in_pdb, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM","HETATM","ANISOU","TER","MODEL","ENDMDL","REMARK")):
                continue
            if line.startswith(("ATOM","HETATM")):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    x,y,z = apply_affine(np.array([[x,y,z]]), mu, R)[0]
                    new = list(line)
                    xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    new[30:54] = list(xyz)
                    new[6:11] = list(f"{serial:5d}")
                    if chain_override is not None:
                        new[21] = chain_override[0]
                    out_handle.write("".join(new))
                    serial += 1
                except Exception:
                    out_handle.write(line)
            else:
                out_handle.write(line)
    return serial

# ---------------------------- Estadística --------------------------------------
def bh_fdr(pvals: np.ndarray):
    """Benjamini–Hochberg FDR (array arbitrario)."""
    x = np.asarray(pvals, float).ravel()
    n = x.size
    order = np.argsort(x)
    ranks = np.empty(n, int); ranks[order] = np.arange(1, n+1)
    q = x * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty_like(x); out[order] = q_sorted
    return np.clip(out.reshape(pvals.shape), 0.0, 1.0)

def chi2_yates_2x2_p(a, b, c, d):
    """
    Chi² 2x2 con corrección de Yates.
    Tabla:
        [a b]
        [c d]
    Devuelve p-valor (cola superior) con df=1: p = erfc( sqrt(chi2/2) ).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)

    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    n  = r1 + r2

    with np.errstate(divide="ignore", invalid="ignore"):
        Ea = r1 * c1 / n
        Eb = r1 * c2 / n
        Ec = r2 * c1 / n
        Ed = r2 * c2 / n

    def comp(O, E):
        diff = np.abs(O - E) - 0.5
        diff = np.maximum(diff, 0.0)
        out = np.zeros_like(E, dtype=float)
        mask = (E > 0)
        out[mask] = (diff[mask] * diff[mask]) / E[mask]
        return out

    chi2 = comp(a,Ea) + comp(b,Eb) + comp(c,Ec) + comp(d,Ed)
    chi2 = np.where(n > 0, chi2, 0.0)

    s = np.sqrt(np.maximum(chi2, 0.0) * 0.5)
    # math.erfc vectorizado para compatibilidad amplia
    p = np.vectorize(lambda z: math.erfc(float(z)))(s)
    p = np.where(np.isfinite(p), p, 1.0)
    return p

def log2_or_haldane(a, b, c, d, eps=0.5):
    """log2(odds ratio) con corrección de Haldane (+eps). Positivo ⇒ enriquecido en fila 1 (A)."""
    num = (a + eps) * (d + eps)
    den = (b + eps) * (c + eps)
    return np.log2(num / den)

# ---------------------------- Voxel helpers ------------------------------------
def voxel_index_xyz(x, y, z, edges):
    """Devuelve (ix,iy,iz) o (None,None,None) si cae fuera del grid (incluye borde derecho por isclose)."""
    ex, ey, ez = edges
    nx, ny, nz = len(ex)-1, len(ey)-1, len(ez)-1
    ix = np.searchsorted(ex, x, side="right") - 1
    iy = np.searchsorted(ey, y, side="right") - 1
    iz = np.searchsorted(ez, z, side="right") - 1
    # incluir exactamente el borde máximo
    if ix == nx and np.isclose(x, ex[-1]): ix = nx-1
    if iy == ny and np.isclose(y, ey[-1]): iy = ny-1
    if iz == nz and np.isclose(z, ez[-1]): iz = nz-1
    if ix<0 or ix>=nx or iy<0 or iy>=ny or iz<0 or iz>=nz:
        return None, None, None
    return int(ix), int(iy), int(iz)

def flat_index(ix, iy, iz, nx, ny, nz):
    return (ix*ny + iy)*nz + iz

# ---------------------------- Helpers para nombres de pestañas -----------------
def strip_parens(name: str) -> str:
    """Quita cualquier ' ( ... )' del nombre y colapsa espacios."""
    s = re.sub(r"\s*\([^)]*\)", "", str(name)).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

def make_sheet_name(base_title: str, used: set) -> str:
    """
    Genera un nombre de hoja válido (<=31 chars, alfanumérico+guiones bajos),
    añadiendo sufijos _2, _3, ... si se repite.
    """
    safe = re.sub(r"[^A-Za-z0-9]+", "_", base_title).strip("_")
    safe = safe[:31] if len(safe) > 31 else safe
    name = safe or "Sheet"
    k = 2
    while name in used:
        suffix = f"_{k}"
        name = (safe[: max(0, 31 - len(suffix))] + suffix) or f"Sheet_{k}"
        k += 1
    used.add(name)
    return name

# ---------------------------- Main --------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Promedio por voxel de químicas por clase (PCA grid) + Excel + PDB por clase + DEBUG grid + pairwise Excel con top4 BW por voxel")
    ap.add_argument("--indir", required=True, help="Carpeta con PDBs alineados")
    ap.add_argument("--pattern", default="*.pdb", help="Patrón glob de PDBs")
    ap.add_argument("--familias", required=True, help="familias.txt (TSV) con la clase en la última columna")

    ap.add_argument("--xlsx", required=True, help="Excel de salida (by-class)")
    ap.add_argument("--xlsx-pairs", help="Excel adicional con comparaciones por pares (una hoja por A vs B)")
    ap.add_argument("--ref-pdb", required=True, help="PDB de referencia a incluir (transformado a PCA) en cada PDB por clase")
    ap.add_argument("--pdb-per-class-dir", required=True, help="Carpeta donde escribir un PDB por clase (mean_density)")

    ap.add_argument("--pdb-chem", nargs="*", choices=CHEM_ORDER, default=CHEM_ORDER, help="Químicas a exportar en PDB")
    ap.add_argument("--pdb-min-mean-density", type=float, default=0.0, help="Umbral de mean_density para voxels")
    ap.add_argument("--pdb-topk-per-chem", type=int, default=0, help="Top-K voxels por química (0 = sin límite)")
    ap.add_argument("--voxel-name", default="VOX", help="Nombre de átomo para voxels")

    # Debug grid
    ap.add_argument("--debug-voxels-pdb", help="PDB con receptor ref (cadena R) + centros del grid (cadena V, resn VOX) sin propiedades químicas")
    ap.add_argument("--debug-voxels-pml", help="Ruta PML opcional para colorear voxels por PC1 (B-factor 0..1)")
    ap.add_argument("--debug-voxels-vdw", type=float, default=0.8, help="Radio de esfera (vdw) para los voxels de debug")

    args = ap.parse_args()

    indir = Path(args.indir); pdbs = sorted(indir.rglob(args.pattern))
    if not pdbs:
        print("[ERROR] No se encontraron PDBs en --indir con --pattern."); sys.exit(1)

    code2class = load_familias_map(Path(args.familias))
    if not code2class:
        print("[ERROR] familias.txt vacío/incorrecto."); sys.exit(2)

    # ---------- Agrupar por clase y recolectar puntos crudos ----------
    group_files: Dict[str, List[Path]] = {}
    raw_points_by_file_and_chem: Dict[Path, Dict[str, np.ndarray]] = {}
    raw_points_with_bw_by_file: Dict[Path, List[Tuple[float,float,float,str]]] = {}
    all_points = []
    skipped = []

    for p in pdbs:
        pref = extract_prefix_from_filename(p)
        rclass = map_prefix_to_class(pref, code2class)
        if rclass is None:
            skipped.append(p.name); continue
        group_files.setdefault(rclass, []).append(p)

        # Por química (para densidades/conteos)
        buckets = read_tm_ca_points_by_chem(p)
        raw_points_by_file_and_chem[p] = buckets
        for chem in CHEM_ORDER:
            arr = buckets[chem]
            if arr.size: all_points.append(arr)

        # Con BW para contadores por voxel
        raw_points_with_bw_by_file[p] = read_tm_ca_points_with_bw(p)

    if not all_points:
        print("[ERROR] No se detectaron CA en TMs (1..7)."); sys.exit(3)

    all_points_arr = np.vstack(all_points)  # (M,3)
    mu, R = pca_fit(all_points_arr)

    # ---------- Transformar a PCA por archivo x química ----------
    pc_points_by_file_and_chem: Dict[Path, Dict[str, np.ndarray]] = {}
    for p, buckets in raw_points_by_file_and_chem.items():
        pc_map = {}
        for chem in CHEM_ORDER:
            arr = buckets[chem]
            pc_map[chem] = apply_affine(arr, mu, R) if arr.size else arr
        pc_points_by_file_and_chem[p] = pc_map

    # ---------- Grid 30/15/15 en PCA ----------
    all_pc = apply_affine(all_points_arr, mu, R)
    mins = np.min(all_pc, axis=0); maxs = np.max(all_pc, axis=0)
    edges, bins = make_grid_edges(mins, maxs)
    nx, ny, nz = len(edges[0])-1, len(edges[1])-1, len(edges[2])-1
    dx = np.diff(edges[0])[0]; dy = np.diff(edges[1])[0]; dz = np.diff(edges[2])[0]
    cell_volume = float(dx*dy*dz)

    # ---------- Promedio por clase + CONTEOS totales por clase ----------
    mean_density: Dict[Tuple[str,str], np.ndarray] = {}
    sem_density:  Dict[Tuple[str,str], np.ndarray] = {}
    total_counts: Dict[Tuple[str,str], np.ndarray] = {}  # para pairwise
    meta_groups = []

    for gname, flist in group_files.items():
        gkey = sanitize_key(gname)
        meta_groups.append({"name": gname, "key": gkey, "num_pdbs": len(flist)})
        for chem in CHEM_ORDER:
            per_file_counts = []
            for p in flist:
                pts = pc_points_by_file_and_chem[p][chem]
                H = histogramdd(pts, edges)     # CONTEOS (enteros)
                per_file_counts.append(H)
            stack_cnt = np.stack(per_file_counts, axis=0)          # (n_files, nx,ny,nz)
            total_counts[(gkey,chem)] = np.sum(stack_cnt, axis=0)  # (nx,ny,nz)
            stack_den = stack_cnt / cell_volume                     # densidades por archivo
            mean = np.mean(stack_den, axis=0)
            mean_density[(gkey,chem)] = mean
            if len(flist) > 1:
                sem = np.std(stack_den, axis=0, ddof=1) / np.sqrt(len(flist))
            else:
                sem = np.zeros_like(mean)
            sem_density[(gkey,chem)] = sem

    # ---------- Contadores BW POR VOXEL y POR CLASE ----------
    Nvox = nx*ny*nz
    bw_vox_by_key: Dict[str, List[Counter]] = {}
    for gname, flist in group_files.items():
        gkey = sanitize_key(gname)
        bw_vox_by_key[gkey] = [Counter() for _ in range(Nvox)]
        for p in flist:
            pts_bw = raw_points_with_bw_by_file[p]  # lista (x,y,z,bw)
            if not pts_bw:
                continue
            coords = np.array([(x,y,z) for (x,y,z,_) in pts_bw], dtype=float)
            coords_pc = apply_affine(coords, mu, R)
            for (xpc,ypc,zpc), (_,_,_,bw) in zip(coords_pc, pts_bw):
                ix,iy,iz = voxel_index_xyz(xpc, ypc, zpc, edges)
                if ix is None:
                    continue
                f = flat_index(ix,iy,iz, nx,ny,nz)
                bw_vox_by_key[gkey][f][bw] += 1

    # ---------- Excel by-class ----------
    xlsx = Path(args.xlsx)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        meta = {
            "num_total_pdbs": sum(len(v) for v in group_files.values()),
            "num_groups": len(group_files),
            "bins": bins,
            "cell_volume": cell_volume,
            "edges_x_min_max": (float(edges[0][0]), float(edges[0][-1])),
            "edges_y_min_max": (float(edges[1][0]), float(edges[1][-1])),
            "edges_z_min_max": (float(edges[2][0]), float(edges[2][-1])),
            "PCA_mu": mu.tolist(),
            "PCA_Rows_are_PC_axes": R.T.tolist(),
            "CHEM_ORDER": CHEM_ORDER,
            "CHEM_LABEL": CHEM_LABEL,
            "groups": meta_groups,
        }
        pd.DataFrame({
            "key": list(meta.keys()),
            "value": [json.dumps(v) if isinstance(v,(list,tuple,dict)) else v for v in meta.values()]
        }).to_excel(xw, sheet_name="meta", index=False)

        for mg in meta_groups:
            gkey = mg["key"]; gname = mg["name"]
            rows = []
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        x_min, x_max = edges[0][ix], edges[0][ix+1]
                        y_min, y_max = edges[1][iy], edges[1][iy+1]
                        z_min, z_max = edges[2][iz], edges[2][iz+1]
                        cx_c = 0.5*(x_min+x_max); cy_c = 0.5*(y_min+y_max); cz_c = 0.5*(z_min+z_max)
                        row = {
                            "ix": ix, "iy": iy, "iz": iz,
                            "x_min": x_min, "x_max": x_max,
                            "y_min": y_min, "y_max": y_max,
                            "z_min": z_min, "z_max": z_max,
                            "cx": cx_c, "cy": cy_c, "cz": cz_c,
                        }
                        for chem in CHEM_ORDER:
                            row[f"mean_density_{chem}"] = float(mean_density[(gkey,chem)][ix,iy,iz])
                            row[f"sem_density_{chem}"]  = float(sem_density[(gkey,chem)][ix,iy,iz])
                        rows.append(row)
            df = pd.DataFrame(rows)
            sname = gname[:31] if gname else gkey[:31]
            df.to_excel(xw, sheet_name=sname, index=False)

    # ---------- (Opcional) Excel por pares ----------
    if args.xlsx_pairs:
        pairs_path = Path(args.xlsx_pairs)

        # Centros del grid (para columnas de posición)
        cx = 0.5*(edges[0][:-1] + edges[0][1:])
        cy = 0.5*(edges[1][:-1] + edges[1][1:])
        cz = 0.5*(edges[2][:-1] + edges[2][1:])
        name2key = {mg["name"]: mg["key"] for mg in meta_groups}

        # Nombres "display" SIN paréntesis
        raw_class_names = [mg["name"] for mg in meta_groups]
        disp_by_raw = {nm: strip_parens(nm) for nm in raw_class_names}

        # Preparar pares y asignar nombres de hoja únicos
        used_sheet_names = set()
        pair_rows = []   # para meta_pairs
        pair_defs = []   # (A_raw, B_raw, A_disp, B_disp, sheet_name)

        for (A_raw, B_raw) in itertools.combinations(raw_class_names, 2):
            A_disp = disp_by_raw[A_raw]
            B_disp = disp_by_raw[B_raw]
            pair_title = f"{A_disp}__vs__{B_disp}"          # <-- sin paréntesis
            sheet_name = make_sheet_name(pair_title, used_sheet_names)
            pair_rows.append({
                "pair": pair_title,                         # <-- sin paréntesis
                "pair_full": f"{A_raw}__vs__{B_raw}",       # referencia informativa (con paréntesis)
                "sheet": sheet_name
            })
            pair_defs.append((A_raw, B_raw, A_disp, B_disp, sheet_name))

        with pd.ExcelWriter(pairs_path, engine="openpyxl") as xw2:
            # Índice de combinaciones
            pd.DataFrame(pair_rows).to_excel(xw2, sheet_name="meta_pairs", index=False)

            # Hojas por par
            for (A_raw, B_raw, A_disp, B_disp, sname) in pair_defs:
                gA, gB = name2key[A_raw], name2key[B_raw]

                # Base dataframe (una fila por voxel)
                rows = []
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            rows.append({
                                "ix": ix, "iy": iy, "iz": iz,
                                "x_min": edges[0][ix], "x_max": edges[0][ix+1],
                                "y_min": edges[1][iy], "y_max": edges[1][iy+1],
                                "z_min": edges[2][iz], "z_max": edges[2][iz+1],
                                "cx": cx[ix], "cy": cy[iy], "cz": cz[iz],
                            })
                df = pd.DataFrame(rows)

                # Estadística por química (2x2 con Yates) + FDR
                for chem in CHEM_ORDER:
                    A_cnt = total_counts[(gA,chem)].ravel()
                    B_cnt = total_counts[(gB,chem)].ravel()
                    A_tot = float(A_cnt.sum()); B_tot = float(B_cnt.sum())
                    A_in  = A_cnt
                    B_in  = B_cnt
                    A_out = A_tot - A_in
                    B_out = B_tot - B_in

                    pvals = chi2_yates_2x2_p(A_in, A_out, B_in, B_out)
                    qvals = bh_fdr(pvals)
                    log2or = log2_or_haldane(A_in, A_out, B_in, B_out, eps=0.5)

                    df[f"pval_{chem}"]   = pvals
                    df[f"qval_{chem}"]   = qvals
                    df[f"log2OR_{chem}"] = log2or

                # top4 BW por voxel para A y B
                def top4_str(counter: Counter):
                    if not counter:
                        return ""
                    return ", ".join([f"{bw}({n})" for bw, n in counter.most_common(4)])

                topA_list, topB_list = [], []
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            f = flat_index(ix, iy, iz, nx, ny, nz)
                            cntA = bw_vox_by_key.get(gA, [Counter()]* (nx*ny*nz))[f]
                            cntB = bw_vox_by_key.get(gB, [Counter()]* (nx*ny*nz))[f]
                            topA_list.append(top4_str(cntA))
                            topB_list.append(top4_str(cntB))

                df["top4_BW_A"] = topA_list
                df["top4_BW_B"] = topB_list

                # Escribir hoja con nombre sin paréntesis
                df.to_excel(xw2, sheet_name=sname, index=False)

        print(f"[OK] Excel (pairwise) → {pairs_path.resolve()}")

    # ---------- PDB por clase: receptor ref + voxels (B = mean_density) ----------
    outdir = Path(args.pdb_per_class_dir); outdir.mkdir(parents=True, exist_ok=True)
    cx_c = 0.5*(edges[0][:-1] + edges[0][1:])
    cy_c = 0.5*(edges[1][:-1] + edges[1][1:])
    cz_c = 0.5*(edges[2][:-1] + edges[2][1:])
    CX, CY, CZ = np.meshgrid(cx_c, cy_c, cz_c, indexing="ij")

    # ---------- Debug grid ----------
    if args.debug_voxels_pdb:
        dbg_path = Path(args.debug_voxels_pdb)
        serial = 1
        with open(dbg_path, "w") as fo:
            fo.write("REMARK  DEBUG: voxel centers only (no chemistry). PCA frame (PC1,PC2,PC3).\n")
            fo.write(f"REMARK  Grid bins: {bins} (PC1 has 30 bins). Chain R=reference receptor, V=voxels.\n")
            serial = transform_pdb_and_write(Path(args.ref_pdb), fo, mu, R, start_serial=serial, chain_override="R")
            Xc, Yc, Zc = CX.ravel(), CY.ravel(), CZ.ravel()
            xmin, xmax = float(cx_c.min()), float(cx_c.max())
            rng = xmax - xmin if (xmax - xmin) > 1e-12 else 1.0
            Bdbg = (Xc - xmin) / rng
            resseq = 9999
            for x,y,z,b in zip(Xc, Yc, Zc, Bdbg):
                fo.write(pdb_line("HETATM", serial, "VOX", "VOX", "V", resseq,
                                  float(x), float(y), float(z), 1.00, float(b), "C"))
                serial += 1
            fo.write("END\n")
        print(f"[OK] Debug voxels PDB → {dbg_path.resolve()}")
        if args.debug_voxels_pml:
            pmlp = Path(args.debug_voxels_pml)
            with open(pmlp, "w") as f:
                f.write(f'load {dbg_path.name}\n')
                f.write('hide everything, all\n')
                f.write('select rec, chain R\n')
                f.write('select vox, chain V\n')
                f.write('show cartoon, rec\n')
                f.write('set cartoon_transparency, 0.35, rec\n')
                f.write('color grey70, rec\n')
                f.write('show spheres, vox\n')
                f.write('ramp_new ramp_pc1, vox, [0.0, 1.0], [blue, white, red]\n')
                f.write('color ramp_pc1, vox\n')
                f.write(f'alter vox, vdw = {args.debug_voxels_vdw}\n')
                f.write('rebuild\n')
                f.write('bg_color white\n')
                f.write('orient\n')
            print(f"[OK] Debug PML → {pmlp.resolve()}")

    # ---------- PDBs por clase (mean_density) ----------
    for mg in meta_groups:
        gkey = mg["key"]; gname = mg["name"]
        out_pdb = outdir / f"voxels_mean_{sanitize_key(gname)}.pdb"
        serial = 1
        with open(out_pdb, "w") as fo:
            fo.write("REMARK  Generated by tm_grid_density_byclass_mean_excel.py\n")
            fo.write("REMARK  Coordinates are in PCA frame (PC1,PC2,PC3). Grid bins: [30,15,15]\n")
            serial = transform_pdb_and_write(Path(args.ref_pdb), fo, mu, R, start_serial=serial, chain_override="R")

            # Inclusión por química
            per_chem_flat = {}
            for chem in args.pdb_chem:
                arr = mean_density[(gkey,chem)]
                flat = arr.ravel()
                cand = np.flatnonzero(flat >= args.pdb_min_mean_density)
                if args.pdb_topk_per_chem and cand.size > args.pdb_topk_per_chem:
                    part = np.argpartition(flat[cand], -args.pdb_topk_per_chem)[-args.pdb_topk_per_chem:]
                    cand = cand[part]
                per_chem_flat[chem] = cand

            # Escribir voxels seleccionados
            resseq = 9999
            CXf, CYf, CZf = CX.ravel(), CY.ravel(), CZ.ravel()
            for chem in args.pdb_chem:
                resn = CHEM_TO_RESN.get(chem, "VOX"); elem = CHEM_TO_ELEM.get(chem, "C")
                arr = mean_density[(gkey,chem)].ravel()
                for idx in per_chem_flat[chem]:
                    x,y,z = float(CXf[idx]), float(CYf[idx]), float(CZf[idx])
                    b = float(arr[idx])
                    fo.write(pdb_line("HETATM", serial, args.voxel_name, resn, "V", resseq, x,y,z, 1.00, b, elem))
                    serial += 1
            fo.write("END\n")

        print(f"[OK] PDB (clase) → {out_pdb}")

    print(f"[OK] Excel (by-class) → {xlsx.resolve()}")
    if skipped:
        print(f"[INFO] PDBs sin clase (omitidos): {len(skipped)}. Ejemplos: {', '.join(skipped[:8])}")

if __name__ == "__main__":
    main()

