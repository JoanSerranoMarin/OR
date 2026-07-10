#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairwise_xlsx_voxels_to_pdb.py

Lee un Excel 'grid_tm_pairwise.xlsx' (generado por tm_grid_density_byclass_mean_excel.py)
y exporta a PDB (+PML) los voxels significativos para una QUÍMICA y un PAR de clases.

- Entrada:
  * Hoja 'meta_pairs' con columnas 'pair' y 'sheet' (map de par -> nombre de pestaña)
  * Una hoja por par A__vs__B con columnas:
      ix,iy,iz, x_min,x_max,y_min,y_max,z_min,z_max, cx,cy,cz,
      pval_<chem>, qval_<chem>, log2OR_<chem>, ... (para cada química)

- Filtros:
  * q-value <= --q
  * |log2OR| >= --min-abs-effect
  * (opcional) --include-nonsig para incluir voxels no significativos (se ocultarán en el PML)

- Codificación PDB:
  * B-factor := log2OR (con signo; positivo = enriquecido en A; negativo = enriquecido en B)
  * Occupancy := 1 - qval
  * A>B  → resn --resn-pos (def: VXP), elem --elem-pos (def: O)
  * B>A  → resn --resn-neg (def: VXN), elem --elem-neg (def: N)
  * Átomo: --voxel-name (def: VOX), cadena V, resseq 9999

- PML:
  * Selecciones por elemento (A_enriquecido vs B_enriquecido)
  * Radio ∝ |B| con tope en p90(|B|)
  * Colores: rampas separadas (A: blanco→rojo; B: blanco→azul)

Requiere: numpy, pandas, openpyxl (o xlsxwriter para escribir PML junto al PDB)
"""

import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

CHEM_ORDER = ["hydrophobic","positive","negative","polar","special"]

def sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def pdb_line(record, serial, name, resn, chain, resseq, x,y,z, occ, b, element):
    name4 = f"{name:>4}" if len(name.strip())==4 else f" {name:<3}"
    return (f"{record:<6}{serial:>5} {name4} {resn:>3} {chain:1}"
            f"{resseq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{occ:>6.2f}{b:>6.2f}          {element:>2}\n")

def find_pair_sheet(xls: pd.ExcelFile, pair_name: str|None, a_name: str|None, b_name: str|None):
    """
    Devuelve (sheet_name, orientation) donde orientation=+1 si A__vs__B, -1 si B__vs__A (para flip del signo).
    Busca en 'meta_pairs'. Si no existe, intenta inferir por nombre de hoja.
    """
    # Normalizadores
    def norm(x): return sanitize(x) if x is not None else None

    # 1) Lee meta_pairs si existe
    if "meta_pairs" in xls.sheet_names:
        meta = pd.read_excel(xls, sheet_name="meta_pairs")
        meta = meta.dropna(subset=["sheet"])
        # Probar match directo por 'pair'
        if pair_name:
            pn = norm(pair_name)
            for _, row in meta.iterrows():
                if norm(str(row.get("pair",""))) == pn or norm(str(row.get("sheet",""))) == pn:
                    # orientación desde pair 'A__vs__B'
                    pair_txt = str(row.get("pair",""))
                    if "__vs__" in pair_txt:
                        A, B = pair_txt.split("__vs__", 1)
                        return str(row["sheet"]), +1  # ya viene como A vs B
                    return str(row["sheet"]), +1
        # Probar por a_name y b_name
        if a_name and b_name:
            an = norm(a_name); bn = norm(b_name)
            # Buscar fila cuyo 'pair' contenga ambos en ese orden
            for _, row in meta.iterrows():
                ptxt = str(row.get("pair",""))
                pnorm = norm(ptxt)
                if pnorm and "__vs__" in ptxt:
                    A, B = ptxt.split("__vs__", 1)
                    if norm(A) == an and norm(B) == bn:
                        return str(row["sheet"]), +1
                    if norm(A) == bn and norm(B) == an:
                        return str(row["sheet"]), -1
            # Si no, buscar cualquier sheet que contenga ambos nombres en cualquier orden
            for _, row in meta.iterrows():
                s = str(row.get("sheet",""))
                sn = norm(s)
                if sn and an in sn and bn in sn:
                    # inferir orientación desde 'pair' si está
                    ptxt = str(row.get("pair",""))
                    if "__vs__" in ptxt:
                        A, B = ptxt.split("__vs__", 1)
                        if norm(A)==an and norm(B)==bn: return s, +1
                        if norm(A)==bn and norm(B)==an: return s, -1
                    # por defecto +1
                    return s, +1

    # 2) Fallback: hoja cuyo nombre contenga ambos tokens
    targets = []
    if pair_name:
        targets.append(norm(pair_name))
    if a_name:
        targets.append(norm(a_name))
    if b_name:
        targets.append(norm(b_name))
    if targets:
        for s in xls.sheet_names:
            sn = norm(s)
            if all(t is None or (sn and t in sn) for t in targets):
                # orientación heurística usando "__vs__" si existe
                if "__vs__" in s:
                    A,B = s.split("__vs__", 1)
                    if a_name and b_name:
                        if norm(A)==norm(a_name) and norm(B)==norm(b_name): return s, +1
                        if norm(A)==norm(b_name) and norm(B)==norm(a_name): return s, -1
                return s, +1
    raise SystemExit("[ERROR] No pude localizar la pestaña del par. Revisa --pair o --a-name/--b-name o la hoja 'meta_pairs'.")

def main():
    ap = argparse.ArgumentParser(description="Exporta voxels significativos desde grid_tm_pairwise.xlsx a PDB+PML")
    ap.add_argument("--xlsx", required=True, help="Ruta a grid_tm_pairwise.xlsx")
    ap.add_argument("--chem", required=True, choices=CHEM_ORDER, help="Química a exportar")
    # Selección del par
    ap.add_argument("--pair", help="Nombre del par como aparece en 'meta_pairs' (ej. 'Class A__vs__Class O2')")
    ap.add_argument("--a-name", help="Nombre de la clase A (si no usas --pair)")
    ap.add_argument("--b-name", help="Nombre de la clase B (si no usas --pair)")
    # Filtros
    ap.add_argument("--q", type=float, default=0.05, help="Umbral FDR q-value")
    ap.add_argument("--min-abs-effect", type=float, default=0.0, help="Umbral de |log2OR|")
    ap.add_argument("--include-nonsig", action="store_true", help="Escribir también voxels no significativos")
    ap.add_argument("--max-points", type=int, default=250000, help="Límite duro de puntos (prioriza |efecto|)")
    # Salida y estilo
    ap.add_argument("--out", required=True, help="PDB de salida")
    ap.add_argument("--overlay-pdb", help="PDB de referencia a anteponer (debe estar en el mismo marco PCA)")
    ap.add_argument("--r-min", type=float, default=0.9)
    ap.add_argument("--r-max", type=float, default=2.2)
    ap.add_argument("--voxel-name", default="VOX")
    ap.add_argument("--resn-pos", default="VXP")
    ap.add_argument("--resn-neg", default="VXN")
    ap.add_argument("--elem-pos", default="O")
    ap.add_argument("--elem-neg", default="N")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise SystemExit(f"[ERROR] No existe: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet, orient = find_pair_sheet(xls, args.pair, args.a_name, args.b_name)

    df = pd.read_excel(xls, sheet_name=sheet)
    # Columnas esperadas
    col_q   = f"qval_{args.chem}"
    col_eff = f"log2OR_{args.chem}"
    for need in ["cx","cy","cz", col_q, col_eff]:
        if need not in df.columns:
            raise SystemExit(f"[ERROR] La hoja '{sheet}' no contiene columna '{need}'")

    # Efecto: si orientación = -1 (tenemos B__vs__A), invertimos el signo para que >0 = A enriquecido
    eff = df[col_eff].astype(float).to_numpy(copy=True)
    if orient == -1:
        eff = -eff

    qv = df[col_q].astype(float).to_numpy(copy=True)
    cx = df["cx"].astype(float).to_numpy()
    cy = df["cy"].astype(float).to_numpy()
    cz = df["cz"].astype(float).to_numpy()

    # Filtros
    is_sig = (qv <= args.q) & (np.abs(eff) >= args.min_abs_effect) & np.isfinite(eff) & np.isfinite(qv)
    export_mask = is_sig | (args.include_nonsig & np.isfinite(eff) & np.isfinite(qv))

    idx = np.flatnonzero(export_mask)
    if idx.size == 0:
        print("[INFO] No hay voxels tras los filtros. (Prueba bajar --min-abs-effect o subir --q)")
    # cap por |efecto|
    if idx.size > args.max_points:
        abse = np.abs(eff[idx])
        keep = np.argsort(abse)[-args.max_points:]
        idx = idx[keep]

    X = cx[idx]; Y = cy[idx]; Z = cz[idx]
    E = eff[idx]
    Q = qv[idx]
    OCC = np.clip(1.0 - Q, 0.0, 1.0)

    # p90 de |B| (entre significativos) para el PML
    if np.any(is_sig):
        p90 = float(np.percentile(np.abs(eff[is_sig]), 90))
    else:
        p90 = float(np.percentile(np.abs(eff[export_mask]), 90)) if np.any(export_mask) else 1.0
    if not np.isfinite(p90) or p90 <= 1e-9:
        p90 = 1.0

    out = Path(args.out)
    serial = 1
    with open(out, "w") as fo:
        fo.write("REMARK  Generated by pairwise_xlsx_voxels_to_pdb.py\n")
        fo.write(f"REMARK  Sheet: {sheet}   chem={args.chem}\n")
        fo.write(f"REMARK  Filters: q<={args.q:.3g}, |log2OR|>={args.min_abs_effect:.3g}, N={idx.size}\n")
        fo.write("REMARK  B-factor = log2OR (signo: >0 A enriquecido; <0 B enriquecido). Occupancy = 1 - q.\n")
        if args.overlay_pdb:
            try:
                with open(args.overlay_pdb, "r", encoding="utf-8", errors="ignore") as fi:
                    for line in fi:
                        if line.startswith(("ATOM","HETATM","ANISOU","TER","MODEL","ENDMDL","REMARK")):
                            fo.write(line)
                            if line.startswith(("ATOM","HETATM")):
                                try:
                                    serial = max(serial, int(line[6:11])+1)
                                except Exception:
                                    pass
            except FileNotFoundError:
                print(f"[WARN] overlay PDB no encontrado: {args.overlay_pdb}")

        resseq = 9999
        for x,y,z,b,oc in zip(X,Y,Z,E,OCC):
            if not np.isfinite(b) or not np.isfinite(oc):
                continue
            if b > 0:
                resn = args.resn_pos; elem = args.elem_pos
            elif b < 0:
                resn = args.resn_neg; elem = args.elem_neg
            else:
                # efecto exactamente 0: si incluimos no-sig, márcalo con elem neutro
                resn = args.resn_pos; elem = args.elem_pos
            fo.write(pdb_line("HETATM", serial, args.voxel_name, resn, "V", resseq,
                              float(x), float(y), float(z), float(oc), float(b), elem))
            serial += 1
        fo.write("END\n")

    # PML
    pml = out.with_suffix(".pml")
    occ_thr = 1.0 - float(args.q)
    with open(pml, "w") as f:
        f.write(f'load {out.name}\n')
        f.write('hide everything, all\n')
        f.write('select rec, not chain V\n')
        f.write('show cartoon, rec\n')
        f.write('set cartoon_transparency, 0.35, rec\n')
        f.write('color grey70, rec\n')

        f.write(f'select vox_pos, chain V and elem {args.elem_pos}\n')
        f.write(f'select vox_neg, chain V and elem {args.elem_neg}\n')
        f.write(f'select vox_pos_sig, vox_pos and occ >= {occ_thr:.6f}\n')
        f.write(f'select vox_neg_sig, vox_neg and occ >= {occ_thr:.6f}\n')
        if args.include_nonsig:
            f.write(f'select vox_pos_nonsig, vox_pos and occ < {occ_thr:.6f}\n')
            f.write(f'select vox_neg_nonsig, vox_neg and occ < {occ_thr:.6f}\n')

        # Esferas solo para significativos
        f.write('show spheres, vox_pos_sig or vox_neg_sig\n')
        f.write('alter vox_pos_sig, b = abs(b)\n')
        f.write('alter vox_neg_sig, b = abs(b)\n')
        f.write(f'alter vox_pos_sig, vdw = {args.r_min} + ({args.r_max}-{args.r_min})*min(b/{p90:.6f}, 1.0)\n')
        f.write(f'alter vox_neg_sig, vdw = {args.r_min} + ({args.r_max}-{args.r_min})*min(b/{p90:.6f}, 1.0)\n')
        f.write('rebuild\n')

        f.write(f'ramp_new ramp_pos, vox_pos_sig, [0.0, {p90:.3f}], [white, red]\n')
        f.write(f'ramp_new ramp_neg, vox_neg_sig, [0.0, {p90:.3f}], [white, blue]\n')
        f.write('color ramp_pos, vox_pos_sig\n')
        f.write('color ramp_neg, vox_neg_sig\n')

        if args.include_nonsig:
            f.write('hide spheres, vox_pos_nonsig or vox_neg_nonsig\n')

        f.write('set sphere_quality, 2\n')
        f.write('bg_color white\n')
        f.write('orient\n')

    print(f"[OK] PDB  → {out.resolve()}")
    print(f"[OK] PML  → {pml.resolve()}")
    print("Nota: B = log2OR (del Excel). Positivo ⇒ enriquecido en la primera clase del par (A).")

if __name__ == "__main__":
    main()

