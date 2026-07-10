#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPCR: residuos en 3.39 y 2.50 + distancia mínima de cadena lateral + clase GPCR
-------------------------------------------------------------------------------

Para cada PDB:
  - Detecta qué residuo hay en 3.39 y 2.50 (usando BW en occ/B-factor, con ventana ±N).
  - Indica el resname y BW "escogido" para cada posición (primer candidato en la ventana).
  - Indica si ese residuo es negativo (ASP o GLU).
  - Calcula la distancia mínima entre átomos pesados de CADENA LATERAL
    de esos dos residuos (excluye átomos de backbone: N, CA, C, O, OXT, OT1, OT2).
  - Anota:
      * state: 'active' / 'inactive' / 'unknown' según el nombre del fichero
        (contenga "_active" o "_inactive").
      * gpcr_class: a partir de un fichero de familias (1ª columna -> última columna).

Salida: CSV con columnas
  file, chain_used, status, skip_reason,
  state, gpcr_class,
  res_3.39_resname, res_3.39_BW, is_neg_3.39,
  res_2.50_resname, res_2.50_BW, is_neg_2.50,
  min_dist_3.39_2.50

Uso:
  python3 gpcr_339_250_metrics_with_class_sidechain.py \
    --in "aligned/*.pdb" \
    --chain A \
    --familias families.txt \
    --bw-window 2 \
    --workers 6 \
    --out gpcr_339_250_metrics_class.csv
"""

import argparse
import glob
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


# ---------- Modelo PDB mínimo + BW a partir de Occ/Bfac ----------

class Atom:
    __slots__ = ("name", "resname", "chain", "resseq", "coord", "occ", "bfac", "element")

    def __init__(self, name, resname, chain, resseq, coord, occ, bfac, element):
        self.name = name.strip()
        self.resname = resname.strip()
        self.chain = (chain or "").strip()
        self.resseq = resseq
        self.coord = coord
        self.occ = occ
        self.bfac = bfac
        self.element = (element or "").strip()


class Residue:
    def __init__(self, resname, chain, resseq):
        self.resname = resname
        self.chain = chain
        self.resseq = resseq
        self.atoms: Dict[str, Atom] = {}
        self.bw: Optional[str] = None

    def add_atom(self, atom: Atom):
        self.atoms[atom.name] = atom

    def get_ca(self):
        return self.atoms.get("CA")

    def heavy_atoms(self):
        return [
            a for a in self.atoms.values()
            if (not a.element) or (a.element.upper() != "H")
        ]


def parse_pdb(path: str, chain_filter: Optional[str] = None):
    """
    Lee el PDB y construye:
      - residues[(chain, resseq)] -> Residue
      - bw_index["H.PP"] -> Residue  (p.ej. "3.49")

    Usa occupancy/B-factor para leer números BW, evitando confundir ocupancias 1.00.
    """
    residues: Dict[Tuple[str, int], Residue] = {}
    chains_present = set()

    with open(path, "r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = (line[21].strip() or "")
            resseq_str = line[22:26]
            try:
                resseq = int(resseq_str)
            except ValueError:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            occ_str = line[54:60].strip()
            bfac_str = line[60:66].strip()
            try:
                occ = float(occ_str) if occ_str else 0.0
            except Exception:
                occ = 0.0
            try:
                bfac = float(bfac_str) if bfac_str else 0.0
            except Exception:
                bfac = 0.0
            element = line[76:78].strip()

            chains_present.add(chain)

            # Filtro de cadena (pero registramos todas para auto_pick_chain)
            if (chain_filter is not None) and (chain != chain_filter):
                continue

            key = (chain, resseq)
            if key not in residues:
                residues[key] = Residue(resname, chain, resseq)
            residues[key].add_atom(
                Atom(name, resname, chain, resseq, (x, y, z), occ, bfac, element)
            )

    # --- Helpers para reconocer BW "válido" ---

    def looks_like_bw(v: float) -> bool:
        # BW típico entre 1.00 y 8.99 (TM1–TM8); ignorar exactamente 1.00 (ocupancia real)
        if not (1.0 <= v < 9.0):
            return False
        if abs(v - 1.00) < 1e-6:
            return False
        return True

    def pick_bw_for_res(res: Residue) -> Optional[str]:
        # 1) Priorizar CA: primero occupancy, luego B-factor
        ca = res.atoms.get("CA")
        if ca:
            if looks_like_bw(ca.occ):
                return f"{ca.occ:.2f}"
            if looks_like_bw(ca.bfac):
                return f"{ca.bfac:.2f}"
        # 2) Buscar en el resto de átomos del residuo (moda)
        vals_occ = [round(a.occ, 2) for a in res.atoms.values() if looks_like_bw(a.occ)]
        if vals_occ:
            moda = Counter(vals_occ).most_common(1)[0][0]
            return f"{moda:.2f}"
        vals_b = [round(a.bfac, 2) for a in res.atoms.values() if looks_like_bw(a.bfac)]
        if vals_b:
            moda = Counter(vals_b).most_common(1)[0][0]
            return f"{moda:.2f}"
        # 3) Nada con pinta de BW
        return None

    bw_index: Dict[str, Residue] = {}
    for res in residues.values():
        bw = pick_bw_for_res(res)
        if bw is not None:
            res.bw = bw
            if bw not in bw_index:  # evitar sobreescritura silenciosa
                bw_index[bw] = res

    return residues, bw_index, chains_present


# ---------- BW helpers ----------

def _parse_bw(bw: str):
    h, p = f"{float(bw):.2f}".split(".")
    return int(h), int(p)


def _format_bw(h: int, p: int):
    return f"{h}.{p:02d}"


def get_candidates_by_bw(bw_index: Dict[str, Residue], target_bw: str, window: int = 2):
    """
    Devuelve lista de (bw_str, residue, delta_posicional)
    donde bw_str es el BW encontrado (p.ej. "3.38"), delta es |pos - target_pos|.
    """
    if not target_bw:
        return []
    h, p = _parse_bw(target_bw)
    cands = []
    # Posición central y ventana ±N
    for q in [p] + [p + d for d in range(-window, window + 1) if d != 0]:
        if 0 <= q <= 99:
            key = _format_bw(h, q)
            if key in bw_index:
                cands.append((key, bw_index[key], abs(q - p)))
    # Orden: primero menor desplazamiento
    cands.sort(key=lambda t: (t[2], abs(_parse_bw(t[0])[1] - p)))
    return cands


# ---------- Utilidades de clasificación ----------

def normalize_resname(rn: str) -> str:
    rn = (rn or "").upper()
    return {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        "ASH": "ASP", "GLH": "GLU",
        "LYN": "LYS", "ARN": "ARG",
    }.get(rn, rn)


# ---------- Geometría (distancia mínima entre cadenas laterales) ----------

# Átomos que consideramos backbone y NO usamos para la distancia de cadena lateral
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "OT1", "OT2"}


def dist(a, b):
    return math.dist(a, b)


def min_dist_between_residues(res1: Residue, res2: Residue) -> float:
    """
    Distancia mínima entre átomos pesados de la CADENA LATERAL
    (excluyendo backbone: N, CA, C, O, OXT, OT1, OT2).
    """
    if res1 is None or res2 is None:
        return float("nan")

    # Heavy atoms de cada residuo, pero solo side chain
    a1 = [a for a in res1.heavy_atoms() if a.name not in BACKBONE_ATOMS]
    a2 = [a for a in res2.heavy_atoms() if a.name not in BACKBONE_ATOMS]

    # Si alguno no tiene cadena lateral definida, devolvemos NaN
    if not a1 or not a2:
        return float("nan")

    return min(dist(x.coord, y.coord) for x in a1 for y in a2)


# ---------- Cálculo de métrica principal ----------

def compute_metrics(path: str, chain: Optional[str], bw_window: int):
    """
    Para un PDB (y una cadena):
      - Busca candidatos BW alrededor de 3.39 y 2.50.
      - Elige el primer candidato (más cercano en BW) para cada posición.
      - Devuelve resname y BW de esos residuos, marca si son negativos (ASP/GLU).
      - Calcula la distancia mínima entre átomos pesados de CADENA LATERAL
        de esos dos residuos.
    """
    residues, bw_index, _ = parse_pdb(path, chain_filter=chain)
    if not bw_index:
        return ("SKIP", "sin BW en occupancy/B-factor"), None

    c_339 = get_candidates_by_bw(bw_index, "3.39", bw_window)
    c_250 = get_candidates_by_bw(bw_index, "2.50", bw_window)

    def pick_primary(cands):
        """
        Devuelve (resname, bw, residue_obj).
        Si no hay candidatos, devuelve ("", "", None).
        """
        if not cands:
            return "", "", None
        bw, res, _ = cands[0]
        rn = normalize_resname(res.resname)
        return rn, bw, res

    res339_name, bw339, res339 = pick_primary(c_339)
    res250_name, bw250, res250 = pick_primary(c_250)

    is_neg_339 = 1 if res339_name in ("ASP", "GLU") else 0
    is_neg_250 = 1 if res250_name in ("ASP", "GLU") else 0

    min_dist_339_250 = float("nan")
    if res339 is not None and res250 is not None:
        min_dist_339_250 = min_dist_between_residues(res339, res250)

    row = {
        "res_3.39_resname": res339_name,
        "res_3.39_BW": bw339,
        "is_neg_3.39": is_neg_339,
        "res_2.50_resname": res250_name,
        "res_2.50_BW": bw250,
        "is_neg_2.50": is_neg_250,
        "min_dist_3.39_2.50": min_dist_339_250,
    }
    return ("OK", "done"), row


# ---------- Anotación de estado y clase (familias) ----------

def load_familias(path: Path) -> dict:
    """
    Lee el fichero de familias (TSV/CSV) y crea un diccionario:
      primera_columna (upper, strip) -> última_columna (clase, strip)
    """
    try:
        fam = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        fam = pd.read_csv(path, sep="\t")
    if fam.shape[1] < 2:
        raise ValueError(
            "El fichero de familias debe tener al menos dos columnas (ID y Class)."
        )
    first_col = fam.columns[0]
    last_col = fam.columns[-1]
    mapping = dict(
        zip(
            fam[first_col].astype(str).str.upper().str.strip(),
            fam[last_col].astype(str).str.strip(),
        )
    )
    return mapping


def infer_state(filename: str) -> str:
    """
    Usa el nombre del fichero para deducir:
      - 'inactive' si contiene '_inactive'
      - 'active'   si contiene '_active'
      - 'unknown'  en cualquier otro caso
    """
    f = str(filename).lower()
    if "_inactive" in f:
        return "inactive"
    if "_active" in f:
        return "active"
    return "unknown"


def file_prefix_lower(filename: str) -> str:
    """
    Prefijo antes del primer "_" (en minúsculas).
    Ej: '5HT1A_active.BWocc.pdb' -> '5ht1a'
    """
    base = os.path.basename(str(filename))
    return base.split("_", 1)[0].lower()


def infer_gpcr_class(filename: str, map_upper: dict) -> str:
    """
    Usa el prefijo del fichero (antes del primer "_") como clave
    para buscar en el diccionario de familias (upper).
    """
    key_up = file_prefix_lower(filename).upper()
    return map_upper.get(key_up, "Unknown")


# ---------- Utilidades CLI ----------

def expand_inputs(inp: str) -> List[str]:
    if os.path.isdir(inp):
        files = []
        for ext in ("*.pdb", "*.ent", "*.PDB"):
            files += glob.glob(os.path.join(inp, ext))
        return sorted(files)
    if any(c in inp for c in "*?[]"):
        return sorted(glob.glob(inp))
    return [inp] if os.path.isfile(inp) else []


def auto_pick_chain(path: str) -> Optional[str]:
    """
    Intenta adivinar qué cadena es la "buena":
      1) Cadena que tenga un residuo BW == 3.50.
      2) Si no, la cadena con más residuos etiquetados con BW.
    """
    residues, bw_index, chains_present = parse_pdb(path, chain_filter=None)
    chains_with_350 = {res.chain for res in residues.values() if res.bw == "3.50"}
    if chains_with_350:
        return sorted(chains_with_350)[0]

    counts = defaultdict(int)
    for res in residues.values():
        if res.bw is not None:
            counts[res.chain] += 1
    return (
        sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if counts
        else None
    )


def _worker(args):
    (path, chain, bw_window, mapping) = args
    try:
        status, row = compute_metrics(path, chain, bw_window)
        tag, msg = status
        base = os.path.basename(path)

        if tag == "OK":
            # Anotar estado y clase
            row["state"] = infer_state(base)
            row["gpcr_class"] = infer_gpcr_class(base, mapping)
            return ("OK", base, chain, row)

        # Si no hay BW en esa cadena, intentar otra cadena automática
        if tag == "SKIP" and (chain is not None):
            auto = auto_pick_chain(path)
            if auto and auto != chain:
                status2, row2 = compute_metrics(path, auto, bw_window)
                if status2[0] == "OK":
                    row2["state"] = infer_state(base)
                    row2["gpcr_class"] = infer_gpcr_class(base, mapping)
                    return ("OK", base, auto, row2)

        # Último intento: sin filtro de cadena (todas las cadenas)
        if tag == "SKIP":
            status3, row3 = compute_metrics(path, None, bw_window)
            if status3[0] == "OK":
                row3["state"] = infer_state(base)
                row3["gpcr_class"] = infer_gpcr_class(base, mapping)
                return ("OK", base, "(all)", row3)

        return ("SKIP", base, chain, msg)

    except Exception as e:
        base = os.path.basename(path)
        return ("ERR", base, chain, str(e))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Residuos en 3.39 y 2.50 (BW) + distancia mínima de cadenas laterales "
            "+ estado (active/inactive) + clase GPCR."
        )
    )
    ap.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Carpeta o patrón de PDBs (p.ej. 'aligned/*.pdb').",
    )
    ap.add_argument(
        "--chain",
        default=None,
        help="Cadena a usar (si se omite, intenta auto-detectar).",
    )
    ap.add_argument(
        "--familias",
        required=True,
        help="Fichero de familias (TSV/CSV). 1ª columna = ID, última = Class.",
    )
    ap.add_argument(
        "--out",
        default="gpcr_339_250_metrics_class.csv",
        help="CSV de salida.",
    )
    ap.add_argument(
        "--bw-window",
        type=int,
        default=2,
        help="Ventana ±N alrededor del BW objetivo (por defecto 2).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Procesos en paralelo (1 para secuencial).",
    )
    args = ap.parse_args()

    files = expand_inputs(args.inp)
    if not files:
        print("No se encontraron PDBs de entrada.", file=sys.stderr)
        sys.exit(2)

    familias_path = Path(args.familias)
    if not familias_path.exists():
        print(
            f"[ERROR] Fichero de familias no encontrado: {familias_path}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Cargar mapping de familias
    mapping = load_familias(familias_path)

    # Preparar trabajos
    todo = []
    for path in files:
        chain = args.chain or auto_pick_chain(path)
        if chain is None:
            print(
                f"[SKIP] {os.path.basename(path)}: cadena no detectada.",
                file=sys.stderr,
            )
            continue
        todo.append((path, chain, args.bw_window, mapping))

    # CSV
    import csv

    fieldnames = [
        "file",
        "chain_used",
        "status",
        "skip_reason",
        "state",
        "gpcr_class",
        "res_3.39_resname",
        "res_3.39_BW",
        "is_neg_3.39",
        "res_2.50_resname",
        "res_2.50_BW",
        "is_neg_2.50",
        "min_dist_3.39_2.50",
    ]

    processed = 0
    with open(args.out, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=fieldnames)
        wr.writeheader()
        fo.flush()

        if args.workers > 1:
            from multiprocessing import Pool

            chunksize = max(1, (len(todo) // (args.workers * 4)) or 1)
            with Pool(processes=args.workers) as pool:
                for res in pool.imap_unordered(_worker, todo, chunksize=chunksize):
                    tag, base, chain_used, payload = res
                    if tag == "OK":
                        row = {
                            "file": base,
                            "chain_used": chain_used,
                            "status": "OK",
                            "skip_reason": "",
                        }
                        row.update(payload)
                        wr.writerow(row)
                        fo.flush()
                        processed += 1
                        print(f"[OK] {base} | chain={chain_used}")
                    elif tag == "SKIP":
                        row = {
                            "file": base,
                            "chain_used": chain_used,
                            "status": "SKIP",
                            "skip_reason": str(payload),
                        }
                        for k in fieldnames:
                            if k not in row:
                                row[k] = float("nan")
                        wr.writerow(row)
                        fo.flush()
                        print(
                            f"[SKIP] {base}: {payload}",
                            file=sys.stderr,
                        )
                    else:
                        print(f"[ERR] {base}: {payload}", file=sys.stderr)
        else:
            # Modo secuencial
            for argsW in todo:
                tag, base, chain_used, payload = _worker(argsW)
                if tag == "OK":
                    row = {
                        "file": base,
                        "chain_used": chain_used,
                        "status": "OK",
                        "skip_reason": "",
                    }
                    row.update(payload)
                    wr.writerow(row)
                    fo.flush()
                    processed += 1
                    print(f"[OK] {base} | chain={chain_used}")
                elif tag == "SKIP":
                    row = {
                        "file": base,
                        "chain_used": chain_used,
                        "status": "SKIP",
                        "skip_reason": str(payload),
                    }
                    for k in fieldnames:
                        if k not in row:
                            row[k] = float("nan")
                    wr.writerow(row)
                    fo.flush()
                    print(
                        f"[SKIP] {base}: {payload}",
                        file=sys.stderr,
                    )
                else:
                    print(f"[ERR] {base}: {payload}", file=sys.stderr)

    print(
        f"[RESUMEN] PDBs procesados (OK): {processed} | CSV: {args.out}"
    )


if __name__ == "__main__":
    main()

