#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPCR ionic / alternative lock distances usando GPCRdb + descarga automática desde PDB.

Resumen
-------

Dentro de este script defines una lista de IDs PDB:

    PDB_IDS = ["5T1A", "6GPS", "6GPX", "7XA3"]

Para cada ID:

1) Descarga el .pdb desde RCSB:
      https://files.rcsb.org/download/<PDB>.pdb

2) Usa la API de GPCRdb:

   - https://gpcrdb.org/services/structure/
     -> para obtener info de ese PDB (receptor, cadena preferida, estado, etc.)
   - https://gpcrdb.org/services/structure/assign_generic_numbers
     -> para anotar el PDB con números genéricos (BW / GPCRdb) en B-factors.

3) A partir del PDB anotado, busca:

   - Ionic lock canónico:  R3.50 – E/D6.30
   - Ionic lock alternativo: D/E3.49 – K/R6.30

   usando una ventana ±N posiciones BW (por defecto 2).

4) Calcula:

   - Distancia mínima entre átomos pesados de las cadenas laterales
   - Distancia CA–CA

5) Escribe un CSV con:

   file, pdb_code, receptor_gpcrdb, class, species, state, chain_used,
   ionic_*, alt_*.

Uso
---

  python3 gpcr_ionic_locks_gpcrdb_remote.py \
      --out gpcr_locks.csv \
      --bw-window 2 \
      --policy loose

Requiere: requests, numpy
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import requests

# ------------------------------------------------------------------
# EDITA AQUÍ: lista de IDs PDB que quieres analizar
# ------------------------------------------------------------------
PDB_IDS = [
"5T1A","6GPS","6GPX","7XA3","4MBS","5UIW","6AKX","6AKY","6MEO","6MET","7O7F","7F1Q","7F1R","7F1S","7F1T","3ODU","3OE0","3OE6","3OE8","3OE9","4RWS","8U4N","8U4O","8U4P","8U4Q","8U4R","8U4S","8U4T","8K3Z","5VEX","5VEW","5NX2","6B3J","6KJV","6KK1","6KK7","6ORV","6LN2","6VCB","7C2E","6X18","6X19","6X1A","6XOX","7LCI","7LCJ","7LCK","7E14","7DUQ","7KI0","7KI1","7DUR","7EVM","7RTB","7S1M","7S3I","7LLL","7LLY","7FIM","7VBI","7RG9","7RGP","7VBH","7S15","7X8R","7X8S","8JIP","8JIR","8JIS","8WG7","8YW3","9IVG","9IVM","6FJ3","6NBF","6NBH","6NBI","7VVJ","7VVK","7VVL","7VVM","7VVN","7VVO","8HA0","8HAF","8HAO","8FLS","8FLQ","8FLT","8FLU","8FLR","7Y35","7Y36","8GW8","8JR9","9JR2","9JR3","4NTJ","4PXZ","4PY0","7PP1","7XXI","6E59","6HLL","6HLO","6HLP","6J20","6J21","7RMH","7RMG","7RMI","7P00","7P02","8U26","6OS9","6OSA","6PWC","6UP7","7UL2","8JPB","8JPC","8JPF","4N6H","4RWA","4RWD","6PT2","6PT3","8F7S","8Y45","4DJH","6B73","6VI4","8F7W","7YIT","8DZP","8DZR","7Y1F","8DZQ","8DZS","8FEG","8EF5","8EF6","8EFB","8EFL","8EFO","8EFQ","8F7Q","8F7R","8K9K","8K9L","4EA3","5DHG","5DHH","8F7X"
]

GPCRDB_STRUCTURE_URL = "https://gpcrdb.org/services/structure/"
GPCRDB_ASSIGN_URL = "https://gpcrdb.org/services/structure/assign_generic_numbers"
RCSB_PDB_URL = "https://files.rcsb.org/download/{code}.pdb"

# ============================================================
# Utilidades geométricas
# ============================================================

def dist(a, b):
    """Distancia euclídea entre dos coordenadas 3D."""
    return math.dist(a, b)


def angle_between(v1, v2):
    """Ángulo (en grados) entre dos vectores 3D."""
    v1 = np.array(v1, float)
    v2 = np.array(v2, float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = np.dot(v1, v2) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))

# ============================================================
# Modelo PDB mínimo (versión con segmentación de fragmentos)
# ============================================================

class Atom:
    __slots__ = ("name", "resname", "chain", "resseq", "coord", "occ", "bfac", "element")

    def __init__(self, name, resname, chain, resseq, coord, occ, bfac, element):
        self.name = name.strip()
        self.resname = resname.strip()
        self.chain = (chain or "").strip()
        self.resseq = resseq  # int (número de residuo PDB)
        self.coord = coord
        self.occ = occ
        self.bfac = bfac
        self.element = (element or "").strip()


class Residue:
    def __init__(self, resname, chain, resseq):
        self.resname = resname
        self.chain = chain
        self.resseq = resseq  # int
        self.atoms: Dict[str, Atom] = {}
        self.bw: Optional[str] = None  # número genérico tipo "3.49"

    def add_atom(self, atom: Atom):
        self.atoms[atom.name] = atom

    def get_ca(self):
        return self.atoms.get("CA")

    def heavy_atoms(self):
        return [a for a in self.atoms.values()
                if (not a.element) or (a.element.upper() != "H")]


def _looks_like_bw(v: float) -> bool:
    """
    Un número BW típico está entre 1.00 y 8.99 (TM1–TM8).
    NO consideramos exactamente 1.00 porque suele ser ocupancia real o ruido.
    """
    if not (1.0 <= v < 9.0):
        return False
    if abs(v - 1.00) < 1e-6:
        return False
    return True


def _pick_bw_for_res(res: Residue) -> Optional[str]:
    """
    Decide qué valor (si lo hay) de los campos occ/bfac es un número BW/GPCRdb
    para este residuo. Se prioriza CA y luego la moda en el resto de átomos.

    Esta lógica asume que el PDB ya ha pasado por assign_generic_numbers de GPCRdb.
    """
    ca = res.atoms.get("CA")
    if ca:
        if _looks_like_bw(ca.occ):
            return f"{ca.occ:.2f}"
        if _looks_like_bw(ca.bfac):
            return f"{ca.bfac:.2f}"

    vals_occ = [round(a.occ, 2) for a in res.atoms.values() if _looks_like_bw(a.occ)]
    if vals_occ:
        moda = Counter(vals_occ).most_common(1)[0][0]
        return f"{moda:.2f}"

    vals_b = [round(a.bfac, 2) for a in res.atoms.values() if _looks_like_bw(a.bfac)]
    if vals_b:
        moda = Counter(vals_b).most_common(1)[0][0]
        return f"{moda:.2f}"

    return None


def parse_pdb_from_lines(lines, chain_filter: Optional[str] = None,
                         gap_threshold: int = 50):
    """
    Parseo PDB -> residuos + índice BW.

    Extra:
      - Detecta fragmentos sintéticos en la MISMA cadena:
        * Ordena los residuos por resseq
        * Si hay saltos grandes (> gap_threshold) define nuevos fragmentos
        * Se queda sólo con el fragmento con más residuos anotados con BW
          (normalmente el 7TM del GPCR).
    """
    residues: Dict[Tuple[str, int], Residue] = {}
    chains_present = set()

    for line in lines:
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

        if (chain_filter is not None) and (chain != chain_filter):
            continue

        key = (chain, resseq)
        if key not in residues:
            residues[key] = Residue(resname, chain, resseq)
        residues[key].add_atom(Atom(name, resname, chain, resseq,
                                    (x, y, z), occ, bfac, element))

    if not residues:
        return {}, {}, chains_present

    # 1) Asignar BW por residuo
    for res in residues.values():
        res.bw = _pick_bw_for_res(res)

    # 2) Segmentar por cadena y resseq según saltos grandes
    chain_to_res = defaultdict(list)
    for (chain, _), res in residues.items():
        chain_to_res[chain].append(res)

    best_segment_keys: Optional[set] = None
    best_bw_count = -1

    for chain, rlist in chain_to_res.items():
        if not rlist:
            continue
        rlist_sorted = sorted(rlist, key=lambda r: r.resseq)
        segments: List[List[Residue]] = []
        current_seg = [rlist_sorted[0]]

        for prev, cur in zip(rlist_sorted, rlist_sorted[1:]):
            if (cur.resseq - prev.resseq) > gap_threshold:
                segments.append(current_seg)
                current_seg = [cur]
            else:
                current_seg.append(cur)
        segments.append(current_seg)

        for seg in segments:
            bw_count = sum(1 for r in seg if r.bw is not None)
            if bw_count > best_bw_count:
                best_bw_count = bw_count
                best_segment_keys = {(r.chain, r.resseq) for r in seg}

    # 3) Nos quedamos sólo con el fragmento principal (más BW)
    if best_segment_keys is not None and best_bw_count > 0:
        residues = {k: r for k, r in residues.items() if k in best_segment_keys}

    # 4) Construir índice BW con ese fragmento
    bw_index: Dict[str, Residue] = {}
    for res in residues.values():
        if res.bw is not None and res.bw not in bw_index:
            bw_index[res.bw] = res

    return residues, bw_index, chains_present


def parse_pdb_from_text(text: str, chain_filter: Optional[str] = None,
                        gap_threshold: int = 50):
    """Conveniencia: parsea un PDB contenido en un string grande."""
    return parse_pdb_from_lines(text.splitlines(),
                                chain_filter=chain_filter,
                                gap_threshold=gap_threshold)

# ============================================================
# Helpers de BW
# ============================================================

def _parse_bw(bw: str):
    h, p = f"{float(bw):.2f}".split(".")
    return int(h), int(p)


def _format_bw(h: int, p: int):
    return f"{h}.{p:02d}"


def get_candidates_by_bw(bw_index: Dict[str, Residue],
                         target_bw: str,
                         window: int = 2):
    """
    Devuelve lista de candidatos (bw, Residue, |delta|) alrededor del BW objetivo
    dentro de una ventana ±window.
    """
    if not target_bw:
        return []
    h, p = _parse_bw(target_bw)
    cands = []
    positions = [p] + [p + d for d in range(-window, window + 1) if d != 0]
    for q in positions:
        if 0 <= q <= 99:
            key = _format_bw(h, q)
            if key in bw_index:
                cands.append((key, bw_index[key], abs(q - p)))
    cands.sort(key=lambda t: (t[2], abs(_parse_bw(t[0])[1] - p)))
    return cands

# ============================================================
# Identidades permitidas / filtros
# ============================================================

def normalize_resname(rn: str) -> str:
    rn = (rn or "").upper()
    return {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG"
    }.get(rn, rn)


def filter_by_resname(cands, allowed: set):
    """Filtra candidatos por nombre de residuo normalizado."""
    return [(bw, res, d) for (bw, res, d) in cands
            if normalize_resname(res.resname) in allowed]


def allowed_resnames_for(metric: str, site: str, policy: str) -> set:
    """
    Reglas para políticas strict/mid/loose:

    alt_lock: 3.49 ácido, 6.30 básico
    """
    policy = (policy or "mid").lower()
    if metric == "alt_lock":  # 3.49 ↔ 6.30 (alternativo)
        if site == "3.49":
            return {"ASP"} if policy == "strict" else {"ASP", "GLU"}
        if site == "6.30":
            # strict/mid: sólo LYS; loose: LYS o ARG
            if policy == "strict":
                return {"LYS"}
            elif policy == "mid":
                return {"LYS"}
            else:  # loose
                return {"LYS", "ARG"}
    return set()


# Mapas de átomos de cadena lateral
ACID_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
BASIC_ATOMS = {
    "ARG": ["NE", "NH1", "NH2"],
    "LYS": ["NZ"],
}


def min_dist_between_residues(res1: Residue,
                              res2: Residue,
                              only_names1=None,
                              only_names2=None) -> float:
    """Distancia mínima heavy-heavy entre dos residuos."""
    if res1 is None or res2 is None:
        return float("nan")
    a1 = res1.heavy_atoms()
    a2 = res2.heavy_atoms()
    if only_names1:
        a1 = [a for a in a1 if a.name in only_names1]
    if only_names2:
        a2 = [a for a in a2 if a.name in only_names2]
    if not a1 or not a2:
        return float("nan")
    return min(dist(x.coord, y.coord) for x in a1 for y in a2)


def ca_distance(res1: Residue, res2: Residue) -> float:
    """Distancia CA–CA entre dos residuos."""
    if res1 is None or res2 is None:
        return float("nan")
    a1 = res1.get_ca()
    a2 = res2.get_ca()
    if a1 is None or a2 is None:
        return float("nan")
    return dist(a1.coord, a2.coord)


def min_heavy_over_candidates_flexible(acid_dict, pos_dict):
    """
    acid_dict: {"ASP": [(bw,res,delta), ...], "GLU": [...]}
    pos_dict : {"ARG": [...], "LYS": [...]}
    Devuelve: (best_dist, used_acid_type, used_pos_type, used_acid_bw, used_pos_bw)
    """
    best = (float("nan"), None, None, None, None)
    found = False
    for atype, alist in acid_dict.items():
        if not alist:
            continue
        a_atoms = ACID_ATOMS.get(atype, None)
        if not a_atoms:
            continue
        for btype, blist in pos_dict.items():
            if not blist:
                continue
            b_atoms = BASIC_ATOMS.get(btype, None)
            if not b_atoms:
                continue
            for abw, ares, _ in alist:
                for bbw, bres, _ in blist:
                    d = min_dist_between_residues(ares, bres, a_atoms, b_atoms)
                    if math.isnan(d):
                        continue
                    if (not found) or (d < best[0]):
                        best = (d, atype, btype, abw, bbw)
                        found = True
    return best

# ============================================================
# Llamadas a GPCRdb y RCSB
# ============================================================

def load_gpcrdb_structure_index():
    """
    Descarga la lista de estructuras de GPCRdb y construye un índice
    por código PDB (en minúsculas).
    """
    print("[INFO] Descargando lista de estructuras de GPCRdb...", file=sys.stderr)
    r = requests.get(GPCRDB_STRUCTURE_URL)
    r.raise_for_status()
    data = r.json()
    index = {}
    for entry in data:
        pdb = entry.get("pdb_code", "").lower()
        if pdb:
            index[pdb] = entry
    print(f"[INFO] Estructuras GPCRdb cargadas: {len(index)}", file=sys.stderr)
    return index


def download_pdb_from_rcsb(pdb_code: str) -> bytes:
    """
    Descarga el archivo PDB desde RCSB (formato .pdb) para un código dado.
    """
    url = RCSB_PDB_URL.format(code=pdb_code.upper())
    print(f"[INFO] Descargando {pdb_code} desde RCSB: {url}", file=sys.stderr)
    r = requests.get(url)
    r.raise_for_status()
    return r.content


def annotate_with_gpcrdb_bytes(pdb_bytes: bytes) -> str:
    """
    Envía un PDB (en bytes) a GPCRdb para que asigne números genéricos.
    Devuelve el texto PDB anotado.
    """
    files = {"pdb_file": ("structure.pdb", pdb_bytes)}
    r = requests.post(GPCRDB_ASSIGN_URL, files=files)
    r.raise_for_status()
    return r.text

# ============================================================
# Cálculo de locks
# ============================================================

def compute_locks_from_pdbtext(pdb_text: str,
                               chain: str,
                               bw_window: int,
                               policy: str):
    """
    Cálculo de:
      - ionic lock R3.50–E/D6.30
      - alt lock D/E3.49–K/R6.30
    sobre el PDB anotado por GPCRdb.
    """
    residues, bw_index, chains_present = parse_pdb_from_text(
        pdb_text, chain_filter=chain
    )

    if not bw_index:
        return ("SKIP", "sin números genéricos detectados en esa cadena/fragmento principal"), None

    # Candidatos por BW
    c_350 = get_candidates_by_bw(bw_index, "3.50", bw_window)
    c_349 = get_candidates_by_bw(bw_index, "3.49", bw_window)
    c_630 = get_candidates_by_bw(bw_index, "6.30", bw_window)

    # ---------- Ionic lock canónico: R3.50 – E/D6.30 ----------
    c350_arg = filter_by_resname(c_350, {"ARG"})
    c630_acid = filter_by_resname(c_630, {"GLU", "ASP"})

    ionic_present = int(bool(c350_arg and c630_acid))

    if ionic_present:
        acid_dict_630 = {
            "ASP": [t for t in c630_acid if normalize_resname(t[1].resname) == "ASP"],
            "GLU": [t for t in c630_acid if normalize_resname(t[1].resname) == "GLU"],
        }
        pos_dict_350 = {
            "ARG": c350_arg,
            "LYS": [],
        }
        ionic_best = min_heavy_over_candidates_flexible(acid_dict_630, pos_dict_350)
        ionic_hh, ionic_acid_type, ionic_pos_type, ionic_bw6, ionic_bw3 = ionic_best

        # CA–CA
        best_ca = float("inf")
        ca_pair = ("", "")
        for bw6, r6, _ in c630_acid:
            for bw3, r3, _ in c350_arg:
                d_ca = ca_distance(r6, r3)
                if not math.isnan(d_ca) and d_ca < best_ca:
                    best_ca = d_ca
                    ca_pair = (bw6, bw3)
        ionic_ca = best_ca if best_ca < float("inf") else float("nan")
    else:
        ionic_hh = float("nan")
        ionic_ca = float("nan")
        ionic_acid_type = None
        ionic_pos_type = None
        ionic_bw6 = None
        ionic_bw3 = None
        ca_pair = ("", "")

    # ---------- Alternative lock: D/E3.49 – K/R6.30 ----------
    allowed_349 = allowed_resnames_for("alt_lock", "3.49", policy)
    allowed_630_alt = allowed_resnames_for("alt_lock", "6.30", policy)

    c349_any = filter_by_resname(c_349, allowed_349) if allowed_349 else []
    c630_any_alt = filter_by_resname(c_630, allowed_630_alt) if allowed_630_alt else []

    alt_present = int(bool(c349_any and c630_any_alt))

    if alt_present:
        acid_dict_349 = {
            "ASP": [t for t in c349_any if normalize_resname(t[1].resname) == "ASP"],
            "GLU": [t for t in c349_any if normalize_resname(t[1].resname) == "GLU"],
        }
        pos_dict_630 = {
            "LYS": [t for t in c630_any_alt if normalize_resname(t[1].resname) == "LYS"],
            "ARG": [t for t in c630_any_alt if normalize_resname(t[1].resname) == "ARG"],
        }
        alt_best = min_heavy_over_candidates_flexible(acid_dict_349, pos_dict_630)
        alt_hh, alt_acid_type, alt_pos_type, alt_bw349, alt_bw630 = alt_best

        # CA–CA
        best_ca_alt = float("inf")
        ca_pair_alt = ("", "")
        for bw3, r3, _ in c349_any:
            for bw6, r6, _ in c630_any_alt:
                d_ca = ca_distance(r3, r6)
                if not math.isnan(d_ca) and d_ca < best_ca_alt:
                    best_ca_alt = d_ca
                    ca_pair_alt = (bw3, bw6)
        alt_ca = best_ca_alt if best_ca_alt < float("inf") else float("nan")
    else:
        alt_hh = float("nan")
        alt_ca = float("nan")
        alt_acid_type = None
        alt_pos_type = None
        alt_bw349 = None
        alt_bw630 = None
        ca_pair_alt = ("", "")

    row = {
        # Ionic lock
        "ionic_lock_present": ionic_present,
        "ionic_lock_min_heavy": ionic_hh,
        "ionic_lock_CA_CA": ionic_ca,
        "ionic_lock_used_6.30_type": (ionic_acid_type or ""),
        "ionic_lock_used_3.50_type": (ionic_pos_type or ""),
        "ionic_lock_used_bw_6.30": (ionic_bw6 or ""),
        "ionic_lock_used_bw_3.50": (ionic_bw3 or ""),
        "ionic_lock_CA_pair_6.30": ca_pair[0],
        "ionic_lock_CA_pair_3.50": ca_pair[1],

        # Alternative lock
        "alt_lock_present": alt_present,
        "alt_lock_min_heavy": alt_hh,
        "alt_lock_CA_CA": alt_ca,
        "alt_lock_used_3.49_type": (alt_acid_type or ""),
        "alt_lock_used_6.30_type": (alt_pos_type or ""),
        "alt_lock_used_bw_3.49": (alt_bw349 or ""),
        "alt_lock_used_bw_6.30": (alt_bw630 or ""),
        "alt_lock_CA_pair_3.49": ca_pair_alt[0],
        "alt_lock_CA_pair_6.30": ca_pair_alt[1],
    }

    return ("OK", "done"), row

# ============================================================
# CLI / main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Cálculo de ionic/alt lock para GPCRs bajando PDBs de RCSB y usando GPCRdb."
    )
    ap.add_argument(
        "--out", default="gpcr_locks.csv",
        help="Nombre del CSV de salida"
    )
    ap.add_argument(
        "--chain", default=None,
        help="Cadena a usar (si se omite, se usa preferred_chain de GPCRdb)"
    )
    ap.add_argument(
        "--bw-window", type=int, default=2,
        help="Ventana ±N alrededor del BW objetivo (por defecto 2)"
    )
    ap.add_argument(
        "--policy", choices=["strict", "mid", "loose"], default="mid",
        help="Flexibilidad de identidades para el alternative lock (por defecto mid)"
    )
    args = ap.parse_args()

    if not PDB_IDS:
        print("La lista PDB_IDS está vacía, edita el script y añade IDs.", file=sys.stderr)
        sys.exit(2)

    # Índice de estructuras de GPCRdb (pdb_code -> info)
    struct_index = load_gpcrdb_structure_index()

    fieldnames = [
        "file",
        "pdb_code",
        "receptor_gpcrdb",
        "class",
        "species",
        "state",
        "chain_used",
        "status",
        "skip_reason",
        "ionic_lock_present",
        "ionic_lock_min_heavy",
        "ionic_lock_CA_CA",
        "ionic_lock_used_6.30_type",
        "ionic_lock_used_3.50_type",
        "ionic_lock_used_bw_6.30",
        "ionic_lock_used_bw_3.50",
        "ionic_lock_CA_pair_6.30",
        "ionic_lock_CA_pair_3.50",
        "alt_lock_present",
        "alt_lock_min_heavy",
        "alt_lock_CA_CA",
        "alt_lock_used_3.49_type",
        "alt_lock_used_6.30_type",
        "alt_lock_used_bw_3.49",
        "alt_lock_used_bw_6.30",
        "alt_lock_CA_pair_3.49",
        "alt_lock_CA_pair_6.30",
    ]

    processed_ok = 0
    with open(args.out, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=fieldnames)
        wr.writeheader()

        for pdb_code in PDB_IDS:
            pdb_code_clean = pdb_code.strip().upper()
            base = pdb_code_clean + ".pdb"
            struct_meta = struct_index.get(pdb_code_clean.lower())

            receptor = struct_meta.get("protein") if struct_meta else ""
            gpcr_class = struct_meta.get("class") if struct_meta else ""
            species = struct_meta.get("species") if struct_meta else ""
            state = struct_meta.get("state") if struct_meta else ""

            chain_used = args.chain or (struct_meta.get("preferred_chain") if struct_meta else None)

            if chain_used is None:
                row = {
                    "file": base,
                    "pdb_code": pdb_code_clean,
                    "receptor_gpcrdb": receptor,
                    "class": gpcr_class,
                    "species": species,
                    "state": state,
                    "chain_used": "",
                    "status": "SKIP",
                    "skip_reason": "No se pudo determinar la cadena (ni --chain ni preferred_chain)",
                }
                wr.writerow(row)
                print(f"[SKIP] {base}: sin cadena", file=sys.stderr)
                continue

            try:
                pdb_bytes = download_pdb_from_rcsb(pdb_code_clean)
            except Exception as e:
                row = {
                    "file": base,
                    "pdb_code": pdb_code_clean,
                    "receptor_gpcrdb": receptor,
                    "class": gpcr_class,
                    "species": species,
                    "state": state,
                    "chain_used": chain_used,
                    "status": "SKIP",
                    "skip_reason": f"Error descargando PDB de RCSB: {e}",
                }
                wr.writerow(row)
                print(f"[SKIP] {base}: fallo al descargar de RCSB", file=sys.stderr)
                continue

            try:
                pdb_text = annotate_with_gpcrdb_bytes(pdb_bytes)
            except Exception as e:
                row = {
                    "file": base,
                    "pdb_code": pdb_code_clean,
                    "receptor_gpcrdb": receptor,
                    "class": gpcr_class,
                    "species": species,
                    "state": state,
                    "chain_used": chain_used,
                    "status": "SKIP",
                    "skip_reason": f"Error llamando a GPCRdb assign_generic_numbers: {e}",
                }
                wr.writerow(row)
                print(f"[SKIP] {base}: fallo en assign_generic_numbers", file=sys.stderr)
                continue

            status, payload = compute_locks_from_pdbtext(
                pdb_text, chain_used, args.bw_window, args.policy
            )

            tag, msg = status
            if tag == "OK":
                row = {
                    "file": base,
                    "pdb_code": pdb_code_clean,
                    "receptor_gpcrdb": receptor,
                    "class": gpcr_class,
                    "species": species,
                    "state": state,
                    "chain_used": chain_used,
                    "status": "OK",
                    "skip_reason": "",
                }
                row.update(payload)
                wr.writerow(row)
                processed_ok += 1
                print(f"[OK] {base} | receptor={receptor or '?'} | chain={chain_used}")
            else:
                row = {
                    "file": base,
                    "pdb_code": pdb_code_clean,
                    "receptor_gpcrdb": receptor,
                    "class": gpcr_class,
                    "species": species,
                    "state": state,
                    "chain_used": chain_used,
                    "status": "SKIP",
                    "skip_reason": msg,
                }
                for k in fieldnames:
                    if k not in row:
                        row[k] = float("nan")
                wr.writerow(row)
                print(f"[SKIP] {base}: {msg}", file=sys.stderr)

    print(f"[RESUMEN] PDBs procesados correctamente: {processed_ok} | CSV: {args.out}")


if __name__ == "__main__":
    main()

