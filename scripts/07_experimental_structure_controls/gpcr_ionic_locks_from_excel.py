#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import subprocess
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------
GPCRDB_STRUCTURE_URL = "https://gpcrdb.org/services/structure/"
GPCRDB_RESIDUES_EXT_URL = "https://gpcrdb.org/services/residues/extended/{entry_name}/"
RCSB_PDB_URL = "https://files.rcsb.org/download/{code}.pdb"

# Máximo número de mismatches permitidos al buscar el 5-mer
MAX_MISMATCH = 0

# ---------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------
def dist(a, b):
    return math.dist(a, b)

# ---------------------------------------------------------------------
# PyMOL helpers (PSE por PDB)
# ---------------------------------------------------------------------
def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _norm_state_folder(state: str) -> str:
    s = (state or "").strip().lower()
    if s == "active":
        return "Active"
    if s == "inactive":
        return "Inactive"
    return "Undefined"


def write_pymol_pml_for_row(row: dict, pdb_path: str, pml_path: str, pse_path: str):
    """
    Genera un .pml que:
      - carga el PDB local
      - selecciona los residuos usados (via atom serial -> byres)
      - crea distancias (distance) entre los dos átomos exactos usados
      - guarda una sesión .pse y sale
    """
    pdb = row["pdb_code"]
    chain = (row.get("chain_used") or "").strip()

    L: List[str] = []
    L.append("reinitialize")
    L.append(f"load {pdb_path}, {pdb}")

    L.append(f"hide everything, {pdb}")
    if chain:
        L.append(f"show cartoon, {pdb} and chain {chain}")
    else:
        L.append(f"show cartoon, {pdb}")

    # estética de distancias
    L.append("set dash_labels, 1")
    L.append("set dash_label_format, '%.2f Å'")
    L.append("set dash_gap, 0.35")
    L.append("set dash_radius, 0.12")
    L.append("set label_size, 18")
    L.append("set stick_radius, 0.2")

    zoom_targets: List[str] = []  # para no referenciar selecciones inexistentes

    def add_dist(tag: str, serial_a_key: str, serial_b_key: str):
        if not (_is_finite_number(row.get(serial_a_key)) and _is_finite_number(row.get(serial_b_key))):
            return

        a = int(float(row[serial_a_key]))
        b = int(float(row[serial_b_key]))

        # Selecciono los átomos por serial (id) y luego expando a residuo con byres
        L.append(f"select {tag}_atomA, {pdb} and id {a}")
        L.append(f"select {tag}_atomB, {pdb} and id {b}")
        L.append(f"select {tag}_resA, byres {tag}_atomA")
        L.append(f"select {tag}_resB, byres {tag}_atomB")

        L.append(f"show sticks, {tag}_resA or {tag}_resB")
        L.append(f"distance {tag}_dist, {tag}_atomA, {tag}_atomB")

        zoom_targets.append(f"({tag}_resA or {tag}_resB)")

    # Ionic: atom6.30 vs atom3.50
    add_dist("ionic", "ionic_atom_6.30_serial", "ionic_atom_3.50_serial")
    # Alt: atom3.49 vs atom6.30
    add_dist("alt", "alt_atom_3.49_serial", "alt_atom_6.30_serial")
    # DRY: atom3.49 vs atom3.50
    add_dist("dry", "dry_atom_3.49_serial", "dry_atom_3.50_serial")

    # Zoom: primero al sitio si hay distancias; si no (raro), al receptor
    if zoom_targets:
        sel_expr = " or ".join(zoom_targets)
        L.append(f"select locks_site, {sel_expr}")
        L.append("zoom locks_site, 10")
    else:
        if chain:
            L.append(f"zoom {pdb} and chain {chain}, 5")
        else:
            L.append(f"zoom {pdb}, 5")

    # Guardar sesión y salir
    L.append(f"save {pse_path}")
    L.append("quit")

    with open(pml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def run_pymol_make_pse(pymol_bin: str, pml_path: str):
    """
    Ejecuta PyMOL en modo batch y devuelve error detallado si falla.
    """
    p = subprocess.run(
        [pymol_bin, "-cq", pml_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"PyMOL falló (code={p.returncode}).\n"
            f"STDOUT:\n{p.stdout}\n"
            f"STDERR:\n{p.stderr}"
        )

# ---------------------------------------------------------------------
# Modelo PDB mínimo
# ---------------------------------------------------------------------
class Atom:
    __slots__ = ("serial", "name", "resname", "chain", "resseq",
                 "coord", "occ", "bfac", "element")

    def __init__(self, serial: int, name: str, resname: str, chain: str,
                 resseq: int, coord, occ: float, bfac: float, element: str):
        self.serial = serial
        self.name = name.strip()
        self.resname = resname.strip()
        self.chain = (chain or "").strip()
        self.resseq = resseq
        self.coord = coord
        self.occ = occ
        self.bfac = bfac
        self.element = (element or "").strip()


class Residue:
    def __init__(self, resname: str, chain: str, resseq: int):
        self.resname = resname
        self.chain = chain
        self.resseq = resseq
        self.atoms: Dict[str, Atom] = {}

    def add_atom(self, atom: Atom):
        self.atoms[atom.name] = atom

    def get_ca(self) -> Optional[Atom]:
        return self.atoms.get("CA")

    def heavy_atoms(self) -> List[Atom]:
        return [a for a in self.atoms.values()
                if (not a.element) or (a.element.upper() != "H")]


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

def resname_to_one(resname: str) -> str:
    resname = (resname or "").upper()
    return THREE_TO_ONE.get(resname, "X")


def normalize_resname(rn: str) -> str:
    rn = (rn or "").upper()
    return {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG"
    }.get(rn, rn)


def parse_pdb_from_text(text: str, chain_filter: str) -> List[Residue]:
    residues_dict: Dict[Tuple[str, int], Residue] = {}

    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = 0
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = (line[21].strip() or "")
        if chain != chain_filter:
            continue
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

        key = (chain, resseq)
        if key not in residues_dict:
            residues_dict[key] = Residue(resname, chain, resseq)
        residues_dict[key].add_atom(Atom(serial, name, resname, chain,
                                         resseq, (x, y, z), occ, bfac, element))

    residues = [res for (_, _), res in sorted(residues_dict.items(),
                                              key=lambda kv: kv[0][1])]
    return residues


def get_pdb_chain_sequence(residues: List[Residue]) -> str:
    return "".join(resname_to_one(normalize_resname(r.resname)) for r in residues)

# ---------------------------------------------------------------------
# GPCRdb residues / BW
# ---------------------------------------------------------------------
class GPCRResidue:
    __slots__ = ("seq_index", "sequence_number", "aa",
                 "segment", "bw_label", "bw_helix", "bw_pos")

    def __init__(self, seq_index: int, sequence_number: int,
                 aa: str, segment: str, bw_label: Optional[str]):
        self.seq_index = seq_index
        self.sequence_number = sequence_number
        self.aa = aa
        self.segment = segment
        self.bw_label = bw_label
        if bw_label is not None:
            try:
                h, p = bw_label.split(".")
                self.bw_helix = int(h)
                self.bw_pos = int(p)
            except Exception:
                self.bw_helix = None
                self.bw_pos = None
        else:
            self.bw_helix = None
            self.bw_pos = None


def load_gpcrdb_structure_index():
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


def classify_state_from_ligands(struct_meta):
    """Ago -> Active; Antagonist / inverse agonist -> Inactive; resto -> Undefined."""
    if not struct_meta:
        return "Undefined"
    ligs = struct_meta.get("ligands") or []
    has_ago = False
    has_antago = False
    for lig in ligs:
        f = (lig.get("function") or "").lower()
        if "inverse agonist" in f or "antagonist" in f:
            has_antago = True
        elif "agonist" in f:
            has_ago = True
    if has_antago and not has_ago:
        return "Inactive"
    if has_ago and not has_antago:
        return "Active"
    return "Undefined"


def download_pdb_from_rcsb(pdb_code: str) -> str:
    url = RCSB_PDB_URL.format(code=pdb_code.upper())
    print(f"[INFO] Descargando {pdb_code} desde RCSB: {url}", file=sys.stderr)
    r = requests.get(url)
    r.raise_for_status()
    return r.text


def fetch_extended_residues(entry_name: str) -> List[GPCRResidue]:
    url = GPCRDB_RESIDUES_EXT_URL.format(entry_name=entry_name)
    print(f"[INFO] Descargando residues/extended para {entry_name}: {url}", file=sys.stderr)
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    residues: List[GPCRResidue] = []
    for i, rec in enumerate(data):
        seqnum = rec.get("sequence_number")
        aa = rec.get("amino_acid")
        seg = rec.get("protein_segment")
        alt = rec.get("alternative_generic_numbers") or []
        bw_label = None
        for alt_g in alt:
            if alt_g.get("scheme") == "BW":
                bw_label = alt_g.get("label")
                break
        residues.append(GPCRResidue(i, seqnum, aa, seg, bw_label))
    return residues

# ---------------------------------------------------------------------
# Búsqueda local de BW mediante 5-mers
# ---------------------------------------------------------------------
def find_canonical_index_for_bw(gresidues: List[GPCRResidue],
                                helix: int, pos: int) -> Optional[int]:
    for gr in gresidues:
        if gr.bw_helix == helix and gr.bw_pos == pos:
            return gr.seq_index
    return None


def find_pdb_index_by_5mer(seq_can: str,
                           seq_pdb: str,
                           center_can_index: int,
                           flank: int,
                           max_mismatch: int = 0) -> Optional[int]:
    """
    Devuelve el índice en la secuencia PDB correspondiente al residuo
    central de un 2*flank+1-mer centrado en center_can_index.
    """
    L = len(seq_can)
    start = center_can_index - flank
    end = center_can_index + flank
    if start < 0 or end >= L:
        return None
    pattern = seq_can[start:end+1]
    w = len(pattern)
    if len(seq_pdb) < w:
        return None

    best_i = None
    best_mm = None
    for i in range(len(seq_pdb) - w + 1):
        window = seq_pdb[i:i+w]
        mism = sum(1 for a, b in zip(pattern, window) if a != b)
        if mism <= max_mismatch:
            if best_mm is None or mism < best_mm:
                best_mm = mism
                best_i = i
    if best_i is None:
        return None
    return best_i + flank


def get_bw_candidates_local(gresidues: List[GPCRResidue],
                            seq_can: str,
                            seq_pdb: str,
                            pdb_residues: List[Residue],
                            helix: int,
                            pos_center: int,
                            bw_window: int,
                            flank_5mer: int) -> List[Tuple[str, Residue, int]]:
    """
    Devuelve lista de candidatos (bw_label, Residue, |delta|)
    para todos los BW de una hélice dentro de [pos_center - bw_window,
    pos_center + bw_window], mapeados a la secuencia PDB
    mediante 5-mers centrados en cada una de esas posiciones.
    """
    cands: List[Tuple[str, Residue, int]] = []
    for delta in range(-bw_window, bw_window + 1):
        pos = pos_center + delta
        if pos <= 0:
            continue
        idx_can = find_canonical_index_for_bw(gresidues, helix, pos)
        if idx_can is None:
            continue
        idx_pdb = find_pdb_index_by_5mer(seq_can, seq_pdb, idx_can,
                                         flank_5mer, MAX_MISMATCH)
        if idx_pdb is None:
            continue
        if not (0 <= idx_pdb < len(pdb_residues)):
            continue
        bw_label = f"{helix}.{pos:02d}"
        cands.append((bw_label, pdb_residues[idx_pdb], abs(delta)))
    return cands

# ---------------------------------------------------------------------
# Distancias de cadena lateral
# ---------------------------------------------------------------------
ACID_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
BASIC_ATOMS = {
    "ARG": ["NE", "NH1", "NH2"],
    "LYS": ["NZ"],
}


def ca_distance(res1: Residue, res2: Residue) -> float:
    if res1 is None or res2 is None:
        return float("nan")
    a1 = res1.get_ca()
    a2 = res2.get_ca()
    if a1 is None or a2 is None:
        return float("nan")
    return dist(a1.coord, a2.coord)


def has_all_atoms(res: Residue, atom_names: List[str]) -> bool:
    return all(name in res.atoms for name in atom_names)


def min_dist_and_atoms(res1: Residue, res2: Residue,
                       only_names1=None, only_names2=None):
    """
    Devuelve (distancia, atom1, atom2) donde atom1/atom2 son los átomos
    concretos que dan la distancia mínima. Si no hay átomos válidos,
    devuelve (NaN, None, None).
    """
    if res1 is None or res2 is None:
        return float("nan"), None, None
    a1 = res1.heavy_atoms()
    a2 = res2.heavy_atoms()
    if only_names1:
        a1 = [a for a in a1 if a.name in only_names1]
    if only_names2:
        a2 = [a for a in a2 if a.name in only_names2]
    if not a1 or not a2:
        return float("nan"), None, None
    best = float("inf")
    best_pair = (None, None)
    for x in a1:
        for y in a2:
            d = dist(x.coord, y.coord)
            if d < best:
                best = d
                best_pair = (x, y)
    return best, best_pair[0], best_pair[1]

# ---------------------------------------------------------------------
# allowed_resnames_for para alt_lock
# ---------------------------------------------------------------------
def allowed_resnames_for(metric: str, site: str, policy: str) -> set:
    policy = (policy or "mid").lower()
    if metric == "alt_lock":
        if site == "3.49":
            return {"ASP"} if policy == "strict" else {"ASP", "GLU"}
        if site == "6.30":
            if policy in ("strict", "mid"):
                return {"LYS"}
            else:
                return {"LYS", "ARG"}
    return set()

# ---------------------------------------------------------------------
# Cálculo de locks para un PDB
# ---------------------------------------------------------------------
def compute_locks_for_pdb(pdb_code: str,
                          struct_meta: Optional[dict],
                          bw_window: int,
                          flank_5mer: int,
                          policy: str) -> Dict[str, object]:
    pdb_code_clean = pdb_code.strip().upper()
    base = pdb_code_clean + ".pdb"

    entry_name = struct_meta.get("protein") if struct_meta else None
    receptor = entry_name or ""
    gpcr_class = struct_meta.get("class") if struct_meta else ""
    species = struct_meta.get("species") if struct_meta else ""
    state_gpcrdb = struct_meta.get("state") if struct_meta else ""
    state_ligand = classify_state_from_ligands(struct_meta) if struct_meta else "Undefined"
    state = state_ligand

    chain_used = struct_meta.get("preferred_chain") if struct_meta else None

    if chain_used is None:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": receptor,
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": "",
            "status": "SKIP",
            "skip_reason": "No se pudo determinar la cadena (preferred_chain nulo)",
        }

    if not entry_name:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": "",
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": chain_used,
            "status": "SKIP",
            "skip_reason": "No se encontró 'protein' (entry_name) en GPCRdb para este PDB",
        }

    # --- PDB ---
    try:
        pdb_text = download_pdb_from_rcsb(pdb_code_clean)
    except Exception as e:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": receptor,
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": chain_used,
            "status": "SKIP",
            "skip_reason": f"Error descargando PDB de RCSB: {e}",
        }

    # Guardar PDB local (para PyMOL / PSE)
    try:
        with open(base, "w", encoding="utf-8") as fh:
            fh.write(pdb_text)
    except Exception as e:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": receptor,
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": chain_used,
            "status": "SKIP",
            "skip_reason": f"No se pudo escribir el PDB local {base}: {e}",
        }

    pdb_residues = parse_pdb_from_text(pdb_text, chain_filter=chain_used)
    if not pdb_residues:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": receptor,
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": chain_used,
            "status": "SKIP",
            "skip_reason": "No hay residuos ATOM/HETATM en esa cadena",
        }

    seq_pdb = get_pdb_chain_sequence(pdb_residues)

    # --- Secuencia canónica ---
    try:
        gresidues = fetch_extended_residues(entry_name)
    except Exception as e:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": receptor,
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": chain_used,
            "status": "SKIP",
            "skip_reason": f"Error llamando a residues/extended para {entry_name}: {e}",
        }

    if not gresidues:
        return {
            "file": base,
            "pdb_code": pdb_code_clean,
            "receptor_gpcrdb": receptor,
            "class": gpcr_class,
            "species": species,
            "state": state,
            "state_gpcrdb": state_gpcrdb,
            "chain_used": chain_used,
            "status": "SKIP",
            "skip_reason": "Lista residues/extended vacía",
        }

    seq_can = "".join(gr.aa for gr in gresidues)

    # ----------------------------------------------------------
    # Candidatos BW en ventana ±bw_window
    # ----------------------------------------------------------
    cands_3_50 = get_bw_candidates_local(
        gresidues, seq_can, seq_pdb, pdb_residues,
        helix=3, pos_center=50, bw_window=bw_window, flank_5mer=flank_5mer
    )
    cands_3_49 = get_bw_candidates_local(
        gresidues, seq_can, seq_pdb, pdb_residues,
        helix=3, pos_center=49, bw_window=bw_window, flank_5mer=flank_5mer
    )
    cands_6_30 = get_bw_candidates_local(
        gresidues, seq_can, seq_pdb, pdb_residues,
        helix=6, pos_center=30, bw_window=bw_window, flank_5mer=flank_5mer
    )

    # ===================================================
    # 1) Ionic lock: R3.50 – E/D6.30
    # ===================================================
    ionic_present = 0
    ionic_sc = float("nan")
    ionic_ca = float("nan")
    ionic_acid_type = ""
    ionic_basic_type = ""
    ionic_bw6 = ""
    ionic_bw3 = ""
    ionic_atom6_name = ""
    ionic_atom6_serial = float("nan")
    ionic_atom3_name = ""
    ionic_atom3_serial = float("nan")

    cands_R = [(bw, res, d) for (bw, res, d) in cands_3_50
               if normalize_resname(res.resname) == "ARG" and has_all_atoms(res, BASIC_ATOMS["ARG"])]
    cands_ED = [(bw, res, d) for (bw, res, d) in cands_6_30
                if normalize_resname(res.resname) in {"ASP", "GLU"}]

    best_pair = None
    best_dist = float("inf")

    for bw6, r6, d6 in cands_ED:
        rn6 = normalize_resname(r6.resname)
        acid_atoms = ACID_ATOMS[rn6]
        if not has_all_atoms(r6, acid_atoms):
            continue
        for bw3, r3, d3 in cands_R:
            d_side, atom6, atom3 = min_dist_and_atoms(
                r6, r3, only_names1=acid_atoms, only_names2=BASIC_ATOMS["ARG"]
            )
            if math.isnan(d_side):
                continue
            if d_side < best_dist:
                best_dist = d_side
                best_pair = (bw6, r6, rn6, atom6,
                             bw3, r3, "ARG", atom3)

    if best_pair is not None:
        bw6, r6, rn6, atom6, bw3, r3, rn3, atom3 = best_pair
        ionic_present = 1
        ionic_sc = best_dist
        ionic_ca = ca_distance(r6, r3)
        ionic_acid_type = rn6
        ionic_basic_type = rn3
        ionic_bw6 = bw6
        ionic_bw3 = bw3
        if atom6 is not None:
            ionic_atom6_name = atom6.name
            ionic_atom6_serial = atom6.serial
        if atom3 is not None:
            ionic_atom3_name = atom3.name
            ionic_atom3_serial = atom3.serial

    # ===================================================
    # 2) Alternative lock: D/E3.49 – K/R6.30
    # ===================================================
    alt_present = 0
    alt_sc = float("nan")
    alt_ca = float("nan")
    alt_acid_type = ""
    alt_basic_type = ""
    alt_bw349 = ""
    alt_bw630 = ""
    alt_atom3_name = ""
    alt_atom3_serial = float("nan")
    alt_atom6_name = ""
    alt_atom6_serial = float("nan")

    allowed_349 = allowed_resnames_for("alt_lock", "3.49", policy)
    allowed_630 = allowed_resnames_for("alt_lock", "6.30", policy)

    cands_349_acid = [(bw, res, d) for (bw, res, d) in cands_3_49
                      if normalize_resname(res.resname) in allowed_349]
    cands_630_basic = [(bw, res, d) for (bw, res, d) in cands_6_30
                       if normalize_resname(res.resname) in allowed_630]

    best_pair_alt = None
    best_dist_alt = float("inf")

    for bw3, r3, d3 in cands_349_acid:
        rn3 = normalize_resname(r3.resname)
        acid_atoms_alt = ACID_ATOMS[rn3]
        if not has_all_atoms(r3, acid_atoms_alt):
            continue
        for bw6, r6, d6 in cands_630_basic:
            rn6 = normalize_resname(r6.resname)
            basic_atoms_alt = BASIC_ATOMS[rn6]
            if not has_all_atoms(r6, basic_atoms_alt):
                continue
            d_side, atom3_alt, atom6_alt = min_dist_and_atoms(
                r3, r6, only_names1=acid_atoms_alt, only_names2=basic_atoms_alt
            )
            if math.isnan(d_side):
                continue
            if d_side < best_dist_alt:
                best_dist_alt = d_side
                best_pair_alt = (bw3, r3, rn3, atom3_alt,
                                 bw6, r6, rn6, atom6_alt)

    if best_pair_alt is not None:
        bw3, r3, rn3, atom3_alt, bw6, r6, rn6, atom6_alt = best_pair_alt
        alt_present = 1
        alt_sc = best_dist_alt
        alt_ca = ca_distance(r3, r6)
        alt_acid_type = rn3
        alt_basic_type = rn6
        alt_bw349 = bw3
        alt_bw630 = bw6
        if atom3_alt is not None:
            alt_atom3_name = atom3_alt.name
            alt_atom3_serial = atom3_alt.serial
        if atom6_alt is not None:
            alt_atom6_name = atom6_alt.name
            alt_atom6_serial = atom6_alt.serial

    # ===================================================
    # 3) DRY: D/E3.49 – R/K3.50
    # ===================================================
    dry_present = 0
    dry_sc = float("nan")
    dry_ca = float("nan")
    dry_acid_type = ""
    dry_basic_type = ""
    dry_bw349 = ""
    dry_bw350 = ""
    dry_atom3_name = ""
    dry_atom3_serial = float("nan")
    dry_atom5_name = ""
    dry_atom5_serial = float("nan")

    cands_349_ED = [(bw, res, d) for (bw, res, d) in cands_3_49
                    if normalize_resname(res.resname) in {"ASP", "GLU"}]
    cands_350_RK = [(bw, res, d) for (bw, res, d) in cands_3_50
                    if normalize_resname(res.resname) in {"ARG", "LYS"}]

    best_pair_dry = None
    best_dist_dry = float("inf")

    for bw3, r3, d3 in cands_349_ED:
        rn3 = normalize_resname(r3.resname)
        acid_atoms_dry = ACID_ATOMS[rn3]
        if not has_all_atoms(r3, acid_atoms_dry):
            continue
        for bw5, r5, d5 in cands_350_RK:
            rn5 = normalize_resname(r5.resname)
            basic_atoms_dry = BASIC_ATOMS[rn5]
            if not has_all_atoms(r5, basic_atoms_dry):
                continue
            d_side, atom3_dry, atom5_dry = min_dist_and_atoms(
                r3, r5, only_names1=acid_atoms_dry, only_names2=basic_atoms_dry
            )
            if math.isnan(d_side):
                continue
            if d_side < best_dist_dry:
                best_dist_dry = d_side
                best_pair_dry = (bw3, r3, rn3, atom3_dry,
                                 bw5, r5, rn5, atom5_dry)

    if best_pair_dry is not None:
        bw3, r3, rn3, atom3_dry, bw5, r5, rn5, atom5_dry = best_pair_dry
        dry_present = 1
        dry_sc = best_dist_dry
        dry_ca = ca_distance(r3, r5)
        dry_acid_type = rn3
        dry_basic_type = rn5
        dry_bw349 = bw3
        dry_bw350 = bw5
        if atom3_dry is not None:
            dry_atom3_name = atom3_dry.name
            dry_atom3_serial = atom3_dry.serial
        if atom5_dry is not None:
            dry_atom5_name = atom5_dry.name
            dry_atom5_serial = atom5_dry.serial

    row = {
        "file": base,
        "pdb_code": pdb_code_clean,
        "receptor_gpcrdb": receptor,
        "class": gpcr_class,
        "species": species,
        "state": state,
        "state_gpcrdb": state_gpcrdb,
        "chain_used": chain_used,
        "status": "OK",
        "skip_reason": "",
        # Ionic lock
        "ionic_present": ionic_present,
        "ionic_sidechain_NO": ionic_sc,
        "ionic_CA_CA": ionic_ca,
        "ionic_used_6.30_type": ionic_acid_type,
        "ionic_used_3.50_type": ionic_basic_type,
        "ionic_used_bw_6.30": ionic_bw6,
        "ionic_used_bw_3.50": ionic_bw3,
        "ionic_atom_6.30_name": ionic_atom6_name,
        "ionic_atom_6.30_serial": ionic_atom6_serial,
        "ionic_atom_3.50_name": ionic_atom3_name,
        "ionic_atom_3.50_serial": ionic_atom3_serial,
        # Alternative lock
        "alt_present": alt_present,
        "alt_sidechain_NO": alt_sc,
        "alt_CA_CA": alt_ca,
        "alt_used_3.49_type": alt_acid_type,
        "alt_used_6.30_type": alt_basic_type,
        "alt_used_bw_3.49": alt_bw349,
        "alt_used_bw_6.30": alt_bw630,
        "alt_atom_3.49_name": alt_atom3_name,
        "alt_atom_3.49_serial": alt_atom3_serial,
        "alt_atom_6.30_name": alt_atom6_name,
        "alt_atom_6.30_serial": alt_atom6_serial,
        # DRY
        "dry_present": dry_present,
        "dry_sidechain_NO": dry_sc,
        "dry_CA_CA": dry_ca,
        "dry_used_3.49_type": dry_acid_type,
        "dry_used_3.50_type": dry_basic_type,
        "dry_used_bw_3.49": dry_bw349,
        "dry_used_bw_3.50": dry_bw350,
        "dry_atom_3.49_name": dry_atom3_name,
        "dry_atom_3.49_serial": dry_atom3_serial,
        "dry_atom_3.50_name": dry_atom5_name,
        "dry_atom_3.50_serial": dry_atom5_serial,
    }

    return row

# ---------------------------------------------------------------------
# Worker para multiprocessing
# ---------------------------------------------------------------------
def worker_task(args):
    pdb_code, struct_index, bw_window, flank_5mer, policy, fieldnames = args
    struct_meta = struct_index.get(pdb_code.lower())
    row = compute_locks_for_pdb(
        pdb_code=pdb_code,
        struct_meta=struct_meta,
        bw_window=bw_window,
        flank_5mer=flank_5mer,
        policy=policy,
    )
    out = {k: (row[k] if k in row else float("nan")) for k in fieldnames}
    status = out["status"]
    print(f"[{status}] {out['pdb_code']} | receptor={out['receptor_gpcrdb']} | chain={out['chain_used']}",
          file=sys.stderr)
    return out

# ---------------------------------------------------------------------
# Estadística (igual que antes)
# ---------------------------------------------------------------------
def safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    return float(np.nanmean(arr)) if arr.size > 0 else float("nan")


def welch_ttest_normal_approx(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan"), float("nan")
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    n1, n2 = x.size, y.size
    denom = math.sqrt(v1/n1 + v2/n2)
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    t_stat = (m1 - m2) / denom
    num = (v1/n1 + v2/n2)**2
    den = (v1**2)/(n1**2*(n1-1)) + (v2**2)/(n2**2*(n2-1))
    df = num/den if den != 0 else float("nan")
    z = abs(t_stat)
    p = 2 * (1 - 0.5*(1 + math.erf(z/math.sqrt(2))))
    return float(t_stat), float(df), float(p)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, float)
    n = pvals.size
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    m = float(n)
    for i in range(n-1, -1, -1):
        if math.isnan(ranked[i]):
            adj[i] = math.nan
            continue
        val = ranked[i] * m / (i+1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty_like(adj)
    out[order] = adj
    return out

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Ionic/alt/DRY locks via BW local 5-mers + ventana BW, leyendo PDBs desde un Excel."
    )
    ap.add_argument("--excel", required=True,
                    help="Excel de entrada (ej. gpcr_active_inactive.xlsx)")
    ap.add_argument("--sheet", default="with_active_and_inactive",
                    help="Hoja con listas de PDBs (por defecto with_active_and_inactive)")
    ap.add_argument("--out", default="gpcr_locks_bw_5mer_window.xlsx",
                    help="Excel de salida")
    ap.add_argument("--bw-window", type=int, default=2,
                    help="Ventana BW ±N alrededor de 3.49/3.50/6.30 (por defecto 2)")
    ap.add_argument("--policy", choices=["strict", "mid", "loose"], default="mid",
                    help="Política de identidades para el alternative lock")
    ap.add_argument("--workers", type=int, default=1,
                    help="Número de procesos en paralelo (ej. 8)")

    # ---- PyMOL: generar sesiones ----
    ap.add_argument("--make-pse", action="store_true",
                    help="Generar un .pse por cada PDB con alguna distancia calculada (requiere PyMOL instalado)")
    ap.add_argument("--pymol-bin", default="pymol",
                    help="Ejecutable de PyMOL (por defecto 'pymol')")
    ap.add_argument("--pse-dir", default="pymol_pse",
                    help="Directorio base donde guardar los .pse (se crean subcarpetas Active/Inactive/Undefined)")
    ap.add_argument("--keep-pml", action="store_true",
                    help="No borrar los .pml intermedios (útil para debug)")

    args = ap.parse_args()

    flank_5mer = 2  # siempre usamos 5-mers para anclar

    # --- leer Excel ---
    if not os.path.isfile(args.excel):
        print(f"ERROR: no se encuentra el Excel '{args.excel}'", file=sys.stderr)
        sys.exit(2)

    xls = pd.ExcelFile(args.excel)
    if args.sheet not in xls.sheet_names:
        print(f"ERROR: la hoja '{args.sheet}' no existe. Hojas: {xls.sheet_names}", file=sys.stderr)
        sys.exit(2)

    df_in = xls.parse(args.sheet)

    pdb_set = set()
    for col in ["active_pdbs", "inactive_pdbs", "other_state_pdbs"]:
        if col not in df_in.columns:
            continue
        for val in df_in[col].dropna():
            if not isinstance(val, str):
                val = str(val)
            parts = [p.strip().upper() for p in val.split(",") if p.strip()]
            pdb_set.update(parts)

    pdb_list = sorted(pdb_set)
    if not pdb_list:
        print("ERROR: no se han encontrado PDBs en las columnas active_pdbs/inactive_pdbs/other_state_pdbs",
              file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Número total de PDBs únicos a procesar: {len(pdb_list)}", file=sys.stderr)

    struct_index = load_gpcrdb_structure_index()

    fieldnames = [
        "file", "pdb_code", "receptor_gpcrdb", "class", "species",
        "state", "state_gpcrdb", "chain_used", "status", "skip_reason",
        "ionic_present", "ionic_sidechain_NO", "ionic_CA_CA",
        "ionic_used_6.30_type", "ionic_used_3.50_type",
        "ionic_used_bw_6.30", "ionic_used_bw_3.50",
        "ionic_atom_6.30_name", "ionic_atom_6.30_serial",
        "ionic_atom_3.50_name", "ionic_atom_3.50_serial",
        "alt_present", "alt_sidechain_NO", "alt_CA_CA",
        "alt_used_3.49_type", "alt_used_6.30_type",
        "alt_used_bw_3.49", "alt_used_bw_6.30",
        "alt_atom_3.49_name", "alt_atom_3.49_serial",
        "alt_atom_6.30_name", "alt_atom_6.30_serial",
        "dry_present", "dry_sidechain_NO", "dry_CA_CA",
        "dry_used_3.49_type", "dry_used_3.50_type",
        "dry_used_bw_3.49", "dry_used_bw_3.50",
        "dry_atom_3.49_name", "dry_atom_3.49_serial",
        "dry_atom_3.50_name", "dry_atom_3.50_serial",
    ]

    todo = [
        (pdb_code, struct_index, args.bw_window, flank_5mer, args.policy, fieldnames)
        for pdb_code in pdb_list
    ]

    rows = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            for out in pool.imap_unordered(worker_task, todo):
                rows.append(out)
    else:
        for params in todo:
            rows.append(worker_task(params))

    df_all = pd.DataFrame(rows)
    n_ok = (df_all["status"] == "OK").sum()
    print(f"[RESUMEN] PDBs procesados correctamente: {n_ok} / {len(df_all)}", file=sys.stderr)

    # ---------------------------------------------------------
    # Generación de sesiones PyMOL (.pse), 1 por PDB con distancias
    # Guardadas en subcarpetas Active/Inactive/Undefined
    # ---------------------------------------------------------
    if args.make_pse:
        # Validar pymol-bin (evita errores tontos tipo "es un directorio" o symlink roto)
        if os.path.isdir(args.pymol_bin):
            print(f"[ERROR] --pymol-bin apunta a un directorio, no a un ejecutable: {args.pymol_bin}", file=sys.stderr)
            sys.exit(2)
        if not shutil_which(args.pymol_bin) and not os.path.isfile(args.pymol_bin):
            print(f"[WARN] No puedo encontrar PyMOL con --pymol-bin='{args.pymol_bin}'. "
                  f"Prueba con una ruta absoluta (p.ej. /usr/bin/pymol).", file=sys.stderr)

        os.makedirs(args.pse_dir, exist_ok=True)
        for sub in ("Active", "Inactive", "Undefined"):
            os.makedirs(os.path.join(args.pse_dir, sub), exist_ok=True)

        df_vis = df_all[(df_all["status"] == "OK") &
                        ((df_all["ionic_present"] == 1) |
                         (df_all["alt_present"] == 1) |
                         (df_all["dry_present"] == 1))].copy()

        print(f"[INFO] Generando .pse para {len(df_vis)} PDB(s)...", file=sys.stderr)

        for _, r in df_vis.iterrows():
            row = r.to_dict()
            pdb = row["pdb_code"]

            state_folder = _norm_state_folder(str(row.get("state", "")))
            out_dir = os.path.join(args.pse_dir, state_folder)

            pdb_path = os.path.abspath(f"{pdb}.pdb")  # se guarda en compute_locks_for_pdb
            pml_path = os.path.abspath(os.path.join(out_dir, f"{pdb}_locks.pml"))
            pse_path = os.path.abspath(os.path.join(out_dir, f"{pdb}_locks.pse"))

            try:
                if not os.path.isfile(pdb_path):
                    print(f"[WARN] No existe {pdb_path}; no puedo generar PSE para {pdb}", file=sys.stderr)
                    continue

                write_pymol_pml_for_row(row, pdb_path=pdb_path, pml_path=pml_path, pse_path=pse_path)
                run_pymol_make_pse(args.pymol_bin, pml_path)

                if os.path.isfile(pse_path):
                    print(f"[PSE] {pse_path}", file=sys.stderr)
                    if not args.keep_pml:
                        try:
                            os.remove(pml_path)
                        except Exception:
                            pass
                else:
                    print(f"[WARN] PyMOL terminó sin error pero NO encuentro el .pse: {pse_path}", file=sys.stderr)

            except Exception as e:
                print(f"[WARN] No se pudo generar PSE para {pdb}: {e}", file=sys.stderr)

    # ----------------- Estadística (igual que antes) -----------------
    df_ok = df_all[df_all["status"] == "OK"].copy()
    df_ok["state_norm"] = df_ok["state"].astype(str).str.strip().str.capitalize()

    metrics = [
        "ionic_sidechain_NO",
        "ionic_CA_CA",
        "alt_sidechain_NO",
        "alt_CA_CA",
        "dry_sidechain_NO",
        "dry_CA_CA",
    ]

    if not df_ok.empty:
        grouped = df_ok.groupby(["receptor_gpcrdb", "class", "state_norm"])[metrics].mean()
        df_means = grouped.reset_index()
    else:
        df_means = pd.DataFrame(columns=["receptor_gpcrdb", "class", "state_norm"] + metrics)

    log2fc_rows = []
    for (receptor, gpcr_class), sub in df_ok.groupby(["receptor_gpcrdb", "class"]):
        states = sub["state_norm"].unique()
        if not {"Active", "Inactive"}.issubset(set(states)):
            continue
        for metric in metrics:
            data_active = sub.loc[sub["state_norm"] == "Active", metric].to_numpy(float)
            data_inact  = sub.loc[sub["state_norm"] == "Inactive", metric].to_numpy(float)
            m_active = safe_mean(data_active)
            m_inact  = safe_mean(data_inact)
            if (not math.isfinite(m_active)) or (not math.isfinite(m_inact)) or m_inact <= 0:
                log2fc = float("nan")
            else:
                log2fc = math.log2(m_active / m_inact)
            log2fc_rows.append({
                "receptor_gpcrdb": receptor,
                "class": gpcr_class,
                "metric": metric,
                "mean_active": m_active,
                "mean_inactive": m_inact,
                "log2FC_active_over_inactive": log2fc,
            })
    df_log2fc = pd.DataFrame(log2fc_rows)

    within_rows = []
    for gpcr_class, sub_class in df_ok.groupby("class"):
        sub_class = sub_class.copy()
        sub_class["state_norm"] = sub_class["state_norm"].astype(str)
        for metric in metrics:
            data_active = sub_class.loc[sub_class["state_norm"] == "Active", metric].to_numpy(float)
            data_inact  = sub_class.loc[sub_class["state_norm"] == "Inactive", metric].to_numpy(float)
            mean_a = safe_mean(data_active)
            mean_i = safe_mean(data_inact)
            n_a = int(np.isfinite(data_active).sum())
            n_i = int(np.isfinite(data_inact).sum())
            if n_a >= 2 and n_i >= 2:
                t_stat, df_w, p_val = welch_ttest_normal_approx(data_active, data_inact)
            else:
                t_stat = df_w = p_val = float("nan")
            within_rows.append({
                "class": gpcr_class,
                "metric": metric,
                "n_active": n_a,
                "mean_active": mean_a,
                "n_inactive": n_i,
                "mean_inactive": mean_i,
                "t_statistic": t_stat,
                "df_welch": df_w,
                "p_value": p_val,
            })

    df_within = pd.DataFrame(within_rows)
    if not df_within.empty:
        df_within["p_adj_BH"] = bh_fdr(df_within["p_value"].to_numpy(float))
    else:
        df_within["p_adj_BH"] = []

    between_rows = []
    if not df_log2fc.empty:
        classes = sorted(df_log2fc["class"].dropna().unique())
        for metric in metrics:
            sub_m = df_log2fc[df_log2fc["metric"] == metric]
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    c1, c2 = classes[i], classes[j]
                    x = sub_m.loc[sub_m["class"] == c1, "log2FC_active_over_inactive"].to_numpy(float)
                    y = sub_m.loc[sub_m["class"] == c2, "log2FC_active_over_inactive"].to_numpy(float)
                    mean1 = safe_mean(x)
                    mean2 = safe_mean(y)
                    n1 = int(np.isfinite(x).sum())
                    n2 = int(np.isfinite(y).sum())
                    if n1 >= 2 and n2 >= 2:
                        t_stat, df_w, p_val = welch_ttest_normal_approx(x, y)
                    else:
                        t_stat = df_w = p_val = float("nan")
                    between_rows.append({
                        "metric": metric,
                        "class1": c1,
                        "class2": c2,
                        "n_receptors_class1": n1,
                        "mean_log2FC_class1": mean1,
                        "n_receptors_class2": n2,
                        "mean_log2FC_class2": mean2,
                        "t_statistic": t_stat,
                        "df_welch": df_w,
                        "p_value": p_val,
                    })

    df_between = pd.DataFrame(between_rows)
    if not df_between.empty:
        df_between["p_adj_BH"] = bh_fdr(df_between["p_value"].to_numpy(float))
    else:
        df_between["p_adj_BH"] = []

    # --- Guardar Excel ---
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="locks_per_pdb", index=False)

        start = 0
        df_means.to_excel(writer, sheet_name="summary_stats", index=False, startrow=start)
        start += df_means.shape[0] + 2

        df_log2fc.to_excel(writer, sheet_name="summary_stats", index=False, startrow=start)
        start += df_log2fc.shape[0] + 2

        df_within.to_excel(writer, sheet_name="summary_stats", index=False, startrow=start)
        start += df_within.shape[0] + 2

        df_between.to_excel(writer, sheet_name="summary_stats", index=False, startrow=start)

    print(f"[OK] Excel generado: {args.out}", file=sys.stderr)


def shutil_which(cmd: str) -> Optional[str]:
    """
    mini-which sin importar shutil (para no tocar demasiadas dependencias del script)
    """
    if os.path.isabs(cmd) and os.access(cmd, os.X_OK):
        return cmd
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, cmd)
        if os.access(cand, os.X_OK) and os.path.isfile(cand):
            return cand
    return None


if __name__ == "__main__":
    main()

