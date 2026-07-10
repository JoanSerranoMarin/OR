#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_ecl2_z_depth_v8_aligned_debug_3ECL2closest_3perTM.py

Corrección importante respecto a V7
-----------------------------------
La referencia basal sigue siendo:
    los 3 residuos con Z más negativa de CADA TM1-TM7

Por tanto, si se detectan los 7 TMs:
    3 residuos/TM × 7 TMs = 21 residuos de referencia

La métrica principal ahora NO usa:
    - ni el centroide de todo el ECL2
    - ni un único residuo ECL2 profundo

Ahora usa:
    el centroide all-atom de los 3 residuos del ECL2 más cercanos en 3D
    al centroide de la referencia basal de 21 residuos TM.

Objetivo
--------
Alinear todos los PDBs usando los residuos TM1-TM7 y medir la profundidad axial
del ECL2 en Z.

Orientación
-----------
Después de alinear el eje principal de los TMs con Z, se fuerza que el ECL2 quede
hacia Z positiva. Así, la base TM-negativa queda abajo y el ECL2 arriba.

Métrica principal
-----------------
    ecl2_closest3res_centroid_delta_z_to_tm_neg3_per_tm_aligned

Definición:
    1) Se calcula el centroide de la referencia basal:
       3 residuos con Z más negativa de cada TM1-TM7 = 21 residuos.
    2) Para cada residuo del ECL2 se calcula su centroide all-atom.
    3) Se seleccionan los 3 residuos ECL2 más cercanos en 3D al centroide basal.
    4) Se calcula el centroide all-atom conjunto de esos 3 residuos ECL2.
    5) Se calcula:

       ΔZ = Z(centroide de esos 3 residuos ECL2 cercanos)
            - Z(centroide basal de los 21 residuos TM)

Interpretación:
    menor ΔZ = los 3 residuos ECL2 más cercanos están más profundos/cerca axialmente
               de la base TM-negativa
    mayor ΔZ = esos 3 residuos ECL2 quedan más alejados axialmente

Outputs
-------
    <out-prefix>_distances.csv
    <out-prefix>_summary.csv
    <out-prefix>_field_detection.csv
    <out-prefix>_failed_files.csv
    <out-prefix>_aligned_pdbs/*.pdb
    <out-prefix>_marker_pdbs/*.pdb

Marker PDBs:
    TMC = centroide de todos los TMs
    BAS = centroide de los 21 residuos base: 3 más negativos por TM
    ECL = centroide global del ECL2
    CEN = centroide all-atom de los 3 residuos ECL2 más cercanos a BAS
    DEP = centroide del residuo ECL2 más profundo en Z
    C3D = centroide del residuo ECL2 más cercano en 3D

Uso:
    python analyze_ecl2_z_depth_v8_aligned_debug_3ECL2closest_3perTM.py \
        --pdb-dir /ruta/a/pdbs \
        --metadata gpcr_ecl2_lengths_real_by_receptor.csv \
        --out-prefix ecl2_depth_v8
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import itertools
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Atom:
    record: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: Optional[float]
    bfactor: Optional[float]
    element: str
    line: str

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def is_hydrogen(self) -> bool:
        atom = self.atom_name.strip().upper()
        elem = self.element.strip().upper()
        return elem == "H" or atom.startswith("H")


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    order: int
    atoms: List[Atom] = field(default_factory=list)
    gpcr_value: Optional[float] = None
    gpcr_field_used: str = ""
    tm_number: Optional[int] = None

    @property
    def resid_label(self) -> str:
        suffix = self.icode if self.icode else ""
        return f"{self.chain}:{self.resname}{self.resseq}{suffix}".strip()


# =============================================================================
# Basic helpers
# =============================================================================

def safe_float(text) -> Optional[float]:
    try:
        if text is None:
            return None
        text = str(text).strip()
        if text == "":
            return None
        val = float(text)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def safe_int(text) -> Optional[int]:
    try:
        if text is None:
            return None
        text = str(text).strip()
        if text == "":
            return None
        return int(text)
    except Exception:
        return None


def aa3_to_aa1(resname: str) -> str:
    table = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEC": "U", "PYL": "O",
    }
    return table.get(resname.upper(), "X")


def infer_receptor_and_state(filename: str) -> Tuple[str, str]:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"\(.*?\)$", "", stem)

    # Archivos tipo 5ht1a_all_inactive.BWocc.pdb dejan ".bwocc" al hacer Path(...).stem.
    stem = re.sub(r"\.bwocc$", "", stem, flags=re.IGNORECASE)

    if "inactive" in stem:
        state = "inactive"
    elif "active" in stem:
        state = "active"
    else:
        state = "unknown"

    receptor = stem
    for token in ["_all_inactive", "_all_active", "_inactive", "_active"]:
        receptor = receptor.replace(token, "")
    receptor = receptor.strip("_")
    receptor = re.sub(r"\.bwocc$", "", receptor, flags=re.IGNORECASE)
    return receptor, state


def looks_like_gpcrdb_value(value: Optional[float]) -> bool:
    if value is None:
        return False
    if value <= 0:
        return False
    integer_part = int(math.floor(value + 1e-8))
    if integer_part not in {1, 2, 3, 4, 5, 6, 7}:
        return False
    frac = abs(value - integer_part)
    if frac < 0.005:
        return False
    return True


def tm_number_from_gpcr_value(value: Optional[float]) -> Optional[int]:
    if not looks_like_gpcrdb_value(value):
        return None
    return int(math.floor(float(value) + 1e-8))


# =============================================================================
# PDB parsing
# =============================================================================

def parse_pdb(path: Path) -> Dict[str, List[Residue]]:
    residues: "OrderedDict[Tuple[str, int, str], Residue]" = OrderedDict()
    order_counter = 0

    with path.open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            parsed = None
            try:
                record = line[0:6].strip()
                serial = int(line[6:11])
                atom_name = line[12:16].strip()
                altloc = line[16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip() or "_"
                resseq = int(line[22:26])
                icode = line[26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = safe_float(line[54:60])
                bfactor = safe_float(line[60:66])
                element = line[76:78].strip() if len(line) >= 78 else ""
                parsed = (record, serial, atom_name, altloc, resname, chain, resseq, icode, x, y, z, occupancy, bfactor, element)
            except Exception:
                parsed = None

            if parsed is None:
                parts = line.split()
                if len(parts) < 11:
                    continue
                try:
                    record = parts[0]
                    serial = int(parts[1])
                    atom_name = parts[2]
                    resname = parts[3]
                    chain = parts[4]
                    resseq = int(parts[5])
                    icode = ""
                    x, y, z = map(float, parts[6:9])
                    occupancy = safe_float(parts[9])
                    bfactor = safe_float(parts[10])
                    element = parts[-1] if parts[-1].isalpha() else ""
                    altloc = ""
                    parsed = (record, serial, atom_name, altloc, resname, chain, resseq, icode, x, y, z, occupancy, bfactor, element)
                except Exception:
                    continue

            record, serial, atom_name, altloc, resname, chain, resseq, icode, x, y, z, occupancy, bfactor, element = parsed

            if altloc not in ("", "A"):
                continue

            key = (chain, resseq, icode)
            if key not in residues:
                residues[key] = Residue(
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    resname=resname,
                    order=order_counter,
                )
                order_counter += 1

            residues[key].atoms.append(
                Atom(
                    record=record,
                    serial=serial,
                    atom_name=atom_name,
                    altloc=altloc,
                    resname=resname,
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element=element,
                    line=line.rstrip("\n"),
                )
            )

    by_chain: Dict[str, List[Residue]] = {}
    for residue in residues.values():
        by_chain.setdefault(residue.chain, []).append(residue)

    for chain in by_chain:
        by_chain[chain].sort(key=lambda r: r.order)

    return by_chain


# =============================================================================
# GPCRdb/Ballesteros field detection
# =============================================================================

def residue_gpcr_value(residue: Residue, field_name: str) -> Optional[float]:
    vals = []
    for atom in residue.atoms:
        value = atom.occupancy if field_name == "occupancy" else atom.bfactor
        if looks_like_gpcrdb_value(value):
            vals.append(round(float(value), 2))
    if not vals:
        return None
    counts = Counter(vals)
    most_common_value, _ = counts.most_common(1)[0]
    return float(most_common_value)


def score_chain_for_field(residues: List[Residue], field_name: str) -> Dict[str, object]:
    valid_residues = 0
    tm_counts = Counter()
    for residue in residues:
        value = residue_gpcr_value(residue, field_name)
        tm = tm_number_from_gpcr_value(value)
        if tm is not None:
            valid_residues += 1
            tm_counts[tm] += 1

    return {
        "field": field_name,
        "valid_residues": valid_residues,
        "tm_counts": dict(tm_counts),
        "tm_numbers_detected": sorted(tm_counts),
        "n_tm_numbers": len(tm_counts),
    }


def choose_best_gpcr_field(by_chain: Dict[str, List[Residue]], gpcr_field: str) -> Tuple[str, str, Dict[str, object]]:
    diagnostics = []

    for chain, residues in by_chain.items():
        fields = ["occupancy", "bfactor"] if gpcr_field in ("auto", "both") else [gpcr_field]
        for field in fields:
            score = score_chain_for_field(residues, field)
            diagnostics.append({"chain": chain, **score})

    if not diagnostics:
        return "", "", {}

    if gpcr_field == "both":
        chain_scores = {}
        for d in diagnostics:
            chain_scores.setdefault(d["chain"], 0)
            chain_scores[d["chain"]] += d["valid_residues"]
        best_chain = max(chain_scores, key=chain_scores.get)
        return best_chain, "both", {"mode": "both", "diagnostics": diagnostics}

    best = sorted(
        diagnostics,
        key=lambda d: (d["n_tm_numbers"], d["valid_residues"], d["field"] == "occupancy"),
        reverse=True,
    )[0]
    return best["chain"], best["field"], best


def annotate_gpcr_values(residues: List[Residue], gpcr_field_used: str) -> None:
    for residue in residues:
        value = None
        field = ""

        if gpcr_field_used == "both":
            occ = residue_gpcr_value(residue, "occupancy")
            bfac = residue_gpcr_value(residue, "bfactor")
            if occ is not None:
                value = occ
                field = "occupancy"
            elif bfac is not None:
                value = bfac
                field = "bfactor"
        else:
            value = residue_gpcr_value(residue, gpcr_field_used)
            field = gpcr_field_used if value is not None else ""

        residue.gpcr_value = value
        residue.gpcr_field_used = field
        residue.tm_number = tm_number_from_gpcr_value(value)


# =============================================================================
# Geometry
# =============================================================================

def coords_from_residues(residues: List[Residue], include_hydrogens: bool = False) -> np.ndarray:
    coords = []
    for residue in residues:
        for atom in residue.atoms:
            if not include_hydrogens and atom.is_hydrogen:
                continue
            coords.append(atom.coord)
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.vstack(coords)


def centroid(coords: np.ndarray) -> np.ndarray:
    return np.mean(coords, axis=0)


def residue_centroid(residue: Residue, transform, include_hydrogens: bool = False) -> np.ndarray:
    coords = coords_from_residues([residue], include_hydrogens=include_hydrogens)
    return centroid(transform(coords))


def tm_residues_all(residues: List[Residue]) -> List[Residue]:
    return [r for r in residues if r.tm_number in {1, 2, 3, 4, 5, 6, 7}]


def select_ecl2(residues: List[Residue]) -> Tuple[List[Residue], Optional[Residue], Optional[Residue]]:
    tm4 = [r for r in residues if r.tm_number == 4]
    tm5 = [r for r in residues if r.tm_number == 5]

    if not tm4 or not tm5:
        return [], None, None

    tm4_last = max(tm4, key=lambda r: r.order)
    tm5_first = min(tm5, key=lambda r: r.order)

    if tm4_last.order >= tm5_first.order:
        return [], tm4_last, tm5_first

    ecl2 = [r for r in residues if tm4_last.order < r.order < tm5_first.order and r.atoms]
    return ecl2, tm4_last, tm5_first


def build_tm_alignment(
    tm_residues: List[Residue],
    ecl2_residues: List[Residue],
    include_hydrogens: bool = False,
) -> Dict[str, object]:
    tm_coords = coords_from_residues(tm_residues, include_hydrogens=include_hydrogens)
    ecl2_coords = coords_from_residues(ecl2_residues, include_hydrogens=include_hydrogens)

    if tm_coords.shape[0] < 3:
        raise ValueError("Too few TM atoms for alignment")
    if ecl2_coords.shape[0] < 1:
        raise ValueError("No ECL2 atoms for orientation")

    tm_center = centroid(tm_coords)
    centered = tm_coords - tm_center

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    z_axis = vt[0].astype(float)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Orientar el eje Z para que el ECL2 quede en Z positiva
    ecl2_center_original = centroid(ecl2_coords)
    ecl2_projection = float(np.dot(ecl2_center_original - tm_center, z_axis))
    if ecl2_projection < 0:
        z_axis = -z_axis

    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, z_axis)) > 0.90:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    basis = np.vstack([x_axis, y_axis, z_axis])
    variances = singular_values ** 2
    explained = float(variances[0] / np.sum(variances)) if np.sum(variances) > 0 else np.nan

    def transform(coords: np.ndarray) -> np.ndarray:
        return (coords - tm_center) @ basis.T

    return {
        "tm_center_original": tm_center,
        "basis": basis,
        "x_axis_original": x_axis,
        "y_axis_original": y_axis,
        "z_axis_original": z_axis,
        "pca_first_axis_explained_variance": explained,
        "transform": transform,
    }


def summarize_reference_residues(ref_residues: List[Residue]) -> Tuple[str, str]:
    labels = ";".join(r.resid_label for r in ref_residues)
    gpcrs = ";".join("" if r.gpcr_value is None else f"{r.gpcr_value:.2f}" for r in ref_residues)
    return labels, gpcrs


def select_tm_neg_per_tm_residues(
    tm_residues: List[Residue],
    transform,
    n_per_tm: int = 3,
    include_hydrogens: bool = False,
) -> Tuple[List[Residue], Dict[int, List[Residue]], Dict[int, List[float]]]:
    """
    Selecciona los n_per_tm residuos con Z más negativa dentro de cada TM.
    Si están detectados TM1-TM7 y n_per_tm=3, devuelve 21 residuos.
    """
    selected_all: List[Residue] = []
    selected_by_tm: Dict[int, List[Residue]] = {}
    selected_z_by_tm: Dict[int, List[float]] = {}

    for tm in range(1, 8):
        tm_specific = [r for r in tm_residues if r.tm_number == tm]
        rows = []
        for residue in tm_specific:
            z = float(residue_centroid(residue, transform, include_hydrogens=include_hydrogens)[2])
            rows.append((z, residue))
        rows.sort(key=lambda x: x[0])

        selected = [residue for z, residue in rows[:n_per_tm]]
        selected_z = [z for z, residue in rows[:n_per_tm]]

        selected_by_tm[tm] = selected
        selected_z_by_tm[tm] = selected_z
        selected_all.extend(selected)

    return selected_all, selected_by_tm, selected_z_by_tm


def get_deepest_and_closest_ecl2_residues(
    ecl2: List[Residue],
    ref_centroid: np.ndarray,
    transform,
    include_hydrogens: bool = False,
) -> Dict[str, object]:
    rows = []
    for residue in ecl2:
        rc = residue_centroid(residue, transform, include_hydrogens=include_hydrogens)
        delta = rc - ref_centroid
        rows.append({
            "residue": residue,
            "centroid": rc,
            "delta": delta,
            "delta_z": float(delta[2]),
            "abs_delta_z": float(abs(delta[2])),
            "dist3d": float(np.linalg.norm(delta)),
        })

    deepest = min(rows, key=lambda r: r["delta_z"])
    closest = min(rows, key=lambda r: r["dist3d"])

    return {
        "all_rows": rows,
        "deepest_z": deepest,
        "closest_3d": closest,
    }


def get_ecl2_closest_n_residue_centroid(
    ecl2: List[Residue],
    ref_centroid: np.ndarray,
    transform,
    n_closest: int = 3,
    include_hydrogens: bool = False,
) -> Dict[str, object]:
    """
    Selecciona los n_closest residuos del ECL2 cuyos centroides de residuo
    están más cerca en 3D del centroide de referencia basal.

    Después calcula el centroide all-atom conjunto de esos residuos ECL2.

    Esta es la métrica principal de V8.
    """
    residue_rows = []

    for residue in ecl2:
        rc = residue_centroid(residue, transform, include_hydrogens=include_hydrogens)
        delta = rc - ref_centroid
        residue_rows.append({
            "residue": residue,
            "residue_centroid": rc,
            "residue_delta_z": float(delta[2]),
            "residue_dist3d": float(np.linalg.norm(delta)),
        })

    residue_rows.sort(key=lambda r: r["residue_dist3d"])

    n = min(n_closest, len(residue_rows))
    selected_rows = residue_rows[:n]
    selected_residues = [r["residue"] for r in selected_rows]

    selected_coords = transform(
        coords_from_residues(selected_residues, include_hydrogens=include_hydrogens)
    )
    selected_centroid = centroid(selected_coords)
    selected_delta = selected_centroid - ref_centroid

    return {
        "n_requested": n_closest,
        "n_selected": n,
        "selected_rows": selected_rows,
        "selected_residues": selected_residues,
        "selected_centroid": selected_centroid,
        "delta": selected_delta,
        "delta_z": float(selected_delta[2]),
        "abs_delta_z": float(abs(selected_delta[2])),
        "dist3d": float(np.linalg.norm(selected_delta)),
        "selected_residue_labels": ";".join(r.resid_label for r in selected_residues),
        "selected_residue_resnames": ";".join(r.resname for r in selected_residues),
        "selected_residue_oneletters": "".join(aa3_to_aa1(r.resname) for r in selected_residues),
        "selected_residue_resseqs": ";".join(str(r.resseq) for r in selected_residues),
        "selected_residue_distances_3d": ";".join(f"{r['residue_dist3d']:.3f}" for r in selected_rows),
        "selected_residue_delta_z_values": ";".join(f"{r['residue_delta_z']:.3f}" for r in selected_rows),
    }


# =============================================================================
# PDB writing
# =============================================================================

def format_pdb_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.00,
    bfactor: float = 0.00,
    element: str = "C",
    record: str = "HETATM",
) -> str:
    return (
        f"{record:<6}{serial:5d} "
        f"{atom_name:<4}"
        f" "
        f"{resname:>3} {chain:1s}"
        f"{resseq:4d}"
        f"    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}"
        f"          {element:>2s}\n"
    )


def write_aligned_pdb(original_path: Path, by_chain: Dict[str, List[Residue]], transform, out_path: Path) -> None:
    serial_to_coord = {}
    for residues in by_chain.values():
        for residue in residues:
            for atom in residue.atoms:
                aligned = transform(atom.coord.reshape(1, 3))[0]
                serial_to_coord[atom.serial] = aligned

    output_lines = []
    with original_path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                serial = safe_int(line[6:11]) if len(line) >= 11 else None
                if serial is not None and serial in serial_to_coord and len(line) >= 54:
                    x, y, z = serial_to_coord[serial]
                    new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                    output_lines.append(new_line if new_line.endswith("\n") else new_line + "\n")
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

    output_lines.append("REMARK Aligned by PCA of TM1-TM7 atoms; ECL2 side forced to +Z\n")
    output_lines.append("REMARK Reference base = 3 most negative-Z residues PER TM; 21 residues total if TM1-TM7 present\n")
    output_lines.append(f"REMARK Original file: {original_path.name}\n")
    out_path.write_text("".join(output_lines), encoding="utf-8")


def write_marker_pdb(
    out_path: Path,
    tm_centroid: np.ndarray,
    tm_neg_per_tm_centroid: np.ndarray,
    ecl2_centroid: np.ndarray,
    closest3res_centroid: np.ndarray,
    deepest_centroid: np.ndarray,
    closest3d_centroid: np.ndarray,
    closest3residue_labels: str,
    deepest_residue_label: str,
    closest3d_residue_label: str,
) -> None:
    lines = []
    lines.append("REMARK Pseudoatoms for visual inspection after TM alignment\n")
    lines.append("REMARK TMC = TM centroid\n")
    lines.append("REMARK BAS = centroid of 3 most negative-Z residues PER TM; 21 residues total if TM1-TM7 present\n")
    lines.append("REMARK ECL = global ECL2 centroid\n")
    lines.append(f"REMARK CEN = centroid of 3 ECL2 residues closest to BAS: {closest3residue_labels}\n")
    lines.append(f"REMARK DEP = deepest ECL2 residue in Z: {deepest_residue_label}\n")
    lines.append(f"REMARK C3D = closest ECL2 residue in 3D: {closest3d_residue_label}\n")

    markers = [
        ("TMC", tm_centroid),
        ("BAS", tm_neg_per_tm_centroid),
        ("ECL", ecl2_centroid),
        ("CEN", closest3res_centroid),
        ("DEP", deepest_centroid),
        ("C3D", closest3d_centroid),
    ]

    for i, (name, coord) in enumerate(markers, start=1):
        lines.append(
            format_pdb_atom_line(
                serial=i,
                atom_name=name,
                resname="MKR",
                chain="Z",
                resseq=i,
                x=float(coord[0]),
                y=float(coord[1]),
                z=float(coord[2]),
                occupancy=1.00,
                bfactor=0.00,
                element="C",
                record="HETATM",
            )
        )

    lines.append("END\n")
    out_path.write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Metadata: optional sensory metadata + class/family table
# =============================================================================

def normalize_receptor_key(text: str) -> str:
    """
    Normaliza IDs para poder emparejar:
        5ht1a, 5ht1a_human, 5HT1A, AA2AR, o51e2, OR51E2, etc.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    if not s:
        return ""
    s = s.replace("_human", "")
    s = s.replace("-", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def read_metadata(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Metadata opcional antigua, normalmente con entry_name y columnas sensoriales.
    Se mantiene por compatibilidad, pero el análisis por clases/familias puede
    hacerse solo con --families.
    """
    if path is None or not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "entry_name" not in (reader.fieldnames or []):
            return {}

        metadata = {}
        for row in reader:
            entry = row.get("entry_name", "")
            if entry:
                metadata[entry] = row
                metadata[normalize_receptor_key(entry)] = row
        return metadata


def sniff_delimiter(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    if "\t" in text:
        return "\t"
    if ";" in text and text.count(";") > text.count(","):
        return ";"
    return ","


def read_family_annotations(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Lee families.txt / tabla GPCRdb con columnas:
        GPCRs (UniProt)
        GPCRs (Gene name)
        Receptor family
        Ligand type
        Class

    Devuelve un diccionario con múltiples claves normalizadas:
        uniprot, gene, uniprot_human, gene_human, etc.
    """
    if path is None or not path.exists():
        return {}

    delimiter = sniff_delimiter(path)
    fam: Dict[str, Dict[str, str]] = {}

    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []

        # Búsqueda flexible por si el encabezado cambia ligeramente
        def find_col(candidates: List[str]) -> Optional[str]:
            lower_map = {c.lower().strip(): c for c in fieldnames}
            for cand in candidates:
                key = cand.lower().strip()
                if key in lower_map:
                    return lower_map[key]
            for c in fieldnames:
                lc = c.lower()
                if all(part in lc for part in candidates[0].lower().split()):
                    return c
            return None

        uniprot_col = find_col(["GPCRs (UniProt)", "UniProt", "GPCRs"])
        gene_col = find_col(["GPCRs (Gene name)", "Gene name", "Gene"])
        receptor_family_col = find_col(["Receptor family"])
        ligand_type_col = find_col(["Ligand type"])
        class_col = find_col(["Class"])

        required = [uniprot_col, receptor_family_col, ligand_type_col, class_col]
        if any(c is None for c in required):
            raise ValueError(
                "No he podido detectar las columnas necesarias en families.txt. "
                f"Columnas encontradas: {fieldnames}"
            )

        for row in reader:
            uniprot = (row.get(uniprot_col, "") or "").strip()
            gene = (row.get(gene_col, "") or "").strip() if gene_col else ""

            if not uniprot and not gene:
                continue

            record = {
                "families_found": "yes",
                "families_uniprot": uniprot,
                "families_gene_name": gene,
                "receptor_family": row.get(receptor_family_col, "") or "",
                "ligand_type": row.get(ligand_type_col, "") or "",
                "receptor_class": row.get(class_col, "") or "",
            }

            keys = set()
            for value in [uniprot, gene]:
                key = normalize_receptor_key(value)
                if key:
                    keys.add(key)
                    keys.add(normalize_receptor_key(key + "_human"))

            for key in keys:
                fam[key] = record

    return fam


def find_family_record_for_result(result: Dict[str, object], families: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Busca la anotación de clase/familia usando varias claves posibles.
    """
    candidate_values = [
        str(result.get("receptor_id", "")),
        str(result.get("entry_name_inferred", "")),
        str(result.get("pdb_file", "")),
    ]

    # Del nombre del PDB se puede inferir de nuevo por robustez
    pdb_file = str(result.get("pdb_file", ""))
    if pdb_file:
        receptor_id, _ = infer_receptor_and_state(pdb_file)
        candidate_values.append(receptor_id)
        candidate_values.append(f"{receptor_id}_human")

    for value in candidate_values:
        key = normalize_receptor_key(value)
        if key in families:
            return families[key]

    return None


def add_metadata_to_result(
    result: Dict[str, object],
    metadata: Dict[str, Dict[str, str]],
    families: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """
    Añade metadata antigua si existe y, sobre todo, clase/familia/ligand_type
    desde families.txt.

    Prioridad:
        1) metadata antigua, si existe
        2) families.txt para receptor_class, receptor_family, ligand_type
    """
    entry = str(result.get("entry_name_inferred", ""))
    receptor_id = str(result.get("receptor_id", ""))

    meta = metadata.get(entry) or metadata.get(normalize_receptor_key(entry)) or metadata.get(normalize_receptor_key(receptor_id))

    if meta:
        result["metadata_found"] = "yes"
        for col in [
            "receptor_name", "accession", "receptor_class", "ligand_type",
            "receptor_family", "subfamily", "species", "sensory_related",
            "associated_sense", "sensory_modality_detail", "sensory_role",
            "classification_rule",
        ]:
            if col in meta:
                result[col] = meta.get(col, "")
    else:
        result["metadata_found"] = "no"

    # No crear clasificación sensorial si no existe; este script ya no agrupa por sensorial/no sensorial.
    result.setdefault("sensory_related", "")
    result.setdefault("associated_sense", "")

    fam_record = find_family_record_for_result(result, families or {})
    if fam_record:
        result["families_found"] = "yes"
        for col in ["families_uniprot", "families_gene_name", "receptor_class", "receptor_family", "ligand_type"]:
            if fam_record.get(col, ""):
                result[col] = fam_record.get(col, "")
    else:
        result["families_found"] = "no"
        result.setdefault("receptor_class", "unannotated")
        result.setdefault("receptor_family", "unannotated")
        result.setdefault("ligand_type", "unannotated")

# =============================================================================
# Main analysis per PDB
# =============================================================================

def analyze_one_pdb(
    path: Path,
    gpcr_field: str,
    n_per_tm: int,
    include_hydrogens: bool,
    aligned_pdb_dir: Path,
    marker_pdb_dir: Path,
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]], Optional[str]]:
    receptor_id, state = infer_receptor_and_state(path.name)

    by_chain = parse_pdb(path)
    if not by_chain:
        return None, None, "No ATOM/HETATM records found"

    chain_id, field_used, field_diag = choose_best_gpcr_field(by_chain, gpcr_field)
    if not chain_id or chain_id not in by_chain:
        return None, field_diag, "No GPCRdb/Ballesteros field detected"

    residues = by_chain[chain_id]
    annotate_gpcr_values(residues, field_used)

    tms = tm_residues_all(residues)
    ecl2, tm4_last, tm5_first = select_ecl2(residues)

    if len(tms) < 30:
        return None, field_diag, f"Too few TM residues detected: {len(tms)}"

    if not ecl2:
        return None, field_diag, "No ECL2 detected between TM4 and TM5"

    try:
        alignment = build_tm_alignment(tms, ecl2, include_hydrogens=include_hydrogens)
    except Exception as exc:
        return None, field_diag, f"Alignment failed: {exc}"

    transform = alignment["transform"]

    tm_coords = transform(coords_from_residues(tms, include_hydrogens=include_hydrogens))
    ecl2_coords = transform(coords_from_residues(ecl2, include_hydrogens=include_hydrogens))

    tm_centroid = centroid(tm_coords)
    ecl2_centroid = centroid(ecl2_coords)

    # Corrected reference: n_per_tm most negative residues from EACH TM
    ref_residues, ref_by_tm, ref_z_by_tm = select_tm_neg_per_tm_residues(
        tm_residues=tms,
        transform=transform,
        n_per_tm=n_per_tm,
        include_hydrogens=include_hydrogens,
    )

    ref_coords = transform(coords_from_residues(ref_residues, include_hydrogens=include_hydrogens))
    ref_centroid = centroid(ref_coords)
    ref_labels, ref_gpcrs = summarize_reference_residues(ref_residues)

    ecl2_res_metrics = get_deepest_and_closest_ecl2_residues(
        ecl2=ecl2,
        ref_centroid=ref_centroid,
        transform=transform,
        include_hydrogens=include_hydrogens,
    )

    # V8 main metric:
    # centroid of atoms from the 3 ECL2 residues closest in 3D to the 21-residue TM reference centroid
    ecl2_closest3res_metrics = get_ecl2_closest_n_residue_centroid(
        ecl2=ecl2,
        ref_centroid=ref_centroid,
        transform=transform,
        n_closest=3,
        include_hydrogens=include_hydrogens,
    )

    deepest = ecl2_res_metrics["deepest_z"]
    closest3d = ecl2_res_metrics["closest_3d"]
    deepest_res = deepest["residue"]
    closest3d_res = closest3d["residue"]

    tm_z_min = float(np.min(tm_coords[:, 2]))
    tm_z_max = float(np.max(tm_coords[:, 2]))
    tm_z_span = tm_z_max - tm_z_min
    tm_counts = Counter(r.tm_number for r in tms)

    aligned_pdb_dir.mkdir(parents=True, exist_ok=True)
    marker_pdb_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = aligned_pdb_dir / path.name
    marker_path = marker_pdb_dir / f"{path.stem}_markers.pdb"

    write_aligned_pdb(path, by_chain, transform, aligned_path)
    write_marker_pdb(
        out_path=marker_path,
        tm_centroid=tm_centroid,
        tm_neg_per_tm_centroid=ref_centroid,
        ecl2_centroid=ecl2_centroid,
        closest3res_centroid=ecl2_closest3res_metrics["selected_centroid"],
        deepest_centroid=deepest["centroid"],
        closest3d_centroid=closest3d["centroid"],
        closest3residue_labels=ecl2_closest3res_metrics["selected_residue_labels"],
        deepest_residue_label=deepest_res.resid_label,
        closest3d_residue_label=closest3d_res.resid_label,
    )

    closest3res_delta_z = float(ecl2_closest3res_metrics["delta_z"])
    deepest_delta_z = float(deepest["delta_z"])
    closest3d_delta_z = float(closest3d["delta_z"])
    ecl2_centroid_delta_z_to_ref = float(ecl2_centroid[2] - ref_centroid[2])

    # Human-readable selected residues per TM
    ref_residues_by_tm_text = {}
    ref_gpcrs_by_tm_text = {}
    ref_z_by_tm_text = {}
    for tm in range(1, 8):
        labels, gpcrs = summarize_reference_residues(ref_by_tm.get(tm, []))
        ref_residues_by_tm_text[f"tm{tm}_neg{n_per_tm}_reference_residues"] = labels
        ref_gpcrs_by_tm_text[f"tm{tm}_neg{n_per_tm}_reference_gpcr_values"] = gpcrs
        ref_z_by_tm_text[f"tm{tm}_neg{n_per_tm}_reference_z_values"] = ";".join(f"{z:.3f}" for z in ref_z_by_tm.get(tm, []))

    result = {
        "pdb_file": path.name,
        "receptor_id": receptor_id,
        "entry_name_inferred": f"{receptor_id}_human",
        "state": state,
        "chain": chain_id,
        "gpcr_field_chosen": field_used,
        "alignment_method": "PCA_TM1_TM7_all_atoms_to_Z_ECL2_positive",
        "reference_definition": f"{n_per_tm}_most_negative_Z_residues_per_TM",
        "include_hydrogens_in_calculation": include_hydrogens,
        "reference_n_residues_per_tm": n_per_tm,
        "reference_total_residues": len(ref_residues),

        "aligned_pdb_path": str(aligned_path),
        "marker_pdb_path": str(marker_path),

        "n_residues_total_chain": len(residues),
        "n_tm_residues": len(tms),
        "n_tm1_residues": tm_counts.get(1, 0),
        "n_tm2_residues": tm_counts.get(2, 0),
        "n_tm3_residues": tm_counts.get(3, 0),
        "n_tm4_residues": tm_counts.get(4, 0),
        "n_tm5_residues": tm_counts.get(5, 0),
        "n_tm6_residues": tm_counts.get(6, 0),
        "n_tm7_residues": tm_counts.get(7, 0),
        "n_ecl2_residues": len(ecl2),

        "tm_centroid_x_aligned": float(tm_centroid[0]),
        "tm_centroid_y_aligned": float(tm_centroid[1]),
        "tm_centroid_z_aligned": float(tm_centroid[2]),
        "ecl2_centroid_x_aligned": float(ecl2_centroid[0]),
        "ecl2_centroid_y_aligned": float(ecl2_centroid[1]),
        "ecl2_centroid_z_aligned": float(ecl2_centroid[2]),

        "tm_z_min_aligned": tm_z_min,
        "tm_z_max_aligned": tm_z_max,
        "tm_z_span_aligned": tm_z_span,

        f"tm_neg{n_per_tm}_per_tm_reference_residues": ref_labels,
        f"tm_neg{n_per_tm}_per_tm_reference_gpcr_values": ref_gpcrs,
        f"tm_neg{n_per_tm}_per_tm_centroid_x_aligned": float(ref_centroid[0]),
        f"tm_neg{n_per_tm}_per_tm_centroid_y_aligned": float(ref_centroid[1]),
        f"tm_neg{n_per_tm}_per_tm_centroid_z_aligned": float(ref_centroid[2]),

        # V8 main metric: centroid of atoms from the 3 ECL2 residues closest to the 21-residue TM reference centroid
        f"ecl2_closest3res_centroid_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned": closest3res_delta_z,
        f"ecl2_closest3res_centroid_abs_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned": abs(closest3res_delta_z),
        f"ecl2_closest3res_centroid_delta_z_norm_by_tm_span_to_tm_neg{n_per_tm}_per_tm_aligned": closest3res_delta_z / tm_z_span if tm_z_span > 0 else np.nan,
        f"ecl2_closest3res_centroid_3d_distance_to_tm_neg{n_per_tm}_per_tm_aligned": float(ecl2_closest3res_metrics["dist3d"]),
        "ecl2_closest3res_n_selected": ecl2_closest3res_metrics["n_selected"],
        "ecl2_closest3res_residues": ecl2_closest3res_metrics["selected_residue_labels"],
        "ecl2_closest3res_resnames": ecl2_closest3res_metrics["selected_residue_resnames"],
        "ecl2_closest3res_oneletters": ecl2_closest3res_metrics["selected_residue_oneletters"],
        "ecl2_closest3res_resseqs": ecl2_closest3res_metrics["selected_residue_resseqs"],
        "ecl2_closest3res_individual_3d_distances_to_reference": ecl2_closest3res_metrics["selected_residue_distances_3d"],
        "ecl2_closest3res_individual_delta_z_to_reference": ecl2_closest3res_metrics["selected_residue_delta_z_values"],
        "ecl2_closest3res_centroid_x_aligned": float(ecl2_closest3res_metrics["selected_centroid"][0]),
        "ecl2_closest3res_centroid_y_aligned": float(ecl2_closest3res_metrics["selected_centroid"][1]),
        "ecl2_closest3res_centroid_z_aligned": float(ecl2_closest3res_metrics["selected_centroid"][2]),

        # Controls kept for diagnostics
        f"deepest_ecl2_residue_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned": deepest_delta_z,
        f"deepest_ecl2_residue_abs_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned": abs(deepest_delta_z),
        f"deepest_ecl2_residue_delta_z_norm_by_tm_span_to_tm_neg{n_per_tm}_per_tm_aligned": deepest_delta_z / tm_z_span if tm_z_span > 0 else np.nan,
        f"deepest_ecl2_residue_3d_distance_to_tm_neg{n_per_tm}_per_tm_aligned": float(deepest["dist3d"]),
        "deepest_ecl2_residue": deepest_res.resid_label,
        "deepest_ecl2_residue_resname": deepest_res.resname,
        "deepest_ecl2_residue_oneletter": aa3_to_aa1(deepest_res.resname),
        "deepest_ecl2_residue_resseq": deepest_res.resseq,
        "deepest_ecl2_residue_centroid_x_aligned": float(deepest["centroid"][0]),
        "deepest_ecl2_residue_centroid_y_aligned": float(deepest["centroid"][1]),
        "deepest_ecl2_residue_centroid_z_aligned": float(deepest["centroid"][2]),

        f"closest3d_ecl2_residue_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned": closest3d_delta_z,
        f"closest3d_ecl2_residue_3d_distance_to_tm_neg{n_per_tm}_per_tm_aligned": float(closest3d["dist3d"]),
        "closest3d_ecl2_residue": closest3d_res.resid_label,
        "closest3d_ecl2_residue_resname": closest3d_res.resname,
        "closest3d_ecl2_residue_oneletter": aa3_to_aa1(closest3d_res.resname),
        "closest3d_ecl2_residue_resseq": closest3d_res.resseq,

        f"ecl2_centroid_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned": ecl2_centroid_delta_z_to_ref,
        f"ecl2_centroid_3d_distance_to_tm_neg{n_per_tm}_per_tm_aligned": float(np.linalg.norm(ecl2_centroid - ref_centroid)),

        "sanity_ecl2_centroid_should_be_positive_z": "ok" if float(ecl2_centroid[2]) > 0 else "warning_ecl2_centroid_not_positive",
        f"sanity_ecl2_closest3res_delta_z_to_tm_neg{n_per_tm}_per_tm_should_be_positive": "ok" if closest3res_delta_z >= -1e-6 else "warning_ecl2_closest3res_below_reference",
        f"sanity_deepest_ecl2_delta_z_to_tm_neg{n_per_tm}_per_tm_should_be_positive": "ok" if deepest_delta_z >= -1e-6 else "warning_deepest_ecl2_below_reference",
        "sanity_reference_total_should_be_21": "ok" if len(ref_residues) == 21 else f"warning_reference_total_{len(ref_residues)}",

        "pca_first_axis_explained_variance": alignment["pca_first_axis_explained_variance"],
        "pca_z_axis_original_x": float(alignment["z_axis_original"][0]),
        "pca_z_axis_original_y": float(alignment["z_axis_original"][1]),
        "pca_z_axis_original_z": float(alignment["z_axis_original"][2]),

        "tm4_last_residue": tm4_last.resid_label if tm4_last else "",
        "tm4_last_gpcr_value": f"{tm4_last.gpcr_value:.2f}" if tm4_last and tm4_last.gpcr_value is not None else "",
        "ecl2_start_residue": ecl2[0].resid_label if ecl2 else "",
        "ecl2_end_residue": ecl2[-1].resid_label if ecl2 else "",
        "tm5_first_residue": tm5_first.resid_label if tm5_first else "",
        "tm5_first_gpcr_value": f"{tm5_first.gpcr_value:.2f}" if tm5_first and tm5_first.gpcr_value is not None else "",
        "ecl2_sequence": "".join(aa3_to_aa1(r.resname) for r in ecl2),
        "tm_numbers_detected": ";".join(str(i) for i in sorted(tm_counts)),
    }

    result.update(ref_residues_by_tm_text)
    result.update(ref_gpcrs_by_tm_text)
    result.update(ref_z_by_tm_text)

    field_row = {
        "pdb_file": path.name,
        "receptor_id": receptor_id,
        "state": state,
        "chain": chain_id,
        "gpcr_field_chosen": field_used,
        "n_tm_residues": len(tms),
        "n_ecl2_residues": len(ecl2),
        "tm_numbers_detected": ";".join(str(i) for i in sorted(tm_counts)),
        "reference_definition": f"{n_per_tm}_most_negative_Z_residues_per_TM",
        "reference_total_residues": len(ref_residues),
        "sanity_reference_total_should_be_21": result["sanity_reference_total_should_be_21"],
        "tm4_last_residue": result["tm4_last_residue"],
        "tm5_first_residue": result["tm5_first_residue"],
        "sanity_ecl2_centroid_should_be_positive_z": result["sanity_ecl2_centroid_should_be_positive_z"],
        f"sanity_ecl2_closest3res_delta_z_to_tm_neg{n_per_tm}_per_tm_should_be_positive": result[f"sanity_ecl2_closest3res_delta_z_to_tm_neg{n_per_tm}_per_tm_should_be_positive"],
        f"sanity_deepest_ecl2_delta_z_to_tm_neg{n_per_tm}_per_tm_should_be_positive": result[f"sanity_deepest_ecl2_delta_z_to_tm_neg{n_per_tm}_per_tm_should_be_positive"],
        "field_diagnostic": str(field_diag),
    }

    return result, field_row, None


# =============================================================================
# Output helpers
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_name(n_per_tm: int) -> str:
    return f"ecl2_closest3res_centroid_delta_z_to_tm_neg{n_per_tm}_per_tm_aligned"


def to_float_or_nan(value) -> float:
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return np.nan
        return val
    except Exception:
        return np.nan


def make_summary_by_group(
    results: List[Dict[str, object]],
    n_per_tm: int,
    group_col: str,
    include_state: bool = True,
    min_group_n: int = 1,
) -> List[Dict[str, object]]:
    """
    Resumen descriptivo por clase/familia/ligand_type, no por sensorialidad.
    """
    metric = metric_name(n_per_tm)
    groups: Dict[Tuple[str, str], List[float]] = {}

    for row in results:
        value = to_float_or_nan(row.get(metric))
        if np.isnan(value):
            continue

        state = row.get("state", "unknown") if include_state else "all"
        group_value = row.get(group_col, "") or "unannotated"
        key = (str(state), str(group_value))
        groups.setdefault(key, []).append(value)

    summary = []
    for (state, group_value), values in sorted(groups.items()):
        arr = np.array(values, dtype=float)
        if len(arr) < min_group_n:
            continue
        summary.append({
            "state": state,
            "group_column": group_col,
            group_col: group_value,
            "metric": metric,
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "q1": float(np.quantile(arr, 0.25)),
            "q3": float(np.quantile(arr, 0.75)),
            "max": float(np.max(arr)),
        })

    return summary


def make_summary(results: List[Dict[str, object]], n_per_tm: int) -> List[Dict[str, object]]:
    """
    Resumen antiguo por associated_sense. Se mantiene solo por compatibilidad.
    Para el nuevo análisis usa make_summary_by_group(..., group_col='receptor_class')
    y make_summary_by_group(..., group_col='receptor_family').
    """
    return make_summary_by_group(results, n_per_tm, "associated_sense", include_state=True)


def holm_adjust(pvalues: List[float]) -> List[float]:
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [np.nan] * m
    running_max = 0.0

    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * p
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    greater = 0
    less = 0
    for xi in x:
        greater += int(np.sum(xi > y))
        less += int(np.sum(xi < y))
    return float((greater - less) / (len(x) * len(y)))


def make_group_stats(
    results: List[Dict[str, object]],
    n_per_tm: int,
    group_col: str,
    min_group_n: int = 3,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Tests por state y group_col:
        - Kruskal-Wallis global
        - Mann-Whitney posthoc por pares con Holm
        - Cliff's delta

    Si scipy no está instalado, devuelve filas con note.
    """
    metric = metric_name(n_per_tm)

    try:
        from scipy.stats import kruskal, mannwhitneyu
    except Exception:
        states = sorted(set(str(r.get("state", "unknown")) for r in results))
        global_rows = [{
            "state": state,
            "group_column": group_col,
            "metric": metric,
            "test": "Kruskal-Wallis",
            "min_group_n": min_group_n,
            "n_groups": 0,
            "H": np.nan,
            "p_value": np.nan,
            "significant_0.05": "",
            "note": "scipy_not_installed",
        } for state in states]
        return global_rows, []

    # Organizar valores: state -> group -> values
    values_by_state_group: Dict[str, Dict[str, List[float]]] = {}
    for row in results:
        value = to_float_or_nan(row.get(metric))
        if np.isnan(value):
            continue
        state = str(row.get("state", "unknown"))
        group_value = str(row.get(group_col, "") or "unannotated")
        values_by_state_group.setdefault(state, {}).setdefault(group_value, []).append(value)

    global_rows: List[Dict[str, object]] = []
    posthoc_rows: List[Dict[str, object]] = []

    for state, group_map in sorted(values_by_state_group.items()):
        valid_groups = {
            g: np.array(v, dtype=float)
            for g, v in sorted(group_map.items())
            if len(v) >= min_group_n
        }

        if len(valid_groups) < 2:
            global_rows.append({
                "state": state,
                "group_column": group_col,
                "metric": metric,
                "test": "Kruskal-Wallis",
                "min_group_n": min_group_n,
                "n_groups": len(valid_groups),
                "H": np.nan,
                "p_value": np.nan,
                "significant_0.05": "",
                "note": "fewer_than_2_groups_after_min_group_n_filter",
            })
            continue

        group_names = list(valid_groups.keys())
        arrays = [valid_groups[g] for g in group_names]

        H, p_global = kruskal(*arrays)
        global_rows.append({
            "state": state,
            "group_column": group_col,
            "metric": metric,
            "test": "Kruskal-Wallis",
            "min_group_n": min_group_n,
            "n_groups": len(valid_groups),
            "H": float(H),
            "p_value": float(p_global),
            "significant_0.05": "yes" if p_global < 0.05 else "no",
            "note": "",
        })

        raw_p = []
        pair_rows = []

        for g1, g2 in itertools.combinations(group_names, 2):
            x = valid_groups[g1]
            y = valid_groups[g2]
            U, p_pair = mannwhitneyu(x, y, alternative="two-sided")
            raw_p.append(float(p_pair))
            pair_rows.append({
                "state": state,
                "group_column": group_col,
                "metric": metric,
                "group1": g1,
                "group2": g2,
                "n1": len(x),
                "n2": len(y),
                "median1": float(np.median(x)),
                "median2": float(np.median(y)),
                "median_diff_group1_minus_group2": float(np.median(x) - np.median(y)),
                "mean1": float(np.mean(x)),
                "mean2": float(np.mean(y)),
                "mean_diff_group1_minus_group2": float(np.mean(x) - np.mean(y)),
                "mannwhitney_U": float(U),
                "p_raw": float(p_pair),
                "cliffs_delta_group1_vs_group2": cliffs_delta(x, y),
            })

        p_holm = holm_adjust(raw_p)
        for row, p_adj in zip(pair_rows, p_holm):
            row["p_holm"] = p_adj
            row["significant_holm_0.05"] = "yes" if p_adj < 0.05 else "no"
            posthoc_rows.append(row)

    return global_rows, posthoc_rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align GPCR PDBs and measure ECL2 axial Z depth using the centroid of the 3 ECL2 residues closest to the 3/TM basal reference."
    )
    parser.add_argument("--pdb-dir", required=True, help="Folder containing PDB files.")
    parser.add_argument("--metadata", default=None, help="Optional CSV with entry_name and sensory annotation. Kept for compatibility.")
    parser.add_argument("--families", default="families.txt", help="TSV/TXT with GPCRs (UniProt), Receptor family, Ligand type and Class. Default: families.txt")
    parser.add_argument("--min-group-n", type=int, default=3, help="Minimum n per group for Kruskal/posthoc tests. Default: 3")
    parser.add_argument("--out-prefix", default="ecl2_depth_v8", help="Prefix for output files.")
    parser.add_argument("--pattern", default="*.pdb", help="Glob pattern for PDB files. Default: *.pdb")
    parser.add_argument(
        "--gpcr-field",
        choices=["auto", "occupancy", "bfactor", "both"],
        default="auto",
        help="Where to read GPCRdb/Ballesteros numbers from.",
    )
    parser.add_argument(
        "--n-per-tm",
        type=int,
        default=3,
        help="Number of most-negative-Z residues selected inside each TM. Default: 3, so 21 total.",
    )
    parser.add_argument(
        "--include-hydrogens",
        action="store_true",
        help="Include hydrogens in PCA/centroid calculations. Default: heavy atoms only.",
    )

    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    metadata_path = Path(args.metadata) if args.metadata else None
    metadata = read_metadata(metadata_path)

    families_path = Path(args.families) if args.families else None
    if families_path is not None and not families_path.exists():
        # Si se ejecuta desde otro directorio, intentar encontrar families.txt dentro del pdb_dir
        candidate = pdb_dir / families_path.name
        if candidate.exists():
            families_path = candidate
    families = read_family_annotations(families_path)

    pdb_files = sorted(pdb_dir.glob(args.pattern))
    if not pdb_files:
        raise SystemExit(f"No PDB files found in {pdb_dir} with pattern {args.pattern}")

    aligned_pdb_dir = Path(f"{args.out_prefix}_aligned_pdbs")
    marker_pdb_dir = Path(f"{args.out_prefix}_marker_pdbs")

    results = []
    field_rows = []
    failed_rows = []

    for i, pdb_file in enumerate(pdb_files, start=1):
        result, field_row, error = analyze_one_pdb(
            path=pdb_file,
            gpcr_field=args.gpcr_field,
            n_per_tm=args.n_per_tm,
            include_hydrogens=args.include_hydrogens,
            aligned_pdb_dir=aligned_pdb_dir,
            marker_pdb_dir=marker_pdb_dir,
        )

        if result is None:
            receptor_id, state = infer_receptor_and_state(pdb_file.name)
            failed_rows.append({
                "pdb_file": pdb_file.name,
                "receptor_id": receptor_id,
                "state": state,
                "error": error,
                "field_diagnostic": str(field_row),
            })
        else:
            add_metadata_to_result(result, metadata, families)
            results.append(result)
            field_rows.append(field_row or {})

        if i % 100 == 0:
            print(f"Processed {i}/{len(pdb_files)} PDBs...")

    distances_csv = Path(f"{args.out_prefix}_distances.csv")
    summary_csv = Path(f"{args.out_prefix}_summary_legacy_by_associated_sense.csv")
    summary_class_csv = Path(f"{args.out_prefix}_summary_by_class.csv")
    summary_family_csv = Path(f"{args.out_prefix}_summary_by_family.csv")
    summary_ligand_csv = Path(f"{args.out_prefix}_summary_by_ligand_type.csv")
    kruskal_class_csv = Path(f"{args.out_prefix}_kruskal_by_class.csv")
    kruskal_family_csv = Path(f"{args.out_prefix}_kruskal_by_family.csv")
    kruskal_ligand_csv = Path(f"{args.out_prefix}_kruskal_by_ligand_type.csv")
    posthoc_class_csv = Path(f"{args.out_prefix}_posthoc_by_class_holm.csv")
    posthoc_family_csv = Path(f"{args.out_prefix}_posthoc_by_family_holm.csv")
    posthoc_ligand_csv = Path(f"{args.out_prefix}_posthoc_by_ligand_type_holm.csv")
    field_csv = Path(f"{args.out_prefix}_field_detection.csv")
    failed_csv = Path(f"{args.out_prefix}_failed_files.csv")

    class_global, class_posthoc = make_group_stats(results, args.n_per_tm, "receptor_class", min_group_n=args.min_group_n)
    family_global, family_posthoc = make_group_stats(results, args.n_per_tm, "receptor_family", min_group_n=args.min_group_n)
    ligand_global, ligand_posthoc = make_group_stats(results, args.n_per_tm, "ligand_type", min_group_n=args.min_group_n)

    write_csv(distances_csv, results)
    write_csv(summary_csv, make_summary(results, args.n_per_tm))
    write_csv(summary_class_csv, make_summary_by_group(results, args.n_per_tm, "receptor_class", include_state=True))
    write_csv(summary_family_csv, make_summary_by_group(results, args.n_per_tm, "receptor_family", include_state=True))
    write_csv(summary_ligand_csv, make_summary_by_group(results, args.n_per_tm, "ligand_type", include_state=True))
    write_csv(kruskal_class_csv, class_global)
    write_csv(kruskal_family_csv, family_global)
    write_csv(kruskal_ligand_csv, ligand_global)
    write_csv(posthoc_class_csv, class_posthoc)
    write_csv(posthoc_family_csv, family_posthoc)
    write_csv(posthoc_ligand_csv, ligand_posthoc)
    write_csv(field_csv, field_rows)
    write_csv(failed_csv, failed_rows)

    print("\nDone.")
    print(f"PDBs found: {len(pdb_files)}")
    print(f"PDBs analyzed: {len(results)}")
    print(f"PDBs failed: {len(failed_rows)}")
    print("Reference definition:")
    print(f"  {args.n_per_tm} most negative-Z residues per TM")
    print(f"  expected total if all 7 TMs detected: {args.n_per_tm * 7}")
    print("Primary metric:")
    print(f"  ecl2_closest3res_centroid_delta_z_to_tm_neg{args.n_per_tm}_per_tm_aligned")
    print("Annotation:")
    print(f"  families file: {families_path if families_path and families_path.exists() else 'not_found'}")
    print(f"  family/class records loaded: {len(families)}")
    print(f"  min group n for tests: {args.min_group_n}")
    print("Outputs:")
    print(f"  {distances_csv}")
    print(f"  {summary_class_csv}")
    print(f"  {summary_family_csv}")
    print(f"  {summary_ligand_csv}")
    print(f"  {kruskal_class_csv}")
    print(f"  {kruskal_family_csv}")
    print(f"  {kruskal_ligand_csv}")
    print(f"  {posthoc_class_csv}")
    print(f"  {posthoc_family_csv}")
    print(f"  {posthoc_ligand_csv}")
    print(f"  {summary_csv}")
    print(f"  {field_csv}")
    print(f"  {failed_csv}")
    print(f"  {aligned_pdb_dir}/")
    print(f"  {marker_pdb_dir}/")


if __name__ == "__main__":
    main()
