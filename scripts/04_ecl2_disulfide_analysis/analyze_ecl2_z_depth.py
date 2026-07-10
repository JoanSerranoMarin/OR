from pathlib import Path

script = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_ecl2_z_depth_v6_aligned_debug.py

Objetivo
--------
Alinear todos los PDBs usando los residuos TM1-TM7 y medir la profundidad axial
del ECL2 en Z.

Esta versión corrige el problema conceptual anterior:

    - Después de alinear el eje principal de los TMs con Z, se fuerza que el ECL2
      quede hacia Z positiva.
    - Por tanto, los 3 residuos TM con Z más negativa representan la "base" del
      paquete transmembrana en ese sistema alineado.
    - El ECL2 debería quedar por encima de esa base. Si no, el script lo marca
      con una alerta.

Métrica principal
-----------------
    deepest_ecl2_residue_delta_z_to_tm_neg3_aligned

Definición:
    1) Se alinean los TMs por PCA usando todos los átomos de TM1-TM7.
    2) Se orienta el eje Z para que el centroide del ECL2 quede en Z positiva.
    3) Se seleccionan los N residuos TM con Z más negativa, N=3 por defecto.
    4) Se calcula el centroide all-atom de esos N residuos TM.
    5) Para cada residuo del ECL2, se calcula su centroide all-atom.
    6) Se elige el residuo del ECL2 con menor ΔZ respecto a la referencia TM-negativa.

        ΔZ = Z(residuo ECL2) - Z(centroide TM-negN)

Interpretación:
    menor ΔZ = el residuo más profundo del ECL2 está más cerca de la base TM-negativa
    mayor ΔZ = el ECL2 queda más alejado axialmente de esa base

Outputs
-------
    <out-prefix>_distances.csv
    <out-prefix>_summary.csv
    <out-prefix>_field_detection.csv
    <out-prefix>_failed_files.csv
    <out-prefix>_aligned_pdbs/*.pdb
    <out-prefix>_marker_pdbs/*.pdb

Los marker PDBs contienen pseudoátomos para inspección:
    TMC = centroide de todos los TMs
    BAS = centroide de los N residuos TM más negativos
    ECL = centroide global del ECL2
    DEP = centroide del residuo ECL2 más profundo en Z
    C3D = centroide del residuo ECL2 más cercano en 3D, como control

Uso
---
    python analyze_ecl2_z_depth_v6_aligned_debug.py \
        --pdb-dir /ruta/a/pdbs \
        --metadata gpcr_ecl2_lengths_real_by_receptor.csv \
        --out-prefix ecl2_depth_v6

"""

from __future__ import annotations

import argparse
import csv
import math
import re
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
    def resid_key(self) -> Tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)

    @property
    def resid_label(self) -> str:
        suffix = self.icode if self.icode else ""
        return f"{self.chain}:{self.resname}{self.resseq}{suffix}".strip()


# =============================================================================
# Helpers
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
    return receptor, state


def looks_like_gpcrdb_value(value: Optional[float]) -> bool:
    """
    Detecta valores tipo 4.61, 5.35, 7.53, etc.
    Excluye valores enteros como 1.00, porque suelen ser occupancies estándar.
    """
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
    """
    Lee ATOM/HETATM. No descarta hidrógenos al parsear, para poder escribir
    el PDB alineado completo. Los hidrógenos se pueden excluir del análisis
    mediante coords_from_residues(..., include_hydrogens=False).
    """
    residues: "OrderedDict[Tuple[str, int, str], Residue]" = OrderedDict()
    order_counter = 0

    with path.open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            parsed = None

            # Fixed-width PDB
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

            # Fallback para PDB-like con split
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

            # Solo altloc principal
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
    coords_aligned = transform(coords)
    return centroid(coords_aligned)


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

    ecl2 = [
        r for r in residues
        if tm4_last.order < r.order < tm5_first.order
        and r.atoms
    ]

    return ecl2, tm4_last, tm5_first


def build_tm_alignment(
    tm_residues: List[Residue],
    ecl2_residues: List[Residue],
    include_hydrogens: bool = False,
) -> Dict[str, object]:
    """
    PCA de los átomos TM. El primer eje PCA se convierte en Z.

    Importante:
    Se fuerza el signo para que el centroide global del ECL2 tenga Z positiva
    respecto al centroide de los TMs. Así, los residuos TM con Z más negativa
    son la base opuesta al lado extracelular/ECL2.
    """
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

    ecl2_center_original = centroid(ecl2_coords)
    ecl2_projection = float(np.dot(ecl2_center_original - tm_center, z_axis))

    # Forzar ECL2 hacia Z positiva
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


def get_deepest_and_closest_ecl2_residues(
    ecl2: List[Residue],
    ref_centroid: np.ndarray,
    transform,
    include_hydrogens: bool = False,
) -> Dict[str, object]:
    """
    Devuelve:
      - deepest_z: residuo con menor ΔZ respecto a la referencia TM-negativa.
      - closest_3d: residuo con menor distancia euclídea 3D respecto a la referencia.

    Para la pregunta de "más dentro" axialmente, usar deepest_z.
    """
    rows = []

    for residue in ecl2:
        rc = residue_centroid(residue, transform, include_hydrogens=include_hydrogens)
        delta = rc - ref_centroid
        dist3d = float(np.linalg.norm(delta))
        rows.append({
            "residue": residue,
            "centroid": rc,
            "delta": delta,
            "delta_z": float(delta[2]),
            "abs_delta_z": float(abs(delta[2])),
            "dist3d": dist3d,
        })

    deepest = min(rows, key=lambda r: r["delta_z"])
    closest = min(rows, key=lambda r: r["dist3d"])

    return {
        "all_rows": rows,
        "deepest_z": deepest,
        "closest_3d": closest,
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
    """
    Escribe el PDB completo con coordenadas alineadas.
    Solo transforma ATOM/HETATM que se hayan parseado.
    """
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
    output_lines.append(f"REMARK Original file: {original_path.name}\n")
    out_path.write_text("".join(output_lines), encoding="utf-8")


def write_marker_pdb(
    out_path: Path,
    tm_centroid: np.ndarray,
    tm_neg_ref_centroid: np.ndarray,
    ecl2_centroid: np.ndarray,
    deepest_centroid: np.ndarray,
    closest3d_centroid: np.ndarray,
    deepest_residue_label: str,
    closest3d_residue_label: str,
) -> None:
    lines = []
    lines.append("REMARK Pseudoatoms for visual inspection after TM alignment\n")
    lines.append("REMARK TMC = TM centroid\n")
    lines.append("REMARK BAS = centroid of TM-neg reference residues\n")
    lines.append("REMARK ECL = global ECL2 centroid\n")
    lines.append(f"REMARK DEP = deepest ECL2 residue in Z: {deepest_residue_label}\n")
    lines.append(f"REMARK C3D = closest ECL2 residue in 3D: {closest3d_residue_label}\n")

    markers = [
        ("TMC", tm_centroid),
        ("BAS", tm_neg_ref_centroid),
        ("ECL", ecl2_centroid),
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
# Metadata
# =============================================================================

def read_metadata(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "entry_name" not in (reader.fieldnames or []):
            return {}
        metadata = {}
        for row in reader:
            entry = row.get("entry_name", "")
            if not entry:
                continue
            metadata[entry] = row
        return metadata


def add_metadata_to_result(result: Dict[str, object], metadata: Dict[str, Dict[str, str]]) -> None:
    entry = str(result.get("entry_name_inferred", ""))
    meta = metadata.get(entry)
    if not meta:
        result["metadata_found"] = "no"
        result.setdefault("sensory_related", "")
        result.setdefault("associated_sense", "")
        return

    result["metadata_found"] = "yes"
    for col in [
        "receptor_name", "accession", "receptor_class", "ligand_type",
        "receptor_family", "subfamily", "species", "sensory_related",
        "associated_sense", "sensory_modality_detail", "sensory_role",
        "classification_rule",
    ]:
        if col in meta:
            result[col] = meta.get(col, "")


# =============================================================================
# Main analysis per PDB
# =============================================================================

def analyze_one_pdb(
    path: Path,
    gpcr_field: str,
    reference_n_residues: int,
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

    # Residue Z after alignment
    tm_residue_z = [
        (r, float(residue_centroid(r, transform, include_hydrogens=include_hydrogens)[2]))
        for r in tms
    ]
    tm_residue_z_sorted = sorted(tm_residue_z, key=lambda x: x[1])

    n_ref = min(reference_n_residues, len(tm_residue_z_sorted))
    tm_neg_ref_residues = [r for r, _ in tm_residue_z_sorted[:n_ref]]
    tm_pos_ref_residues = [r for r, _ in tm_residue_z_sorted[-n_ref:]]

    tm_neg_ref_coords = transform(coords_from_residues(tm_neg_ref_residues, include_hydrogens=include_hydrogens))
    tm_pos_ref_coords = transform(coords_from_residues(tm_pos_ref_residues, include_hydrogens=include_hydrogens))
    tm_neg_ref_centroid = centroid(tm_neg_ref_coords)
    tm_pos_ref_centroid = centroid(tm_pos_ref_coords)

    tm_neg_labels, tm_neg_gpcrs = summarize_reference_residues(tm_neg_ref_residues)
    tm_pos_labels, tm_pos_gpcrs = summarize_reference_residues(tm_pos_ref_residues)

    ecl2_res_metrics = get_deepest_and_closest_ecl2_residues(
        ecl2=ecl2,
        ref_centroid=tm_neg_ref_centroid,
        transform=transform,
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

    # Generate aligned PDB and marker PDB for every file
    aligned_pdb_dir.mkdir(parents=True, exist_ok=True)
    marker_pdb_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = aligned_pdb_dir / path.name
    marker_path = marker_pdb_dir / f"{path.stem}_markers.pdb"

    write_aligned_pdb(path, by_chain, transform, aligned_path)
    write_marker_pdb(
        out_path=marker_path,
        tm_centroid=tm_centroid,
        tm_neg_ref_centroid=tm_neg_ref_centroid,
        ecl2_centroid=ecl2_centroid,
        deepest_centroid=deepest["centroid"],
        closest3d_centroid=closest3d["centroid"],
        deepest_residue_label=deepest_res.resid_label,
        closest3d_residue_label=closest3d_res.resid_label,
    )

    # Sanity flags
    deepest_delta_z = float(deepest["delta_z"])
    closest3d_delta_z = float(closest3d["delta_z"])
    ecl2_centroid_delta_z_to_base = float(ecl2_centroid[2] - tm_neg_ref_centroid[2])

    result = {
        "pdb_file": path.name,
        "receptor_id": receptor_id,
        "entry_name_inferred": f"{receptor_id}_human",
        "state": state,
        "chain": chain_id,
        "gpcr_field_chosen": field_used,
        "alignment_method": "PCA_TM1_TM7_all_atoms_to_Z_ECL2_positive",
        "include_hydrogens_in_calculation": include_hydrogens,
        "reference_n_residues": reference_n_residues,

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

        f"tm_neg{reference_n_residues}_reference_residues": tm_neg_labels,
        f"tm_neg{reference_n_residues}_reference_gpcr_values": tm_neg_gpcrs,
        f"tm_neg{reference_n_residues}_centroid_x_aligned": float(tm_neg_ref_centroid[0]),
        f"tm_neg{reference_n_residues}_centroid_y_aligned": float(tm_neg_ref_centroid[1]),
        f"tm_neg{reference_n_residues}_centroid_z_aligned": float(tm_neg_ref_centroid[2]),

        f"tm_pos{reference_n_residues}_reference_residues": tm_pos_labels,
        f"tm_pos{reference_n_residues}_reference_gpcr_values": tm_pos_gpcrs,
        f"tm_pos{reference_n_residues}_centroid_z_aligned": float(tm_pos_ref_centroid[2]),

        # Main metric: deepest ECL2 residue in axial Z
        f"deepest_ecl2_residue_delta_z_to_tm_neg{reference_n_residues}_aligned": deepest_delta_z,
        f"deepest_ecl2_residue_abs_delta_z_to_tm_neg{reference_n_residues}_aligned": abs(deepest_delta_z),
        f"deepest_ecl2_residue_delta_z_norm_by_tm_span_to_tm_neg{reference_n_residues}_aligned": deepest_delta_z / tm_z_span if tm_z_span > 0 else np.nan,
        f"deepest_ecl2_residue_3d_distance_to_tm_neg{reference_n_residues}_aligned": float(deepest["dist3d"]),
        f"deepest_ecl2_residue": deepest_res.resid_label,
        f"deepest_ecl2_residue_resname": deepest_res.resname,
        f"deepest_ecl2_residue_oneletter": aa3_to_aa1(deepest_res.resname),
        f"deepest_ecl2_residue_resseq": deepest_res.resseq,
        f"deepest_ecl2_residue_centroid_x_aligned": float(deepest["centroid"][0]),
        f"deepest_ecl2_residue_centroid_y_aligned": float(deepest["centroid"][1]),
        f"deepest_ecl2_residue_centroid_z_aligned": float(deepest["centroid"][2]),

        # Control metric: ECL2 residue closest in 3D
        f"closest3d_ecl2_residue_delta_z_to_tm_neg{reference_n_residues}_aligned": closest3d_delta_z,
        f"closest3d_ecl2_residue_abs_delta_z_to_tm_neg{reference_n_residues}_aligned": abs(closest3d_delta_z),
        f"closest3d_ecl2_residue_3d_distance_to_tm_neg{reference_n_residues}_aligned": float(closest3d["dist3d"]),
        f"closest3d_ecl2_residue": closest3d_res.resid_label,
        f"closest3d_ecl2_residue_resname": closest3d_res.resname,
        f"closest3d_ecl2_residue_oneletter": aa3_to_aa1(closest3d_res.resname),
        f"closest3d_ecl2_residue_resseq": closest3d_res.resseq,
        f"closest3d_ecl2_residue_centroid_z_aligned": float(closest3d["centroid"][2]),

        # Global ECL2 centroid metrics, only as diagnostics
        f"ecl2_centroid_delta_z_to_tm_neg{reference_n_residues}_aligned": ecl2_centroid_delta_z_to_base,
        f"ecl2_centroid_3d_distance_to_tm_neg{reference_n_residues}_aligned": float(np.linalg.norm(ecl2_centroid - tm_neg_ref_centroid)),

        # Sanity checks
        "sanity_ecl2_centroid_should_be_positive_z": "ok" if float(ecl2_centroid[2]) > 0 else "warning_ecl2_centroid_not_positive",
        f"sanity_deepest_ecl2_delta_z_to_tm_neg{reference_n_residues}_should_be_positive": "ok" if deepest_delta_z >= -1e-6 else "warning_deepest_ecl2_below_tm_negative_reference",

        # Alignment diagnostics
        "pca_first_axis_explained_variance": alignment["pca_first_axis_explained_variance"],
        "pca_z_axis_original_x": float(alignment["z_axis_original"][0]),
        "pca_z_axis_original_y": float(alignment["z_axis_original"][1]),
        "pca_z_axis_original_z": float(alignment["z_axis_original"][2]),

        # Topology diagnostics
        "tm4_last_residue": tm4_last.resid_label if tm4_last else "",
        "tm4_last_gpcr_value": f"{tm4_last.gpcr_value:.2f}" if tm4_last and tm4_last.gpcr_value is not None else "",
        "ecl2_start_residue": ecl2[0].resid_label if ecl2 else "",
        "ecl2_end_residue": ecl2[-1].resid_label if ecl2 else "",
        "tm5_first_residue": tm5_first.resid_label if tm5_first else "",
        "tm5_first_gpcr_value": f"{tm5_first.gpcr_value:.2f}" if tm5_first and tm5_first.gpcr_value is not None else "",
        "ecl2_sequence": "".join(aa3_to_aa1(r.resname) for r in ecl2),
        "tm_numbers_detected": ";".join(str(i) for i in sorted(tm_counts)),
    }

    field_row = {
        "pdb_file": path.name,
        "receptor_id": receptor_id,
        "state": state,
        "chain": chain_id,
        "gpcr_field_chosen": field_used,
        "n_tm_residues": len(tms),
        "n_ecl2_residues": len(ecl2),
        "tm_numbers_detected": ";".join(str(i) for i in sorted(tm_counts)),
        "tm4_last_residue": result["tm4_last_residue"],
        "tm5_first_residue": result["tm5_first_residue"],
        "sanity_ecl2_centroid_should_be_positive_z": result["sanity_ecl2_centroid_should_be_positive_z"],
        f"sanity_deepest_ecl2_delta_z_to_tm_neg{reference_n_residues}_should_be_positive": result[f"sanity_deepest_ecl2_delta_z_to_tm_neg{reference_n_residues}_should_be_positive"],
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
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_summary(results: List[Dict[str, object]], reference_n_residues: int) -> List[Dict[str, object]]:
    metric = f"deepest_ecl2_residue_delta_z_to_tm_neg{reference_n_residues}_aligned"
    groups = {}

    for row in results:
        state = row.get("state", "unknown")
        sense = row.get("associated_sense", "") or "unannotated"
        key = (state, sense)
        groups.setdefault(key, []).append(float(row[metric]))

    summary = []
    for (state, sense), values in sorted(groups.items()):
        arr = np.array(values, dtype=float)
        summary.append({
            "state": state,
            "associated_sense": sense,
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


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align GPCR PDBs by TM PCA and measure ECL2 axial Z depth. Writes aligned PDBs for every input."
    )
    parser.add_argument("--pdb-dir", required=True, help="Folder containing PDB files.")
    parser.add_argument("--metadata", default=None, help="Optional CSV with entry_name and sensory annotation.")
    parser.add_argument("--out-prefix", default="ecl2_depth_v6", help="Prefix for output files.")
    parser.add_argument("--pattern", default="*.pdb", help="Glob pattern for PDB files. Default: *.pdb")
    parser.add_argument(
        "--gpcr-field",
        choices=["auto", "occupancy", "bfactor", "both"],
        default="auto",
        help="Where to read GPCRdb/Ballesteros numbers from.",
    )
    parser.add_argument(
        "--reference-n-residues",
        type=int,
        default=3,
        help="Number of most-negative-Z TM residues used as base reference.",
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
            reference_n_residues=args.reference_n_residues,
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
            add_metadata_to_result(result, metadata)
            results.append(result)
            field_rows.append(field_row or {})

        if i % 100 == 0:
            print(f"Processed {i}/{len(pdb_files)} PDBs...")

    distances_csv = Path(f"{args.out_prefix}_distances.csv")
    summary_csv = Path(f"{args.out_prefix}_summary.csv")
    field_csv = Path(f"{args.out_prefix}_field_detection.csv")
    failed_csv = Path(f"{args.out_prefix}_failed_files.csv")

    write_csv(distances_csv, results)
    write_csv(summary_csv, make_summary(results, args.reference_n_residues))
    write_csv(field_csv, field_rows)
    write_csv(failed_csv, failed_rows)

    print("\nDone.")
    print(f"PDBs found: {len(pdb_files)}")
    print(f"PDBs analyzed: {len(results)}")
    print(f"PDBs failed: {len(failed_rows)}")
    print("Primary metric:")
    print(f"  deepest_ecl2_residue_delta_z_to_tm_neg{args.reference_n_residues}_aligned")
    print("Outputs:")
    print(f"  {distances_csv}")
    print(f"  {summary_csv}")
    print(f"  {field_csv}")
    print(f"  {failed_csv}")
    print(f"  {aligned_pdb_dir}/")
    print(f"  {marker_pdb_dir}/")


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/analyze_ecl2_z_depth_v6_aligned_debug.py")
path.write_text(script, encoding="utf-8")
path.chmod(0o755)

print(f"Script escrito: {path}")
print(f"Tamaño: {path.stat().st_size / 1024:.1f} KB")
