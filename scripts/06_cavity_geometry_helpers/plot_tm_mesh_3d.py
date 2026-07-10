#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualise TM-based polygon mesh (triangular surface) for a GPCR PDB,
with TM-specific colouring for Cα atoms and explicit low/high vertices.

This script:
- detects TM residues from the occupancy field (1.xx .. 8.xx),
- aligns the TM bundle so its main axis coincides with the Z axis,
- builds a polygonal "envelope" from the lowest and highest Cα of each TM,
- optionally builds a second envelope from selected BW positions,
- plots the triangular mesh in 3D,
- optionally colours Cα atoms by TM and highlights the low/high vertices.

Usage example
-------------
python plot_tm_mesh_3d_colored.py \
    --in or8k3_all_active.pdb \
    --tm-include 1-7 \
    --bw 1.53,1.54 \
    --show-ca
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================
# Basic PDB utilities
# ============================

def is_atom_line(line: str) -> bool:
    """Return True if the line corresponds to an ATOM or HETATM record."""
    rec = line[:6]
    return rec.startswith("ATOM") or rec.startswith("HETATM")


def parse_xyz(line: str) -> np.ndarray:
    """Extract (x, y, z) coordinates from a PDB line as a NumPy array."""
    return np.array(
        [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ],
        dtype=float,
    )


def parse_keys(line: str):
    """
    Extract residue key (chain, resSeq, iCode) from a PDB line.

    This is used to group atoms by residue.
    """
    chain = line[21].strip() or " "
    resSeq = int(line[22:26])
    iCode = line[26].strip()
    return (chain, resSeq, iCode)


def parse_occ(line: str):
    """
    Parse occupancy as float.

    In your annotation, this encodes TM or TM.BW (e.g. 1.53). If parsing
    fails, returns None.
    """
    s = line[54:60].strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_atom_name(line: str) -> str:
    """Return the atom name (e.g. 'CA')."""
    return line[12:16].strip()


def group_by_residue(lines):
    """
    Group all ATOM/HETATM records by residue key and collect coordinates.

    Returns
    -------
    resid_to_idx : dict
        (chain, resSeq, iCode) -> list of line indices.
    atom_xyz : (N_atoms, 3) array
        Coordinates of all atoms in the order they appear.
    atom_map : (N_atoms,) array
        Maps rows of atom_xyz back to line indices in 'lines'.
    """
    resid_to_idx = defaultdict(list)
    atom_xyz = []
    atom_map = []

    for i, ln in enumerate(lines):
        if is_atom_line(ln):
            resid_to_idx[parse_keys(ln)].append(i)
            atom_xyz.append(parse_xyz(ln))
            atom_map.append(i)

    if atom_xyz:
        atom_xyz = np.vstack(atom_xyz)
        atom_map = np.array(atom_map, dtype=int)
    else:
        atom_xyz = np.zeros((0, 3), float)
        atom_map = np.zeros((0,), int)

    return resid_to_idx, atom_xyz, atom_map


def residue_ca(lines, idxs):
    """
    Return Cα coordinates for a residue.

    If a CA atom is not present, the geometric centroid of all atoms in the
    residue is used as fallback.
    """
    ca = None
    coords = []
    for i in idxs:
        if not is_atom_line(lines[i]):
            continue
        p = parse_xyz(lines[i])
        coords.append(p)
        if parse_atom_name(lines[i]) == "CA":
            ca = p

    if ca is not None:
        return ca

    if coords:
        return np.mean(np.vstack(coords), axis=0)

    # Extremely unusual: empty residue
    return np.zeros(3, float)


# ============================
# TM selection from occupancy
# ============================

def expand_tm_include(spec: str):
    """
    Parse a TM selection string like "1-7,8" into a set of integers.

    Examples
    --------
    "1-7"      -> {1,2,3,4,5,6,7}
    "1-3,6,7"  -> {1,2,3,6,7}
    """
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            for t in range(min(a, b), max(a, b) + 1):
                out.add(t)
        else:
            out.add(int(part))
    return out


def collect_tm_residues(lines, resid_to_idx, tm_include="1-7"):
    """
    Identify TM residues based on the occupancy field.

    Occupancy is assumed to encode TM indices as 1.xx, 2.xx, ..., 8.xx.
    The integer part (int(occ)) is taken as TM id.

    Parameters
    ----------
    tm_include : str
        String like "1-7" that selects which TM ids are considered.

    Returns
    -------
    tm_keys : list
        Residue keys for residues that belong to the selected TMs.
    """
    tm_set = expand_tm_include(tm_include)
    tm_keys = []

    for key, idxs in resid_to_idx.items():
        tm_found = None
        for i in idxs:
            v = parse_occ(lines[i])
            if v is None:
                continue
            tm = int(v)
            if 1 <= tm <= 8:
                tm_found = tm
                break
        if tm_found is not None and tm_found in tm_set:
            tm_keys.append(key)

    if not tm_keys:
        raise RuntimeError(
            "No TM residues found in occupancy field (expected 1.xx..8.xx). "
            "Check your PDB annotation and --tm-include."
        )

    return tm_keys


# ============================
# Alignment utilities
# ============================

def com_from_residues(lines, resid_to_idx, keys):
    """
    Compute centre of mass (simple average) of all atoms in the given residues.
    """
    num = np.zeros(3, float)
    den = 0.0
    for k in keys:
        for i in resid_to_idx[k]:
            if not is_atom_line(lines[i]):
                continue
            num += parse_xyz(lines[i])
            den += 1.0
    return num / max(den, 1e-12)


def fit_tls_axis(points: np.ndarray) -> np.ndarray:
    """
    Fit dominant axis of a point cloud using SVD.

    Returns a unit vector along the principal axis.
    """
    P = np.asarray(points, dtype=float)
    P0 = P.mean(axis=0, keepdims=True)
    X = P - P0
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    axis = VT[0, :]
    return axis / np.linalg.norm(axis)


def R_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Construct a rotation matrix that rotates vector 'a' onto vector 'b'.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        # Vectors are (almost) parallel: identity rotation
        return np.eye(3)

    if np.isclose(c, -1.0):
        # Vectors are opposite: choose an arbitrary axis orthogonal to 'a'
        orth = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(orth, a)) > 0.9:
            orth = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, orth)
        v /= np.linalg.norm(v)
        # 180-degree rotation around v
        return -np.eye(3) + 2.0 * np.outer(v, v)

    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))


# ============================
# Mesh building (triangles)
# ============================

def build_mesh_from_tms(walls, tm_ids_sorted):
    """
    Build a triangular mesh from TM low/high points.

    walls[tm_id] must contain {"low": (3,), "high": (3,)} in aligned coordinates.

    The mesh consists of:
    - a bottom cap built as a triangle fan from the 'low' points,
    - a top cap built as a triangle fan from the 'high' points,
    - lateral walls between each pair of neighbouring TMs, each quadrilateral
      split into two triangles.
    """
    tris = []

    lows = [walls[tid]["low"] for tid in tm_ids_sorted]
    highs = [walls[tid]["high"] for tid in tm_ids_sorted]
    n = len(tm_ids_sorted)

    # --- Bottom and top caps as triangle fans ---
    for i in range(1, n - 1):
        # Bottom triangle: low[0], low[i], low[i+1]
        tris.append(np.vstack([lows[0], lows[i], lows[i + 1]]))
        # Top triangle: high[0], high[i+1], high[i]
        tris.append(np.vstack([highs[0], highs[i + 1], highs[i]]))

    # --- Lateral walls between neighbouring TMs ---
    for i in range(n):
        j = (i + 1) % n  # wrap-around: last TM connects to first
        li, lj = lows[i], lows[j]
        hi, hj = highs[i], highs[j]
        # Quadrilateral [li - lj - hj - hi] -> two triangles:
        tris.append(np.vstack([li, lj, hj]))
        tris.append(np.vstack([li, hj, hi]))

    return tris


# ============================
# BW vertices (optional)
# ============================

def parse_bw_list(bw_str: str):
    """Parse '1.53,1.54,2.43' -> [1.53, 1.54, 2.43]."""
    vals = []
    for tok in bw_str.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except Exception:
            pass

    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def match_bw_value(occ: float, target: float, tol=0.015) -> bool:
    """
    Return True if occupancy 'occ' matches a BW-like value 'target' within a
    small tolerance. Integer parts must match (TM index), and the fractional
    part must be close to the desired BW position.
    """
    if occ is None:
        return False
    if int(occ) != int(target):
        return False
    return (abs(occ - target) <= tol) or (round(occ, 2) == round(target, 2))


def collect_bw_vertices(lines, resid_to_idx, bw_values, R, com_tm):
    """
    Build low/high points per TM using only residues whose occupancy matches
    one of the supplied BW values.

    Returns
    -------
    walls_bw : dict or None
        Mapping tm_id -> {"low": (3,), "high": (3,)} in aligned coordinates,
        or None if not enough vertices.
    tm_ids_sorted_bw : list of int
        TM ids used for BW polyhedron, sorted by angle in XY.
    """
    if not bw_values:
        return None, []

    by_tm = defaultdict(list)

    for key, idxs in resid_to_idx.items():
        occ = None
        for i in idxs:
            v = parse_occ(lines[i])
            if v is not None:
                occ = v
                break
        if occ is None:
            continue

        for b in bw_values:
            if match_bw_value(occ, b):
                ca = residue_ca(lines, idxs)
                p = R @ (ca - com_tm)
                tm_id = int(b)  # TM index from integer part
                by_tm[tm_id].append(p)
                break

    if len(by_tm) < 3:
        return None, []

    centers_xy = {}
    walls_bw = {}

    for tm_id, pts in by_tm.items():
        P = np.vstack(pts)
        z = P[:, 2]
        low = P[np.argmin(z)]
        high = P[np.argmax(z)]
        walls_bw[tm_id] = {"low": low, "high": high}
        c_xy = 0.5 * (low[:2] + high[:2])
        centers_xy[tm_id] = math.atan2(c_xy[1], c_xy[0])

    tm_ids_sorted_bw = sorted(walls_bw.keys(), key=lambda t: centers_xy[t])
    return walls_bw, tm_ids_sorted_bw


# ============================
# Plot helpers
# ============================

def set_equal_aspect_3d(ax):
    """
    Set equal aspect ratio (same scale) for x, y, z axes in a 3D plot.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)
    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    ax.set_xlim3d(x_mid - max_range / 2.0, x_mid + max_range / 2.0)
    ax.set_ylim3d(y_mid - max_range / 2.0, y_mid + max_range / 2.0)
    ax.set_zlim3d(z_mid - max_range / 2.0, z_mid + max_range / 2.0)


def plot_mesh(tris, ax, facecolor, edgecolor="k", alpha=0.3,
              linewidth=0.3, label=None):
    """
    Add a triangular mesh to a 3D axis as a Poly3DCollection.
    """
    if not tris:
        return

    poly = Poly3DCollection(
        tris,
        facecolors=facecolor,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=linewidth,
    )
    if label is not None:
        poly.set_label(label)
    ax.add_collection3d(poly)


# ============================
# Main routine
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Plot 3D TM-based polygon mesh (total and optional BW polyhedron), "
                    "colouring Cα by TM."
    )
    parser.add_argument("--in", dest="inpdb", required=True,
                        help="Input PDB file with TM/BW annotation in occupancy.")
    parser.add_argument("--tm-include", default="1-7",
                        help="TM indices to include, e.g. '1-7' or '1-4,6,7'.")
    parser.add_argument("--bw", type=str, default=None,
                        help="Optional BW list for a second polyhedron, e.g. '1.53,1.54'.")
    parser.add_argument("--show-ca", action="store_true",
                        help="Scatter-plot TM Cα atoms coloured by TM and highlight low/high.")
    args = parser.parse_args()

    # --- Read PDB ---
    with open(args.inpdb, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # --- Group atoms by residue ---
    resid_to_idx, atom_xyz, atom_map = group_by_residue(lines)

    # --- Select TM residues from occupancy ---
    tm_keys = collect_tm_residues(lines, resid_to_idx, args.tm_include)

    # --- CA coordinates for alignment ---
    ca_pts = np.vstack([residue_ca(lines, resid_to_idx[k]) for k in tm_keys])

    # --- Define TM bundle reference frame ---
    com_tm = com_from_residues(lines, resid_to_idx, tm_keys)
    axis = fit_tls_axis(ca_pts - com_tm)
    R = R_from_a_to_b(axis, np.array([0.0, 0.0, 1.0]))

    # --- Transform TM Cα into aligned frame and group by TM id ---
    tms = defaultdict(list)  # TM_id -> list of Cα coords in aligned frame
    for k in tm_keys:
        idxs = resid_to_idx[k]
        ca = residue_ca(lines, idxs)
        ca_r = R @ (ca - com_tm)

        # TM id from integer part of occupancy
        occ = None
        for i in idxs:
            v = parse_occ(lines[i])
            if v is not None:
                occ = v
                break
        if occ is None:
            continue
        tm_id = int(occ)
        tms[tm_id].append(ca_r)

    # Keep only TMs with at least two Cα points
    tm_ids = sorted([tid for tid in tms if len(tms[tid]) >= 2])
    if len(tm_ids) < 3:
        raise RuntimeError("Not enough TMs (≥3) to build a closed polygon.")

    # --- Order TMs by azimuthal angle (around the bundle) ---
    angles = {}
    for tid in tm_ids:
        A = np.vstack(tms[tid])
        c = A[:, :2].mean(axis=0)  # XY centre
        angles[tid] = math.atan2(c[1], c[0])

    tm_ids_sorted = sorted(tm_ids, key=lambda t: angles[t])

    # --- Compute low/high vertices for TOTAL solid ---
    walls_total = {}
    # We also keep per-TM arrays for plotting
    per_tm_ca = {}

    for tid in tm_ids_sorted:
        A = np.vstack(tms[tid])  # (n_points, 3)
        per_tm_ca[tid] = A
        z = A[:, 2]
        low = A[np.argmin(z)]
        high = A[np.argmax(z)]
        walls_total[tid] = {"low": low, "high": high}

    # --- Build triangular mesh for TOTAL solid ---
    tris_total = build_mesh_from_tms(walls_total, tm_ids_sorted)

    # --- Optional BW polyhedron ---
    tris_bw = []
    walls_bw = None
    tm_ids_bw = []
    if args.bw:
        bw_vals = parse_bw_list(args.bw)
        walls_bw, tm_ids_bw = collect_bw_vertices(
            lines, resid_to_idx, bw_vals, R, com_tm
        )
        if tm_ids_bw:
            tris_bw = build_mesh_from_tms(walls_bw, tm_ids_bw)
        else:
            print("[INFO] BW polyhedron could not be built: no matching BW residues found.")

    # ============================
    # 3D plotting
    # ============================

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot TOTAL TM envelope
    plot_mesh(tris_total, ax,
              facecolor="tab:blue",
              edgecolor="k",
              alpha=0.25,
              linewidth=0.3,
              label="TM envelope")

    # Plot BW envelope if available
    if tris_bw:
        plot_mesh(tris_bw, ax,
                  facecolor="tab:orange",
                  edgecolor="k",
                  alpha=0.35,
                  linewidth=0.4,
                  label="BW polyhedron")

    # --- Scatter TM CA points coloured by TM ---
    if args.show_ca:
        cmap = plt.get_cmap("tab10")
        for idx, tid in enumerate(tm_ids_sorted):
            A = per_tm_ca[tid]
            color = cmap(idx % cmap.N)
            ax.scatter(
                A[:, 0], A[:, 1], A[:, 2],
                s=15,
                c=[color],
                alpha=0.8,
                label=f"TM{tid} Cα" if idx == 0 else None  # avoid too many legend entries
            )

        # Highlight TOTAL low/high vertices in black
        lows_plot = np.vstack([walls_total[tid]["low"] for tid in tm_ids_sorted])
        highs_plot = np.vstack([walls_total[tid]["high"] for tid in tm_ids_sorted])

        ax.scatter(
            lows_plot[:, 0], lows_plot[:, 1], lows_plot[:, 2],
            s=40,
            c="black",
            marker="s",
            alpha=0.9,
            label="low vertices"
        )
        ax.scatter(
            highs_plot[:, 0], highs_plot[:, 1], highs_plot[:, 2],
            s=60,
            c="black",
            marker="^",
            alpha=0.9,
            label="high vertices"
        )

        # If BW vertices exist, show them in red
        if walls_bw and tm_ids_bw:
            bw_pts = []
            for tid in tm_ids_bw:
                bw_pts.append(walls_bw[tid]["low"])
                bw_pts.append(walls_bw[tid]["high"])
            bw_pts = np.vstack(bw_pts)
            ax.scatter(
                bw_pts[:, 0], bw_pts[:, 1], bw_pts[:, 2],
                s=70,
                c="red",
                marker="o",
                alpha=0.9,
                label="BW low/high"
            )

    # Axis labels and view
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    ax.view_init(elev=20, azim=-60)
    set_equal_aspect_3d(ax)

    # Build legend from plotted artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

