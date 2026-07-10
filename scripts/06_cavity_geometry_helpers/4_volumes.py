#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometric cavity volume estimation for GPCRs using TM Cα-based envelopes
and a ray-casting ("spike") filter.

Overview
--------
This script estimates the free (void) volume enclosed by the transmembrane (TM)
region of a GPCR based purely on geometric criteria:

1) TM identification from PDB occupancy:
   - The PDB occupancy field is assumed to encode TM/BW indices as 1.xx, 2.xx,
     ..., 8.xx, where the integer part is the TM number and the fractional part
     may encode a Ballesteros–Weinstein (BW) position.

2) Alignment of the TM bundle:
   - TM residues (TM1–TM7 by default) are used to define a bundle-based frame:
       * centre of mass of TM atoms is translated to the origin,
       * main axis of the TM bundle is obtained via SVD and aligned with +Z.

3) Construction of a TM envelope (polyhedron):
   - For each TM, the lowest and highest Cα along Z (in the aligned frame)
     are identified ("low" and "high" points).
   - These points are connected into a closed polyhedron (non-planar top and
     bottom caps plus lateral faces between neighbouring TMs).

4) Voxelisation of the interior:
   - The polyhedron is sliced along Z at a user-defined spacing.
   - In each slice, the intersection with the polyhedron is converted into
     2D polygons in the XY plane.
   - A regular XY grid is generated inside the union of these polygons,
     and the resulting (x, y, z) points are candidate voxel centres.
   - Candidate voxels overlapping any van der Waals sphere (vdW + probe)
     are discarded.

5) Ray-casting ("spike") filter:
   - For each remaining voxel, N rays (spike_dirs) are cast in quasi-uniform
     directions on the unit sphere up to a maximum length (spike_max).
   - A voxel is kept only if every ray hits at least one protein vdW sphere
     within spike_max (using vdW + spike_probe as radius).
   - This enforces a "fully buried" criterion: voxels must be enclosed by
     protein in all sampled directions.

6) Optional BW-defined polyhedron:
   - An optional list of BW positions (e.g. "1.53,1.57,2.43,...") is used to
     define a second polyhedron (BW polygon) constructed from those anchors
     only, again using low/high points in Z per TM.
   - Voxelisation and spike filtering are repeated inside that BW polyhedron,
     and both the total and BW-restricted volumes can be written as PDBs.

The main entry point is `run(...)`, which is wired to a command-line interface
via `parse_args()` and the `__main__` block.
"""

# -------------------------------------------------------------------------
# Package bootstrap: make sure NumPy is available (even in restricted envs)
# -------------------------------------------------------------------------
import os
import sys
import subprocess
import importlib
import site   # noqa: F401  (imported for completeness; not used directly)

def ensure_package(pkg, version=None, target=None):
    """
    Ensure a Python package is importable, installing it into a user-writable
    directory if necessary.

    Parameters
    ----------
    pkg : str
        Name of the package to import/install (e.g. "numpy").
    version : str or None, optional
        If provided, install this exact version (e.g. "1.26.4").
        If None, install the latest available version.
    target : str or None, optional
        Target directory where the package should be installed (e.g. a path
        under $HOME). If None, a default 'pylibs' directory under TMPDIR
        or the user's home directory will be used.

    Notes
    -----
    - This is mainly useful in notebook / sandboxed environments where the
      user does not have admin privileges.
    - The function:
        * tries to import the module,
        * if import fails, calls `pip install --target <target>`,
        * then adds <target> to sys.path and re-imports the module.
    """
    modname = pkg  # for numpy, package name == module name
    try:
        importlib.import_module(modname)
        return  # package already available
    except ImportError:
        pass

    # Choose a user-writable installation base
    base = os.environ.get("TMPDIR") or os.path.expanduser("~")
    target = target or os.path.join(base, "pylibs")
    os.makedirs(target, exist_ok=True)

    # Build pip spec and command
    spec = f"{pkg}=={version}" if version else pkg
    cmd = [
        sys.executable,
        "-m", "pip", "install",
        "--upgrade",
        "--no-warn-script-location",
        "--target", target,
        spec,
    ]

    # Remove proxy settings that may cause 503s in some environments
    env = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)

    print(f"[setup] Installing {spec} into {target} ...", flush=True)
    subprocess.check_call(cmd, env=env)

    # Make sure Python can see that directory
    if target not in sys.path:
        sys.path.insert(0, target)
    importlib.invalidate_caches()
    importlib.import_module(modname)


# Guarantee NumPy is available (version compatible with Python 3.9)
ensure_package("numpy", "1.26.4")

import numpy as np
import argparse
import math
from collections import defaultdict


# =========================================================================
# PDB utilities
# =========================================================================

def is_atom_line(line):
    """
    Return True if a PDB line corresponds to an ATOM or HETATM record.

    Parameters
    ----------
    line : str
        PDB file line.

    Returns
    -------
    bool
        True if the record is ATOM/HETATM, False otherwise.
    """
    rec = line[:6]
    return rec.startswith("ATOM") or rec.startswith("HETATM")


def parse_xyz(line):
    """
    Parse Cartesian coordinates (x, y, z) from a PDB line.

    Parameters
    ----------
    line : str
        PDB file line in standard format.

    Returns
    -------
    np.ndarray, shape (3,)
        (x, y, z) coordinates as float32/float64.
    """
    return np.array(
        [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ],
        float,
    )


def rewrite_xyz(line, x, y, z):
    """
    Rewrite the XYZ coordinates of a PDB ATOM/HETATM line.

    Parameters
    ----------
    line : str
        Original PDB line.
    x, y, z : float
        New coordinates to be written.

    Returns
    -------
    str
        Modified PDB line with updated coordinates.
    """
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def parse_keys(line):
    """
    Extract a residue key (chain, resSeq, iCode) from a PDB line.

    Parameters
    ----------
    line : str

    Returns
    -------
    tuple
        (chain, resSeq, iCode), where:
        - chain : str (single character or space),
        - resSeq : int,
        - iCode : str (insertion code, may be an empty string).
    """
    chain = line[21].strip() or " "
    resSeq = int(line[22:26])
    iCode = line[26].strip()
    return (chain, resSeq, iCode)


def parse_occ(line):
    """
    Parse the occupancy field from a PDB line as float.

    Parameters
    ----------
    line : str

    Returns
    -------
    float or None
        Occupancy as float, or None if parsing fails.

    Notes
    -----
    In this workflow, the occupancy is repurposed to store TM/BW information,
    e.g. 1.53 for TM1 position 53 in the Ballesteros–Weinstein system.
    """
    s = line[54:60].strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_atom_name(line):
    """
    Extract the atom name from a PDB line (e.g. "CA", "N", "C1", ...).

    Parameters
    ----------
    line : str

    Returns
    -------
    str
        The stripped atom name.
    """
    return line[12:16].strip()


def parse_element(line):
    """
    Determine the chemical element of an atom from a PDB line.

    Parameters
    ----------
    line : str

    Returns
    -------
    str
        Element symbol (e.g. "C", "O", "N", "S", ...).

    Notes
    -----
    - If columns 77–78 (element field) are non-empty, they are used.
    - Otherwise, the element is inferred from the alphabetic part of the
      atom name (first letter, uppercase), defaulting to 'C'.
    """
    el = line[76:78].strip()
    if el:
        return el
    nm = parse_atom_name(line)
    return (''.join([c for c in nm if c.isalpha()]) or 'C')[0].upper()


def group_by_residue(lines):
    """
    Group PDB ATOM/HETATM records by residue and collect all atom coordinates.

    Parameters
    ----------
    lines : list of str
        Lines of a PDB file.

    Returns
    -------
    resid_to_idx : dict
        Maps residue keys (chain, resSeq, iCode) to lists of line indices.
    atom_xyz : np.ndarray, shape (N_atoms, 3)
        Cartesian coordinates of all atoms in the order they appear.
    atom_map : np.ndarray, shape (N_atoms,)
        Maps row indices in atom_xyz back to line indices in 'lines'.
    """
    resid_to_idx = defaultdict(list)
    atom_xyz = []
    atom_map = []

    for i, ln in enumerate(lines):
        if is_atom_line(ln):
            resid_to_idx[parse_keys(ln)].append(i)
            atom_xyz.append(parse_xyz(ln))
            atom_map.append(i)

    return resid_to_idx, np.vstack(atom_xyz), np.array(atom_map, int)


# =========================================================================
# TM selection from occupancy
# =========================================================================

def expand_tm_include(spec):
    """
    Expand a TM selection specification into a set of TM indices.

    Parameters
    ----------
    spec : str
        A string like "1-7" or "1-3,6,7", where ranges are allowed.

    Returns
    -------
    set of int
        Set of TM indices included in the specification.
    """
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
        else:
            out.add(int(part))
    return out


def collect_tm_residues(lines, resid_to_idx, tm_include="1-7"):
    """
    Collect residues belonging to selected TMs based on the occupancy field.

    Parameters
    ----------
    lines : list of str
        PDB lines.
    resid_to_idx : dict
        Residue mapping as returned by group_by_residue().
    tm_include : str, optional
        TM indices to include, e.g. "1-7". Occupancy is assumed to store
        values like 1.xx, 2.xx, ..., 8.xx; the integer part 1..8 is taken
        as the TM index.

    Returns
    -------
    tm_keys : list
        List of residue keys belonging to TMs included in tm_include.

    Raises
    ------
    RuntimeError
        If no TM residues are detected.
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
        raise RuntimeError("No TM residues detected in occupancy (expected 1.xx..8.xx).")
    return tm_keys


# =========================================================================
# Van der Waals radii and basic geometry
# =========================================================================

# Simple VdW radii table for common elements (Å)
VDW = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}

def vdw_radius(el):
    """
    Get a van der Waals radius (Å) for a given element.

    Parameters
    ----------
    el : str
        Element symbol (case-insensitive).

    Returns
    -------
    float
        VdW radius in Å. Defaults to 1.70 (carbon-like) if unknown.
    """
    return VDW.get(el.upper(), 1.70)


def residue_ca(lines, idxs):
    """
    Compute Cα coordinates for a residue, or fallback to centroid.

    Parameters
    ----------
    lines : list of str
        PDB lines.
    idxs : list of int
        Indices of PDB lines belonging to this residue.

    Returns
    -------
    np.ndarray, shape (3,)
        Cα coordinates if present; otherwise, the centroid of all atoms.
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
    return ca if ca is not None else np.mean(np.vstack(coords), axis=0)


def com_from_residues(lines, resid_to_idx, keys):
    """
    Compute a simple centre of mass (unweighted average of atom positions)
    over a set of residues.

    Parameters
    ----------
    lines : list of str
    resid_to_idx : dict
        Residue mapping from group_by_residue().
    keys : list
        Residue keys for which to include atoms.

    Returns
    -------
    np.ndarray, shape (3,)
        Average Cartesian coordinates (centre of mass).
    """
    num = np.zeros(3)
    den = 0.0
    for k in keys:
        for i in resid_to_idx[k]:
            if not is_atom_line(lines[i]):
                continue
            num += parse_xyz(lines[i])
            den += 1.0
    return num / max(den, 1e-12)


def fit_tls_axis(points):
    """
    Fit the principal axis of a point cloud via singular value decomposition.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Coordinates of TM Cα atoms (or other bundle-defining atoms).

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector along the dominant axis of the point cloud.
    """
    P = np.asarray(points)
    P0 = P.mean(axis=0, keepdims=True)
    X = P - P0
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    axis = VT[0, :]
    return axis / np.linalg.norm(axis)


def R_from_a_to_b(a, b):
    """
    Construct a 3x3 rotation matrix that rotates vector 'a' onto vector 'b'.

    Parameters
    ----------
    a, b : array-like, shape (3,)
        Input vectors.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix R such that R @ a ~ b.

    Notes
    -----
    - Uses the standard axis-angle formula from cross and dot products.
    - Handles the parallel (a ~ b) and antiparallel (a ~ -b) cases explicitly.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1.0):
        # Almost parallel: identity
        return np.eye(3)
    if np.isclose(c, -1.0):
        # Antiparallel: choose arbitrary orthogonal axis
        orth = np.array([1, 0, 0], float)
        if abs(np.dot(orth, a)) > 0.9:
            orth = np.array([0, 1, 0], float)
        v = np.cross(a, orth)
        v /= np.linalg.norm(v)
        # 180-degree rotation around v
        return -np.eye(3) + 2 * np.outer(v, v)
    s = np.linalg.norm(v)
    K = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + K + K @ K * ((1 - c) / (s**2))


# =========================================================================
# Mesh construction (triangular polyhedron)
# =========================================================================

def build_mesh_from_tms(walls, tm_ids_sorted):
    """
    Build a triangular mesh (polyhedron) from TM low/high points.

    Parameters
    ----------
    walls : dict
        Mapping TM id -> {"low": (3,), "high": (3,)} arrays in aligned coords.
    tm_ids_sorted : list of int
        TM ids ordered by azimuthal angle around the bundle.

    Returns
    -------
    list of np.ndarray
        List of triangles. Each triangle is a (3, 3) array of vertices.

    Notes
    -----
    - Bottom cap is built as a triangle fan from low[0] to all other lows.
    - Top cap is built as a triangle fan from high[0] to all other highs.
    - Lateral walls between neighbouring TMs i and j are formed by the
      quadrilateral [low_i, low_j, high_j, high_i], split into two triangles.
    """
    tris = []
    lows = [walls[tid]["low"] for tid in tm_ids_sorted]
    highs = [walls[tid]["high"] for tid in tm_ids_sorted]
    n = len(tm_ids_sorted)

    # Bottom and top caps as triangle fans
    for i in range(1, n - 1):
        tris.append(np.vstack([lows[0], lows[i], lows[i + 1]]))
        tris.append(np.vstack([highs[0], highs[i + 1], highs[i]]))

    # Lateral walls between neighbouring TMs
    for i in range(n):
        j = (i + 1) % n
        li, lj = lows[i], lows[j]
        hi, hj = highs[i], highs[j]
        tris.append(np.vstack([li, lj, hj]))
        tris.append(np.vstack([li, hj, hi]))
    return tris


def tri_plane_intersect_segment(tri, z, eps=1e-6):
    """
    Intersect a triangle with a horizontal plane z = constant.

    Parameters
    ----------
    tri : np.ndarray, shape (3, 3)
        Triangle vertices.
    z : float
        Z value of the slicing plane.
    eps : float, optional
        Numerical tolerance for considering a point to lie on the plane.

    Returns
    -------
    (np.ndarray, np.ndarray) or None
        Pair of endpoints (3D points) of the intersection segment, or None
        if there is no intersection or it degenerates.

    Notes
    -----
    - Handles cases where a vertex lies exactly on the plane.
    - The triangle-plane intersection of a non-degenerate triangle and a plane
      is either empty, a single point, or a line segment; here we only
      return the segment case (two distinct points).
    """
    V = tri
    Z = V[:, 2] - z
    pts = []
    # Check each edge of the triangle
    for a, b, Za, Zb in (
        (V[0], V[1], Z[0], Z[1]),
        (V[1], V[2], Z[1], Z[2]),
        (V[2], V[0], Z[2], Z[0]),
    ):
        # If a vertex lies near the plane, keep it as an intersection point
        if abs(Za) < eps and not any(np.allclose(a, q, atol=eps) for q in pts):
            pts.append(a)
        # If the edge crosses the plane, compute the intersection
        if Za * Zb < -eps * eps:
            t = Za / (Za - Zb)
            p = a + t * (b - a)
            pts.append(p)
    # Deduplicate intersection points
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) == 2:
        return np.array(uniq[0]), np.array(uniq[1])
    else:
        return None


def segments_to_polygons(segments, eps=1e-6):
    """
    Stitch a set of line segments into closed polygons in the XY plane.

    Parameters
    ----------
    segments : list of (np.ndarray, np.ndarray)
        List of segment endpoints (3D). Only x,y components are used.
    eps : float, optional
        Tolerance for matching endpoints.

    Returns
    -------
    list of np.ndarray
        List of polygons. Each polygon is an array of shape (n_vertices, 2)
        in XY coordinates.

    Notes
    -----
    - The function attempts to chain segments into loops by matching endpoints.
    - Colinear points along the loop are simplified.
    - Only polygons with at least 3 distinct vertices (plus closure) are kept.
    """
    unused = segments[:]
    polys = []

    while unused:
        p0, p1 = unused.pop()
        loop = [p0, p1]
        changed = True
        # Greedy chaining of segments to extend the loop
        while changed:
            changed = False
            for k, (a, b) in enumerate(unused):
                if np.linalg.norm(loop[-1] - a) < eps:
                    loop.append(b)
                    unused.pop(k)
                    changed = True
                    break
                if np.linalg.norm(loop[-1] - b) < eps:
                    loop.append(a)
                    unused.pop(k)
                    changed = True
                    break
        # Close loop if last != first
        if np.linalg.norm(loop[0] - loop[-1]) >= eps:
            loop.append(loop[0])

        poly_xy = np.array(loop)[:, :2]

        # Remove consecutive duplicates/near-duplicates
        keep = [0]
        for i in range(1, len(poly_xy) - 1):
            if np.linalg.norm(poly_xy[i] - poly_xy[keep[-1]]) > eps:
                keep.append(i)
        keep.append(len(poly_xy) - 1)
        poly_xy = poly_xy[keep]

        # polygon must have at least 3 unique vertices + closure
        if len(poly_xy) >= 4:
            polys.append(poly_xy[:-1])  # drop closing duplicate

    return polys


def point_in_polygon(x, y, poly_xy):
    """
    Test if a point (x, y) lies inside a polygon using ray casting.

    Parameters
    ----------
    x, y : float
        Coordinates of the test point.
    poly_xy : np.ndarray, shape (n, 2)
        Polygon vertices (closed or open, but assumed ordered).

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    inside = False
    n = poly_xy.shape[0]
    xj, yj = poly_xy[-1]
    for i in range(n):
        xi, yi = poly_xy[i]
        cond = ((yi > y) != (yj > y)) and (
            x
            < (xj - xi) * (y - yi) / (yj - yi + 1e-12)
            + xi
        )
        if cond:
            inside = not inside
        xj, yj = xi, yi
    return inside


# =========================================================================
# Spike directions (Fibonacci sphere) and ray-sphere filter
# =========================================================================

def fibonacci_directions(n=20):
    """
    Generate approximately uniform directions on the unit sphere using a
    Fibonacci spiral construction.

    Parameters
    ----------
    n : int, optional
        Number of directions to generate.

    Returns
    -------
    np.ndarray, shape (n, 3)
        Array of unit vectors (x, y, z).
    """
    if n < 1:
        return np.zeros((0, 3), float)
    dirs = np.zeros((n, 3), float)
    phi = (1.0 + 5.0**0.5) / 2.0  # golden ratio
    ga = 2.0 * math.pi * (1.0 - 1.0 / phi)
    for k in range(n):
        # z coordinate from +1 to -1
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = max(0.0, 1.0 - z * z) ** 0.5
        t = k * ga
        x = r * math.cos(t)
        y = r * math.sin(t)
        dirs[k] = [x, y, z]
    return dirs


def filter_voxels_by_spikes(
    pts_free,
    atom_pos,
    atom_vdw,
    max_len=10.0,
    n_dirs=20,
    spike_probe=0.0,
    chunk_atoms=1000,
    block_pts=2000,
):
    """
    Apply the "spike" ray-casting filter to candidate voxels.

    A voxel is kept only if, for each of the n_dirs directions, a ray of
    length up to max_len intersects at least one protein vdW sphere
    (radius = atom_vdw + spike_probe).

    Parameters
    ----------
    pts_free : np.ndarray, shape (N, 3)
        Candidate voxel centres (assumed to be free of steric clashes).
    atom_pos : np.ndarray, shape (M, 3)
        Atom coordinates (aligned frame).
    atom_vdw : np.ndarray, shape (M,)
        Van der Waals radii for each atom.
    max_len : float, optional
        Maximum ray length (Å).
    n_dirs : int, optional
        Number of spike directions (rays) per voxel.
    spike_probe : float, optional
        Extra probe added to vdW radii for the ray-sphere intersection test.
    chunk_atoms : int, optional
        Number of atoms to process per chunk (memory/performance tuning).
    block_pts : int, optional
        Number of voxels to process per block (memory/performance tuning).

    Returns
    -------
    np.ndarray, dtype=bool, shape (N,)
        Boolean mask indicating which voxels pass the spike filter
        (True = kept, False = discarded).
    """
    if pts_free.size == 0:
        return np.zeros(0, dtype=bool)

    dirs = fibonacci_directions(n_dirs)
    # Effective radii used for intersection test (vdW + spike_probe)
    radii = atom_vdw + float(spike_probe)
    r2 = radii * radii
    M = atom_pos.shape[0]
    keep_all = np.ones(pts_free.shape[0], dtype=bool)

    # Process voxels in blocks to limit memory use
    for p0 in range(0, pts_free.shape[0], block_pts):
        p1 = min(p0 + block_pts, pts_free.shape[0])
        P = pts_free[p0:p1]  # (m, 3)
        m = P.shape[0]
        keep = np.ones(m, dtype=bool)

        # For each direction, check whether each voxel "sees" any sphere
        for u in dirs:  # u: (3,)
            if not keep.any():
                break  # early exit if all voxels already discarded

            hits_any = np.zeros(m, dtype=bool)

            # Process atoms in chunks
            for a0 in range(0, M, chunk_atoms):
                a1 = min(a0 + chunk_atoms, M)
                C = atom_pos[a0:a1]   # (k, 3)
                r2k = r2[a0:a1]       # (k,)
                rk = radii[a0:a1]     # (k,)

                diff = P[:, None, :] - C[None, :, :]  # (m, k, 3)
                d2 = np.sum(diff * diff, axis=2)      # (m, k)

                # Distance threshold: sphere can be hit only if centre is
                # within (max_len + radius) of the voxel
                thresh = (max_len + rk)[None, :] ** 2
                near = d2 <= thresh
                if not near.any():
                    continue

                # Project diff onto ray direction u
                um = np.einsum("ijk,k->ij", diff, u)  # (m, k)
                # Discriminant of ray-sphere intersection
                disc = um * um - (d2 - r2k[None, :])  # (m, k)
                hit = disc >= 0.0
                if not hit.any():
                    continue

                sqrtD = np.sqrt(np.maximum(disc, 0.0))
                t_enter = -um - sqrtD
                t_exit = -um + sqrtD

                # We require that the intersection (if any) is in front of
                # the voxel (t_exit >= 0) and not beyond max_len
                hit &= (t_exit >= 0.0) & (t_enter <= max_len)
                hit &= near

                # hits_any[v] becomes True if voxel v hits ANY sphere in this direction
                hits_any |= hit.any(axis=1)

            # To keep the voxel overall, it must see at least one atom in THIS direction
            keep &= hits_any
            if not keep.any():
                break

        keep_all[p0:p1] = keep

    return keep_all


# =========================================================================
# BW parsing and BW-based vertices
# =========================================================================

def parse_bw_list(bw_str):
    """
    Parse a string of BW-like values into a list of floats.

    Parameters
    ----------
    bw_str : str
        String like "1.53,1.57,2.43,2.46".

    Returns
    -------
    list of float
        Parsed BW values with duplicates removed (first occurrence kept).
    """
    vals = []
    for tok in bw_str.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except Exception:
            pass
    # Remove duplicates while preserving order
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def match_bw_value(occ, target, tol=0.015):
    """
    Decide whether an occupancy value matches a BW target within tolerance.

    Parameters
    ----------
    occ : float or None
        Occupancy from PDB line, repurposed to hold TM.BW.
    target : float
        Desired BW-like value (e.g. 1.53).
    tol : float, optional
        Allowed absolute deviation.

    Returns
    -------
    bool
        True if occ matches the target BW within the tolerance.
    """
    if occ is None:
        return False
    if int(occ) != int(target):
        return False
    return abs(occ - target) <= tol or round(occ, 2) == round(target, 2)


def collect_bw_vertices(lines, resid_to_idx, bw_values, R, com_tm):
    """
    Build low/high vertices per TM using only residues whose occupancy matches
    any of the supplied BW values.

    Parameters
    ----------
    lines : list of str
        PDB lines.
    resid_to_idx : dict
        Residue mapping.
    bw_values : list of float
        BW-like occupancy targets (e.g. [1.53, 1.57, 2.43, ...]).
    R : np.ndarray, shape (3,3)
        Rotation matrix used to align the TM bundle to +Z.
    com_tm : np.ndarray, shape (3,)
        Centre of mass of TM residues (translation used for alignment).

    Returns
    -------
    walls : dict or None
        Mapping TM id -> {"low": (3,), "high": (3,)} in aligned coords.
    tm_ids_sorted : list of int
        TM ids used in this BW polyhedron, sorted by azimuthal angle.

    Notes
    -----
    - Each BW value (e.g. 1.53) implies a TM index via its integer part (1).
    - For each residue whose occupancy matches any BW value:
        * the residue Cα is transformed into the aligned frame,
        * stored in by_tm[TM_index].
    - For each TM, low/high along Z are extracted and then the TMs are sorted
      by their XY centre angle to build a closed polygon.
    - At least 3 TMs are required to build a valid polyhedron.
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
                by_tm[int(b)].append(p)
                break

    if len(by_tm) < 3:
        return None, []

    centers_xy = {}
    walls = {}
    for tm_id, pts in by_tm.items():
        P = np.vstack(pts)
        z = P[:, 2]
        low = P[int(np.argmin(z))]
        high = P[int(np.argmax(z))]
        walls[tm_id] = {"low": low, "high": high}
        c_xy = 0.5 * (low[:2] + high[:2])
        centers_xy[tm_id] = math.atan2(c_xy[1], c_xy[0])

    tm_ids_sorted = sorted(walls.keys(), key=lambda t: centers_xy[t])
    return walls, tm_ids_sorted


# =========================================================================
# Voxelisation: slice polyhedron, build polygons, fill grid
# =========================================================================

def mesh_intersection_polys(tris, z):
    """
    Intersect a triangular mesh with a plane z = constant and build polygons.

    Parameters
    ----------
    tris : list of np.ndarray
        List of triangles (3x3 arrays).
    z : float
        Z value of the slicing plane.

    Returns
    -------
    list of np.ndarray
        List of polygons in XY (each an array of shape (n_vertices, 2)).
    """
    segments = []
    for tri in tris:
        seg = tri_plane_intersect_segment(tri, z)
        if seg is not None:
            segments.append(seg)
    return segments_to_polygons(segments)


def voxelize_inside_mesh(
    tris,
    spacing,
    atom_pos,
    atom_rad,
    zmin,
    zmax,
    chunk_atoms=4000,
    chunk_pts=30000,
):
    """
    Voxelise the interior of a polyhedral mesh and discard sterically occupied voxels.

    Parameters
    ----------
    tris : list of np.ndarray
        Triangular mesh (list of 3x3 arrays).
    spacing : float
        Voxel grid spacing (Å).
    atom_pos : np.ndarray, shape (M, 3)
        Atom coordinates (aligned).
    atom_rad : np.ndarray, shape (M,)
        Exclusion radii for each atom (vdW + probe).
    zmin, zmax : float
        Z limits for slicing the mesh.
    chunk_atoms : int, optional
        Number of atoms per chunk in distance checks.
    chunk_pts : int, optional
        Number of voxels per chunk in distance checks.

    Returns
    -------
    pts_free : np.ndarray, shape (N, 3)
        Voxel centres that are inside the mesh and do not overlap any atom.
    z_slices : np.ndarray
        Z positions of slice planes used during voxelisation.
    """
    L = zmax - zmin
    Ns = max(1, int(math.ceil(L / spacing)))
    if Ns > 1:
        z_slices = np.linspace(
            zmin + 0.5 * L / Ns, zmax - 0.5 * L / Ns, Ns
        )
    else:
        z_slices = np.array([0.5 * (zmin + zmax)])

    all_pts = []
    for z in z_slices:
        polys = mesh_intersection_polys(tris, z)
        if not polys:
            continue

        # Compute XY bounding box of all polygons
        xmins = [p[:, 0].min() for p in polys]
        xmaxs = [p[:, 0].max() for p in polys]
        ymins = [p[:, 1].min() for p in polys]
        ymaxs = [p[:, 1].max() for p in polys]
        xmin, xmax = min(xmins), max(xmaxs)
        ymin, ymax = min(ymins), max(ymaxs)

        xs = np.arange(xmin, xmax + 1e-6, spacing)
        ys = np.arange(ymin, ymax + 1e-6, spacing)
        if xs.size == 0 or ys.size == 0:
            continue

        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        XY = np.column_stack([XX.ravel(), YY.ravel()])

        # Mask of points inside the union of polygons
        inside = np.zeros(len(XY), dtype=bool)
        for poly in polys:
            inside |= np.array(
                [point_in_polygon(p[0], p[1], poly) for p in XY]
            )
        if not np.any(inside):
            continue

        # Build 3D candidate points (voxel centres)
        P3 = np.column_stack([XY[inside], np.full(np.count_nonzero(inside), z)])

        # Exclude voxels overlapping atoms (vdW+probe)
        occupied = np.zeros(P3.shape[0], dtype=bool)
        for a0 in range(0, atom_pos.shape[0], chunk_atoms):
            a1 = min(a0 + chunk_atoms, atom_pos.shape[0])
            apos = atom_pos[a0:a1]
            arad = atom_rad[a0:a1]
            for p0 in range(0, P3.shape[0], chunk_pts):
                p1 = min(p0 + chunk_pts, P3.shape[0])
                pts = P3[p0:p1]
                d2 = np.sum((pts[:, None, :] - apos[None, :, :]) ** 2, axis=2)
                occupied[p0:p1] |= np.any(d2 <= (arad[None, :] ** 2), axis=1)

        P3_free = P3[~occupied]
        if P3_free.size:
            all_pts.append(P3_free)

    if all_pts:
        return np.vstack(all_pts), z_slices
    else:
        return np.zeros((0, 3), float), z_slices


# =========================================================================
# Main driver: alignment, mesh, voxelisation, spike filter, BW polyhedron
# =========================================================================

def run(
    inpdb,
    outpdb,
    tm_include,
    spacing,
    probe,
    voxels_free,
    voxels_bwpoly,
    caps_pdb,
    spike_dirs,
    spike_max,
    spike_probe,
    bw_str,
):
    """
    Main workflow: read PDB, align TM bundle, build polyhedron, voxelise,
    apply spike filter, and optionally process a BW-defined polyhedron.

    Parameters
    ----------
    inpdb : str
        Input PDB path.
    outpdb : str or None
        Optional output PDB path for the aligned structure.
    tm_include : str
        TM ids to include, e.g. "1-7".
    spacing : float
        Voxel spacing (Å).
    probe : float
        Extra probe added to vdW for initial exclusion (vdW + probe).
    voxels_free : str or None
        If not None, path to write all final voxels inside the total TM
        envelope (after spike filter) as a PDB (HETATM VOX).
    voxels_bwpoly : str or None
        If not None, path to write voxels inside the BW polyhedron (after
        spike filter) as a PDB.
    caps_pdb : str or None
        If not None, path to write diagnostic low/high points per TM as PDB.
    spike_dirs : int
        Number of spike directions (rays) per voxel.
    spike_max : float
        Maximum ray length (Å) for the spike filter.
    spike_probe : float
        Extra probe added to vdW in the spike intersection test.
    bw_str : str or None
        BW list string, e.g. "1.53,1.57,2.43,..." for BW polyhedron
        construction.
    """
    # --- Read PDB file ---
    with open(inpdb, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    resid_to_idx, atom_xyz, atom_map = group_by_residue(lines)

    # --- Identify TM residues from occupancy ---
    tm_keys = collect_tm_residues(lines, resid_to_idx, tm_include)

    # --- Alignment: define TM bundle frame ---
    ca_pts = np.vstack([residue_ca(lines, resid_to_idx[k]) for k in tm_keys])
    com_tm = com_from_residues(lines, resid_to_idx, tm_keys)
    axis = fit_tls_axis(ca_pts - com_tm)
    R = R_from_a_to_b(axis, np.array([0.0, 0.0, 1.0]))

    # Transform all atom coordinates
    xyz_c = atom_xyz - com_tm
    xyz_r = (R @ xyz_c.T).T

    # Optionally write aligned PDB
    if outpdb:
        out_lines = list(lines)
        j = 0
        for i in range(len(lines)):
            if is_atom_line(lines[i]):
                x, y, z = xyz_r[j]
                out_lines[i] = rewrite_xyz(lines[i], x, y, z)
                j += 1
        with open(outpdb, "w") as g:
            for ln in out_lines:
                g.write(ln + "\n")

    # --- Build Cα sets per TM (aligned frame) for the TOTAL solid ---
    tms = defaultdict(list)
    for k in tm_keys:
        idxs = resid_to_idx[k]
        ca = residue_ca(lines, idxs)
        ca_r = R @ (ca - com_tm)
        tm_id = int(parse_occ(lines[idxs[0]]))
        tms[tm_id].append(ca_r)

    tm_ids = sorted([tid for tid in tms if len(tms[tid]) >= 2])
    if len(tm_ids) < 3:
        raise RuntimeError("Not enough TMs (≥3) to close the polygon.")

    # Order TMs by azimuthal angle in XY for the TOTAL solid
    angles = {}
    for tid in tm_ids:
        A = np.vstack(tms[tid])
        c = A[:, :2].mean(axis=0)
        angles[tid] = math.atan2(c[1], c[0])
    tm_ids_sorted = sorted(tm_ids, key=lambda t: angles[t])

    # Extract low/high per TM for the TOTAL solid
    walls_total = {}
    lows = []
    highs = []
    for tid in tm_ids_sorted:
        A = np.vstack(tms[tid])
        z = A[:, 2]
        low, high = A[int(np.argmin(z))], A[int(np.argmax(z))]
        walls_total[tid] = {"low": low, "high": high}
        lows.append(low)
        highs.append(high)
    lows = np.vstack(lows)
    highs = np.vstack(highs)

    # --- Triangular mesh and voxelisation of the TOTAL solid ---
    tris_total = build_mesh_from_tms(walls_total, tm_ids_sorted)
    zmin_total = min(lows[:, 2].min(), highs[:, 2].min())
    zmax_total = max(lows[:, 2].max(), highs[:, 2].max())

    # Atom positions and exclusion radii (vdW + probe) for initial voxel pruning
    elements = [parse_element(lines[i]) for i in atom_map]
    atom_pos = xyz_r
    atom_vdw = np.array([VDW.get(el.upper(), 1.70) for el in elements])
    atom_excl = atom_vdw + float(probe)

    pts_total, _ = voxelize_inside_mesh(
        tris_total,
        spacing,
        atom_pos,
        atom_excl,
        zmin_total,
        zmax_total,
    )

    # --- Ray-casting ("spike") filter for TOTAL voxels ---
    if pts_total.shape[0]:
        keep_total = filter_voxels_by_spikes(
            pts_total,
            atom_pos,
            atom_vdw,
            max_len=float(spike_max),
            n_dirs=int(spike_dirs),
            spike_probe=float(spike_probe),
            chunk_atoms=1000,
            block_pts=2000,
        )
        pts_total = pts_total[keep_total]

    volume_total = pts_total.shape[0] * (spacing**3)
    print(
        f"[TOTAL] Spike-filtered free volume: {volume_total:.1f} Å^3  |  "
        f"N_voxels = {pts_total.shape[0]}"
    )

    # Optionally write all TOTAL voxels to PDB
    if voxels_free:
        with open(voxels_free, "w") as fv:
            s = 1
            for (x, y, z) in pts_total:
                fv.write(
                    f"HETATM{s:>5}  VOX VOX VALL     "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                    f"{1.00:>6.2f}{0.00:>6.2f}\n"
                )
                s += 1

    # --- Optional BW polyhedron and volume ---
    if bw_str:
        bw_vals = parse_bw_list(bw_str)
        walls_bw, tm_ids_bw = collect_bw_vertices(
            lines, resid_to_idx, bw_vals, R, com_tm
        )
        if not tm_ids_bw:
            print("[BW] Warning: could not build BW polyhedron "
                  "(are those BW positions present in the PDB?).")
        else:
            tris_bw = build_mesh_from_tms(walls_bw, tm_ids_bw)
            lows_bw = np.vstack([walls_bw[tid]["low"] for tid in tm_ids_bw])
            highs_bw = np.vstack([walls_bw[tid]["high"] for tid in tm_ids_bw])
            zmin_bw = min(lows_bw[:, 2].min(), highs_bw[:, 2].min())
            zmax_bw = max(lows_bw[:, 2].max(), highs_bw[:, 2].max())

            pts_bw, _ = voxelize_inside_mesh(
                tris_bw,
                spacing,
                atom_pos,
                atom_excl,
                zmin_bw,
                zmax_bw,
            )

            if pts_bw.shape[0]:
                keep_bw = filter_voxels_by_spikes(
                    pts_bw,
                    atom_pos,
                    atom_vdw,
                    max_len=float(spike_max),
                    n_dirs=int(spike_dirs),
                    spike_probe=float(spike_probe),
                    chunk_atoms=1000,
                    block_pts=2000,
                )
                pts_bw = pts_bw[keep_bw]

            volume_bw = pts_bw.shape[0] * (spacing**3)
            print(
                f"[BW-POLY] TMs={tm_ids_bw} | Z=[{zmin_bw:.2f},{zmax_bw:.2f}] Å | "
                f"Volume: {volume_bw:.1f} Å^3 | N_voxels={pts_bw.shape[0]}"
            )

            if voxels_bwpoly:
                with open(voxels_bwpoly, "w") as fb:
                    s = 1
                    for (x, y, z) in pts_bw:
                        fb.write(
                            f"HETATM{s:>5}  VOX VOX VBW      "
                            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                            f"{1.00:>6.2f}{0.00:>6.2f}\n"
                        )
                        s += 1

    # --- Optional caps PDB (diagnostic low/high points) ---
    if caps_pdb:
        with open(caps_pdb, "w") as fcap:
            s = 1
            # Lower caps
            for tid in tm_ids_sorted:
                p = walls_total[tid]["low"]
                fcap.write(
                    f"HETATM{s:>5}  CBL CAP L   1    "
                    f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}\n"
                )
                s += 1
            # Upper caps
            for tid in tm_ids_sorted:
                p = walls_total[tid]["high"]
                fcap.write(
                    f"HETATM{s:>5}  CBT CAP U   1    "
                    f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}\n"
                )
                s += 1


# =========================================================================
# Command-line interface
# =========================================================================

def parse_args():
    """
    Parse command-line arguments for the cavity volume calculation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes corresponding to the defined options.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Free volume inside a TM Cα-defined solid (non-planar caps) "
            "with ray-casting filter and optional BW polyhedron."
        )
    )
    ap.add_argument("--in", dest="inpdb", required=True,
                    help="Input PDB file.")
    ap.add_argument("--out", dest="outpdb", default=None,
                    help="Optional output PDB with aligned coordinates.")
    ap.add_argument("--tm-include", default="1-7",
                    help="TM indices to include, e.g. '1-7'.")
    ap.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Voxel spacing (Å).",
    )
    ap.add_argument(
        "--probe",
        type=float,
        default=0.0,
        help="Probe added to vdW radii for initial steric exclusion (vdW + probe).",
    )
    ap.add_argument(
        "--voxels-free",
        dest="voxels_free",
        default=None,
        help="PDB output for ALL internal voxels (TOTAL solid).",
    )
    ap.add_argument(
        "--voxels-bwpoly",
        dest="voxels_bwpoly",
        default=None,
        help="PDB output for voxels inside BW polyhedron.",
    )
    ap.add_argument(
        "--caps-pdb",
        dest="caps_pdb",
        default=None,
        help="PDB with low/high cap points per TM (diagnostics).",
    )
    # Spike (ray-casting) parameters
    ap.add_argument(
        "--spike-dirs",
        type=int,
        default=20,
        help="Number of spike directions (rays) used in the ray-casting filter.",
    )
    ap.add_argument(
        "--spike-max",
        type=float,
        default=10.0,
        help="Maximum ray length (Å).",
    )
    ap.add_argument(
        "--spike-probe",
        type=float,
        default=0.0,
        help="Extra probe added to vdW radii in the spike intersection test.",
    )
    # BW anchors
    ap.add_argument(
        "--bw",
        type=str,
        default=None,
        help='Comma-separated BW list, e.g. "1.53,1.57,2.43,2.46,...".',
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.inpdb,
        args.outpdb,
        args.tm_include,
        args.spacing,
        args.probe,
        args.voxels_free,
        args.voxels_bwpoly,
        args.caps_pdb,
        args.spike_dirs,
        args.spike_max,
        args.spike_probe,
        args.bw,
    )

