#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPCR distances/angles 

Summary

Calculates heavy–heavy/CA–CA distances and dihedrals (chi1/chi2 + TM6 kink) from BW candidates within a ±N-position window (default 2). Does not compute volumes. The “alternative” metrics (alternative ionic lock, network, DRY) can be rigid or flexible by residue identity, while keeping positional rigidity (BW ± window).

Policy (--policy)

strict: canonical identities
    alt_lock: 3.49 = ASP ; 6.30 = LYS
    network: 3.39 = GLU ; 6.40 = HIS
    dry: 3.49 = ASP ; 3.50 = ARG

mid (default): conservative acid↔acid, canonical basic
    alt_lock: 3.49 ∈ {ASP, GLU} ; 6.30 = LYS
    network: 3.39 ∈ {GLU, ASP} ; 6.40 = HIS
    dry: 3.49 ∈ {ASP, GLU} ; 3.50 = ARG

loose: broadens the basic side when reasonable (exploratory)
    alt_lock: 3.49 ∈ {ASP, GLU} ; 6.30 ∈ {LYS, ARG}
    network: 3.39 ∈ {GLU, ASP} ; 6.40 ∈ {HIS, LYS, ARG}
    dry: 3.49 ∈ {ASP, GLU} ; 3.50 ∈ {ARG, LYS}

For each flexible metric, it also reports WHICH residue@BW combination was used for the minimum (columns with suffix _used).

Use
---
  python3 1_gpcr_distances_only.py \
    --in "." \
    --chain A \
    --bw-window 2 \
    --out gpcr_distances.csv \
    --policy mid \
    --workers 6 \
    --familias familias.txt

Output
CSV including, among others:
ionic_lock_R3.50_E/D6.30, ionic_lock_CA_R3.50_E/D6.30,
ionic_lock_used_3.50, ionic_lock_used_6.30,
ionic_lock_used_bw_3.50, ionic_lock_used_bw_6.30,
alt_lock_min_dist, alt_lock_CA_3.49_6.30,
network_min_dist, dry_min_dist, dry_CA_3.49_3.50,
tm3_tm6_ic_CA_3.50_6.34, connector_CA_3.40_6.44, sodium_net_D2.50_N7.49, portal_EC_CA_5.39_6.58,
W6.48_chi2_deg, Y7.53_chi1_deg, CWxP_kink_TM6_deg
plus columns with suffixes _present and _used for traceability.

If --familias is provided, also adds:
  - state: 'active' / 'inactive' / 'unknown'
  - gpcr_class: mapped from familias.txt (first column -> last column)

Additionally, generates an Excel file (same basename as --out, with .xlsx) with sheets:
  - all
  - only_trad   (ionic_lock_present=1, alt_lock_present=0)
  - only_alt    (ionic_lock_present=0, alt_lock_present=1)
  - both        (ionic_lock_present=1, alt_lock_present=1)
  - none        (ionic_lock_present=0, alt_lock_present=0)
"""

import argparse, glob, math, os, sys
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd  # Para Excel y familias

# ---------- Geometría básica ----------
def dist(a, b): return math.dist(a, b)

def torsion(p1, p2, p3, p4):
    b1 = np.array(p2) - np.array(p1)
    b2 = np.array(p3) - np.array(p2)
    b3 = np.array(p4) - np.array(p3)
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2))
    x = float(np.dot(n1, n2)); y = float(np.dot(m1, n2))
    return math.degrees(math.atan2(y, x))

def angle_between(v1, v2):
    v1 = np.array(v1, float); v2 = np.array(v2, float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return float("nan")
    c = np.dot(v1, v2) / (n1*n2); c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))

# ---------- Modelo PDB mínimo + BW a partir de Occ/Bfac ----------
class Atom:
    __slots__=("name","resname","chain","resseq","coord","occ","bfac","element")
    def __init__(self, name, resname, chain, resseq, coord, occ, bfac, element):
        self.name=name.strip(); self.resname=resname.strip(); self.chain=(chain or "").strip()
        self.resseq=resseq; self.coord=coord; self.occ=occ; self.bfac=bfac; self.element=(element or "").strip()

class Residue:
    def __init__(self, resname, chain, resseq):
        self.resname=resname; self.chain=chain; self.resseq=resseq
        self.atoms: Dict[str, Atom] = {}; self.bw: Optional[str] = None
    def add_atom(self, atom: Atom): self.atoms[atom.name]=atom
    def get_ca(self): return self.atoms.get("CA")
    def heavy_atoms(self): return [a for a in self.atoms.values() if (not a.element) or (a.element.upper()!="H")]

def parse_pdb(path: str, chain_filter: Optional[str] = None):
    """
    Lee el PDB y construye:
      - residues[(chain, resseq)] -> Residue
      - bw_index["H.PP"] -> Residue  (p.ej., "3.49")
    Corregido: prioriza CA y detecta valores tipo BW en occupancy/B-factor.
    Evita confundir ocupancias 1.00 con números BW.
    """
    residues: Dict[Tuple[str,int], Residue] = {}
    chains_present=set()

    with open(path, "r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            name=line[12:16].strip(); resname=line[17:20].strip()
            chain=(line[21].strip() or "")
            resseq_str=line[22:26]
            try: resseq=int(resseq_str)
            except ValueError: continue
            x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
            occ_str=line[54:60].strip(); bfac_str=line[60:66].strip()
            try: occ=float(occ_str) if occ_str else 0.0
            except Exception: occ=0.0
            try: bfac=float(bfac_str) if bfac_str else 0.0
            except Exception: bfac=0.0
            element=line[76:78].strip()
            chains_present.add(chain)
            if (chain_filter is not None) and (chain != chain_filter):
                continue
            key=(chain, resseq)
            if key not in residues: residues[key]=Residue(resname, chain, resseq)
            residues[key].add_atom(Atom(name,resname,chain,resseq,(x,y,z),occ,bfac,element))

    # --- Helpers para reconocer BW "válido" ---
    def looks_like_bw(v: float) -> bool:
        # BW típico entre 1.00 y 8.99 (TM1–TM8); ignorar exactamente 1.00 (ocupancia real)
        if not (1.0 <= v < 9.0): return False
        if abs(v - 1.00) < 1e-6: return False
        return True

    def pick_bw_for_res(res: Residue) -> Optional[str]:
        # 1) Priorizar CA: primero occupancy, luego B-factor
        ca = res.atoms.get("CA")
        if ca:
            if looks_like_bw(ca.occ):  return f"{ca.occ:.2f}"
            if looks_like_bw(ca.bfac): return f"{ca.bfac:.2f}"
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
def _parse_bw(bw: str): h,p = f"{float(bw):.2f}".split("."); return int(h), int(p)
def _format_bw(h:int, p:int): return f"{h}.{p:02d}"

def get_candidates_by_bw(bw_index: Dict[str, Residue], target_bw: str, window: int=2):
    if not target_bw: return []
    h,p=_parse_bw(target_bw); cands=[]
    for q in [p]+[p+d for d in range(-window,window+1) if d!=0]:
        if 0<=q<=99:
            key=_format_bw(h,q)
            if key in bw_index: cands.append((key,bw_index[key],abs(q-p)))
    cands.sort(key=lambda t:(t[2],abs(_parse_bw(t[0])[1]-p)))
    return cands

def get_candidate_set(bw_index: Dict[str, Residue], bws: List[str], window: int):
    out=[]; seen=set()
    for bw in bws:
        for key,res,delta in get_candidates_by_bw(bw_index,bw,window):
            if key not in seen: out.append((key,res,delta)); seen.add(key)
    return out

# ---------- Clasificación y utilidades ----------
def normalize_resname(rn: str) -> str:
    rn=(rn or "").upper()
    return {"HID":"HIS","HIE":"HIS","HIP":"HIS","ASH":"ASP","GLH":"GLU","LYN":"LYS","ARN":"ARG"}.get(rn,rn)

def filter_by_resname(cands, allowed: set):
    return [(bw,res,d) for (bw,res,d) in cands if normalize_resname(res.resname) in allowed]

# ---------- Distancias/dihedros ----------
def min_dist_between_residues(res1: Residue, res2: Residue, only_names1=None, only_names2=None) -> float:
    if res1 is None or res2 is None: return float("nan")
    a1=res1.heavy_atoms(); a2=res2.heavy_atoms()
    if only_names1: a1=[a for a in a1 if a.name in only_names1]
    if only_names2: a2=[a for a in a2 if a.name in only_names2]
    if not a1 or not a2: return float("nan")
    return min(dist(x.coord,y.coord) for x in a1 for y in a2)

def ca_distance(res1: Residue, res2: Residue) -> float:
    if (res1 is None) or (res2 is None): return float("nan")
    a1=res1.get_ca(); a2=res2.get_ca()
    if (a1 is None) or (a2 is None): return float("nan")
    return dist(a1.coord,a2.coord)

def torsion_chi(res: Residue, chi="chi1"):
    if res is None: return float("nan")
    rn=normalize_resname(res.resname)
    if chi=="chi1": names=("N","CA","CB","CG")
    elif chi=="chi2":
        if rn not in ("TRP","TYR","PHE"): return float("nan")
        names=("CA","CB","CG","CD1")
    else: return float("nan")
    try: p=[res.atoms[n].coord for n in names]
    except KeyError: return float("nan")
    return torsion(*p)

def chi2_aromatic(res: Residue):
    rn=normalize_resname(res.resname) if res else ""
    if rn not in ("TRP","TYR","PHE"): return float("nan")
    try:
        p1=res.atoms["CA"].coord; p2=res.atoms["CB"].coord; p3=res.atoms["CG"].coord
    except Exception: return float("nan")
    p4=None
    for name in ("CD1","CD2","CE1","CE2"):
        if name in res.atoms: p4=res.atoms[name].coord; break
    if p4 is None: return float("nan")
    return torsion(p1,p2,p3,p4)

# ---------- CWxP kink (TM6) ----------
def centroid(res_list):
    acc=[]
    for r in res_list:
        a=r.get_ca()
        if a is not None: acc.append(a.coord)
    return tuple(np.mean(np.array(acc,float),axis=0)) if acc else None

def cwXp_kink_tm6_deg(bw_index, window=2):
    def pick_set(bws):
        cands=get_candidate_set(bw_index,bws,window); cands=sorted(cands,key=lambda t:t[2])
        return [r for _,r,_ in cands]
    pre=pick_set(["6.44","6.46","6.48"]); post=pick_set(["6.52","6.54","6.56"])
    pivot_cand=get_candidates_by_bw(bw_index,"6.50",window)
    pivot=pivot_cand[0][1] if pivot_cand else None
    if (not pre) or (not post) or (pivot is None) or (pivot.get_ca() is None): return float("nan")
    c_pre=centroid(pre); c_post=centroid(post); c_pivot=pivot.get_ca().coord
    if c_pre is None or c_post is None: return float("nan")
    v1=np.array(c_pivot)-np.array(c_pre); v2=np.array(c_post)-np.array(c_pivot)
    ang=angle_between(v1,v2); return float("nan") if math.isnan(ang) else 180.0-ang

# ---------- Política de identidades permitidas ----------
def allowed_resnames_for(metric: str, site: str, policy: str) -> set:
    policy = (policy or "mid").lower()
    if metric == "alt_lock":  # 3.49 ↔ 6.30
        if site == "3.49":
            return {"ASP"} if policy=="strict" else {"ASP","GLU"}
        if site == "6.30":
            return {"LYS"} if policy in ("strict","mid") else {"LYS","ARG"}
    if metric == "network":   # 3.39 ↔ 6.40
        if site == "3.39":
            return {"GLU"} if policy=="strict" else {"GLU","ASP"}
        if site == "6.40":
            return {"HIS"} if policy in ("strict","mid") else {"HIS","LYS","ARG"}
    if metric == "dry":       # 3.49 ↔ 3.50
        if site == "3.49":
            return {"ASP"} if policy=="strict" else {"ASP","GLU"}
        if site == "3.50":
            return {"ARG"} if policy in ("strict","mid") else {"ARG","LYS"}
    return set()

# Mapas de nombres atómicos por tipo
ACID_ATOMS = {"ASP": ["OD1","OD2"], "GLU": ["OE1","OE2"]}
BASIC_ATOMS = {
    "ARG": ["NE","NH1","NH2"],
    "LYS": ["NZ"],
    "HIS": ["ND1","NE2"],
}

def min_heavy_over_candidates_flexible(acid_dict, pos_dict):
    """
    acid_dict: {"ASP": [(bw,res,delta), ...], "GLU": [...]}
    pos_dict : {"ARG": [...], "LYS": [...], "HIS": [...]}
    Devuelve: (best_dist, used_acid_type, used_pos_type, used_acid_bw, used_pos_bw)
    """
    best = (float("nan"), None, None, None, None)
    found = False
    for atype, alist in acid_dict.items():
        if not alist: continue
        a_atoms = ACID_ATOMS.get(atype, None)
        if not a_atoms: continue
        for btype, blist in pos_dict.items():
            if not blist: continue
            b_atoms = BASIC_ATOMS.get(btype, None)
            if not b_atoms: continue
            for abw, ares, _ in alist:
                for bbw, bres, _ in blist:
                    d = min_dist_between_residues(ares, bres, a_atoms, b_atoms)
                    if math.isnan(d): continue
                    if (not found) or (d < best[0]):
                        best = (d, atype, btype, abw, bbw)
                        found = True
    return best

# ---------- Helpers de anotación de clase/estado ----------
def load_familias(path: Path):
    try:
        fam = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        fam = pd.read_csv(path, sep="\t")
    if fam.shape[1] < 2:
        raise ValueError("familias.txt must have at least two columns (ID and Class).")
    first_col = fam.columns[0]
    last_col  = fam.columns[-1]
    mapping = dict(zip(fam[first_col].astype(str).str.upper().str.strip(),
                       fam[last_col].astype(str).str.strip()))
    return mapping

def infer_state(filename: str) -> str:
    f = str(filename).lower().strip()
    if f.endswith("_active.pdb"):
        return "active"
    if f.endswith("_inactive.pdb"):
        return "inactive"
    return "unknown"

def file_prefix_lower(filename: str) -> str:
    base = Path(str(filename)).name
    return base.split("_", 1)[0].lower()

def infer_gpcr_class(filename: str, map_upper: dict) -> str:
    if map_upper is None:
        return "Unknown"
    key_up = file_prefix_lower(filename).upper()
    return map_upper.get(key_up, "Unknown")

def infer_receptor_name(filename: str) -> str:
    """
    Devuelve el nombre del receptor sin sufijos como:
      - _all_active.pdb
      - _all_inactive.pdb
      - _active.pdb
      - _inactive.pdb
    """
    base = Path(str(filename)).name
    for suffix in ["_all_active.pdb", "_all_inactive.pdb",
                   "_active.pdb", "_inactive.pdb"]:
        if base.endswith(suffix):
            return base[:-len(suffix)]
    # Por defecto, nombre sin extensión
    return Path(base).stem

# ---------- Orquestación por PDB ----------
def compute_distances(path, chain, bw_window, policy):
    residues, bw_index, _ = parse_pdb(path, chain_filter=chain)
    if not bw_index: return ("SKIP","sin BW en occupancy/B-factor"), None

    # Candidatos por BW
    c_350=get_candidates_by_bw(bw_index,"3.50",bw_window)
    c_349=get_candidates_by_bw(bw_index,"3.49",bw_window)
    c_340=get_candidates_by_bw(bw_index,"3.40",bw_window)
    c_339=get_candidates_by_bw(bw_index,"3.39",bw_window)
    c_630=get_candidates_by_bw(bw_index,"6.30",bw_window)
    c_640=get_candidates_by_bw(bw_index,"6.40",bw_window)
    c_644=get_candidates_by_bw(bw_index,"6.44",bw_window)
    c_648_all=get_candidates_by_bw(bw_index,"6.48",bw_window)
    c_634=get_candidates_by_bw(bw_index,"6.34",bw_window)
    c_539=get_candidates_by_bw(bw_index,"5.39",bw_window)
    c_658=get_candidates_by_bw(bw_index,"6.58",bw_window)
    c_250=get_candidates_by_bw(bw_index,"2.50",bw_window)
    c_749=get_candidates_by_bw(bw_index,"7.49",bw_window)
    c_753=get_candidates_by_bw(bw_index,"7.53",bw_window)

    # Helper para recuperar el residuo a partir del tipo y BW usados
    def _pick_res_from_dict(cdict, rtype, bwkey):
        if (rtype is None) or (bwkey is None):
            return None
        for bw, res, _ in cdict.get(rtype, []):
            if bw == bwkey:
                return res
        return None

    # ---- Métricas con política ----
    # Alternative ionic lock: (3.49 ácido) ↔ (6.30 básico)
    allowed_349 = allowed_resnames_for("alt_lock", "3.49", policy)
    allowed_630 = allowed_resnames_for("alt_lock", "6.30", policy)
    c349_any = filter_by_resname(c_349, allowed_349) if allowed_349 else []
    c630_any = filter_by_resname(c_630, allowed_630) if allowed_630 else []
    acid_dict_349 = {
        "ASP": [t for t in c349_any if normalize_resname(t[1].resname)=="ASP"],
        "GLU": [t for t in c349_any if normalize_resname(t[1].resname)=="GLU"],
    }
    pos_dict_630 = {
        "LYS": [t for t in c630_any if normalize_resname(t[1].resname)=="LYS"],
        "ARG": [t for t in c630_any if normalize_resname(t[1].resname)=="ARG"],
    }
    alt_best = min_heavy_over_candidates_flexible(acid_dict_349, pos_dict_630)
    alt_lock, alt_used_349, alt_used_630, alt_used_bw349, alt_used_bw630 = alt_best
    alt_lock_present = int(bool(c349_any and c630_any))

    # network: (3.39 ácido) ↔ (6.40 básico)
    allowed_339 = allowed_resnames_for("network", "3.39", policy)
    allowed_640 = allowed_resnames_for("network", "6.40", policy)
    c339_any = filter_by_resname(c_339, allowed_339) if allowed_339 else []
    c640_any = filter_by_resname(c_640, allowed_640) if allowed_640 else []
    acid_dict_339 = {
        "ASP": [t for t in c339_any if normalize_resname(t[1].resname)=="ASP"],
        "GLU": [t for t in c339_any if normalize_resname(t[1].resname)=="GLU"],
    }
    pos_dict_640 = {
        "HIS": [t for t in c640_any if normalize_resname(t[1].resname)=="HIS"],
        "LYS": [t for t in c640_any if normalize_resname(t[1].resname)=="LYS"],
        "ARG": [t for t in c640_any if normalize_resname(t[1].resname)=="ARG"],
    }
    net_best = min_heavy_over_candidates_flexible(acid_dict_339, pos_dict_640)
    network_eh, net_used_339, net_used_640, net_used_bw339, net_used_bw640 = net_best
    network_eh_present = int(bool(c339_any and c640_any))

    # dry: (3.49 ácido) ↔ (3.50 básico)
    allowed_349_dry = allowed_resnames_for("dry", "3.49", policy)
    allowed_350_dry = allowed_resnames_for("dry", "3.50", policy)
    c350_any = filter_by_resname(c_350, allowed_350_dry) if allowed_350_dry else []
    c349_any_dry = filter_by_resname(c_349, allowed_349_dry) if allowed_349_dry else []
    acid_dict_349_dry = {
        "ASP": [t for t in c349_any_dry if normalize_resname(t[1].resname)=="ASP"],
        "GLU": [t for t in c349_any_dry if normalize_resname(t[1].resname)=="GLU"],
    }
    pos_dict_350 = {
        "ARG": [t for t in c350_any if normalize_resname(t[1].resname)=="ARG"],
        "LYS": [t for t in c350_any if normalize_resname(t[1].resname)=="LYS"],
    }
    dry_best = min_heavy_over_candidates_flexible(acid_dict_349_dry, pos_dict_350)
    dry, dry_used_349, dry_used_350, dry_used_bw349, dry_used_bw350 = dry_best

    # ---- CA–CA para alt_lock y DRY, y N–O/CA para ionic lock ----

    # CA–CA del alternative ionic lock (misma pareja que alt_lock_min_dist)
    alt_lock_ca = float("nan")
    if not math.isnan(alt_lock):
        res349_alt = _pick_res_from_dict(acid_dict_349, alt_used_349, alt_used_bw349)
        res630_alt = _pick_res_from_dict(pos_dict_630, alt_used_630, alt_used_bw630)
        if res349_alt is not None and res630_alt is not None:
            alt_lock_ca = ca_distance(res349_alt, res630_alt)

    # Ionic lock canónico R3.50–E/D6.30 (N–O + Cα–Cα)
    ionic_lock = float("nan")
    ionic_lock_ca = float("nan")
    ionic_lock_present = 0
    ionic_used_630 = None
    ionic_used_350 = None
    ionic_used_bw630 = None
    ionic_used_bw350 = None

    cand_350_R = filter_by_resname(c_350, {"ARG"}) if c_350 else []
    cand_630_D = filter_by_resname(c_630, {"ASP"}) if c_630 else []
    cand_630_E = filter_by_resname(c_630, {"GLU"}) if c_630 else []

    if cand_350_R and (cand_630_D or cand_630_E):
        ionic_lock_present = 1
        acid_dict_630 = {
            "ASP": cand_630_D,
            "GLU": cand_630_E,
        }
        pos_dict_350_ionic = {
            "ARG": cand_350_R,
        }
        ion_best = min_heavy_over_candidates_flexible(acid_dict_630, pos_dict_350_ionic)
        ionic_lock, ionic_used_630, ionic_used_350, ionic_used_bw630, ionic_used_bw350 = ion_best
        if not math.isnan(ionic_lock):
            res630_ion = _pick_res_from_dict(acid_dict_630, ionic_used_630, ionic_used_bw630)
            res350_ion = _pick_res_from_dict(pos_dict_350_ionic, ionic_used_350, ionic_used_bw350)
            if res630_ion is not None and res350_ion is not None:
                ionic_lock_ca = ca_distance(res350_ion, res630_ion)

    # CA–CA para la métrica DRY (misma pareja que dry_min_dist)
    dry_ca = float("nan")
    if not math.isnan(dry):
        res349_dry = _pick_res_from_dict(acid_dict_349_dry, dry_used_349, dry_used_bw349)
        res350_dry = _pick_res_from_dict(pos_dict_350, dry_used_350, dry_used_bw350)
        if (res349_dry is not None) and (res350_dry is not None):
            dry_ca = ca_distance(res349_dry, res350_dry)

    # ---- Otras métricas (como antes) ----
    tm3_tm6_ic = float("nan")
    if c_350 and c_634:
        best = float("inf")
        for _, r1, _ in c_350:
            for _, r2, _ in c_634:
                d = ca_distance(r1, r2)
                if not math.isnan(d) and d < best:
                    best = d
        tm3_tm6_ic = best if best < float("inf") else float("nan")

    connector = float("nan")
    if c_340 and c_644:
        best = float("inf")
        for _, r1, _ in c_340:
            for _, r2, _ in c_644:
                d = ca_distance(r1, r2)
                if not math.isnan(d) and d < best:
                    best = d
        connector = best if best < float("inf") else float("nan")

    sodium_net = float("nan")
    if c_250 and c_749:
        best = float("inf")
        for _, r1, _ in c_250:
            for _, r2, _ in c_749:
                d = min_dist_between_residues(r1, r2)
                if not math.isnan(d) and d < best:
                    best = d
        sodium_net = best if best < float("inf") else float("nan")

    portal_ec = float("nan")
    if c_539 and c_658:
        best = float("inf")
        for _, r1, _ in c_539:
            for _, r2, _ in c_658:
                d = ca_distance(r1, r2)
                if not math.isnan(d) and d < best:
                    best = d
        portal_ec = best if best < float("inf") else float("nan")

    # min CA entre sets TM5(EC) y TM6(EC)
    tm5_ec_set=get_candidate_set(bw_index,["5.38","5.39","5.40","5.41"],bw_window)
    tm6_ec_set=get_candidate_set(bw_index,["6.55","6.56","6.57","6.58","6.59"],bw_window)
    portal_ec_ca_minset = float("nan")
    if tm5_ec_set and tm6_ec_set:
        best = float("inf")
        for _, r1, _ in tm5_ec_set:
            for _, r2, _ in tm6_ec_set:
                d = ca_distance(r1, r2)
                if not math.isnan(d) and d < best: best = d
        portal_ec_ca_minset = best if best < float("inf") else float("nan")

    portal_ec_heavy_minset = float("nan")
    if tm5_ec_set and tm6_ec_set:
        best = float("inf")
        for _, r1, _ in tm5_ec_set:
            for _, r2, _ in tm6_ec_set:
                d = min_dist_between_residues(r1, r2)
                if not math.isnan(d) and d < best: best = d
        portal_ec_heavy_minset = best if best < float("inf") else float("nan")

    # W6.48 χ2 aromática priorizando TRP>TYR>PHE
    def arom_rank(resname:str)->int:
        rn=normalize_resname(resname); return {"TRP":0,"TYR":1,"PHE":2}.get(rn,3)
    c_648_arom=[t for t in c_648_all if normalize_resname(t[1].resname) in ("TRP","TYR","PHE")]
    c_648_sorted=sorted(c_648_arom,key=lambda t:(arom_rank(t[1].resname),t[2]))
    if c_648_sorted:
        w648_res_bw,w648_res,_=c_648_sorted[0]
        w648_chi2=chi2_aromatic(w648_res); w648_used_bw=w648_res_bw; w648_used_resname=normalize_resname(w648_res.resname)
    else:
        w648_chi2=float("nan"); w648_used_bw=""; w648_used_resname=""

    y753_res=c_753[0][1] if c_753 else None
    y753_chi1=torsion_chi(y753_res,"chi1")

    kink_tm6=cwXp_kink_tm6_deg(bw_index,bw_window)

    row={
        # Locks (canónico y alternativo) + network + DRY
        "ionic_lock_R3.50_E/D6.30": ionic_lock,
        "ionic_lock_present": ionic_lock_present,
        "ionic_lock_CA_R3.50_E/D6.30": ionic_lock_ca,
        "ionic_lock_used_3.50": (ionic_used_350 or ""),
        "ionic_lock_used_6.30": (ionic_used_630 or ""),
        "ionic_lock_used_bw_3.50": (ionic_used_bw350 or ""),
        "ionic_lock_used_bw_6.30": (ionic_used_bw630 or ""),

        "alt_lock_min_dist": alt_lock,
        "alt_lock_present": alt_lock_present,
        "alt_lock_CA_3.49_6.30": alt_lock_ca,
        "alt_lock_used_3.49": (alt_used_349 or ""),
        "alt_lock_used_6.30": (alt_used_630 or ""),
        "alt_lock_used_bw_3.49": (alt_used_bw349 or ""),
        "alt_lock_used_bw_6.30": (alt_used_bw630 or ""),

        "network_min_dist": network_eh,
        "network_present": network_eh_present,
        "network_used_3.39": (net_used_339 or ""),
        "network_used_6.40": (net_used_640 or ""),
        "network_used_bw_3.39": (net_used_bw339 or ""),
        "network_used_bw_6.40": (net_used_bw640 or ""),

        "dry_min_dist": dry,
        "dry_CA_3.49_3.50": dry_ca,
        "dry_used_3.49": (dry_used_349 or ""),
        "dry_used_3.50": (dry_used_350 or ""),
        "dry_used_bw_3.49": (dry_used_bw349 or ""),
        "dry_used_bw_3.50": (dry_used_bw350 or ""),

        # Distancias adicionales
        "tm3_tm6_ic_CA_3.50_6.34": tm3_tm6_ic,
        "connector_CA_3.40_6.44": connector,
        "sodium_net_D2.50_N7.49": sodium_net,
        "portal_EC_CA_5.39_6.58": portal_ec,
        "portal_EC_CA_minset": portal_ec_ca_minset,
        "portal_EC_heavy_minset": portal_ec_heavy_minset,

        # Angulos/dihedros
        "W6.48_chi2_deg": w648_chi2,
        "Y7.53_chi1_deg": y753_chi1,
        "W6.48_used_BW": w648_used_bw,
        "W6.48_used_resname": w648_used_resname,
        "CWxP_kink_TM6_deg": kink_tm6,
    }
    return ("OK","done"), row

# ---------- Utilidades CLI ----------
def expand_inputs(inp: str):
    if os.path.isdir(inp):
        files=[]
        for ext in ("*.pdb","*.ent","*.PDB"): files+=glob.glob(os.path.join(inp,ext))
        return sorted(files)
    if any(c in inp for c in "*?[]"): return sorted(glob.glob(inp))
    return [inp] if os.path.isfile(inp) else []

def auto_pick_chain(path: str) -> Optional[str]:
    residues, bw_index, chains_present = parse_pdb(path, chain_filter=None)
    chains_with_350={res.chain for res in residues.values() if res.bw=="3.50"}
    if chains_with_350: return sorted(chains_with_350)[0]
    counts=defaultdict(int)
    for res in residues.values():
        if res.bw is not None: counts[res.chain]+=1
    return sorted(counts.items(), key=lambda kv:(-kv[1],kv[0]))[0][0] if counts else None

def _worker(args):
    (path, chain, bw_window, policy)=args
    try:
        status,row = compute_distances(path, chain, bw_window, policy)
        tag,msg=status; base=os.path.basename(path)
        if tag=="OK": return ("OK", base, chain, row)
        if tag=="SKIP" and (chain is not None):
            auto = auto_pick_chain(path)
            if auto and auto != chain:
                status2,row2 = compute_distances(path, auto, bw_window, policy)
                if status2[0]=="OK":
                    return ("OK", base, auto, row2)
        if tag=="SKIP":
            status3,row3 = compute_distances(path, None, bw_window, policy)
            if status3[0]=="OK":
                return ("OK", base, "(all)", row3)
        return ("SKIP", base, chain, msg)
    except Exception as e:
        base=os.path.basename(path)
        return ("ERR", base, chain, str(e))

def main():
    ap=argparse.ArgumentParser(description="GPCR distances/angles ONLY (sin volúmenes) con política de flexibilidad por identidad residuo↔BW.")
    ap.add_argument("--in", dest="inp", required=True, help="Carpeta o patrón de PDBs")
    ap.add_argument("--chain", default=None, help="Cadena a usar (si se omite, intenta auto-detectar)")
    ap.add_argument("--out", default="gpcr_distances.csv", help="CSV de salida")
    ap.add_argument("--bw-window", type=int, default=2, help="Ventana ±N alrededor del BW objetivo")
    ap.add_argument("--policy", choices=["strict","mid","loose"], default="mid", help="Flexibilidad por identidad (ver cabecera)")
    ap.add_argument("--workers", type=int, default=6, help="Procesos en paralelo")
    ap.add_argument("--familias", default=None, help="Ruta a familias.txt (TSV/CSV). Usa 1ª columna -> última columna para gpcr_class.")
    args=ap.parse_args()

    files = expand_inputs(args.inp)
    if not files:
        print("No se encontraron PDBs de entrada.", file=sys.stderr); sys.exit(2)

    # Cargar mapping de familias si se proporciona
    mapping = None
    if args.familias is not None:
        fam_path = Path(args.familias)
        if not fam_path.exists():
            print(f"[ERROR] familias.txt not found: {fam_path}", file=sys.stderr); sys.exit(2)
        mapping = load_familias(fam_path)
        print(f"[INFO] familias cargadas desde {fam_path} | entradas: {len(mapping)}")

    # Preparar trabajos
    todo=[]
    for path in files:
        chain = args.chain or auto_pick_chain(path)
        if chain is None:
            print(f"[SKIP] {os.path.basename(path)}: cadena no detectada.", file=sys.stderr)
            continue
        todo.append((path, chain, args.bw_window, args.policy))

    # CSV
    import csv
    fieldnames=[
        "file","receptor","chain_used","status","skip_reason",
        "state","gpcr_class",
        "ionic_lock_R3.50_E/D6.30","ionic_lock_present","ionic_lock_CA_R3.50_E/D6.30",
        "ionic_lock_used_3.50","ionic_lock_used_6.30","ionic_lock_used_bw_3.50","ionic_lock_used_bw_6.30",
        "alt_lock_min_dist","alt_lock_present","alt_lock_CA_3.49_6.30",
        "alt_lock_used_3.49","alt_lock_used_6.30","alt_lock_used_bw_3.49","alt_lock_used_bw_6.30",
        "network_min_dist","network_present","network_used_3.39","network_used_6.40","network_used_bw_3.39","network_used_bw_6.40",
        "dry_min_dist","dry_CA_3.49_3.50","dry_used_3.49","dry_used_3.50","dry_used_bw_3.49","dry_used_bw_3.50",
        "tm3_tm6_ic_CA_3.50_6.34","connector_CA_3.40_6.44","sodium_net_D2.50_N7.49",
        "portal_EC_CA_5.39_6.58","portal_EC_CA_minset","portal_EC_heavy_minset",
        "W6.48_chi2_deg","Y7.53_chi1_deg","W6.48_used_BW","W6.48_used_resname","CWxP_kink_TM6_deg"
    ]

    processed=0
    all_rows = []  # Para luego generar el Excel

    with open(args.out,"w",newline="") as fo:
        wr=csv.DictWriter(fo, fieldnames=fieldnames); wr.writeheader(); fo.flush()
        if args.workers > 1:
            from multiprocessing import Pool
            chunksize = max(1, (len(todo)//(args.workers*4)) or 1)
            with Pool(processes=args.workers) as pool:
                for res in pool.imap_unordered(_worker, todo, chunksize=chunksize):
                    tag, base, chain_used, payload = res
                    state = infer_state(base)
                    gpcr_class = infer_gpcr_class(base, mapping)
                    receptor = infer_receptor_name(base)
                    if tag=="OK":
                        row={
                            "file":base,
                            "receptor": receptor,
                            "chain_used":chain_used,
                            "status":"OK",
                            "skip_reason":"",
                            "state": state,
                            "gpcr_class": gpcr_class
                        }
                        row.update(payload)
                        wr.writerow(row); fo.flush(); processed+=1
                        all_rows.append(row)
                        print(f"[OK] {base} | chain={chain_used}")
                    elif tag=="SKIP":
                        row={
                            "file":base,
                            "receptor": receptor,
                            "chain_used":chain_used,
                            "status":"SKIP",
                            "skip_reason":str(payload),
                            "state": state,
                            "gpcr_class": gpcr_class
                        }
                        for k in fieldnames:
                            if k not in row: row[k]=float("nan")
                        wr.writerow(row); fo.flush()
                        all_rows.append(row)
                        print(f"[SKIP] {base}: {payload}", file=sys.stderr)
                    else:
                        print(f"[ERR] {base}: {payload}", file=sys.stderr)
        else:
            for argsW in todo:
                tag, base, chain_used, payload = _worker(argsW)
                state = infer_state(base)
                gpcr_class = infer_gpcr_class(base, mapping)
                receptor = infer_receptor_name(base)
                if tag=="OK":
                    row={
                        "file":base,
                        "receptor": receptor,
                        "chain_used":chain_used,
                        "status":"OK",
                        "skip_reason":"",
                        "state": state,
                        "gpcr_class": gpcr_class
                    }
                    row.update(payload)
                    wr.writerow(row); fo.flush(); processed+=1
                    all_rows.append(row)
                    print(f"[OK] {base} | chain={chain_used}")
                elif tag=="SKIP":
                    row={
                        "file":base,
                        "receptor": receptor,
                        "chain_used":chain_used,
                        "status":"SKIP",
                        "skip_reason":str(payload),
                        "state": state,
                        "gpcr_class": gpcr_class
                    }
                    for k in fieldnames:
                        if k not in row: row[k]=float("nan")
                    wr.writerow(row); fo.flush()
                    all_rows.append(row)
                    print(f"[SKIP] {base}: {payload}", file=sys.stderr)
                else:
                    print(f"[ERR] {base}: {payload}", file=sys.stderr)

    # ---- Generar Excel con pestañas de clasificación ----
    if all_rows:
        df = pd.DataFrame(all_rows)

        # Nombre del Excel basado en el nombre del CSV
        excel_out = os.path.splitext(args.out)[0] + ".xlsx"

        # Hoja "all" con TODO y subsets solo con status OK
        df_ok = df[df["status"] == "OK"].copy()

        with pd.ExcelWriter(excel_out) as writer:
            df.to_excel(writer, sheet_name="all", index=False)

            if not df_ok.empty:
                only_trad = df_ok[(df_ok["ionic_lock_present"] == 1) &
                                  (df_ok["alt_lock_present"] == 0)]
                only_alt = df_ok[(df_ok["ionic_lock_present"] == 0) &
                                 (df_ok["alt_lock_present"] == 1)]
                both = df_ok[(df_ok["ionic_lock_present"] == 1) &
                             (df_ok["alt_lock_present"] == 1)]
                none = df_ok[(df_ok["ionic_lock_present"] == 0) &
                             (df_ok["alt_lock_present"] == 0)]

                only_trad.to_excel(writer, sheet_name="only_trad", index=False)
                only_alt.to_excel(writer, sheet_name="only_alt", index=False)
                both.to_excel(writer, sheet_name="both", index=False)
                none.to_excel(writer, sheet_name="none", index=False)

        print(f"[INFO] Excel generado: {excel_out}")

    print(f"[RESUMEN] PDBs procesados (OK): {processed} | CSV: {args.out}")

if __name__ == "__main__":
    main()

