#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotates gpcr_distances.csv with:
  - state: 'active' / 'inactive' (if the filename ends with active.BWocc.pdb or inactive.BWocc.pdb)
  - gpcr_class: mapped from familias.txt (first column -> last column).
    * The mapping prefix is the part before the first "_" in the filename (lowercase in the CSV);
      in familias.txt it appears in UPPERCASE in the first column.

Usage:
  python3 2_anotate_class.py \
    --in-csv gpcr_distances.csv \
    --familias families.txt \
    --out-csv gpcr_distances_annot.csv

Notes:
- Assumes the CSV has a column named 'file' with the PDB filename.
- familias.txt can be TSV or CSV; the script tries to infer the separator.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

def infer_sep(path: Path):
    # try pandas' inference; if it fails, assume tab-separated
    try:
        return pd.read_csv(path, sep=None, engine="python", nrows=5)
    except Exception:
        return pd.read_csv(path, sep="\t", nrows=5)

def load_familias(path: Path):
    # try to infer the separator by loading the entire file
    try:
        fam = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        fam = pd.read_csv(path, sep="\t")
    if fam.shape[1] < 2:
        raise ValueError("familias.txt must have at least two columns (ID and Class).")
    first_col = fam.columns[0]
    last_col  = fam.columns[-1]
    # dictionary: FIRST_COLUMN (uppercase) -> LAST_COLUMN (class)
    mapping = dict(zip(fam[first_col].astype(str).str.upper().str.strip(),
                       fam[last_col].astype(str).str.strip()))
    return mapping

def infer_state(filename: str) -> str:
    f = str(filename).lower().strip()
    # exactly as requested: ends exactly with active/inactive.BWocc.pdb
    if f.endswith("_active.pdb"):
        return "active"
    if f.endswith("_inactive.pdb"):
        return "inactive"
    return "unknown"

def file_prefix_lower(filename: str) -> str:
    base = Path(str(filename)).name
    return base.split("_", 1)[0].lower()

def infer_gpcr_class(filename: str, map_upper: dict) -> str:
    key_up = file_prefix_lower(filename).upper()
    return map_upper.get(key_up, "Unknown")

def main():
    ap = argparse.ArgumentParser(description="Annotate gpcr_distances.csv with 'state' and 'gpcr_class' using familias.txt.")
    ap.add_argument("--in-csv", required=True, help="Path to gpcr_distances.csv")
    ap.add_argument("--familias", required=True, help="Path to familias.txt (TSV/CSV). Uses 1st column -> last column.")
    ap.add_argument("--out-csv", required=True, help="Output path for the annotated CSV")
    ap.add_argument("--file-col", default="file", help="Name of the column with the filename (default: 'file')")
    ap.add_argument("--fail-on-unknown", action="store_true",
                    help="If set, raise an error if any row ends up with gpcr_class='Unknown'")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    fam_txt = Path(args.familias)
    out_csv = Path(args.out_csv)

    if not in_csv.exists():
        print(f"[ERROR] Input CSV not found: {in_csv}", file=sys.stderr); sys.exit(2)
    if not fam_txt.exists():
        print(f"[ERROR] familias.txt not found: {fam_txt}", file=sys.stderr); sys.exit(2)

    # Load data
    df = pd.read_csv(in_csv)
    if args.file_col not in df.columns:
        print(f"[ERROR] Column '{args.file_col}' not found in {in_csv}. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    mapping = load_familias(fam_txt)

    # Add columns
    df["state"] = df[args.file_col].apply(infer_state)
    df["gpcr_class"] = df[args.file_col].apply(lambda s: infer_gpcr_class(s, mapping))

    if args.fail_on_unknown and (df["gpcr_class"] == "Unknown").any():
        unknowns = df.loc[df["gpcr_class"] == "Unknown", args.file_col].tolist()
        print(f"[ERROR] Found {len(unknowns)} without class (examples: {unknowns[:5]})", file=sys.stderr)
        sys.exit(3)

    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Annotated CSV -> {out_csv} | rows: {len(df)} | Unknowns: {(df['gpcr_class']=='Unknown').sum()}")

if __name__ == "__main__":
    main()

